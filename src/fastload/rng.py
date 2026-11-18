"""Random-number handling that reproduces LeRobot's DataLoader.

LeRobot draws data randomness in three places, all on the CPU:

1. When a DataLoader iterator starts, the main process draws ``base_seed`` from the global torch RNG, and a
   ``RandomSampler`` draws its seed (``EpisodeAwareSampler``, which ``lerobot_train`` uses for every map-style
   dataset since 0.6, seeds its own generator from ``(seed, epoch)`` and draws nothing). With
   ``persistent_workers`` (``lerobot_train``'s default when ``num_workers > 0``) only the first epoch creates an
   iterator; later epochs ``_reset`` it: no ``base_seed`` draw and no worker reseeding.
2. Each DataLoader worker ``w`` reseeds ``random``, ``torch`` and ``numpy`` from ``base_seed + w`` when it
   starts, and batch ``k`` of an epoch is fetched by worker ``k % num_workers`` (``_reset`` restarts the worker
   cycle at worker 0). With ``num_workers=0`` the main process fetches, so image transforms consume the main
   RNG (one batch ahead, because accelerate's ``DataLoaderShard`` fetches the next batch before yielding the
   current one).
3. ``LeRobotDataset.__getitem__`` applies ``image_transforms`` per camera, which draws the transform
   parameters (``RandomSubsetApply`` selection, ColorJitter factors, ...) from the worker's RNG.

FastLoader gets (1) by iterating a real DataLoader over row indices (prepared by the real accelerate),
emulates (2) with saved RNG states, and for (3) draws the parameters exactly like torchvision does but
applies the transform kernels to the batch on the GPU (:class:`TransformReplayer`).
"""

from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader as _dl
from torchvision.transforms import v2

# torchvision >= 0.21 calls make_params()/transform(); older releases _get_params()/_transform().
_TRANSFORM_ATTR = "transform" if hasattr(v2.Transform, "make_params") else "_transform"


# --------------------------------------------------------------------------------------------
# image transforms: draw parameters on the CPU, apply kernels anywhere
# --------------------------------------------------------------------------------------------
def _collect_transforms(root: Any) -> list[v2.Transform]:
    """Every v2.Transform reachable from ``root``, including ones held in plain lists / dicts
    (LeRobot's ImageTransforms and RandomSubsetApply do not register their children as submodules)."""
    found: list[v2.Transform] = []
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, v2.Transform):
            found.append(obj)
            for v in vars(obj).values():
                visit(v)
            for m in obj._modules.values():
                visit(m)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)

    visit(root)
    return found


class TransformReplayer:
    """Splits a torchvision-v2 transform pipeline into parameter drawing and kernel application.

    ``record(shape)`` runs the pipeline on a zero-cost proxy tensor of ``shape`` with every leaf transform's
    kernel replaced by a recorder, so exactly the random draws of a real call happen (same calls, same
    order) and the drawn parameters are returned as a list of ``(leaf, params)``. ``apply(x, ops)`` then
    runs the real kernels with those parameters on ``x`` (any device).
    """

    def __init__(self, tf: Callable):
        self.tf = tf
        self.leaves = _collect_transforms(tf)
        if not self.leaves:
            raise TypeError(f"fastload: image_transforms must be torchvision.transforms.v2 transforms, got {type(tf)}")

    def record(self, shape: tuple[int, ...],
               dtype: torch.dtype = torch.float32) -> list[tuple[v2.Transform, dict[str, Any]]]:
        ops: list[tuple[v2.Transform, dict[str, Any]]] = []
        proxy = torch.zeros((), dtype=dtype).expand(shape)  # dtype of the real input (uint8 video since 0.6)

        def recorder(leaf):
            def _rec(inpt, params):
                ops.append((leaf, params))
                return inpt
            return _rec

        for leaf in self.leaves:
            setattr(leaf, _TRANSFORM_ATTR, recorder(leaf))  # instance attribute shadows the method
        try:
            self.tf(proxy)
        finally:
            for leaf in self.leaves:
                delattr(leaf, _TRANSFORM_ATTR)
        return ops

    @staticmethod
    def apply(x: torch.Tensor, ops: list[tuple[v2.Transform, dict[str, Any]]]) -> torch.Tensor:
        for leaf, params in ops:
            if x.device.type != "cpu" and _is_seeded_noise(leaf, params, x):
                x = _seeded_noise_cpu_draw(x, params)
            else:
                x = getattr(leaf, _TRANSFORM_ATTR)(x, params)
        return x


def _is_seeded_noise(leaf: Any, params: dict[str, Any], x: torch.Tensor) -> bool:
    """LeRobot's GaussianNoise: noise from ``torch.Generator(device=x.device).manual_seed(seed)`` (float only)."""
    return type(leaf).__name__ == "GaussianNoise" and "seed" in params and "std" in params and x.is_floating_point()


def _seeded_noise_cpu_draw(x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """GaussianNoise on a GPU tensor with the noise LeRobot draws on the CPU (a CUDA generator with the same seed
    gives different numbers); the arithmetic then runs on the GPU (float rounding only)."""
    g = torch.Generator(device="cpu").manual_seed(params["seed"])
    noise = torch.randn(x.shape, dtype=x.dtype, generator=g).to(x.device, non_blocking=True)
    return (x + noise * (params["std"] / 255.0)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------------------------
# DataLoader worker RNG streams
# --------------------------------------------------------------------------------------------
class _State:
    __slots__ = ("torch", "py", "np")

    def __init__(self, torch_state, py_state, np_state):
        self.torch, self.py, self.np = torch_state, py_state, np_state


def _capture() -> _State:
    return _State(torch.get_rng_state(), random.getstate(), np.random.get_state())


def _restore(s: _State) -> None:
    torch.set_rng_state(s.torch)
    random.setstate(s.py)
    np.random.set_state(s.np)


def _worker_state(base_seed: int, worker_id: int) -> _State:
    """The RNG state a DataLoader worker starts with (torch.utils.data._utils.worker._worker_loop)."""
    seed = base_seed + worker_id
    g = torch.Generator()
    g.manual_seed(seed)
    py = random.Random(seed).getstate()
    np_rs = np.random.RandomState(_dl._utils.worker._generate_state(base_seed, worker_id))
    return _State(g.get_state(), py, np_rs.get_state())


class WorkerRNG:
    """RNG streams of ``num_workers`` emulated DataLoader workers (0 = draw from the process RNG).

    With persistent workers the streams are seeded by the first epoch only and carry on across epochs
    (``begin_epoch(None)``; see :class:`IndexDataset`).

    ``private_seed`` instead gives a single stream that never touches the process RNG (used outside the
    LeRobot-order mode so that augmentation cannot shift the training RNG).
    """

    def __init__(self, num_workers: int, private_seed: int | None = None):
        self.num_workers = int(num_workers)
        self.private_seed = private_seed
        self._states: list[_State] = []
        if private_seed is not None:
            self._states = [_worker_state(int(private_seed), 0)]

    def begin_epoch(self, base_seed: int | None) -> None:
        """A new DataLoader iterator: workers start from ``base_seed`` (None = persistent workers carry on)."""
        if self.private_seed is None and self.num_workers > 0 and base_seed is not None:
            self._states = [_worker_state(int(base_seed), w) for w in range(self.num_workers)]

    def draw(self, batch_index: int, fn: Callable[[], Any]) -> Any:
        if self.private_seed is None and self.num_workers == 0:
            return fn()  # main-process fetching: the global RNG, like LeRobot
        w = 0 if self.private_seed is not None else batch_index % self.num_workers
        if w >= len(self._states):
            raise RuntimeError("fastload: worker RNG used before begin_epoch()")
        saved = _capture()
        _restore(self._states[w])
        try:
            return fn()
        finally:
            self._states[w] = _capture()
            _restore(saved)


# --------------------------------------------------------------------------------------------
# index DataLoader: the real sampler / accelerate sharding / base_seed draws, without loading data
# --------------------------------------------------------------------------------------------
class IndexDataset(Dataset):
    """Stands in for the LeRobotDataset in a DataLoader that yields row indices only.

    The index DataLoader itself has no workers. With ``persistent=True`` it emulates a DataLoader with
    ``persistent_workers=True``: the first iterator draws ``base_seed`` as usual, later ones restore the RNG
    that drew it (PyTorch only ``_reset``s a persistent iterator), so sampler draws and the process RNG
    advance exactly like LeRobot's DataLoader, and ``on_new_iterator`` gets ``None`` (keep worker streams).
    """

    def __init__(self, n: int, persistent: bool = False):
        self.n = int(n)
        self.persistent = bool(persistent)
        self.iterators = 0  # iterators created so far (epochs started)
        self.on_new_iterator: Callable[[int | None], None] | None = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return int(idx)


def collate_rows(rows: list[int]) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64)


_orig_iter_init = _dl._BaseDataLoaderIter.__init__


def _iter_init(self, loader):
    ds = loader.dataset
    if not isinstance(ds, IndexDataset):
        return _orig_iter_init(self, loader)
    reuse = ds.persistent and ds.iterators > 0
    gen = loader.generator
    saved = None
    if reuse:
        # A persistent iterator is reset, not rebuilt: undo this new iterator's base_seed draw. Nothing else in
        # __init__ draws (sampler iterators draw only when advanced).
        saved = gen.get_state() if gen is not None else torch.get_rng_state()
    _orig_iter_init(self, loader)
    if saved is not None:
        if gen is not None:
            gen.set_state(saved)
        else:
            torch.set_rng_state(saved)
    ds.iterators += 1
    if ds.on_new_iterator is not None:
        ds.on_new_iterator(None if reuse else int(self._base_seed))


if not getattr(_dl._BaseDataLoaderIter.__init__, "_fastload", False):
    _iter_init._fastload = True  # type: ignore[attr-defined]
    _dl._BaseDataLoaderIter.__init__ = _iter_init
