"""Random-number handling that reproduces LeRobot's DataLoader.

LeRobot draws data randomness in three places, all on the CPU:

1. When a DataLoader iterator starts, the main process draws ``base_seed`` from the DataLoader's ``generator``
   (``lerobot_train`` 0.7 passes ``torch.Generator().manual_seed(seed)``; without one, from the global torch RNG),
   and a ``RandomSampler`` draws its seed (``EpisodeAwareSampler``, which ``lerobot_train`` uses for every map-style
   dataset, seeds its own generator from ``(seed, epoch)`` and draws nothing). With ``persistent_workers``
   (``lerobot_train``'s default when ``num_workers > 0``) only the first epoch creates an iterator; later epochs
   ``_reset`` it: no ``base_seed`` draw and no worker reseeding.
2. Each DataLoader worker ``w`` reseeds ``random``, ``torch`` and ``numpy`` from ``base_seed + w`` when it
   starts, and batch ``k`` of an epoch is fetched by worker ``k % num_workers`` (``_reset`` restarts the worker
   cycle at worker 0). With ``num_workers=0`` the main process fetches, so image transforms consume the main
   RNG (one batch ahead, because accelerate's ``DataLoaderShard`` fetches the next batch before yielding the
   current one).
3. ``LeRobotDataset.__getitems__`` (one call per batch; the same draws as per-item ``__getitem__``) applies
   ``image_transforms`` per sample and camera (``apply_rgb_transforms``), which draws the transform parameters
   from the worker's default generators.

FastLoader gets (1) by iterating a real DataLoader over row indices (prepared by the real accelerate),
emulates (2) with saved RNG states, and for (3) draws the parameters exactly like LeRobot does and applies the
transform math separately (:func:`make_replayer`).
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
# image transforms: draw parameters on the CPU, apply them anywhere
# --------------------------------------------------------------------------------------------
def make_replayer(tf: Callable) -> "TransformReplayer | ImageTransformsReplayer":
    """Replayer for a dataset's ``image_transforms``: LeRobot's ``ImageTransforms`` (0.7, batched transforms) or a
    plain ``torchvision.transforms.v2`` pipeline (custom scripts)."""
    from fastload import _lr

    if isinstance(tf, _lr.lerobot_image_transforms_types()[0]):
        return ImageTransformsReplayer(tf)
    return TransformReplayer(tf)


def _collect_transforms(root: Any) -> list[v2.Transform]:
    """Every v2.Transform reachable from ``root``, including ones held in plain lists / dicts."""
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
            raise TypeError(
                "fastload: image_transforms must be lerobot.transforms.ImageTransforms or torchvision.transforms.v2 "
                f"transforms, got {type(tf).__module__}.{type(tf).__qualname__}"
            )

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
            # The class's kernel, not the instance attribute: ``record`` (main thread) temporarily shadows it with
            # a recorder while the prefetch thread may be applying earlier parameters (augment_device="cpu").
            x = getattr(type(leaf), _TRANSFORM_ATTR)(leaf, x, params)
        return x

    def apply_batch(self, x: torch.Tensor, ops: list) -> torch.Tensor:
        """Every sample of ``x`` with its own parameters."""
        return torch.stack([self.apply(x[i], ops[i]) for i in range(len(ops))])


_INNER_OPS = "_fastload_inner_ops"  # recorded draws of a PerSampleTransform's wrapped transform, per sample


def _tree_to(params: Any, device: torch.device) -> Any:
    if isinstance(params, torch.Tensor):
        return params.to(device, non_blocking=True)
    if isinstance(params, dict):
        return {k: (v if k == _INNER_OPS else _tree_to(v, device)) for k, v in params.items()}
    if isinstance(params, list):
        return [_tree_to(v, device) for v in params]
    return params


def _tree_cat(trees: list[Any]) -> Any:
    """Per-sample parameter trees (leading dim 1) -> one batch tree (leading dim B)."""
    first = trees[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(trees)
    if isinstance(first, dict):
        return {k: _tree_cat([t[k] for t in trees]) for k in first}
    if isinstance(first, list):
        return [_tree_cat([t[i] for t in trees]) for i in range(len(first))]
    return first  # flags shared by every sample (uint8_input)


class ImageTransformsReplayer:
    """LeRobot 0.7 ``ImageTransforms`` split into parameter drawing and application.

    On the ``"dataloader"`` backend LeRobot calls ``ImageTransforms(frames)`` per sample and camera in a DataLoader
    worker: ``BatchedImageTransforms`` on a batch of one -> ``apply_to_frames(tf, frames, generator=None)``, which
    converts uint8 frames to float32 / 255, draws every parameter with ``tf.make_params(shape, cpu, None)`` from
    the worker's default generators, tags them with ``uint8_input``, runs ``tf.transform`` and rounds uint8 input
    back. The only draws outside ``make_params`` are those of a ``PerSampleTransform`` without a generator
    (torchvision v2 classes and ``JPEGCompression``), whose wrapped transform draws while it runs, in
    application order (``BatchedRandomSubsetApply._transform_one``).

    ``record(shape, dtype)`` makes exactly these draws (same calls, same order, same shapes, on the CPU) and returns
    the parameter tree, with the wrapped transforms' draws recorded by :class:`TransformReplayer`. ``apply`` runs
    LeRobot's transform math with them on one sample (bit-identical on the CPU); ``apply_batch`` on a batch.
    """

    def __init__(self, tf: Any):
        from fastload import _lr

        _, self._subset_cls, self._per_sample_cls, self._tag_params = _lr.lerobot_image_transforms_types()
        self.tf = tf
        self.root = tf.tf  # BatchedRandomSubsetApply, or BatchedIdentity when disabled
        self._inner: dict[int, TransformReplayer] = {}
        for m in self.root.modules():
            if isinstance(m, self._per_sample_cls):
                self._inner[id(m)] = TransformReplayer(m.per_sample)
        self.has_per_sample = bool(self._inner)

    @staticmethod
    def _five(shape: tuple[int, ...]) -> tuple[int, ...]:
        """ImageTransforms adds the batch dim, BatchedImageTransforms the frame dim of a (C, H, W) image."""
        return (1, 1, *shape) if len(shape) == 3 else (1, *shape)

    # ---- drawing ------------------------------------------------------------------------------------------
    def record(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> dict[str, Any]:
        if not (dtype == torch.uint8 or dtype.is_floating_point):
            raise TypeError(f"Expected uint8 or floating point images, got {dtype}.")
        uint8 = dtype == torch.uint8
        five = torch.Size(self._five(tuple(shape)))
        params = self._tag_params(self.root.make_params(five, torch.device("cpu"), None), uint8)
        self._record_run_draws(self.root, params, five, uint8)
        return params

    def _record_run_draws(self, tf: Any, params: dict[str, Any], shape: torch.Size, uint8: bool) -> None:
        """The draws ``tf.transform`` makes on a batch of one (a PerSampleTransform without generator)."""
        if isinstance(tf, self._subset_cls):
            indices = params["selected"][0].tolist()
            for index in indices if tf.random_order else sorted(indices):  # BatchedRandomSubsetApply._transform_one
                self._record_run_draws(tf.transforms[index], params["transforms"][index], shape, uint8)
        elif isinstance(tf, self._per_sample_cls) and params.get("seed") is None:
            # PerSampleTransform.transform: the wrapped transform gets each sample (N, C, H, W) in the caller's dtype
            dtype = torch.uint8 if uint8 else torch.float32
            params[_INNER_OPS] = [self._inner[id(tf)].record(tuple(shape[1:]), dtype) for _ in range(shape[0])]

    # ---- applying -----------------------------------------------------------------------------------------
    def _run(self, tf: Any, frames: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """``tf.transform(frames, params)``, with the recorded draws of per-sample transforms instead of new ones."""
        if isinstance(tf, self._subset_cls) and frames.shape[0] == 1:
            indices = params["selected"][0].tolist()
            for index in indices if tf.random_order else sorted(indices):
                frames = self._run(tf.transforms[index], frames, params["transforms"][index])
            return frames
        if isinstance(tf, self._per_sample_cls) and _INNER_OPS in params:
            uint8_input = params.get("uint8_input", False)  # mirrors PerSampleTransform.transform
            inner = self._inner[id(tf)]
            outputs = []
            for sample, ops in zip(frames.unbind(0), params[_INNER_OPS], strict=True):
                if uint8_input:
                    sample = (sample * 255.0).round_().clamp_(0, 255).to(torch.uint8)
                out = inner.apply(sample, ops)
                outputs.append(out.to(frames.dtype) / 255.0 if uint8_input else out)
            return torch.stack(outputs)
        return tf.transform(frames, params)

    def _apply_frames(self, frames: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """``apply_to_frames`` on (B, N, C, H, W) frames with given parameters."""
        uint8_input = frames.dtype == torch.uint8
        if uint8_input:
            # On CUDA, dividing by a Python scalar becomes x * (1/255); a 0-dim tensor keeps a true division.
            div = torch.tensor(255.0, dtype=torch.float32, device=frames.device) if frames.is_cuda else 255.0
            work = frames.to(torch.float32) / div
        else:
            work = frames.to(torch.float32)
        if work.device.type != "cpu":
            params = _tree_to(params, work.device)
        out = self._run(self.root, work, params)
        if uint8_input:
            return (out * 255.0).round_().clamp_(0, 255).to(torch.uint8)
        return out.to(frames.dtype)

    def apply(self, x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """``ImageTransforms(x)`` for one sample ``x`` ((C, H, W) or (T, C, H, W)) with recorded parameters."""
        frames = x.unsqueeze(0)
        frames = frames.unsqueeze(1) if frames.ndim == 4 else frames
        out = self._apply_frames(frames, params)
        out = out.squeeze(1) if x.ndim == 3 else out
        return out.squeeze(0)

    def apply_batch(self, x: torch.Tensor, ops: list[dict[str, Any]]) -> torch.Tensor:
        """Every sample of ``x`` ((B, C, H, W) or (B, T, C, H, W)) with its own parameters.

        On CUDA, without per-sample (torchvision) transforms, the batch runs through LeRobot's batched math in one
        call (as on LeRobot's "gpu" backend: every candidate transform runs and is masked into the samples that drew
        it); otherwise sample by sample exactly as in LeRobot's workers (bit-identical on the CPU).
        """
        if x.is_cuda and not self.has_per_sample and len(ops) > 1:
            frames = x.unsqueeze(1) if x.ndim == 4 else x
            out = self._apply_frames(frames, _tree_cat(ops))
            return out.squeeze(1) if x.ndim == 4 else out
        return torch.stack([self.apply(x[i], ops[i]) for i in range(len(ops))])


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
