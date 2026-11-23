"""FastLoader: batches assembled directly from the decoded-frame cache, with a VRAM tier.

Data layout per process (rank):

* host tier  - the cache memmap on local disk / page cache (always holds every frame)
* GPU tier   - a copy of as many of this rank's episodes as fit in free VRAM (sized per run)

Each step a background thread gathers the frames of an upcoming batch that are not in the GPU
tier from the memmap and copies them to the GPU on a side CUDA stream. The training thread then
takes the cached frames straight from VRAM and merges both parts. Non-image columns (state,
action, indices, ...) live on the GPU for the whole run.

The produced batch has the same keys, dtypes and values as
``default_collate([LeRobotDataset[i] for i in rows])``.

Batch order (``order``):

* ``"lerobot"`` (default) - the rows of every batch come from a real DataLoader over row indices
  built with LeRobot's sampler arguments and prepared by the real accelerate, so each rank gets the
  same batches, in the same order, with the same RNG draws as LeRobot's DataLoader. Image-transform
  parameters are drawn from emulated DataLoader-worker RNG streams (``num_workers`` must be LeRobot's)
  and applied on the GPU. The GPU tier holds a random subset of all frames.
* ``"shard"`` - each rank owns whole episodes and shuffles within them (fastest multi-GPU; the GPU
  tier holds this rank's own episodes). Transform parameters use a private RNG stream.

Incomplete cache (``cache_mode="during_training"``): a :class:`fastload.fill.CacheFiller` (decoder processes) fills
it. The prefetch thread asks it for the frames of the next batches (cache misses only, observation history
included) and waits, before reading a batch, until that batch's frames are filled. Nothing else changes: batches,
order and RNG draws are those of a complete cache, since decoding happens in other processes and the frames are
stored before they are read. The GPU tier waits until the cache is complete.

Size-limited cache (``max_cache_gb``): the frames of the episodes it does not store are decoded for the batches that
need them by the same decoder processes (:class:`fastload.fill.TransientFrames`), a few batches ahead, into a small
scratch area, and gathered with the cached frames. Again only the source of the bytes changes, not batches, order or
RNG draws. The GPU tier holds stored frames only.
"""

from __future__ import annotations

import atexit
import logging
import math
import queue
import threading
import time
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader as _TorchDataLoader

from fastload import _lr
from fastload.cache import DEFAULT_MAX_CACHE_GB, DEFAULT_MIN_FREE_DISK_GB
from fastload.dataset import CachedLeRobotDataset
from fastload.decode_hook import active_hook, apply_hook, hook_name
from fastload.rng import IndexDataset, TransformReplayer, WorkerRNG, collate_rows

logger = logging.getLogger("fastload")

PLACEMENTS = ("auto", "gpu", "host")
ORDERS = ("lerobot", "shard")
_ACTIVE_LOADERS: "weakref.WeakSet[FastLoader]" = weakref.WeakSet()


def active_loaders() -> list["FastLoader"]:
    return list(_ACTIVE_LOADERS)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _dist_info(device: torch.device | str | None) -> tuple[int, int, torch.device]:
    """(rank, world_size, device) from accelerate if initialised, else torch.distributed, else single."""
    try:
        from accelerate.state import PartialState

        if PartialState._shared_state:  # accelerate already initialised (e.g. by lerobot_train)
            st = PartialState()
            return st.process_index, st.num_processes, torch.device(device) if device else st.device
    except Exception:  # pragma: no cover - accelerate missing or not initialised
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank, world = 0, 1
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    return rank, world, torch.device(device)


def _load_columns(dataset: CachedLeRobotDataset, names: list[str]) -> dict[str, np.ndarray]:
    """Numeric parquet columns as numpy, with LeRobot's dtypes (float -> float32, int -> int64)."""
    data = dataset.hf_dataset.select_columns(names).with_format("numpy")[:]
    out = {}
    for name in names:
        arr = data[name]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack([np.asarray(x) for x in arr])
        arr = np.asarray(arr)
        # hf_transform_to_torch calls torch.tensor() on python values: floats -> float32, ints -> int64.
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32, copy=False)
        elif arr.dtype.kind in "iu":
            arr = arr.astype(np.int64, copy=False)
        out[name] = arr
    return out


CHUNK_BYTES = int(1e9)  # GPU tier allocation unit; shrinking drops whole chunks without copying


@dataclass
class _GpuTier:
    slot: np.ndarray | None = None  # [total_frames] int64 position in the tier, -1 = not on GPU
    tensors: dict[str, list[torch.Tensor]] = field(default_factory=dict)  # key -> chunks of [F, C, H, W] uint8
    chunk_frames: int = 1  # F: frames per chunk (position p lives in chunk p // F at row p % F)
    nbytes: int = 0
    frames: int = 0
    released: bool = False


@dataclass
class _Prepared:
    rows: np.ndarray
    tier: _GpuTier
    rows_dev: torch.Tensor
    tasks: list[str]
    subtasks: list[str] | None
    cams: dict[str, dict[str, Any]]
    event: Any = None
    keepalive: list[torch.Tensor] = field(default_factory=list)
    batch_index: int = -1
    ops: Any = None  # image-transform parameters already applied on the CPU (augment_device="cpu")


# --------------------------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------------------------
class FastLoader:
    """Iterable over training batches for one process. One ``iter()`` = one epoch.

    Args:
        dataset: a :class:`CachedLeRobotDataset` (or a LeRobotDataset plus ``cache_dir``).
        batch_size: per-process batch size (same meaning as DataLoader's ``batch_size``).
        sampler: optional sampler, as passed to DataLoader (LeRobot: ``EpisodeAwareSampler``).
        shuffle: shuffle every epoch (DataLoader's ``shuffle``).
        drop_last: drop the last incomplete batch of an epoch.
        order: ``"lerobot"`` (same batches and RNG draws as LeRobot's DataLoader) or ``"shard"``
            (per-rank episode shards); see the module docstring.
        num_workers: the ``num_workers`` of the DataLoader being reproduced (``order="lerobot"``): image
            transforms draw from the RNG streams those workers would have. No worker processes are started.
        decode_hook_device: where a wrapper around LeRobot's ``decode_video_frames`` (e.g. a resize in the
            data loader) runs: ``"cpu"`` = exactly as LeRobot (bit-identical; those cameras skip the GPU
            tier), ``"gpu"`` = on the batch on the GPU (faster, float-rounding differences).
        memory_guard: after the GPU tier exists, check free VRAM every step and drop tier chunks before the
            model runs out of memory (the peak use of the run can grow after the tier was sized).
        augment_device: where image transforms run: ``"gpu"`` = on the batch on the GPU (fast; augmented pixels
            differ from LeRobot's CPU result by float rounding), ``"cpu"`` = LeRobot's exact CPU path in the
            prefetch thread (bit-identical; no GPU tier).
        placement: ``"auto"`` - grow a GPU tier after ``warmup_steps`` batches, sized from the measured
            peak memory of the run; ``"gpu"`` - put this rank's whole shard on the GPU now;
            ``"host"`` - never use VRAM for frames.
        warmup_steps: batches to serve before measuring memory (auto mode). Must include at least one
            optimizer step so optimizer state is counted.
        gpu_margin_gb: VRAM left free on top of the measured peak (default: max(2 GB, 5% of the GPU)).
        max_gpu_cache_gb: upper bound on the GPU tier size.
        prefetch: batches prepared ahead by the background thread (at least 4 while the cache is being filled).
        seed: shuffling seed (default: derived from torch's initial seed). Shards are seed-independent.
        cache_mode: when ``dataset`` is not a CachedLeRobotDataset yet: ``"prebuilt"`` (the cache in ``cache_dir``
            must be complete), ``"build_first"`` (build it now) or ``"during_training"`` (fill it while serving
            batches); see :meth:`CachedLeRobotDataset.from_dataset`. Whenever the dataset's cache is incomplete, the
            loader fills it.
        decode_workers: decoder processes that fill an incomplete cache (default: about half the CPU cores divided
            by the processes on this node).
        background_fill: while decoders are idle, also decode frames no batch has asked for yet, so the cache
            completes (and becomes a regular prebuilt cache) even if training never reads some frames.
        min_free_disk_gb: free disk that must remain after the cache is written (``cache_mode`` other than
            prebuilt); a safety margin, not a size limit.
        max_cache_gb: size limit of a cache created here (``cache_mode`` build_first / during_training): a larger
            dataset stores whole episodes up to this size; the decoder processes decode the others for the batches
            that need them. A prebuilt cache keeps the limit it was built with.
    """

    def __init__(
        self,
        dataset: CachedLeRobotDataset,
        batch_size: int,
        *,
        sampler: Any = None,
        shuffle: bool = True,
        drop_last: bool = False,
        placement: str = "auto",
        warmup_steps: int = 20,
        gpu_margin_gb: float | None = None,
        max_gpu_cache_gb: float | None = None,
        prefetch: int = 2,
        seed: int | None = None,
        device: torch.device | str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        cache_dir: str | None = None,
        order: str = "lerobot",
        num_workers: int = 0,
        decode_hook_device: str = "cpu",
        memory_guard: bool = True,
        augment_device: str = "gpu",
        cache_mode: str = "prebuilt",
        decode_workers: int | None = None,
        background_fill: bool = True,
        min_free_disk_gb: float | None = DEFAULT_MIN_FREE_DISK_GB,
        max_cache_gb: float | None = DEFAULT_MAX_CACHE_GB,
    ):
        if not isinstance(dataset, CachedLeRobotDataset):
            if cache_dir is None:
                raise ValueError("dataset is not a CachedLeRobotDataset; pass cache_dir=...")
            dataset = CachedLeRobotDataset.from_dataset(dataset, cache_dir, cache_mode=cache_mode,
                                                        min_free_disk_gb=min_free_disk_gb,
                                                        build_workers=decode_workers, max_cache_gb=max_cache_gb)
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        if decode_hook_device not in ("cpu", "gpu"):
            raise ValueError(f"decode_hook_device must be 'cpu' or 'gpu', got {decode_hook_device!r}")
        self.order = order
        self.num_workers = int(num_workers or 0)
        self.memory_guard = bool(memory_guard)
        if augment_device not in ("cpu", "gpu"):
            raise ValueError(f"augment_device must be 'cpu' or 'gpu', got {augment_device!r}")
        r, w, dev = _dist_info(device)
        self.rank = r if rank is None else rank
        self.world_size = w if world_size is None else world_size
        self.device = torch.device(dev)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.placement = placement if self.is_cuda else "host"
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = bool(shuffle or getattr(sampler, "shuffle", False))
        self.warmup_steps = max(1, int(warmup_steps))
        self.gpu_margin_gb = gpu_margin_gb
        self.max_gpu_cache_gb = max_gpu_cache_gb
        self.prefetch = max(1, int(prefetch))
        self.seed = int(seed) if seed is not None else int(torch.initial_seed() % (2**31))
        self.cache = dataset.fastload_cache
        meta = dataset.meta

        # ---- camera keys and the static per-frame tables --------------------------------------
        self.cam_keys = list(meta.camera_keys)
        self.cache.check_compatible(dataset, keys=self.cam_keys)
        self.video_keys = set(meta.video_keys)
        self.cam_shape = {k: self.cache.frame_shape(k) for k in self.cam_keys}
        self.frame_bytes = sum(self.cache.frame_nbytes(k) for k in self.cam_keys)

        hf_names = [n for n in dataset.hf_dataset.column_names if n not in self.cam_keys]
        cols_np = _load_columns(dataset, hf_names)
        self.columns_cpu = {n: torch.from_numpy(np.ascontiguousarray(a)) for n, a in cols_np.items()
                            if a.dtype.kind in "fiub"}
        self._index_np = cols_np["index"].astype(np.int64)
        self._episode_np = cols_np["episode_index"].astype(np.int64)
        self._task_np = cols_np["task_index"].astype(np.int64) if "task_index" in cols_np else None
        self._subtask_np = (cols_np["subtask_index"].astype(np.int64)
                            if "subtask_index" in cols_np and getattr(meta, "subtasks", None) is not None
                            else None)
        self.task_names = ([meta.tasks.iloc[i].name for i in range(len(meta.tasks))]
                           if self._task_np is not None else [])
        self.subtask_names = ([meta.subtasks.iloc[i].name for i in range(len(meta.subtasks))]
                              if self._subtask_np is not None else None)
        self._ep_from_np = np.asarray(meta.episodes["dataset_from_index"], dtype=np.int64)
        self._ep_to_np = np.asarray(meta.episodes["dataset_to_index"], dtype=np.int64)
        self.delta_np = {k: np.asarray(v, dtype=np.int64) for k, v in (dataset.delta_indices or {}).items()}

        # absolute frame index -> row of hf_dataset (only differs when a subset of episodes is loaded)
        rel = None
        if dataset._absolute_to_relative_idx is not None:
            rel = np.full(int(meta.total_frames), -1, dtype=np.int64)
            rel[self._index_np] = np.arange(len(self._index_np))
        to_dev = (lambda t: t.to(self.device)) if self.is_cuda else (lambda t: t)
        self.columns = {n: to_dev(t) for n, t in self.columns_cpu.items()}
        self._rel_of_abs = to_dev(torch.from_numpy(rel)) if rel is not None else None
        self._ep_from = to_dev(torch.from_numpy(self._ep_from_np))
        self._ep_to = to_dev(torch.from_numpy(self._ep_to_np))
        self.delta = {k: to_dev(torch.from_numpy(v)) for k, v in self.delta_np.items()}
        self._div255 = torch.tensor(255.0, dtype=torch.float32, device=self.device) if self.is_cuda else 255

        # ---- decode_video_frames wrapper (e.g. a data-loader resize): same function, same place ----
        self.decode_hook = active_hook()
        self.hook_device = decode_hook_device
        self.hook_keys = [k for k in self.cam_keys if k in self.video_keys] if self.decode_hook else []
        self.out_shape = dict(self.cam_shape)  # per-frame [C, H, W] after the hook
        for key in self.hook_keys:
            c, h, w = self.cam_shape[key]
            probe = self._run_hook(key, torch.zeros((1, c, h, w), dtype=torch.float32), None, [0.0])
            if probe.dim() != 4 or probe.shape[0] != 1 or probe.dtype != torch.float32:
                raise RuntimeError(f"fastload: decode hook returned {probe.dtype}{tuple(probe.shape)} for "
                                   f"{key}; expected float32 [T, C, H, W]")
            self.out_shape[key] = tuple(probe.shape[1:])
        if self.hook_keys:
            logger.info(f"fastload: decode_video_frames wrapper {hook_name(self.decode_hook)} applied to cached "
                        f"frames on the {self.hook_device} "
                        + ", ".join(f"{k} {self.cam_shape[k][1:]}->{self.out_shape[k][1:]}" for k in self.hook_keys))
        # Frames run through the hook on the CPU never use the GPU tier (it holds raw uint8 frames).
        self.tier_keys = [k for k in self.cam_keys if not (k in self.hook_keys and self.hook_device == "cpu")]
        self.tier_frame_bytes = sum(self.cache.frame_nbytes(k) for k in self.tier_keys)
        if not self.tier_keys:
            self.placement = "host"

        # ---- image transforms: parameters drawn on the CPU, kernels applied on the device -------
        tf = dataset.image_transforms
        self.replayer = TransformReplayer(tf) if tf is not None else None
        # augment_device="cpu": images take LeRobot's exact path (uint8 -> float32 / 255 -> decode hook ->
        # transforms, all on the CPU), so they are bit-identical; no camera can use the GPU tier then.
        self.cpu_augment = self.replayer is not None and augment_device == "cpu"
        if self.cpu_augment:
            self.tier_keys, self.tier_frame_bytes, self.placement = [], 0, "host"
            logger.info("fastload: image transforms on the CPU (bit-identical to LeRobot; GPU tier off)")
        self.sample_shape = {}
        for key in self.cam_keys:
            c, h, w = self.out_shape[key]
            t = len(self.delta_np[key]) if key in self.delta_np else 0
            squeeze = key in self.video_keys and t == 1
            self.sample_shape[key] = (c, h, w) if t == 0 or squeeze else (t, c, h, w)

        # ---- rows this rank draws from --------------------------------------------------------
        has_indices = sampler is not None and hasattr(sampler, "indices")
        base_rows = (np.asarray(sampler.indices, dtype=np.int64) if has_indices
                     else np.arange(len(dataset), dtype=np.int64))
        self._index_loader = None
        self._prepared_index_loader = None
        if order == "lerobot":
            # Same sampler arguments as the DataLoader being replaced; it only yields row indices.
            self._index_ds = IndexDataset(len(dataset))
            self.rng = WorkerRNG(self.num_workers)
            self._index_ds.on_new_iterator = self.rng.begin_epoch
            self._index_loader = _TorchDataLoader(
                self._index_ds, batch_size=self.batch_size, shuffle=bool(shuffle) if sampler is None else False,
                sampler=sampler, drop_last=drop_last, num_workers=0, collate_fn=collate_rows,
            )
            _install_accelerate_hook()
            self.local_rows = base_rows
            self.local_episodes = np.unique(self._episode_np[base_rows])
        else:
            if sampler is not None and not has_indices:
                raise TypeError("order='shard' only supports samplers exposing `.indices` (EpisodeAwareSampler).")
            self.rng = WorkerRNG(0, private_seed=self.seed + 7919 * self.rank)
            self.local_rows, self.local_episodes = self._shard(base_rows)
        if len(self.local_rows) == 0:
            raise RuntimeError(f"rank {self.rank}: no frames ({len(base_rows)} frames for {self.world_size} ranks)")

        self._tier = _GpuTier()
        self._tier_deferred: tuple[str, int | None] | None = None  # GPU tier waiting for the cache to complete
        self._served = 0
        self._epoch = 0
        self._grown = self.placement != "auto"
        # ---- an incomplete cache (cache_mode="during_training"): decoder processes fill it -------------------
        # ---- a size-limited cache: the same processes decode the frames it does not store ------------------
        self._filler = None
        self._fill_lookahead = max(8, 2 * self.prefetch)  # batches whose frames are requested ahead
        self._stored = self.cache.stored()  # None, or bool per frame
        self._transient = None
        self._transient_lookahead = max(4, 2 * self.prefetch)  # batches whose unstored frames are decoded ahead
        if not self.cache.complete or self.cache.limited:
            from fastload.fill import CacheFiller, default_decode_workers

            slots = reserve = 0
            if self.cache.limited:
                deltas = {tuple(self.delta_np[k].tolist()) for k in self.cam_keys if k in self.delta_np}
                per_batch = self.batch_size * (1 + sum(len(d) for d in deltas))  # bound of _needed_frames(rows)
                reserve = 2 * per_batch  # the batch being assembled (and one re-assembled on the training thread)
                slots = reserve + self._transient_lookahead * per_batch
            self._filler = CacheFiller(self.cache, _lr.dataset_source(dataset), self._ep_from_np, self._ep_to_np,
                                       decode_workers or default_decode_workers(), background=background_fill,
                                       transient_slots=slots, transient_reserve=reserve)
            self._transient = self._filler.transient
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None
        self._iterators: "weakref.WeakSet[_EpochIterator]" = weakref.WeakSet()
        _ACTIVE_LOADERS.add(self)

        shard_gb = len(self.local_rows) * self.frame_bytes / 1e9
        logger.info(
            f"fastload: rank {self.rank}/{self.world_size} order={self.order} {len(self.local_episodes)} episodes, "
            f"{len(self.local_rows)} frames ({shard_gb:.1f} GB of frames), placement={self.placement}, "
            f"device={self.device}"
            + (f", RNG of {self.num_workers} DataLoader workers" if self.order == "lerobot" else "")
        )
        if self.placement == "gpu":
            self.build_gpu_cache(budget_bytes=None)

    # ------------------------------------------------------------------------------------------
    def _shard(self, base_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assign whole episodes round-robin after sorting by (task, episode) -> tasks spread evenly."""
        eps_of_rows = self._episode_np[base_rows]
        episodes = np.unique(eps_of_rows)
        if self._task_np is not None:
            first_row = {}
            for row, ep in zip(base_rows, eps_of_rows):
                first_row.setdefault(int(ep), int(row))
            task_of_ep = np.array([self._task_np[first_row[int(e)]] for e in episodes])
            episodes = episodes[np.lexsort((episodes, task_of_ep))]
        mine = episodes[self.rank::self.world_size]
        keep = np.isin(eps_of_rows, mine)
        return base_rows[keep], np.sort(mine)

    # ------------------------------------------------------------------------------------------
    # LeRobot-order mode: the index DataLoader and its accelerate-prepared version
    # ------------------------------------------------------------------------------------------
    @property
    def index_loader(self):
        """The DataLoader over row indices (``order="lerobot"``); accelerate prepares this one."""
        return self._index_loader

    def attach_prepared_index_loader(self, prepared) -> None:
        self._prepared_index_loader = prepared

    def _batch_source(self):
        if self._prepared_index_loader is not None:
            return self._prepared_index_loader
        if self.world_size > 1:
            raise RuntimeError(
                "fastload: order='lerobot' with several processes needs the loader to go through "
                "accelerator.prepare(...) (as lerobot-train does), so batches are sharded like LeRobot's"
            )
        return self._index_loader

    def __len__(self) -> int:
        if self.order == "lerobot":
            return len(self._batch_source())
        n = len(self.local_rows)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        it = _EpochIterator(self, self._epoch)
        self._epoch += 1
        self._iterators.add(it)
        return it

    # ------------------------------------------------------------------------------------------
    # GPU tier
    # ------------------------------------------------------------------------------------------
    @property
    def gpu_cache_fraction(self) -> float:
        return self._tier.frames / max(1, len(self.local_rows))

    def _margin(self, total: int) -> float:
        return self.gpu_margin_gb * 1e9 if self.gpu_margin_gb is not None else max(2e9, 0.05 * total)

    def _headroom(self) -> tuple[float, float]:
        """(VRAM still available at the run's peak, margin): free memory minus how far the peak reservation
        went above the current one (the model's transient need)."""
        free, total = torch.cuda.mem_get_info(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)
        current = torch.cuda.memory_reserved(self.device)
        return free - max(0, peak - current), self._margin(total)

    def _auto_budget(self) -> int:
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)
        current = torch.cuda.memory_reserved(self.device)
        margin = self._margin(total)
        budget = int(free - max(0, peak - current) - margin)
        logger.info(
            f"fastload: rank {self.rank} memory after {self._served} steps: free {free / 1e9:.1f} GB, "
            f"peak reserved {peak / 1e9:.1f} GB, now reserved {current / 1e9:.1f} GB, margin {margin / 1e9:.1f} GB "
            f"-> frame budget {max(budget, 0) / 1e9:.1f} GB"
        )
        return budget

    def build_gpu_cache(self, budget_bytes: int | None) -> None:
        """(Re)build the GPU tier with at most ``budget_bytes`` (None = all of this rank's frames)."""
        if not self.is_cuda or not self.tier_keys:
            return
        if self._filling:
            # Frames that are not decoded yet must not be copied to VRAM: build the tier once the cache is complete.
            if self._tier_deferred is None:
                logger.info(f"fastload: rank {self.rank} GPU tier deferred until the cache is complete")
            self._tier_deferred = ("budget", budget_bytes)
            return
        self.release_gpu_cache()
        fb = self.tier_frame_bytes
        need = len(self.local_rows) * fb
        budget = need if budget_bytes is None else min(int(budget_bytes), need)
        if self.max_gpu_cache_gb is not None:
            budget = min(budget, int(self.max_gpu_cache_gb * 1e9))
        if budget < fb:
            logger.info(f"fastload: rank {self.rank} no VRAM left for frames; host tier only")
            return
        t0 = time.perf_counter()
        # Whole episodes, in a seed-fixed random order, until the budget is used (a size-limited cache: stored ones).
        order = np.random.default_rng(self.seed + 7919 * self.rank).permutation(self.local_episodes)
        ranges, frames = [], 0
        for ep in order:
            a, b = int(self._ep_from_np[ep]), int(self._ep_to_np[ep])
            if self._stored is not None and (b <= a or not self._stored[a]):
                continue
            if (frames + b - a) * fb > budget:
                break
            ranges.append((a, b))
            frames += b - a
        if frames == 0:
            return
        frame_ids = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in ranges])
        per_chunk = max(1, CHUNK_BYTES // fb)
        slot = np.full(int(self.dataset.meta.total_frames), -1, dtype=np.int64)
        tensors: dict[str, list[torch.Tensor]] = {k: [] for k in self.tier_keys}
        try:
            mm = {k: self.cache.array(k) for k in self.tier_keys}
            for start in range(0, frames, per_chunk):
                ids = frame_ids[start:start + per_chunk]
                for key in self.tier_keys:
                    c, h, w = self.cam_shape[key]
                    chunk = torch.empty((len(ids), c, h, w), dtype=torch.uint8, device=self.device)
                    chunk.copy_(torch.from_numpy(np.asarray(mm[key][ids])))
                    tensors[key].append(chunk)
            torch.cuda.synchronize(self.device)
        except torch.OutOfMemoryError:
            tensors.clear()
            torch.cuda.empty_cache()
            logger.warning(f"fastload: rank {self.rank} OOM while building a {budget / 1e9:.1f} GB GPU tier; "
                           "retrying with half")
            self.build_gpu_cache(budget // 2)
            return
        slot[frame_ids] = np.arange(frames)
        self._tier = _GpuTier(slot=slot, tensors=tensors, chunk_frames=per_chunk, nbytes=frames * fb, frames=frames)
        logger.info(
            f"fastload: rank {self.rank} GPU tier {frames * fb / 1e9:.1f} GB = "
            f"{100 * self.gpu_cache_fraction:.0f}% of local frames ({len(tensors[self.tier_keys[0]])} chunks), "
            f"filled in {time.perf_counter() - t0:.1f}s"
        )

    def _replace_tier(self, new: _GpuTier) -> None:
        old = self._tier
        self._tier = new
        old.released = True
        old.tensors = {}  # drops this tier's references; chunks kept by `new` stay alive
        for it in list(self._iterators):
            it._invalidate()
        if self.is_cuda:
            torch.cuda.empty_cache()

    def release_gpu_cache(self) -> None:
        if self._tier.tensors:
            self._replace_tier(_GpuTier())

    def _truncate_tier(self, keep_bytes: float) -> None:
        """Keep only the first chunks of the GPU tier that fit in ``keep_bytes`` (no copies)."""
        old = self._tier
        if not old.tensors:
            return
        per_chunk = old.chunk_frames
        n_chunks = len(old.tensors[self.tier_keys[0]])
        keep = max(0, min(n_chunks, int(keep_bytes // (per_chunk * self.tier_frame_bytes))))
        if keep >= n_chunks:
            keep = n_chunks - 1  # always free at least one chunk
        if keep <= 0:
            self.release_gpu_cache()
            return
        limit = keep * per_chunk
        slot = old.slot.copy()
        slot[slot >= limit] = -1
        frames = min(old.frames, limit)
        self._replace_tier(_GpuTier(slot=slot, tensors={k: v[:keep] for k, v in old.tensors.items()},
                                    chunk_frames=per_chunk, nbytes=frames * self.tier_frame_bytes, frames=frames))

    def shrink_gpu_cache(self, fraction: float = 0.5) -> None:
        """Free VRAM (e.g. after an OOM): keep about ``fraction`` of the current GPU tier."""
        before = self._tier.nbytes
        self._truncate_tier(before * fraction)
        self._grown = True
        logger.warning(f"fastload: rank {self.rank} GPU tier {before / 1e9:.1f} -> {self._tier.nbytes / 1e9:.1f} GB")

    def _guard_memory(self) -> None:
        """Drop tier chunks when the run's peak memory has grown into the margin left at sizing time."""
        headroom, margin = self._headroom()
        if headroom >= margin / 2:
            return
        before = self._tier.nbytes
        self._truncate_tier(before - (margin - headroom))
        logger.warning(
            f"fastload: rank {self.rank} VRAM headroom {headroom / 1e9:.2f} GB < half the margin "
            f"({margin / 1e9:.1f} GB): GPU tier {before / 1e9:.1f} -> {self._tier.nbytes / 1e9:.1f} GB"
        )

    def _after_batch_served(self) -> None:
        self._served += 1
        if self._tier_deferred is not None and not self._filling:
            kind, budget = self._tier_deferred
            self._tier_deferred = None
            self.build_gpu_cache(self._auto_budget() if kind == "auto" else budget)
        elif not self._grown and self._served >= self.warmup_steps:
            self._grown = True
            if self._filling and self.is_cuda and self.tier_keys:
                logger.info(f"fastload: rank {self.rank} GPU tier deferred until the cache is complete")
                self._tier_deferred = ("auto", None)  # sized from the memory measured then
            else:
                self.build_gpu_cache(self._auto_budget())
        elif self.memory_guard and self.is_cuda and self._tier.tensors:
            self._guard_memory()

    # ------------------------------------------------------------------------------------------
    # incomplete cache (cache_mode="during_training")
    # ------------------------------------------------------------------------------------------
    @property
    def _filling(self) -> bool:
        return self._filler is not None and not self._filler.complete

    @property
    def cache_complete(self) -> bool:
        return self._filler is None or self._filler.complete

    @property
    def fill_stats(self) -> dict[str, Any] | None:
        """Counters of the cache filler (tasks, frames decoded, time waited for frames), None without one."""
        return dict(self._filler.stats, complete=self._filler.complete) if self._filler is not None else None

    def finish_cache(self, timeout: float | None = None) -> bool:
        """Decode every frame the cache still lacks and wait (``timeout`` seconds at most). Returns whether the
        cache is complete. Training does not need this: the filler completes the cache in the background."""
        return self._filler.finish(timeout) if self._filler is not None else True

    def _transient_frames(self, rows: np.ndarray) -> np.ndarray:
        """Frames of the samples ``rows`` that a size-limited cache does not store (sorted, unique)."""
        f = self._needed_frames(rows)
        return np.unique(f[~self._stored[f]])

    def _read(self, key: str, flat: np.ndarray, th: Any) -> np.ndarray:
        """uint8 frames ``flat`` of camera ``key`` (a copy): from the cache, and those it does not store from the
        scratch rows of ``th`` (a :class:`fastload.fill.TransientHandle`)."""
        mm = self.cache.array(key)
        if th is None:
            return mm[flat]
        st = self._stored[flat]
        if st.all():
            return mm[flat]
        out = np.empty((len(flat), *self.cam_shape[key]), dtype=np.uint8)
        if st.any():
            out[st] = mm[flat[st]]
        out[~st] = self._transient.read(key, th, flat[~st])
        return out

    def _needed_frames(self, rows: np.ndarray) -> np.ndarray:
        """Absolute frame indices of every camera frame of the samples ``rows`` (observation history included)."""
        abs_idx = self._index_np[rows]
        ep = self._episode_np[rows]
        frm, to = self._ep_from_np[ep], self._ep_to_np[ep]
        parts = [abs_idx]
        for d in {tuple(self.delta_np[k].tolist()) for k in self.cam_keys if k in self.delta_np}:
            q = np.clip(abs_idx[:, None] + np.asarray(d, dtype=np.int64)[None, :], frm[:, None], to[:, None] - 1)
            parts.append(q.reshape(-1))
        return np.concatenate(parts)

    # ------------------------------------------------------------------------------------------
    # batch assembly
    # ------------------------------------------------------------------------------------------
    def _run_hook(self, key: str, frames: torch.Tensor, row: int | None, ts: list[float]) -> torch.Tensor:
        """Apply the decode_video_frames wrapper to one sample's frames ``[T, C, H, W]`` (float32 in [0, 1])."""
        ds = self.dataset
        ep_idx = int(self._episode_np[row]) if row is not None else int(self._episode_np[0])
        path = ds.root / ds.meta.get_video_file_path(ep_idx, key)
        return apply_hook(self.decode_hook, frames, path, list(ts), ds.tolerance_s, ds.video_backend)

    def _hook_timestamps(self, key: str, q: np.ndarray, ep: np.ndarray) -> list[list[float]]:
        """The timestamps LeRobot passes to decode_video_frames for each sample (episode-shifted)."""
        eps = self.dataset.meta.episodes
        ts = self.cache.timestamps[q.reshape(len(ep), -1)]
        out = []
        for i, e in enumerate(ep):
            off = float(eps[int(e)][f"videos/{key}/from_timestamp"])
            out.append([off + float(t) for t in ts[i]])
        return out

    def _draw_ops(self, batch_index: int, n: int) -> list[dict[str, list]] | None:
        """Transform parameters for the ``n`` samples of a batch (cameras in LeRobot's order)."""
        if self.replayer is None:
            return None
        rec, shapes = self.replayer.record, self.sample_shape
        return self.rng.draw(batch_index, lambda: [{k: rec(shapes[k]) for k in self.cam_keys} for _ in range(n)])

    def _squeeze(self, key: str, img: torch.Tensor) -> torch.Tensor:
        """[B, T, C, H, W] -> LeRobot's per-sample layout (no time axis without delta / for 1 video frame)."""
        if key in self.delta_np:
            return img[:, 0] if key in self.video_keys and img.shape[1] == 1 else img
        return img[:, 0]

    def _prepare(self, rows: np.ndarray, batch_index: int = -1, ops: list | None = None,
                 transient: Any = None) -> _Prepared:
        """CPU work + host->GPU copies for one batch (runs in the background thread). ``transient``: scratch rows
        acquired ahead for the frames a size-limited cache does not store (acquired here otherwise)."""
        th = transient
        try:
            if self._transient is not None:
                if th is None:
                    tf = self._transient_frames(rows)
                    th = self._transient.acquire(tf) if len(tf) else None
                if th is not None:
                    self._transient.wait(th)
            return self._prepare_frames(rows, batch_index, ops, th)
        finally:
            if th is not None:
                self._transient.release(th)  # the frames were copied out of the scratch rows

    def _prepare_frames(self, rows: np.ndarray, batch_index: int, ops: list | None, th: Any) -> _Prepared:
        if self._filling:  # wait until every frame this batch reads is decoded and stored
            self._filler.ensure(self._needed_frames(rows))
        tier = self._tier
        abs_idx = self._index_np[rows]
        ep = self._episode_np[rows]
        frm, to = self._ep_from_np[ep], self._ep_to_np[ep]
        tasks = [self.task_names[t] for t in self._task_np[rows]] if self._task_np is not None else []
        subtasks = ([self.subtask_names[t] for t in self._subtask_np[rows]]
                    if self._subtask_np is not None else None)

        stream_ctx = torch.cuda.stream(self._copy_stream) if self.is_cuda else _null()
        dev_ctx = torch.cuda.device(self.device) if self.is_cuda else _null()
        keep: list[torch.Tensor] = []

        def to_dev(a: np.ndarray | torch.Tensor, pin: bool = False) -> torch.Tensor:
            t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
            if not self.is_cuda:
                return t
            if pin:
                t = t.pin_memory()
                keep.append(t)
            return t.to(self.device, non_blocking=True)

        cams: dict[str, dict[str, Any]] = {}
        with dev_ctx, stream_ctx:
            rows_dev = to_dev(rows)
            for key in self.cam_keys:
                if key in self.delta_np:
                    q = np.clip(abs_idx[:, None] + self.delta_np[key][None, :], frm[:, None], to[:, None] - 1)
                else:
                    q = abs_idx[:, None]
                t = q.shape[1]
                flat = q.reshape(-1)
                entry: dict[str, Any] = {"t": t, "n": flat.shape[0]}
                if key in self.hook_keys:
                    entry["ts"] = self._hook_timestamps(key, q, ep)
                if self.cpu_augment:
                    # LeRobot's exact image path on the CPU: uint8 -> float32 / 255 -> decode hook -> squeeze ->
                    # transforms (here if the parameters are known, else in _finalize on the main thread).
                    c, h, w = self.cam_shape[key]
                    raw = torch.from_numpy(np.ascontiguousarray(self._read(key, flat, th)))
                    x = (raw.to(torch.float32) / 255).view(len(rows), t, c, h, w)
                    if key in self.hook_keys:
                        x = torch.stack([self._run_hook(key, x[i], int(rows[i]), entry["ts"][i]) for i in range(len(rows))])
                    x = self._squeeze(key, x)
                    if ops is not None:
                        x = torch.stack([TransformReplayer.apply(x[i], ops[i][key]) for i in range(len(rows))])
                        entry["final"] = to_dev(x, pin=True)
                    else:
                        entry["cpu_img"] = x
                    cams[key] = entry
                    continue
                if key in self.hook_keys and self.hook_device == "cpu":
                    # Exactly LeRobot's data path: uint8 -> float32 / 255 -> wrapper, per sample, on the CPU.
                    c, h, w = self.cam_shape[key]
                    raw = torch.from_numpy(np.ascontiguousarray(self._read(key, flat, th)))
                    x = (raw.to(torch.float32) / 255).view(len(rows), t, c, h, w)
                    y = torch.stack([self._run_hook(key, x[i], int(rows[i]), entry["ts"][i]) for i in range(len(rows))])
                    entry["post"] = to_dev(y, pin=True)
                    cams[key] = entry
                    continue
                slot = tier.slot[flat] if tier.slot is not None else np.full(flat.shape, -1, dtype=np.int64)
                host_pos = np.nonzero(slot < 0)[0]
                gpu_pos = np.nonzero(slot >= 0)[0]
                if len(host_pos):
                    data = self._read(key, flat[host_pos], th)  # memmap gather -> [h, C, H, W]
                    entry["host_pos"] = to_dev(host_pos)
                    entry["host_data"] = to_dev(data, pin=True)
                if len(gpu_pos):
                    pos_in_tier = slot[gpu_pos]
                    chunk_of = pos_in_tier // tier.chunk_frames
                    groups = []
                    for cid in np.unique(chunk_of):
                        sel = chunk_of == cid
                        groups.append((int(cid), to_dev(gpu_pos[sel]), to_dev(pos_in_tier[sel] % tier.chunk_frames)))
                    entry["gpu"] = groups
                cams[key] = entry
            event = None
            if self.is_cuda:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
        return _Prepared(rows=rows, tier=tier, rows_dev=rows_dev, tasks=tasks,
                         subtasks=subtasks, cams=cams, event=event, keepalive=keep, batch_index=batch_index,
                         ops=ops)

    def _sync(self, prep: _Prepared) -> None:
        """Make the training stream wait for the side-stream copies of ``prep``."""
        if not self.is_cuda:
            return
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(prep.event)
        for entry in prep.cams.values():
            for name in ("host_pos", "host_data", "post", "final"):
                if name in entry:
                    entry[name].record_stream(cur)
            for _, pos, off in entry.get("gpu", ()):
                pos.record_stream(cur)
                off.record_stream(cur)
        prep.rows_dev.record_stream(cur)

    def _finalize(self, prep: _Prepared, ops: list[dict[str, list]] | None = None) -> dict[str, Any]:
        """GPU-side gathers on the training stream; returns the collated batch."""
        if prep.tier.released:  # its GPU tier was freed or shrunk: rebuild the batch from the host tier
            prep = self._prepare(prep.rows, prep.batch_index, prep.ops)
        self._sync(prep)
        tier = prep.tier

        rows = prep.rows_dev
        n_b = len(prep.rows)
        batch: dict[str, Any] = {}
        abs_idx = self.columns["index"][rows]
        ep = self.columns["episode_index"][rows]
        frm, to = self._ep_from[ep], self._ep_to[ep]

        for name, col in self.columns.items():
            if name in self.delta:
                q = abs_idx[:, None] + self.delta[name][None, :]
                batch[f"{name}_is_pad"] = (q < frm[:, None]) | (q >= to[:, None])
                qc = torch.minimum(torch.maximum(q, frm[:, None]), to[:, None] - 1)
                rel = self._rel_of_abs[qc] if self._rel_of_abs is not None else qc
                batch[name] = col[rel]
            else:
                batch[name] = col[rows]

        if self.replayer is not None and ops is None and prep.ops is None:
            raise RuntimeError("fastload: image transforms are enabled but no parameters were drawn for this batch")
        for key in self.cam_keys:
            entry = prep.cams[key]
            t = entry["t"]
            if "final" in entry or "cpu_img" in entry:  # augment_device="cpu": LeRobot's CPU path
                if key in self.delta:
                    q = abs_idx[:, None] + self.delta[key][None, :]
                    batch[f"{key}_is_pad"] = (q < frm[:, None]) | (q >= to[:, None])
                if "final" in entry:
                    batch[key] = entry["final"]
                else:  # num_workers=0: parameters come from the process RNG at LeRobot's time (main thread)
                    x = entry["cpu_img"]
                    x = torch.stack([TransformReplayer.apply(x[i], ops[i][key]) for i in range(n_b)])
                    batch[key] = x.to(self.device)
                continue
            if "post" in entry:
                img = entry["post"]  # [B, T, C, H, W] float32, already through the decode hook on the CPU
            else:
                c, h, w = self.cam_shape[key]
                out = torch.empty((entry["n"], c, h, w), dtype=torch.uint8, device=self.device)
                if "host_pos" in entry:
                    out.index_copy_(0, entry["host_pos"], entry["host_data"])
                for cid, pos, off in entry.get("gpu", ()):
                    out.index_copy_(0, pos, tier.tensors[key][cid].index_select(0, off))
                # LeRobot's decoders compute float32(x) / 255 on CPU. On CUDA, dividing by a Python scalar is
                # turned into x * (1/255) (1-ulp differences); a 0-dim CUDA tensor keeps a true division.
                img = (out.to(torch.float32) / self._div255).view(n_b, t, c, h, w)
                if key in self.hook_keys:  # decode hook on the device
                    img = torch.stack([self._run_hook(key, img[i], int(prep.rows[i]), entry["ts"][i])
                                       for i in range(n_b)])
            if key in self.delta:
                q = abs_idx[:, None] + self.delta[key][None, :]
                batch[f"{key}_is_pad"] = (q < frm[:, None]) | (q >= to[:, None])
                if key in self.video_keys and t == 1:
                    img = img[:, 0]  # LeRobot squeezes single-timestamp video queries
            else:
                img = img[:, 0]
            if ops is not None:  # per sample, with the parameters LeRobot's worker would have drawn
                img = torch.stack([TransformReplayer.apply(x, ops[i][key]) for i, x in enumerate(img)])
            batch[key] = img

        if self._task_np is not None:
            batch["task"] = prep.tasks
        if prep.subtasks is not None:
            batch["subtask"] = prep.subtasks
        return batch

    def close(self) -> None:
        """Stop prefetch threads and free the GPU tier. Called automatically at interpreter exit."""
        if self._filler is not None:  # first: a prefetch thread waiting for frames stops waiting
            self._filler.close()
        for it in list(self._iterators):
            it.close()
        self.release_gpu_cache()
        _ACTIVE_LOADERS.discard(self)

    def get_batch(self, rows: list[int] | np.ndarray, ops: list[dict[str, list]] | None = None) -> dict[str, Any]:
        """Assemble the batch for explicit dataset rows (synchronous; used by tests and benchmarks).

        With image transforms enabled, ``ops`` are the per-sample parameters; if omitted they are drawn
        from the loader's RNG as batch 0.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if ops is None and self.replayer is not None:
            ops = self._draw_ops(0, len(rows))
        return self._finalize(self._prepare(rows), ops)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _EpochIterator:
    def __init__(self, loader: FastLoader, epoch: int):
        self.loader = loader
        if loader.order == "lerobot":
            # Iterating the (accelerate-prepared) index DataLoader performs exactly the RNG draws LeRobot's
            # DataLoader makes when an epoch starts (RNG sync, base_seed, sampler) and yields this rank's
            # batches. Row indices carry no further randomness, so the whole epoch is listed at once.
            self.batches = [np.asarray(b, dtype=np.int64) for b in loader._batch_source()]
        else:
            rows = loader.local_rows
            if loader.shuffle:
                g = np.random.default_rng(loader.seed + 1_000_003 * epoch + 7919 * loader.rank)
                rows = rows[g.permutation(len(rows))]
            bs = loader.batch_size
            n_batches = len(rows) // bs if loader.drop_last else math.ceil(len(rows) / bs)
            self.batches = [rows[i * bs:(i + 1) * bs] for i in range(n_batches)]
        self._served = 0
        self._ops: dict[int, list] = {}
        # augment_device="cpu": the prefetch thread applies the transforms, so it needs the parameters up front.
        # Worker RNG streams (num_workers > 0) and the private stream do not depend on when they are drawn,
        # so the whole epoch is drawn here; with num_workers=0 LeRobot draws from the process RNG during
        # each step, so those batches are transformed on the main thread instead.
        self._pre_ops = None
        rng = loader.rng
        if loader.cpu_augment and (rng.private_seed is not None or rng.num_workers > 0):
            self._pre_ops = [loader._draw_ops(i, len(b)) for i, b in enumerate(self.batches)]
        # While the cache is being filled, frames of the next batches are requested ahead (decoded in parallel by
        # the filler's processes) and a few more prepared batches absorb decoding jitter.
        self._fill = loader._filler if loader._filling else None
        self._requested = 0
        # Size-limited cache: scratch rows for the frames it does not store, acquired (and decoding queued) a few
        # batches ahead; batch index -> TransientHandle until the batch is prepared.
        self._transient = loader._transient
        self._t_requested = 0
        self._handles: dict[int, Any] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=loader.prefetch if self._fill is None and self._transient is None
                                               else max(loader.prefetch, 4))
        self._stop = threading.Event()
        self._pending: list[tuple[int, np.ndarray]] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._produce, daemon=True, name="fastload-prefetch")
        self._thread.start()

    def _put(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for i, rows in enumerate(self.batches):
                if self._fill is not None and not self._fill.complete:
                    ahead = min(len(self.batches), i + self.loader._fill_lookahead)
                    while self._requested < ahead:
                        self._fill.request(self.loader._needed_frames(self.batches[self._requested]))
                        self._requested += 1
                if self._transient is not None:
                    self._t_requested = max(self._t_requested, i)
                    ahead = min(len(self.batches), i + self.loader._transient_lookahead)
                    while self._t_requested < ahead:
                        tf = self.loader._transient_frames(self.batches[self._t_requested])
                        if len(tf):
                            h = self._transient.acquire(tf, block=False)
                            if h is None:
                                break  # scratch rows busy: ask again after this batch
                            self._handles[self._t_requested] = h
                        self._t_requested += 1
                pre = self._pre_ops[i] if self._pre_ops is not None else None
                if self._stop.is_set():
                    return
                prep = self.loader._prepare(rows, i, pre, self._handles.pop(i, None))
                if not self._put(prep):
                    return
            self._put(StopIteration)
        except BaseException as e:  # surface errors in the training thread
            self._put(e)
        finally:
            for h in list(self._handles.values()):
                self._transient.release(h)
            self._handles.clear()

    def close(self) -> None:
        """Stop the prefetch thread (idempotent)."""
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)

    def _invalidate(self) -> None:
        """Drop prepared batches that reference a released GPU tier (they are re-prepared on use)."""
        drained = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for item in drained:
            if isinstance(item, _Prepared):
                self._pending.append((item.batch_index, item.rows))
            else:
                self._queue.put(item)

    def __iter__(self):
        return self

    def _ops_for(self, k: int) -> list | None:
        """Transform parameters of batch ``k``, drawn in the main thread in LeRobot's order.

        accelerate's DataLoaderShard fetches batch k+1 before yielding batch k, so with
        ``num_workers=0`` LeRobot draws batch k+1's parameters during the step that returns batch k.
        """
        if self.loader.replayer is None or self._pre_ops is not None:
            return None
        for j in (k, k + 1):
            if j < len(self.batches) and j not in self._ops:
                self._ops[j] = self.loader._draw_ops(j, len(self.batches[j]))
        return self._ops.pop(k)

    def __next__(self) -> dict[str, Any]:
        k = self._served
        if self._pending:
            idx, rows = self._pending.pop(0)
            assert idx == k, (idx, k)
            pre = self._pre_ops[idx] if self._pre_ops is not None else None
            batch = self.loader._finalize(self.loader._prepare(rows, idx, pre), self._ops_for(k))
        else:
            item = self._queue.get()
            if item is StopIteration:
                self._thread.join()
                raise StopIteration
            if isinstance(item, BaseException):
                self._stop.set()
                raise item
            batch = self.loader._finalize(item, self._ops_for(k))
        self._served += 1
        self.loader._after_batch_served()
        return batch

    def __del__(self):
        self._stop.set()


_ACCELERATE_HOOKED = False


def _install_accelerate_hook() -> None:
    """Make ``accelerator.prepare(fast_loader)`` prepare its index DataLoader exactly like a DataLoader."""
    global _ACCELERATE_HOOKED
    if _ACCELERATE_HOOKED:
        return
    try:
        from accelerate import Accelerator
    except ImportError:  # pragma: no cover
        return
    orig = Accelerator._prepare_one

    def _prepare_one(self, obj, first_pass=False, device_placement=None):
        if first_pass and isinstance(obj, FastLoader) and obj.index_loader is not None:
            prepared = self.prepare_data_loader(obj.index_loader, device_placement=device_placement)
            obj.attach_prepared_index_loader(prepared)
            return obj
        return orig(self, obj, first_pass=first_pass, device_placement=device_placement)

    Accelerator._prepare_one = _prepare_one
    _ACCELERATE_HOOKED = True


def _close_all() -> None:
    for ld in list(_ACTIVE_LOADERS):
        try:
            ld.close()
        except Exception:  # pragma: no cover - best effort at interpreter exit
            pass


atexit.register(_close_all)
