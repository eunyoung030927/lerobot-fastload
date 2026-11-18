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

from fastload.dataset import CachedLeRobotDataset
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


@dataclass
class _GpuTier:
    slot: np.ndarray | None = None  # [total_frames] int64, -1 = not on GPU (CPU copy only)
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)  # key -> [M, C, H, W] uint8
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
        placement: ``"auto"`` - grow a GPU tier after ``warmup_steps`` batches, sized from the measured
            peak memory of the run; ``"gpu"`` - put this rank's whole shard on the GPU now;
            ``"host"`` - never use VRAM for frames.
        warmup_steps: batches to serve before measuring memory (auto mode). Must include at least one
            optimizer step so optimizer state is counted.
        gpu_margin_gb: VRAM left free on top of the measured peak (default: max(2 GB, 5% of the GPU)).
        max_gpu_cache_gb: upper bound on the GPU tier size.
        prefetch: batches prepared ahead by the background thread.
        seed: shuffling seed (default: derived from torch's initial seed). Shards are seed-independent.
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
    ):
        if not isinstance(dataset, CachedLeRobotDataset):
            if cache_dir is None:
                raise ValueError("dataset is not a CachedLeRobotDataset; pass cache_dir=...")
            dataset = CachedLeRobotDataset.from_dataset(dataset, cache_dir)
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        self.order = order
        self.num_workers = int(num_workers or 0)
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

        # ---- image transforms: parameters drawn on the CPU, kernels applied on the device -------
        tf = dataset.image_transforms
        self.replayer = TransformReplayer(tf) if tf is not None else None
        self.sample_shape = {}
        for key in self.cam_keys:
            c, h, w = self.cam_shape[key]
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
        self._served = 0
        self._epoch = 0
        self._grown = self.placement != "auto"
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

    def _auto_budget(self) -> int:
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)
        current = torch.cuda.memory_reserved(self.device)
        margin = (self.gpu_margin_gb * 1e9 if self.gpu_margin_gb is not None else max(2e9, 0.05 * total))
        budget = int(free - max(0, peak - current) - margin)
        logger.info(
            f"fastload: rank {self.rank} memory after {self._served} steps: free {free / 1e9:.1f} GB, "
            f"peak reserved {peak / 1e9:.1f} GB, now reserved {current / 1e9:.1f} GB, margin {margin / 1e9:.1f} GB "
            f"-> frame budget {max(budget, 0) / 1e9:.1f} GB"
        )
        return budget

    def build_gpu_cache(self, budget_bytes: int | None) -> None:
        """(Re)build the GPU tier with at most ``budget_bytes`` (None = whole local shard)."""
        if not self.is_cuda:
            return
        self.release_gpu_cache()
        need = len(self.local_rows) * self.frame_bytes
        budget = need if budget_bytes is None else min(int(budget_bytes), need)
        if self.max_gpu_cache_gb is not None:
            budget = min(budget, int(self.max_gpu_cache_gb * 1e9))
        if budget < self.frame_bytes:
            logger.info(f"fastload: rank {self.rank} no VRAM left for frames; host tier only")
            return
        t0 = time.perf_counter()
        # Whole episodes, in a seed-fixed random order, until the budget is used.
        order = np.random.default_rng(self.seed + 7919 * self.rank).permutation(self.local_episodes)
        ranges, frames = [], 0
        for ep in order:
            a, b = int(self._ep_from_np[ep]), int(self._ep_to_np[ep])
            if (frames + b - a) * self.frame_bytes > budget:
                break
            ranges.append((a, b))
            frames += b - a
        if frames == 0:
            return
        slot = np.full(int(self.dataset.meta.total_frames), -1, dtype=np.int64)
        tensors = {}
        try:
            for key in self.cam_keys:
                c, h, w = self.cam_shape[key]
                tensors[key] = torch.empty((frames, c, h, w), dtype=torch.uint8, device=self.device)
            mm = {k: self.cache.array(k) for k in self.cam_keys}
            pos = 0
            for a, b in ranges:
                slot[a:b] = np.arange(pos, pos + b - a)
                for key in self.cam_keys:
                    with warnings.catch_warnings():  # read-only memmap slice; it is only copied from
                        warnings.simplefilter("ignore", UserWarning)
                        src = torch.from_numpy(mm[key][a:b])
                    tensors[key][pos:pos + b - a].copy_(src)
                pos += b - a
            torch.cuda.synchronize(self.device)
        except torch.OutOfMemoryError:
            tensors.clear()
            torch.cuda.empty_cache()
            logger.warning(f"fastload: rank {self.rank} OOM while building a {budget / 1e9:.1f} GB GPU tier; "
                           "retrying with half")
            self.build_gpu_cache(budget // 2)
            return
        self._tier = _GpuTier(slot=slot, tensors=tensors, nbytes=frames * self.frame_bytes, frames=frames)
        logger.info(
            f"fastload: rank {self.rank} GPU tier {frames * self.frame_bytes / 1e9:.1f} GB = "
            f"{100 * self.gpu_cache_fraction:.0f}% of local frames, filled in {time.perf_counter() - t0:.1f}s"
        )

    def release_gpu_cache(self) -> None:
        old = self._tier
        if old.tensors:
            self._tier = _GpuTier()
            old.released = True
            old.tensors.clear()  # frees VRAM even if prepared batches still reference `old`
            for it in list(self._iterators):
                it._invalidate()
            if self.is_cuda:
                torch.cuda.empty_cache()

    def shrink_gpu_cache(self, fraction: float = 0.5) -> None:
        """Free VRAM (e.g. after an OOM): keep ``fraction`` of the current GPU tier."""
        keep = int(self._tier.nbytes * fraction)
        self.release_gpu_cache()
        if keep > 0:
            self.build_gpu_cache(keep)
        self._grown = True

    def _after_batch_served(self) -> None:
        self._served += 1
        if not self._grown and self._served >= self.warmup_steps:
            self._grown = True
            self.build_gpu_cache(self._auto_budget())

    # ------------------------------------------------------------------------------------------
    # batch assembly
    # ------------------------------------------------------------------------------------------
    def _draw_ops(self, batch_index: int, n: int) -> list[dict[str, list]] | None:
        """Transform parameters for the ``n`` samples of a batch (cameras in LeRobot's order)."""
        if self.replayer is None:
            return None
        rec, shapes = self.replayer.record, self.sample_shape
        return self.rng.draw(batch_index, lambda: [{k: rec(shapes[k]) for k in self.cam_keys} for _ in range(n)])

    def _prepare(self, rows: np.ndarray, batch_index: int = -1) -> _Prepared:
        """CPU work + host->GPU copies for one batch (runs in the background thread)."""
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

        def to_dev(a: np.ndarray, pin: bool = False) -> torch.Tensor:
            t = torch.from_numpy(np.ascontiguousarray(a))
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
                    q = abs_idx
                flat = q.reshape(-1)
                slot = tier.slot[flat] if tier.slot is not None else np.full(flat.shape, -1, dtype=np.int64)
                host_pos = np.nonzero(slot < 0)[0]
                gpu_pos = np.nonzero(slot >= 0)[0]
                entry: dict[str, Any] = {"shape": q.shape, "n": flat.shape[0]}
                if len(host_pos):
                    data = self.cache.array(key)[flat[host_pos]]  # memmap gather -> [h, C, H, W]
                    entry["host_pos"] = to_dev(host_pos)
                    entry["host_data"] = to_dev(data, pin=True)
                if len(gpu_pos):
                    entry["gpu_pos"] = to_dev(gpu_pos)
                    entry["gpu_slot"] = to_dev(slot[gpu_pos])
                cams[key] = entry
            event = None
            if self.is_cuda:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
        return _Prepared(rows=rows, tier=tier, rows_dev=rows_dev, tasks=tasks,
                         subtasks=subtasks, cams=cams, event=event, keepalive=keep, batch_index=batch_index)

    def _sync(self, prep: _Prepared) -> None:
        """Make the training stream wait for the side-stream copies of ``prep``."""
        if not self.is_cuda:
            return
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(prep.event)
        for entry in prep.cams.values():
            for name in ("host_pos", "host_data", "gpu_pos", "gpu_slot"):
                if name in entry:
                    entry[name].record_stream(cur)
        prep.rows_dev.record_stream(cur)

    def _finalize(self, prep: _Prepared, ops: list[dict[str, list]] | None = None) -> dict[str, Any]:
        """GPU-side gathers on the training stream; returns the collated batch."""
        if prep.tier.released:  # its GPU tier was freed (shrink after OOM): rebuild from the host tier
            prep = self._prepare(prep.rows, prep.batch_index)
        self._sync(prep)
        tier = prep.tier

        rows = prep.rows_dev
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

        if self.replayer is not None and ops is None:
            raise RuntimeError("fastload: image transforms are enabled but no parameters were drawn for this batch")
        for key in self.cam_keys:
            entry = prep.cams[key]
            c, h, w = self.cam_shape[key]
            out = torch.empty((entry["n"], c, h, w), dtype=torch.uint8, device=self.device)
            if "host_pos" in entry:
                out.index_copy_(0, entry["host_pos"], entry["host_data"])
            if "gpu_pos" in entry:
                out.index_copy_(0, entry["gpu_pos"], tier.tensors[key].index_select(0, entry["gpu_slot"]))
            # LeRobot's decoders compute float32(x) / 255 on CPU. On CUDA, dividing by a Python scalar is
            # turned into x * (1/255) (1-ulp differences); a 0-dim CUDA tensor keeps a true division.
            img = out.to(torch.float32) / self._div255
            img = img.view(*entry["shape"], c, h, w)
            if key in self.delta:
                q = abs_idx[:, None] + self.delta[key][None, :]
                batch[f"{key}_is_pad"] = (q < frm[:, None]) | (q >= to[:, None])
                if key in self.video_keys and img.shape[1] == 1:
                    img = img.squeeze(1)  # LeRobot squeezes single-timestamp video queries
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
        self._queue: queue.Queue = queue.Queue(maxsize=loader.prefetch)
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
                if self._stop.is_set() or not self._put(self.loader._prepare(rows, i)):
                    return
            self._put(StopIteration)
        except BaseException as e:  # surface errors in the training thread
            self._put(e)

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
        if self.loader.replayer is None:
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
            batch = self.loader._finalize(self.loader._prepare(rows, idx), self._ops_for(k))
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
