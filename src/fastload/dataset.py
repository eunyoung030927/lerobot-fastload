"""LeRobotDataset that reads camera frames from a fastload cache instead of decoding video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastload._lerobot_version import check_lerobot_version

check_lerobot_version()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from fastload import _lr  # noqa: E402
from fastload.cache import DEFAULT_MAX_CACHE_GB, DEFAULT_MIN_FREE_DISK_GB, FrameCache  # noqa: E402
from fastload.decode_hook import active_hook, apply_hook  # noqa: E402


class CachedLeRobotDataset(LeRobotDataset):
    """Drop-in LeRobotDataset whose ``_query_videos`` reads uint8 frames from the cache.

    Everything else (parquet rows, delta timestamps, padding, image transforms, task strings) is
    LeRobot's own code, so ``dataset[i]`` is identical to the original dataset's ``dataset[i]``.
    Works with a regular ``torch.utils.data.DataLoader`` (any ``num_workers``) and is also the input
    of :class:`fastload.loader.FastLoader`.
    """

    fastload_cache: FrameCache

    @classmethod
    def from_dataset(cls, dataset: LeRobotDataset, cache_dir: str | Path, cache_mode: str = "prebuilt",
                     min_free_disk_gb: float | None = DEFAULT_MIN_FREE_DISK_GB,
                     build_workers: int | None = None,
                     max_cache_gb: float | None = DEFAULT_MAX_CACHE_GB) -> "CachedLeRobotDataset":
        """Convert an existing LeRobotDataset in place (keeps all of its configuration).

        ``cache_mode``: ``"prebuilt"`` (the cache must be complete), ``"build_first"`` (build a missing / incomplete
        cache now, with ``build_workers`` decoder processes) or ``"during_training"`` (allocate it now; a
        :class:`fastload.FastLoader` fills it while it serves batches; frames not filled yet are decoded by LeRobot
        when the dataset is indexed directly). See :func:`fastload.fill.ensure_cache` for several processes per node.
        ``max_cache_gb``: size limit of a cache created here; frames a size-limited cache does not store are decoded
        by LeRobot when the dataset is indexed (by FastLoader's decoder processes in a FastLoader).
        """
        if isinstance(dataset, cls):
            return dataset
        if type(dataset) is not LeRobotDataset:
            raise TypeError(f"Only plain LeRobotDataset is supported, got {type(dataset).__name__}.")
        if cache_mode != "prebuilt":
            from fastload.fill import ensure_cache

            ensure_cache(cache_mode, cache_dir, lambda: _lr.dataset_source(dataset), min_free_disk_gb,
                         num_workers=build_workers, max_cache_gb=max_cache_gb)
        cache = FrameCache(cache_dir, allow_incomplete=cache_mode == "during_training", repo_id=dataset.repo_id,
                           root=dataset.root)
        cache.check_compatible(dataset, keys=list(dataset.meta.video_keys))
        dataset.__class__ = cls
        dataset._fastload_attach(cache)
        return dataset  # type: ignore[return-value]

    @classmethod
    def from_repo(cls, repo_id: str, cache_dir: str | Path, cache_mode: str = "prebuilt",
                  **kwargs: Any) -> "CachedLeRobotDataset":
        """Same arguments as LeRobotDataset (root, episodes, delta_timestamps, image_transforms, ...)."""
        return cls.from_dataset(LeRobotDataset(repo_id, **kwargs), cache_dir, cache_mode=cache_mode)

    def _fastload_attach(self, cache: FrameCache) -> None:
        self.fastload_cache = cache
        episodes = self.meta.episodes
        self._fl_ep_from = np.asarray(episodes["dataset_from_index"], dtype=np.int64)
        self._fl_ep_to = np.asarray(episodes["dataset_to_index"], dtype=np.int64)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        cache = self.fastload_cache
        start, end = int(self._fl_ep_from[ep_idx]), int(self._fl_ep_to[ep_idx])
        ep_ts = cache.timestamps[start:end]
        item: dict[str, torch.Tensor] = {}
        fallback: dict[str, list[float]] = {}
        for key, query_ts in query_timestamps.items():
            if key not in cache.keys:
                fallback[key] = query_ts
                item[key] = None  # type: ignore[assignment]  # keeps LeRobot's key order
                continue
            rel = _nearest(ep_ts, np.asarray(query_ts, dtype=np.float64), self.tolerance_s, key, ep_idx)
            avail = cache.available()  # None when every frame is in the cache
            if avail is not None and not bool(np.all(avail[start + rel])):
                # frames not in the cache (a size-limited cache does not store them, or cache_mode="during_training"
                # has not filled them yet): LeRobot's own decoding
                fallback[key] = query_ts
                item[key] = None  # type: ignore[assignment]  # keeps LeRobot's key order
                continue
            frames = np.asarray(cache.array(key)[start + rel])  # [T, C, H, W] uint8 copy
            # Same arithmetic as LeRobot's decoders: uint8 -> float32, then / 255.
            frames = torch.from_numpy(frames).to(torch.float32) / 255
            hook = active_hook()
            if hook is not None:  # a wrapper around decode_video_frames (e.g. a resize) runs as with LeRobot
                ep = self.meta.episodes[ep_idx]
                shifted = [ep[f"videos/{key}/from_timestamp"] + ts for ts in query_ts]
                path = self.root / self.meta.get_video_file_path(ep_idx, key)
                frames = apply_hook(hook, frames, path, shifted, self.tolerance_s, self.video_backend)
            item[key] = frames.squeeze(0)
        if fallback:
            item.update(super()._query_videos(fallback, ep_idx))
        return item


def _nearest(ep_ts: np.ndarray, query: np.ndarray, tolerance_s: float, key: str, ep_idx: int) -> np.ndarray:
    """Index (within the episode) of the frame whose timestamp is closest to each query."""
    pos = np.clip(np.searchsorted(ep_ts, query), 1, len(ep_ts) - 1) if len(ep_ts) > 1 else np.zeros(
        len(query), dtype=np.int64)
    if len(ep_ts) > 1:
        left = ep_ts[pos - 1]
        right = ep_ts[pos]
        pos = np.where(np.abs(query - left) <= np.abs(right - query), pos - 1, pos)
    err = np.abs(ep_ts[pos].astype(np.float64) - query)
    if np.any(err > tolerance_s):
        raise ValueError(
            f"fastload: no cached frame within tolerance_s={tolerance_s} for {key} in episode {ep_idx} "
            f"(query {query[err.argmax()]:.6f}s, nearest {ep_ts[pos[err.argmax()]]:.6f}s)"
        )
    return pos
