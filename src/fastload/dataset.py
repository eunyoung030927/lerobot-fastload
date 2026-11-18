"""LeRobotDataset that reads camera frames from a fastload cache instead of decoding video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from fastload.cache import FrameCache
from fastload.decode_hook import active_hook, apply_hook


class CachedLeRobotDataset(LeRobotDataset):
    """Drop-in LeRobotDataset whose ``_query_videos`` reads uint8 frames from the cache.

    Everything else (parquet rows, delta timestamps, padding, image transforms, task strings) is
    LeRobot's own code, so ``dataset[i]`` is identical to the original dataset's ``dataset[i]``.
    Works with a regular ``torch.utils.data.DataLoader`` (any ``num_workers``) and is also the input
    of :class:`fastload.loader.FastLoader`.
    """

    fastload_cache: FrameCache

    @classmethod
    def from_dataset(cls, dataset: LeRobotDataset, cache_dir: str | Path) -> "CachedLeRobotDataset":
        """Convert an existing LeRobotDataset in place (keeps all of its configuration)."""
        if isinstance(dataset, cls):
            return dataset
        if type(dataset) is not LeRobotDataset:
            raise TypeError(f"Only plain LeRobotDataset is supported, got {type(dataset).__name__}.")
        cache = FrameCache(cache_dir)
        cache.check_compatible(dataset, keys=list(dataset.meta.video_keys))
        dataset.__class__ = cls
        dataset._fastload_attach(cache)
        return dataset  # type: ignore[return-value]

    @classmethod
    def from_repo(cls, repo_id: str, cache_dir: str | Path, **kwargs: Any) -> "CachedLeRobotDataset":
        """Same arguments as LeRobotDataset (root, episodes, delta_timestamps, image_transforms, ...)."""
        return cls.from_dataset(LeRobotDataset(repo_id, **kwargs), cache_dir)

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
                continue
            rel = _nearest(ep_ts, np.asarray(query_ts, dtype=np.float64), self.tolerance_s, key, ep_idx)
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
