"""Read access to a fastload decoded-frame cache.

Cache layout (one directory per LeRobot dataset)::

    <cache_dir>/
      fastload_meta.json     format version, source dataset fingerprint, per-key shapes
      timestamps.npy         float32 [total_frames]  (copy of the dataset `timestamp` column)
      <camera_key>.npy       uint8   [total_frames, C, H, W]  indexed by the absolute frame `index`
      progress/ep_XXXXXX.done  per-episode markers written while building (allows resuming)
      COMPLETE               written last; a cache without it is never used for training

Pixels are stored exactly as LeRobot decodes them (uint8 before the `/255` conversion), so
`uint8 -> float32 / 255` reproduces LeRobotDataset outputs bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT_VERSION = 1
META_FILE = "fastload_meta.json"
COMPLETE_FILE = "COMPLETE"
TIMESTAMPS_FILE = "timestamps.npy"
PROGRESS_DIR = "progress"


def key_to_filename(key: str) -> str:
    """Camera keys contain dots (``observation.images.front``); keep them readable but path-safe."""
    return key.replace("/", "__") + ".npy"


def info_fingerprint(dataset_root: str | Path) -> str:
    """SHA-256 of the dataset's ``meta/info.json``; ties a cache to one dataset version."""
    return hashlib.sha256((Path(dataset_root) / "meta" / "info.json").read_bytes()).hexdigest()


class FrameCache:
    """Lazily opened, read-only view of a built cache.

    Memory maps are opened on first use in each process and are dropped when the object is
    pickled, so the cache can be passed to DataLoader workers (fork or spawn) without copying
    the arrays.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No fastload cache at {self.cache_dir} ({META_FILE} missing). "
                "Build it first with `python -m fastload.build_cache`."
            )
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        if self.meta.get("format_version") != CACHE_FORMAT_VERSION:
            raise RuntimeError(
                f"Cache format {self.meta.get('format_version')} != supported {CACHE_FORMAT_VERSION}; rebuild it."
            )
        if not (self.cache_dir / COMPLETE_FILE).exists():
            raise RuntimeError(
                f"Cache at {self.cache_dir} is incomplete (no {COMPLETE_FILE}). "
                "Re-run `python -m fastload.build_cache` with the same arguments to finish it."
            )
        self.keys: list[str] = list(self.meta["keys"])
        self.total_frames: int = int(self.meta["total_frames"])
        self._arrays: dict[str, np.ndarray] | None = None
        self._timestamps: np.ndarray | None = None
        self._pid: int | None = None

    # -- pickling: never ship the memory maps themselves ------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_timestamps"] = None
        state["_pid"] = None
        return state

    def _ensure_open(self) -> None:
        if self._arrays is not None and self._pid == os.getpid():
            return
        self._arrays = {
            key: np.load(self.cache_dir / self.meta["keys"][key]["file"], mmap_mode="r") for key in self.keys
        }
        self._timestamps = np.load(self.cache_dir / TIMESTAMPS_FILE, mmap_mode="r")
        self._pid = os.getpid()

    def array(self, key: str) -> np.ndarray:
        self._ensure_open()
        return self._arrays[key]  # type: ignore[index]

    @property
    def timestamps(self) -> np.ndarray:
        self._ensure_open()
        return self._timestamps  # type: ignore[return-value]

    def frame_shape(self, key: str) -> tuple[int, int, int]:
        return tuple(self.meta["keys"][key]["shape"])  # type: ignore[return-value]

    def frame_nbytes(self, key: str) -> int:
        c, h, w = self.frame_shape(key)
        return c * h * w

    @property
    def nbytes(self) -> int:
        return sum(self.frame_nbytes(k) for k in self.keys) * self.total_frames

    # -- validation -------------------------------------------------------------------------
    def check_compatible(self, dataset: Any, keys: list[str] | None = None) -> None:
        """Raise if this cache was not built from ``dataset`` (same info.json, same frame count)."""
        fp = info_fingerprint(dataset.root)
        if fp != self.meta["info_fingerprint"]:
            raise RuntimeError(
                f"Cache {self.cache_dir} was built from a different version of {dataset.repo_id} "
                "(meta/info.json changed). Rebuild the cache."
            )
        if int(dataset.meta.total_frames) != self.total_frames:
            raise RuntimeError(
                f"Cache has {self.total_frames} frames but the dataset has {dataset.meta.total_frames}."
            )
        missing = [k for k in (keys or []) if k not in self.keys]
        if missing:
            raise RuntimeError(f"Cache {self.cache_dir} has no frames for {missing}; rebuild with these keys.")
