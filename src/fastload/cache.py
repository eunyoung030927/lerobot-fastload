"""Read access to a fastload decoded-frame cache.

Cache layout (one directory per LeRobot dataset)::

    <cache_dir>/
      fastload_meta.json     format version, source dataset fingerprint, per-key shapes (written atomically,
                             after the other files are allocated)
      timestamps.npy         float32 [total_frames]  (copy of the dataset `timestamp` column)
      <camera_key>.npy       uint8   [total_frames, C, H, W]  indexed by the absolute frame `index`
      filled.npy             uint8   [total_frames]  1 = every camera of that frame is written and flushed
      progress/ep_XXXXXX.done  per-episode markers written by build_cache (allows resuming)
      COMPLETE               written last (atomic rename); a cache without it is only used by
                             ``cache_mode="during_training"``, which reads filled frames and decodes the rest

Pixels are stored exactly as LeRobot decodes them (uint8 before the `/255` conversion), so
`uint8 -> float32 / 255` reproduces LeRobotDataset outputs bit for bit. Caches built before ``filled.npy``
existed are still valid (a complete cache does not need it).

Nothing in this module imports LeRobot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT_VERSION = 1
META_FILE = "fastload_meta.json"
COMPLETE_FILE = "COMPLETE"
TIMESTAMPS_FILE = "timestamps.npy"
FILLED_FILE = "filled.npy"
PROGRESS_DIR = "progress"

# How a training run gets its cache (``--fastload.cache_mode``).
CACHE_MODES = ("prebuilt", "build_first", "during_training")
DEFAULT_MIN_FREE_DISK_GB = 20.0


def key_to_filename(key: str) -> str:
    """Camera keys contain dots (``observation.images.front``); keep them readable but path-safe."""
    return key.replace("/", "__") + ".npy"


def info_fingerprint(dataset_root: str | Path) -> str:
    """SHA-256 of the dataset's ``meta/info.json``; ties a cache to one dataset version."""
    return hashlib.sha256((Path(dataset_root) / "meta" / "info.json").read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------
# cache state, atomic markers, disk space
# ------------------------------------------------------------------------------------------------
def cache_status(cache_dir: str | Path) -> str:
    """``"complete"`` (COMPLETE written), ``"partial"`` (meta written, frames missing) or ``"missing"``."""
    d = Path(cache_dir)
    if not (d / META_FILE).exists():
        return "missing"
    return "complete" if (d / COMPLETE_FILE).exists() else "partial"


def layout_ready(cache_dir: str | Path) -> bool:
    """Files allocated and a ``filled`` bitmap present: ``cache_mode="during_training"`` can fill this cache."""
    d = Path(cache_dir)
    return (d / META_FILE).exists() and (d / FILLED_FILE).exists()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` so that readers see either no file or the whole file (rename is atomic)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_complete(cache_dir: str | Path) -> None:
    """Mark a cache complete (idempotent; several processes may do it at the same time)."""
    atomic_write_text(Path(cache_dir) / COMPLETE_FILE, time.strftime("%Y-%m-%d %H:%M:%S"))


def missing_cache_message(cache_dir: str | Path, repo_id: str | None = None, root: str | Path | None = None,
                          partial: bool = False) -> str:
    """The error for ``cache_mode="prebuilt"`` when the cache is missing or incomplete."""
    cmd = (f"python -m fastload.build_cache --repo_id {repo_id or '<repo_id>'}"
           + (f" --root {root}" if root else " [--root <dataset dir>]") + f" --cache_dir {cache_dir}")
    what = (f"The fastload cache at {cache_dir} is incomplete (no {COMPLETE_FILE})" if partial
            else f"No fastload cache at {cache_dir}")
    return (
        f"{what}, and --fastload.cache_mode=prebuilt (the default) needs a complete one. Either\n"
        f"  * build it first (resumable):  {cmd}\n"
        "  * or let training build it:     --fastload.cache_mode=build_first (decode everything, then train)\n"
        "                                  --fastload.cache_mode=during_training (the first epoch decodes the frames "
        "it needs and stores them; later epochs read the cache)"
    )


class DiskSpaceError(RuntimeError):
    """Not enough free disk for the rest of a cache plus the ``min_free_disk_gb`` margin."""


def _existing_ancestor(path: Path) -> Path:
    path = path.absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def allocated_bytes(path: str | Path) -> int:
    """Bytes a file already occupies on disk: the written blocks of a sparse file (Linux), or its size where
    ``st_blocks`` does not exist (Windows: NTFS allocates a file created at full size right away)."""
    st = os.stat(path)
    blocks = getattr(st, "st_blocks", None)
    return min(st.st_size, blocks * 512) if blocks is not None else st.st_size


def cache_file_sizes(meta: dict[str, Any]) -> dict[str, int]:
    """Full size of every file of the cache described by ``meta`` (frame arrays, timestamps, filled bitmap)."""
    n = int(meta["total_frames"])
    sizes = {v["file"]: n * int(np.prod(v["shape"])) for v in meta["keys"].values()}
    sizes[TIMESTAMPS_FILE] = 4 * n
    sizes[FILLED_FILE] = n
    return sizes


def check_free_disk(cache_dir: str | Path, file_sizes: dict[str, int], min_free_disk_gb: float | None) -> None:
    """Raise :class:`DiskSpaceError` unless ``free disk - bytes still to write >= min_free_disk_gb``.

    ``file_sizes``: file name in ``cache_dir`` -> full size in bytes. Files that exist count with what they already
    occupy, so a resumed or partly filled cache only needs space for the rest. Creates nothing.
    """
    if min_free_disk_gb is None or min_free_disk_gb < 0:
        return
    d = Path(cache_dir)
    total = sum(file_sizes.values())
    to_write = 0
    for name, size in file_sizes.items():
        f = d / name
        to_write += max(0, size - allocated_bytes(f)) if f.exists() else size
    free = shutil.disk_usage(_existing_ancestor(d)).free
    if free - to_write < float(min_free_disk_gb) * 1e9:
        def gb(b: float) -> str:
            return f"{b / 1e9:.1f} GB"

        raise DiskSpaceError(
            f"fastload: not enough free disk for the cache at {d}: cache size {gb(total)} ({gb(to_write)} still to "
            f"write), free {gb(free)}; writing it would leave {gb(free - to_write)}, less than "
            f"min_free_disk_gb={min_free_disk_gb:g} (a safety margin kept free, not a cache size limit). Options:\n"
            "  * put the cache on another disk (--fastload.cache_dir / build_cache --cache_dir)\n"
            "  * cache fewer cameras (python -m fastload.build_cache --keys ...; uncached cameras are then decoded by "
            "LeRobot, with --fastload.loader=dataloader)\n"
            "  * lower the margin (--fastload.min_free_disk_gb / build_cache --min_free_disk_gb) if you are sure\n"
            "  * train with plain lerobot-train (decodes video on the fly, needs no cache)"
        )


class FrameCache:
    """Lazily opened, read-only view of a built cache.

    Memory maps are opened on first use in each process and are dropped when the object is
    pickled, so the cache can be passed to DataLoader workers (fork or spawn) without copying
    the arrays.

    ``allow_incomplete=True`` (``cache_mode="during_training"``) also opens a cache whose frames are still being
    filled: :attr:`complete` is False and :meth:`filled` tells which frames are written.
    """

    def __init__(self, cache_dir: str | Path, allow_incomplete: bool = False, repo_id: str | None = None,
                 root: str | Path | None = None):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / META_FILE
        if not meta_path.exists():
            raise FileNotFoundError(missing_cache_message(self.cache_dir, repo_id, root))
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        if self.meta.get("format_version") != CACHE_FORMAT_VERSION:
            raise RuntimeError(
                f"Cache format {self.meta.get('format_version')} != supported {CACHE_FORMAT_VERSION}; rebuild it."
            )
        self.complete = (self.cache_dir / COMPLETE_FILE).exists()
        if not self.complete:
            if not allow_incomplete:
                raise RuntimeError(missing_cache_message(self.cache_dir, repo_id, root, partial=True))
            if not (self.cache_dir / FILLED_FILE).exists():
                raise RuntimeError(f"Cache at {self.cache_dir} is incomplete and has no {FILLED_FILE}, so it "
                                   "cannot be filled during training")
        self.keys: list[str] = list(self.meta["keys"])
        self.total_frames: int = int(self.meta["total_frames"])
        self._arrays: dict[str, np.ndarray] | None = None
        self._timestamps: np.ndarray | None = None
        self._filled: np.ndarray | None = None
        self._pid: int | None = None

    # -- pickling: never ship the memory maps themselves ------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_timestamps"] = None
        state["_filled"] = None
        state["_pid"] = None
        return state

    def _ensure_open(self) -> None:
        if self._arrays is not None and self._pid == os.getpid():
            return
        self._arrays = {
            key: np.load(self.cache_dir / self.meta["keys"][key]["file"], mmap_mode="r") for key in self.keys
        }
        self._timestamps = np.load(self.cache_dir / TIMESTAMPS_FILE, mmap_mode="r")
        self._filled = None
        self._pid = os.getpid()

    def array(self, key: str) -> np.ndarray:
        self._ensure_open()
        return self._arrays[key]  # type: ignore[index]

    @property
    def timestamps(self) -> np.ndarray:
        self._ensure_open()
        return self._timestamps  # type: ignore[return-value]

    def filled(self) -> np.ndarray | None:
        """Read-only view of the ``filled`` bitmap (the file other processes write), None once complete."""
        if self.complete:
            return None
        self._ensure_open()
        if self._filled is None:
            self._filled = np.load(self.cache_dir / FILLED_FILE, mmap_mode="r")
        return self._filled

    def mark_complete(self) -> None:
        self.complete = True
        self._filled = None

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
