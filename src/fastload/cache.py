"""Read access to a fastload decoded-frame cache.

Cache layout (one directory per LeRobot dataset)::

    <cache_dir>/
      fastload_meta.json     format version, source dataset fingerprint, per-key shapes (written atomically,
                             after the other files are allocated)
      timestamps.npy         float32 [total_frames]  (copy of the dataset `timestamp` column)
      <camera_key>.npy       uint8   [total_frames, C, H, W]  indexed by the absolute frame `index`
      filled.npy             uint8   [total_frames]  1 = every camera of that frame is written and flushed
      stored.npy             uint8   [total_frames]  only in a size-limited cache (``max_cache_gb``): 1 = the frame
                             belongs to the episodes the cache stores; the other frames are never written (the frame
                             files are sparse, so they take no disk space) and are decoded during training
      progress/ep_XXXXXX.done  per-episode markers written by build_cache (allows resuming)
      CREATED_BY_TRAINING    the cache was made by a training run (``cache_mode`` build_first / during_training);
                             ``fastload.train`` deletes such a cache when training ends normally (``delete_cache``)
      COMPLETE               written last (atomic rename): every stored frame is written. A cache without it is only
                             used by ``cache_mode="during_training"``, which reads filled frames and decodes the rest

Pixels are stored exactly as LeRobot decodes them (uint8 before the `/255` conversion), so
`uint8 -> float32 / 255` reproduces LeRobotDataset outputs bit for bit. Caches built before ``filled.npy``
existed are still valid (a complete cache does not need it).

Size limit: a dataset whose frames take more than ``max_cache_gb`` stores whole episodes, spread evenly over the
dataset, up to that size (``stored.npy``); the frames of the other episodes are decoded while training runs (by
FastLoader's decoder processes, or by LeRobot's own decoding with a plain DataLoader).

Nothing in this module imports LeRobot.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

CACHE_FORMAT_VERSION = 1
META_FILE = "fastload_meta.json"
COMPLETE_FILE = "COMPLETE"
TIMESTAMPS_FILE = "timestamps.npy"
FILLED_FILE = "filled.npy"
STORED_FILE = "stored.npy"
PROGRESS_DIR = "progress"
CREATED_BY_TRAINING_FILE = "CREATED_BY_TRAINING"
USERS_DIR = ".users"  # one lease file per training process that uses the cache
TRANSIENT_DIR = ".transient"  # per-process scratch for frames outside a size-limited cache

# How a training run gets its cache (``--fastload.cache_mode``).
CACHE_MODES = ("prebuilt", "build_first", "during_training")
DEFAULT_MIN_FREE_DISK_GB = 20.0
DEFAULT_MAX_CACHE_GB = 300.0

logger = logging.getLogger("fastload")


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


def frame_bytes(meta: dict[str, Any]) -> int:
    """Bytes of one frame index over every cached camera."""
    return sum(int(np.prod(v["shape"])) for v in meta["keys"].values())


def stored_frame_count(meta: dict[str, Any]) -> int:
    """Frames the cache stores (all of them unless the cache is size-limited)."""
    return int(meta.get("stored_frames", meta["total_frames"]))


def cache_file_sizes(meta: dict[str, Any]) -> dict[str, int]:
    """Apparent size of every file of the cache described by ``meta`` (frame arrays, timestamps, bitmaps). Frames a
    size-limited cache does not store are holes of the (sparse) frame files: see :func:`cache_disk_bytes`."""
    n = int(meta["total_frames"])
    sizes = {v["file"]: n * int(np.prod(v["shape"])) for v in meta["keys"].values()}
    sizes[TIMESTAMPS_FILE] = 4 * n
    sizes[FILLED_FILE] = n
    if stored_frame_count(meta) < n:
        sizes[STORED_FILE] = n
    return sizes


def cache_disk_bytes(meta: dict[str, Any]) -> int:
    """Disk space the finished cache takes: the stored frames plus timestamps and bitmaps."""
    frame_files = {v["file"] for v in meta["keys"].values()}
    other = sum(s for f, s in cache_file_sizes(meta).items() if f not in frame_files)
    return stored_frame_count(meta) * frame_bytes(meta) + other


def select_stored_episodes(lengths: np.ndarray, frame_nbytes: int, budget_bytes: float) -> np.ndarray:
    """Whole episodes whose frames fit in ``budget_bytes``, spread evenly over the dataset (bool per episode).

    Every episode adds its share ``budget / total`` of its size to a credit; an episode is stored once the credit
    covers it (and it still fits), so the stored episodes are spread over the episode order (tasks, recording
    sessions) instead of being the first ones."""
    lengths = np.asarray(lengths, dtype=np.int64)
    sizes = lengths * int(frame_nbytes)
    total = int(sizes.sum())
    keep = np.zeros(len(lengths), dtype=bool)
    if total <= budget_bytes:
        keep[:] = True
        return keep
    share = float(budget_bytes) / total
    credit, used = 0.0, 0
    for e, b in enumerate(sizes.tolist()):
        credit += share * b
        if b and credit >= b and used + b <= budget_bytes:
            keep[e] = True
            credit -= b
            used += b
    return keep


def check_free_disk(cache_dir: str | Path, meta: dict[str, Any], min_free_disk_gb: float | None) -> None:
    """Raise :class:`DiskSpaceError` unless ``free disk - bytes still to write >= min_free_disk_gb``.

    The bytes to write are those of the finished cache (:func:`cache_disk_bytes`, only the stored frames of a
    size-limited cache) minus what its files already occupy, so a resumed or partly filled cache only needs space for
    the rest. Creates nothing.
    """
    if min_free_disk_gb is None or min_free_disk_gb < 0:
        return
    d = Path(cache_dir)
    total = cache_disk_bytes(meta)
    have = sum(allocated_bytes(d / name) for name in cache_file_sizes(meta) if (d / name).exists())
    to_write = max(0, total - have)
    free = shutil.disk_usage(_existing_ancestor(d)).free
    if free - to_write < float(min_free_disk_gb) * 1e9:
        def gb(b: float) -> str:
            return f"{b / 1e9:.1f} GB"

        raise DiskSpaceError(
            f"fastload: not enough free disk for the cache at {d}: cache size {gb(total)} ({gb(to_write)} still to "
            f"write), free {gb(free)}; writing it would leave {gb(free - to_write)}, less than "
            f"min_free_disk_gb={min_free_disk_gb:g} (a safety margin kept free, not a cache size limit). Options:\n"
            "  * a smaller cache: --fastload.max_cache_gb / build_cache --max_cache_gb (stores that many GB of whole "
            "episodes; the other frames are decoded during training)\n"
            "  * put the cache on another disk (--fastload.cache_dir / build_cache --cache_dir)\n"
            "  * cache fewer cameras (python -m fastload.build_cache --keys ...; uncached cameras are then decoded by "
            "LeRobot, with --fastload.loader=dataloader)\n"
            "  * lower the margin (--fastload.min_free_disk_gb / build_cache --min_free_disk_gb) if you are sure\n"
            "  * train with plain lerobot-train (decodes video on the fly, needs no cache)"
        )


# ------------------------------------------------------------------------------------------------
# training runs that use a cache: leases, and deleting a cache a training run created
# ------------------------------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5  # access denied: exists
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return not ok or code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - someone else's process
        return True
    return True


def _lease_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _live_leases(cache_dir: Path) -> list[str]:
    """Lease files of processes that may still use the cache; stale ones (dead processes on this host) are removed."""
    d = cache_dir / USERS_DIR
    if not d.is_dir():
        return []
    host = socket.gethostname()
    live = []
    for f in d.iterdir():
        h, _, pid = f.name.rpartition("-")
        if h == host and pid.isdigit() and not _pid_alive(int(pid)):
            f.unlink(missing_ok=True)  # a crashed run
            continue
        live.append(f.name)
    return live


class _NodeLock:
    """Exclusive lock (node-local file in the temp dir, keyed by the cache path) around lease changes and deletion."""

    def __init__(self, cache_dir: Path):
        key = hashlib.sha1(str(cache_dir.absolute()).encode()).hexdigest()[:16]
        self.path = Path(tempfile.gettempdir()) / f"fastload-{key}.lock"
        self.fd: int | None = None

    def __enter__(self) -> "_NodeLock":
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows: msvcrt
            import msvcrt

            while True:
                try:
                    msvcrt.locking(self.fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            try:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_UN)
            except ImportError:  # pragma: no cover
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(self.fd)


def acquire_lease(cache_dir: str | Path) -> Path | None:
    """Record that this process uses the cache (so another run that ends does not delete it). None if the cache
    directory does not exist."""
    d = Path(cache_dir)
    if not d.is_dir():
        return None
    with _NodeLock(d):
        (d / USERS_DIR).mkdir(exist_ok=True)
        lease = d / USERS_DIR / _lease_name()
        lease.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return lease


def created_by_training(cache_dir: str | Path) -> bool:
    return (Path(cache_dir) / CREATED_BY_TRAINING_FILE).exists()


def mark_created_by_training(cache_dir: str | Path) -> None:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / CREATED_BY_TRAINING_FILE, time.strftime("%Y-%m-%d %H:%M:%S"))


def release_lease(cache_dir: str | Path, lease: Path | None, delete: bool) -> bool:
    """Drop this process's lease. With ``delete``, also delete the cache if a training run created it and no other
    process still holds a lease (the last process of a run deletes it). Returns whether the cache was deleted."""
    d = Path(cache_dir)
    if not d.is_dir():
        return False
    with _NodeLock(d):
        if lease is not None:
            lease.unlink(missing_ok=True)
        if not delete or not created_by_training(d):
            return False
        others = _live_leases(d)
        if others:
            logger.info(f"fastload: cache {d} is still used by {len(others)} other process(es) ({', '.join(others[:4])}); "
                        "not deleting it")
            return False
        size = sum(allocated_bytes(f) for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    logger.info(f"fastload: deleted the cache {d} ({size / 1e9:.1f} GB on disk): a training run created it and training "
                "has ended (--fastload.delete_cache=false keeps it)")
    return True


class FrameCache:
    """Lazily opened, read-only view of a built cache.

    Memory maps are opened on first use in each process and are dropped when the object is
    pickled, so the cache can be passed to DataLoader workers (fork or spawn) without copying
    the arrays.

    ``allow_incomplete=True`` (``cache_mode="during_training"``) also opens a cache whose frames are still being
    filled: :attr:`complete` is False and :meth:`filled` tells which frames are written.

    A size-limited cache (:attr:`limited`) stores only some episodes: :meth:`stored` tells which frames, and
    :meth:`available` which frames can be read now (stored and, while filling, written).
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
        self.stored_frames: int = stored_frame_count(self.meta)
        self.limited: bool = self.stored_frames < self.total_frames
        if self.limited and not (self.cache_dir / STORED_FILE).exists():
            raise RuntimeError(f"Cache {self.cache_dir} stores {self.stored_frames} of {self.total_frames} frames but "
                               f"has no {STORED_FILE}; rebuild it.")
        self._arrays: dict[str, np.ndarray] | None = None
        self._timestamps: np.ndarray | None = None
        self._filled: np.ndarray | None = None
        self._stored: np.ndarray | None = None
        self._pid: int | None = None

    # -- pickling: never ship the memory maps themselves ------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_timestamps"] = None
        state["_filled"] = None
        state["_stored"] = None
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
        self._stored = None
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

    def stored(self) -> np.ndarray | None:
        """Bool per frame: the frames a size-limited cache stores; None when it stores every frame."""
        if not self.limited:
            return None
        self._ensure_open()
        if self._stored is None:
            self._stored = np.load(self.cache_dir / STORED_FILE).astype(bool)
        return self._stored

    def available(self) -> np.ndarray | None:
        """Frames that can be read from the cache now (non-zero = readable), or None when every frame can.

        Complete cache: its stored frames. Incomplete cache: the frames written so far (only stored frames are ever
        written). The array of an incomplete cache is a live view of the bitmap other processes write."""
        if self.complete:
            return self.stored()
        return self.filled()

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
        """Bytes of the stored frames (a size-limited cache does not hold the others)."""
        return sum(self.frame_nbytes(k) for k in self.keys) * self.stored_frames

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
        built = self.meta.get("video_backend")
        used = getattr(dataset, "_video_backend", None) or getattr(dataset, "video_backend", None)
        if getattr(dataset.meta, "video_keys", None) and built and isinstance(used, str) and built != used:
            import logging

            logging.warning(
                f"fastload: cache {self.cache_dir} was decoded with video_backend={built!r} but this dataset uses "
                f"{used!r}; the two decoders can give slightly different pixels, so batches may not match plain "
                f"LeRobot. Rebuild with `build_cache ... --video_backend {used}` (or pass "
                f"--dataset.video_backend={built} to training)."
            )
