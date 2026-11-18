"""Getting the cache without a separate build step: ``cache_mode="build_first"`` and ``"during_training"``.

* :func:`ensure_cache` runs before training (``fastload.train`` calls it before ``lerobot_train.train()``, i.e. before
  the Accelerator / process group exists): ``build_first`` builds the whole cache with :func:`fastload.build_cache`;
  ``during_training`` only allocates the files and the ``filled`` bitmap. With several processes on a node only
  ``LOCAL_RANK`` 0 does it (the cache directory is node-local); the other local ranks poll until it is ready and stop
  with the leader's error if it failed.
* :class:`CacheFiller` serves a FastLoader whose cache is not complete yet: a pool of decoder processes (the same
  workers as ``build_cache``, so the frames are the same bytes) decodes the frames that upcoming batches need, and,
  while the pool has idle workers, the rest of the dataset in order. Each task writes its frames, flushes them, then
  sets their ``filled`` bytes. When every frame is filled, ``COMPLETE`` is written (atomic rename) and the cache is an
  ordinary prebuilt cache from then on. Several processes on a node may fill the same cache: a frame written twice
  gets the same bytes, and any process may set ``filled`` bytes (one byte per frame, set only after its data).

Nothing here imports LeRobot directly (dataset access goes through :mod:`fastload.build_cache`'s workers and
:mod:`fastload._lr`), so the module is the same on every branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import tempfile
import threading
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fastload.cache import (
    CACHE_MODES,
    META_FILE,
    DiskSpaceError,
    FrameCache,
    cache_file_sizes,
    cache_status,
    check_free_disk,
    layout_ready,
    missing_cache_message,
    write_complete,
)

logger = logging.getLogger("fastload")


# ------------------------------------------------------------------------------------------------
# processes on this node
# ------------------------------------------------------------------------------------------------
def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0") or 0)


def local_world_size() -> int:
    return max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1") or 1))


def default_decode_workers() -> int:
    """About half the CPU cores, split between the training processes of this node (at least 1)."""
    return max(1, (os.cpu_count() or 2) // 2 // local_world_size())


def _failed_marker(cache_dir: str | Path) -> Path:
    """Where the local leader reports a failure: node-local temp dir, keyed by the cache and the launcher process
    (all local ranks of one ``torchrun`` / ``accelerate launch`` share their parent), so a marker of an earlier run
    is never read, and nothing is created in the cache directory."""
    key = hashlib.sha1(str(Path(cache_dir).absolute()).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"fastload-{key}-{os.getppid()}.failed"


def lead_or_wait(cache_dir: str | Path, what: str, work: Callable[[], None], ready: Callable[[], bool],
                 poll_s: float = 1.0, log_every_s: float = 60.0) -> None:
    """``LOCAL_RANK`` 0 runs ``work``; the other local ranks wait until ``ready()`` (or the leader failed)."""
    marker = _failed_marker(cache_dir)
    if local_rank() == 0:
        marker.unlink(missing_ok=True)
        try:
            work()
        except BaseException as e:
            try:
                marker.write_text(f"{type(e).__name__}: {e}")
            except OSError:  # pragma: no cover
                pass
            raise
        return
    t0 = last = time.time()
    logger.info(f"fastload: local rank {local_rank()} waits for local rank 0 to {what} at {cache_dir}")
    while not ready():
        if marker.exists():
            time.sleep(0.2)  # let the leader finish writing it
            raise SystemExit(f"fastload: local rank 0 failed to {what} at {cache_dir}; stopping too. Its error:\n"
                             f"{marker.read_text()}")
        time.sleep(poll_s)
        if time.time() - last >= log_every_s:
            last = time.time()
            logger.info(f"fastload: local rank {local_rank()} still waiting for local rank 0 to {what} "
                        f"({last - t0:.0f}s)")
    logger.info(f"fastload: local rank {local_rank()}: local rank 0 finished ({what}) after {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------
# building in a separate process (no decoder state or memory left in the training process)
# ------------------------------------------------------------------------------------------------
def _isolated_build(kwargs: dict[str, Any], q) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from fastload.build_cache import build_cache
        from fastload.decode_hook import remove_hook

        remove_hook()  # spawned: a training script that wraps decode_video_frames at import time was re-imported
        build_cache(**kwargs)
        q.put(None)
    except BaseException as e:
        detail = "" if isinstance(e, DiskSpaceError) else "\n" + traceback.format_exc()[-6000:]
        q.put((type(e).__name__, f"{e}{detail}"))
        raise SystemExit(1)


def run_build_cache(**kwargs: Any) -> None:
    """:func:`fastload.build_cache.build_cache` in a spawned process; its error is raised here."""
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p = ctx.Process(target=_isolated_build, args=(kwargs, q), name="fastload-build-cache")
    p.start()
    p.join()
    err = None if q.empty() else q.get()
    if err is None and p.exitcode != 0:
        err = ("RuntimeError", f"the cache-building process exited with status {p.exitcode}")
    if err is not None:
        name, msg = err
        raise (DiskSpaceError if name == "DiskSpaceError" else RuntimeError)(msg)


def ensure_cache(cache_mode: str, cache_dir: str | Path, source: dict[str, Any] | Callable[[], dict[str, Any]],
                 min_free_disk_gb: float | None, num_workers: int | None = None) -> str:
    """Make the cache usable in ``cache_mode`` before training starts. Returns the cache status afterwards.

    ``source``: the dataset to decode (``repo_id``, ``root``, ``revision``, ``video_backend``, ``tolerance_s``), or a
    callable returning it (only called when something has to be built).
    """
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    cache_dir = Path(cache_dir)
    status = cache_status(cache_dir)
    src = (lambda: source()) if callable(source) else (lambda: source)  # noqa: E731
    if cache_mode == "prebuilt":
        if status != "complete":
            s = src()
            raise FileNotFoundError(missing_cache_message(cache_dir, s.get("repo_id"), s.get("root"),
                                                          partial=status == "partial"))
        return status
    if status == "complete":
        logger.info(f"fastload: cache {cache_dir} is complete (cache_mode={cache_mode} has nothing to do)")
        return status
    if cache_mode == "build_first":
        def work() -> None:
            s = src()
            logger.info(f"fastload: cache_mode=build_first: building the cache for {s['repo_id']} at {cache_dir} "
                        "before training starts (resumable)")
            t0 = time.time()
            run_build_cache(**s, cache_dir=str(cache_dir), num_workers=num_workers,
                            min_free_disk_gb=min_free_disk_gb)
            logger.info(f"fastload: cache built in {time.time() - t0:.0f}s; training starts")

        lead_or_wait(cache_dir, "build the cache", work, lambda: cache_status(cache_dir) == "complete")
        return cache_status(cache_dir)
    # during_training: allocate the files (once); training fills them
    if layout_ready(cache_dir):
        meta = json.loads((cache_dir / META_FILE).read_text())
        check_free_disk(cache_dir, cache_file_sizes(meta), min_free_disk_gb)
        return status

    def work() -> None:
        s = src()
        logger.info(f"fastload: cache_mode=during_training: allocating the cache for {s['repo_id']} at {cache_dir}; "
                    "the first epoch decodes the frames it needs and stores them")
        run_build_cache(**s, cache_dir=str(cache_dir), min_free_disk_gb=min_free_disk_gb, layout_only=True)

    lead_or_wait(cache_dir, "allocate the cache files", work, lambda: layout_ready(cache_dir))
    return cache_status(cache_dir)


# ------------------------------------------------------------------------------------------------
# filling during training
# ------------------------------------------------------------------------------------------------
class FillError(RuntimeError):
    pass


class CacheFiller:
    """Decoder pool that fills an incomplete cache for one training process (see the module docstring).

    Args:
        cache: the (incomplete) :class:`FrameCache`.
        source: dataset arguments for the decoder processes (``fastload._lr.dataset_source``).
        ep_from, ep_to: frame range of every episode (indexed by episode index).
        workers: decoder processes.
        background: decode the frames no batch has asked for yet while workers are idle (in order, starting at
            this process's share of the episodes), so the cache completes even if training never reads some frames.
    """

    TASK_FRAMES = 64  # frames per decoding task at most
    MERGE_GAP = 4  # needed frames of one episode this close together are decoded in one task
    EXTEND = 8  # a needed span grows by up to this many unfilled frames (seeking decoders decode them anyway)
    SCAN = 1 << 16  # frames scanned at a time for background work
    STALL_S = 600.0  # waiting this long without any task finishing is an error (e.g. a decoder process died)

    def __init__(self, cache: FrameCache, source: dict[str, Any], ep_from: np.ndarray, ep_to: np.ndarray,
                 workers: int, background: bool = True):
        from fastload.build_cache import _worker_init

        if cache.complete:
            raise ValueError("cache is already complete")
        self.cache = cache
        self.cache_dir = cache.cache_dir
        self.n = cache.total_frames
        self.filled = cache.filled()  # read-only view of the bitmap the decoder processes write
        self.ep_from = np.asarray(ep_from, dtype=np.int64)
        self.ep_to = np.asarray(ep_to, dtype=np.int64)
        self.workers = max(1, int(workers))
        self.background = bool(background)
        self.inflight = np.zeros(self.n, dtype=bool)  # frames in this process's queued / running tasks
        self.complete = False
        self.stats = {"tasks": 0, "frames": 0, "background_tasks": 0, "background_frames": 0, "wait_s": 0.0,
                      "started": time.time()}
        self._cv = threading.Condition()
        self._outstanding = 0
        self._last_progress = time.time()
        self._error: BaseException | None = None
        self._closed = False
        self._finish = False
        n_ep = len(self.ep_from)
        first_ep = (local_rank() * n_ep) // local_world_size() if n_ep else 0
        self._cursor = int(self.ep_from[first_ep]) if n_ep else 0
        already = int(np.count_nonzero(self.filled))
        wargs = dict(source, cache_dir=str(self.cache_dir), keys=list(cache.keys))
        self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_worker_init, initargs=(wargs,))
        self._thread = threading.Thread(target=self._background_loop, daemon=True, name="fastload-fill")
        self._thread.start()
        logger.info(f"fastload: cache {self.cache_dir} has {already} of {self.n} frames; decoding the rest during "
                    f"training with {self.workers} decoder processes"
                    + (" (needed frames first, the rest while decoders are idle)" if self.background else
                       " (only frames that batches need)"))

    # -- tasks ---------------------------------------------------------------------------------
    def _submit_locked(self, ep: int, lo: int, hi: int, background: bool) -> None:
        from fastload.build_cache import _fill_task

        if self._outstanding == 0:
            self._last_progress = time.time()
        self.inflight[lo:hi] = True
        self._outstanding += 1
        self._pool.apply_async(_fill_task, (int(ep), int(lo), int(hi)),
                               callback=partial(self._done, lo, hi, background),
                               error_callback=partial(self._failed, lo, hi))

    def _done(self, lo: int, hi: int, background: bool, _result: Any) -> None:
        with self._cv:
            self.inflight[lo:hi] = False
            self._outstanding -= 1
            self._last_progress = time.time()
            self.stats["tasks"] += 1
            self.stats["frames"] += hi - lo
            if background:
                self.stats["background_tasks"] += 1
                self.stats["background_frames"] += hi - lo
            self._cv.notify_all()

    def _failed(self, lo: int, hi: int, exc: BaseException) -> None:
        with self._cv:
            self.inflight[lo:hi] = False
            self._outstanding -= 1
            if self._error is None:
                self._error = exc
            self._cv.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise FillError(f"fastload: decoding frames into the cache {self.cache_dir} failed: "
                            f"{type(self._error).__name__}: {self._error}") from self._error
        if self._closed:
            raise FillError("fastload: the cache filler was closed")

    def _free(self, lo: int, hi: int) -> bool:
        """Frames ``lo:hi`` are all unfilled and not queued by this process."""
        return not (self.filled[lo:hi].any() or self.inflight[lo:hi].any())

    def _submit_needed_locked(self, frames: np.ndarray) -> None:
        """Tasks for sorted, unfilled, not-queued ``frames``: runs within one episode, merged across small gaps of
        free frames and extended by a few free frames, at most ``TASK_FRAMES`` long."""
        eps = np.searchsorted(self.ep_to, frames, side="right")
        i, n = 0, len(frames)
        while i < n:
            e, lo = int(eps[i]), int(frames[i])
            end = min(int(self.ep_to[e]), lo + self.TASK_FRAMES)
            hi, j = lo + 1, i + 1
            while (j < n and eps[j] == e and frames[j] < end and frames[j] - hi <= self.MERGE_GAP
                   and self._free(hi, int(frames[j]))):
                hi = int(frames[j]) + 1
                j += 1
            stop = min(end, hi + self.EXTEND)
            while hi < stop and self.filled[hi] == 0 and not self.inflight[hi]:
                hi += 1
            while j < n and frames[j] < hi:
                j += 1
            self._submit_locked(e, lo, hi, background=False)
            i = j

    def _missing(self, frames: np.ndarray) -> np.ndarray:
        f = np.unique(np.asarray(frames, dtype=np.int64))
        return f[self.filled[f] == 0]

    def request(self, frames: np.ndarray) -> None:
        """Queue the decoding of the unfilled ``frames`` (does not wait)."""
        if self.complete:
            return
        with self._cv:
            self._raise_if_failed()
            f = self._missing(frames)
            f = f[~self.inflight[f]]
            if len(f):
                self._submit_needed_locked(f)

    def ensure(self, frames: np.ndarray) -> None:
        """Return once every frame in ``frames`` is filled (decoding the missing ones first)."""
        if self.complete:
            return
        f = self._missing(frames)
        if not len(f):
            return
        t0 = time.perf_counter()
        with self._cv:
            while True:
                self._raise_if_failed()
                f = f[self.filled[f] == 0]
                if not len(f):
                    break
                todo = f[~self.inflight[f]]  # e.g. a frame another rank started but has not finished
                if len(todo):
                    self._submit_needed_locked(todo)
                elif self._outstanding and time.time() - self._last_progress > self.STALL_S:
                    raise FillError(f"fastload: no decoding task finished for {self.STALL_S:.0f}s while frames of "
                                    "this batch were queued (a decoder process may have died)")
                self._cv.wait(timeout=0.05)
        self.stats["wait_s"] += time.perf_counter() - t0

    # -- background filling and completion -------------------------------------------------------
    def _next_free_span_locked(self) -> tuple[int, int, int] | None:
        for start in (self._cursor, 0):
            pos = start
            while pos < self.n:
                w = min(self.n, pos + self.SCAN)
                free = np.flatnonzero((self.filled[pos:w] == 0) & ~self.inflight[pos:w])
                if len(free):
                    lo = pos + int(free[0])
                    e = int(np.searchsorted(self.ep_to, lo, side="right"))
                    stop = min(int(self.ep_to[e]), lo + self.TASK_FRAMES)
                    seg = (self.filled[lo:stop] == 0) & ~self.inflight[lo:stop]
                    hi = lo + (int(np.argmin(seg)) if not seg.all() else len(seg))
                    self._cursor = hi
                    return e, lo, hi
                pos = w
        return None

    def _background_loop(self) -> None:
        try:
            while True:
                with self._cv:
                    if self._closed or self._error is not None:
                        return
                    if self._outstanding == 0 and bool(np.all(self.filled)):
                        break  # every frame stored (by this process, or by batches / other processes)
                    if self._outstanding >= self.workers or not (self.background or self._finish):
                        self._cv.wait(timeout=0.5)
                        continue
                    span = self._next_free_span_locked()
                    if span is not None:
                        self._submit_locked(*span, background=True)
                        continue
                    self._cv.wait(timeout=0.5)  # remaining frames are in flight (here or in another process)
            self._mark_complete()
        except BaseException as e:  # pragma: no cover - surfaced by the next ensure()/request()
            with self._cv:
                if self._error is None:
                    self._error = e
                self._cv.notify_all()

    def _mark_complete(self) -> None:
        write_complete(self.cache_dir)
        dt = time.time() - self.stats["started"]
        logger.info(f"fastload: cache complete: all {self.n} frames stored at {self.cache_dir} ({dt:.0f}s, "
                    f"{self.stats['frames']} decoded by this process); it is a regular prebuilt cache from now on")
        with self._cv:
            self.complete = True
            self._cv.notify_all()
        self._pool.close()
        self._pool.join()

    def finish(self, timeout: float | None = None) -> bool:
        """Decode every remaining frame now (also with ``background=False``) and wait. Returns ``complete``."""
        deadline = None if timeout is None else time.time() + timeout
        with self._cv:
            self._finish = True
            self._cv.notify_all()
            while not self.complete:
                self._raise_if_failed()
                if deadline is not None and time.time() >= deadline:
                    break
                self._cv.wait(timeout=0.5)
        return self.complete

    def close(self) -> None:
        """Stop decoding (queued and running tasks are dropped; their frames stay unfilled and are decoded by the
        next run). Idempotent."""
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread and self.complete:
            self._thread.join(timeout=60)  # _mark_complete closes the pool itself
        try:
            if not self.complete:
                self._pool.terminate()
            self._pool.join()
        except ValueError:  # pragma: no cover - pool being closed by _mark_complete right now
            pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10)
