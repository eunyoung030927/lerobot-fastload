"""Decode a LeRobot dataset's camera streams once into a uint8 memmap cache.

Usage::

    python -m fastload.build_cache --repo_id <repo> --root <dataset dir> --cache_dir <out dir> \
        --num_workers 16

Frames are decoded through LeRobot's own ``DatasetReader._query_videos`` (same backend, same timestamp
matching, ``return_uint8=True`` as in ``lerobot_train``), so the cache holds exactly the pixels LeRobot would
have produced during training. Depth cameras (``meta.depth_keys``, 12-bit / float) are not cached; datasets
read through :class:`fastload.CachedLeRobotDataset` decode them with LeRobot's code.
The build is resumable: finished episodes are recorded under ``progress/`` (and every written frame in
``filled.npy``) and skipped on rerun. Before writing, the free disk space is checked (``--min_free_disk_gb``).
A dataset whose frames take more than ``--max_cache_gb`` (default 300) gets a size-limited cache: whole episodes,
spread over the dataset, up to that size; training decodes the frames of the other episodes as it goes.

The decoding workers here (``_worker_init`` / ``_decode_range``) are also the ones that fill a cache during
training (``--fastload.cache_mode=during_training``, :mod:`fastload.fill`), so both store the same bytes.
"""

from __future__ import annotations

import argparse
import json
import logging
import mmap
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastload import _lr

from fastload.cache import (
    CACHE_FORMAT_VERSION,
    COMPLETE_FILE,
    DEFAULT_MAX_CACHE_GB,
    DEFAULT_MIN_FREE_DISK_GB,
    FILLED_FILE,
    META_FILE,
    PROGRESS_DIR,
    STORED_FILE,
    TIMESTAMPS_FILE,
    atomic_write_text,
    cache_disk_bytes,
    check_free_disk,
    info_fingerprint,
    key_to_filename,
    mark_created_by_training,
    select_stored_episodes,
    write_complete,
)

logger = logging.getLogger("fastload.build_cache")

# Decoded float32 frames held per worker at once (bytes). Keeps 480x640 videos from blowing RAM.
_CHUNK_FLOAT_BYTES = 256 * 1024 * 1024


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """uint8 frames as they are; float32 frames (parquet images: uint8 / 255) inverted exactly."""
    if frames.dtype == torch.uint8:
        return frames.numpy()
    return frames.mul(255).round_().clamp_(0, 255).to(torch.uint8).numpy()


def _flush_rows(arr: np.memmap, lo: int, hi: int) -> None:
    """Flush rows ``lo:hi`` of a memmap to disk (msync of just that range; the whole map as a fallback)."""
    try:
        row = int(np.prod(arr.shape[1:], dtype=np.int64)) * arr.itemsize
        mm = arr._mmap  # numpy maps from offset rounded down to the allocation granularity
        start = arr.offset - arr.offset % mmap.ALLOCATIONGRANULARITY
        off = arr.offset - start + lo * row
        aligned = off - off % mmap.PAGESIZE
        mm.flush(aligned, off - aligned + (hi - lo) * row)
    except Exception:  # pragma: no cover - numpy internals changed
        arr.flush()


# ------------------------------------------------------------------------------------------------
# worker side (build_cache's pool and the cache-filling pool of cache_mode=during_training)
# ------------------------------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(args: dict[str, Any]) -> None:
    from fastload.decode_hook import remove_hook

    torch.set_num_threads(1)
    remove_hook()  # the cache stores the decoder's own frames (a spawned worker may re-import a training script)
    ds = _lr.open_dataset_for_cache(args["repo_id"], args["root"], args["revision"], args["video_backend"],
                                    args["tolerance_s"])
    cache_dir = Path(args["cache_dir"])
    meta = json.loads((cache_dir / META_FILE).read_text())
    ep_from, ep_to = _lr.episode_bounds(ds.meta)
    _W.clear()
    _W.update(
        ds=ds,
        cache_dir=cache_dir,
        keys=args["keys"],
        ep_from=ep_from,
        ep_to=ep_to,
        arrays={k: np.load(cache_dir / meta["keys"][k]["file"], mmap_mode="r+") for k in args["keys"]},
        filled=np.load(cache_dir / FILLED_FILE, mmap_mode="r+"),
        timestamps=np.load(cache_dir / TIMESTAMPS_FILE, mmap_mode="r"),
        chunk={k: max(8, _CHUNK_FLOAT_BYTES // (4 * int(np.prod(meta["keys"][k]["shape"]))))
               for k in args["keys"]},
    )


def _decode_range(ep_idx: int, lo: int, hi: int) -> int:
    """Decode frames ``lo:hi`` (absolute, inside episode ``ep_idx``) of every cached camera, write and flush them,
    then mark them filled. Concurrent writers of the same frames write the same bytes."""
    ds = _W["ds"]
    ts_all = _W["timestamps"]
    for key in _W["keys"]:
        out = _W["arrays"][key]
        step = _W["chunk"][key]
        for s in range(lo, hi, step):
            e = min(s + step, hi)
            out[s:e] = _to_uint8(_lr.read_frames(ds, key, ep_idx, s, e, ts_all[s:e]))
        _flush_rows(out, lo, hi)
    filled = _W["filled"]
    filled[lo:hi] = 1  # only after the frames are on disk: readers never see a frame before it is complete
    _flush_rows(filled, lo, hi)
    return hi - lo


def _decode_episode(ep_idx: int) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    start, end = int(_W["ep_from"][ep_idx]), int(_W["ep_to"][ep_idx])
    _decode_range(ep_idx, start, end)
    (_W["cache_dir"] / PROGRESS_DIR / f"ep_{ep_idx:06d}.done").touch()
    return ep_idx, end - start, time.perf_counter() - t0


def _fill_task(ep_idx: int, lo: int, hi: int) -> tuple[int, int]:
    """One task of the during-training filling pool (:mod:`fastload.fill`)."""
    return lo, lo + _decode_range(ep_idx, lo, hi)


def _scratch_arrays(scratch_dir: str) -> dict[str, np.ndarray]:
    """This worker's maps of a loader's scratch files (opened once per scratch directory)."""
    cur = _W.get("scratch")
    if cur is None or cur[0] != scratch_dir:
        meta = json.loads((_W["cache_dir"] / META_FILE).read_text())
        arrays = {k: np.load(Path(scratch_dir) / meta["keys"][k]["file"], mmap_mode="r+") for k in _W["keys"]}
        _W["scratch"] = cur = (scratch_dir, arrays)
    return cur[1]


def _transient_task(scratch_dir: str, ep_idx: int, lo: int, hi: int, frames: np.ndarray, slots: np.ndarray) -> int:
    """Decode frames ``lo:hi`` of episode ``ep_idx`` (every cached camera, as :func:`_decode_range`) and write the
    sorted absolute ``frames`` among them to rows ``slots`` of the scratch files: frames a size-limited cache does
    not store (:class:`fastload.fill.TransientFrames`). Nothing is written to the cache itself."""
    ds = _W["ds"]
    ts_all = _W["timestamps"]
    scratch = _scratch_arrays(scratch_dir)
    frames = np.asarray(frames, dtype=np.int64)
    slots = np.asarray(slots, dtype=np.int64)
    for key in _W["keys"]:
        out = scratch[key]
        step = _W["chunk"][key]
        for s in range(lo, hi, step):
            e = min(s + step, hi)
            sel = (frames >= s) & (frames < e)
            if not sel.any():
                continue
            dec = _to_uint8(_lr.read_frames(ds, key, ep_idx, s, e, ts_all[s:e]))
            out[slots[sel]] = dec[frames[sel] - s]
    return len(frames)


# ------------------------------------------------------------------------------------------------
# main side
# ------------------------------------------------------------------------------------------------
def build_cache(
    repo_id: str,
    cache_dir: str | Path,
    root: str | Path | None = None,
    revision: str | None = None,
    video_backend: str | None = None,
    tolerance_s: float = 1e-4,
    keys: list[str] | None = None,
    num_workers: int | None = None,
    verify: int = 32,
    overwrite: bool = False,
    min_free_disk_gb: float | None = DEFAULT_MIN_FREE_DISK_GB,
    layout_only: bool = False,
    max_cache_gb: float | None = DEFAULT_MAX_CACHE_GB,
    created_by_training: bool = False,
) -> Path:
    """Build (or resume building) the cache for one LeRobot dataset. Returns the cache dir.

    ``min_free_disk_gb``: refuse (before writing anything) unless ``free disk - bytes still to write`` stays at
    least this large (None / negative = no check). ``layout_only``: only allocate the files and the ``filled``
    bitmap (what ``cache_mode="during_training"`` fills), decode nothing. ``max_cache_gb``: if the frames take more,
    store whole episodes (spread over the dataset) up to this size; the others are decoded during training (None /
    negative = no limit). It applies when the cache is created; a resumed cache keeps its episodes.
    ``created_by_training``: mark a new cache as made by a training run (deleted when that training ends).
    """
    from fastload.decode_hook import active_hook, hook_name

    if active_hook() is not None:
        raise RuntimeError(
            f"fastload: decode_video_frames is wrapped ({hook_name(active_hook())}); build the cache in a process "
            "without the wrapper. The cache stores the decoder's own frames and the wrapper is applied at training time."
        )
    cache_dir = Path(cache_dir)
    root = str(root) if root is not None else None
    ds = _lr.open_dataset_for_cache(repo_id, root, revision, video_backend, tolerance_s)
    _lr.check_storage_format(ds.meta)
    depth = set(_lr.depth_keys(ds.meta))
    if keys:
        keys = list(keys)
        unknown = [k for k in keys if k not in ds.meta.camera_keys]
        if unknown:
            raise ValueError(f"{unknown} are not camera keys of {repo_id}: {ds.meta.camera_keys}")
        if depth & set(keys):
            raise ValueError(f"{sorted(depth & set(keys))} are depth cameras; fastload does not cache depth frames")
    else:
        keys = [k for k in ds.meta.camera_keys if k not in depth]
        if depth:
            logger.warning(f"{repo_id}: depth cameras {sorted(depth)} are not cached (12-bit / float frames); "
                           "CachedLeRobotDataset decodes them with LeRobot, FastLoader refuses such datasets")
    if not keys:
        raise ValueError(f"{repo_id} has no (non-depth) camera keys; nothing to cache.")

    n = int(ds.meta.total_frames)
    hf = ds.hf_dataset.select_columns(["index", "timestamp"]).with_format("numpy")[:]
    index = np.asarray(hf["index"]).astype(np.int64)
    if index.shape[0] != n or not np.array_equal(index, np.arange(n)):
        raise RuntimeError("Expected the `index` column to be 0..total_frames-1; unsupported dataset layout.")
    timestamps = np.asarray(hf["timestamp"], dtype=np.float32)
    ep_from, ep_to = _lr.episode_bounds(ds.meta)

    # Frame shape = what LeRobot actually returns (C, H, W), probed on frame 0.
    ep0 = int(np.searchsorted(ep_to, 0, side="right"))
    shapes = {key: list(_lr.read_frames(ds, key, ep0, 0, 1, timestamps[:1]).shape[-3:]) for key in keys}

    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "repo_id": repo_id,
        "dataset_root": str(ds.root),
        "info_fingerprint": info_fingerprint(ds.root),
        "total_frames": n,
        "total_episodes": int(ds.meta.total_episodes),
        "fps": int(ds.meta.fps),
        "video_backend": _lr.video_backend(ds),
        "tolerance_s": tolerance_s,
        "keys": {k: {"file": key_to_filename(k), "shape": shapes[k], "dtype": "uint8",
                     "source": "video" if k in ds.meta.video_keys else "image"} for k in keys},
    }
    try:
        import lerobot

        meta["lerobot_version"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # pragma: no cover
        meta["lerobot_version"] = "unknown"

    # Size limit: whole episodes, spread over the dataset, up to max_cache_gb of frames.
    fb = int(sum(np.prod(s) for s in shapes.values()))
    limit = None if max_cache_gb is None or max_cache_gb < 0 else float(max_cache_gb) * 1e9
    keep_ep = select_stored_episodes(ep_to - ep_from, fb, limit if limit is not None else float("inf"))
    stored = None
    if not keep_ep.all():
        stored = np.zeros(n, dtype=np.uint8)
        for e in np.flatnonzero(keep_ep):
            stored[ep_from[e]:ep_to[e]] = 1
        meta["stored_frames"] = int(stored.sum())
        meta["stored_episodes"] = int(keep_ep.sum())
        meta["max_cache_gb"] = float(max_cache_gb)

    meta_path = cache_dir / META_FILE
    resume = False
    if meta_path.exists() and not overwrite:
        old = json.loads(meta_path.read_text())
        same = all(old.get(f) == meta[f] for f in ("format_version", "info_fingerprint", "total_frames", "keys"))
        if not same:
            raise RuntimeError(
                f"{cache_dir} already holds a cache for a different dataset/keys. Use --overwrite to replace it."
            )
        resume = True
        if old.get("stored_frames", n) != meta.get("stored_frames", n):
            logger.warning(
                f"{cache_dir} stores {old.get('stored_frames', n)} of {n} frames (max_cache_gb="
                f"{old.get('max_cache_gb', 'none')}); keeping that (max_cache_gb={max_cache_gb} would store "
                f"{meta.get('stored_frames', n)}). Delete the cache or pass --overwrite to change it."
            )
        for f in ("stored_frames", "stored_episodes", "max_cache_gb"):
            meta.pop(f, None)
            if f in old:
                meta[f] = old[f]
        stored = np.load(cache_dir / STORED_FILE) if (cache_dir / STORED_FILE).exists() else None
        keep_ep = (np.ones(len(ep_from), dtype=bool) if stored is None
                   else (stored[np.minimum(ep_from, max(n - 1, 0))] > 0) & (ep_to > ep_from))
    elif cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)
    if stored is not None and not resume:
        logger.info(
            f"{repo_id}: the frames take {n * fb / 1e9:.1f} GB > max_cache_gb={max_cache_gb:g}: the cache stores "
            f"{int(keep_ep.sum())} of {len(keep_ep)} episodes ({meta['stored_frames'] * fb / 1e9:.1f} GB, "
            f"{100 * meta['stored_frames'] / max(n, 1):.0f}% of the frames); training decodes the others as it goes"
        )
        if meta["stored_frames"] == 0:
            logger.warning(f"{repo_id}: no episode fits in max_cache_gb={max_cache_gb:g}; the cache stores nothing")
    # Disk space before anything is written (a resumed cache only needs room for what is not on disk yet).
    check_free_disk(cache_dir, meta, min_free_disk_gb)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if created_by_training and not resume:
        mark_created_by_training(cache_dir)
    (cache_dir / PROGRESS_DIR).mkdir(exist_ok=True)
    if not resume:
        if stored is not None:
            tmp = cache_dir / f".{STORED_FILE}.{os.getpid()}.npy"
            np.save(tmp, stored)
            os.replace(tmp, cache_dir / STORED_FILE)
        for key in keys:
            c, h, w = shapes[key]
            arr = np.lib.format.open_memmap(cache_dir / key_to_filename(key), mode="w+", dtype=np.uint8,
                                            shape=(n, c, h, w))
            del arr
        np.save(cache_dir / TIMESTAMPS_FILE, timestamps)
    done = {int(p.stem.split("_")[1]) for p in (cache_dir / PROGRESS_DIR).glob("ep_*.done")}
    if not (cache_dir / FILLED_FILE).exists():
        # A fresh cache, or one started before filled.npy existed: frames of finished episodes are filled.
        tmp = cache_dir / f".{FILLED_FILE}.{os.getpid()}.tmp"
        filled = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n,))
        for e in done:
            filled[ep_from[e]:ep_to[e]] = 1
        filled.flush()
        del filled
        os.replace(tmp, cache_dir / FILLED_FILE)
    if not resume:
        atomic_write_text(meta_path, json.dumps(meta, indent=2))  # last: the layout is ready
    if layout_only:
        filled = np.load(cache_dir / FILLED_FILE, mmap_mode="r")
        logger.info(f"{repo_id}: cache layout at {cache_dir} ({n} frames, keys={keys}, "
                    f"{cache_disk_bytes(meta) / 1e9:.2f} GB when filled"
                    + (f", stores {meta['stored_frames']} frames" if stored is not None else "")
                    + f", {int(np.count_nonzero(filled))} frames already filled)")
        return cache_dir
    (cache_dir / COMPLETE_FILE).unlink(missing_ok=True)

    # Episodes whose frames are all filled (e.g. by a run with cache_mode=during_training) are done as well.
    filled = np.load(cache_dir / FILLED_FILE, mmap_mode="r")
    todo = [e for e in range(int(ds.meta.total_episodes))
            if keep_ep[e] and e not in done and not bool(np.all(filled[ep_from[e]:ep_to[e]]))]
    del filled
    logger.info(f"{repo_id}: {n} frames, keys={keys}, cache={cache_disk_bytes(meta) / 1e9:.1f} GB at {cache_dir}; "
                f"{len(todo)} of {int(keep_ep.sum())} stored episodes to decode"
                + (f" ({ds.meta.total_episodes} in the dataset)" if stored is not None else ""))

    num_workers = max(0, (os.cpu_count() or 2) // 2 if num_workers is None else num_workers)
    wargs = dict(repo_id=repo_id, root=root, revision=revision, video_backend=video_backend,
                 tolerance_s=tolerance_s, cache_dir=str(cache_dir), keys=keys)
    t0 = time.perf_counter()
    frames_done = 0
    if todo:
        if num_workers == 0:
            _worker_init(wargs)
            it = map(_decode_episode, todo)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(num_workers, initializer=_worker_init, initargs=(wargs,))
            it = pool.imap_unordered(_decode_episode, todo)
        try:
            for i, (_, nf, _) in enumerate(it, 1):
                frames_done += nf
                if i % max(1, len(todo) // 20) == 0 or i == len(todo):
                    dt = time.perf_counter() - t0
                    logger.info(f"  {i}/{len(todo)} episodes, {frames_done / max(dt, 1e-9):.0f} frames/s")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    if verify > 0:
        _verify(ds, cache_dir, keys, verify, None if stored is None else np.flatnonzero(stored))
    write_complete(cache_dir)
    logger.info(f"done in {time.perf_counter() - t0:.0f}s -> {cache_dir}")
    return cache_dir


def _verify(ds, cache_dir: Path, keys: list[str], n_samples: int, rows: np.ndarray | None = None) -> None:
    """Compare random frames (of ``rows``, default all) against LeRobotDataset.__getitem__ (bit-exact; ``ds`` has
    return_uint8=True)."""
    meta = json.loads((cache_dir / META_FILE).read_text())
    arrays = {k: np.load(cache_dir / meta["keys"][k]["file"], mmap_mode="r") for k in keys}
    rng = random.Random(0)
    pool = range(len(ds)) if rows is None else [int(r) for r in rows]
    idxs = rng.sample(pool, min(n_samples, len(pool)))
    saved_delta, saved_tf = _lr.delta_indices(ds), _lr.image_transforms(ds)
    _lr.set_delta_indices(ds, None)
    _lr.set_image_transforms(ds, None)
    try:
        for i in idxs:
            item = ds[i]
            for k in keys:
                ref = item[k]
                got = torch.from_numpy(np.array(arrays[k][i]))
                if ref.dtype != torch.uint8:  # parquet images are float32 / 255 even with return_uint8
                    got = got.to(torch.float32) / 255
                if not torch.equal(ref, got):
                    diff = (ref - got).abs().max().item()
                    raise RuntimeError(f"verify failed: frame {i} key {k} differs (max abs diff {diff:.3g})")
    finally:
        _lr.set_delta_indices(ds, saved_delta)
        _lr.set_image_transforms(ds, saved_tf)
    logger.info(f"verify: {len(idxs)} random frames x {len(keys)} keys match LeRobotDataset exactly")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None, help="local dataset directory (as for LeRobotDataset)")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--video_backend", default=None, help="torchcodec | pyav (default: LeRobot's default)")
    p.add_argument("--tolerance_s", type=float, default=1e-4)
    p.add_argument("--keys", nargs="*", default=None, help="camera keys to cache (default: all)")
    p.add_argument("--num_workers", type=int, default=None, help="decode processes (default: cpu_count/2)")
    p.add_argument("--verify", type=int, default=32, help="random frames to check bit-exactly (0 = skip)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--min_free_disk_gb", type=float, default=DEFAULT_MIN_FREE_DISK_GB,
                   help="refuse unless free disk minus the bytes still to write stays above this (GB; a safety "
                        "margin, not a size limit; -1 = no check)")
    p.add_argument("--max_cache_gb", type=float, default=DEFAULT_MAX_CACHE_GB,
                   help="largest cache (GB of frames): a bigger dataset stores whole episodes up to this size and "
                        "training decodes the others (-1 = no limit)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from fastload.cache import DiskSpaceError

    try:
        build_cache(a.repo_id, a.cache_dir, root=a.root, revision=a.revision, video_backend=a.video_backend,
                    tolerance_s=a.tolerance_s, keys=a.keys, num_workers=a.num_workers, verify=a.verify,
                    overwrite=a.overwrite, min_free_disk_gb=a.min_free_disk_gb, max_cache_gb=a.max_cache_gb)
    except DiskSpaceError as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
