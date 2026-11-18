"""Decode a LeRobot dataset's camera streams once into a uint8 memmap cache.

Usage::

    python -m fastload.build_cache --repo_id <repo> --root <dataset dir> --cache_dir <out dir> \
        --num_workers 16

Frames are decoded through LeRobot's own ``DatasetReader._query_videos`` (same backend, same timestamp
matching, ``return_uint8=True`` as in ``lerobot_train``), so the cache holds exactly the pixels LeRobot would
have produced during training. Depth cameras (``meta.depth_keys``, 12-bit / float) are not cached; datasets
read through :class:`fastload.CachedLeRobotDataset` decode them with LeRobot's code.
The build is resumable: finished episodes are recorded under ``progress/`` and skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    META_FILE,
    PROGRESS_DIR,
    TIMESTAMPS_FILE,
    info_fingerprint,
    key_to_filename,
)

logger = logging.getLogger("fastload.build_cache")

# Decoded float32 frames held per worker at once (bytes). Keeps 480x640 videos from blowing RAM.
_CHUNK_FLOAT_BYTES = 256 * 1024 * 1024


def _open_dataset(repo_id: str, root: str | None, revision: str | None, video_backend: str | None,
                  tolerance_s: float):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # return_uint8=True: video frames come back as the decoder's raw uint8 (no float round trip).
    return LeRobotDataset(
        repo_id, root=root, revision=revision, video_backend=video_backend, tolerance_s=tolerance_s,
        return_uint8=True,
    )


def _to_uint8(frames: torch.Tensor) -> np.ndarray:
    """uint8 frames as they are; float32 frames (parquet images: uint8 / 255) inverted exactly."""
    if frames.dtype == torch.uint8:
        return frames.numpy()
    return frames.mul(255).round_().clamp_(0, 255).to(torch.uint8).numpy()


# ------------------------------------------------------------------------------------------------
# worker side
# ------------------------------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(args: dict[str, Any]) -> None:
    torch.set_num_threads(1)
    ds = _open_dataset(args["repo_id"], args["root"], args["revision"], args["video_backend"],
                       args["tolerance_s"])
    cache_dir = Path(args["cache_dir"])
    meta = json.loads((cache_dir / META_FILE).read_text())
    _W.clear()
    _W.update(
        ds=ds,
        cache_dir=cache_dir,
        keys=args["keys"],
        video_keys=set(ds.meta.video_keys),
        arrays={k: np.load(cache_dir / meta["keys"][k]["file"], mmap_mode="r+") for k in args["keys"]},
        timestamps=np.load(cache_dir / TIMESTAMPS_FILE, mmap_mode="r"),
        chunk={k: max(8, _CHUNK_FLOAT_BYTES // (4 * int(np.prod(meta["keys"][k]["shape"]))))
               for k in args["keys"]},
    )


def _decode_episode(ep_idx: int) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    ds = _W["ds"]
    ep = ds.meta.episodes[ep_idx]
    start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    ts_all = _W["timestamps"][start:end]
    for key in _W["keys"]:
        out = _W["arrays"][key]
        step = _W["chunk"][key]
        for s in range(0, end - start, step):
            e = min(s + step, end - start)
            if key in _W["video_keys"]:
                frames = _lr.query_videos(ds, {key: [float(t) for t in ts_all[s:e]]}, ep_idx)[key]
                if frames.ndim == 3:  # LeRobot squeezes a single-timestamp query
                    frames = frames.unsqueeze(0)
            else:  # image stored in parquet
                frames = torch.stack(ds.hf_dataset[start + s:start + e][key])
            out[start + s:start + e] = _to_uint8(frames)
        out.flush()
    (_W["cache_dir"] / PROGRESS_DIR / f"ep_{ep_idx:06d}.done").touch()
    return ep_idx, end - start, time.perf_counter() - t0


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
) -> Path:
    """Build (or resume building) the cache for one LeRobot dataset. Returns the cache dir."""
    from fastload.decode_hook import active_hook, hook_name

    if active_hook() is not None:
        raise RuntimeError(
            f"fastload: decode_video_frames is wrapped ({hook_name(active_hook())}); build the cache in a process "
            "without the wrapper. The cache stores the decoder's own frames and the wrapper is applied at training time."
        )
    cache_dir = Path(cache_dir)
    root = str(root) if root is not None else None
    ds = _open_dataset(repo_id, root, revision, video_backend, tolerance_s)
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

    # Frame shape = what LeRobot actually returns (C, H, W), probed on frame 0.
    shapes: dict[str, list[int]] = {}
    for key in keys:
        if key in ds.meta.video_keys:
            f = _lr.query_videos(ds, {key: [float(timestamps[0])]}, int(ds.hf_dataset[0]["episode_index"]))[key]
        else:
            f = ds.hf_dataset[0][key]
        shapes[key] = list(f.shape[-3:])

    fingerprint = info_fingerprint(ds.root)
    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "repo_id": repo_id,
        "dataset_root": str(ds.root),
        "info_fingerprint": fingerprint,
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
    elif cache_dir.exists() and overwrite:
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / PROGRESS_DIR).mkdir(exist_ok=True)
    if not resume:
        for key in keys:
            c, h, w = shapes[key]
            arr = np.lib.format.open_memmap(cache_dir / key_to_filename(key), mode="w+", dtype=np.uint8,
                                            shape=(n, c, h, w))
            del arr
        np.save(cache_dir / TIMESTAMPS_FILE, timestamps)
        meta_path.write_text(json.dumps(meta, indent=2))
    (cache_dir / COMPLETE_FILE).unlink(missing_ok=True)

    done = {int(p.stem.split("_")[1]) for p in (cache_dir / PROGRESS_DIR).glob("ep_*.done")}
    todo = [e for e in range(int(ds.meta.total_episodes)) if e not in done]
    total_gb = sum(np.prod(s) for s in shapes.values()) * n / 1e9
    logger.info(f"{repo_id}: {n} frames, keys={keys}, cache={total_gb:.1f} GB at {cache_dir}; "
                f"{len(todo)} of {ds.meta.total_episodes} episodes to decode")

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
        _verify(ds, cache_dir, keys, verify)
    (cache_dir / COMPLETE_FILE).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(f"done in {time.perf_counter() - t0:.0f}s -> {cache_dir}")
    return cache_dir


def _verify(ds, cache_dir: Path, keys: list[str], n_samples: int) -> None:
    """Compare random frames against LeRobotDataset.__getitem__ (bit-exact; ``ds`` has return_uint8=True)."""
    meta = json.loads((cache_dir / META_FILE).read_text())
    arrays = {k: np.load(cache_dir / meta["keys"][k]["file"], mmap_mode="r") for k in keys}
    rng = random.Random(0)
    idxs = rng.sample(range(len(ds)), min(n_samples, len(ds)))
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
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_cache(a.repo_id, a.cache_dir, root=a.root, revision=a.revision, video_backend=a.video_backend,
                tolerance_s=a.tolerance_s, keys=a.keys, num_workers=a.num_workers, verify=a.verify,
                overwrite=a.overwrite)


if __name__ == "__main__":
    main()
