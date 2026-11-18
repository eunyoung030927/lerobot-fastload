"""Equivalence test: cache / FastLoader outputs must equal LeRobotDataset outputs exactly.

Run as a script (recommended)::

    python tests/test_equivalence.py --repo_id <repo> --root <dataset dir> --cache_dir <cache dir>

or with pytest, after exporting FASTLOAD_TEST_REPO, FASTLOAD_TEST_ROOT, FASTLOAD_TEST_CACHE.

Checks (each with ``return_uint8=True`` - lerobot_train since LeRobot 0.6, uint8 video frames - and
``return_uint8=False``, the LeRobotDataset default)
  1. CachedLeRobotDataset[i] == LeRobotDataset[i] (episode starts/ends + random frames), with an
     action chunk and a 2-frame observation history (video delta queries and padding), and the cached
     dataset decodes no video.
  2. FastLoader.get_batch(rows) == collate(LeRobotDataset.__getitems__(rows)) (what a DataLoader calls; the
     collate is lerobot_train's: lerobot_collate_fn, which keeps language columns as lists, for datasets with
     language columns, else default_collate), same keys in the same order, with frames from the host tier only,
     from a partial GPU tier, and from a full GPU tier.
  3. The same with a subset of episodes loaded (absolute -> relative index mapping).
  4. order="shard": ranks get disjoint whole episodes that cover the data, and with LeRobot's
     EpisodeAwareSampler (subset of episodes, absolute_to_relative_idx) the shards cover its rows.

Batch order, augmentation and RNG equivalence with order="lerobot": tests/test_lerobot_order.py.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import default_collate


def _eq(a, b, path: str, errors: list[str]) -> None:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            errors.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
            return
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape:
            errors.append(f"{path}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
        elif not torch.equal(a, b):
            d = (a.double() - b.double()).abs().max().item() if a.dtype != torch.bool else "bool"
            errors.append(f"{path}: values differ (max abs diff {d})")
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errors.append(f"{path}: keys {sorted(set(a) ^ set(b))} only on one side")
        elif list(a) != list(b):
            errors.append(f"{path}: key order {list(a)} vs {list(b)}")
        for k in set(a) & set(b):
            _eq(a[k], b[k], f"{path}.{k}", errors)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if list(a) != list(b):
            errors.append(f"{path}: lists differ")
    elif a != b:
        errors.append(f"{path}: {a!r} vs {b!r}")


def _lerobot_subset_indexing_fixed() -> bool:
    """LeRobot < 0.4.4 computes delta windows from the relative row when `episodes=` is a subset."""
    import inspect

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = inspect.getsource(LeRobotDataset.__getitem__)
    try:  # LeRobot >= 0.5 reads items in DatasetReader.get_item (0.7: get_items)
        from lerobot.datasets.dataset_reader import DatasetReader

        src += inspect.getsource(DatasetReader.get_item)
        src += inspect.getsource(getattr(DatasetReader, "get_items", DatasetReader.get_item))
    except ImportError:
        pass
    return "_get_query_indices(abs_idx" in src


def _pick_rows(ds, n: int, seed: int = 0) -> list[int]:
    """Episode boundary rows (where padding/clamping happens) plus random rows."""
    rng = random.Random(seed)
    ep = np.asarray(ds.hf_dataset.select_columns(["episode_index"]).with_format("numpy")[:]["episode_index"])
    starts = [0] + list(np.nonzero(np.diff(ep))[0] + 1)
    ends = [s - 1 for s in starts[1:]] + [len(ep) - 1]
    special = sorted(set(rng.sample(starts, min(4, len(starts))) + rng.sample(ends, min(4, len(ends)))))
    special += [min(e, s + 1) for s, e in zip(starts[:2], ends[:2])]
    rest = rng.sample(range(len(ep)), min(max(0, n - len(special)), len(ep)))
    return special + rest


def run(repo_id: str, root: str | None, cache_dir: str, n: int = 48, batch: int = 8, horizon: int = 50,
        backend: str | None = None, device: str | None = None) -> list[str]:
    errors: list[str] = []
    for return_uint8 in (True, False):
        print(f" return_uint8={return_uint8}")
        errors += [f"return_uint8={return_uint8}: {e}" for e in
                   _run(repo_id, root, cache_dir, n, batch, horizon, backend, device, return_uint8)]
    return errors


def _run(repo_id: str, root: str | None, cache_dir: str, n: int, batch: int, horizon: int,
         backend: str | None, device: str | None, return_uint8: bool) -> list[str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from fastload import CachedLeRobotDataset, FastLoader

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    fps = meta.fps
    dt = {"action": [i / fps for i in range(horizon)], "observation.state": [-1 / fps, 0.0]}
    dt.update({k: [-1 / fps, 0.0] for k in meta.camera_keys})
    dt = {k: v for k, v in dt.items() if k in meta.features}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    errors: list[str] = []

    def make(episodes=None, cached=False):
        ds = LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=dt, video_backend=backend,
                            return_uint8=return_uint8)
        return CachedLeRobotDataset.from_dataset(ds, cache_dir) if cached else ds

    # lerobot_train's collate rule: lerobot_collate_fn (keeps the language columns as lists) for datasets with them
    from lerobot.utils.collate import lerobot_collate_fn

    collate = lerobot_collate_fn if meta.has_language_columns else default_collate
    collate_fn = lerobot_collate_fn if meta.has_language_columns else None

    def check_loader(ref, cached, rows, tag):
        import fastload.loader as fl_loader

        loader = FastLoader(cached, batch, shuffle=False, placement="host", device=device, rank=0, world_size=1,
                            collate_fn=collate_fn)
        tiers = [("host", None)]
        if loader.is_cuda:
            shard = len(loader.local_rows) * loader.tier_frame_bytes
            # small chunks so the tier has several of them (gathers across chunks, shrinking drops chunks)
            fl_loader.CHUNK_BYTES = max(loader.tier_frame_bytes, shard // 7)
            tiers += [("gpu-partial", shard // 2), ("gpu-full", None), ("gpu-after-memory-guard", "guard")]
        for tier_name, budget in tiers:
            if budget == "guard":  # memory guard: pretend the peak grew into the margin -> drop about half the chunks
                loader.build_gpu_cache(None)
                half = loader._tier.nbytes / 2
                loader._headroom = lambda: (0.0, half)
                loader._guard_memory()
                del loader._headroom
            elif tier_name != "host":
                loader.build_gpu_cache(budget)
            for s in range(0, len(rows), batch):
                chunk = rows[s:s + batch]
                got = loader.get_batch(chunk)
                exp = collate(ref.__getitems__([int(i) for i in chunk]))
                _eq(got, exp, f"{tag}[{tier_name}] rows {chunk[:3]}..", errors)
            print(f"  {tag} FastLoader {tier_name}: {'OK' if not errors else 'FAIL'}"
                  + (f" (gpu tier {100 * loader.gpu_cache_fraction:.0f}%)" if tier_name != "host" else ""))
        loader.release_gpu_cache()

    # 1 + 2: full dataset
    ref, cached = make(), make(cached=True)
    rows = _pick_rows(ref, n)
    for i in rows:
        _eq(cached[i], ref[i], f"item[{i}]", errors)
    print(f"  CachedLeRobotDataset items ({len(rows)} rows): {'OK' if not errors else 'FAIL'}")
    if meta.video_keys:  # the cached dataset must not decode video at all (equal items alone would not show it)
        import lerobot.datasets.video_utils as vu

        # the backend decoders only: replacing video_utils.decode_video_frames itself would look like a user
        # wrapper around it to fastload (decode_hook.active_hook)
        names = [n for n in dir(vu) if n.startswith("decode_video_frames_")]
        saved = {n: getattr(vu, n) for n in names}

        def refuse(*a, **k):
            raise AssertionError("video decoded")

        try:
            for name in names:
                setattr(vu, name, refuse)
            cached[rows[0]]
            print("  CachedLeRobotDataset decodes no video: OK")
        except AssertionError:
            errors.append("CachedLeRobotDataset still decodes video (cache not used)")
            print("  CachedLeRobotDataset decodes no video: FAIL")
        finally:
            for name, fn in saved.items():
                setattr(vu, name, fn)
    check_loader(ref, cached, rows, "full")
    if meta.has_language_columns:  # lerobot_train always collates them with lerobot_collate_fn: others refused
        try:
            FastLoader(cached, batch, placement="host", device="cpu", rank=0, world_size=1)
            errors.append("language columns: FastLoader accepted collate_fn=None")
        except ValueError:
            pass
        print(f"  language columns: lerobot_collate_fn reproduced, default collate refused: "
              f"{'OK' if not any('language' in e for e in errors) else 'FAIL'}")

    # 3: subset of episodes (non-contiguous)
    if meta.total_episodes >= 4:
        eps = sorted(random.Random(1).sample(range(meta.total_episodes), min(5, meta.total_episodes // 2)))
        ref_s, cached_s = make(eps), make(eps, cached=True)
        rows_s = _pick_rows(ref_s, n // 2)
        n_err = len(errors)
        for i in rows_s:
            _eq(cached_s[i], ref_s[i], f"subset item[{i}]", errors)
        print(f"  episodes={eps} CachedLeRobotDataset items: {'OK' if len(errors) == n_err else 'FAIL'}")
        if _lerobot_subset_indexing_fixed():
            check_loader(ref_s, cached_s, rows_s, f"episodes={eps}")
        else:
            print("  episodes subset FastLoader: SKIPPED - this LeRobot passes the relative row to "
                  "_get_query_indices (fixed in 0.4.4); FastLoader implements the fixed behaviour")

    # 4: sharding
    shards = [FastLoader(cached, batch, placement="host", device="cpu", rank=r, world_size=3, order="shard",
                         collate_fn=collate_fn) for r in range(3)]
    all_rows = np.concatenate([s.local_rows for s in shards])
    if len(all_rows) != len(cached) or len(np.unique(all_rows)) != len(cached):
        errors.append("sharding: ranks are not a disjoint cover of the dataset")
    ep_sets = [set(s.local_episodes.tolist()) for s in shards]
    if any(ep_sets[i] & ep_sets[j] for i in range(3) for j in range(i + 1, 3)):
        errors.append("sharding: an episode is split across ranks")
    print(f"  sharding over 3 ranks: episodes {[len(s) for s in ep_sets]}: "
          f"{'OK' if not any('sharding' in e for e in errors) else 'FAIL'}")

    # 4b: order="shard" with LeRobot's EpisodeAwareSampler on a subset (its .indices are relative rows)
    if meta.total_episodes >= 4:
        from lerobot.datasets.sampler import EpisodeAwareSampler

        smp = EpisodeAwareSampler(cached_s.meta.episodes["dataset_from_index"],
                                  cached_s.meta.episodes["dataset_to_index"], episode_indices_to_use=cached_s.episodes,
                                  drop_n_last_frames=2, shuffle=True, seed=0,
                                  absolute_to_relative_idx=cached_s.absolute_to_relative_idx)
        expect = np.asarray(smp.indices, dtype=np.int64)
        shards = [FastLoader(cached_s, batch, sampler=smp, placement="host", device="cpu", rank=r, world_size=2,
                             order="shard", collate_fn=collate_fn) for r in range(2)]
        got = np.sort(np.concatenate([s.local_rows for s in shards]))
        ok = np.array_equal(got, np.sort(expect)) and expect.max() < len(cached_s)
        if not ok:
            errors.append("sharding with EpisodeAwareSampler: shard rows != sampler.indices")
        print(f"  sharding with EpisodeAwareSampler (episodes subset, {len(expect)} rows): {'OK' if ok else 'FAIL'}")
    return errors


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--video_backend", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    errors = run(a.repo_id, a.root, a.cache_dir, a.n, a.batch, a.horizon, a.video_backend, a.device)
    for e in errors[:30]:
        print("  -", e)
    print("PASS" if not errors else f"FAIL ({len(errors)} mismatches)")
    return 0 if not errors else 1


def test_equivalence():  # pytest entry point
    import pytest

    repo = os.environ.get("FASTLOAD_TEST_REPO")
    if not repo:
        pytest.skip("set FASTLOAD_TEST_REPO / FASTLOAD_TEST_ROOT / FASTLOAD_TEST_CACHE")
    errors = run(repo, os.environ.get("FASTLOAD_TEST_ROOT"), os.environ["FASTLOAD_TEST_CACHE"])
    assert not errors, errors[:10]


if __name__ == "__main__":
    sys.exit(main())
