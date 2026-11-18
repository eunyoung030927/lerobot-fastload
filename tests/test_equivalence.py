"""Equivalence test: cache / FastLoader outputs must equal LeRobotDataset outputs exactly.

Run as a script (recommended)::

    python tests/test_equivalence.py --repo_id <repo> --root <dataset dir> --cache_dir <cache dir>

or with pytest, after exporting FASTLOAD_TEST_REPO, FASTLOAD_TEST_ROOT, FASTLOAD_TEST_CACHE.

Checks
  1. CachedLeRobotDataset[i] == LeRobotDataset[i] (episode starts/ends + random frames), with an
     action chunk and a 2-frame observation history (video delta queries and padding).
  2. FastLoader.get_batch(rows) == default_collate([LeRobotDataset[i] for i in rows]) with frames
     from the host tier only, from a partial GPU tier, and from a full GPU tier.
  3. The same with a subset of episodes loaded (absolute -> relative index mapping).
  4. DDP sharding: ranks get disjoint whole episodes that cover the data.
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

    return "_get_query_indices(abs_idx" in inspect.getsource(LeRobotDataset.__getitem__)


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
        ds = LeRobotDataset(repo_id, root=root, episodes=episodes, delta_timestamps=dt, video_backend=backend)
        return CachedLeRobotDataset.from_dataset(ds, cache_dir) if cached else ds

    def check_loader(ref, cached, rows, tag):
        loader = FastLoader(cached, batch, shuffle=False, placement="host", device=device, rank=0, world_size=1)
        tiers = [("host", None)]
        if loader.is_cuda:
            shard = len(loader.local_rows) * loader.frame_bytes
            tiers += [("gpu-partial", shard // 2), ("gpu-full", None)]
        for tier_name, budget in tiers:
            if tier_name != "host":
                loader.build_gpu_cache(budget)
            for s in range(0, len(rows), batch):
                chunk = rows[s:s + batch]
                got = loader.get_batch(chunk)
                exp = default_collate([ref[i] for i in chunk])
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
    check_loader(ref, cached, rows, "full")

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
    shards = [FastLoader(cached, batch, placement="host", device="cpu", rank=r, world_size=3) for r in range(3)]
    all_rows = np.concatenate([s.local_rows for s in shards])
    if len(all_rows) != len(cached) or len(np.unique(all_rows)) != len(cached):
        errors.append("sharding: ranks are not a disjoint cover of the dataset")
    ep_sets = [set(s.local_episodes.tolist()) for s in shards]
    if any(ep_sets[i] & ep_sets[j] for i in range(3) for j in range(i + 1, 3)):
        errors.append("sharding: an episode is split across ranks")
    print(f"  sharding over 3 ranks: episodes {[len(s) for s in ep_sets]}: "
          f"{'OK' if not any('sharding' in e for e in errors) else 'FAIL'}")
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
