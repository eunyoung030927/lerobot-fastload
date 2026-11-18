"""Data-loading throughput: LeRobot video DataLoader vs cache DataLoader vs FastLoader.

Measures only data loading (batches delivered on the GPU), not model compute, so the numbers are
the ceiling each loader can feed. Single process::

    python benchmarks/bench_loader.py --repo_id <repo> --root <dir> --cache_dir <cache> --batch 32

Multi-GPU (each rank loads its own shard, like DDP training)::

    torchrun --nproc_per_node 2 benchmarks/bench_loader.py ... --modes fast_host,fast_gpu
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def _to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _time(it, steps: int, warmup: int, device, to_device: bool) -> tuple[float, float]:
    for _ in range(warmup):
        b = next(it)
        if to_device:
            b = _to_device(b, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        b = next(it)
        if to_device:
            b = _to_device(b, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    return dt / steps * 1e3, steps / dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--video_backend", default=None)
    p.add_argument("--modes", default="video,cache_dl,fast_host,fast_gpu")
    a = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import cycle

    from fastload import CachedLeRobotDataset, FastLoader

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")

    meta = LeRobotDatasetMetadata(a.repo_id, root=a.root)
    dt = {"action": [i / meta.fps for i in range(a.horizon)]} if "action" in meta.features else None

    def dataset(cached: bool):
        ds = LeRobotDataset(a.repo_id, root=a.root, delta_timestamps=dt, video_backend=a.video_backend)
        return CachedLeRobotDataset.from_dataset(ds, a.cache_dir) if cached else ds

    def dataloader(ds):
        sampler = (torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
                   if world > 1 else None)
        return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=sampler is None, sampler=sampler,
                                           num_workers=a.num_workers, pin_memory=device.type == "cuda",
                                           prefetch_factor=2 if a.num_workers > 0 else None,
                                           persistent_workers=a.num_workers > 0)

    results = []
    for mode in a.modes.split(","):
        ld = None
        if mode == "video":
            it, to_dev = cycle(dataloader(dataset(False))), True
        elif mode == "cache_dl":
            it, to_dev = cycle(dataloader(dataset(True))), True
        elif mode in ("fast_host", "fast_gpu"):
            if mode == "fast_gpu" and device.type != "cuda":
                continue
            ld = FastLoader(dataset(True), a.batch, placement="gpu" if mode == "fast_gpu" else "host",
                            device=device, rank=rank, world_size=world)
            it, to_dev = cycle(ld), False
        else:
            raise SystemExit(f"unknown mode {mode}")
        ms, bps = _time(it, a.steps, a.warmup, device, to_dev)
        results.append((mode, ms, bps * a.batch))
        del it
        if ld is not None:
            ld.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        print(f"\nper-rank batch {a.batch}, world {world}, workers {a.num_workers} (video/cache_dl)")
        print(f"{'mode':<10} {'ms/batch':>10} {'samples/s/rank':>16}")
        for mode, ms, sps in results:
            print(f"{mode:<10} {ms:>10.1f} {sps:>16.0f}")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
