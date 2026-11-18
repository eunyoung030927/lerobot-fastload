"""`lerobot-train` with fastload data loading.

Usage: the normal lerobot-train arguments plus ``--fastload.*`` options::

    accelerate launch --num_processes 8 -m fastload.train \
        --dataset.repo_id=<repo> --dataset.root=<dir> --policy.type=pi05 --batch_size=32 --steps=30000 \
        --fastload.cache_dir=/nvme/fastload/<repo>

This runs LeRobot's own ``lerobot.scripts.lerobot_train.train`` unchanged except for three hooks:

1. ``make_dataset`` returns a :class:`CachedLeRobotDataset` (frames from the cache, not video).
2. The ``DataLoader`` that train() builds for that dataset becomes a :class:`FastLoader`
   (``--fastload.loader=dataloader`` keeps LeRobot's DataLoader and only swaps the video decoding).
3. ``update_policy`` retries once after a CUDA OOM by shrinking the GPU frame tier (single process).

Checkpoints, train_config.json, logging, eval and resume are LeRobot's; ``--fastload.*`` options are
not written into train_config.json, so checkpoints stay loadable by plain LeRobot.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from dataclasses import dataclass
from typing import Any

import torch

from fastload.dataset import CachedLeRobotDataset
from fastload.loader import ORDERS, PLACEMENTS, FastLoader, active_loaders

logger = logging.getLogger("fastload")

PREFIX = "--fastload."


@dataclass
class FastLoadConfig:
    cache_dir: str | None = None
    # "fast" = FastLoader (GPU tier + prefetch thread); "dataloader" = LeRobot's DataLoader over the cache.
    loader: str = "fast"
    # "lerobot" = same batches, order and data RNG draws as LeRobot's DataLoader (needs the same --num_workers);
    # "shard" = per-rank episode shards (fastest multi-GPU, not LeRobot's order).
    order: str = "lerobot"
    placement: str = "auto"  # auto | gpu | host
    warmup_steps: int = 20
    gpu_margin_gb: float | None = None
    max_gpu_cache_gb: float | None = None
    prefetch: int = 2
    oom_retry: bool = True

    def validate(self) -> None:
        if not self.cache_dir:
            raise SystemExit("fastload: --fastload.cache_dir=<dir> is required (build it with fastload.build_cache)")
        if self.loader not in ("fast", "dataloader"):
            raise SystemExit(f"fastload: --fastload.loader must be fast|dataloader, got {self.loader}")
        if self.order not in ORDERS:
            raise SystemExit(f"fastload: --fastload.order must be one of {ORDERS}, got {self.order}")
        if self.placement not in PLACEMENTS:
            raise SystemExit(f"fastload: --fastload.placement must be one of {PLACEMENTS}, got {self.placement}")


def _convert(value: str, typ: Any) -> Any:
    t = str(typ)
    if value.lower() in ("none", "null") and "None" in t:
        return None
    if "bool" in t:
        return value.lower() in ("1", "true", "yes", "y")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    return value


def split_args(argv: list[str]) -> tuple[FastLoadConfig, list[str]]:
    """Pull ``--fastload.name=value`` (or ``--fastload.name value``) out of argv."""
    cfg = FastLoadConfig()
    fields = {f.name: f.type for f in dataclasses.fields(FastLoadConfig)}
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith(PREFIX):
            body = arg[len(PREFIX):]
            if "=" in body:
                name, value = body.split("=", 1)
            else:
                name, value = body, argv[i + 1] if i + 1 < len(argv) else ""
                i += 1
            if name not in fields:
                raise SystemExit(f"fastload: unknown option --fastload.{name} (known: {sorted(fields)})")
            setattr(cfg, name, _convert(value, fields[name]))
        else:
            rest.append(arg)
        i += 1
    return cfg, rest


def install_hooks(fl: FastLoadConfig):
    """Patch lerobot_train in place. Returns a callable that undoes the patches."""
    import lerobot.scripts.lerobot_train as lt

    for name in ("make_dataset", "update_policy", "train"):
        if not hasattr(lt, name):
            raise RuntimeError(f"fastload: lerobot.scripts.lerobot_train.{name} not found; unsupported lerobot version")

    orig_make_dataset = lt.make_dataset
    orig_update_policy = lt.update_policy
    orig_dataloader = torch.utils.data.DataLoader
    state = {"fast_loaders": 0}

    def make_dataset(cfg):
        ds = orig_make_dataset(cfg)
        ds = CachedLeRobotDataset.from_dataset(ds, fl.cache_dir)
        logger.info(f"fastload: frames from cache {fl.cache_dir} (loader={fl.loader})")
        return ds

    class DataLoaderDispatch(orig_dataloader):  # type: ignore[misc, valid-type]
        """Constructs a FastLoader when given a CachedLeRobotDataset, a normal DataLoader otherwise."""

        def __new__(cls, dataset=None, *args, **kwargs):
            if fl.loader == "fast" and isinstance(dataset, CachedLeRobotDataset):
                state["fast_loaders"] += 1
                sampler = kwargs.get("sampler")
                ignored = {k: kwargs[k] for k in ("pin_memory", "prefetch_factor") if k in kwargs}
                num_workers = int(kwargs.get("num_workers", 0) or 0)
                logger.info(f"fastload: FastLoader replaces DataLoader (order={fl.order}, no worker processes; "
                            f"num_workers={num_workers} only sets the emulated worker RNG; ignoring {ignored})")
                return FastLoader(
                    dataset,
                    batch_size=kwargs.get("batch_size", 1),
                    sampler=sampler,
                    shuffle=bool(kwargs.get("shuffle", False)),
                    drop_last=bool(kwargs.get("drop_last", False)),
                    placement=fl.placement,
                    warmup_steps=fl.warmup_steps,
                    gpu_margin_gb=fl.gpu_margin_gb,
                    max_gpu_cache_gb=fl.max_gpu_cache_gb,
                    prefetch=fl.prefetch,
                    order=fl.order,
                    num_workers=num_workers,
                )
            return orig_dataloader(dataset, *args, **kwargs)

    def update_policy(train_metrics, policy, batch, optimizer, *args, **kwargs):
        try:
            return orig_update_policy(train_metrics, policy, batch, optimizer, *args, **kwargs)
        except torch.OutOfMemoryError:
            loaders = [ld for ld in active_loaders() if ld._tier.tensors]
            world = loaders[0].world_size if loaders else 1
            if not fl.oom_retry or not loaders or world > 1:
                if loaders and world > 1:
                    logger.error("fastload: OOM with a GPU frame tier under multi-process training; rerun with "
                                 "a larger --fastload.gpu_margin_gb or --fastload.max_gpu_cache_gb")
                raise
            optimizer.zero_grad(set_to_none=True)
            for ld in loaders:
                ld.shrink_gpu_cache(0.5)
            logger.warning("fastload: CUDA OOM -> GPU frame tier halved, retrying the step once")
            return orig_update_policy(train_metrics, policy, batch, optimizer, *args, **kwargs)

    lt.make_dataset = make_dataset
    lt.update_policy = update_policy
    torch.utils.data.DataLoader = DataLoaderDispatch

    def undo():
        lt.make_dataset = orig_make_dataset
        lt.update_policy = orig_update_policy
        torch.utils.data.DataLoader = orig_dataloader

    return undo, state


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    fl, rest = split_args(argv)
    fl.validate()
    sys.argv = [sys.argv[0], *rest]  # lerobot's parser reads sys.argv

    import lerobot.scripts.lerobot_train as lt

    undo, state = install_hooks(fl)
    try:
        try:  # lerobot >= 0.4.4
            from lerobot.utils.import_utils import register_third_party_plugins

            register_third_party_plugins()
        except ImportError:
            pass
        lt.train()
    finally:
        undo()
    if fl.loader == "fast" and state["fast_loaders"] == 0:
        logger.warning("fastload: train() never built a DataLoader for the cached dataset; FastLoader was unused")


if __name__ == "__main__":
    main()
