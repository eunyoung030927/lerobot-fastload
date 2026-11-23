"""`lerobot-train` with fastload data loading.

Usage: the normal lerobot-train arguments plus ``--fastload.*`` options::

    accelerate launch --num_processes 8 -m fastload.train \
        --dataset.repo_id=<repo> --dataset.root=<dir> --policy.type=pi05 --batch_size=32 --steps=30000 \
        --fastload.cache_dir=/nvme/fastload/<repo>

The cache (``--fastload.cache_mode``): ``prebuilt`` (default) needs a complete cache from ``fastload.build_cache``;
``build_first`` builds a missing / incomplete one before ``lerobot_train.train()`` starts (before the Accelerator and
the process group exist, so no collective can time out; only ``LOCAL_RANK`` 0 of each node builds, the other local
ranks wait for it); ``during_training`` allocates it there and FastLoader fills it while the first epoch runs (see
:mod:`fastload.fill`). The dataset to decode comes from the ``--dataset.*`` arguments (or the ``train_config.json`` of
``--config_path`` when resuming). A cache created this way holds at most ``--fastload.max_cache_gb`` (default 300 GB)
of frames, whole episodes spread over the dataset; the frames of the other episodes are decoded during training. It
is deleted when training ends normally (``--fastload.delete_cache``, default true; a cache from ``build_cache`` is
never deleted, nor one another training process still uses).

This runs LeRobot's own ``lerobot.scripts.lerobot_train.train`` unchanged except for three hooks:

1. ``make_dataset`` returns a :class:`CachedLeRobotDataset` (frames from the cache, not video).
2. The ``DataLoader`` that train() builds for that dataset becomes a :class:`FastLoader`
   (``--fastload.loader=dataloader`` keeps LeRobot's DataLoader and only swaps the video decoding).
3. ``update_policy`` handles a CUDA OOM: one process -> halve the GPU frame tier and retry the step;
   several processes -> stop all of them at once with a note that ``python -m fastload.supervise`` uses to
   restart from the last checkpoint with a smaller tier. (FastLoader also shrinks the tier before an OOM
   when the run's peak memory grows; see ``memory_guard``.)

Checkpoints, train_config.json, logging, eval and resume are LeRobot's; ``--fastload.*`` options are
not written into train_config.json, so checkpoints stay loadable by plain LeRobot.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fastload import _lr
from fastload.cache import (
    CACHE_MODES,
    DEFAULT_MAX_CACHE_GB,
    DEFAULT_MIN_FREE_DISK_GB,
    DiskSpaceError,
    acquire_lease,
    cache_status,
    missing_cache_message,
    release_lease,
)
from fastload.dataset import CachedLeRobotDataset
from fastload.loader import ORDERS, PLACEMENTS, FastLoader, active_loaders

logger = logging.getLogger("fastload")

PREFIX = "--fastload."


@dataclass
class FastLoadConfig:
    cache_dir: str | None = None
    # "prebuilt" = a complete cache from `python -m fastload.build_cache` is required; "build_first" = build a
    # missing / incomplete cache before training starts; "during_training" = the first epoch decodes the frames it
    # needs into the cache (in decoder processes), later epochs read it.
    cache_mode: str = "prebuilt"
    # Decoder processes that build (build_first) or fill (during_training) the cache. Default: build_first -> half
    # the CPU cores; during_training -> half the CPU cores divided by the training processes on the node.
    decode_workers: int | None = None
    # during_training: while decoders are idle, decode frames no batch asked for yet, so the cache completes.
    background_fill: bool = True
    # Refuse to write the cache unless this much disk stays free afterwards (GB; a safety margin, -1 = no check).
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB
    # Largest cache build_first / during_training create (GB of frames): a bigger dataset stores whole episodes up to
    # this size and the frames of the others are decoded during training (-1 = no limit).
    max_cache_gb: float = DEFAULT_MAX_CACHE_GB
    # Delete the cache when training ends normally, if a training run created it (build_first / during_training) and
    # no other training process uses it. A cache from `python -m fastload.build_cache` is never deleted.
    delete_cache: bool = True
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
    # A wrapper around LeRobot's decode_video_frames (e.g. a data-loader resize) runs on the
    # "cpu" (bit-identical to LeRobot) or on the "gpu" (faster, float-rounding differences).
    decode_hook_device: str = "cpu"
    # Image transforms (augmentation) on the "gpu" (fast; pixels differ from LeRobot by float rounding)
    # or on the "cpu" (LeRobot's exact path in the prefetch thread: bit-identical, no VRAM tier).
    augment_device: str = "gpu"
    # Check free VRAM every step and drop GPU-tier chunks before the model runs out of memory.
    memory_guard: bool = True
    # CUDA OOM in a step: one process -> halve the GPU tier and retry the step; several processes -> stop all
    # of them at once and leave a note for `python -m fastload.supervise` to restart from the last checkpoint.
    oom_retry: bool = True

    def validate(self) -> None:
        if not self.cache_dir:
            raise SystemExit("fastload: --fastload.cache_dir=<dir> is required (build it with fastload.build_cache)")
        if self.loader not in ("fast", "dataloader"):
            raise SystemExit(f"fastload: --fastload.loader must be fast|dataloader, got {self.loader}")
        if self.cache_mode not in CACHE_MODES:
            raise SystemExit(f"fastload: --fastload.cache_mode must be one of {'|'.join(CACHE_MODES)}, "
                             f"got {self.cache_mode}")
        if self.cache_mode == "during_training" and self.loader != "fast":
            raise SystemExit("fastload: --fastload.cache_mode=during_training needs --fastload.loader=fast (FastLoader "
                             "fills the cache); use build_first with --fastload.loader=dataloader")
        if self.order not in ORDERS:
            raise SystemExit(f"fastload: --fastload.order must be one of {ORDERS}, got {self.order}")
        if self.augment_device not in ("cpu", "gpu"):
            raise SystemExit(f"fastload: --fastload.augment_device must be cpu|gpu, got {self.augment_device}")
        if self.decode_hook_device not in ("cpu", "gpu"):
            raise SystemExit(f"fastload: --fastload.decode_hook_device must be cpu|gpu, got {self.decode_hook_device}")
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


EXIT_OOM = 86  # exit status of a rank stopped by stop_after_oom (fastload.supervise looks for it)
NOTE_DIR_ENV = "FASTLOAD_SUPERVISE_DIR"
# Testing only: "RANK:STEP" raises a CUDA OOM on that rank at that step of a fresh (not resumed) run.
DEBUG_OOM_ENV = "FASTLOAD_DEBUG_OOM"


def _debug_oom_now(step: int) -> bool:
    spec = os.environ.get(DEBUG_OOM_ENV)
    if not spec or any(a.startswith("--resume") and a.endswith("true") for a in sys.argv):
        return False
    rank, at = (int(x) for x in spec.split(":"))
    return int(os.environ.get("RANK", 0)) == rank and step == at


def oom_note_dir(output_dir: str | None) -> Path | None:
    d = os.environ.get(NOTE_DIR_ENV)
    if d:
        return Path(d)
    return Path(output_dir) / "fastload_oom" if output_dir else None


def stop_after_oom(loaders: list[FastLoader], output_dir: str | None, world: int) -> None:
    """Write an OOM note for fastload.supervise and terminate this rank immediately (never returns)."""
    rank = loaders[0].rank if loaders else int(os.environ.get("RANK", 0))
    tier_bytes = int(sum(ld._tier.nbytes for ld in loaders))
    tier_gb = tier_bytes / 1e9
    note = {
        "rank": rank,
        "world_size": world,
        "output_dir": output_dir,
        "gpu_tier_bytes": tier_bytes,
        "time": time.time(),
    }
    d = oom_note_dir(output_dir)
    if d is not None:
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"rank{rank}.json").write_text(json.dumps(note))
        except OSError as e:  # pragma: no cover
            logger.error(f"fastload: could not write the OOM note to {d}: {e}")
    logger.error(
        f"fastload: rank {rank}: CUDA OOM during a multi-process step (GPU tier {tier_gb:.1f} GB). Stopping all "
        f"processes; restart from the last checkpoint with --fastload.max_gpu_cache_gb={tier_gb / 2:.4g} "
        "(python -m fastload.supervise does this automatically)"
    )
    for h in logging.getLogger().handlers + logger.handlers:
        try:
            h.flush()
        except Exception:  # pragma: no cover
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(EXIT_OOM)


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
        state["output_dir"] = str(cfg.output_dir) if getattr(cfg, "output_dir", None) else None
        ds = orig_make_dataset(cfg)
        # build_first / during_training already prepared the cache in main() (prepare_cache)
        ds = CachedLeRobotDataset.from_dataset(ds, fl.cache_dir, cache_mode=fl.cache_mode,
                                               min_free_disk_gb=fl.min_free_disk_gb, build_workers=fl.decode_workers,
                                               max_cache_gb=fl.max_cache_gb)
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
                    decode_hook_device=fl.decode_hook_device,
                    memory_guard=fl.memory_guard,
                    augment_device=fl.augment_device,
                    decode_workers=fl.decode_workers,
                    background_fill=fl.background_fill,
                )
            return orig_dataloader(dataset, *args, **kwargs)

    def update_policy(train_metrics, policy, batch, optimizer, *args, **kwargs):
        state["steps"] = state.get("steps", 0) + 1
        try:
            if _debug_oom_now(state["steps"]):
                raise torch.OutOfMemoryError(f"fastload: simulated CUDA OOM ({DEBUG_OOM_ENV})")
            return orig_update_policy(train_metrics, policy, batch, optimizer, *args, **kwargs)
        except torch.OutOfMemoryError:
            loaders = [ld for ld in active_loaders() if ld._tier.tensors]
            world = max([ld.world_size for ld in active_loaders()] or [1])
            if world > 1 and fl.oom_retry:
                # The other ranks are already waiting in this step's gradient all-reduce, so this rank cannot
                # redo the step alone. Stop every process now (instead of after the NCCL timeout) and leave
                # a note so `python -m fastload.supervise` restarts from the last checkpoint with a smaller tier.
                stop_after_oom(loaders, state.get("output_dir"), world)
            if not fl.oom_retry or not loaders:
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


def _close_loaders() -> None:
    for ld in active_loaders():
        try:
            ld.close()
        except Exception as e:  # pragma: no cover
            logger.warning(f"fastload: closing a FastLoader failed: {e}")


def prepare_cache(fl: FastLoadConfig, train_args: list[str]) -> None:
    """Before ``lerobot_train.train()``: check (prebuilt), build (build_first) or allocate (during_training) the cache.

    Runs before the Accelerator / process group exists (LeRobot 0.4's ``train()`` creates the Accelerator itself), so
    a long build cannot hit a collective timeout. With several processes per node only ``LOCAL_RANK`` 0 builds; the
    others wait (:func:`fastload.fill.ensure_cache`).
    """
    from fastload.fill import ensure_cache

    status = cache_status(fl.cache_dir)
    if fl.cache_mode == "prebuilt":
        if status != "complete":
            try:
                src = _lr.dataset_source_from_cli(train_args)
            except SystemExit:
                src = {}
            raise SystemExit("fastload: " + missing_cache_message(fl.cache_dir, src.get("repo_id"), src.get("root"),
                                                                  partial=status == "partial"))
        return
    # lerobot_train configures logging only inside train() (and replaces these handlers then)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
    try:
        ensure_cache(fl.cache_mode, fl.cache_dir, lambda: _lr.dataset_source_from_cli(train_args),
                     fl.min_free_disk_gb, num_workers=fl.decode_workers, max_cache_gb=fl.max_cache_gb)
    except DiskSpaceError as e:
        raise SystemExit(str(e)) from None


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    fl, rest = split_args(argv)
    fl.validate()
    prepare_cache(fl, rest)
    sys.argv = [sys.argv[0], *rest]  # lerobot's parser reads sys.argv

    import lerobot.scripts.lerobot_train as lt

    undo, state = install_hooks(fl)
    lease = acquire_lease(fl.cache_dir)  # another run ending meanwhile does not delete the cache under this one
    finished = False
    try:
        try:  # lerobot >= 0.4.4
            from lerobot.utils.import_utils import register_third_party_plugins

            register_third_party_plugins()
        except ImportError:
            pass
        lt.train()
        finished = True
    finally:
        undo()
        _close_loaders()  # decoder processes and scratch files go before the cache may be deleted
        release_lease(fl.cache_dir, lease, delete=fl.delete_cache and finished)
    if fl.loader == "fast" and state["fast_loaders"] == 0:
        logger.warning("fastload: train() never built a DataLoader for the cached dataset; FastLoader was unused")


if __name__ == "__main__":
    main()
