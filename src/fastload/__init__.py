"""fastload: decode LeRobot videos once, then train from a uint8 cache held in RAM and VRAM."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["CachedLeRobotDataset", "FastLoader", "FrameCache", "build_cache", "make_loader", "__version__"]


def __getattr__(name: str) -> Any:  # lazy imports keep `python -m fastload.build_cache` light
    if name == "CachedLeRobotDataset":
        from fastload.dataset import CachedLeRobotDataset

        return CachedLeRobotDataset
    if name == "FastLoader":
        from fastload.loader import FastLoader

        return FastLoader
    if name == "FrameCache":
        from fastload.cache import FrameCache

        return FrameCache
    if name == "build_cache":
        from fastload.build_cache import build_cache

        return build_cache
    if name == "make_loader":
        return make_loader
    raise AttributeError(name)


def make_loader(dataset: Any, batch_size: int, *, cache_dir: str | None = None, **kwargs: Any):
    """FastLoader for an existing LeRobotDataset, for custom training scripts.

    Replace::

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, ...)
        policy, optimizer, dataloader, sched = accelerator.prepare(policy, optimizer, dataloader, sched)

    with::

        dataloader = fastload.make_loader(dataset, bs, cache_dir=CACHE_DIR, shuffle=True, num_workers=NUM_WORKERS,
                                          persistent_workers=PERSISTENT_WORKERS, generator=GENERATOR)
        policy, optimizer, dataloader, sched = accelerator.prepare(policy, optimizer, dataloader, sched)

    (create it after the Accelerator so it can read rank / world size / device). ``accelerator.prepare``
    returns the same FastLoader; with the default ``order="lerobot"`` it shards and seeds the batch order
    exactly as it would a DataLoader (required with several processes). Pass the ``num_workers`` and
    ``persistent_workers`` (and ``generator``) of the DataLoader you are replacing so image-transform randomness
    matches (and its ``collate_fn`` if it is LeRobot's ``lerobot_collate_fn``). Keyword arguments are those of
    :class:`fastload.loader.FastLoader` (sampler, shuffle, order, num_workers, placement, ...), including
    ``cache_mode="during_training"`` (no separate build step: the loader decodes missing frames into the cache).
    """
    from fastload.dataset import CachedLeRobotDataset
    from fastload.loader import FastLoader

    if not isinstance(dataset, CachedLeRobotDataset) and cache_dir is None:
        raise ValueError("pass cache_dir=... (or a CachedLeRobotDataset)")
    # FastLoader converts the dataset (cache_mode / min_free_disk_gb / decode_workers apply there)
    return FastLoader(dataset, batch_size, cache_dir=cache_dir, **kwargs)
