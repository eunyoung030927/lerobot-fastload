"""LeRobot 0.4 specifics used to build and fill the cache (build_cache, ``cache_mode=build_first / during_training``).

:mod:`fastload.cache` and :mod:`fastload.fill` do not import LeRobot; everything they need from it goes through here
(and through :mod:`fastload.build_cache`'s worker processes). LeRobot is imported lazily, so
``python -m fastload.build_cache`` stays light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def open_dataset_for_cache(repo_id: str, root: str | None, revision: str | None, video_backend: str | None,
                           tolerance_s: float):
    """The dataset the cache is decoded from: every episode, no delta timestamps, no transforms. LeRobot 0.4
    returns frames as float32 (``uint8 / 255``); :func:`fastload.build_cache._to_uint8` inverts that exactly."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, revision=revision, video_backend=video_backend,
                          tolerance_s=tolerance_s)


def episode_bounds(meta) -> tuple[np.ndarray, np.ndarray]:
    """(dataset_from_index, dataset_to_index) of every episode, indexed by episode index."""
    eps = meta.episodes
    return (np.asarray(eps["dataset_from_index"], dtype=np.int64),
            np.asarray(eps["dataset_to_index"], dtype=np.int64))


def read_frames(ds, key: str, ep_idx: int, start: int, stop: int, timestamps: np.ndarray) -> torch.Tensor:
    """Frames ``start:stop`` (absolute indices, inside episode ``ep_idx``) of camera ``key`` as LeRobot reads them
    (``ds`` from :func:`open_dataset_for_cache`): video through the dataset's own ``_query_videos`` at the frames'
    timestamps, parquet images as stored (both float32 / 255). Returns ``[stop - start, C, H, W]``."""
    if key in ds.meta.video_keys:
        frames = ds._query_videos({key: [float(t) for t in timestamps]}, ep_idx)[key]
        if frames.ndim == 3:  # LeRobot squeezes a single-timestamp query
            frames = frames.unsqueeze(0)
        return frames
    return torch.stack(ds.hf_dataset[start:stop][key])  # all episodes loaded: row == absolute index


def dataset_source(ds) -> dict[str, Any]:
    """Arguments of :func:`open_dataset_for_cache` that reopen ``ds`` (a live LeRobotDataset) in another process."""
    return dict(repo_id=ds.repo_id, root=str(ds.root), revision=ds.revision, video_backend=ds.video_backend,
                tolerance_s=float(ds.tolerance_s))


def _cli_value(argv: list[str], name: str) -> str | None:
    """Last value of ``--name=value`` / ``--name value`` in ``argv`` (as draccus: the last one wins)."""
    flag, val = f"--{name}", None
    for i, a in enumerate(argv):
        if a.startswith(flag + "="):
            val = a[len(flag) + 1:]
        elif a == flag and i + 1 < len(argv):
            val = argv[i + 1]
    return val


def dataset_source_from_cli(argv: list[str]) -> dict[str, Any]:
    """The dataset ``lerobot_train`` will train on, from its CLI arguments, before ``train()`` parses them.

    ``--config_path=<train_config.json or its directory>`` (resume) gives the defaults; ``--dataset.repo_id``,
    ``--dataset.root``, ``--dataset.revision`` and ``--dataset.video_backend`` override them, as in LeRobot's
    parser. Missing values take ``DatasetConfig``'s defaults. LeRobot 0.4's ``make_dataset`` passes no
    ``tolerance_s``, so it is ``LeRobotDataset``'s default.
    """
    import inspect
    import json
    from pathlib import Path

    src: dict[str, Any] = {"repo_id": None, "root": None, "revision": None, "video_backend": None}
    cfg_path = _cli_value(argv, "config_path")
    if cfg_path:
        p = Path(cfg_path)
        if p.is_dir():
            p = p / "train_config.json"
        if not p.is_file():
            raise SystemExit(f"fastload: --config_path={cfg_path} is not a local train_config.json; pass "
                             "--dataset.repo_id / --dataset.root explicitly to build the cache")
        data = json.loads(p.read_text())
        ds_cfg = data.get("dataset") or {}
        for k in ("repo_id", "root", "revision", "video_backend"):
            src[k] = ds_cfg.get(k)
        if ds_cfg.get("streaming"):
            src["streaming"] = True
    for k in ("repo_id", "root", "revision", "video_backend"):
        v = _cli_value(argv, f"dataset.{k}")
        if v is not None:
            src[k] = None if v.lower() in ("none", "null") else v
    v = _cli_value(argv, "dataset.streaming")
    if v is not None:
        src["streaming"] = v.lower() in ("1", "true", "yes")
    if src.pop("streaming", False):
        raise SystemExit("fastload: --dataset.streaming=true is not supported (there is nothing to cache)")
    if not src["repo_id"]:
        raise SystemExit("fastload: --dataset.repo_id is required to build the cache")
    if isinstance(src["repo_id"], list) or str(src["repo_id"]).startswith("["):
        raise SystemExit("fastload: several datasets (--dataset.repo_id=[...]) are not supported")
    if src["video_backend"] is None:  # DatasetConfig's default
        from lerobot.datasets.video_utils import get_safe_default_codec

        src["video_backend"] = get_safe_default_codec()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src["tolerance_s"] = float(inspect.signature(LeRobotDataset.__init__).parameters["tolerance_s"].default)
    return src
