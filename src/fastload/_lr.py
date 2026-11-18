"""The LeRobot internals fastload touches, for LeRobot 0.7.x (one module per branch).

``LeRobotDataset`` is a front object and all reading happens in ``dataset.reader``
(``lerobot.datasets.dataset_reader.DatasetReader`` for the default ``"lerobot"`` storage format). A DataLoader
calls ``LeRobotDataset.__getitems__`` -> ``reader.get_items`` (``__getitem__`` -> ``get_item`` is the same code
on a batch of one): base rows, delta windows + padding, then ``reader._query_videos`` per item ->
``dataset_reader.decode_video_frames(..., return_uint8=, is_depth=)``, then per item ``apply_rgb_transforms``
(cameras in ``meta.camera_keys`` order), depth units and ``task``.

Changes from 0.6 that fastload follows:

* Video frames are inserted after the tabular columns and the ``*_is_pad`` masks (dict order of items and
  batches); ``task`` stays last.
* ``lerobot_collate_fn`` keeps the language columns (``LANGUAGE_COLUMNS``) as per-sample Python lists instead of
  dropping them; it appends them after the collated tensors, in the iteration order of ``_PYTHON_LIST_KEYS``.
* Image transforms are ``lerobot.transforms.ImageTransforms`` (an ``nn.Module`` running
  ``BatchedImageTransforms`` on a batch of one); see :mod:`fastload.rng`.
* Non-default storage formats (``meta.storage_format``, e.g. ``"lance"``) use another reader and have no
  ``hf_dataset``; fastload refuses them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.language import LANGUAGE_COLUMNS
from lerobot.datasets.storage import DEFAULT_STORAGE_FORMAT

# The module whose global ``decode_video_frames`` the reader calls (wrappers are installed there).
DECODE_MODULE = "lerobot.datasets.dataset_reader"
# Backend decoders that ``video_utils.decode_video_frames`` dispatches to (those present are routed).
BACKEND_DECODERS = ("decode_video_frames_torchcodec", "decode_video_frames_pyav")

__all__ = ["DECODE_MODULE", "BACKEND_DECODERS", "LANGUAGE_COLUMNS", "DEFAULT_STORAGE_FORMAT"]


def reader(ds) -> DatasetReader:
    r = ds._ensure_reader()
    if r.hf_dataset is None:
        r.load_and_activate()
    return r


def storage_format(meta) -> str:
    return str(getattr(meta, "storage_format", DEFAULT_STORAGE_FORMAT) or DEFAULT_STORAGE_FORMAT)


def check_storage_format(meta) -> None:
    fmt = storage_format(meta)
    if fmt != DEFAULT_STORAGE_FORMAT:
        raise NotImplementedError(
            f"fastload: storage_format={fmt!r} is not supported: fastload caches datasets in LeRobot's default "
            f"{DEFAULT_STORAGE_FORMAT!r} layout (parquet + mp4, read by DatasetReader). Train this dataset with "
            "plain lerobot-train."
        )


def delta_indices(ds) -> dict | None:
    return reader(ds).delta_indices


def set_delta_indices(ds, value) -> None:
    reader(ds).delta_indices = value


def absolute_to_relative_idx(ds) -> dict | None:
    return reader(ds)._absolute_to_relative_idx


def video_backend(ds) -> str | None:
    return ds._video_backend


def tolerance_s(ds) -> float:
    return ds.tolerance_s


def return_uint8(ds) -> bool:
    """True if the reader returns RGB video frames as uint8 (``LeRobotDataset(return_uint8=True)``)."""
    return bool(getattr(reader(ds), "_return_uint8", False))


def image_transforms(ds) -> Any:
    return ds.image_transforms


def set_image_transforms(ds, tf) -> None:
    ds.set_image_transforms(tf)


def query_videos(ds, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
    """LeRobot's own decoding of ``query_timestamps`` (episode-relative) for episode ``ep_idx``."""
    return reader(ds)._query_videos(query_timestamps, ep_idx)


def normalize_rgb_frames(frames: torch.Tensor, return_uint8: bool) -> torch.Tensor:
    """What LeRobot's decoders return for raw uint8 RGB frames (uint8, or float32 / 255)."""
    from lerobot.datasets.video_utils import normalize_rgb_frames as f

    return f(frames, return_uint8)


def shift_timestamps(timestamps, offset: float) -> list[float]:
    from lerobot.datasets.utils import shift_timestamps as f

    return f(timestamps, offset)


def depth_keys(meta) -> list[str]:
    return list(getattr(meta, "depth_keys", []) or [])


def has_language_columns(meta) -> bool:
    return bool(getattr(meta, "has_language_columns", False))


def language_collate_fn():
    """The collate ``lerobot_train`` uses for datasets with language columns."""
    from lerobot.utils.collate import lerobot_collate_fn

    return lerobot_collate_fn


def language_batch_keys(columns) -> list[str]:
    """Language columns in the order ``lerobot_collate_fn`` appends them to a batch in this process.

    ``_PYTHON_LIST_KEYS`` is a set of strings: its iteration order depends on the process's hash seed, so batches
    collated in spawned DataLoader workers can list these keys in another order (not reproducible across LeRobot
    runs either). Values are unaffected; only their relative position among themselves can differ."""
    from lerobot.utils import collate

    keys = [k for k in collate._PYTHON_LIST_KEYS if k in columns]
    return keys + [k for k in columns if k not in keys]


def subtasks(meta):
    """``get_item`` returns no ``subtask`` strings (since 0.6)."""
    return None


def item_key_order(hf_columns: list[str], delta_keys: list[str], video_keys: list[str]) -> list[str]:
    """Key order of a 0.7 ``get_items`` item: base columns, ``*_is_pad`` (delta order), video keys, ``task``."""
    order = list(hf_columns) + [f"{k}_is_pad" for k in delta_keys]
    order += [k for k in video_keys if k not in order]
    return order + ["task"]


def sampler_rows(sampler) -> np.ndarray:
    """``sampler.indices`` (dataset rows, unshuffled) as int64, vectorised for ``EpisodeAwareSampler``."""
    starts, cum = getattr(sampler, "_starts", None), getattr(sampler, "_cum_lengths", None)
    if starts is None or cum is None:
        return np.asarray(sampler.indices, dtype=np.int64)
    lengths = np.diff(np.concatenate([[0], np.asarray(cum, dtype=np.int64)]))
    rows = np.concatenate([np.arange(s, s + n, dtype=np.int64) for s, n in zip(starts, lengths)])
    rel = getattr(sampler, "_absolute_to_relative", None)
    if rel is not None:
        rows = np.fromiter((rel[int(i)] for i in rows), dtype=np.int64, count=len(rows))
    return rows


# ---- image transforms (lerobot.transforms, rewritten in 0.7) -------------------------------------------------
def lerobot_image_transforms_types():
    """(ImageTransforms, BatchedRandomSubsetApply, PerSampleTransform, _tag_params)."""
    from lerobot.transforms import transforms as t

    return t.ImageTransforms, t.BatchedRandomSubsetApply, t.PerSampleTransform, t._tag_params


# ---- cache building / filling (build_cache, cache_mode=build_first / during_training) -----------------------
def open_dataset_for_cache(repo_id: str, root: str | None, revision: str | None, video_backend: str | None,
                           tolerance_s: float):
    """The dataset the cache is decoded from: every episode, ``return_uint8=True`` (video frames come back as the
    decoder's raw uint8, no float round trip), no delta timestamps, no transforms."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, revision=revision, video_backend=video_backend,
                          tolerance_s=tolerance_s, return_uint8=True)


def episode_bounds(meta) -> tuple[np.ndarray, np.ndarray]:
    """(dataset_from_index, dataset_to_index) of every episode, indexed by episode index."""
    eps = meta.episodes
    return (np.asarray(eps["dataset_from_index"], dtype=np.int64),
            np.asarray(eps["dataset_to_index"], dtype=np.int64))


def read_frames(ds, key: str, ep_idx: int, start: int, stop: int, timestamps: np.ndarray) -> torch.Tensor:
    """Frames ``start:stop`` (absolute indices, inside episode ``ep_idx``) of camera ``key`` as LeRobot reads them
    (``ds`` from :func:`open_dataset_for_cache`): video through the reader's own ``_query_videos`` at the frames'
    timestamps (uint8), parquet images as stored (float32 / 255). Returns ``[stop - start, C, H, W]``."""
    if key in ds.meta.video_keys:
        frames = query_videos(ds, {key: [float(t) for t in timestamps]}, ep_idx)[key]
        if frames.ndim == 3:  # LeRobot squeezes a single-timestamp query
            frames = frames.unsqueeze(0)
        return frames
    return torch.stack(ds.hf_dataset[start:stop][key])  # all episodes loaded: row == absolute index


def dataset_source(ds) -> dict[str, Any]:
    """Arguments of :func:`open_dataset_for_cache` that reopen ``ds`` (a live LeRobotDataset) in another process."""
    return dict(repo_id=ds.repo_id, root=str(ds.root), revision=ds.revision, video_backend=video_backend(ds),
                tolerance_s=float(tolerance_s(ds)))


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
    ``--dataset.root``, ``--dataset.revision``, ``--dataset.video_backend`` and ``--tolerance_s`` override them,
    as in LeRobot's parser. Missing values take ``DatasetConfig`` / ``TrainPipelineConfig``'s defaults.
    """
    import json
    from pathlib import Path

    src: dict[str, Any] = {"repo_id": None, "root": None, "revision": None, "video_backend": None,
                           "tolerance_s": None}
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
        src["tolerance_s"] = data.get("tolerance_s")
        if ds_cfg.get("streaming"):
            src["streaming"] = True
    for k in ("repo_id", "root", "revision", "video_backend"):
        v = _cli_value(argv, f"dataset.{k}")
        if v is not None:
            src[k] = None if v.lower() in ("none", "null") else v
    v = _cli_value(argv, "tolerance_s")
    if v is not None:
        src["tolerance_s"] = float(v)
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
        from lerobot.utils.import_utils import get_safe_default_video_backend

        src["video_backend"] = get_safe_default_video_backend()
    if src["tolerance_s"] is None:  # TrainPipelineConfig's default
        from lerobot.configs.train import TrainPipelineConfig

        src["tolerance_s"] = TrainPipelineConfig.__dataclass_fields__["tolerance_s"].default
    src["tolerance_s"] = float(src["tolerance_s"])
    return src
