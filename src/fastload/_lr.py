"""The LeRobot internals fastload touches, for LeRobot 0.6.x (one module per branch).

``LeRobotDataset`` is a front object and all reading happens in ``dataset.reader``
(``lerobot.datasets.dataset_reader.DatasetReader``): ``__getitem__`` -> ``reader.get_item`` ->
``reader._query_videos`` -> ``dataset_reader.decode_video_frames(..., return_uint8=, is_depth=)``.

Changes from 0.5 that fastload follows:

* ``return_uint8`` (``lerobot_train`` builds datasets with ``return_uint8=True``): RGB video frames stay uint8
  (no ``/255``); image transforms run on uint8 in the worker and the training loop divides on the device.
  Parquet image columns are still float32 in [0, 1] (``hf_transform_to_torch`` -> ``ToTensor``).
* Depth cameras (``meta.depth_keys``): 12-bit / float frames, not cached by fastload.
* Language columns (``LANGUAGE_COLUMNS``): passed through by ``get_item`` and dropped again by
  ``lerobot_collate_fn`` (the collate ``lerobot_train`` uses when ``meta.has_language_columns``).
* ``get_item`` no longer adds ``subtask``.
* ``EpisodeAwareSampler`` maps to relative rows itself (``absolute_to_relative_idx``); ``.indices`` is a
  property over relative rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.language import LANGUAGE_COLUMNS

# The module whose global ``decode_video_frames`` the reader calls (wrappers are installed there).
DECODE_MODULE = "lerobot.datasets.dataset_reader"
# Backend decoders that ``video_utils.decode_video_frames`` dispatches to (those present are routed).
BACKEND_DECODERS = ("decode_video_frames_torchcodec", "decode_video_frames_pyav", "decode_video_frames_torchvision")

__all__ = ["DECODE_MODULE", "BACKEND_DECODERS", "LANGUAGE_COLUMNS"]


def reader(ds) -> DatasetReader:
    r = ds._ensure_reader()
    if r.hf_dataset is None:
        r.load_and_activate()
    return r


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


def depth_keys(meta) -> list[str]:
    return list(getattr(meta, "depth_keys", []) or [])


def has_language_columns(meta) -> bool:
    return bool(getattr(meta, "has_language_columns", False))


def language_collate_fn():
    """The collate ``lerobot_train`` uses for datasets with language columns."""
    from lerobot.utils.collate import lerobot_collate_fn

    return lerobot_collate_fn


def subtasks(meta):
    """0.6 ``get_item`` no longer returns ``subtask`` strings."""
    return None


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
