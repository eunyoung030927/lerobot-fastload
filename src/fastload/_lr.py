"""The LeRobot internals fastload touches, for LeRobot 0.5.x (one module per branch).

From 0.5 on, ``LeRobotDataset`` is a front object and all reading happens in ``dataset.reader``
(``lerobot.datasets.dataset_reader.DatasetReader``): ``__getitem__`` -> ``reader.get_item`` ->
``reader._query_videos`` -> ``dataset_reader.decode_video_frames``.
"""

from __future__ import annotations

from typing import Any

from lerobot.datasets.dataset_reader import DatasetReader

# The module whose global ``decode_video_frames`` the reader calls (wrappers are installed there).
DECODE_MODULE = "lerobot.datasets.dataset_reader"


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


def image_transforms(ds) -> Any:
    return ds.image_transforms


def set_image_transforms(ds, tf) -> None:
    ds.set_image_transforms(tf)


def query_videos(ds, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
    """LeRobot's own decoding of ``query_timestamps`` (episode-relative) for episode ``ep_idx``."""
    return reader(ds)._query_videos(query_timestamps, ep_idx)


def subtasks(meta):
    return getattr(meta, "subtasks", None)
