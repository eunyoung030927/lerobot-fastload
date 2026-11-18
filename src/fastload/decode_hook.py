"""Post-decode hooks: user code that wraps LeRobot's ``decode_video_frames``.

Some training scripts change frames right after video decoding by wrapping
``lerobot.datasets.lerobot_dataset.decode_video_frames``, e.g. resizing 480x640 to 240x320 in the data
loader workers::

    decode = lerobot_dataset.decode_video_frames
    lerobot_dataset.decode_video_frames = lambda *a, **k: resize(decode(*a, **k))

fastload never decodes video, so such a wrapper would silently stop running. Instead fastload calls the
wrapper itself with LeRobot's backend decoders temporarily answering with the cached frames: the wrapper
sees exactly what LeRobot's decoder would have returned (float32 in [0, 1], ``[T, C, H, W]``) and its own
code runs unchanged, at the same point of the pipeline (before ``image_transforms``).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import torch

_tls = threading.local()
_installed = False


def _vu():
    import lerobot.datasets.video_utils as vu

    return vu


def _install() -> None:
    """Route LeRobot's backend decoders through a thread-local override (once per process)."""
    global _installed
    if _installed:
        return
    vu = _vu()
    for name in ("decode_video_frames_torchcodec", "decode_video_frames_torchvision"):
        orig = getattr(vu, name)

        def routed(*args, _orig=orig, **kwargs):
            frames = getattr(_tls, "frames", None)
            if frames is None:
                return _orig(*args, **kwargs)
            _tls.consumed = True
            return frames

        routed.__wrapped__ = orig  # type: ignore[attr-defined]
        setattr(vu, name, routed)
    _installed = True


def active_hook() -> Callable | None:
    """The wrapper installed around ``lerobot_dataset.decode_video_frames``, or None."""
    import lerobot.datasets.lerobot_dataset as lds

    fn = lds.decode_video_frames
    return None if fn is _vu().decode_video_frames else fn


def hook_name(hook: Callable | None) -> str | None:
    if hook is None:
        return None
    return f"{getattr(hook, '__module__', '?')}.{getattr(hook, '__qualname__', repr(hook))}"


def apply_hook(hook: Callable, frames: torch.Tensor, video_path: str | Path, timestamps: list[float],
               tolerance_s: float, backend: str | None) -> torch.Tensor:
    """Run ``hook`` as LeRobot's ``_query_videos`` would, with ``frames`` as the decoder's output."""
    _install()
    _tls.frames, _tls.consumed = frames, False
    try:
        out = hook(video_path, timestamps, tolerance_s, backend)
    finally:
        consumed = _tls.consumed
        _tls.frames = None
    if not consumed:
        raise RuntimeError(
            f"fastload: the decode_video_frames wrapper {hook_name(hook)} did not call LeRobot's decoder, so it "
            "cannot be applied to cached frames"
        )
    return out
