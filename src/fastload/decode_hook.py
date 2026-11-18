"""Post-decode hooks: user code that wraps LeRobot's ``decode_video_frames``.

Some training scripts change frames right after video decoding by wrapping
``lerobot.datasets.dataset_reader.decode_video_frames`` (the name LeRobot's reader calls), e.g. resizing
480x640 to 240x320 in the data loader workers::

    decode = dataset_reader.decode_video_frames
    dataset_reader.decode_video_frames = lambda *a, **k: resize(decode(*a, **k))

fastload never decodes video, so such a wrapper would silently stop running. Instead fastload calls the
wrapper itself, with the same arguments as LeRobot's reader (``video_path, timestamps, tolerance_s, backend,
return_uint8=..., is_depth=False``), while LeRobot's backend decoders temporarily answer with the cached
frames: the wrapper sees exactly what LeRobot's decoder would have returned (``[T, C, H, W]``; uint8 when
``return_uint8`` - as in ``lerobot_train`` since LeRobot 0.6 - else float32 in [0, 1]) and its own code runs
unchanged, at the same point of the pipeline (before ``image_transforms``).
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from typing import Callable

import torch

_tls = threading.local()
_installed = False


def _vu():
    import lerobot.datasets.video_utils as vu

    return vu


def _as_decoded(frames: torch.Tensor, return_uint8: bool) -> torch.Tensor:
    """Cached uint8 frames as the backend decoder returns them (uint8, or float32 / 255)."""
    if return_uint8:
        return frames
    div = torch.tensor(255.0, dtype=torch.float32, device=frames.device) if frames.is_cuda else 255
    return frames.to(torch.float32) / div


def _install() -> None:
    """Route LeRobot's backend decoders through a thread-local override (once per process)."""
    global _installed
    if _installed:
        return
    from fastload._lr import BACKEND_DECODERS

    vu = _vu()
    for name in BACKEND_DECODERS:  # LeRobot 0.6: torchcodec and pyav (torchvision was removed)
        orig = getattr(vu, name, None)
        if orig is None:
            continue
        sig = inspect.signature(orig)

        def routed(*args, _orig=orig, _sig=sig, **kwargs):
            frames = getattr(_tls, "frames", None)
            if frames is None:
                return _orig(*args, **kwargs)
            bound = _sig.bind_partial(*args, **kwargs).arguments
            _tls.consumed = True
            return _as_decoded(frames, bool(bound.get("return_uint8", False)))

        routed.__wrapped__ = orig  # type: ignore[attr-defined]
        setattr(vu, name, routed)
    _installed = True


def active_hook() -> Callable | None:
    """The wrapper installed around the decode_video_frames that LeRobot's reader calls, or None."""
    import importlib

    from fastload._lr import DECODE_MODULE

    fn = importlib.import_module(DECODE_MODULE).decode_video_frames
    return None if fn is _vu().decode_video_frames else fn


def hook_name(hook: Callable | None) -> str | None:
    if hook is None:
        return None
    return f"{getattr(hook, '__module__', '?')}.{getattr(hook, '__qualname__', repr(hook))}"


def apply_hook(hook: Callable, frames: torch.Tensor, video_path: str | Path, timestamps: list[float],
               tolerance_s: float, backend: str | None, return_uint8: bool) -> torch.Tensor:
    """Run ``hook`` as LeRobot's ``_query_videos`` would, with ``frames`` (cached uint8 ``[T, C, H, W]``) as the
    decoder's raw output."""
    if frames.dtype != torch.uint8:
        raise TypeError(f"fastload: apply_hook expects the cached uint8 frames, got {frames.dtype}")
    _install()
    _tls.frames, _tls.consumed = frames, False
    try:
        out = hook(video_path, timestamps, tolerance_s, backend, return_uint8=return_uint8, is_depth=False)
    finally:
        consumed = _tls.consumed
        _tls.frames = None
    if not consumed:
        raise RuntimeError(
            f"fastload: the decode_video_frames wrapper {hook_name(hook)} did not call LeRobot's decoder, so it "
            "cannot be applied to cached frames"
        )
    return out


def remove_hook() -> None:
    """Undo a wrapper around the decode_video_frames LeRobot's reader calls, in this process only. For processes
    that only decode into the cache (build_cache / cache-filling workers): the cache stores the decoder's own frames
    and the wrapper is applied at training time. A spawned worker re-imports the training script, which may install
    the wrapper at import time."""
    import importlib

    from fastload._lr import DECODE_MODULE

    mod = importlib.import_module(DECODE_MODULE)
    if mod.decode_video_frames is not _vu().decode_video_frames:
        mod.decode_video_frames = _vu().decode_video_frames
