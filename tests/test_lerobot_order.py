"""order="lerobot" test: same batches, image augmentation and RNG state as LeRobot's DataLoader.

For each case the same seeded "training loop" runs twice, once over LeRobot's DataLoader and once over
FastLoader, both passed through ``accelerator.prepare`` like lerobot-train does. The reference DataLoader is
built as in ``lerobot_train`` 0.6.1: dataset with ``return_uint8=True``, ``EpisodeAwareSampler(seed=...,
absolute_to_relative_idx=...)``, ``persistent_workers`` / ``prefetch_factor=4`` / ``multiprocessing_context=
"spawn"`` when ``num_workers > 0``, and ``lerobot_collate_fn`` when the dataset has language columns (the
``random`` sampler case is a custom script's ``shuffle=True``). Both sides then get the loop's
``uint8 -> float32 / 255`` camera conversion. The loop body draws from the process RNG (like SmolVLA / GR00T
sampling the flow-matching time on the CPU), so a shifted RNG anywhere would show up. Compared per batch:
every tensor and string, and after every epoch the torch (CPU and CUDA) / numpy / python RNG states.

    python tests/test_lerobot_order.py                       # synthetic image dataset (CPU is enough)
    python tests/test_lerobot_order.py --repo_id R --root D --cache_dir C [--device cuda]

On CUDA the transform kernels run on the GPU, so augmented images may differ from LeRobot's CPU
result by float rounding (reported as max abs diff, accepted below --atol); everything else must match
exactly. Multi-process sharding: run the same script under ``accelerate launch --num_processes 2``.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def make_synthetic(root: Path, episodes: int = 6, fps: int = 10, videos: bool = False, language: bool = False) -> str:
    """Tiny LeRobot dataset with two cameras (PNG images, or mp4 videos with ``videos=True``).

    ``language=True`` adds LeRobot 0.6's ``language_persistent`` / ``language_events`` columns (the way the
    annotation pipeline writes them), so ``meta.has_language_columns`` is True.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "fastload/synthetic"
    dtype = "video" if videos else "image"
    shapes = {"observation.image": (64, 96, 3), "observation.wrist": (64, 64, 3)} if videos else \
        {"observation.image": (24, 32, 3), "observation.wrist": (16, 16, 3)}
    features = {
        **{k: {"dtype": dtype, "shape": s, "names": ["height", "width", "channel"]} for k, s in shapes.items()},
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id, fps=fps, features=features, root=root, use_videos=videos)
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        for _ in range(int(rng.integers(15, 30))):
            ds.add_frame({
                **{k: rng.integers(0, 256, s, dtype=np.uint8) for k, s in shapes.items()},
                "observation.state": rng.normal(size=3).astype(np.float32),
                "action": rng.normal(size=2).astype(np.float32),
                "task": f"task {ep % 2}",
            })
        ds.save_episode()
    if hasattr(ds, "finalize"):
        ds.finalize()
    if language:
        _add_language_columns(Path(root))
    return repo_id


def _add_language_columns(root: Path) -> None:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT, language_feature_info

    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        ts = table.column("timestamp").to_pylist()
        ep = table.column("episode_index").to_pylist()
        persistent = [[{"role": "assistant", "content": f"subtask {e}.{int(t * 10) // 5}", "style": "subtask",
                        "timestamp": float(t), "camera": None, "tool_calls": None}] for t, e in zip(ts, ep)]
        events = [[{"role": "user", "content": f"event {e}", "style": "interjection", "camera": None,
                    "tool_calls": None}] if i % 7 == 0 else [] for i, (t, e) in enumerate(zip(ts, ep))]
        table = table.append_column(LANGUAGE_PERSISTENT, pa.array(persistent))
        table = table.append_column(LANGUAGE_EVENTS, pa.array(events))
        tmp = path.with_suffix(".tmp")
        write_table_one_row_group_per_episode(table, tmp)
        tmp.replace(path)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].update({k: {**v, "shape": list(v["shape"])} for k, v in language_feature_info().items()})
    info_path.write_text(json.dumps(info, indent=4))


def _states():
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.zeros(0, dtype=torch.uint8)
    return torch.get_rng_state(), np.random.get_state(), random.getstate(), cuda


def _same_states(a, b) -> bool:
    return (torch.equal(a[0], b[0]) and a[2] == b[2] and torch.equal(a[3], b[3])
            and all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a[1], b[1])))


def run(kind, args, accelerator, num_workers, sampler_kind, tf_cfg, persistent):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.transforms import ImageTransforms
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.random_utils import set_seed

    from fastload import CachedLeRobotDataset, FastLoader

    set_seed(args.seed, accelerator=accelerator)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = meta.fps
    cams = list(meta.camera_keys)
    dt = {"action": [i / fps for i in range(4)], cams[0]: [-1 / fps, 0.0]}
    tf = ImageTransforms(tf_cfg) if tf_cfg is not None else None
    episodes = list(range(args.episodes)) if args.episodes else None
    # as datasets/factory.py make_dataset (lerobot_train): return_uint8=True
    ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=dt, image_transforms=tf, episodes=episodes,
                        return_uint8=True)
    sampler = None
    if sampler_kind == "episode_aware":  # as lerobot_train 0.6.1 (every map-style dataset)
        sampler = EpisodeAwareSampler(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
                                      episode_indices_to_use=ds.episodes, drop_n_last_frames=2, shuffle=True,
                                      seed=args.seed, absolute_to_relative_idx=ds.absolute_to_relative_idx)
    collate_fn = lerobot_collate_fn if ds.meta.has_language_columns else None
    common = dict(batch_size=args.batch, shuffle=sampler is None, sampler=sampler, drop_last=False,
                  collate_fn=collate_fn)
    if kind == "lerobot":
        workers_enabled = num_workers > 0  # lerobot_train._dataloader_worker_kwargs
        dl = torch.utils.data.DataLoader(
            ds, num_workers=num_workers, pin_memory=accelerator.device.type == "cuda",
            prefetch_factor=4 if workers_enabled else None, persistent_workers=persistent and workers_enabled,
            multiprocessing_context="spawn" if workers_enabled else None, **common)
    else:
        ds = CachedLeRobotDataset.from_dataset(ds, args.cache_dir)
        dl = FastLoader(ds, order="lerobot", num_workers=num_workers, persistent_workers=persistent,
                        placement=args.placement, device=accelerator.device, decode_hook_device=args.hook_device,
                        augment_device=args.augment_device, **common)
    dl = accelerator.prepare(dl)

    batches, states = [], []
    for _ in range(args.epochs):
        for b in dl:
            b = dict(b)
            b["_camera_dtypes"] = {cam: str(b[cam].dtype) for cam in cams if cam in b}  # what the loader emitted
            for cam in cams:  # lerobot_train's loop: uint8 cameras -> float32 / 255 on the device
                if cam in b and b[cam].dtype == torch.uint8:
                    b[cam] = b[cam].to(dtype=torch.float32) / 255.0
            batches.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in b.items()})
            torch.rand(1)  # a CPU draw in the training step (SmolVLA / GR00T time sampling)
            if accelerator.device.type == "cuda":
                torch.randn(1, device=accelerator.device)  # a GPU draw (pi0.5 / diffusion noise)
        states.append(_states())
    if kind == "fast":
        dl.close()
    else:  # stop persistent workers before the next case
        base = getattr(dl, "base_dataloader", dl)
        it = getattr(base, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
            base._iterator = None
    return batches, states, cams


def compare(ref, got, cams, atol) -> list[str]:
    (rb, rs, _), (gb, gs, _) = ref, got
    errors = []
    if len(rb) != len(gb):
        return [f"batch count {len(rb)} vs {len(gb)}"]
    worst = 0.0
    for i, (a, b) in enumerate(zip(rb, gb)):
        if set(a) != set(b):
            errors.append(f"batch {i}: keys differ {sorted(set(a) ^ set(b))}")
            continue
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, torch.Tensor):
                if x.dtype != y.dtype or x.shape != y.shape:
                    errors.append(f"batch {i} {k}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}")
                elif not torch.equal(x, y):
                    d = (x.double() - y.double()).abs().max().item() if x.dtype != torch.bool else float("inf")
                    worst = max(worst, d) if k in cams else worst
                    if k not in cams or d > atol:
                        errors.append(f"batch {i} {k}: max abs diff {d}")
            elif x != y:
                errors.append(f"batch {i} {k}: {x!r} vs {y!r}")
    for e, (a, b) in enumerate(zip(rs, gs)):
        if not _same_states(a, b):
            errors.append(f"epoch {e}: process RNG state differs after the epoch")
    if worst:
        print(f"    images: max abs diff {worst:.3g} (GPU kernels vs LeRobot's CPU kernels)")
    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id")
    p.add_argument("--root")
    p.add_argument("--cache_dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--placement", default="host")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--workers", default="0,2")
    p.add_argument("--persistent", default="both", choices=["both", "on", "off"],
                   help="persistent_workers for num_workers > 0 (lerobot_train's default is on)")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--episodes", type=int, default=0, help="use only the first N episodes (0 = all)")
    p.add_argument("--video", action="store_true", help="synthetic dataset with mp4 videos instead of images")
    p.add_argument("--language", action="store_true",
                   help="synthetic dataset with language columns (lerobot_collate_fn, as lerobot_train)")
    p.add_argument("--decode_hook", choices=["none", "resize"], default="none",
                   help="wrap LeRobot's decode_video_frames with a half-size bilinear antialiased resize "
                        "(like a data-loader resize in a training script)")
    p.add_argument("--hook_device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--augment_device", choices=["cpu", "gpu"], default="gpu",
                   help="cpu: augmented images must match LeRobot bit for bit (use with --atol 0)")
    args = p.parse_args()

    import subprocess

    from accelerate import Accelerator
    from lerobot.transforms import ImageTransformsConfig

    tmp = None
    if args.repo_id is None:
        tmp = tempfile.TemporaryDirectory()
        args.root = str(Path(tmp.name) / "ds")
        args.repo_id = make_synthetic(Path(args.root), videos=args.video, language=args.language)
        args.cache_dir = str(Path(tmp.name) / "cache")
        # In a separate process: decoding videos here would leave open decoders that DataLoader workers
        # inherit (LeRobot's global decoder cache), which breaks LeRobot's own worker decoding.
        subprocess.check_call([sys.executable, "-m", "fastload.build_cache", "--repo_id", args.repo_id, "--root",
                               args.root, "--cache_dir", args.cache_dir, "--num_workers", "1", "--verify", "8"])
    if args.decode_hook == "resize":
        import importlib

        import torch.nn.functional as F

        from fastload._lr import DECODE_MODULE

        lds = importlib.import_module(DECODE_MODULE)  # where LeRobot's reader looks up decode_video_frames
        decode = lds.decode_video_frames

        def decode_resized(*a, **k):
            # A data-loader resize as a training script would write it. lerobot_train reads with return_uint8=True
            # since LeRobot 0.6, so the decoder returns uint8: resize in float and round back to uint8 to keep the
            # dtype LeRobot's pipeline (transforms on uint8, loop-side / 255) expects.
            x = decode(*a, **k)
            size = (x.shape[-2] // 2, x.shape[-1] // 2)
            if x.dtype == torch.uint8:
                y = F.interpolate(x.to(torch.float32), size=size, mode="bilinear", antialias=True)
                return y.round_().clamp_(0, 255).to(torch.uint8)
            return F.interpolate(x, size=size, mode="bilinear", antialias=True)

        lds.decode_video_frames = decode_resized

    accelerator = Accelerator(cpu=args.device == "cpu")
    tf_on = ImageTransformsConfig(enable=True, max_num_transforms=3)
    pers = {"both": (False, True), "on": (True,), "off": (False,)}[args.persistent]
    cases = [(w, s, t, pw) for w in [int(x) for x in args.workers.split(",")]
             for s in ("random", "episode_aware") for t in (None, tf_on) for pw in (pers if w > 0 else (False,))]
    failed = 0
    for w, s, t, pw in cases:
        name = (f"num_workers={w}" + (f" persistent={'on' if pw else 'off'}" if w > 0 else "")
                + f" sampler={s} transforms={'on' if t else 'off'}")
        ref = run("lerobot", args, accelerator, w, s, t, pw)
        got = run("fast", args, accelerator, w, s, t, pw)
        errors = compare(ref, got, ref[2], args.atol)
        tag = f"[rank {accelerator.process_index}/{accelerator.num_processes}] " if accelerator.num_processes > 1 else ""
        print(f"  {tag}{name}: {'OK' if not errors else 'FAIL'} ({len(ref[0])} batches)", flush=True)
        for e in errors[:8]:
            print(f"    {e}")
        failed += bool(errors)
    print(f"{tag}{'PASS' if not failed else f'FAIL ({failed} cases)'}", flush=True)
    if tmp is not None:
        tmp.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
