"""order="lerobot" test: same batches, image augmentation and RNG state as LeRobot's DataLoader.

For each case the same seeded "training loop" runs twice, once over LeRobot's DataLoader and once over
FastLoader, both passed through ``accelerator.prepare`` like lerobot-train does. The loop body draws
from the process RNG (like SmolVLA / GR00T sampling the flow-matching time on the CPU), so a shifted
RNG anywhere would show up. Compared per batch: every tensor and string, and after every epoch the
torch (CPU and CUDA) / numpy / python RNG states.

    python tests/test_lerobot_order.py                       # synthetic image dataset (CPU is enough)
    python tests/test_lerobot_order.py --repo_id R --root D --cache_dir C [--device cuda]

On CUDA the transform kernels run on the GPU, so augmented images may differ from LeRobot's CPU
result by float rounding (reported as max abs diff, accepted below --atol); everything else must match
exactly. Multi-process sharding: run the same script under ``accelerate launch --num_processes 2``.

``--cache_mode during_training``: every case starts FastLoader on an EMPTY cache (under ``--fill_dir``), which it
fills while serving the epochs; afterwards the cache must be complete and every file equal to the one
``fastload.build_cache`` made (``--cache_dir``).
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Facts about the last FastLoader run of run(): per-epoch seconds, the cache filler's counters, ...
LAST_RUN: dict = {}

def make_synthetic(root: Path, episodes: int = 6, fps: int = 10, videos: bool = False) -> str:
    """Tiny LeRobot dataset with two cameras (PNG images, or mp4 videos with ``videos=True``)."""
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
    return repo_id


def _states():
    cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.zeros(0, dtype=torch.uint8)
    return torch.get_rng_state(), np.random.get_state(), random.getstate(), cuda


def _same_states(a, b) -> bool:
    return (torch.equal(a[0], b[0]) and a[2] == b[2] and torch.equal(a[3], b[3])
            and all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a[1], b[1])))


def _transform_configs():
    """``off``: no augmentation; ``on``: LeRobot's default augmentation (3 random transforms)."""
    from lerobot.datasets.transforms import ImageTransformsConfig

    return {"off": None, "on": ImageTransformsConfig(enable=True, max_num_transforms=3)}


def run(kind, args, accelerator, num_workers, sampler_kind, tf_cfg):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.datasets.transforms import ImageTransforms
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
    ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=dt, image_transforms=tf, episodes=episodes)
    sampler = None
    if sampler_kind == "episode_aware":
        sampler = EpisodeAwareSampler(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
                                      episode_indices_to_use=ds.episodes, drop_n_last_frames=2, shuffle=True)
    common = dict(batch_size=args.batch, shuffle=sampler is None, sampler=sampler, drop_last=False)
    if kind == "lerobot":
        dl = torch.utils.data.DataLoader(ds, num_workers=num_workers, pin_memory=False,
                                         prefetch_factor=2 if num_workers > 0 else None, **common)
    else:
        mode = getattr(args, "cache_mode", "prebuilt")
        ds = CachedLeRobotDataset.from_dataset(ds, args.cache_dir, cache_mode=mode)
        dl = FastLoader(ds, order="lerobot", num_workers=num_workers, placement=args.placement,
                        device=accelerator.device, decode_hook_device=args.hook_device,
                        augment_device=args.augment_device, cache_mode=mode,
                        decode_workers=getattr(args, "decode_workers", None),
                        background_fill=getattr(args, "background_fill", True), **common)
    dl = accelerator.prepare(dl)

    batches, states, epoch_s = [], [], []
    max_batches = getattr(args, "max_batches", None) if kind == "fast" else None
    for _ in range(args.epochs):
        t0 = time.perf_counter()
        for b in dl:
            if max_batches is not None and len(batches) >= max_batches:
                break  # an interrupted run (tests/test_cache_modes.py: restart)
            batches.append({k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in b.items()})
            torch.rand(1)  # a CPU draw in the training step (SmolVLA / GR00T time sampling)
            if accelerator.device.type == "cuda":
                torch.randn(1, device=accelerator.device)  # a GPU draw (pi0.5 / diffusion noise)
        epoch_s.append(time.perf_counter() - t0)
        states.append(_states())
        if max_batches is not None and len(batches) >= max_batches:
            break
    if kind == "fast":
        LAST_RUN.clear()
        LAST_RUN.update(epoch_s=epoch_s, complete_after_epochs=dl.cache_complete, fill_stats=dl.fill_stats)
        if getattr(args, "finish_cache", False):
            LAST_RUN["finished"] = dl.finish_cache()
            LAST_RUN["fill_stats_final"] = dl.fill_stats  # including what finish_cache decoded
        dl.close()
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


def compare_caches(got_dir, ref_dir) -> list[str]:
    """``got_dir`` must be a complete cache whose frame arrays, timestamps and metadata equal ``ref_dir``'s."""
    import json

    from fastload.cache import COMPLETE_FILE, FILLED_FILE, META_FILE, TIMESTAMPS_FILE

    got_dir, ref_dir = Path(got_dir), Path(ref_dir)
    errors = []
    if not (got_dir / COMPLETE_FILE).exists():
        errors.append(f"{got_dir}: no {COMPLETE_FILE}")
    gm, rm = (json.loads((d / META_FILE).read_text()) for d in (got_dir, ref_dir))
    for f in ("format_version", "info_fingerprint", "total_frames", "total_episodes", "keys", "video_backend"):
        if gm.get(f) != rm.get(f):
            errors.append(f"meta {f}: {gm.get(f)!r} vs {rm.get(f)!r}")
    files = [v["file"] for v in rm["keys"].values()] + [TIMESTAMPS_FILE, FILLED_FILE]
    for name in files:
        a, b = np.load(got_dir / name, mmap_mode="r"), np.load(ref_dir / name, mmap_mode="r")
        if a.dtype != b.dtype or a.shape != b.shape:
            errors.append(f"{name}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        elif not np.array_equal(a, b):
            bad = np.unique(np.nonzero(np.asarray(a) != np.asarray(b))[0])
            errors.append(f"{name}: {len(bad)} frames differ (first {bad[:5].tolist()})")
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
    p.add_argument("--samplers", default="random,episode_aware")
    p.add_argument("--transforms", default="off,on", help="off | on (LeRobot's default augmentation)")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--episodes", type=int, default=0, help="use only the first N episodes (0 = all)")
    p.add_argument("--video", action="store_true", help="synthetic dataset with mp4 videos instead of images")
    p.add_argument("--decode_hook", choices=["none", "resize"], default="none",
                   help="wrap LeRobot's decode_video_frames with a half-size bilinear antialiased resize "
                        "(like a data-loader resize in a training script)")
    p.add_argument("--hook_device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--augment_device", choices=["cpu", "gpu"], default="gpu",
                   help="cpu: augmented images must match LeRobot bit for bit (use with --atol 0)")
    p.add_argument("--cache_mode", choices=["prebuilt", "during_training"], default="prebuilt",
                   help="during_training: every case starts from an empty cache under --fill_dir and must end with a "
                        "cache identical to --cache_dir's (build_cache)")
    p.add_argument("--fill_dir", default=None, help="where the during_training caches go (default: a temp dir)")
    p.add_argument("--decode_workers", type=int, default=2)
    args = p.parse_args()
    args.finish_cache = args.cache_mode == "during_training"

    import subprocess

    from accelerate import Accelerator

    tmp = None
    if args.repo_id is None:
        tmp = tempfile.TemporaryDirectory()
        args.root = str(Path(tmp.name) / "ds")
        args.repo_id = make_synthetic(Path(args.root), videos=args.video)
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
            x = decode(*a, **k)
            return F.interpolate(x, size=(x.shape[-2] // 2, x.shape[-1] // 2), mode="bilinear", antialias=True)

        lds.decode_video_frames = decode_resized

    accelerator = Accelerator(cpu=args.device == "cpu")
    tf_cfgs = _transform_configs()
    cases = [(w, s, t) for w in [int(x) for x in args.workers.split(",")]
             for s in args.samplers.split(",") for t in args.transforms.split(",")]
    failed = 0
    tag = f"[rank {accelerator.process_index}/{accelerator.num_processes}] " if accelerator.num_processes > 1 else ""
    built_cache = args.cache_dir
    fill_dir = Path(args.fill_dir) if args.fill_dir else (Path(tmp.name) / "fill" if tmp is not None else None)
    if args.cache_mode == "during_training" and fill_dir is None:
        raise SystemExit("--cache_mode during_training with --cache_dir needs --fill_dir")
    for ci, (w, s, t) in enumerate(cases):
        name = f"num_workers={w} sampler={s} transforms={t}"
        ref = run("lerobot", args, accelerator, w, s, tf_cfgs[t])
        if args.cache_mode == "during_training":
            args.cache_dir = str(fill_dir / f"case{ci}")  # empty: FastLoader fills it during the epochs
        got = run("fast", args, accelerator, w, s, tf_cfgs[t])
        errors = compare(ref, got, ref[2], args.atol)
        if args.cache_mode == "during_training":
            st = LAST_RUN.get("fill_stats") or {}
            print(f"    {tag}epoch times {', '.join(f'{x:.2f}s' for x in LAST_RUN['epoch_s'])}; complete after the "
                  f"epochs: {LAST_RUN['complete_after_epochs']}; decoded {st.get('frames')} frames in "
                  f"{st.get('tasks')} tasks ({st.get('background_frames')} in the background), waited "
                  f"{st.get('wait_s', 0):.2f}s for frames")
            errors += compare_caches(args.cache_dir, built_cache)
            args.cache_dir = built_cache
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
