"""order="lerobot" test: same batches, image augmentation and RNG state as LeRobot's DataLoader.

For each case the same seeded "training loop" runs twice, once over LeRobot's DataLoader and once over
FastLoader, both passed through ``accelerator.prepare`` like lerobot-train does. Both DataLoaders are built by
``lerobot_train.make_dataloaders`` itself (LeRobot 0.7: ``EpisodeAwareSampler(seed, absolute_to_relative_idx)``,
``generator=torch.Generator().manual_seed(seed)``, ``persistent_workers`` / ``prefetch_factor`` /
``multiprocessing_context`` from TrainPipelineConfig's defaults when ``num_workers > 0``, ``lerobot_collate_fn``
when the dataset has language columns); on the fastload side with ``fastload.train``'s DataLoader hook installed,
so ``make_dataloaders`` builds a FastLoader. The dataset is built as ``datasets/factory.py`` does
(``return_uint8=True``, ``ImageTransforms`` on the "dataloader" backend). The ``random`` sampler case is a custom
script's ``shuffle=True`` with the same arguments. Both sides then go through the loop's
``_preprocess_dataset_batch`` (uint8 -> float32 / 255). The loop body draws from the process RNGs (like SmolVLA /
GR00T / pi0.5 sampling the flow-matching time on the CPU), so a shifted RNG anywhere would show up. Compared per
batch: every tensor, string and language list, the key order, and after every epoch the torch (CPU and CUDA) /
numpy / python RNG states.

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
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


_RESIZE_HOOK_ENV = "FASTLOAD_TEST_RESIZE_HOOK"
# Facts about the last FastLoader run of run(): per-epoch seconds, the cache filler's counters, ...
LAST_RUN: dict = {}


def _install_resize_hook() -> None:
    """Wrap LeRobot's decode_video_frames with a half-size resize (a data-loader resize as a training script
    would write it). Since LeRobot 0.6 lerobot_train reads with return_uint8=True, so for uint8 input it resizes
    in float and rounds back to uint8, keeping the dtype LeRobot's pipeline expects."""
    import importlib

    import torch.nn.functional as F

    from fastload._lr import DECODE_MODULE

    lds = importlib.import_module(DECODE_MODULE)  # where LeRobot's reader looks up decode_video_frames
    if getattr(lds.decode_video_frames, "_fastload_test_resize", False):
        return
    decode = lds.decode_video_frames

    def decode_resized(*a, **k):
        x = decode(*a, **k)
        size = (x.shape[-2] // 2, x.shape[-1] // 2)
        if x.dtype == torch.uint8:
            y = F.interpolate(x.to(torch.float32), size=size, mode="bilinear", antialias=True)
            return y.round_().clamp_(0, 255).to(torch.uint8)
        return F.interpolate(x, size=size, mode="bilinear", antialias=True)

    decode_resized._fastload_test_resize = True
    lds.decode_video_frames = decode_resized


if os.environ.get(_RESIZE_HOOK_ENV) == "1":  # spawned DataLoader worker of a --decode_hook resize run
    _install_resize_hook()


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


def _transform_configs():
    """``on``: lerobot_train's default augmentation. ``all``: every LeRobot 0.7 transform type, including
    per-sample torchvision ones (JPEGCompression, RandomGrayscale via PerSampleTransform), in random order."""
    from lerobot.transforms import ImageTransformConfig as C
    from lerobot.transforms import ImageTransformsConfig

    tfs = {
        "brightness": C(type="ColorJitter", kwargs={"brightness": (0.8, 1.2)}),
        "jitter": C(type="ColorJitter",
                    kwargs={"contrast": (0.8, 1.2), "saturation": (0.5, 1.5), "hue": (-0.05, 0.05)}),
        "sharpness": C(type="SharpnessJitter", kwargs={"sharpness": (0.5, 1.5)}),
        "affine": C(type="RandomAffine", kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)}),
        "rotation": C(type="RandomRotation", kwargs={"degrees": 5.0}),
        "noise": C(type="GaussianNoise", kwargs={"std": (5.0, 25.0)}),
        "blur": C(type="MotionBlur", kwargs={"kernel_size": (3, 7)}),
        "patches": C(type="GaussianPatchBrightness"),
        "shadow": C(type="RandomShadow"),
        "dropout": C(type="CoarseDropout"),
        "gamma": C(type="GammaCorrection"),
        "planckian": C(type="PlanckianJitter"),
        "jpeg": C(type="JPEGCompression", kwargs={"quality": (30, 90)}),
        "grayscale": C(type="RandomGrayscale", kwargs={"p": 0.5}),
    }
    return {
        "off": None,
        "on": ImageTransformsConfig(enable=True, max_num_transforms=3),
        "all": ImageTransformsConfig(enable=True, max_num_transforms=5, random_order=True, tfs=tfs),
    }


def _train_cfg(args, num_workers: int, persistent: bool):
    """The fields of TrainPipelineConfig that lerobot_train.make_dataloaders reads (its defaults otherwise)."""
    from types import SimpleNamespace

    from lerobot.configs.train import TrainPipelineConfig

    fields = TrainPipelineConfig.__dataclass_fields__
    return SimpleNamespace(
        seed=args.seed, num_workers=num_workers, batch_size=args.batch, resume=False, checkpoint_path=None,
        prefetch_factor=fields["prefetch_factor"].default, persistent_workers=persistent,
        dataloader_multiprocessing_context=fields["dataloader_multiprocessing_context"].default,
        max_eval_samples=0, dataset=SimpleNamespace(streaming=False),
        trainable_config=SimpleNamespace(drop_n_first_frames=0, drop_n_last_frames=2),
    )


def run(kind, args, accelerator, num_workers, sampler_kind, tf_cfg, persistent):
    import lerobot.scripts.lerobot_train as lt
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.distributed import ParallelDims
    from lerobot.transforms import ImageTransforms
    from lerobot.utils.random_utils import set_seed

    from fastload import CachedLeRobotDataset
    from fastload.train import FastLoadConfig, install_hooks

    set_seed(args.seed, accelerator=accelerator)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    fps = meta.fps
    cams = list(meta.camera_keys)
    dt = {"action": [i / fps for i in range(4)], cams[0]: [-1 / fps, 0.0]}
    # as datasets/factory.py (lerobot_train): ImageTransforms on the "dataloader" backend, return_uint8=True
    tf = ImageTransforms(tf_cfg) if tf_cfg is not None else None
    episodes = list(range(args.episodes)) if args.episodes else None
    ds = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=dt, image_transforms=tf, episodes=episodes,
                        return_uint8=True)
    cfg = _train_cfg(args, num_workers, persistent)
    n = accelerator.num_processes
    pdims = ParallelDims(dp_replicate=n, dp_shard=1, ring=1, ulysses=1, world_size=n,
                         device_type=accelerator.device.type)
    undo = None
    mode = getattr(args, "cache_mode", "prebuilt")
    if kind == "fast":
        ds = CachedLeRobotDataset.from_dataset(ds, args.cache_dir, cache_mode=mode)
        fl = FastLoadConfig(cache_dir=args.cache_dir, order="lerobot", placement=args.placement,
                            decode_hook_device=args.hook_device, augment_device=args.augment_device,
                            cache_mode=mode, decode_workers=getattr(args, "decode_workers", None),
                            background_fill=getattr(args, "background_fill", True))
        undo, _ = install_hooks(fl)  # fastload.train's DataLoader dispatch: the DataLoader below is a FastLoader
    try:
        if sampler_kind == "episode_aware":  # lerobot_train 0.7: every map-style dataset
            dl, _ = lt.make_dataloaders(cfg, ds, None, 0, pdims)
        else:  # a custom script's shuffle=True, otherwise with make_dataloaders' arguments
            workers = num_workers > 0
            dl = torch.utils.data.DataLoader(
                ds, num_workers=num_workers, batch_size=args.batch, shuffle=True, sampler=None,
                generator=torch.Generator().manual_seed(cfg.seed), pin_memory=pdims.device_type == "cuda",
                drop_last=False, collate_fn=lt.lerobot_collate_fn if ds.meta.has_language_columns else None,
                prefetch_factor=cfg.prefetch_factor if workers else None,
                persistent_workers=cfg.persistent_workers and workers,
                multiprocessing_context=cfg.dataloader_multiprocessing_context if workers else None)
    finally:
        if undo is not None:
            undo()
    if kind == "fast":
        assert type(dl).__name__ == "FastLoader", type(dl)
    dl = accelerator.prepare(dl)

    batches, states, epoch_s = [], [], []
    max_batches = getattr(args, "max_batches", None) if kind == "fast" else None
    for _ in range(args.epochs):
        t0 = time.perf_counter()
        for b in dl:
            if max_batches is not None and len(batches) >= max_batches:
                break  # an interrupted run (tests/test_cache_modes.py: restart)
            b = dict(b)
            b["_camera_dtypes"] = {cam: str(b[cam].dtype) for cam in cams if cam in b}  # what the loader emitted
            # lerobot_train's loop: uint8 cameras -> float32 / 255 on the device (no rename, identity preprocessor)
            b = lt._preprocess_dataset_batch(b, cams, {}, lambda x: x)
            # clone: a worker's tensors arrive in shared memory, one open file each until freed; keeping them all
            # would exceed the open-file limit (ulimit -n 1024) on larger test datasets
            batches.append({k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in b.items()})
            torch.rand(1)  # a CPU draw in the training step (SmolVLA / GR00T / pi0.5 time sampling)
            np.random.rand()
            random.random()
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
    else:  # stop persistent workers before the next case
        base = getattr(dl, "base_dataloader", dl)
        it = getattr(base, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
            base._iterator = None
    return batches, states, cams


def _key_order(batch: dict) -> list[str]:
    """Key order of a batch, the language columns as one block: ``lerobot_collate_fn`` appends them in the iteration
    order of a set of strings, which depends on the hash seed of the process that collates (a spawned DataLoader
    worker has its own), so their order among themselves is not reproducible even between LeRobot runs."""
    from lerobot.datasets.language import LANGUAGE_COLUMNS

    out: list[str] = []
    for k in batch:
        k = "<language columns>" if k in LANGUAGE_COLUMNS else k
        if not out or out[-1] != k:
            out.append(k)
    return out


def compare(ref, got, cams, atol, max_frac: float | None = None) -> list[str]:
    (rb, rs, _), (gb, gs, _) = ref, got
    errors = []
    if len(rb) != len(gb):
        return [f"batch count {len(rb)} vs {len(gb)}"]
    worst = 0.0
    n_img = n_off = 0  # image elements compared / beyond atol (for max_frac)
    for i, (a, b) in enumerate(zip(rb, gb)):
        if set(a) != set(b):
            errors.append(f"batch {i}: keys differ {sorted(set(a) ^ set(b))}")
            continue
        if _key_order(a) != _key_order(b):
            errors.append(f"batch {i}: key order {list(a)} vs {list(b)}")
        for k in a:
            x, y = a[k], b[k]
            if isinstance(x, torch.Tensor):
                if x.dtype != y.dtype or x.shape != y.shape:
                    errors.append(f"batch {i} {k}: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}")
                else:
                    if k in cams:
                        n_img += x.numel()
                    if torch.equal(x, y):
                        continue
                    diff = (x.double() - y.double()).abs() if x.dtype != torch.bool else None
                    d = diff.max().item() if diff is not None else float("inf")
                    worst = max(worst, d) if k in cams else worst
                    if k in cams and max_frac is not None:
                        n_off += int((diff > atol).sum())
                    elif k not in cams or d > atol:
                        errors.append(f"batch {i} {k}: max abs diff {d}")
            elif x != y:
                errors.append(f"batch {i} {k}: {x!r} vs {y!r}"[:300])
    for e, (a, b) in enumerate(zip(rs, gs)):
        if not _same_states(a, b):
            errors.append(f"epoch {e}: process RNG state differs after the epoch")
    if max_frac is not None and n_img:
        frac = n_off / n_img
        print(f"    images: {n_off} of {n_img} values beyond atol ({frac:.2e}, limit {max_frac:.0e})")
        if frac > max_frac:
            errors.append(f"images: {frac:.2e} of values beyond atol (limit {max_frac:.0e})")
    print(f"    images: max abs diff {worst:.3g}" + (" (GPU kernels vs LeRobot's CPU kernels)" if worst else ""))
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
    p.add_argument("--persistent", default="both", choices=["both", "on", "off"],
                   help="persistent_workers for num_workers > 0 (lerobot_train's default is on)")
    p.add_argument("--samplers", default="random,episode_aware")
    p.add_argument("--transforms", default="off,on,all",
                   help="off | on (lerobot_train's default augmentation) | all (every 0.7 transform type)")
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
    p.add_argument("--cache_mode", choices=["prebuilt", "during_training"], default="prebuilt",
                   help="during_training: every case starts from an empty cache under --fill_dir and must end with a "
                        "cache identical to --cache_dir's (build_cache)")
    p.add_argument("--fill_dir", default=None, help="where the during_training caches go (default: a temp dir)")
    p.add_argument("--decode_workers", type=int, default=2)
    args = p.parse_args()
    args.finish_cache = args.cache_mode == "during_training"
    args.max_frac = None
    if args.video and args.augment_device == "gpu" and args.device != "cpu":
        # LeRobot 0.7 augments uint8 video frames in float and rounds back to uint8 after each transform. GPU
        # kernels differ from the CPU ones in the last float bits: a pixel on a rounding boundary lands one level
        # away, and threshold-like transforms (JPEG compression, ...) can turn that into a larger jump at single
        # pixels. So in this mode images are judged by the fraction of values more than one level apart.
        # augment_device=cpu has no such difference (compared exactly).
        args.atol = max(args.atol, 1 / 255 + 1e-6)
        args.max_frac = 1e-3
        print(f"  (augment_device=gpu on uint8 video: values more than one uint8 level apart must be < {args.max_frac:.0e})")

    import subprocess

    from accelerate import Accelerator

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
        # Spawned DataLoader workers (multiprocessing_context="spawn" since LeRobot 0.6) re-import this module, so
        # the wrapper is installed at import time, like a training script that installs it at module level.
        os.environ[_RESIZE_HOOK_ENV] = "1"
        _install_resize_hook()

    accelerator = Accelerator(cpu=args.device == "cpu")
    tf_cfgs = _transform_configs()
    pers = {"both": (False, True), "on": (True,), "off": (False,)}[args.persistent]
    cases = [(w, s, t, pw) for w in [int(x) for x in args.workers.split(",")]
             for s in args.samplers.split(",") for t in args.transforms.split(",")
             for pw in (pers if w > 0 else (False,))]
    failed = 0
    tag = f"[rank {accelerator.process_index}/{accelerator.num_processes}] " if accelerator.num_processes > 1 else ""
    built_cache = args.cache_dir
    fill_dir = Path(args.fill_dir) if args.fill_dir else (Path(tmp.name) / "fill" if tmp is not None else None)
    if args.cache_mode == "during_training" and fill_dir is None:
        raise SystemExit("--cache_mode during_training with --cache_dir needs --fill_dir")
    for ci, (w, s, t, pw) in enumerate(cases):
        name = (f"num_workers={w}" + (f" persistent={'on' if pw else 'off'}" if w > 0 else "")
                + f" sampler={s} transforms={t}")
        ref = run("lerobot", args, accelerator, w, s, tf_cfgs[t], pw)
        if args.cache_mode == "during_training":
            args.cache_dir = str(fill_dir / f"case{ci}")  # empty: FastLoader fills it during the epochs
        got = run("fast", args, accelerator, w, s, tf_cfgs[t], pw)
        errors = compare(ref, got, ref[2], args.atol, args.max_frac)
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
    print(f"{tag}{'PASS' if not failed else f'FAIL ({failed} of {len(cases)} cases)'}", flush=True)
    if tmp is not None:
        tmp.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
