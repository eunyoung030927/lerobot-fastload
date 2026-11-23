"""Cache modes: ``--fastload.cache_mode`` prebuilt / build_first / during_training and ``min_free_disk_gb``.

Self-contained (synthetic datasets from ``tests/test_lerobot_order.make_synthetic``, PNG images and mp4 videos; CPU
is enough)::

    python tests/test_cache_modes.py                      # every test, both variants
    python tests/test_cache_modes.py --tests during_training,restart --variants video

Tests:

* ``during_training``: ``tests/test_lerobot_order.py --cache_mode during_training``: every case starts FastLoader on an
  EMPTY cache and runs 2 epochs next to LeRobot's DataLoader (num_workers 0 and 2, random and episode-aware sampler,
  transforms off / on, ``augment_device=cpu`` with atol 0): batches, key order, dtypes and per-epoch RNG states must be
  identical, and the filled cache must be COMPLETE and equal, file by file, to the cache ``build_cache`` made.
* ``restart`` (30 episodes): a run is stopped after a few batches (background filling off, so the cache stays
  partial); a second run on the same cache must decode exactly the frames still missing (the filled ones are
  reused), match LeRobot, and end with a cache equal to ``build_cache``'s.
* ``build_first``: ``python -m fastload.train`` (ACT, CPU, 6 steps) with ``--fastload.cache_mode=build_first`` on a
  missing cache: the cache is built before training (log lines), training runs, the cache equals ``build_cache``'s.
* ``min_free_disk``: a huge ``min_free_disk_gb`` makes ``build_cache``, ``build_first`` and ``during_training``
  (``fastload.train`` and ``make_loader``) refuse with the disk-space error and create no cache files.
* ``prebuilt_missing``: ``fastload.train`` / ``CachedLeRobotDataset.from_dataset`` without a cache: the error names
  ``build_first``, ``during_training`` and the ``build_cache`` command.
* ``multiprocess``: ``torch.distributed.run --nproc_per_node 2`` (on Windows: two ranks started here with the same
  environment) runs ``test_lerobot_order.py --cache_mode during_training`` with both ranks filling one shared empty
  cache per case; both ranks must PASS.
* ``losses``: 40 steps of ACT with plain ``lerobot.scripts.lerobot_train`` and with ``fastload.train
  --fastload.cache_mode=during_training`` (num_workers=2, image transforms on, mp4 data): identical logged losses.
* ``timing``: epoch 1 (filling) vs epoch 2 (cache) of FastLoader on a larger synthetic mp4 dataset.

All datasets use every episode (LeRobot < 0.4.4 computes delta windows from the relative row when ``episodes=`` is
a subset; see ``tests/test_equivalence.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO / "src"))

ALL_TESTS = ("during_training", "restart", "build_first", "min_free_disk", "prebuilt_missing", "multiprocess",
             "limited", "delete_cache", "losses", "timing")
HUGE_GB = "1e9"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("USE_LIBUV", "0")  # torch.distributed on Windows builds without libuv
    return env


def _run(cmd: list[str], log: Path | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], env=_env(), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=str(REPO))
    if log is not None:
        log.write_text(p.stdout + "\n----- stderr -----\n" + p.stderr, encoding="utf-8")
    return p


def _noise_free(text: str) -> str:
    skip = ("Starting second pass", "TorchCodec", "torchcodec", "Map:", "examples/s")
    return "\n".join(line for line in text.splitlines() if not any(s in line for s in skip))


class Data:
    """A synthetic dataset and the cache build_cache made from it."""

    def __init__(self, work: Path, variant: str, episodes: int = 6):
        from test_lerobot_order import make_synthetic

        self.variant = variant
        self.root = work / f"ds_{variant}_{episodes}"
        self.cache = work / f"cache_{variant}_{episodes}"
        if (self.cache / "COMPLETE").exists():  # made earlier in this work dir (restart and timing share one)
            self.repo_id = "fastload/synthetic"
            return
        self.repo_id = make_synthetic(self.root, episodes=episodes, videos=variant == "video")
        p = _run([sys.executable, "-m", "fastload.build_cache", "--repo_id", self.repo_id, "--root", self.root,
                  "--cache_dir", self.cache, "--num_workers", "1", "--verify", "8"])
        if p.returncode != 0:
            raise RuntimeError(f"build_cache failed:\n{p.stderr[-3000:]}")

    def total_frames(self) -> int:
        return int(json.loads((self.cache / "fastload_meta.json").read_text())["total_frames"])


def _order_args(data: Data, **kw) -> SimpleNamespace:
    """The argument namespace test_lerobot_order.run() reads."""
    base = dict(repo_id=data.repo_id, root=str(data.root), cache_dir=str(data.cache), device="cpu", placement="host",
                batch=8, epochs=2, seed=1000, episodes=0, hook_device="cpu", augment_device="cpu",
                cache_mode="prebuilt", decode_workers=2, background_fill=True, finish_cache=False, max_batches=None)
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------------------------------------
# tests
# ------------------------------------------------------------------------------------------------
def test_during_training(data: Data, work: Path) -> list[str]:
    fill = work / f"fill_{data.variant}"
    p = _run([sys.executable, TESTS_DIR / "test_lerobot_order.py", "--repo_id", data.repo_id, "--root", data.root,
              "--cache_dir", data.cache, "--fill_dir", fill, "--cache_mode", "during_training",
              "--samplers", "random,episode_aware", "--workers", "0,2", "--transforms", "off,on",
              "--augment_device", "cpu", "--atol", "0"], log=work / f"during_training_{data.variant}.log")
    out = _noise_free(p.stdout)
    print("\n".join("    " + line for line in out.splitlines() if line.strip()))
    if p.returncode != 0 or not out.rstrip().endswith("PASS"):
        return [f"test_lerobot_order --cache_mode during_training failed (exit {p.returncode}): "
                f"{_noise_free(p.stderr)[-2000:]}"]
    return []


def test_restart(data: Data, work: Path) -> list[str]:
    from accelerate import Accelerator

    from fastload.cache import COMPLETE_FILE, FILLED_FILE
    from test_lerobot_order import LAST_RUN, _transform_configs, compare, compare_caches, run

    # A larger dataset than the other tests: the loader requests the frames of at least 8 batches ahead (each run of
    # needed frames extended by a few), which on the 6-episode dataset already covers every frame, so run 1 would
    # complete the cache before it is stopped.
    data = Data(work, data.variant, episodes=30)
    errors: list[str] = []
    cache = work / f"restart_{data.variant}"
    acc = Accelerator(cpu=True)
    tf = _transform_configs()["on"]
    n = data.total_frames()
    # run 1: stopped after 3 batches of 4; only frames batches asked for are decoded (background filling off)
    a1 = _order_args(data, cache_dir=str(cache), cache_mode="during_training", background_fill=False, max_batches=3,
                     batch=4)
    run("fast", a1, acc, 2, "episode_aware", tf)
    filled1 = int(np.count_nonzero(np.load(cache / FILLED_FILE)))
    print(f"    run 1 (stopped after 3 batches): {filled1} of {n} frames filled, "
          f"{LAST_RUN['fill_stats']['frames']} decoded")
    if (cache / COMPLETE_FILE).exists() or not 0 < filled1 < n:
        errors.append(f"run 1 should leave a partial cache: {filled1} of {n} frames, "
                      f"COMPLETE={(cache / COMPLETE_FILE).exists()}")
    # A crash in the middle of writing leaves frames written but not marked: unmark (and zero) every other frame
    # of the first half, so run 2 has more to redo than the few frames run 1 never reached.
    filled = np.load(cache / FILLED_FILE, mmap_mode="r+")
    drop = np.arange(0, n // 2, 2)
    for v in json.loads((cache / "fastload_meta.json").read_text())["keys"].values():
        arr = np.load(cache / v["file"], mmap_mode="r+")
        arr[drop] = 0
        arr.flush()
        del arr
    filled[drop] = 0
    filled.flush()
    del filled
    filled1 = int(np.count_nonzero(np.load(cache / FILLED_FILE)))
    print(f"    after unmarking {len(drop)} frames: {filled1} of {n} filled")
    # run 2: the same cache; must decode only what is missing and match LeRobot exactly
    a2 = _order_args(data, cache_dir=str(cache), cache_mode="during_training", finish_cache=True)
    ref = run("lerobot", a2, acc, 2, "episode_aware", tf)
    got = run("fast", a2, acc, 2, "episode_aware", tf)
    errors += compare(ref, got, ref[2], 0.0)
    if LAST_RUN.get("fill_stats_final") is None:  # the cache was complete when run 2 started
        return errors + ["run 2 had nothing to fill (cache complete after run 1)"]
    decoded2 = LAST_RUN["fill_stats_final"]["frames"]  # the epochs plus the rest finish_cache() decoded
    print(f"    run 2: decoded {decoded2} frames (missing after run 1: {n - filled1}), epoch times "
          f"{', '.join(f'{x:.2f}s' for x in LAST_RUN['epoch_s'])}")
    if decoded2 != n - filled1:
        errors.append(f"run 2 decoded {decoded2} frames, expected exactly the {n - filled1} missing ones")
    errors += compare_caches(cache, data.cache)
    return errors


def _act_args(data: Data, out: Path, steps: int, num_workers: int = 0, transforms: bool = False) -> list[str]:
    return [f"--dataset.repo_id={data.repo_id}", f"--dataset.root={data.root}", "--policy.type=act",
            "--policy.device=cpu", "--policy.push_to_hub=false", "--policy.chunk_size=4", "--policy.n_action_steps=4",
            "--batch_size=4", f"--steps={steps}", "--log_freq=1", f"--num_workers={num_workers}",
            "--save_checkpoint=false", "--wandb.enable=false", "--eval_freq=0", f"--output_dir={out}",
            f"--dataset.image_transforms.enable={'true' if transforms else 'false'}"]


def test_build_first(data: Data, work: Path) -> list[str]:
    from fastload.cache import cache_status
    from test_lerobot_order import compare_caches

    cache = work / f"build_first_{data.variant}"
    out = work / f"out_build_first_{data.variant}"
    # build with the reference cache's decoder: LeRobot's default backend can depend on the process (e.g. torchcodec
    # loads in a bare build_cache process but not after lerobot_train's imports), and compare_caches checks it
    ref_backend = json.loads((data.cache / "fastload_meta.json").read_text()).get("video_backend")
    p = _run([sys.executable, "-m", "fastload.train", *_act_args(data, out, 6),
              *([f"--dataset.video_backend={ref_backend}"] if ref_backend else []),
              f"--fastload.cache_dir={cache}", "--fastload.cache_mode=build_first", "--fastload.delete_cache=false"],
             log=work / f"build_first_{data.variant}.log")
    text = p.stdout + p.stderr
    errors = []
    if p.returncode != 0:
        return [f"fastload.train build_first exited {p.returncode}: {_noise_free(p.stderr)[-3000:]}"]
    for needle in ("cache_mode=build_first: building the cache", "cache built in", "step:6"):
        if needle not in text:
            errors.append(f"log line {needle!r} missing")
    if cache_status(cache) != "complete":
        errors.append("cache not complete after build_first")
    else:
        errors += compare_caches(cache, data.cache)
    print(f"    build_first: exit {p.returncode}, log lines: " + "; ".join(
        line.strip()[-110:] for line in text.splitlines() if "fastload: cache" in line or "step:6 " in line))
    return errors


def _refused(p: subprocess.CompletedProcess, cache: Path, what: str, needles: tuple[str, ...]) -> list[str]:
    text = p.stdout + p.stderr
    errors = []
    if p.returncode == 0:
        errors.append(f"{what}: exited 0")
    for needle in needles:
        if needle not in text:
            errors.append(f"{what}: {needle!r} not in the output: {_noise_free(text)[-1500:]}")
    if cache.exists():
        errors.append(f"{what}: created {cache} ({sorted(x.name for x in cache.iterdir())})")
    return errors


def test_min_free_disk(data: Data, work: Path) -> list[str]:
    from fastload import make_loader
    from fastload.cache import DiskSpaceError

    errors = []
    needles = ("not enough free disk", "min_free_disk_gb", "plain lerobot-train", "--keys")
    cache = work / f"disk_build_cache_{data.variant}"
    p = _run([sys.executable, "-m", "fastload.build_cache", "--repo_id", data.repo_id, "--root", data.root,
              "--cache_dir", cache, "--min_free_disk_gb", HUGE_GB])
    errors += _refused(p, cache, "build_cache", needles)
    msg = next((ln for ln in (p.stdout + p.stderr).splitlines() if "not enough free disk" in ln), "")
    print(f"    build_cache: exit {p.returncode}: {msg[:200]}")
    for mode in ("build_first", "during_training"):
        cache = work / f"disk_{mode}_{data.variant}"
        p = _run([sys.executable, "-m", "fastload.train", *_act_args(data, work / f"out_disk_{mode}", 2),
                  f"--fastload.cache_dir={cache}", f"--fastload.cache_mode={mode}",
                  f"--fastload.min_free_disk_gb={HUGE_GB}"])
        errors += _refused(p, cache, f"fastload.train {mode}", needles)
        print(f"    fastload.train {mode}: exit {p.returncode}")
    # custom script: make_loader(cache_mode="during_training")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cache = work / f"disk_make_loader_{data.variant}"
    ds = LeRobotDataset(data.repo_id, root=data.root)
    try:
        make_loader(ds, 4, cache_dir=str(cache), cache_mode="during_training", min_free_disk_gb=float(HUGE_GB))
        errors.append("make_loader(during_training) did not refuse")
    except DiskSpaceError as e:
        print(f"    make_loader during_training: DiskSpaceError ({str(e).splitlines()[0][:120]}...)")
    if cache.exists():
        errors.append(f"make_loader created {cache}")
    return errors


def test_prebuilt_missing(data: Data, work: Path) -> list[str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from fastload import CachedLeRobotDataset

    errors = []
    needles = ("--fastload.cache_mode=build_first", "--fastload.cache_mode=during_training",
               "python -m fastload.build_cache --repo_id")
    cache = work / f"missing_{data.variant}"
    p = _run([sys.executable, "-m", "fastload.train", *_act_args(data, work / "out_missing", 2),
              f"--fastload.cache_dir={cache}"])
    errors += _refused(p, cache, "fastload.train prebuilt", needles)
    first = next((ln for ln in (p.stdout + p.stderr).splitlines() if "No fastload cache" in ln), "")
    print(f"    fastload.train prebuilt: exit {p.returncode}: {first[:160]}")
    ds = LeRobotDataset(data.repo_id, root=data.root)
    try:
        CachedLeRobotDataset.from_dataset(ds, cache)
        errors.append("from_dataset(prebuilt) on a missing cache did not raise")
    except FileNotFoundError as e:
        errors += [f"from_dataset message lacks {n!r}" for n in needles if n not in str(e)]
    return errors


def _launch_two(script_args: list, log: Path) -> subprocess.CompletedProcess:
    """Two processes of one node, like ``torchrun --nproc_per_node 2``. On Windows torch is built without libuv and
    torchrun's rendezvous TCPStore ignores USE_LIBUV=0, so the ranks are started here with the same environment
    variables (RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, MASTER_ADDR/PORT; both share this process as parent)."""
    if sys.platform != "win32":
        return _run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--standalone",
                     *script_args], log=log)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    print("  $ [2 ranks, env://] " + " ".join(str(c) for c in script_args), flush=True)
    procs = []
    for rank in range(2):
        env = dict(_env(), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), USE_LIBUV="0")
        procs.append(subprocess.Popen([sys.executable, *[str(a) for a in script_args]], env=env, cwd=str(REPO),
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                      errors="replace"))
    outs = [p.communicate(timeout=3600) for p in procs]
    rc = max(abs(p.returncode) for p in procs)
    out, err = "".join(o for o, _ in outs), "".join(e for _, e in outs)
    log.write_text(out + "\n----- stderr -----\n" + err, encoding="utf-8")
    return subprocess.CompletedProcess(script_args, rc, out, err)


def test_multiprocess(data: Data, work: Path) -> list[str]:
    fill = work / f"mp_fill_{data.variant}"
    p = _launch_two([TESTS_DIR / "test_lerobot_order.py", "--repo_id", data.repo_id, "--root", data.root,
                     "--cache_dir", data.cache, "--fill_dir", fill, "--cache_mode", "during_training",
                     "--samplers", "episode_aware", "--workers", "0,2", "--transforms", "on",
                     "--augment_device", "cpu", "--atol", "0"], log=work / f"multiprocess_{data.variant}.log")
    out = _noise_free(p.stdout)
    print("\n".join("    " + line for line in out.splitlines() if line.strip()))
    passes = [line for line in out.splitlines() if re.match(r"\[rank \d/2\] PASS", line.strip())]
    if p.returncode != 0 or len(passes) != 2:
        return [f"multi-process run failed (exit {p.returncode}, {len(passes)} ranks PASS): "
                f"{_noise_free(p.stderr)[-3000:]}"]
    return []


def _full_gb(data: Data) -> float:
    meta = json.loads((data.cache / "fastload_meta.json").read_text())
    return meta["total_frames"] * sum(int(np.prod(v["shape"])) for v in meta["keys"].values()) / 1e9


def test_limited(data: Data, work: Path) -> list[str]:
    """Size-limited cache (half the frames): build_cache stores whole episodes only (the rest takes no disk on Linux),
    and FastLoader (prebuilt / during_training) and LeRobot's DataLoader over the cache give LeRobot's batches."""
    errors: list[str] = []
    limited = Path(str(data.cache) + "_limited")
    common = ["--repo_id", data.repo_id, "--root", data.root, "--cache_dir", data.cache, "--cache_fraction", "0.5",
              "--samplers", "episode_aware", "--workers", "0,2", "--transforms", "off,on",
              "--augment_device", "cpu", "--atol", "0"]
    runs = [("prebuilt, FastLoader", []), ("prebuilt, LeRobot DataLoader", ["--loader", "dataloader"]),
            ("during_training, FastLoader", ["--cache_mode", "during_training", "--fill_dir", work / f"lim_fill_{data.variant}"])]
    for name, extra in runs:
        p = _run([sys.executable, TESTS_DIR / "test_lerobot_order.py", *common, *extra],
                 log=work / f"limited_{data.variant}_{len(extra)}.log")
        out = _noise_free(p.stdout)
        print(f"    [{name}]")
        print("\n".join("      " + line for line in out.splitlines() if line.strip()))
        if p.returncode != 0 or not out.rstrip().endswith("PASS"):
            errors.append(f"{name}: exit {p.returncode}: {_noise_free(p.stderr)[-2000:]}")
    if not (limited / "fastload_meta.json").exists():
        return errors + [f"no size-limited cache at {limited}"]
    meta = json.loads((limited / "fastload_meta.json").read_text())
    stored = np.load(limited / "stored.npy")
    n, kept = meta["total_frames"], int(stored.sum())
    fb = sum(int(np.prod(v["shape"])) for v in meta["keys"].values())
    print(f"    limited cache: {kept} of {n} frames stored ({meta.get('stored_episodes')} of {meta['total_episodes']} "
          f"episodes), {kept * fb} of max {meta['max_cache_gb'] * 1e9:.0f} bytes")
    if not 0 < kept < n or kept * fb > meta["max_cache_gb"] * 1e9:
        errors.append(f"stored {kept} of {n} frames ({kept * fb} bytes) for max_cache_gb={meta['max_cache_gb']}")
    filled = np.load(limited / "filled.npy")
    if not np.array_equal(filled > 0, stored > 0):
        errors.append("filled frames != stored frames")
    if hasattr(os, "statvfs"):  # sparse files: frames not stored take no disk space
        from fastload.cache import allocated_bytes

        on_disk = sum(allocated_bytes(limited / v["file"]) for v in meta["keys"].values())
        print(f"    frame files on disk: {on_disk} bytes (stored frames {kept * fb}, full {n * fb})")
        if on_disk > kept * fb + 64 * 4096 * len(meta["keys"]):
            errors.append(f"frame files take {on_disk} bytes on disk, stored frames only {kept * fb}")
    return errors


def test_delete_cache(data: Data, work: Path) -> list[str]:
    """A cache a training run created is deleted when training ends normally; not with delete_cache=false, not a
    build_cache cache, not while another process holds a lease, not after a failed run."""
    from fastload.cache import USERS_DIR, cache_status

    errors: list[str] = []

    def train(cache: Path, tag: str, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
        cmd = [sys.executable, "-m", "fastload.train", *_act_args(data, work / f"out_del_{tag}_{data.variant}", 3),
               f"--fastload.cache_dir={cache}", *extra]
        if env:
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
        try:
            return _run(cmd, log=work / f"delete_{tag}_{data.variant}.log")
        finally:
            if env:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    # 1. build_first (default delete_cache=true): gone after training
    c1 = work / f"del_bf_{data.variant}"
    p = train(c1, "bf", "--fastload.cache_mode=build_first")
    ok = p.returncode == 0 and not c1.exists() and "deleted the cache" in p.stdout + p.stderr
    print(f"    build_first: exit {p.returncode}, cache exists after: {c1.exists()}")
    if not ok:
        errors.append(f"build_first: exit {p.returncode}, cache still exists: {c1.exists()}: "
                      f"{_noise_free(p.stderr)[-1500:]}")
    # 2. during_training with delete_cache=false: kept
    c2 = work / f"del_keep_{data.variant}"
    p = train(c2, "keep", "--fastload.cache_mode=during_training", "--fastload.delete_cache=false")
    print(f"    during_training delete_cache=false: exit {p.returncode}, kept: {c2.exists()}")
    if p.returncode != 0 or cache_status(c2) == "missing":
        errors.append(f"delete_cache=false: exit {p.returncode}, cache status {cache_status(c2)}")
    # 3. the build_cache cache (prebuilt): never deleted
    p = train(data.cache, "prebuilt")
    print(f"    prebuilt (build_cache): exit {p.returncode}, kept: {data.cache.exists()}")
    if p.returncode != 0 or cache_status(data.cache) != "complete":
        errors.append(f"prebuilt: exit {p.returncode}, cache status {cache_status(data.cache)}")
    # 4. c2 was created by training: a live lease of another process keeps it, then it is deleted
    lease = c2 / USERS_DIR / f"{__import__('socket').gethostname()}-{os.getpid()}"
    lease.parent.mkdir(exist_ok=True)
    lease.write_text("test")
    p = train(c2, "leased", "--fastload.cache_mode=during_training")
    print(f"    another process holds a lease: exit {p.returncode}, kept: {c2.exists()}")
    if p.returncode != 0 or not c2.exists() or "still used by" not in p.stdout + p.stderr:
        errors.append(f"leased: exit {p.returncode}, kept {c2.exists()}")
    # 5. a failed run (simulated OOM, no retry) keeps it for resuming
    lease.unlink()
    p = train(c2, "failed", "--fastload.cache_mode=during_training", "--fastload.oom_retry=false",
              env={"FASTLOAD_DEBUG_OOM": "0:2"})
    print(f"    failed run: exit {p.returncode}, kept: {c2.exists()}")
    if p.returncode == 0 or not c2.exists():
        errors.append(f"failed run: exit {p.returncode}, kept {c2.exists()}")
    # 6. finally a normal run deletes it (the failed run's lease is stale)
    p = train(c2, "last", "--fastload.cache_mode=during_training")
    print(f"    normal run after that: exit {p.returncode}, deleted: {not c2.exists()}")
    if p.returncode != 0 or c2.exists():
        errors.append(f"last run: exit {p.returncode}, cache still exists: {c2.exists()}")
    return errors


def _losses(text: str) -> list[tuple[int, str]]:
    return [(int(m.group(1)), m.group(2)) for m in re.finditer(r"step:(\S+?) .*?loss:(\S+)", text)
            if m.group(1).isdigit()]


def test_losses(data: Data, work: Path) -> list[str]:
    steps = 40
    a = _run([sys.executable, "-m", "lerobot.scripts.lerobot_train",
              *_act_args(data, work / "out_loss_lerobot", steps, num_workers=2, transforms=True)],
             log=work / "losses_lerobot.log")
    if a.returncode:
        return [f"lerobot_train failed (exit {a.returncode}): {_noise_free(a.stderr)[-3000:]}"]
    la = _losses(a.stdout + a.stderr)
    print(f"    lerobot_train losses ({len(la)}): {' '.join(v for _, v in la)}")
    if len(la) != steps:
        return [f"expected {steps} logged losses, got {len(la)}"]
    errors = []
    # the whole cache, and a cache limited to half the frames (the others decoded by FastLoader's decoder processes)
    for tag, limit in (("full", []), ("half", [f"--fastload.max_cache_gb={_full_gb(data) / 2!r}"])):
        b = _run([sys.executable, "-m", "fastload.train",
                  *_act_args(data, work / f"out_loss_fastload_{tag}", steps, num_workers=2, transforms=True),
                  f"--fastload.cache_dir={work / f'loss_cache_{tag}'}", "--fastload.cache_mode=during_training", *limit],
                 log=work / f"losses_fastload_{tag}.log")
        if b.returncode:
            errors.append(f"fastload.train ({tag} cache) failed (exit {b.returncode}): {_noise_free(b.stderr)[-3000:]}")
            continue
        text = b.stdout + b.stderr
        lb = _losses(text)
        print(f"    fastload.train losses, {tag} cache ({len(lb)}): {' '.join(v for _, v in lb)}")
        if la != lb:
            errors.append(f"{tag} cache: logged losses differ: {[(x, y) for x, y in zip(la, lb) if x != y][:5]}")
        if limit and "the cache stores" not in text:
            errors.append("half cache: no size-limit log line")
    return errors


def test_timing(work: Path, episodes: int = 30) -> list[str]:
    from accelerate import Accelerator

    from test_lerobot_order import LAST_RUN, run

    data = Data(work, "video", episodes=episodes)
    acc = Accelerator(cpu=True)
    a = _order_args(data, cache_dir=str(work / "timing_fill"), cache_mode="during_training", batch=16)
    run("fast", a, acc, 2, "episode_aware", None)
    e1, e2 = LAST_RUN["epoch_s"]
    st = LAST_RUN["fill_stats"]
    print(f"    during_training ({data.total_frames()} frames, 2 cameras, {a.decode_workers} decoder processes): "
          f"epoch 1 {e1:.2f}s (waited {st['wait_s']:.2f}s for frames, {st['frames']} frames decoded, "
          f"{st['background_frames']} in the background), epoch 2 {e2:.2f}s; complete after the epochs: "
          f"{LAST_RUN['complete_after_epochs']}")
    a = _order_args(data, batch=16)
    run("fast", a, acc, 2, "episode_aware", None)
    p1, p2 = LAST_RUN["epoch_s"]
    print(f"    prebuilt (same data): epoch 1 {p1:.2f}s, epoch 2 {p2:.2f}s")
    t0 = time.perf_counter()
    ref = run("lerobot", a, acc, 2, "episode_aware", None)
    print(f"    plain LeRobot DataLoader (num_workers=2, decoding video): 2 epochs {time.perf_counter() - t0:.2f}s "
          f"({len(ref[0])} batches)")
    return [] if e2 < e1 else [f"epoch 2 ({e2:.2f}s) not faster than epoch 1 ({e1:.2f}s)"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tests", default=",".join(ALL_TESTS))
    p.add_argument("--variants", default="image,video")
    p.add_argument("--work_dir", default=None, help="keep the data and logs here (default: a temp dir)")
    args = p.parse_args()
    tests = args.tests.split(",")
    unknown = set(tests) - set(ALL_TESTS)
    if unknown:
        raise SystemExit(f"unknown tests {sorted(unknown)}; known: {ALL_TESTS}")
    tmp = None
    if args.work_dir:
        work = Path(args.work_dir).resolve()  # subprocesses run with cwd=REPO
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
    else:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows: files a process still maps
        work = Path(tmp.name)
    results: list[tuple[str, list[str], float]] = []
    per_variant = [t for t in tests if t not in ("losses", "timing")]
    for variant in args.variants.split(","):
        if not per_variant:
            break
        data = Data(work, variant)
        for t in per_variant:
            print(f"[{t} / {variant}]", flush=True)
            t0 = time.perf_counter()
            errs = globals()[f"test_{t}"](data, work)
            results.append((f"{t} / {variant}", errs, time.perf_counter() - t0))
            print(f"  -> {'OK' if not errs else 'FAIL'} ({time.perf_counter() - t0:.0f}s)", flush=True)
            for e in errs[:8]:
                print(f"     {e}")
    if "losses" in tests:
        print("[losses / video]", flush=True)
        t0 = time.perf_counter()
        errs = test_losses(Data(work, "video") if not (work / "ds_video_6").exists() else _existing(work), work)
        results.append(("losses / video", errs, time.perf_counter() - t0))
        print(f"  -> {'OK' if not errs else 'FAIL'}", flush=True)
        for e in errs[:8]:
            print(f"     {e}")
    if "timing" in tests:
        print("[timing / video]", flush=True)
        t0 = time.perf_counter()
        errs = test_timing(work)
        results.append(("timing / video", errs, time.perf_counter() - t0))
        print(f"  -> {'OK' if not errs else 'FAIL'}", flush=True)
    print("\nsummary:")
    for name, errs, dt in results:
        print(f"  {name}: {'OK' if not errs else 'FAIL'} ({dt:.0f}s)")
    failed = sum(bool(e) for _, e, _ in results)
    print("PASS" if not failed else f"FAIL ({failed} of {len(results)})")
    if tmp is not None:
        tmp.cleanup()
    return 1 if failed else 0


def _existing(work: Path) -> Data:
    """The mp4 Data already made in ``work`` (do not rebuild it)."""
    d = Data.__new__(Data)
    d.variant, d.root, d.cache, d.repo_id = "video", work / "ds_video_6", work / "cache_video_6", "fastload/synthetic"
    return d


if __name__ == "__main__":
    sys.exit(main())
