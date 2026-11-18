"""Restart multi-GPU fastload training after a CUDA OOM, from the last checkpoint, with a smaller GPU tier.

Under several processes a rank that runs out of memory cannot redo its step alone (the other ranks are
already waiting in the gradient all-reduce), so ``fastload.train`` stops every process immediately and
writes a note. This wrapper reads the note and runs the same launch command again::

    python -m fastload.supervise [--max_restarts 3] -- \\
        accelerate launch --num_processes 8 -m fastload.train <lerobot-train args> --fastload.cache_dir=...

* a checkpoint exists  -> ``--config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json
  --resume=true`` (LeRobot's own resume: model, optimizer, scheduler, step and RNG state) plus the
  ``--fastload.*`` options, with ``--fastload.max_gpu_cache_gb`` set to half of the tier that ran out;
* no checkpoint yet    -> the original command again with the smaller tier; the failed output directory
  is renamed to ``<output_dir>.oom<N>`` so LeRobot does not refuse to start.

Any other failure is returned as is (no restart).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fastload.train import NOTE_DIR_ENV

_MAX_TIER = "--fastload.max_gpu_cache_gb"


def _train_args_start(cmd: list[str]) -> int:
    for i, tok in enumerate(cmd):
        t = tok.replace("\\", "/")
        if t == "fastload.train" or t.endswith("fastload-train") or t.endswith("fastload-train.exe") \
                or t.endswith("fastload/train.py"):
            return i + 1
    raise SystemExit("fastload.supervise: the command must run `-m fastload.train` (or fastload-train)")


def _read_notes(d: Path) -> list[dict]:
    notes = []
    for f in sorted(d.glob("rank*.json")):
        try:
            notes.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            pass
    return notes


def restart_command(cmd: list[str], notes: list[dict], attempt: int) -> list[str] | None:
    """The command for the next attempt, or None when a smaller GPU tier cannot help."""
    tier_bytes = min(int(n["gpu_tier_bytes"]) for n in notes)
    if tier_bytes <= 0:
        return None
    new_max = f"{tier_bytes / 2 / 1e9:.4g}"  # GB
    i = _train_args_start(cmd)
    train_args = [a for a in cmd[i:] if not a.startswith(_MAX_TIER)]
    output_dir = next((n["output_dir"] for n in notes if n.get("output_dir")), None)
    ckpt = Path(output_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json" if output_dir else None
    if ckpt is not None and ckpt.exists():
        fl_args = [a for a in train_args if a.startswith("--fastload.")]
        new_train = [f"--config_path={ckpt}", "--resume=true", *fl_args]
    else:
        if output_dir and Path(output_dir).exists():
            moved = f"{output_dir}.oom{attempt}"
            shutil.move(output_dir, moved)
            print(f"fastload.supervise: no checkpoint yet; moved {output_dir} -> {moved}", flush=True)
        new_train = train_args
    return [*cmd[:i], *new_train, f"{_MAX_TIER}={new_max}"]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit(__doc__)
    sep = argv.index("--")
    p = argparse.ArgumentParser(prog="python -m fastload.supervise")
    p.add_argument("--max_restarts", type=int, default=3)
    a = p.parse_args(argv[:sep])
    cmd = argv[sep + 1:]
    _train_args_start(cmd)

    note_dir = Path(tempfile.mkdtemp(prefix="fastload_oom_"))
    env = dict(os.environ, **{NOTE_DIR_ENV: str(note_dir)})
    rc = 0
    try:
        for attempt in range(a.max_restarts + 1):
            for f in note_dir.glob("rank*.json"):
                f.unlink()
            print(f"fastload.supervise: attempt {attempt}: {' '.join(cmd)}", flush=True)
            rc = subprocess.call(cmd, env=env)
            if rc == 0:
                return 0
            notes = _read_notes(note_dir)
            if not notes:
                print(f"fastload.supervise: exited with status {rc} without an OOM note; not restarting", flush=True)
                return rc
            if attempt == a.max_restarts:
                break
            nxt = restart_command(cmd, notes, attempt + 1)
            if nxt is None:
                print("fastload.supervise: CUDA OOM with no GPU frame tier left; fastload cannot free more memory "
                      "(lower --batch_size or use more GPUs)", flush=True)
                return rc
            cmd = nxt
        print(f"fastload.supervise: giving up after {a.max_restarts} restarts", flush=True)
        return rc
    finally:
        shutil.rmtree(note_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
