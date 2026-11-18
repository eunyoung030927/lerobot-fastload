"""fastload.supervise: restarts after an OOM note, from the last checkpoint, with half the GPU tier.

A fake ``fastload/train.py`` plays the training run: it records its arguments and, on the first attempt,
writes an OOM note the way ``fastload.train.stop_after_oom`` does and exits with status 1.

    python tests/test_supervise.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

FAKE_TRAIN = r'''
import json, os, sys
from pathlib import Path
log = Path(os.environ["FAKE_LOG"])
calls = json.loads(log.read_text()) if log.exists() else []
calls.append(sys.argv[1:])
log.write_text(json.dumps(calls))
if len(calls) == 1:
    out = os.environ["FAKE_OUT"]
    if os.environ.get("FAKE_CKPT"):
        p = Path(out) / "checkpoints" / "last" / "pretrained_model"
        p.mkdir(parents=True, exist_ok=True)
        (p / "train_config.json").write_text("{}")
    else:
        Path(out).mkdir(parents=True, exist_ok=True)
    note = Path(os.environ["FASTLOAD_SUPERVISE_DIR"]) / "rank1.json"
    note.write_text(json.dumps({"rank": 1, "world_size": 2, "output_dir": out, "gpu_tier_bytes": 10_000_000_000}))
    sys.exit(1)
'''


def run_case(with_checkpoint: bool) -> list[str]:
    import os

    from fastload.supervise import main

    errors = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "fastload").mkdir()
        script = d / "fastload" / "train.py"
        script.write_text(FAKE_TRAIN)
        out = d / "run"
        os.environ.update(FAKE_LOG=str(d / "calls.json"), FAKE_OUT=str(out))
        os.environ.pop("FAKE_CKPT", None)
        if with_checkpoint:
            os.environ["FAKE_CKPT"] = "1"
        args = ["--dataset.repo_id=x", f"--output_dir={out}", "--fastload.cache_dir=/c", "--fastload.max_gpu_cache_gb=50"]
        rc = main(["--max_restarts", "2", "--", sys.executable, str(script), *args])
        calls = json.loads((d / "calls.json").read_text())
        if rc != 0:
            errors.append(f"exit status {rc}")
        if len(calls) != 2:
            errors.append(f"{len(calls)} runs, expected 2")
        else:
            second = calls[1]
            if "--fastload.max_gpu_cache_gb=5" not in second or "--fastload.max_gpu_cache_gb=50" in second:
                errors.append(f"tier not halved: {second}")
            if "--fastload.cache_dir=/c" not in second:
                errors.append("fastload options dropped")
            if with_checkpoint:
                ckpt = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
                if f"--config_path={ckpt}" not in second or "--resume=true" not in second:
                    errors.append(f"not resumed from the checkpoint: {second}")
                if "--dataset.repo_id=x" in second:
                    errors.append("resume should take the run's settings from train_config.json")
            else:
                if "--dataset.repo_id=x" not in second or "--resume=true" in second:
                    errors.append(f"expected the original arguments without resume: {second}")
                if not Path(f"{out}.oom1").exists():
                    errors.append("failed output dir was not moved aside")
    return errors


def main() -> int:
    failed = 0
    for ckpt in (True, False):
        errors = run_case(ckpt)
        print(f"  {'with' if ckpt else 'without'} checkpoint: {'OK' if not errors else 'FAIL'}")
        for e in errors:
            print(f"    {e}")
        failed += bool(errors)
    print("PASS" if not failed else "FAIL")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
