"""Record small training observations; this is not a held-out policy benchmark."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from superblock.experiment import run_experiment
from superblock.monitor import load_checkpoint


def _last_csv(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {key: float(value) for key, value in list(csv.DictReader(stream))[-1].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,42,2026")
    parser.add_argument("--motion-days", type=int, default=5)
    parser.add_argument("--motion-epochs", type=int, default=3)
    parser.add_argument("--task-days", type=int, default=10)
    parser.add_argument("--steps-per-day", type=int, default=100)
    parser.add_argument("--output-root", default="artifacts/method-checks")
    args = parser.parse_args()
    try:
        seeds = [int(token.strip()) for token in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if len(set(seeds)) != len(seeds):
        parser.error("--seeds must not contain duplicates")
    if min(args.motion_days, args.motion_epochs, args.task_days, args.steps_per_day) <= 0:
        parser.error("training budgets must be positive")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("start Python with PYTHONHASHSEED=0 for this observation protocol")

    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    root = Path(args.output_root).expanduser().resolve() / name
    root.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "kind": "training_observation_not_held_out_evaluation",
        "python": platform.python_version(),
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "budget": vars(args),
        "seeds": seeds,
        "runs": [],
    }
    print(f"Output: {root}", flush=True)
    for seed in seeds:
        common = ["--seed", str(seed), "--steps-per-day", str(args.steps_per_day)]
        tasks = common + ["--days", str(args.task_days)]
        with (root / f"seed-{seed}.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                motion = run_experiment("motion", common + ["--max-days", str(args.motion_days), "--epochs-per-night", str(args.motion_epochs)], runs_dir=root, run_id=f"seed-{seed}-motion")
            history = load_checkpoint(str(motion / "train.ckpt"))["history"]
            motion_digest = hashlib.sha256((motion / "train.ckpt").read_bytes()).hexdigest()
            summary["runs"].append({"seed": seed, "stage": "motion", "run_dir": motion.name, "first": history[0], "last": history[-1]})
            print(f"seed={seed}: motion completed", flush=True)
            input_args = ["--motion-checkpoint-path", str(motion / "train.ckpt")]
            for policy in ("heuristic", "qlearn"):
                with contextlib.redirect_stdout(log):
                    forage = run_experiment("forage", tasks + input_args + ["--policy", policy], runs_dir=root, run_id=f"seed-{seed}-forage-{policy}")
                row = _last_csv(forage / "forage_metrics.csv")
                summary["runs"].append({"seed": seed, "stage": "forage", "policy": policy, "run_dir": forage.name, "last": row})
                print(f"seed={seed}: {policy} completed", flush=True)
            with contextlib.redirect_stdout(log):
                evade = run_experiment("evade", tasks + input_args, runs_dir=root, run_id=f"seed-{seed}-evade")
            summary["runs"].append({"seed": seed, "stage": "evade", "run_dir": evade.name, "last": _last_csv(evade / "evade_metrics.csv")})
            print(f"seed={seed}: evade completed", flush=True)
            if hashlib.sha256((motion / "train.ckpt").read_bytes()).hexdigest() != motion_digest:
                raise RuntimeError("A task overwrote the shared motion checkpoint")
        (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {root / 'summary.json'}")


if __name__ == "__main__":
    main()
