"""Run existing trainers in isolated, inspectable experiment directories."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

STAGES = {"motion": "superblock.train", "forage": "superblock.forage_train", "evade": "superblock.evade_train"}
OUTPUTS = {
    "motion": {"checkpoint_path": "train.ckpt", "dashboard_path": "dashboard.html"},
    "forage": {
        "out_checkpoint": "forage.ckpt",
        "out_metrics_csv": "forage_metrics.csv",
        "out_attempts_csv": "forage_attempts.csv",
        "out_dashboard": "forage_dashboard.html",
        "motion_output_checkpoint": "train_forage_tuned.ckpt",
    },
    "evade": {"dashboard_path": "evade_dashboard.html", "metrics_csv_path": "evade_metrics.csv"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_info() -> dict:
    root = Path(__file__).resolve().parent.parent
    info: dict = {"python": platform.python_version(), "git_commit": None, "git_dirty": None}
    # Do not report the revision of an unrelated enclosing checkout after installation.
    if not (root / ".git").exists():
        return info
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def run_experiment(
    stage: str, training_args: list[str], *, runs_dir: str | Path = "artifacts/runs", run_id: str | None = None
) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    module = importlib.import_module(STAGES[stage])
    parser = module.build_parser()
    parser.allow_abbrev = False
    managed = set(OUTPUTS[stage]) | {"master_dashboard_path"}
    output_options = {option for action in parser._actions if action.dest in managed for option in action.option_strings}
    if any(token.split("=", 1)[0] in output_options for token in training_args):
        raise ValueError("Experiment output paths are managed automatically; choose --runs-dir/--run-id instead")
    args = parser.parse_args(training_args)
    if stage == "motion" and args.resume:
        raise ValueError("Use python -m superblock.train --resume for checkpoint continuation; experiments start fresh")
    for name in ("max_days", "days", "steps_per_day", "batch_size", "save_every_days", "motion_batch_size"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + stage + "-" + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("run_id must be a simple name, up to 80 letters, digits, dots, underscores or hyphens")
    root = Path(runs_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    run_dir.mkdir()  # An existing run is never overwritten.

    input_path = Path(args.motion_checkpoint_path).expanduser().resolve() if stage != "motion" else None
    for name, filename in OUTPUTS[stage].items():
        setattr(args, name, str(run_dir / filename))
    args.master_dashboard_path = str(run_dir / "master_dashboard.html")
    if input_path is not None:
        args.motion_checkpoint_path = str(run_dir / "input_motion.ckpt")

    config = {
        "schema_version": 1,
        "stage": stage,
        "training_args": training_args,
        "parameters": vars(args).copy(),
        "source": _source_info(),
        "input_checkpoint": {"source": str(input_path), "sha256": None} if input_path else None,
    }
    state = {"schema_version": 1, "run_id": run_id, "stage": stage, "status": "running", "started_at": _now()}
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "run.json", state)
    try:
        if input_path is not None:
            shutil.copyfile(input_path, args.motion_checkpoint_path)
            config["input_checkpoint"]["sha256"] = hashlib.sha256(Path(args.motion_checkpoint_path).read_bytes()).hexdigest()
            _write_json(run_dir / "config.json", config)
        if stage == "motion":
            args._propagate_interrupt = True
        module.run(args)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        raise
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    else:
        state["status"] = "completed"
    finally:
        state["finished_at"] = _now()
        state["outputs"] = sorted(str(path.relative_to(run_dir)) for path in run_dir.iterdir() if path.is_file())
        _write_json(run_dir / "run.json", state)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    training_args: list[str] = []
    if "--" in argv:
        separator = argv.index("--")
        argv, training_args = argv[:separator], argv[separator + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--runs-dir", default="artifacts/runs")
    parser.add_argument("--run-id", help="Optional unique experiment name; existing names are rejected")
    options = parser.parse_args(argv)
    try:
        directory = run_experiment(options.stage, training_args, runs_dir=options.runs_dir, run_id=options.run_id)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Experiment saved: {directory}")


if __name__ == "__main__":
    main()
