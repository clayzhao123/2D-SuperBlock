"""A short, headless motion -> forage -> evade smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiment


def run_demo(runs_dir: str | Path = "artifacts/runs") -> list[Path]:
    motion = run_experiment("motion", ["--max-days", "2", "--steps-per-day", "16", "--epochs-per-night", "1"], runs_dir=runs_dir)
    input_args = ["--motion-checkpoint-path", str(motion / "train.ckpt")]
    short_args = ["--days", "2", "--steps-per-day", "16", "--hunger-interval", "2", "--hunger-death-steps", "8"]
    forage = run_experiment("forage", input_args + short_args + ["--policy", "qlearn"], runs_dir=runs_dir)
    evade = run_experiment("evade", input_args + short_args, runs_dir=runs_dir)
    return [motion, forage, evade]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="artifacts/runs")
    args = parser.parse_args()
    for path in run_demo(args.runs_dir):
        print(f"{path.name}: {path / 'master_dashboard.html'}")
    print("Smoke run completed; this checks execution, not policy convergence.")


if __name__ == "__main__":
    main()
