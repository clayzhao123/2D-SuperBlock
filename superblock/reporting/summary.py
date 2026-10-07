"""Keep each training summary with its own output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..master_dashboard import write_master_dashboard


def write_training_summary(args: argparse.Namespace, stage: str) -> None:
    output = Path(getattr(args, "master_dashboard_path", "artifacts/master_dashboard.html"))
    root = output.parent
    paths = {
        "motion_ckpt_path": str(root / "train.ckpt"),
        "curiosity_report_path": str(root / "curiosity_report.txt"),
        "forage_metrics_csv_path": str(root / "forage_metrics.csv"),
        "forage_ckpt_path": str(root / "forage.ckpt"),
        "motion_dashboard_path": str(root / "dashboard.html"),
        "forage_dashboard_path": str(root / "forage_dashboard.html"),
        "evade_metrics_csv_path": str(root / "evade_metrics.csv"),
        "evade_dashboard_path": str(root / "evade_dashboard.html"),
    }
    if stage == "motion":
        paths.update(motion_ckpt_path=args.checkpoint_path, motion_dashboard_path=args.dashboard_path)
    elif stage == "forage":
        paths.update(
            motion_ckpt_path=args.motion_output_checkpoint,
            forage_metrics_csv_path=args.out_metrics_csv,
            forage_ckpt_path=args.out_checkpoint,
            forage_dashboard_path=args.out_dashboard,
        )
    elif stage == "evade":
        paths.update(
            motion_ckpt_path=args.motion_checkpoint_path,
            evade_metrics_csv_path=args.metrics_csv_path,
            evade_dashboard_path=args.dashboard_path,
        )
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    write_master_dashboard(str(output), **paths)
