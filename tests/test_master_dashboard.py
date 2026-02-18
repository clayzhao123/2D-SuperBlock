from pathlib import Path

from superblock.buffer import ReplayBuffer
from superblock.master_dashboard import write_master_dashboard
from superblock.models import ForwardModel
from superblock.monitor import save_checkpoint


def test_master_dashboard_renders_sections_and_fallbacks(tmp_path: Path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=1,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[
            {
                "day_idx": 1.0,
                "invalid_move_ratio": 0.1,
                "pos_mse": 0.02,
                "exact_acc": 0.4,
                "near_acc": 0.6,
                "score": 0.7,
            }
        ],
        visible_cells=[(19, 19)],
    )

    forage_csv = tmp_path / "forage_metrics.csv"
    forage_csv.write_text(
        "day_idx,hungry_success_rate_total,hungry_latency_mean_total,deaths_total\n"
        "1,0.5,12,0\n",
        encoding="utf-8",
    )

    out = tmp_path / "master_dashboard.html"
    write_master_dashboard(
        out_path=str(out),
        motion_ckpt_path=str(motion_ckpt),
        curiosity_report_path=str(tmp_path / "missing_curiosity.txt"),
        forage_metrics_csv_path=str(forage_csv),
        forage_ckpt_path=str(tmp_path / "missing_forage.ckpt"),
        motion_dashboard_path="artifacts/dashboard.html",
        forage_dashboard_path="artifacts/forage_dashboard.html",
    )

    html = out.read_text(encoding="utf-8")
    assert "Superblock Master Dashboard" in html
    assert "Motion" in html
    assert "Curiosity" in html
    assert "Forage" in html
    assert "未运行 explore" in html
    assert "dashboard.html" in html
    assert "forage_dashboard.html" in html


def test_master_dashboard_parses_curiosity_report(tmp_path: Path) -> None:
    report = tmp_path / "curiosity_report.txt"
    report.write_text("visited_cells=77 total_cells=1681 coverage=0.0458\n", encoding="utf-8")

    out = tmp_path / "master_dashboard.html"
    write_master_dashboard(
        out_path=str(out),
        motion_ckpt_path=str(tmp_path / "missing_train.ckpt"),
        curiosity_report_path=str(report),
        forage_metrics_csv_path=str(tmp_path / "missing_forage.csv"),
        forage_ckpt_path=str(tmp_path / "missing_forage.ckpt"),
    )

    html = out.read_text(encoding="utf-8")
    assert "visited_cells=77" in html
    assert "coverage=0.0458" in html
