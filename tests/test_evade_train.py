from argparse import Namespace
from pathlib import Path

from superblock.buffer import ReplayBuffer
from superblock.evade_train import run_with_callbacks
from superblock.models import ForwardModel
from superblock.monitor import save_checkpoint


def test_evade_train_callbacks_and_outputs(tmp_path: Path) -> None:
    ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(ckpt),
        day_idx=1,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )

    metrics_path = tmp_path / "evade_metrics.csv"
    dashboard_path = tmp_path / "evade_dashboard.html"

    step_calls = 0
    day_rows: list[dict[str, float]] = []

    def on_step(_day_idx: int, _step_idx: int, _env, _walk) -> None:  # type: ignore[no-untyped-def]
        nonlocal step_calls
        step_calls += 1

    def on_day_end(row: dict[str, float]) -> None:
        day_rows.append(row)

    run_with_callbacks(
        Namespace(
            seed=42,
            days=2,
            steps_per_day=3,
            grass_area=5,
            grass_count=2,
            epsilon=0.1,
            motion_checkpoint_path=str(ckpt),
            dashboard_path=str(dashboard_path),
            metrics_csv_path=str(metrics_path),
        ),
        on_step=on_step,
        on_day_end=on_day_end,
    )

    assert step_calls > 0
    assert len(day_rows) == 2
    assert metrics_path.exists()
    assert dashboard_path.exists()
