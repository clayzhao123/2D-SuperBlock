import csv

from superblock.buffer import ReplayBuffer
from superblock.forage_train import build_parser, run, run_with_callbacks
from superblock.models import ForwardModel
from superblock.monitor import save_checkpoint


def test_forage_train_parser_defaults_to_no_motion_training() -> None:
    args = build_parser().parse_args([])
    assert args.motion_epochs_per_day == 0
    assert args.food_spawn_mode == "static"
    assert args.food_spawn_radius == 8
    assert args.max_food_on_map == 6
    assert args.policy == "heuristic"
    assert args.eat_mode == "overlap"


def test_forage_train_outputs_forage_metrics_without_motion_columns(tmp_path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=0,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )

    args = build_parser().parse_args(
        [
            "--days",
            "1",
            "--steps-per-day",
            "8",
            "--food-count",
            "1",
            "--hunger-interval",
            "2",
            "--hunger-death-steps",
            "20",
            "--vision-radius",
            "20",
            "--motion-checkpoint-path",
            str(motion_ckpt),
            "--out-checkpoint",
            str(tmp_path / "forage.ckpt"),
            "--out-metrics-csv",
            str(tmp_path / "forage_metrics.csv"),
            "--out-dashboard",
            str(tmp_path / "forage_dashboard.html"),
            "--out-attempts-csv",
            str(tmp_path / "forage_attempts.csv"),
        ]
    )
    run(args)

    with open(tmp_path / "forage_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert "hungry_success_rate_today" in row
    assert "hungry_latency_mean_today" in row
    assert "steps_to_first_food_seen_today" in row
    assert "food_seen_flag_today" in row
    assert "food_count_on_map_today" in row
    assert "unreachable_food_seen_today" in row
    assert "motion_score" not in row

    with open(tmp_path / "forage_dashboard.html", encoding="utf-8") as f:
        html = f.read()
    assert "累计觅食成功率" in html
    assert "累计平均觅食耗时 hungry→eat" in html
    assert "运动模型分数（仅在显式启用 motion 训练时显示）" not in html

    with open(tmp_path / "forage_attempts.csv", newline="", encoding="utf-8") as f:
        attempt_rows = list(csv.DictReader(f))
    assert all(key in attempt_rows[0] for key in ["day_idx", "attempt_idx", "start_center", "target_food", "success", "latency_steps"])


def test_forage_train_callbacks_receive_step_and_day_events(tmp_path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=0,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )

    args = build_parser().parse_args(
        [
            "--days",
            "1",
            "--steps-per-day",
            "4",
            "--food-count",
            "1",
            "--hunger-interval",
            "2",
            "--hunger-death-steps",
            "20",
            "--vision-radius",
            "3",
            "--motion-checkpoint-path",
            str(motion_ckpt),
            "--out-checkpoint",
            str(tmp_path / "forage.ckpt"),
            "--out-metrics-csv",
            str(tmp_path / "forage_metrics.csv"),
            "--out-dashboard",
            str(tmp_path / "forage_dashboard.html"),
        ]
    )

    step_events: list[tuple[int, int]] = []
    day_events: list[dict[str, float]] = []

    def on_step(day_idx: int, step_idx: int, _env, _day_path) -> None:
        step_events.append((day_idx, step_idx))

    def on_day_end(metrics: dict[str, float]) -> None:
        day_events.append(metrics)

    run_with_callbacks(args, on_step=on_step, on_day_end=on_day_end)

    assert step_events
    assert day_events
    assert int(day_events[0]["day_idx"]) == 1


def test_forage_train_qlearn_saves_q_table(tmp_path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=0,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )

    out_ckpt = tmp_path / "forage.ckpt"
    args = build_parser().parse_args(
        [
            "--policy",
            "qlearn",
            "--days",
            "1",
            "--steps-per-day",
            "8",
            "--food-count",
            "1",
            "--hunger-interval",
            "1",
            "--hunger-death-steps",
            "20",
            "--vision-radius",
            "4",
            "--motion-checkpoint-path",
            str(motion_ckpt),
            "--out-checkpoint",
            str(out_ckpt),
            "--out-metrics-csv",
            str(tmp_path / "forage_metrics.csv"),
            "--out-dashboard",
            str(tmp_path / "forage_dashboard.html"),
        ]
    )
    run(args)

    import pickle

    with open(out_ckpt, "rb") as f:
        payload = pickle.load(f)
    assert payload["policy"] == "qlearn"
    assert isinstance(payload["q_table"], dict)
