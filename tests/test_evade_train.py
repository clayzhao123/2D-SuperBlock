import pickle
from argparse import Namespace
from pathlib import Path

from superblock.buffer import ReplayBuffer
from superblock.evade_train import (
    _choose_grass_escape_action,
    _resolve_motion_checkpoint_path,
    resolve_motion_checkpoint_path,
    run_with_callbacks,
)
from superblock.models import ForwardModel
from superblock.monitor import save_checkpoint
from superblock.policy_curiosity import CuriosityMemory, CuriosityPolicy
from superblock.evade_env import EvadeEnv


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


def test_resolve_motion_checkpoint_path_from_forage_ckpt(tmp_path: Path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=1,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )
    forage_ckpt = tmp_path / "forage.ckpt"
    forage_ckpt.write_bytes(pickle.dumps({"motion_checkpoint_path": "train.ckpt"}))

    resolved = _resolve_motion_checkpoint_path(str(forage_ckpt))

    assert resolved == str(motion_ckpt)


def test_choose_grass_escape_action_prefers_closer_grass() -> None:
    env = EvadeEnv(seed=1, grass_area=1, grass_count=0)
    state, _ = env.reset_day(1)
    env.base_env.points = [(10, 10)]
    state = env.base_env.get_state()
    env.superhacker_pos = (10, 12)
    env.grass_cells = {(10, 9)}

    policy = CuriosityPolicy(forward_model=ForwardModel(), memory=CuriosityMemory(), epsilon=0.0)
    action = _choose_grass_escape_action(
        state=state,
        env=env,
        policy=policy,
        memory_grass={(10, 9)},
        rng=__import__("random").Random(1),
    )

    assert action is not None
    next_points, invalid = env.base_env.peek_step(env.base_env.points, action)
    assert invalid is False
    assert abs(next_points[0][0] - 10) + abs(next_points[0][1] - 9) <= 1


def test_public_resolve_motion_checkpoint_path_alias(tmp_path: Path) -> None:
    motion_ckpt = tmp_path / "train.ckpt"
    save_checkpoint(
        str(motion_ckpt),
        day_idx=1,
        model=ForwardModel(),
        buffer=ReplayBuffer(),
        history=[],
        visible_cells=[(19, 19)],
    )

    assert resolve_motion_checkpoint_path(str(motion_ckpt)) == str(motion_ckpt)
