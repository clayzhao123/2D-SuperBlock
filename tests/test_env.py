from superblock.env import Action, SuperblockEnv
from superblock.utils import shoelace_area


def test_area_always_one_after_steps() -> None:
    env = SuperblockEnv()
    actions = [
        Action(0, "+x", "CW"),
        Action(1, "+y", "CCW"),
        Action(2, "-x", "CW"),
        Action(3, "-y", "CCW"),
    ]
    for action in actions * 8:
        env.step(action)
        assert shoelace_area(env.points) == 1.0


def test_valid_translate_moves_all_points_exactly_one() -> None:
    env = SuperblockEnv()
    before = list(env.points)
    _, _, invalid = env.step(Action(0, "+x", "CW"))
    assert invalid is False

    # Undo label rotation to compare geometric translation only.
    rotated_back = [env.points[1], env.points[2], env.points[3], env.points[0]]
    deltas = [(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(rotated_back, before)]
    assert deltas == [(1, 0)] * 4


def test_invalid_translate_keeps_position_but_changes_orientation() -> None:
    init = [(40, 39), (40, 40), (39, 40), (39, 39)]
    env = SuperblockEnv(init_points=init)
    before = list(env.points)

    _, _, invalid = env.step(Action(1, "+x", "CW"))

    assert invalid is True
    assert set(env.points) == set(before)
    assert env.points != before
