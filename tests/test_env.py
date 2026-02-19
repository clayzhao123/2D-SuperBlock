from superblock.env import Action, SuperblockEnv


def test_single_cell_moves_one_step() -> None:
    env = SuperblockEnv()
    before = list(env.points)
    _, _, invalid = env.step(Action(0, "+x", "CW"))
    assert invalid is False
    assert env.points == [(before[0][0] + 1, before[0][1])]


def test_invalid_move_keeps_position() -> None:
    env = SuperblockEnv(init_points=[(40, 40)])
    before = list(env.points)
    _, _, invalid = env.step(Action(0, "+x", "CW"))
    assert invalid is True
    assert env.points == before
