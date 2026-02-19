from superblock.env import Action
from superblock.evade_env import EvadeEnv


def test_grass_hides_superblock_from_vision() -> None:
    env = EvadeEnv(seed=1, grass_area=1, grass_count=0)
    env.reset_day(1)
    env.base_env.points = [(5, 5)]
    env.superhacker_pos = (5, 2)
    env.grass_cells = {(5, 5)}
    assert env._in_vision(env.superhacker_pos, (5, 5)) is False


def test_chase_can_catch_target() -> None:
    env = EvadeEnv(seed=1, grass_area=1, grass_count=0)
    env.reset_day(1)
    env.base_env.points = [(40, 40)]
    env.superhacker_pos = (40, 39)
    env.mode = "chase"
    _state, _img, _invalid, done, caught = env.step(Action(0, "+x", "CW"))
    assert done is True
    assert caught is True
