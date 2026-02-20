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


def test_hungry_eat_resets_hunger_state() -> None:
    env = EvadeEnv(
        seed=1,
        grass_area=1,
        grass_count=0,
        food_count=1,
        hunger_interval=1,
        hunger_death_steps=2,
    )
    env.reset_day(1)
    env.food_cells = [(3, 2)]
    _state, _img, _invalid, done, caught = env.step(Action(0, "+x", "CW"))

    assert done is False
    assert caught is False
    assert env.last_ate_food is True
    assert env.hungry is False
    assert env.forage_attempts == 1
    assert env.forage_success == 1


def test_hunger_can_kill_without_hacker() -> None:
    env = EvadeEnv(
        seed=1,
        grass_area=1,
        grass_count=0,
        food_count=0,
        hunger_interval=1,
        hunger_death_steps=1,
    )
    env.reset_day(1)
    _state, _img, _invalid, done, caught = env.step(Action(0, "+x", "CW"))

    assert done is True
    assert caught is False
    assert env.last_hunger_death is True
