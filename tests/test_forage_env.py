from superblock.env import Action
from superblock.forage_env import ForageEnv
from superblock.utils import GRID_MAX


def test_food_count_and_bounds() -> None:
    env = ForageEnv(seed=7, food_count=5)
    env.reset_day(day_idx=1)
    assert len(env.food_cells) == 5
    assert all(0 <= x <= GRID_MAX and 0 <= y <= GRID_MAX for x, y in env.food_cells)


def test_hunger_trigger_and_attempt_count() -> None:
    env = ForageEnv(seed=7, hunger_interval=3, hunger_death_steps=50)
    env.reset_day(day_idx=1)
    for _ in range(3):
        env.step(Action(0, "+x", "CW"))
    assert env.hungry is True
    assert env.forage_attempts == 1


def test_hungry_death_sets_done() -> None:
    env = ForageEnv(seed=7, hunger_interval=1, hunger_death_steps=2)
    env.reset_day(day_idx=1)
    env.food_cells = [(40, 40)]
    _, _, _, done1, _ = env.step(Action(0, "+x", "CW"))
    _, _, _, done2, _ = env.step(Action(0, "+x", "CW"))
    assert done1 is False
    assert done2 is True


def test_reach_food_in_hungry_state_resets_hunger_and_adds_success() -> None:
    env = ForageEnv(
        seed=7,
        hunger_interval=1,
        hunger_death_steps=10,
        init_points=[(40, 39), (40, 40), (39, 40), (39, 39)],
        eat_mode="center",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [env.center_cell()]
    env.step(Action(0, "+x", "CW"))
    assert env.forage_success == 1
    assert env.hungry is False
    assert env.steps_since_last_meal == 0
    assert env.hungry_steps == 0


def test_overlap_mode_eats_food_on_any_occupied_cell() -> None:
    env = ForageEnv(
        seed=7,
        hunger_interval=1,
        hunger_death_steps=10,
        init_points=[(10, 10), (11, 10), (11, 11), (10, 11)],
        eat_mode="overlap",
        occupy_mode="shape",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [(11, 11)]

    env.step(Action(0, "+x", "CW"))

    assert env.forage_success == 1
    assert env.hungry is False


def test_overlap_mode_can_eat_odd_coordinate_food() -> None:
    env = ForageEnv(
        seed=7,
        hunger_interval=1,
        hunger_death_steps=10,
        init_points=[(10, 10), (11, 10), (11, 11), (10, 11)],
        eat_mode="overlap",
        occupy_mode="shape",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [(11, 10)]

    env.step(Action(0, "+x", "CW"))

    assert env.forage_success == 1


def test_eating_food_removes_food_from_map() -> None:
    env = ForageEnv(
        seed=7,
        hunger_interval=1,
        hunger_death_steps=10,
        init_points=[(10, 10), (11, 10), (11, 11), (10, 11)],
        eat_mode="overlap",
        occupy_mode="shape",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [(11, 11), (30, 30)]

    _, _, _, _, ate_food = env.step(Action(0, "+x", "CW"))

    assert ate_food is True
    assert (11, 11) not in env.food_cells
    assert (30, 30) in env.food_cells


def test_default_occupy_mode_is_single_cell() -> None:
    env = ForageEnv(init_points=[(10, 10), (11, 10), (11, 11), (10, 11)])
    env.reset_day(day_idx=1)
    assert env.occupied_cells() == [env.center_cell()]


def test_all_edges_vision_can_see_food_near_corner() -> None:
    env = ForageEnv(
        init_points=[(10, 10), (11, 10), (11, 11), (10, 11)],
        vision_from="all_edges",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [(13, 10)]
    assert env.observe_visible_food(vision_radius=2) == [(13, 10)]


def test_center_vision_cannot_see_food_only_near_corner() -> None:
    env = ForageEnv(
        init_points=[(10, 10), (11, 10), (11, 11), (10, 11)],
        vision_from="center",
    )
    env.reset_day(day_idx=1)
    env.food_cells = [(13, 10)]
    assert env.observe_visible_food(vision_radius=2) == []


def test_spawn_on_hunger_adds_food_near_center() -> None:
    env = ForageEnv(
        seed=7,
        food_count=0,
        hunger_interval=1,
        hunger_death_steps=20,
        food_spawn_mode="spawn_on_hunger",
        food_spawn_radius=4,
        max_food_on_map=3,
    )
    env.reset_day(day_idx=1)
    assert len(env.food_cells) == 0

    env.step(Action(0, "+x", "CW"))

    assert env.hungry is True
    assert len(env.food_cells) >= 1
