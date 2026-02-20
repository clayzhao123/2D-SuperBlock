from superblock.buffer import ReplayBuffer, Transition
from superblock.env import SuperblockEnv
from superblock.models import ForwardModel
from superblock.train import action_to_onehot, parse_visible_cells, train_night


def test_parse_visible_cells() -> None:
    cells = parse_visible_cells("19:19,20:21")
    assert cells == [(19, 19), (20, 21)]


def test_parse_visible_cells_empty_defaults_center() -> None:
    assert parse_visible_cells("") == [(19, 19)]


def test_train_night_handles_single_cell_state_dimension() -> None:
    env = SuperblockEnv(init_points=[(2, 2)])
    model = ForwardModel(seed=42)
    buffer = ReplayBuffer()
    rng = __import__("random").Random(42)

    state_t, img_t = env.reset()
    for _ in range(8):
        action = env.sample_action(rng)
        state_tp1, img_tp1, invalid = env.step(action)
        buffer.add(
            Transition(
                state_t=state_t.copy(),
                action_vec=action_to_onehot(action),
                invalid_move=invalid,
                img_t=[row[:] for row in img_t],
                state_tp1=state_tp1.copy(),
                img_tp1=[row[:] for row in img_tp1],
            )
        )
        state_t, img_t = state_tp1, img_tp1

    pos_mse, exact_acc, near_acc = train_night(model, buffer, batch_size=4, lr=1e-2, epochs=1, seed=42)

    assert pos_mse >= 0.0
    assert 0.0 <= exact_acc <= 1.0
    assert 0.0 <= near_acc <= 1.0
