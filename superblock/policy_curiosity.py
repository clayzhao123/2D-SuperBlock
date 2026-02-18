from __future__ import annotations

import random
from dataclasses import dataclass

from .env import Action, MOVE_DELTAS, SuperblockEnv, TURN_DIRS
from .models import ForwardModel
from .monitor import load_checkpoint
from .utils import position_key


_COORD_SCALE = 40.0


def state_to_center_cell(state: list[int]) -> tuple[int, int]:
    return position_key(state)


def action_to_onehot(action: Action) -> list[float]:
    vec = [0.0] * 10
    vec[action.leg_id] = 1.0
    move_idx = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}[action.move_dir]
    vec[4 + move_idx] = 1.0
    turn_idx = {"CW": 0, "CCW": 1}[action.turn_dir]
    vec[8 + turn_idx] = 1.0
    return vec


def load_forward_model_from_ckpt(path: str) -> ForwardModel:
    payload = load_checkpoint(path)
    model = ForwardModel()
    model.w1 = payload["model"]["w1"]
    model.b1 = payload["model"]["b1"]
    model.w2 = payload["model"]["w2"]
    model.b2 = payload["model"]["b2"]
    return model


class CuriosityMemory:
    def __init__(self) -> None:
        self.visits: dict[tuple[int, int], int] = {}

    def update(self, points_or_state: list[tuple[int, int]] | list[int]) -> None:
        if points_or_state and isinstance(points_or_state[0], int):
            cell = state_to_center_cell(points_or_state)  # type: ignore[arg-type]
        else:
            cell = position_key(points_or_state)  # type: ignore[arg-type]
        self.visits[cell] = self.visits.get(cell, 0) + 1

    def count(self, center_cell: tuple[int, int]) -> int:
        return self.visits.get(center_cell, 0)


@dataclass
class CuriosityPolicy:
    forward_model: ForwardModel
    memory: CuriosityMemory
    epsilon: float = 0.05
    invalid_penalty: float = 1e6

    def candidate_actions(self) -> list[Action]:
        return [Action(0, move_dir, turn_dir) for move_dir in MOVE_DELTAS for turn_dir in TURN_DIRS]

    def _predict_next_state(self, state_t: list[int], action: Action) -> list[int]:
        model_in = [v / _COORD_SCALE for v in state_t] + action_to_onehot(action)
        pred = self.forward_model.predict_batch([model_in])[0]
        return [round(v * _COORD_SCALE) for v in pred]

    def _is_invalid(self, env: SuperblockEnv, action: Action) -> bool:
        _, invalid = env.peek_step(env.points, action)
        return invalid

    def select_action(self, state_t: list[int], env: SuperblockEnv, rng: random.Random) -> Action:
        actions = self.candidate_actions()
        if rng.random() < self.epsilon:
            return rng.choice(actions)

        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            pred_state = self._predict_next_state(state_t, action)
            pred_center = state_to_center_cell(pred_state)
            novelty = 1.0 / (1.0 + self.memory.count(pred_center))
            invalid = self._is_invalid(env, action)
            score = novelty - (self.invalid_penalty if invalid else 0.0)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
