"""Compatibility imports; maintain shared exploration in agents/curiosity.py."""

from .agents.curiosity import (
    CuriosityMemory,
    CuriosityPolicy,
    _COORD_SCALE,
    action_to_onehot,
    load_forward_model_from_ckpt,
    state_to_center_cell,
)

__all__ = [
    "CuriosityMemory", "CuriosityPolicy", "action_to_onehot",
    "load_forward_model_from_ckpt", "state_to_center_cell",
]
