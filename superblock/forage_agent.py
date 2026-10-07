"""Compatibility imports; maintain rule-based foraging in agents/heuristic.py."""

from .agents.heuristic import FoodMemory, ForagePolicy, _min_food_distance
from .agents.curiosity import CuriosityMemory, CuriosityPolicy

__all__ = ["FoodMemory", "ForagePolicy", "CuriosityMemory", "CuriosityPolicy"]
