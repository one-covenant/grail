"""Rubric adapters for computing rewards in environments.

Provides a thin adapter over the existing RewardVector so environments can
produce a decomposed component dict alongside the scalar reward at each step.
"""

from __future__ import annotations

from typing import Any

from .base import RewardVector
from .core import Rubric


class RewardVectorRubric(Rubric):
    """Adapter that uses RewardVector to compute reward and components.

    Step shaping defaults to 0.0; the typical use is single-turn where the
    terminal step returns the full scalar reward and component breakdown.
    """

    def __init__(self, reward_vector: RewardVector) -> None:
        self._rv = reward_vector

    def step_reward(
        self, *, parsed: Any, context: Any, turn_index: int
    ) -> tuple[float, dict[str, float]]:
        # Components from individual rewards
        try:
            # If parsed is raw completion text, RewardVector will internally parse via its parser
            completion = parsed if isinstance(parsed, str) else parsed
            comps_list = self._rv.compute_individual_rewards(completion, context)
            components = {f"r{i}": float(v) for i, v in enumerate(comps_list)}
            total = float(self._rv.compute_reward(completion, context))
            return total, components
        except Exception:
            # Robust default on parser errors
            return 0.0, {}
