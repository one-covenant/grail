"""GRAIL Environments - Scalable RL environments for various tasks."""

# Import base classes
from .base import Parser, RewardVector  # noqa: F401

# Import SAT-specific implementations
from .sat import (  # noqa: F401
    SATParser,
    SATProblem,
    SATRolloutGenerator,
    create_sat_reward_vector,
    generate_sat_problem,
    sat_correctness_reward,
)


def get_sat_reward_bounds() -> tuple[float, float]:
    """Return SAT reward bounds or permissive defaults on failure.

    Used by validators to configure reward bounds for SAT validation.

    Returns:
        Tuple of (low, high) reward bounds as floats.
        Returns (-inf, inf) if bounds cannot be determined.
    """
    try:
        sat_rv = create_sat_reward_vector()
        low, high = sat_rv.reward_bounds()
        return float(low), float(high)
    except Exception:
        return float("-inf"), float("inf")


__all__ = [
    # Base classes
    "Parser",
    "RewardVector",
    # SAT implementations
    "SATProblem",
    "SATParser",
    "generate_sat_problem",
    "sat_correctness_reward",
    "create_sat_reward_vector",
    "SATRolloutGenerator",
    # Helper functions
    "get_sat_reward_bounds",
]
