"""GRAIL Environments - Step-only RL environments for various tasks.

Exports:
- Core API: MultiTurnEnv, SingleTurnEnv, types
- SAT: problem generator, parser, reward, env (all from sat_env)
- GSM8K: dataset-backed env
- Loop: AgentEnvLoop, GRPORollout
- Legacy: Parser, RewardVector
"""

from .base import Parser, RewardVector  # noqa: F401
from .core import (  # noqa: F401
    ChatMessage,
    MultiTurnEnv,
    Observation,
    RewardBreakdown,
    Rubric,
    SingleTurnEnv,
    TaskSource,
)
from .gsm8k_env import GSM8KEnv  # noqa: F401
from .loop import AgentEnvLoop, GRPORollout  # noqa: F401
from .providers import GSM8KTaskSource, SATTaskSource  # noqa: F401
from .rubric import RewardVectorRubric  # noqa: F401
from .sat_env import (  # noqa: F401
    SATEnv,
    SATParser,
    SATProblem,
    create_sat_prompt,
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
    # Base classes (legacy)
    "Parser",
    "RewardVector",
    # Core step-only API
    "ChatMessage",
    "Observation",
    "RewardBreakdown",
    "TaskSource",
    "Rubric",
    "MultiTurnEnv",
    "SingleTurnEnv",
    # Providers and rubrics
    "SATTaskSource",
    "GSM8KTaskSource",
    "RewardVectorRubric",
    # Environments
    "SATEnv",
    "GSM8KEnv",
    # SAT public API (validators use these)
    "SATProblem",
    "SATParser",
    "generate_sat_problem",
    "sat_correctness_reward",
    "create_sat_reward_vector",
    "create_sat_prompt",
    # Loop and rollouts
    "AgentEnvLoop",
    "GRPORollout",
    # Helper functions
    "get_sat_reward_bounds",
]
