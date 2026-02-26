"""Shared reward components for environments.

This module contains env-agnostic reward functions that can be reused across
different environments (GSM8K, SAT, CODE, etc.). Each environment can compose
these shared rewards with env-specific correctness functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .base import Parser, RewardVector


def thinking_format_reward(parsed: Any, context: Any) -> float:
    """Reward for having thinking tags.

    Checks for thinking tag blocks in completion (mode-dependent tags).
    Returns 0.5 if thinking block detected, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0
    return 0.5 if parsed.get("has_thinking") else 0.0


def answer_format_reward(parsed: Any, context: Any) -> float:
    """Reward for having answer tags.

    Checks for <SOLUTION> blocks in completion.
    Returns 0.3 if answer block detected, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0
    return 0.3 if parsed.get("has_answer") else 0.0


def soft_format_reward(parsed: Any, context: Any) -> float:
    """Reward for having any answer formatting.

    Checks if answer is present but doesn't validate strict format.
    Returns 0.2 if answer detected, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0
    return 0.2 if parsed.get("has_answer") else 0.0


def no_trailing_reward(parsed: Any, context: Any) -> float:
    """Penalize trailing text after answer.

    Rewards concise answers: max 0.2 when no trailing text,
    decreases by 0.001 per trailing character.
    Returns max(0.0, 0.2 - 0.001 * trailing_chars).
    """
    if not isinstance(parsed, dict):
        return 0.0
    if not parsed.get("has_answer"):
        return 0.0
    trailing = int(parsed.get("trailing_after_answer", 0))
    return max(0.0, 0.2 - 0.001 * trailing)


def correctness_reward(parsed: Any, context: Any) -> float:
    """Stub correctness reward.

    This is overridden by env-specific implementations.
    Returns 0.0 by default.
    """
    return 0.0


def create_thinking_reward_vector(
    correctness_weight: float = 0.7,
    thinking_weight: float = 0.15,
    answer_weight: float = 0.1,
    no_trailing_weight: float = 0.05,
    parser: Parser | None = None,
) -> RewardVector:
    """Create reward vector with thinking + answer format components.

    Used by both SATEnv and GSM8KEnv as base factory.
    Each env overrides the correctness function with env-specific logic.

    Args:
        correctness_weight: Weight for correctness component (default 0.7)
        thinking_weight: Weight for thinking format (default 0.15)
        answer_weight: Weight for answer format (default 0.1)
        no_trailing_weight: Weight for penalizing trailing text (default 0.05)
        parser: Optional parser for preprocessing completions

    Returns:
        RewardVector combining all components with specified weights.
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            correctness_reward,
            thinking_format_reward,
            answer_format_reward,
            no_trailing_reward,
        ],
    )
    weights = [correctness_weight, thinking_weight, answer_weight, no_trailing_weight]

    return RewardVector(
        reward_functions,
        weights,
        parser=parser,
        bounds=[
            (0.0, 1.0),  # correctness
            (0.0, 0.5),  # thinking
            (0.0, 0.3),  # answer
            (0.0, 0.2),  # no_trailing
        ],
    )
