"""Reward functions for affinetes environments."""

from __future__ import annotations

from typing import Any


def _compare_outputs_normalized(predicted: str, expected: str) -> bool:
    """Whitespace-normalized output comparison (mirrors affinetes compare_outputs)."""
    return predicted.strip().split() == expected.strip().split()


def trace_correctness_reward(parsed: Any, context: Any) -> float:
    """Correctness for trace env. context = expected_output string."""
    if not isinstance(parsed, dict):
        return 0.0
    answer = parsed.get("answer_text")
    if answer is None:
        return 0.0
    expected = context if isinstance(context, str) else str(context)
    return 1.0 if _compare_outputs_normalized(answer, expected) else 0.0


def logic_correctness_reward(parsed: Any, context: Any) -> float:
    """Correctness for logic env. context = ground_truth dict with game_data + task_type.

    Uses cached verifier instances loaded at module level.
    """
    if not isinstance(parsed, dict):
        return 0.0
    answer = parsed.get("answer_text")
    if answer is None:
        return 0.0
    if not isinstance(context, dict):
        return 0.0
    task_type = context.get("task_type", "")
    verifier = _get_verifier(task_type)
    if verifier is None:
        return 0.0
    game_data = context.get("game_data")
    if game_data is None:
        return 0.0
    try:
        from ._loader import load_logic_verifiers

        _, Data = load_logic_verifiers()
        data = Data.from_json(game_data) if isinstance(game_data, dict) else game_data
        return 1.0 if verifier.verify(data, answer) else 0.0
    except Exception:
        return 0.0


# Lazy-loaded verifier cache
_verifier_cache: dict | None = None


def _get_verifier(task_type: str) -> Any:
    global _verifier_cache
    if _verifier_cache is None:
        from ._loader import load_logic_verifiers

        verifier_classes, _ = load_logic_verifiers()
        _verifier_cache = {name: cls() for name, cls in verifier_classes.items()}
    return _verifier_cache.get(task_type)
