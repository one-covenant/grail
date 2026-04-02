"""Advantage estimation strategies for GRPO training.

Computes per-rollout advantages from raw rewards within a group. Orthogonal to
the loss aggregation variant (GRPO_VARIANT), which controls how per-token losses
are normalized.
"""

from __future__ import annotations


def compute_advantages(rewards: list[float], estimator: str = "grpo") -> list[float]:
    """Compute advantages from rewards using the selected estimation strategy.

    Args:
        rewards: Raw rewards for each rollout in a group.
        estimator: Strategy name. One of: "grpo", "dr_grpo", "dapo".

    Returns:
        List of advantage values (same length as rewards).

    Raises:
        ValueError: If estimator name is unknown.
    """
    if not rewards:
        return []

    _dispatch = {
        "grpo": _grpo_advantages,
        "dr_grpo": _dr_grpo_advantages,
        "dapo": _dapo_advantages,
    }
    fn = _dispatch.get(estimator)
    if fn is None:
        raise ValueError(
            f"Unknown advantage estimator '{estimator}'. Must be one of {sorted(_dispatch.keys())}"
        )
    return fn(rewards)


def _grpo_advantages(rewards: list[float]) -> list[float]:
    """GRPO: zero-mean, variance-normalized advantages."""
    n = len(rewards)
    mean_reward = sum(rewards) / n
    centered = [r - mean_reward for r in rewards]
    std = (sum(a * a for a in centered) / n) ** 0.5
    denom = max(std, 1e-8)
    return [a / denom for a in centered]


def _dr_grpo_advantages(rewards: list[float]) -> list[float]:
    """DR-GRPO: zero-mean, no variance normalization.

    Preserves reward magnitude signal that standard GRPO discards.
    """
    n = len(rewards)
    mean_reward = sum(rewards) / n
    return [r - mean_reward for r in rewards]


def _dapo_advantages(rewards: list[float]) -> list[float]:
    """DAPO: rank-based advantages.

    Converts rewards to fractional ranks (average rank for ties), then
    normalizes to [-1, 1] range. Robust to outlier rewards since only
    relative ordering matters.
    """
    n = len(rewards)
    # Compute fractional ranks (average rank for ties)
    sorted_indices = sorted(range(n), key=lambda i: rewards[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and rewards[sorted_indices[j]] == rewards[sorted_indices[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j
    # Normalize to [-1, 1] range
    max_rank = n - 1 if n > 1 else 1
    return [2.0 * r / max_rank - 1.0 for r in ranks]
