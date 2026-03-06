"""GRPO advantage estimation."""

from __future__ import annotations


def compute_advantages(rewards: list[float]) -> list[float]:
    """GRPO advantages: zero-mean within group, variance-normalized."""
    n = len(rewards)
    if n == 0:
        return []
    mean_reward = sum(rewards) / n
    centered = [r - mean_reward for r in rewards]
    std = (sum(a * a for a in centered) / n) ** 0.5
    denom = max(std, 1e-8)
    return [a / denom for a in centered]
