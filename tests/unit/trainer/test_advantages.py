"""Tests for trainer-side advantage estimation strategies."""

from __future__ import annotations

import pytest

from grail.trainer.advantages import compute_advantages

# ── GRPO (zero-mean, variance-normalized) ────────────────────────────


def test_grpo_zero_mean() -> None:
    advs = compute_advantages([1.0, 2.0, 3.0, 4.0, 5.0], estimator="grpo")
    assert len(advs) == 5
    assert abs(sum(advs)) < 1e-6
    # Default estimator should be grpo
    assert compute_advantages([1.0, 2.0, 3.0]) == compute_advantages(
        [1.0, 2.0, 3.0], estimator="grpo"
    )


def test_grpo_variance_normalized() -> None:
    advs = compute_advantages([1.0, 2.0, 3.0, 4.0], estimator="grpo")
    variance = sum(a * a for a in advs) / len(advs)
    assert abs(variance - 1.0) < 1e-6


def test_grpo_uniform_rewards() -> None:
    advs = compute_advantages([3.0, 3.0, 3.0], estimator="grpo")
    assert all(abs(a) < 1e-6 for a in advs)


def test_grpo_empty() -> None:
    assert compute_advantages([], estimator="grpo") == []


# ── DR-GRPO (zero-mean, no variance normalization) ──────────────────


def test_dr_grpo_preserves_magnitude() -> None:
    advs = compute_advantages([0.0, 10.0], estimator="dr_grpo")
    # mean = 5.0, so advantages should be [-5.0, 5.0] (not normalized by std)
    assert abs(advs[0] - (-5.0)) < 1e-6
    assert abs(advs[1] - 5.0) < 1e-6


def test_dr_grpo_uniform() -> None:
    advs = compute_advantages([2.0, 2.0, 2.0], estimator="dr_grpo")
    assert all(abs(a) < 1e-6 for a in advs)


# ── DAPO (rank-based) ───────────────────────────────────────────────


def test_dapo_rank_ordering() -> None:
    advs = compute_advantages([10.0, 20.0, 30.0, 40.0], estimator="dapo")
    for i in range(len(advs) - 1):
        assert advs[i] < advs[i + 1]


def test_dapo_ties() -> None:
    advs = compute_advantages([1.0, 2.0, 2.0, 3.0], estimator="dapo")
    assert abs(advs[1] - advs[2]) < 1e-6


def test_dapo_range() -> None:
    advs = compute_advantages([1.0, 2.0, 3.0], estimator="dapo")
    # Lowest rank maps to -1.0, highest to 1.0
    assert abs(advs[0] - (-1.0)) < 1e-6
    assert abs(advs[2] - 1.0) < 1e-6


# ── Dispatch / cross-cutting ────────────────────────────────────────


def test_unknown_estimator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        compute_advantages([1.0, 2.0], estimator="ppo")


def test_all_estimators_produce_zero_mean() -> None:
    rewards = [0.1, 0.5, 0.3, 0.9, 0.2]
    for est in ("grpo", "dr_grpo", "dapo"):
        advs = compute_advantages(rewards, estimator=est)
        assert abs(sum(advs)) < 1e-6, f"{est} advantages don't sum to 0"
