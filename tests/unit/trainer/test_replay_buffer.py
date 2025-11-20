"""Unit tests for replay buffer implementations."""

from __future__ import annotations

import pytest

from grail.trainer.algorithms.grpo import GRPOGroup, GRPORollout
from grail.trainer.replay_buffer import RecencyWeightedBuffer, create_replay_buffer


def create_mock_group(group_id: str, num_rollouts: int = 5) -> GRPOGroup:
    """Create a mock GRPO group for testing.

    Args:
        group_id: Group identifier
        num_rollouts: Number of rollouts in the group

    Returns:
        Mock GRPOGroup instance
    """
    rollouts = [
        GRPORollout(
            tokens=[1, 2, 3, 4, 5],
            prompt_length=2,
            completion_length=3,
            advantage=float(i - num_rollouts // 2),
            reward=float(i),
            success=i % 2 == 0,
            nonce=i,
            rollout_group=group_id,
            token_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
        )
        for i in range(num_rollouts)
    ]
    return GRPOGroup(group_id=group_id, rollouts=rollouts)


class TestRecencyWeightedBuffer:
    """Tests for RecencyWeightedBuffer implementation."""

    def test_initialization(self) -> None:
        """Test buffer initialization with valid parameters."""
        buffer = RecencyWeightedBuffer(
            max_windows=5,
            recent_window_fraction=0.5,
            decay_factor=0.7,
        )

        stats = buffer.get_stats()
        assert stats["windows"] == 0
        assert stats["total_groups"] == 0
        assert stats["oldest_window"] is None
        assert stats["newest_window"] is None

    def test_initialization_invalid_params(self) -> None:
        """Test buffer initialization with invalid parameters."""
        with pytest.raises(ValueError, match="max_windows must be >= 1"):
            RecencyWeightedBuffer(max_windows=0)

        with pytest.raises(ValueError, match="recent_window_fraction"):
            RecencyWeightedBuffer(max_windows=5, recent_window_fraction=1.5)

        with pytest.raises(ValueError, match="decay_factor"):
            RecencyWeightedBuffer(max_windows=5, decay_factor=-0.1)

    def test_add_single_window(self) -> None:
        """Test adding a single window of groups."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        groups = [create_mock_group(f"group_{i}") for i in range(10)]
        buffer.add_window(window=100, groups=groups)

        stats = buffer.get_stats()
        assert stats["windows"] == 1
        assert stats["total_groups"] == 10
        assert stats["oldest_window"] == 100
        assert stats["newest_window"] == 100

    def test_add_multiple_windows(self) -> None:
        """Test adding multiple windows."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        for window in [100, 110, 120]:
            groups = [create_mock_group(f"w{window}_g{i}") for i in range(5)]
            buffer.add_window(window=window, groups=groups)

        stats = buffer.get_stats()
        assert stats["windows"] == 3
        assert stats["total_groups"] == 15
        assert stats["oldest_window"] == 100
        assert stats["newest_window"] == 120

    def test_eviction_on_capacity(self) -> None:
        """Test oldest window is evicted when capacity is reached."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        # Add 4 windows (should evict first one)
        for window in [100, 110, 120, 130]:
            groups = [create_mock_group(f"w{window}_g{i}") for i in range(5)]
            buffer.add_window(window=window, groups=groups)

        stats = buffer.get_stats()
        assert stats["windows"] == 3
        assert stats["total_groups"] == 15
        assert stats["oldest_window"] == 110  # 100 was evicted
        assert stats["newest_window"] == 130

        # Critical: verify we can sample after eviction (would fail with KeyError bug)
        sampled = buffer.sample_groups(max_groups=10, seed=42)
        assert len(sampled) > 0 and len(sampled) <= 10
        assert all(any(w in g.group_id for w in ["w110_", "w120_", "w130_"]) for g in sampled)

    def test_sample_single_window(self) -> None:
        """Test sampling from a single window."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        groups = [create_mock_group(f"group_{i}") for i in range(20)]
        buffer.add_window(window=100, groups=groups)

        # Sample 10 groups
        sampled = buffer.sample_groups(max_groups=10, seed=42)
        assert len(sampled) == 10

        # Verify determinism
        sampled2 = buffer.sample_groups(max_groups=10, seed=42)
        assert len(sampled2) == 10
        assert [g.group_id for g in sampled] == [g.group_id for g in sampled2]

    def test_sample_multiple_windows_recency_bias(self) -> None:
        """Test recency-weighted sampling across multiple windows."""
        buffer = RecencyWeightedBuffer(
            max_windows=3,
            recent_window_fraction=0.5,
            decay_factor=0.7,
        )

        # Add 3 windows with 100 groups each
        for window in [100, 110, 120]:
            groups = [create_mock_group(f"w{window}_g{i}") for i in range(100)]
            buffer.add_window(window=window, groups=groups)

        # Sample 100 groups (should favor window 120)
        sampled = buffer.sample_groups(max_groups=100, seed=42)
        assert len(sampled) == 100

        # Count samples per window
        counts = {100: 0, 110: 0, 120: 0}
        for group in sampled:
            for window in [100, 110, 120]:
                if f"w{window}_" in group.group_id:
                    counts[window] += 1
                    break

        # Most recent window (120) should have most samples (~50%)
        # Middle window (110) should have ~35%
        # Oldest window (100) should have ~15%
        assert counts[120] > counts[110] > counts[100]
        assert counts[120] >= 40  # At least 40% from most recent

    def test_sample_more_than_available(self) -> None:
        """Test sampling more groups than available."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        groups = [create_mock_group(f"group_{i}") for i in range(10)]
        buffer.add_window(window=100, groups=groups)

        # Request 20 groups but only 10 available
        sampled = buffer.sample_groups(max_groups=20, seed=42)
        assert len(sampled) == 10

    def test_sample_empty_buffer(self) -> None:
        """Test sampling from empty buffer."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        sampled = buffer.sample_groups(max_groups=10, seed=42)
        assert len(sampled) == 0

    def test_clear_before(self) -> None:
        """Test clearing windows before a threshold."""
        buffer = RecencyWeightedBuffer(max_windows=5)

        # Add windows 100, 110, 120, 130
        for window in [100, 110, 120, 130]:
            groups = [create_mock_group(f"w{window}_g{i}") for i in range(5)]
            buffer.add_window(window=window, groups=groups)

        # Clear windows before 120
        buffer.clear_before(window=120)

        stats = buffer.get_stats()
        assert stats["windows"] == 2
        assert stats["total_groups"] == 10
        assert stats["oldest_window"] == 120
        assert stats["newest_window"] == 130

    def test_add_duplicate_window(self) -> None:
        """Test adding same window twice (should update)."""
        buffer = RecencyWeightedBuffer(max_windows=3)

        groups1 = [create_mock_group(f"group_{i}") for i in range(10)]
        buffer.add_window(window=100, groups=groups1)

        groups2 = [create_mock_group(f"new_group_{i}") for i in range(5)]
        buffer.add_window(window=100, groups=groups2)

        stats = buffer.get_stats()
        assert stats["windows"] == 1
        assert stats["total_groups"] == 5  # Updated to new groups

    def test_sample_after_multiple_evictions(self) -> None:
        """Test sampling works correctly after multiple evictions (regression test for KeyError bug)."""
        buffer = RecencyWeightedBuffer(max_windows=2)

        # Add windows sequentially, forcing evictions
        for window in [100, 110, 120, 130, 140, 150]:
            groups = [create_mock_group(f"w{window}_g{i}") for i in range(10)]
            buffer.add_window(window=window, groups=groups)

            # Sample after each addition to ensure buffer state is consistent
            sampled = buffer.sample_groups(max_groups=5, seed=window)
            assert len(sampled) > 0, f"Failed to sample after adding window {window}"

        # Final state should have only last 2 windows
        stats = buffer.get_stats()
        assert stats["windows"] == 2
        assert stats["oldest_window"] == 140
        assert stats["newest_window"] == 150

        # Verify all sampled groups are from valid windows only
        sampled = buffer.sample_groups(max_groups=20, seed=42)
        assert all("w140_" in g.group_id or "w150_" in g.group_id for g in sampled)

    def test_single_window_capacity(self) -> None:
        """Test buffer with max_windows=1 (current production config)."""
        buffer = RecencyWeightedBuffer(max_windows=1)

        # Add first window
        groups1 = [create_mock_group(f"w100_g{i}") for i in range(50)]
        buffer.add_window(window=100, groups=groups1)

        stats = buffer.get_stats()
        assert stats["windows"] == 1
        assert stats["total_groups"] == 50
        assert stats["oldest_window"] == 100
        assert stats["newest_window"] == 100

        # Sample from single window
        sampled = buffer.sample_groups(max_groups=20, seed=42)
        assert len(sampled) == 20
        assert all("w100_" in g.group_id for g in sampled)

        # Add second window → should evict first
        groups2 = [create_mock_group(f"w110_g{i}") for i in range(30)]
        buffer.add_window(window=110, groups=groups2)

        stats = buffer.get_stats()
        assert stats["windows"] == 1
        assert stats["total_groups"] == 30
        assert stats["oldest_window"] == 110
        assert stats["newest_window"] == 110

        # Sample after eviction → should only get w110 groups
        sampled = buffer.sample_groups(max_groups=15, seed=42)
        assert len(sampled) == 15
        assert all("w110_" in g.group_id for g in sampled)
        assert not any("w100_" in g.group_id for g in sampled)

        # Add third window → should evict second
        groups3 = [create_mock_group(f"w120_g{i}") for i in range(40)]
        buffer.add_window(window=120, groups=groups3)

        stats = buffer.get_stats()
        assert stats["windows"] == 1
        assert stats["total_groups"] == 40
        assert stats["oldest_window"] == 120
        assert stats["newest_window"] == 120

        # Sample all available groups (more than available)
        sampled = buffer.sample_groups(max_groups=100, seed=42)
        assert len(sampled) == 40  # Can only get what's available
        assert all("w120_" in g.group_id for g in sampled)


class TestReplayBufferFactory:
    """Tests for replay buffer factory function."""

    def test_create_recency_weighted_buffer(self) -> None:
        """Test creating recency weighted buffer via factory."""
        buffer = create_replay_buffer(
            buffer_type="recency_weighted",
            max_windows=5,
            recent_window_fraction=0.6,
            decay_factor=0.8,
        )

        assert isinstance(buffer, RecencyWeightedBuffer)
        stats = buffer.get_stats()
        assert stats["windows"] == 0

    def test_create_unknown_buffer_type(self) -> None:
        """Test factory with unknown buffer type."""
        with pytest.raises(ValueError, match="Unknown buffer type"):
            create_replay_buffer(
                buffer_type="unknown_type",
                max_windows=5,
            )
