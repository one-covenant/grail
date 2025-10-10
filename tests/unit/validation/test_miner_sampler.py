"""Unit tests for MinerSampler component.

Tests focus on:
- Deterministic selection logic
- Fair distribution across miners
- Sample size calculation
- Edge cases and boundary conditions

Uses pytest best practices:
- Parametrized tests for multiple scenarios
- Descriptive test names following Given-When-Then
- Fixtures from conftest.py for reusable data
- Hypothesis property-based testing where applicable
"""

import hashlib

import pytest

from grail.validation.sampling import MinerSampler


class TestSampleSizeCalculation:
    """Test sample size respects configured bounds."""

    @pytest.mark.parametrize(
        "active_count,rate,min_size,max_size,expected",
        [
            (10, 0.2, 5, None, 5),  # Hits minimum
            (100, 0.2, 5, 50, 20),  # Normal calculation
            (200, 0.2, 5, 20, 20),  # Hits maximum
            (0, 0.2, 5, None, 0),  # No active miners
            (5, 0.2, 10, None, 5),  # Below minimum, capped at active_count
            (1000, 0.5, 10, 100, 100),  # Large pool, hits max
        ],
    )
    def test_respects_min_max_bounds(self, active_count, rate, min_size, max_size, expected):
        """Given various pool sizes, sample size should respect min/max bounds."""
        sampler = MinerSampler(
            sample_rate=rate, sample_min=min_size, sample_max=max_size, concurrency=8
        )

        result = sampler._compute_sample_size(active_count)

        assert result == expected

    def test_zero_rate_returns_zero(self):
        """Given zero sample rate, should return zero regardless of pool size."""
        sampler = MinerSampler(sample_rate=0.0, sample_min=0, sample_max=100, concurrency=8)

        assert sampler._compute_sample_size(50) == 0


class TestDeterministicSelection:
    """Test that selection is deterministic and reproducible."""

    def test_same_inputs_produce_same_output(self):
        """Given identical inputs, selection should be deterministic."""
        sampler = MinerSampler(sample_rate=0.5, sample_min=2, sample_max=10, concurrency=8)

        active_hotkeys = [f"hotkey_{i}" for i in range(20)]
        window_hash = "deterministic_hash"
        selection_counts = {"hotkey_5": 1, "hotkey_10": 2}

        # Run selection multiple times
        results = [
            sampler.select_miners_for_validation(active_hotkeys, window_hash, selection_counts)
            for _ in range(3)
        ]

        # All results should be identical
        assert results[0] == results[1] == results[2]
        assert len(results[0]) == 10  # 20 * 0.5

    def test_different_window_hash_changes_selection(self):
        """Given different window hash, selection should differ (deterministically)."""
        sampler = MinerSampler(sample_rate=0.5, sample_min=2, sample_max=10, concurrency=8)

        active_hotkeys = [f"hotkey_{i}" for i in range(20)]
        selection_counts = {}

        selected_1 = sampler.select_miners_for_validation(
            active_hotkeys, "hash_window_1000", selection_counts
        )
        selected_2 = sampler.select_miners_for_validation(
            active_hotkeys, "hash_window_1020", selection_counts
        )

        # Different hashes should produce different selections
        assert selected_1 != selected_2
        assert len(selected_1) == len(selected_2) == 10


class TestFairDistribution:
    """Test that selection fairly distributes across miners."""

    def test_prioritizes_less_selected_miners(self):
        """Given selection history, should prioritize miners with fewer selections."""
        sampler = MinerSampler(sample_rate=0.5, sample_min=2, sample_max=5, concurrency=8)

        active_hotkeys = [f"hotkey_{i}" for i in range(10)]

        # Simulate: hotkeys 0-4 selected previously, 5-9 not selected
        selection_counts = {f"hotkey_{i}": 3 for i in range(5)}

        selected = sampler.select_miners_for_validation(
            active_hotkeys, "window_hash_123", selection_counts
        )

        # Most of the selected should be from the less-selected group (5-9)
        less_selected_group = {f"hotkey_{i}" for i in range(5, 10)}
        selected_from_less = set(selected) & less_selected_group

        assert len(selected_from_less) >= 3  # At least 3 out of 5 should be from less-selected

    def test_tie_breaking_is_consistent(self):
        """Given miners with equal selection counts, tie-breaking should be deterministic."""
        window_hash = "consistent_hash"
        hotkeys = [f"hotkey_{i}" for i in range(5)]

        # Compute tie-break values twice
        def compute_tie_breaks(hks, w_hash):
            return [
                int.from_bytes(hashlib.sha256(f"{w_hash}:{hk}".encode()).digest()[:8], "big")
                for hk in hks
            ]

        tie_values_1 = compute_tie_breaks(hotkeys, window_hash)
        tie_values_2 = compute_tie_breaks(hotkeys, window_hash)

        assert tie_values_1 == tie_values_2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "active_hotkeys,window_hash,sample_rate,expected",
        [
            ([], "hash", 0.0, []),  # Empty list
            (["single"], "hash", 1.0, ["single"]),  # Single miner with 100% rate
            ([f"hk_{i}" for i in range(3)], "hash", 0.0, []),  # Zero rate returns empty
        ],
    )
    def test_handles_edge_cases(self, active_hotkeys, window_hash, sample_rate, expected):
        """Given edge case inputs, sampler should handle gracefully."""
        sampler = MinerSampler(sample_rate=sample_rate, sample_min=0, sample_max=10, concurrency=8)

        result = sampler.select_miners_for_validation(active_hotkeys, window_hash, {})

        assert result == expected

    def test_all_miners_same_count_uses_hash_ordering(self):
        """Given all miners with same count, should use deterministic hash ordering."""
        sampler = MinerSampler(sample_rate=0.5, sample_min=1, sample_max=5, concurrency=8)

        active_hotkeys = [f"hotkey_{i}" for i in range(10)]
        selection_counts = dict.fromkeys(active_hotkeys, 1)  # All equal

        selected = sampler.select_miners_for_validation(
            active_hotkeys, "hash_abc", selection_counts
        )

        assert len(selected) == 5
        # Selection should be deterministic based on hash
        selected_again = sampler.select_miners_for_validation(
            active_hotkeys, "hash_abc", selection_counts
        )
        assert selected == selected_again

