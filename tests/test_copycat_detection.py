"""Test copycat detection for validator pipelines.

Verifies that the copycat tracker correctly identifies miners submitting
identical or overlapping completions within windows and across intervals.
"""

from __future__ import annotations

from collections import Counter

import pytest

from grail.validation.copycat import (
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_WINDOW_THRESHOLD,
    CopycatTracker,
    compute_completion_digest,
)


def generate_sat_completion_tokens(
    prompt_tokens: list[int], seed: int, unique: bool = False
) -> list[int]:
    """Generate realistic SAT completion tokens."""
    import random

    rng = random.Random(seed)
    length = rng.randint(20, 50)
    if unique:
        completion = [rng.randint(0, 50000) for _ in range(length)]
    else:
        base = 1000 + (seed % 100)
        completion = [base + i for i in range(length)]
    return prompt_tokens + completion


def make_rollout_data(
    prompt_tokens: list[int], seeds: list[int], unique_flags: list[bool]
) -> list[str]:
    """Generate multiple rollout digests from seeds."""
    prompt_len = len(prompt_tokens)
    digests = []
    for seed, is_unique in zip(seeds, unique_flags):
        tokens = generate_sat_completion_tokens(prompt_tokens, seed, is_unique)
        digest = compute_completion_digest({"tokens": tokens}, {"prompt_length": prompt_len})
        if digest:
            digests.append(digest)
    return digests


class TestCompletionDigest:
    """Test digest computation for copycat detection."""

    def test_digest_basic_properties(self) -> None:
        """Digest is deterministic and slices prompt correctly."""
        rollout = {"prompt_length": 2}
        # Deterministic
        commit = {"tokens": [1, 2, 3, 4, 5]}
        d1 = compute_completion_digest(commit, rollout)
        d2 = compute_completion_digest(commit, rollout)
        assert d1 == d2 and d1 is not None
        # Different completions → different digests
        commit2 = {"tokens": [1, 2, 3, 9, 10]}
        assert d1 != compute_completion_digest(commit2, rollout)
        # Same completion, different prompts → same digest
        commit3 = {"tokens": [99, 98, 3, 4, 5]}
        assert d1 == compute_completion_digest(commit3, rollout)

    def test_digest_edge_cases(self) -> None:
        """Handles missing metadata and empty tokens."""
        # Missing prompt_length
        assert compute_completion_digest({"tokens": [1, 2, 3]}, {}) is not None
        # Empty tokens
        assert compute_completion_digest({"tokens": []}, {"prompt_length": 0}) is None


class TestCopycatTracker:
    """Test copycat tracking across windows and intervals."""

    @pytest.fixture
    def tracker(self) -> CopycatTracker:
        """Fresh tracker for each test."""
        return CopycatTracker()

    def test_no_overlap_no_violations(self, tracker: CopycatTracker) -> None:
        """No violations when miners have unique completions."""
        tracker.reset_interval(0)
        miner_rollouts = {
            "miner_a": (Counter({"digest_1": 5}), 5),
            "miner_b": (Counter({"digest_2": 5}), 5),
            "miner_c": (Counter({"digest_3": 5}), 5),
        }
        w_cheat, w_det, i_cheat, i_det, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert len(w_cheat) == 0
        assert len(w_det) == 0
        assert len(i_cheat) == 0
        assert len(i_det) == 0

    def test_window_threshold_violation(self, tracker: CopycatTracker) -> None:
        """Flags miners exceeding window threshold."""
        tracker.reset_interval(0)
        # miner_a: 10 rollouts, 6 shared with miner_b
        # miner_b: 10 rollouts, 6 shared with miner_a
        # Ratio: 6 / min(10, 10) = 0.6 > COPYCAT_WINDOW_THRESHOLD (0.5)
        miner_rollouts = {
            "miner_a": (Counter({"shared": 6, "unique_a": 4}), 10),
            "miner_b": (Counter({"shared": 6, "unique_b": 4}), 10),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert "miner_a" in w_cheat
        assert "miner_b" in w_cheat
        assert len(w_det) == 1
        assert w_det[0].shared == 6
        assert w_det[0].ratio == 0.6

    def test_window_below_threshold_no_violation(self, tracker: CopycatTracker) -> None:
        """No violation when overlap below window threshold."""
        tracker.reset_interval(0)
        # Ratio: 4 / min(10, 10) = 0.4 < 0.5
        miner_rollouts = {
            "miner_a": (Counter({"shared": 4, "unique_a": 6}), 10),
            "miner_b": (Counter({"shared": 4, "unique_b": 6}), 10),
        }
        w_cheat, w_det, _, _, w_all, _ = tracker.ingest_window(100, miner_rollouts)
        assert len(w_cheat) == 0
        assert len(w_det) == 0
        # But should appear in all_pairs
        assert len(w_all) == 1
        assert w_all[0].ratio == 0.4

    def test_interval_accumulation(self, tracker: CopycatTracker) -> None:
        """Interval-level detection accumulates across windows."""
        tracker.reset_interval(0)
        # Accumulate windows with 8 shared each (0.8 per window, high overlap)
        # Window threshold 0.5 triggers per-window, but we test interval (0.75)
        rollouts = {
            "miner_a": (Counter({"shared": 8, "unique": 2}), 10),
            "miner_b": (Counter({"shared": 8, "unique": 2}), 10),
        }
        # Ingest multiple windows to accumulate interval overlap
        for i in range(10):
            _, _, i_cheat, i_det, _, _ = tracker.ingest_window(100 * i, rollouts)
        # After 10 windows: 80 shared / 100 total = 0.8 > 0.75
        assert "miner_a" in i_cheat
        assert "miner_b" in i_cheat
        assert i_det[0].ratio >= COPYCAT_INTERVAL_THRESHOLD

    def test_interval_reset_clears_state(self, tracker: CopycatTracker) -> None:
        """Interval reset clears accumulated statistics."""
        tracker.reset_interval(0)
        miner_rollouts = {
            "miner_a": (Counter({"shared": 8}), 10),
            "miner_b": (Counter({"shared": 8}), 10),
        }
        tracker.ingest_window(100, miner_rollouts)
        assert tracker.interval_totals["miner_a"] == 10

        # Reset to new interval
        tracker.reset_interval(1)
        assert tracker.interval_totals["miner_a"] == 0
        assert len(tracker.interval_pair_overlap) == 0

    def test_three_way_overlap(self, tracker: CopycatTracker) -> None:
        """Multiple miners sharing digests creates pairwise violations."""
        tracker.reset_interval(0)
        # All three share same digest (severe copying)
        miner_rollouts = {
            "miner_a": (Counter({"shared_all": 8}), 10),
            "miner_b": (Counter({"shared_all": 8}), 10),
            "miner_c": (Counter({"shared_all": 8}), 10),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        # All three should be flagged (appears in 3 pairs)
        assert "miner_a" in w_cheat
        assert "miner_b" in w_cheat
        assert "miner_c" in w_cheat
        # Should have 3 pairs: (a,b), (a,c), (b,c)
        assert len(w_det) == 3

    def test_asymmetric_counts(self, tracker: CopycatTracker) -> None:
        """Uses min(total_a, total_b) as denominator."""
        tracker.reset_interval(0)
        # miner_a: 20 rollouts, 8 shared
        # miner_b: 10 rollouts, 8 shared
        # Ratio: 8 / min(20, 10) = 8 / 10 = 0.8 > 0.5
        miner_rollouts = {
            "miner_a": (Counter({"shared": 8, "unique_a": 12}), 20),
            "miner_b": (Counter({"shared": 8, "unique_b": 2}), 10),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert "miner_a" in w_cheat
        assert "miner_b" in w_cheat
        assert w_det[0].denominator == 10
        assert w_det[0].ratio == 0.8
        # Verify threshold constants are in range
        assert 0.0 < COPYCAT_WINDOW_THRESHOLD < 1.0
        assert 0.0 < COPYCAT_INTERVAL_THRESHOLD < 1.0


class TestCopycatWithRealisticSAT:
    """Test copycat detection with realistic SAT prompts and completions."""

    @pytest.fixture
    def tracker(self) -> CopycatTracker:
        """Fresh tracker for each test."""
        return CopycatTracker()

    @pytest.fixture
    def sat_prompt_tokens(self, sat_prompts: list[str]) -> list[int]:
        """Convert SAT prompt to mock token IDs."""
        from transformers import AutoTokenizer

        from grail.shared.constants import MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Use first SAT prompt and tokenize it
        tokens = tokenizer.encode(sat_prompts[0])
        return tokens

    def test_unique_sat_completions_no_violation(
        self, tracker: CopycatTracker, sat_prompt_tokens: list[int]
    ) -> None:
        """Miners with unique SAT solutions produce no violations."""
        tracker.reset_interval(0)
        miner_rollouts = {}
        for i, miner_id in enumerate(["alice", "bob", "charlie"]):
            seeds = [i * 100 + j for j in range(5)]
            digests = make_rollout_data(sat_prompt_tokens, seeds, [True] * 5)
            miner_rollouts[miner_id] = (Counter(digests), len(digests))

        w_cheat, _, i_cheat, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert len(w_cheat) == 0
        assert len(i_cheat) == 0

    def test_copied_sat_completions_detected(
        self, tracker: CopycatTracker, sat_prompt_tokens: list[int]
    ) -> None:
        """Detects when miners copy SAT completions from each other."""
        tracker.reset_interval(0)
        # alice & bob: 8 copied + 2 unique (80% overlap)
        # charlie: all unique
        alice_data = make_rollout_data(
            sat_prompt_tokens,
            [999] * 8 + [1000, 1001],
            [False] * 8 + [True, True],
        )
        bob_data = make_rollout_data(
            sat_prompt_tokens,
            [999] * 8 + [2000, 2001],
            [False] * 8 + [True, True],
        )
        charlie_data = make_rollout_data(sat_prompt_tokens, list(range(3000, 3010)), [True] * 10)

        miner_rollouts = {
            "alice": (Counter(alice_data), len(alice_data)),
            "bob": (Counter(bob_data), len(bob_data)),
            "charlie": (Counter(charlie_data), len(charlie_data)),
        }

        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)

        # alice and bob flagged (8/10 = 0.8 > 0.5), charlie not
        assert "alice" in w_cheat and "bob" in w_cheat
        assert "charlie" not in w_cheat
        assert any({"alice", "bob"} == {v.miner_a, v.miner_b} for v in w_det)

    def test_multiple_sat_prompts_mixed_overlap(
        self, tracker: CopycatTracker, sat_prompts: list[str]
    ) -> None:
        """Test with multiple SAT prompts, some copied, some unique."""
        from transformers import AutoTokenizer

        from grail.shared.constants import MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tracker.reset_interval(0)

        # miner1 & miner2: 3 copied + 7 unique (30% overlap, below threshold)
        # miner3: all unique
        miner_rollouts = {}
        for i, miner_id in enumerate(["miner1", "miner2", "miner3"]):
            prompt_tokens = tokenizer.encode(sat_prompts[i % len(sat_prompts)])
            if i < 2:
                # 3 copied, 7 unique
                seeds = [5555] * 3 + list(range(i * 1000, i * 1000 + 7))
                flags = [False] * 3 + [True] * 7
            else:
                seeds = list(range(3000, 3010))
                flags = [True] * 10
            digests = make_rollout_data(prompt_tokens, seeds, flags)
            miner_rollouts[miner_id] = (Counter(digests), len(digests))

        _, _, _, _, w_all, _ = tracker.ingest_window(200, miner_rollouts)

        # 30% overlap < 50% threshold (not flagged)
        pair = [v for v in w_all if {"miner1", "miner2"} == {v.miner_a, v.miner_b}]
        if pair:
            assert pair[0].ratio < COPYCAT_WINDOW_THRESHOLD
