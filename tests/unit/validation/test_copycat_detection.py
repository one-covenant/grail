"""Test copycat detection for validator pipelines.

Verifies that the copycat tracker correctly identifies miners submitting
identical or overlapping completions within windows and across intervals.
"""

from __future__ import annotations

from collections import Counter

import pytest

from grail.shared.digest import compute_completion_digest
from grail.validation.copycat_service import (
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_WINDOW_THRESHOLD,
    CopycatTracker,
)
from tests.conftest import TEST_MODEL_ID


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
    for seed, is_unique in zip(seeds, unique_flags, strict=False):
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
        # miner_a: 100 rollouts, 6 shared with miner_b
        # miner_b: 100 rollouts, 6 shared with miner_a
        # Ratio: 6 / min(100, 100) = 0.06 > COPYCAT_WINDOW_THRESHOLD (0.05)
        miner_rollouts = {
            "miner_a": (Counter({"shared": 6, "unique_a": 94}), 100),
            "miner_b": (Counter({"shared": 6, "unique_b": 94}), 100),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert "miner_a" in w_cheat
        assert "miner_b" in w_cheat
        assert len(w_det) == 1
        assert w_det[0].shared == 6
        assert w_det[0].ratio == 0.06

    def test_window_below_threshold_no_violation(self, tracker: CopycatTracker) -> None:
        """No violation when overlap below window threshold."""
        tracker.reset_interval(0)
        # Ratio: 4 / min(100, 100) = 0.04 < 0.05
        miner_rollouts = {
            "miner_a": (Counter({"shared": 4, "unique_a": 96}), 100),
            "miner_b": (Counter({"shared": 4, "unique_b": 96}), 100),
        }
        w_cheat, w_det, _, _, w_all, _ = tracker.ingest_window(100, miner_rollouts)
        assert len(w_cheat) == 0
        assert len(w_det) == 0
        # But should appear in all_pairs
        assert len(w_all) == 1
        assert w_all[0].ratio == 0.04

    def test_interval_accumulation(self, tracker: CopycatTracker) -> None:
        """Interval-level detection accumulates across windows."""
        tracker.reset_interval(0)
        # Each window: 4 shared out of 100 = 0.04 per window (below window threshold)
        # but accumulated over 10 windows: 40 shared / 1000 total = 0.04 > 0.03
        rollouts = {
            "miner_a": (Counter({"shared": 4, "unique_a": 96}), 100),
            "miner_b": (Counter({"shared": 4, "unique_b": 96}), 100),
        }
        # Ingest multiple windows to accumulate interval overlap
        for i in range(10):
            _, _, i_cheat, i_det, _, _ = tracker.ingest_window(100 * i, rollouts)
        # After 10 windows: 40 shared / 1000 total = 0.04 > 0.03
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
        # All three share 8 out of 100 = 0.08 > 0.05 (severe copying)
        miner_rollouts = {
            "miner_a": (Counter({"shared_all": 8, "unique_a": 92}), 100),
            "miner_b": (Counter({"shared_all": 8, "unique_b": 92}), 100),
            "miner_c": (Counter({"shared_all": 8, "unique_c": 92}), 100),
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
        # miner_a: 200 rollouts, 8 shared
        # miner_b: 100 rollouts, 8 shared
        # Ratio: 8 / min(200, 100) = 8 / 100 = 0.08 > 0.05
        miner_rollouts = {
            "miner_a": (Counter({"shared": 8, "unique_a": 192}), 200),
            "miner_b": (Counter({"shared": 8, "unique_b": 92}), 100),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)
        assert "miner_a" in w_cheat
        assert "miner_b" in w_cheat
        assert w_det[0].denominator == 100
        assert w_det[0].ratio == 0.08
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

        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID)
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
        # alice & bob: 8 copied + 92 unique out of 100 (8% overlap > 5% threshold)
        # charlie: all unique
        alice_seeds = [999] * 8 + list(range(1000, 1092))
        alice_flags = [False] * 8 + [True] * 92
        bob_seeds = [999] * 8 + list(range(2000, 2092))
        bob_flags = [False] * 8 + [True] * 92
        alice_data = make_rollout_data(sat_prompt_tokens, alice_seeds, alice_flags)
        bob_data = make_rollout_data(sat_prompt_tokens, bob_seeds, bob_flags)
        charlie_data = make_rollout_data(sat_prompt_tokens, list(range(3000, 3100)), [True] * 100)

        miner_rollouts = {
            "alice": (Counter(alice_data), len(alice_data)),
            "bob": (Counter(bob_data), len(bob_data)),
            "charlie": (Counter(charlie_data), len(charlie_data)),
        }

        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, miner_rollouts)

        # alice and bob flagged (8/100 = 0.08 > 0.05), charlie not
        assert "alice" in w_cheat and "bob" in w_cheat
        assert "charlie" not in w_cheat
        assert any({"alice", "bob"} == {v.miner_a, v.miner_b} for v in w_det)

    def test_multiple_sat_prompts_mixed_overlap(
        self, tracker: CopycatTracker, sat_prompts: list[str]
    ) -> None:
        """Test with multiple SAT prompts, some copied, some unique."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID)
        tracker.reset_interval(0)

        # miner1 & miner2: 3 copied + 97 unique (3% overlap, below 5% threshold)
        # miner3: all unique
        miner_rollouts = {}
        for i, miner_id in enumerate(["miner1", "miner2", "miner3"]):
            prompt_tokens = tokenizer.encode(sat_prompts[i % len(sat_prompts)])
            if i < 2:
                # 3 copied, 97 unique
                seeds = [5555] * 3 + list(range(i * 1000, i * 1000 + 97))
                flags = [False] * 3 + [True] * 97
            else:
                seeds = list(range(3000, 3100))
                flags = [True] * 100
            digests = make_rollout_data(prompt_tokens, seeds, flags)
            miner_rollouts[miner_id] = (Counter(digests), len(digests))

        _, _, _, _, w_all, _ = tracker.ingest_window(200, miner_rollouts)

        # 3% overlap < 5% threshold (not flagged)
        pair = [v for v in w_all if {"miner1", "miner2"} == {v.miner_a, v.miner_b}]
        if pair:
            assert pair[0].ratio < COPYCAT_WINDOW_THRESHOLD
