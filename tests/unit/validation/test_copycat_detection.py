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
    MinerCopycatSubmission,
)
from tests.conftest import TEST_MODEL_ID


def _sub(
    digest_counter: Counter[str],
    total: int,
    upload_time: float | None = None,
) -> MinerCopycatSubmission:
    """Build a :class:`MinerCopycatSubmission` for test ergonomics.

    Centralising construction here keeps the tests focused on their
    assertions and means that if we add new fields to the submission
    dataclass in a later tier, we only update this one helper.
    """
    return MinerCopycatSubmission(
        digest_counter=digest_counter,
        total_rollouts=total,
        upload_time=upload_time,
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
        submissions = {
            "miner_a": _sub(Counter({"digest_1": 5}), 5, upload_time=100.0),
            "miner_b": _sub(Counter({"digest_2": 5}), 5, upload_time=110.0),
            "miner_c": _sub(Counter({"digest_3": 5}), 5, upload_time=120.0),
        }
        w_cheat, w_det, i_cheat, i_det, _, _ = tracker.ingest_window(100, submissions)
        assert len(w_cheat) == 0
        assert len(w_det) == 0
        assert len(i_cheat) == 0
        assert len(i_det) == 0

    def test_window_threshold_violation(self, tracker: CopycatTracker) -> None:
        """Flags the later uploader (copier) when a pair exceeds threshold.

        With directional gating, only the later uploader is added to the
        cheater set. The earlier uploader is the victim and is not gated.
        """
        tracker.reset_interval(0)
        # miner_a: 100 rollouts, 6 shared with miner_b
        # miner_b: 100 rollouts, 6 shared with miner_a
        # Ratio: 6 / min(100, 100) = 0.06 > COPYCAT_WINDOW_THRESHOLD (0.05)
        # miner_a uploaded first (t=100.0), miner_b uploaded later (t=200.0)
        # so miner_b is the copier, miner_a is the victim.
        submissions = {
            "miner_a": _sub(Counter({"shared": 6, "unique_a": 94}), 100, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 6, "unique_b": 94}), 100, upload_time=200.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, submissions)
        assert w_cheat == {"miner_b"}, "only the later uploader should be flagged"
        assert "miner_a" not in w_cheat, "victim must not be gated (Tier 0 bugfix)"
        assert len(w_det) == 1
        assert w_det[0].shared == 6
        assert w_det[0].ratio == 0.06
        assert w_det[0].copier == "miner_b"
        assert w_det[0].victim == "miner_a"

    def test_window_below_threshold_no_violation(self, tracker: CopycatTracker) -> None:
        """No violation when overlap below window threshold."""
        tracker.reset_interval(0)
        # Ratio: 4 / min(100, 100) = 0.04 < 0.05
        submissions = {
            "miner_a": _sub(Counter({"shared": 4, "unique_a": 96}), 100, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 4, "unique_b": 96}), 100, upload_time=200.0),
        }
        w_cheat, w_det, _, _, w_all, _ = tracker.ingest_window(100, submissions)
        assert len(w_cheat) == 0
        assert len(w_det) == 0
        # But should appear in all_pairs
        assert len(w_all) == 1
        assert w_all[0].ratio == 0.04

    def test_interval_accumulation(self, tracker: CopycatTracker) -> None:
        """Interval-level detection accumulates across windows.

        Uses distinct upload times so directional attribution can resolve
        the copier across every contributing window.
        """
        tracker.reset_interval(0)
        # Each window: 4 shared out of 100 = 0.04 per window (below window threshold)
        # but accumulated over 10 windows: 40 shared / 1000 total = 0.04 > 0.03
        submissions = {
            "miner_a": _sub(Counter({"shared": 4, "unique_a": 96}), 100, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 4, "unique_b": 96}), 100, upload_time=200.0),
        }
        # Ingest multiple windows to accumulate interval overlap
        for i in range(10):
            _, _, i_cheat, i_det, _, _ = tracker.ingest_window(100 * i, submissions)
        # After 10 windows: 40 shared / 1000 total = 0.04 > 0.03
        assert i_cheat == {"miner_b"}
        assert i_det[0].ratio >= COPYCAT_INTERVAL_THRESHOLD
        assert i_det[0].copier == "miner_b"
        assert i_det[0].victim == "miner_a"

    def test_interval_reset_clears_state(self, tracker: CopycatTracker) -> None:
        """Interval reset clears accumulated statistics."""
        tracker.reset_interval(0)
        submissions = {
            "miner_a": _sub(Counter({"shared": 8}), 10, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 8}), 10, upload_time=200.0),
        }
        tracker.ingest_window(100, submissions)
        assert tracker.interval_totals["miner_a"] == 10

        # Reset to new interval
        tracker.reset_interval(1)
        assert tracker.interval_totals["miner_a"] == 0
        assert len(tracker.interval_pair_overlap) == 0

    def test_three_way_overlap(self, tracker: CopycatTracker) -> None:
        """Multiple miners sharing digests produce pairwise violations.

        The earliest uploader is the victim; every later uploader is a
        copier and is added to the cheater set.
        """
        tracker.reset_interval(0)
        # All three share 8 out of 100 = 0.08 > 0.05 (severe copying)
        submissions = {
            "miner_a": _sub(Counter({"shared_all": 8, "unique_a": 92}), 100, upload_time=100.0),
            "miner_b": _sub(Counter({"shared_all": 8, "unique_b": 92}), 100, upload_time=200.0),
            "miner_c": _sub(Counter({"shared_all": 8, "unique_c": 92}), 100, upload_time=300.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, submissions)
        # miner_a is the earliest and therefore the victim on every pair;
        # miner_b is later than miner_a (copier) and earlier than miner_c
        # (victim), miner_c is the latest (copier on every pair it touches).
        assert w_cheat == {"miner_b", "miner_c"}
        assert "miner_a" not in w_cheat, "earliest uploader must not be gated"
        # Should have 3 pair records: (a,b), (a,c), (b,c)
        assert len(w_det) == 3

    def test_asymmetric_counts(self, tracker: CopycatTracker) -> None:
        """Uses min(total_a, total_b) as denominator; gates the later uploader."""
        tracker.reset_interval(0)
        # miner_a: 200 rollouts, 8 shared
        # miner_b: 100 rollouts, 8 shared
        # Ratio: 8 / min(200, 100) = 8 / 100 = 0.08 > 0.05
        submissions = {
            "miner_a": _sub(Counter({"shared": 8, "unique_a": 192}), 200, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 8, "unique_b": 92}), 100, upload_time=200.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, submissions)
        assert w_cheat == {"miner_b"}
        assert w_det[0].denominator == 100
        assert w_det[0].ratio == 0.08
        assert w_det[0].copier == "miner_b"
        assert w_det[0].victim == "miner_a"
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
        submissions: dict[str, MinerCopycatSubmission] = {}
        for i, miner_id in enumerate(["alice", "bob", "charlie"]):
            seeds = [i * 100 + j for j in range(5)]
            digests = make_rollout_data(sat_prompt_tokens, seeds, [True] * 5)
            submissions[miner_id] = _sub(Counter(digests), len(digests), upload_time=100.0 + i)

        w_cheat, _, i_cheat, _, _, _ = tracker.ingest_window(100, submissions)
        assert len(w_cheat) == 0
        assert len(i_cheat) == 0

    def test_copied_sat_completions_detected(
        self, tracker: CopycatTracker, sat_prompt_tokens: list[int]
    ) -> None:
        """Detects when miners copy SAT completions from each other.

        With directional gating, only the later uploader (``bob``) is
        flagged. Alice, who uploaded first, is the victim and keeps her
        metrics.
        """
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

        submissions = {
            "alice": _sub(Counter(alice_data), len(alice_data), upload_time=100.0),
            "bob": _sub(Counter(bob_data), len(bob_data), upload_time=200.0),
            "charlie": _sub(Counter(charlie_data), len(charlie_data), upload_time=300.0),
        }

        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(100, submissions)

        # Only the later uploader (bob) is flagged; alice is the victim.
        assert w_cheat == {"bob"}
        assert "alice" not in w_cheat
        assert "charlie" not in w_cheat
        alice_bob_pairs = [v for v in w_det if {"alice", "bob"} == {v.miner_a, v.miner_b}]
        assert alice_bob_pairs, "alice↔bob violation should be recorded"
        assert alice_bob_pairs[0].copier == "bob"
        assert alice_bob_pairs[0].victim == "alice"

    def test_multiple_sat_prompts_mixed_overlap(
        self, tracker: CopycatTracker, sat_prompts: list[str]
    ) -> None:
        """Test with multiple SAT prompts, some copied, some unique."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_ID)
        tracker.reset_interval(0)

        # miner1 & miner2: 3 copied + 97 unique (3% overlap, below 5% threshold)
        # miner3: all unique
        submissions: dict[str, MinerCopycatSubmission] = {}
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
            submissions[miner_id] = _sub(Counter(digests), len(digests), upload_time=100.0 + i)

        _, _, _, _, w_all, _ = tracker.ingest_window(200, submissions)

        # 3% overlap < 5% threshold (not flagged)
        pair = [v for v in w_all if {"miner1", "miner2"} == {v.miner_a, v.miner_b}]
        if pair:
            assert pair[0].ratio < COPYCAT_WINDOW_THRESHOLD


class TestDirectionalGating:
    """Regression tests for the Tier 0 victim-punishment fix.

    Background:
        The pre-Tier-0 detector flagged **both** sides of a violating pair
        with ``flagged.update((miner_a, miner_b))``. On subnet 81 this
        actively penalised honest singleton miners (UIDs 53 and 7) who
        were being scraped by coldkey ``5CAL2bA5...``: the scrapers'
        interval-scope overlap crossed the 3% threshold against UIDs 53
        and 7, and the gating logic zeroed the victims' metrics too.

    These tests pin the new semantics:

    - Only the later uploader (copier) is added to the cheater set.
    - When direction cannot be resolved (missing or equal timestamps), the
      detector ABSTAINS: no one is gated, even though a violation is logged.
    """

    @pytest.fixture
    def tracker(self) -> CopycatTracker:
        return CopycatTracker()

    @pytest.fixture
    def overlapping_digests(self) -> tuple[Counter[str], Counter[str]]:
        """Two digest counters that exceed the window threshold.

        Returns a pair where ``shared / min(total_a, total_b) = 0.06``,
        i.e. just above ``COPYCAT_WINDOW_THRESHOLD`` (0.05).
        """
        return (
            Counter({"shared": 6, "unique_a": 94}),
            Counter({"shared": 6, "unique_b": 94}),
        )

    def test_only_later_uploader_is_gated(
        self,
        tracker: CopycatTracker,
        overlapping_digests: tuple[Counter[str], Counter[str]],
    ) -> None:
        """Later uploader is the copier; earlier uploader is the untouched victim."""
        tracker.reset_interval(0)
        counter_a, counter_b = overlapping_digests
        submissions = {
            "victim": _sub(counter_a, 100, upload_time=1_000.0),
            "copier": _sub(counter_b, 100, upload_time=1_050.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert w_cheat == {"copier"}, "only the copier should be gated"
        assert "victim" not in w_cheat, "victim must never be gated"

        assert len(w_det) == 1
        violation = w_det[0]
        assert violation.copier == "copier"
        assert violation.victim == "victim"
        assert violation.ratio >= COPYCAT_WINDOW_THRESHOLD

    def test_role_swap_when_upload_order_inverts(
        self,
        tracker: CopycatTracker,
        overlapping_digests: tuple[Counter[str], Counter[str]],
    ) -> None:
        """Swapping upload order swaps copier/victim — proves it's not alphabetical."""
        tracker.reset_interval(0)
        counter_a, counter_b = overlapping_digests
        submissions = {
            "alpha": _sub(counter_a, 100, upload_time=2_000.0),  # later
            "bravo": _sub(counter_b, 100, upload_time=1_000.0),  # earlier
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert w_cheat == {"alpha"}
        assert w_det[0].copier == "alpha"
        assert w_det[0].victim == "bravo"

    def test_missing_upload_time_abstains(
        self,
        tracker: CopycatTracker,
        overlapping_digests: tuple[Counter[str], Counter[str]],
    ) -> None:
        """If either upload_time is None, no one is gated.

        This is the safe default: the detector prefers a missed detection
        to punishing the wrong side. The violation should still appear in
        the details list so monitoring can see it.
        """
        tracker.reset_interval(0)
        counter_a, counter_b = overlapping_digests
        submissions = {
            "miner_a": _sub(counter_a, 100, upload_time=None),
            "miner_b": _sub(counter_b, 100, upload_time=1_000.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert w_cheat == set(), "unresolved direction must gate nobody"
        assert len(w_det) == 1, "violation still recorded for observability"
        assert w_det[0].copier is None
        assert w_det[0].victim is None

    def test_equal_upload_time_abstains(
        self,
        tracker: CopycatTracker,
        overlapping_digests: tuple[Counter[str], Counter[str]],
    ) -> None:
        """Equal timestamps (within resolution) are treated as unresolved."""
        tracker.reset_interval(0)
        counter_a, counter_b = overlapping_digests
        submissions = {
            "miner_a": _sub(counter_a, 100, upload_time=4_242.0),
            "miner_b": _sub(counter_b, 100, upload_time=4_242.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert w_cheat == set(), "equal timestamps must abstain"
        assert len(w_det) == 1
        assert w_det[0].copier is None
        assert w_det[0].victim is None

    def test_below_threshold_is_not_gated_even_with_direction(
        self,
        tracker: CopycatTracker,
    ) -> None:
        """Directional gating only applies to pairs that exceed the threshold."""
        tracker.reset_interval(0)
        # 4 / 100 = 0.04 < 0.05 window threshold
        submissions = {
            "miner_a": _sub(Counter({"shared": 4, "u_a": 96}), 100, upload_time=100.0),
            "miner_b": _sub(Counter({"shared": 4, "u_b": 96}), 100, upload_time=200.0),
        }
        w_cheat, w_det, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert w_cheat == set()
        assert w_det == []

    def test_three_way_earliest_is_spared(self, tracker: CopycatTracker) -> None:
        """In an N-way cluster, the earliest uploader is never gated.

        Reproduces the layout that caused the bug: three miners all share
        the same completion set at a rate above threshold. Under the old
        symmetric flagger, all three were zeroed. Under directional
        gating, only the two later uploaders are gated.
        """
        tracker.reset_interval(0)
        submissions = {
            "earliest": _sub(Counter({"shared_all": 8, "unique_a": 92}), 100, upload_time=1.0),
            "mid": _sub(Counter({"shared_all": 8, "unique_b": 92}), 100, upload_time=2.0),
            "latest": _sub(Counter({"shared_all": 8, "unique_c": 92}), 100, upload_time=3.0),
        }
        w_cheat, _, _, _, _, _ = tracker.ingest_window(1, submissions)

        assert "earliest" not in w_cheat, (
            "the earliest uploader must not be gated — regression guard for "
            "the Tier 0 victim-punishment fix"
        )
        assert w_cheat == {"mid", "latest"}

    def test_copier_gating_persists_across_interval(
        self,
        tracker: CopycatTracker,
    ) -> None:
        """At interval scope, the copier is flagged using the most recent window's
        upload times, and the victim remains untouched across the whole interval.
        """
        tracker.reset_interval(0)
        # 4 shared per window → below window threshold but above 3% interval
        # threshold after 10 windows.
        submissions = {
            "victim": _sub(Counter({"shared": 4, "u_a": 96}), 100, upload_time=100.0),
            "copier": _sub(Counter({"shared": 4, "u_b": 96}), 100, upload_time=200.0),
        }
        for i in range(10):
            _, _, i_cheat, i_det, _, _ = tracker.ingest_window(100 * i, submissions)

        assert i_cheat == {"copier"}, "victim must not accumulate as a cheater"
        assert all(v.copier == "copier" and v.victim == "victim" for v in i_det)

    def test_interval_attribution_survives_victim_going_offline(
        self,
        tracker: CopycatTracker,
    ) -> None:
        """Regression: interval-scope attribution must still work when the
        victim is absent from the current window.

        Reproduces the blocker found independently by two reviewers against
        an earlier draft of this change. Before the fix,
        ``CopycatTracker._find_cheaters`` at interval scope used only the
        current window's ``uploads`` map. When the victim went offline
        mid-interval, ``_attribute_pair_direction`` returned
        :meth:`_PairAttribution.unresolved` because ``uploads[victim]`` was
        missing, and the accumulated cheat escaped gating entirely.

        The fix archives per-miner upload times on the tracker as
        ``interval_upload_times``, updated on each ingested window and
        cleared on ``reset_interval``. This test locks in the expected
        post-fix behaviour: a victim who stops submitting must still be
        recognised as the earlier uploader, and the copier must still be
        flagged at interval scope.
        """
        tracker.reset_interval(0)

        # Windows 1-5: both submit, accumulate interval overlap.
        # Per-window ratio 4/100 = 0.04 (just under window threshold 0.05)
        # but interval after 5 windows is 20/500 = 0.04 > interval threshold 0.03.
        both = {
            "victim": _sub(Counter({"shared": 4, "u_a": 96}), 100, upload_time=100.0),
            "copier": _sub(Counter({"shared": 4, "u_b": 96}), 100, upload_time=200.0),
        }
        for i in range(5):
            tracker.ingest_window(100 * i, both)

        # Window 6: victim goes offline, only copier submits. The
        # submissions map has no entry for the victim, so the current-window
        # uploads view also has no entry for them. The pre-fix code would
        # abstain here; the fix must remember the victim's upload time
        # from windows 1-5.
        copier_only = {
            "copier": _sub(Counter({"u_b_new": 100}), 100, upload_time=600.0),
        }
        _, _, i_cheat, i_det, _, _ = tracker.ingest_window(600, copier_only)

        assert i_cheat == {"copier"}, (
            "interval-scope attribution must still fire on the absent victim's "
            "accumulated overlap — this is the regression guard for the "
            "reviewers' blocker finding"
        )
        assert "victim" not in i_cheat, "victim is offline but must not be gated"
        victim_copier_pairs = [v for v in i_det if {"victim", "copier"} == {v.miner_a, v.miner_b}]
        assert victim_copier_pairs, "violation record should still exist"
        assert victim_copier_pairs[0].copier == "copier"
        assert victim_copier_pairs[0].victim == "victim"

    def test_reset_interval_clears_archived_upload_times(
        self,
        tracker: CopycatTracker,
    ) -> None:
        """`reset_interval` must clear interval_upload_times along with
        interval_pair_overlap and interval_totals. Otherwise a stale
        upload_time from a previous interval could leak into attribution.
        """
        tracker.reset_interval(0)
        submissions = {
            "miner_a": _sub(Counter({"x": 10}), 10, upload_time=100.0),
            "miner_b": _sub(Counter({"y": 10}), 10, upload_time=200.0),
        }
        tracker.ingest_window(0, submissions)
        assert tracker.interval_upload_times == {
            "miner_a": 100.0,
            "miner_b": 200.0,
        }

        tracker.reset_interval(1)
        assert tracker.interval_upload_times == {}, (
            "reset_interval must clear archived upload times to avoid "
            "cross-interval timestamp leakage"
        )
