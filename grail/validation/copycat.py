"""Copycat detection utilities for validator pipelines.

Rationale
---------
- We want to detect miners copying each other's rollouts. We hash the completion
  token IDs into stable digests and perform pairwise overlap comparisons.
- Two scopes are checked:
  - Per-window: flag if shared rollouts between two miners exceed
    COPYCAT_WINDOW_THRESHOLD of the smaller miner's total for that window.
  - Per-interval: same criterion but cumulative across the submission interval.
- This is pairwise (not group-wise) and uses counters per digest to avoid O(N^2)
  text operations; we accumulate min(count_i, count_j) per digest per pair.

Performance
-----------
- Time: O(R_w) per window to ingest, where R_w is rollouts across selected miners;
  aggregation over an interval is O(R_I). Hot digests can induce up to O(k^2)
  pair updates for k miners sharing a digest (bounded by sampling).
- Memory: O(U) where U is unique digests stored this interval; resets each interval.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

LOGGER = logging.getLogger("grail.copycat")

COPYCAT_WINDOW_THRESHOLD = 0.5
COPYCAT_INTERVAL_THRESHOLD = 0.75


def compute_completion_digest(commit_data: dict, rollout_meta: dict) -> str | None:
    """Return a SHA-256 digest of completion token IDs for copycat detection.

    We canonically slice completion tokens as tokens[prompt_length:] so identical
    completions map to the same digest across validators regardless of prompt.
    On failure to slice, we fall back to hashing the full token list.
    """

    try:
        tokens = commit_data.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            return None

        try:
            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
        except Exception:  # pragma: no cover - defensive fallback
            prompt_len = 0

        completion_ids = tokens[prompt_len:]
        digest_input = json.dumps(
            completion_ids, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(digest_input).hexdigest()
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.debug("Completion digest computation failed (%s)", exc)
        try:
            digest_input = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode()
            return hashlib.sha256(digest_input).hexdigest()
        except Exception as fallback_exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to hash tokens for copycat digest (%s)", fallback_exc)
            return None


@dataclass(frozen=True)
class CopycatViolation:
    """Details about a detected copycat relationship between two miners."""

    miner_a: str
    miner_b: str
    shared: int
    denominator: int
    ratio: float
    threshold: float
    scope: Literal["window", "interval"]
    window_start: int


class CopycatTracker:
    """Track rollout overlap between miners within windows and submission intervals.

    This tracker accumulates, per digest, which miners produced it and how many
    times (multiplicity). Overlap for a miner pair is computed as the sum over
    digests of min(count_a, count_b). Window-level overlap is reset each ingest;
    interval-level overlap is maintained until :meth:`reset_interval`.
    """

    def __init__(self) -> None:
        self.current_interval_id: int | None = None
        self.interval_pair_overlap: defaultdict[frozenset[str], int] = defaultdict(int)
        self.interval_totals: defaultdict[str, int] = defaultdict(int)

    def reset_interval(self, interval_id: int) -> None:
        """Reset interval statistics when a new submission interval starts.

        Parameters
        ----------
        interval_id: int
            Monotonic identifier (e.g., target_window // WEIGHT_SUBMISSION_INTERVAL_BLOCKS).
        """

        if self.current_interval_id == interval_id:
            return
        self.current_interval_id = interval_id
        self.interval_pair_overlap.clear()
        self.interval_totals.clear()

    def ingest_window(
        self,
        window_start: int,
        miner_rollouts: dict[str, tuple[Counter[str], int]],
    ) -> tuple[
        set[str],
        list[CopycatViolation],
        set[str],
        list[CopycatViolation],
        list[CopycatViolation],
        list[CopycatViolation],
    ]:
        """Ingest a window of rollouts and compute copycat cheaters.

        Parameters
        ----------
        window_start: int
            Window start block number (used in violation details).
        miner_rollouts: dict[str, tuple[Counter[str], int]]
            Map of miner hotkey -> (digest_counter, total_rollouts_in_window).

        Returns
        -------
        window_cheaters: set[str]
            Miners flagged at the window scope (threshold COPYCAT_WINDOW_THRESHOLD).
        window_details: list[CopycatViolation]
            Per-pair violation details for window scope (for logging/debug/audit).
        interval_cheaters: set[str]
            Miners flagged at the interval scope (threshold COPYCAT_INTERVAL_THRESHOLD).
        interval_details: list[CopycatViolation]
            Per-pair violation details for interval scope.
        window_all_pairs: list[CopycatViolation]
            All observed miner pairs this window with ratio, shared and denominator
            populated (threshold set to COPYCAT_WINDOW_THRESHOLD; may be below).
        interval_all_pairs: list[CopycatViolation]
            All observed miner pairs for the current interval with ratio, shared and
            denominator populated (threshold set to COPYCAT_INTERVAL_THRESHOLD; may be below).
        """

        window_pair_overlap: defaultdict[frozenset[str], int] = defaultdict(int)
        window_totals = {miner: total for miner, (_, total) in miner_rollouts.items()}

        digest_map: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for miner, (counter, _) in miner_rollouts.items():
            for digest, count in counter.items():
                digest_map[digest].append((miner, count))

        for miners in digest_map.values():
            if len(miners) < 2:
                continue
            for (miner_a, count_a), (miner_b, count_b) in combinations(miners, 2):
                overlap = min(count_a, count_b)
                if overlap <= 0:
                    continue
                pair_key = frozenset((miner_a, miner_b))
                window_pair_overlap[pair_key] += overlap
                self.interval_pair_overlap[pair_key] += overlap

        for miner, (_, total_rollouts) in miner_rollouts.items():
            self.interval_totals[miner] += total_rollouts

        window_cheaters, window_details = self._find_cheaters(
            pair_overlap=window_pair_overlap,
            totals=window_totals,
            threshold=COPYCAT_WINDOW_THRESHOLD,
            scope="window",
            window_start=window_start,
        )
        interval_cheaters, interval_details = self._find_cheaters(
            pair_overlap=self.interval_pair_overlap,
            totals=self.interval_totals,
            threshold=COPYCAT_INTERVAL_THRESHOLD,
            scope="interval",
            window_start=window_start,
        )

        # Build complete pair lists (not only those exceeding threshold)
        window_all_pairs: list[CopycatViolation] = []
        for pair_key, shared in window_pair_overlap.items():
            miner_a, miner_b = tuple(pair_key)
            denom = min(window_totals.get(miner_a, 0), window_totals.get(miner_b, 0))
            if denom <= 0:
                continue
            ratio = shared / float(denom)
            window_all_pairs.append(
                CopycatViolation(
                    miner_a=miner_a,
                    miner_b=miner_b,
                    shared=shared,
                    denominator=denom,
                    ratio=ratio,
                    threshold=COPYCAT_WINDOW_THRESHOLD,
                    scope="window",
                    window_start=window_start,
                )
            )

        interval_all_pairs: list[CopycatViolation] = []
        for pair_key, shared in self.interval_pair_overlap.items():
            miner_a, miner_b = tuple(pair_key)
            denom = min(self.interval_totals.get(miner_a, 0), self.interval_totals.get(miner_b, 0))
            if denom <= 0:
                continue
            ratio = shared / float(denom)
            interval_all_pairs.append(
                CopycatViolation(
                    miner_a=miner_a,
                    miner_b=miner_b,
                    shared=shared,
                    denominator=denom,
                    ratio=ratio,
                    threshold=COPYCAT_INTERVAL_THRESHOLD,
                    scope="interval",
                    window_start=window_start,
                )
            )

        return (
            window_cheaters,
            window_details,
            interval_cheaters,
            interval_details,
            window_all_pairs,
            interval_all_pairs,
        )

    def _find_cheaters(
        self,
        pair_overlap: dict[frozenset[str], int],
        totals: dict[str, int],
        threshold: float,
        scope: Literal["window", "interval"],
        window_start: int,
    ) -> tuple[set[str], list[CopycatViolation]]:
        """Return flagged miners and detailed violations for a given scope.

        The ratio criterion is shared / min(total_a, total_b) >= threshold.

        Returns
        -------
        flagged: set[str]
            Miners appearing in at least one violating pair.
        details: list[CopycatViolation]
            One record per violating pair capturing magnitude and scope.
        """
        flagged: set[str] = set()
        details: list[CopycatViolation] = []
        for pair_key, shared in pair_overlap.items():
            if shared <= 0:
                continue
            miner_a, miner_b = tuple(pair_key)
            total_a = totals.get(miner_a, 0)
            total_b = totals.get(miner_b, 0)
            denominator = min(total_a, total_b)
            if denominator <= 0:
                continue
            ratio = shared / float(denominator)
            if ratio >= threshold:
                flagged.update((miner_a, miner_b))
                details.append(
                    CopycatViolation(
                        miner_a=miner_a,
                        miner_b=miner_b,
                        shared=shared,
                        denominator=denominator,
                        ratio=ratio,
                        threshold=threshold,
                        scope=scope,
                        window_start=window_start,
                    )
                )
        return flagged, details


COPYCAT_TRACKER = CopycatTracker()

__all__ = [
    "COPYCAT_INTERVAL_THRESHOLD",
    "COPYCAT_TRACKER",
    "COPYCAT_WINDOW_THRESHOLD",
    "CopycatTracker",
    "CopycatViolation",
    "compute_completion_digest",
]
