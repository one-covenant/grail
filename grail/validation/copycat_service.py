"""Copycat detection service for GRAIL validation.

Handles detection of miners copying rollouts from each other, both within
individual windows and across submission intervals.

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

import contextlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

from ..shared.digest import compute_completion_digest
from .miner_validator import FAILURE_FLAG_KEY

logger = logging.getLogger(__name__)

# Copycat detection thresholds
COPYCAT_WINDOW_THRESHOLD = 0.5
COPYCAT_INTERVAL_THRESHOLD = 0.75


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

        Args:
            interval_id: Monotonic identifier (e.g., target_window // WEIGHT_SUBMISSION_INTERVAL_BLOCKS).
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

        Args:
            window_start: Window start block number (used in violation details).
            miner_rollouts: Map of miner hotkey -> (digest_counter, total_rollouts_in_window).

        Returns:
            Tuple of:
            - window_cheaters: Miners flagged at the window scope
            - window_details: Per-pair violation details for window scope
            - interval_cheaters: Miners flagged at the interval scope
            - interval_details: Per-pair violation details for interval scope
            - window_all_pairs: All observed miner pairs this window
            - interval_all_pairs: All observed miner pairs for the current interval
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

        Returns:
            Tuple of:
            - flagged: Miners appearing in at least one violating pair
            - details: One record per violating pair capturing magnitude and scope
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


class CopycatService:
    """Service for detecting and gating copycat miners.

    Responsibilities:
    1. Track rollout overlap between miners (window and interval scopes)
    2. Detect cheaters exceeding thresholds
    3. Zero out metrics for detected cheaters
    4. Filter cheater rollouts from upload list
    5. Log copycat metrics to monitoring

    Design:
    - Uses global COPYCAT_TRACKER singleton (managed lifecycle)
    - Stateless service - all state in tracker
    - Clean separation of detection vs. gating
    """

    def __init__(self):
        """Initialize copycat service.

        Note: Uses global COPYCAT_TRACKER singleton to maintain state across windows.
        """
        # All state is in the global COPYCAT_TRACKER, this is just a wrapper
        pass

    def reset_interval(self, interval_id: int) -> None:
        """Reset copycat tracking for a new submission interval.

        Args:
            interval_id: Monotonic interval identifier (e.g., window // interval_length)
        """
        COPYCAT_TRACKER.reset_interval(interval_id)
        logger.info(f"Reset copycat tracker for interval {interval_id}")

    async def detect_cheaters(
        self,
        window: int,
        miner_rollout_counters: dict[str, tuple[Counter[str], int]],
        uid_by_hotkey: dict[str, int],
        monitor: Any | None = None,
    ) -> tuple[set[str], set[str], list[CopycatViolation]]:
        """Detect copycat cheaters for a window.

        Args:
            window: Window start block number
            miner_rollout_counters: Map of hotkey -> (digest_counter, total_rollouts)
            uid_by_hotkey: Mapping of hotkey to UID (for monitor namespacing)
            monitor: Optional monitoring client

        Returns:
            Tuple of:
            - window_cheaters: Miners exceeding window threshold
            - interval_cheaters: Miners exceeding interval threshold
            - all_violations: All violation details for logging
        """
        if not miner_rollout_counters:
            return set(), set(), []

        # Time the copycat detection if monitor available
        timer_ctx = (
            monitor.timer("validation/copycat_detector") if monitor else contextlib.nullcontext()
        )

        with timer_ctx:
            (
                window_cheaters,
                window_violation_details,
                interval_cheaters,
                interval_violation_details,
                window_all_pairs,
                interval_all_pairs,
            ) = COPYCAT_TRACKER.ingest_window(window, miner_rollout_counters)

        # Log detection metrics to monitor
        if monitor:
            try:
                await monitor.log_gauge(
                    "validation/copycat/window_cheaters", float(len(window_cheaters))
                )
                await monitor.log_gauge(
                    "validation/copycat/interval_cheaters", float(len(interval_cheaters))
                )
            except Exception:
                pass

        # Log all violations
        all_violations = window_violation_details + interval_violation_details
        for violation in all_violations:
            logger.warning(
                "Copycat overlap detected: miners %s & %s shared=%d denom=%d ratio=%.3f "
                "threshold=%.2f scope=%s window=%d",
                violation.miner_a,
                violation.miner_b,
                violation.shared,
                violation.denominator,
                violation.ratio,
                violation.threshold,
                violation.scope,
                violation.window_start,
            )

        # Log per-miner proximity metrics to monitor
        if monitor:
            await self._log_miner_proximity_metrics(
                window_all_pairs=window_all_pairs,
                interval_all_pairs=interval_all_pairs,
                uid_by_hotkey=uid_by_hotkey,
                monitor=monitor,
            )

        return window_cheaters, interval_cheaters, all_violations

    async def _log_miner_proximity_metrics(
        self,
        window_all_pairs: list[CopycatViolation],
        interval_all_pairs: list[CopycatViolation],
        uid_by_hotkey: dict[str, int],
        monitor: Any,
    ) -> None:
        """Log per-miner copycat proximity metrics.

        For each miner, logs:
        - window_max_ratio: Maximum overlap ratio with any peer (window scope)
        - interval_max_ratio: Maximum overlap ratio with any peer (interval scope)
        - window_proximity: Ratio divided by threshold (how close to violation)
        - interval_proximity: Ratio divided by threshold (how close to violation)

        Args:
            window_all_pairs: All miner pairs from window scope
            interval_all_pairs: All miner pairs from interval scope
            uid_by_hotkey: Mapping of hotkey to UID for namespace
            monitor: Monitoring client
        """
        try:
            logger.info("Window all pairs logging started")
            logger.info("Interval all pairs: %s", interval_all_pairs)
            logger.info("Window all pairs: %s", window_all_pairs)

            # Build per-miner max ratios
            window_max_ratio: defaultdict[str, float] = defaultdict(float)
            for v in window_all_pairs:
                window_max_ratio[v.miner_a] = max(window_max_ratio[v.miner_a], v.ratio)
                window_max_ratio[v.miner_b] = max(window_max_ratio[v.miner_b], v.ratio)

            interval_max_ratio: defaultdict[str, float] = defaultdict(float)
            for v in interval_all_pairs:
                interval_max_ratio[v.miner_a] = max(interval_max_ratio[v.miner_a], v.ratio)
                interval_max_ratio[v.miner_b] = max(interval_max_ratio[v.miner_b], v.ratio)

            # Emit gauges per miner namespace (uid/hotkey-aware)
            all_miners = set(list(window_max_ratio.keys()) + list(interval_max_ratio.keys()))
            for miner_hk in all_miners:
                uid_str = str(uid_by_hotkey.get(miner_hk, miner_hk))
                wr = float(window_max_ratio.get(miner_hk, 0.0))
                ir = float(interval_max_ratio.get(miner_hk, 0.0))

                await monitor.log_gauge(f"{uid_str}/copycat/window_max_ratio", wr)
                await monitor.log_gauge(f"{uid_str}/copycat/interval_max_ratio", ir)

                # Log proximity to thresholds (ratio / threshold)
                await monitor.log_gauge(
                    f"{uid_str}/copycat/window_proximity",
                    wr / COPYCAT_WINDOW_THRESHOLD if COPYCAT_WINDOW_THRESHOLD > 0 else 0.0,
                )
                await monitor.log_gauge(
                    f"{uid_str}/copycat/interval_proximity",
                    ir / COPYCAT_INTERVAL_THRESHOLD if COPYCAT_INTERVAL_THRESHOLD > 0 else 0.0,
                )

        except Exception as e:
            logger.debug(f"Failed to log copycat proximity metrics: {e}")

    def apply_gating(
        self,
        cheaters: set[str],
        violations: list[CopycatViolation],
        window_metrics: dict[str, dict[str, int]],
        uid_by_hotkey: dict[str, int],
        window: int,
    ) -> None:
        """Apply gating to detected cheaters by zeroing their metrics.

        Modifies window_metrics in-place to zero out valid/successful/unique counts
        for cheaters and sets their failure flag.

        Args:
            cheaters: Set of cheater hotkeys
            violations: All violation details (for logging)
            window_metrics: Dict of hotkey -> metrics dict (modified in-place)
            uid_by_hotkey: Mapping of hotkey to UID
            window: Window number (for logging context)
        """
        # Build violation map for per-miner scopes/ratios
        violation_map: defaultdict[str, list[CopycatViolation]] = defaultdict(list)
        for violation in violations:
            violation_map[violation.miner_a].append(violation)
            violation_map[violation.miner_b].append(violation)

        for cheater in cheaters:
            # Get or create metrics dict
            metrics = window_metrics.get(cheater)
            if metrics is None:
                # Cheater had no metrics yet - create zero metrics
                metrics = {
                    "valid": 0,
                    "checked": 0,
                    "total": 0,
                    "estimated_valid": 0,
                    "estimated_successful": 0,
                    "estimated_unique": 0,
                    "successful": 0,
                    "unique": 0,
                    "prompt_valid": 0,
                    "prompt_mismatch": 0,
                }
                window_metrics[cheater] = metrics
            else:
                # Zero out the counts that contribute to weight
                metrics["valid"] = 0
                metrics["estimated_valid"] = 0
                metrics["estimated_successful"] = 0
                metrics["estimated_unique"] = 0
                metrics["successful"] = 0
                metrics["unique"] = 0

            # Set failure flag
            metrics[FAILURE_FLAG_KEY] = 1

            # Log gating with violation details
            uid_str = str(uid_by_hotkey.get(cheater, cheater))
            scopes = {v.scope for v in violation_map.get(cheater, [])}
            ratios = ", ".join(f"{v.ratio:.3f}" for v in violation_map.get(cheater, []))

            logger.warning(
                f"[window={window} uid={uid_str}] Copycat gating applied "
                f"(scopes={','.join(sorted(scopes)) if scopes else 'unknown'} "
                f"ratios={ratios or 'n/a'})"
            )

    def filter_cheater_rollouts(
        self,
        rollouts: list[dict],
        cheaters: set[str],
    ) -> list[dict]:
        """Filter out rollouts from detected cheaters.

        Args:
            rollouts: List of rollout dicts (must have 'hotkey' field)
            cheaters: Set of cheater hotkeys

        Returns:
            Filtered list of rollouts excluding cheaters
        """
        if not cheaters or not rollouts:
            return rollouts

        filtered = [r for r in rollouts if r.get("hotkey") not in cheaters]

        if len(filtered) < len(rollouts):
            logger.info(
                f"Filtered {len(rollouts) - len(filtered)} rollouts from "
                f"{len(cheaters)} detected cheaters"
            )

        return filtered


# Global instances
COPYCAT_TRACKER = CopycatTracker()
COPYCAT_SERVICE = CopycatService()

__all__ = [
    "COPYCAT_INTERVAL_THRESHOLD",
    "COPYCAT_TRACKER",
    "COPYCAT_WINDOW_THRESHOLD",
    "CopycatTracker",
    "CopycatViolation",
    "CopycatService",
    "COPYCAT_SERVICE",
    "compute_completion_digest",
]
