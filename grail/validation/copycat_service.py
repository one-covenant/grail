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
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

from ..shared.digest import compute_completion_digest
from .miner_validator import FAILURE_FLAG_KEY

logger = logging.getLogger(__name__)

# Copycat detection thresholds
#
# Calibrated from first principles for MBPP code-generation at temperature 0.7:
# - Completion digests are SHA-256 of token IDs → exact token-for-token match required.
# - At temp 0.7, even short canonical solutions (~15 tokens) have <1% per-pair
#   match probability; longer solutions (~50+ tokens) are near zero.
# - With 16 rollouts/problem, ~10% of MBPP problems are short enough for occasional
#   exact matches → natural overlap baseline is ~1-2% of rollouts.
# - Window threshold (5%): ~2.5× above natural ceiling; single-window data is noisier.
# - Interval threshold (3%): accumulated across 12 windows, variance averages out,
#   so we can be more sensitive to persistent low-level copying.
COPYCAT_WINDOW_THRESHOLD = 0.05
COPYCAT_INTERVAL_THRESHOLD = 0.03


# Upload-time map propagated from ``MinerSampler.discover_active_miners``.
# Values are S3 ``LastModified`` unix timestamps; ``None`` means the
# timestamp could not be retrieved for that hotkey. Miner-supplied row
# timestamps are NEVER stored here because they are forgeable.
UploadTimesByHotkey = dict[str, float | None]


@dataclass(frozen=True)
class MinerCopycatSubmission:
    """All per-miner state that copycat detection needs for one window.

    Packaging this into a value object (rather than threading parallel
    ``digest_counter``, ``total_rollouts``, ``upload_time`` parameters
    through every layer from ``MinerSampler`` to ``CopycatTracker``) gives
    us two concrete wins:

    1. The public ``detect_cheaters`` API takes one argument per miner
       instead of three, and a new copycat signal added in a later tier
       (e.g. a ``prompt_digest_counter`` for Tier 1's cross-hotkey prompt
       collision check) becomes a new field on this dataclass rather than
       yet another parallel parameter plumbed through four modules.
    2. Callers are forced to assemble all of a miner's state in one place
       (currently :meth:`WindowProcessor.process_window`), which means
       there is exactly one point where ``upload_time`` can be forgotten,
       rather than four independent opportunities to drop it on the floor
       and silently reintroduce the victim-punishment bug.

    Fields:
        digest_counter: Counter of completion-token digests as produced by
            :func:`grail.shared.digest.compute_completion_digest`.
        total_rollouts: Number of rollouts in the miner's submitted parquet.
            Used as the denominator for the overlap ratio.
        upload_time: Unix timestamp from the S3 ``LastModified`` header of
            the parquet object, propagated from
            ``file_exists_with_deadline``. ``None`` means the timestamp was
            not retrievable; in that case, directional attribution for any
            pair involving this miner abstains rather than gating the
            wrong side. NEVER populate this from miner-supplied row data.
    """

    digest_counter: Counter[str]
    total_rollouts: int
    upload_time: float | None = None


@dataclass(frozen=True)
class CopycatViolation:
    """Details about a detected copycat relationship between two miners.

    When directional attribution succeeds (both miners have an S3 LastModified
    timestamp and the timestamps differ), ``copier`` is the hotkey that
    uploaded later and ``victim`` is the other one. Only the copier is added
    to the cheater set for gating; the victim is left intact. When direction
    cannot be resolved (either upload time is missing, or they are equal
    within resolution), ``copier`` and ``victim`` are both ``None`` and the
    pair is logged but not gated.
    """

    miner_a: str
    miner_b: str
    shared: int
    denominator: int
    ratio: float
    threshold: float
    scope: Literal["window", "interval"]
    window_start: int
    copier: str | None = None
    victim: str | None = None


@dataclass(frozen=True)
class _PairAttribution:
    """Result of directional attribution for a violating miner pair.

    This is an internal value object; its sole purpose is to decouple the
    "who uploaded later" decision from the "gate the copier and build a
    violation record" logic inside :meth:`CopycatTracker._find_cheaters`.
    Making it a named, frozen dataclass (rather than a bare tuple or inline
    ``None`` checks) means the two downstream consumers — gating and logging
    — read the same property rather than repeating the None-checks.
    """

    copier: str | None
    victim: str | None

    @property
    def is_resolved(self) -> bool:
        """True when both parties are known and distinct."""
        return self.copier is not None and self.victim is not None

    @classmethod
    def unresolved(cls) -> _PairAttribution:
        """Sentinel for a pair whose direction cannot be determined.

        Callers MUST treat this as "log but do not gate". Returning this
        value is how we guarantee the pre-Tier-0 victim-punishment bug
        cannot regress: an unresolved attribution carries no hotkey to add
        to the cheater set.
        """
        return cls(copier=None, victim=None)

    @classmethod
    def resolved(cls, copier: str, victim: str) -> _PairAttribution:
        """Direction has been determined: ``copier`` uploaded after ``victim``."""
        return cls(copier=copier, victim=victim)


def _attribute_pair_direction(
    miner_a: str,
    miner_b: str,
    uploads: Mapping[str, float | None],
) -> _PairAttribution:
    """Decide which side of a violating pair is the copier.

    Uses only object-store ``LastModified`` timestamps propagated from
    ``file_exists_with_deadline`` (``grail/infrastructure/comms.py:988``).
    Never uses miner-supplied row timestamps — those are forgeable and would
    reintroduce the bug this module is designed to prevent.

    Abstains (returns :meth:`_PairAttribution.unresolved`) when:

    - either miner's upload time is unknown, or
    - the two upload times are equal within resolution.

    We prefer a missed detection over punishing a victim, so abstention is
    the safe default whenever the signal is ambiguous.

    This function is intentionally module-level and stateless so it is
    trivially testable and so future detectors (e.g. a multi-signal variant
    in Tier 1) can reuse it without instantiating the tracker.
    """
    t_a = uploads.get(miner_a)
    t_b = uploads.get(miner_b)
    if t_a is None or t_b is None:
        return _PairAttribution.unresolved()
    if t_a == t_b:
        return _PairAttribution.unresolved()
    if t_a > t_b:
        return _PairAttribution.resolved(copier=miner_a, victim=miner_b)
    return _PairAttribution.resolved(copier=miner_b, victim=miner_a)


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
        # Per-miner upload timestamp (S3 LastModified) archived across the
        # current interval. Only real (non-None) timestamps are ever
        # written. We type the value as ``float | None`` to match
        # :data:`UploadTimesByHotkey` exactly so the map can be passed
        # straight into ``_find_cheaters`` without a second copy or a
        # covariant Mapping wrapper. Used for interval-scope directional
        # attribution so a pair whose overlap accumulated across multiple
        # windows can still be attributed when one miner is absent from
        # the current window. See the regression test
        # ``test_interval_attribution_survives_victim_going_offline``.
        self.interval_upload_times: UploadTimesByHotkey = {}

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
        self.interval_upload_times.clear()

    def ingest_window(
        self,
        window_start: int,
        submissions: dict[str, MinerCopycatSubmission],
    ) -> tuple[
        set[str],
        list[CopycatViolation],
        set[str],
        list[CopycatViolation],
        list[CopycatViolation],
        list[CopycatViolation],
    ]:
        """Ingest a window of submissions and compute copycat cheaters.

        Args:
            window_start: Window start block number (used in violation details).
            submissions: Per-hotkey :class:`MinerCopycatSubmission` records.
                Each submission bundles the completion-digest counter, the
                total rollout count, and the S3 ``LastModified`` timestamp
                for that miner. Caller is
                :meth:`WindowProcessor.process_window`, which assembles the
                submissions from ``MinerResults`` plus the
                ``discover_active_miners`` upload-time map.

        Returns:
            Tuple of:
            - window_cheaters: Copiers flagged at the window scope (victims not included).
            - window_details: Per-pair violation details for window scope.
            - interval_cheaters: Copiers flagged at the interval scope.
            - interval_details: Per-pair violation details for interval scope.
            - window_all_pairs: All observed miner pairs this window.
            - interval_all_pairs: All observed miner pairs for the current interval.
        """
        window_pair_overlap: defaultdict[frozenset[str], int] = defaultdict(int)
        window_totals = {miner: s.total_rollouts for miner, s in submissions.items()}
        # Extract the upload-time view once so _find_cheaters does not need
        # to know about MinerCopycatSubmission internals; keeps the
        # attribution helper decoupled from the submission schema.
        uploads: UploadTimesByHotkey = {miner: s.upload_time for miner, s in submissions.items()}

        digest_map: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for miner, submission in submissions.items():
            for digest, count in submission.digest_counter.items():
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

        for miner, submission in submissions.items():
            self.interval_totals[miner] += submission.total_rollouts
            # Archive the most recent upload time for each miner seen in
            # this interval. Interval-scope attribution must reference this
            # map, not the current window's uploads, because
            # interval_pair_overlap accumulates across windows and one of
            # the pair may be absent from the current window. Without this
            # archive, absent victims would cause _attribute_pair_direction
            # to abstain and the accumulated cheat would escape gating.
            if submission.upload_time is not None:
                self.interval_upload_times[miner] = submission.upload_time

        window_cheaters, window_details = self._find_cheaters(
            pair_overlap=window_pair_overlap,
            totals=window_totals,
            threshold=COPYCAT_WINDOW_THRESHOLD,
            scope="window",
            window_start=window_start,
            uploads=uploads,
        )
        interval_cheaters, interval_details = self._find_cheaters(
            pair_overlap=self.interval_pair_overlap,
            totals=self.interval_totals,
            threshold=COPYCAT_INTERVAL_THRESHOLD,
            scope="interval",
            window_start=window_start,
            uploads=self.interval_upload_times,
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
        uploads: Mapping[str, float | None],
    ) -> tuple[set[str], list[CopycatViolation]]:
        """Return flagged copiers and detailed violations for a given scope.

        The ratio criterion is ``shared / min(total_a, total_b) >= threshold``.

        Directional attribution:
            When a pair exceeds the threshold, the later uploader (by S3
            LastModified in ``uploads``) is identified as the copier and is
            the *only* hotkey added to the cheater set. The earlier
            uploader (the victim) is deliberately left intact. When
            direction cannot be resolved (see :func:`_attribute_pair_direction`
            for the abstain criteria), the violation is logged but NEITHER
            side is gated. We prefer a missed detection over punishing a
            victim.

        Returns:
            Tuple of:
            - flagged: Copiers appearing in at least one directionally
              resolved violating pair. Victims are never included.
            - details: One record per violating pair. When direction is
              resolved, the record carries ``copier`` and ``victim``; when
              unresolved, both are ``None``.
        """
        flagged: set[str] = set()
        details: list[CopycatViolation] = []
        for pair_key, shared in pair_overlap.items():
            if shared <= 0:
                continue
            miner_a, miner_b = tuple(pair_key)
            denominator = min(totals.get(miner_a, 0), totals.get(miner_b, 0))
            if denominator <= 0:
                continue
            ratio = shared / float(denominator)
            if ratio < threshold:
                continue

            attribution = _attribute_pair_direction(miner_a, miner_b, uploads)
            if not attribution.is_resolved:
                # Log the violation but do not gate either side. This is
                # the explicit fix for the pre-Tier-0 victim-punishment
                # bug: we used to ``flagged.update((miner_a, miner_b))``
                # here, which zeroed the victim's metrics alongside the
                # attacker's. On the subnet 81 investigation this was
                # actively penalising UID 53 and UID 7 (Oriea) while the
                # real copycats (UIDs 20/229) stayed below the window
                # threshold.
                logger.warning(
                    "Copycat threshold exceeded but upload-time direction "
                    "unresolved; suppressing gating. miner_a=%s miner_b=%s "
                    "shared=%d denom=%d ratio=%.3f threshold=%.2f scope=%s "
                    "window=%d",
                    miner_a,
                    miner_b,
                    shared,
                    denominator,
                    ratio,
                    threshold,
                    scope,
                    window_start,
                )
            elif attribution.copier is not None:
                flagged.add(attribution.copier)

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
                    copier=attribution.copier,
                    victim=attribution.victim,
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
        submissions: dict[str, MinerCopycatSubmission],
        uid_by_hotkey: dict[str, int],
        monitor: Any | None = None,
    ) -> tuple[set[str], set[str], list[CopycatViolation]]:
        """Detect copycat cheaters for a window.

        Args:
            window: Window start block number.
            submissions: Per-hotkey :class:`MinerCopycatSubmission` records
                assembled by :meth:`WindowProcessor.process_window`. Each
                record bundles the miner's digest counter, rollout total,
                and S3 ``LastModified`` upload timestamp; there is no
                parallel ``uploads_by_hotkey`` parameter because everything
                the detector needs for one miner lives in one object.
            uid_by_hotkey: Mapping of hotkey to UID (for monitor namespacing).
            monitor: Optional monitoring client.

        Returns:
            Tuple of:
            - window_cheaters: Copiers exceeding the window threshold
              (victims excluded by directional attribution).
            - interval_cheaters: Copiers exceeding the interval threshold.
            - all_violations: All violation details for logging.
        """
        if not submissions:
            return set(), set(), []

        # Time the copycat detection if monitor available
        timer_ctx = (
            monitor.timer("profiling/copycat_detector") if monitor else contextlib.nullcontext()
        )

        with timer_ctx:
            (
                window_cheaters,
                window_violation_details,
                interval_cheaters,
                interval_violation_details,
                window_all_pairs,
                interval_all_pairs,
            ) = COPYCAT_TRACKER.ingest_window(window, submissions)

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

        # Log all violations. When direction is resolved (copier/victim set),
        # surface them explicitly so operators can tell at a glance who was
        # gated and who was the victim. When direction is unresolved, log as
        # an unresolved pair — only the copier (if any) is gated by
        # ``_find_cheaters``, so this is purely informational.
        all_violations = window_violation_details + interval_violation_details
        for violation in all_violations:
            if violation.copier is not None and violation.victim is not None:
                logger.warning(
                    "Copycat gated: copier=%s victim=%s shared=%d denom=%d "
                    "ratio=%.3f threshold=%.2f scope=%s window=%d",
                    violation.copier,
                    violation.victim,
                    violation.shared,
                    violation.denominator,
                    violation.ratio,
                    violation.threshold,
                    violation.scope,
                    violation.window_start,
                )
            else:
                logger.warning(
                    "Copycat overlap (direction unresolved, no gating): "
                    "miners %s & %s shared=%d denom=%d ratio=%.3f "
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
            from ..logging_utils import miner_log_context

            uid = uid_by_hotkey.get(cheater, cheater)
            scopes = {v.scope for v in violation_map.get(cheater, [])}
            ratios = ", ".join(f"{v.ratio:.3f}" for v in violation_map.get(cheater, []))

            with miner_log_context(uid, window):
                logger.warning(
                    f"Copycat gating applied "
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
