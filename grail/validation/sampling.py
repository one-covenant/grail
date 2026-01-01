"""Miner discovery and sampling for validation.

Provides deterministic, fair sampling of miners for validation windows.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import deque
from typing import Any

from ..shared.constants import MIN_ROLLOUT_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)


class MinerSampler:
    """Handles miner discovery and deterministic sampling for validation.

    This class encapsulates the logic for:
    1. Discovering active miners (those with window files)
    2. Deterministic sampling based on selection history
    3. Fair distribution across validation windows
    4. Rolling history management for coverage metrics

    All sampling is deterministic given the same inputs, ensuring
    validators agree on which miners to check.
    """

    def __init__(
        self,
        sample_rate: float,
        sample_min: int,
        sample_max: int | None,
        concurrency: int = 8,
    ):
        """Initialize miner sampler.

        Args:
            sample_rate: Fraction of active miners to sample (0.0-1.0)
            sample_min: Minimum number of miners to sample
            sample_max: Maximum number of miners to sample (None = unlimited)
            concurrency: Max concurrent file existence checks
        """
        self._sample_rate = sample_rate
        self._sample_min = sample_min
        self._sample_max = sample_max
        self._concurrency = concurrency

        logger.info(
            "Initialized MinerSampler (rate=%.2f, min=%d, max=%s, concurrency=%d)",
            sample_rate,
            sample_min,
            sample_max or "unlimited",
            concurrency,
        )

    async def discover_active_miners(
        self,
        meta_hotkeys: list[str],
        window: int,
        chain_manager: Any,
        uid_by_hotkey: dict[str, int] | None = None,
        heartbeat_callback: Any = None,
        deadline_ts: float | None = None,
    ) -> list[str]:
        """Find miners with window files for the given window.

        Active miners are those that uploaded:
        `grail/windows/{hotkey}-window-{window}.parquet`

        This method checks file existence concurrently with bounded parallelism
        and proper timeout handling.

        Args:
            meta_hotkeys: All hotkeys from metagraph
            window: Window start block number
            chain_manager: Chain manager for bucket credentials
            uid_by_hotkey: Optional mapping for logging
            heartbeat_callback: Optional callback to update watchdog heartbeat
            deadline_ts: If set, require LastModified <= this deadline (unix seconds)

        Returns:
            List of hotkeys with available window files
        """
        semaphore = asyncio.Semaphore(self._concurrency)
        late_counter: dict[str, int] = {"count": 0}
        too_small_counter: dict[str, int] = {"count": 0}

        async def _check(hotkey: str) -> tuple[str, bool]:
            filename = f"grail/windows/{hotkey}-window-{window}.parquet"
            bucket = chain_manager.get_bucket_for_hotkey(hotkey)
            uid = uid_by_hotkey.get(hotkey) if uid_by_hotkey else None
            miner_id = f"uid={uid}" if uid is not None else f"hotkey={hotkey[:12]}..."

            # Skip miners without a committed bucket (no fallback)
            if bucket is None:
                return hotkey, False

            async with semaphore:
                # Update heartbeat to prevent watchdog timeout during long
                # operations. With many miners (100+), even with timeouts
                # this can take several minutes total.
                if heartbeat_callback:
                    try:
                        heartbeat_callback()
                    except Exception:
                        pass

                start_time = time.time()
                try:
                    # Single HEAD request checks: existence, size >= 200 bytes, and deadline
                    from ..infrastructure.comms import file_exists_with_deadline

                    exists, was_late, too_small, upload_time = await asyncio.wait_for(
                        file_exists_with_deadline(
                            key=filename,
                            credentials=bucket,
                            use_write=False,
                            max_upload_time=deadline_ts,
                            min_size_bytes=MIN_ROLLOUT_FILE_SIZE_BYTES,
                        ),
                        timeout=6.0,
                    )

                    # Check if file is too small (0 rollouts)
                    if too_small:
                        too_small_counter["count"] = int(too_small_counter.get("count", 0)) + 1
                        logger.debug(
                            "Too small window file ignored for %s window=%s (likely 0 rollouts)",
                            miner_id,
                            window,
                        )
                        return hotkey, False

                    # Check if file was late
                    if was_late:
                        late_counter["count"] = int(late_counter.get("count", 0)) + 1
                        logger.info(
                            "Late upload ignored for %s window=%s uploaded_at=%.0f deadline=%.0f",
                            miner_id,
                            window,
                            upload_time or -1,
                            deadline_ts,
                        )
                        return hotkey, False

                    return hotkey, bool(exists)
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    logger.debug(
                        "Window file check TIMEOUT for %s window=%s after %.2fs",
                        miner_id,
                        window,
                        elapsed,
                    )
                    return hotkey, False
                except Exception as e:
                    logger.error(f"Error checking window file for hotkey {hotkey}: {e}")
                    return hotkey, False

        results = await asyncio.gather(*(_check(hk) for hk in meta_hotkeys))
        active = [hk for hk, ok in results if ok]

        logger.info(
            "Discovered %d/%d active miners for window %d", len(active), len(meta_hotkeys), window
        )
        if too_small_counter.get("count", 0) > 0:
            logger.info(
                "Filtered out %d miners with too-small files (0 rollouts) for window %d",
                int(too_small_counter.get("count", 0)),
                window,
            )
        if deadline_ts is not None and late_counter.get("count", 0) > 0:
            logger.info(
                "Ignored %d late miners for window %d",
                int(late_counter.get("count", 0)),
                window,
            )

        return active

    def select_miners_for_validation(
        self,
        active_hotkeys: list[str],
        window_hash: str,
        selection_counts: dict[str, int],
    ) -> list[str]:
        """Select miners for validation using deterministic sampling.

        Selection is:
        1. Fair - prioritizes miners with fewer recent selections
        2. Deterministic - same inputs always produce same output
        3. Bounded - respects min/max sample size constraints

        Args:
            active_hotkeys: List of active miner hotkeys
            window_hash: Block hash for deterministic tie-breaking
            selection_counts: Rolling count of selections per miner

        Returns:
            Deterministically selected subset of miners to validate
        """
        sample_size = self._compute_sample_size(len(active_hotkeys))
        if sample_size == 0:
            return []

        # Tie-breaking using window hash ensures deterministic selection
        def _tie_break(hk: str) -> int:
            dig = hashlib.sha256(f"{window_hash}:{hk}".encode()).digest()
            return int.from_bytes(dig[:8], "big")

        # Sort by (selection count, deterministic tie-breaker)
        # Miners with fewer selections come first
        ranked = sorted(
            active_hotkeys, key=lambda hk: (int(selection_counts.get(hk, 0)), _tie_break(hk))
        )

        selected = ranked[:sample_size]

        logger.info(
            "Selected %d/%d miners for validation (rate=%.2f)",
            len(selected),
            len(active_hotkeys),
            len(selected) / len(active_hotkeys) if active_hotkeys else 0.0,
        )

        return selected

    def update_rolling_history(
        self,
        history: deque[set[str]],
        counts: dict[str, int],
        new_set: set[str],
        horizon: int,
    ) -> None:
        """Update rolling window history and counts.

        Maintains a sliding window of miner sets and their cumulative counts
        over the last N windows (horizon).

        Args:
            history: Deque of miner sets per window
            counts: Cumulative counts per miner
            new_set: Set of miners for current window
            horizon: Number of windows to track

        Note:
            Modifies history and counts in-place.
        """
        # Remove oldest window if at capacity
        if len(history) >= horizon:
            old_set = history.popleft()
            for hk in old_set:
                counts[hk] = max(0, int(counts.get(hk, 0)) - 1)

        # Add new window
        history.append(new_set)
        for hk in new_set:
            counts[hk] = int(counts.get(hk, 0)) + 1

    def _compute_sample_size(self, active_count: int) -> int:
        """Compute sample size based on active miner count.

        Applies rate-based sampling with min/max constraints.

        Args:
            active_count: Number of active miners

        Returns:
            Number of miners to sample
        """
        if active_count <= 0:
            return 0

        # Rate-based sample size
        rate_k = int(math.ceil(active_count * self._sample_rate))

        # Apply min/max constraints
        k = max(self._sample_min, rate_k)
        if self._sample_max is not None:
            k = min(k, self._sample_max)

        # Can't sample more than available
        return min(k, active_count)
