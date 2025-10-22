"""Utilities for window timing and logging across neurons."""

import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def calculate_blocks_to_window(current_block: int, target_window: int) -> int:
    """Calculate blocks remaining until the target window.

    Args:
        current_block: Current blockchain block
        target_window: Target window start block

    Returns:
        Number of blocks remaining (0 if already past target)
    """
    return max(0, target_window - current_block)


def estimate_time_to_window(
    blocks_remaining: int, secs_per_block: float = 12.0
) -> tuple[timedelta, datetime]:
    """Estimate time remaining until window and compute ETA.

    Args:
        blocks_remaining: Number of blocks to wait
        secs_per_block: Estimated seconds per block (Bittensor mainnet: ~12s)

    Returns:
        Tuple of (timedelta for human-readable duration, datetime for ETA)
    """
    est_seconds = int(blocks_remaining * secs_per_block)
    duration = timedelta(seconds=est_seconds)
    eta = datetime.now() + duration
    return duration, eta


def format_duration(duration: timedelta) -> str:
    """Format timedelta as human-readable string without microseconds.

    Args:
        duration: Time duration

    Returns:
        Formatted string like "1:23:45"
    """
    return str(duration).split(".")[0]


def calculate_next_window(last_processed_window: int, window_length: int) -> int:
    """Calculate the next window to be processed.

    Args:
        last_processed_window: Last successfully processed window start block
        window_length: Blocks per window

    Returns:
        Next window start block (0 if never processed, else last + length)
    """
    if last_processed_window < 0:
        return 0
    return last_processed_window + window_length


def log_window_wait_initial(
    current_block: int,
    last_processed_window: int,
    window_length: int,
    secs_per_block: float = 12.0,
) -> None:
    """Log initial message when starting to wait for a window.

    Uses INFO level with emoji for visibility. Logs:
    - Next window to be processed
    - Current block and last processed window
    - Time estimate with duration and ETA

    Args:
        current_block: Current block
        last_processed_window: Last successfully processed window
        window_length: Blocks per window
        secs_per_block: Estimated seconds per block (default 12s for Bittensor)
    """
    # Calculate when next window becomes available
    next_window = calculate_next_window(last_processed_window, window_length)
    blocks_to_wait = calculate_blocks_to_window(current_block, next_window)
    duration, eta = estimate_time_to_window(blocks_to_wait, secs_per_block)

    logger.info(
        "⏳ Waiting for window %d (current: block %d, last processed: %d) | ETA: %s (~%s)",
        next_window,
        current_block,
        last_processed_window,
        eta.strftime("%H:%M:%S"),
        format_duration(duration),
    )


def log_window_wait_periodic(
    next_window: int,
    elapsed_seconds: int,
) -> None:
    """Log periodic status update while waiting for window.

    Uses INFO level with emoji. Logs elapsed time since start of wait.
    Called every 120 seconds during wait.

    Args:
        next_window: Next window being waited for
        elapsed_seconds: Seconds elapsed since starting to wait
    """
    elapsed = timedelta(seconds=elapsed_seconds)
    logger.info(
        "⏳ Still waiting for window %d | Elapsed: %s",
        next_window,
        format_duration(elapsed),
    )


class WindowWaitTracker:
    """Tracks window wait state across multiple log calls.

    Manages timing for initial and periodic logging to avoid spam.
    """

    def __init__(self, log_interval_secs: int = 120) -> None:
        """Initialize tracker.

        Args:
            log_interval_secs: Seconds between periodic log messages (default 120s = 2min)
        """
        self.log_interval_secs = log_interval_secs
        self._start_time: float | None = None
        self._last_log_time: float | None = None

    def should_log_initial(self) -> bool:
        """Check if we should log the initial wait message.

        Returns:
            True on first call (tracker not yet active)
        """
        if self._start_time is None:
            self._start_time = time.monotonic()
            self._last_log_time = self._start_time
            return True
        return False

    def should_log_periodic(self) -> bool:
        """Check if enough time has passed for periodic log.

        Returns:
            True if log_interval_secs have elapsed since last log
        """
        if self._start_time is None or self._last_log_time is None:
            return False

        current_time = time.monotonic()
        last_log_time = self._last_log_time  # Type narrowing helper
        if current_time - last_log_time >= self.log_interval_secs:
            self._last_log_time = current_time
            return True
        return False

    def get_elapsed_seconds(self) -> int:
        """Get seconds elapsed since tracking started.

        Returns:
            Elapsed seconds (0 if not started)
        """
        if self._start_time is None:
            return 0
        return int(time.monotonic() - self._start_time)

    def reset(self) -> None:
        """Reset tracker state (e.g., when window becomes available)."""
        self._start_time = None
        self._last_log_time = None
