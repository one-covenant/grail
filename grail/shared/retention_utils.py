"""Shared checkpoint retention policy utilities.

This module provides a unified retention policy for determining which checkpoint
windows should be kept in both remote storage (publisher) and local cache (consumer).

For chained deltas, retention must keep entire chains from anchor (FULL) to tip.
"""

from __future__ import annotations

from grail.shared.constants import (
    CHECKPOINT_MILESTONE_INTERVAL,
    DELTA_BASE_INTERVAL,
    WINDOW_LENGTH,
)

SAFETY_MARGIN_WINDOWS = 5


def _anchor_stride() -> int:
    """Calculate the anchor stride (blocks between FULL checkpoints)."""
    return max(1, int(DELTA_BASE_INTERVAL)) * int(WINDOW_LENGTH)


def compute_retention_windows(
    current_window: int,
    bootstrap_windows: int = 10,
) -> set[int]:
    """Calculate which checkpoint windows should be retained.

    For chained deltas, we must keep entire chains from anchor (FULL) to tip.
    This ensures miners can always reconstruct the current state by:
    1. Starting from an anchor (FULL checkpoint)
    2. Applying sequential deltas to reach the current window

    Retention policy:
    - Keep all windows from current anchor to now (active chain)
    - Keep previous anchor and its entire chain (for miners catching up)
    - Keep milestone checkpoints (every CHECKPOINT_MILESTONE_INTERVAL)
    - Keep bootstrap windows (windows 0-N for initial network state)

    Args:
        current_window: Current window number
        bootstrap_windows: Number of initial windows to always keep (default 10)

    Returns:
        Set of window numbers to retain
    """
    if current_window < 0:
        return set()

    keep: set[int] = set()
    stride = _anchor_stride()

    # Bootstrap windows (0 to N, capped at current)
    keep.update(
        w for w in range(0, bootstrap_windows * WINDOW_LENGTH, WINDOW_LENGTH) if w <= current_window
    )

    # Current anchor with safety margin, and all windows from there to now
    current_anchor = max(
        0, (current_window // stride) * stride - SAFETY_MARGIN_WINDOWS * WINDOW_LENGTH
    )
    keep.update(range(current_anchor, current_window + 1, WINDOW_LENGTH))

    # Previous anchor chain (for miners catching up)
    prev_anchor = current_anchor - stride
    if prev_anchor >= 0:
        keep.update(range(prev_anchor, current_anchor, WINDOW_LENGTH))

    # Milestone checkpoints (long-term preservation)
    if CHECKPOINT_MILESTONE_INTERVAL > 0:
        interval = CHECKPOINT_MILESTONE_INTERVAL * WINDOW_LENGTH
        keep.update(range(0, current_window + 1, interval))

    return keep


def get_anchor_window(target_window: int) -> int:
    """Get the anchor window (nearest preceding FULL checkpoint) for a given window.

    Args:
        target_window: The window to find the anchor for

    Returns:
        The anchor window number
    """
    stride = _anchor_stride()
    return (target_window // stride) * stride


def is_anchor_window(window: int) -> bool:
    """Check if a window is an anchor (FULL checkpoint) window.

    Args:
        window: The window number to check

    Returns:
        True if this window is an anchor window
    """
    return window % _anchor_stride() == 0
