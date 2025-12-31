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

    RETENTION POLICY (updated for cold start recovery):
    - Keep previous anchor + current anchor chains (complete delta chains)
    - Keep milestone checkpoints for long-term preservation
    - Keep bootstrap windows for initial network state

    This ensures:
    1. Cold start validators can download the latest ready FULL checkpoint
    2. All deltas needed to reach current window are available
    3. No orphaned FULL checkpoints with missing delta chains

    Why keep previous anchor:
    - Latest anchor checkpoint may not be ready yet (still uploading)
    - Validators fall back to previous anchor checkpoint
    - Need complete delta chain from previous anchor to current

    Previous implementation used safety margin, causing issues:
    - Validators downloaded old FULL checkpoints outside retention window
    - Tried to apply deleted deltas, causing reconstruction failures

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

    # Latest FULL checkpoint (current anchor) + complete chain to current window
    # Keep from PREVIOUS anchor to ensure complete chain for any anchor checkpoint
    # This handles cases where latest anchor isn't ready yet, so validators use previous anchor
    current_anchor = (current_window // stride) * stride
    prev_anchor = max(0, current_anchor - stride)
    keep.update(range(prev_anchor, current_window + WINDOW_LENGTH, WINDOW_LENGTH))

    # Milestone checkpoints (long-term preservation)
    # These are kept for historical analysis, separate from operational chain
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
