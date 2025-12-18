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
    keep: set[int] = set()
    if current_window < 0:
        return keep

    # Always keep bootstrap windows
    for i in range(bootstrap_windows):
        keep.add(i * WINDOW_LENGTH)

    # Calculate anchor stride (blocks between FULL checkpoints)
    delta_base_interval_windows = max(1, int(DELTA_BASE_INTERVAL))
    anchor_stride = delta_base_interval_windows * int(WINDOW_LENGTH)

    # Calculate current anchor (last FULL boundary)
    current_anchor = (current_window // anchor_stride) * anchor_stride

    # Keep all windows from current anchor to now (the active chain)
    w = current_anchor
    while w <= current_window:
        keep.add(w)
        w += WINDOW_LENGTH

    # Keep previous anchor and its chain (for miners catching up)
    prev_anchor = current_anchor - anchor_stride
    if prev_anchor >= 0:
        keep.add(prev_anchor)
        # Keep entire chain from previous anchor to current anchor
        w = prev_anchor
        while w < current_anchor:
            keep.add(w)
            w += WINDOW_LENGTH

    # Keep milestone checkpoints (long-term preservation)
    if CHECKPOINT_MILESTONE_INTERVAL > 0:
        interval_blocks = CHECKPOINT_MILESTONE_INTERVAL * WINDOW_LENGTH
        if interval_blocks > 0:
            milestone = (current_window // interval_blocks) * interval_blocks
            while milestone >= 0:
                keep.add(milestone)
                milestone -= interval_blocks

    return keep


def get_anchor_window(target_window: int) -> int:
    """Get the anchor window (nearest preceding FULL checkpoint) for a given window.

    Args:
        target_window: The window to find the anchor for

    Returns:
        The anchor window number
    """
    delta_base_interval_windows = max(1, int(DELTA_BASE_INTERVAL))
    anchor_stride = delta_base_interval_windows * int(WINDOW_LENGTH)
    return (target_window // anchor_stride) * anchor_stride


def is_anchor_window(window: int) -> bool:
    """Check if a window is an anchor (FULL checkpoint) window.

    Args:
        window: The window number to check

    Returns:
        True if this window is an anchor window
    """
    delta_base_interval_windows = max(1, int(DELTA_BASE_INTERVAL))
    anchor_stride = delta_base_interval_windows * int(WINDOW_LENGTH)
    return window % anchor_stride == 0
