"""Replay buffer for storing and sampling GRPO groups across multiple windows.

Provides abstract interface and concrete implementations for managing training data
from multiple windows with configurable sampling strategies.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from grail.trainer.algorithms.grpo import GRPOGroup

logger = logging.getLogger(__name__)


class ReplayBuffer(ABC):
    """Abstract base class for GRPO group replay buffers.

    Provides interface for storing groups from completed windows and sampling
    them deterministically for training. Implementations control eviction policy
    and sampling strategy.
    """

    @abstractmethod
    def add_window(self, window: int, groups: list[GRPOGroup]) -> None:
        """Add groups from a completed window.

        Args:
            window: Window number
            groups: List of GRPO groups from this window
        """

    @abstractmethod
    def sample_groups(self, max_groups: int, seed: int) -> list[GRPOGroup]:
        """Sample groups deterministically using seed.

        Args:
            max_groups: Maximum number of groups to sample
            seed: Random seed for deterministic sampling

        Returns:
            List of sampled GRPO groups
        """

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics.

        Returns:
            Dictionary with keys: windows, total_groups, memory_mb, oldest_window, newest_window
        """

    @abstractmethod
    def clear_before(self, window: int) -> None:
        """Remove all windows before specified window.

        Args:
            window: Window threshold (exclusive)
        """


class RecencyWeightedBuffer(ReplayBuffer):
    """Replay buffer with exponential decay weighting favoring recent windows.

    Sampling strategy:
    - Most recent window: recent_window_fraction of samples (default 50%)
    - Remaining samples distributed across older windows with exponential decay
    - Ensures diversity while prioritizing fresh data

    Example with 3 windows (oldest to newest: W1, W2, W3), max_groups=100:
    - W3 (newest): 50 groups (50%)
    - W2: 35 groups (35%, decay_factor=0.7 applied once)
    - W1 (oldest): 15 groups (15%, decay_factor=0.7 applied twice)
    """

    def __init__(
        self,
        max_windows: int,
        recent_window_fraction: float = 0.5,
        decay_factor: float = 0.7,
    ) -> None:
        """Initialize recency-weighted replay buffer.

        Args:
            max_windows: Maximum number of windows to retain
            recent_window_fraction: Fraction of samples from most recent window (0.0-1.0)
            decay_factor: Exponential decay for older windows (0.0-1.0)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if max_windows < 1:
            raise ValueError(f"max_windows must be >= 1, got {max_windows}")
        if not 0.0 <= recent_window_fraction <= 1.0:
            raise ValueError(
                f"recent_window_fraction must be in [0.0, 1.0], got {recent_window_fraction}"
            )
        if not 0.0 <= decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be in [0.0, 1.0], got {decay_factor}")

        self._max_windows = max_windows
        self._recent_fraction = recent_window_fraction
        self._decay_factor = decay_factor
        self._windows: dict[int, list[GRPOGroup]] = {}

    def add_window(self, window: int, groups: list[GRPOGroup]) -> None:
        """Add groups from a completed window."""
        if window in self._windows:
            logger.warning(
                "Window %s already exists in replay buffer, updating with %d groups",
                window,
                len(groups),
            )
            self._windows[window] = groups
            return

        self._windows[window] = groups

        # Evict oldest window if over capacity
        if len(self._windows) > self._max_windows:
            oldest_window = min(self._windows.keys())
            evicted_groups = len(self._windows[oldest_window])
            del self._windows[oldest_window]
            logger.info(
                "Replay buffer evicted oldest window %s (%d groups, capacity=%d)",
                oldest_window,
                evicted_groups,
                self._max_windows,
            )

    def sample_groups(self, max_groups: int, seed: int) -> list[GRPOGroup]:
        """Sample groups with recency weighting.

        Args:
            max_groups: Maximum number of groups to sample
            seed: Random seed for deterministic sampling

        Returns:
            List of sampled GRPO groups
        """
        if not self._windows or max_groups <= 0:
            return []

        rng = random.Random(seed)
        sorted_windows = sorted(self._windows.keys())  # Oldest to newest
        num_windows = len(sorted_windows)
        allocations = self._compute_allocations(max_groups, num_windows)

        # Sample from each window according to allocation
        sampled: list[GRPOGroup] = []
        for idx, window in enumerate(sorted_windows):
            available = self._windows[window]
            allocation = allocations[idx]
            k = min(allocation, len(available))
            if k > 0:
                sampled.extend(rng.sample(available, k))

        # Fill remaining quota from most recent window if undersampled
        if len(sampled) < max_groups:
            remaining_quota = max_groups - len(sampled)
            most_recent_window = sorted_windows[-1]
            available = self._windows[most_recent_window]
            unused = [g for g in available if g not in sampled]
            k = min(remaining_quota, len(unused))
            if k > 0:
                sampled.extend(rng.sample(unused, k))

        logger.debug(
            "Sampled %d groups from %d windows (requested=%d, seed=%d)",
            len(sampled),
            num_windows,
            max_groups,
            seed,
        )

        return sampled

    def _compute_allocations(self, max_groups: int, num_windows: int) -> list[int]:
        """Compute sample allocation per window with exponential decay.

        Args:
            max_groups: Total budget of groups to allocate
            num_windows: Number of windows in buffer

        Returns:
            List of integers (allocations) ordered from oldest to newest window
        """
        if num_windows == 1:
            return [max_groups]

        # Allocate to most recent window first
        recent_allocation = int(max_groups * self._recent_fraction)
        remaining = max_groups - recent_allocation

        # Distribute remaining budget across older windows with decay
        # weights[i] = decay^(num_windows - 1 - i) for i in [0, num_windows-2]
        weights = [self._decay_factor ** (num_windows - 1 - i) for i in range(num_windows - 1)]
        total_weight = sum(weights)

        if total_weight > 0:
            old_allocations = [int(remaining * (w / total_weight)) for w in weights]
        else:
            old_allocations = [0] * (num_windows - 1)

        # Return oldest → newest order
        return old_allocations + [recent_allocation]

    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        if not self._windows:
            return {
                "windows": 0,
                "total_groups": 0,
                "memory_mb": 0.0,
                "oldest_window": None,
                "newest_window": None,
            }

        total_groups = sum(len(groups) for groups in self._windows.values())
        memory_mb = (total_groups * 10_000) / (1024 * 1024)
        sorted_windows = sorted(self._windows.keys())

        return {
            "windows": len(self._windows),
            "total_groups": total_groups,
            "memory_mb": round(memory_mb, 2),
            "oldest_window": sorted_windows[0],
            "newest_window": sorted_windows[-1],
        }

    def clear_before(self, window: int) -> None:
        """Remove all windows before specified window.

        Args:
            window: Window threshold (exclusive)
        """
        to_remove = [w for w in self._windows if w < window]
        if not to_remove:
            return

        for w in to_remove:
            del self._windows[w]

        logger.info(
            "Cleared %d windows before window %s from replay buffer", len(to_remove), window
        )


def create_replay_buffer(
    buffer_type: str,
    max_windows: int,
    recent_window_fraction: float = 0.5,
    decay_factor: float = 0.7,
) -> ReplayBuffer:
    """Factory function for creating replay buffers.

    Args:
        buffer_type: Type of buffer ('recency_weighted')
        max_windows: Maximum number of windows to retain
        recent_window_fraction: Fraction of samples from most recent window
        decay_factor: Exponential decay for older windows

    Returns:
        Configured replay buffer instance

    Raises:
        ValueError: If buffer_type is not recognized
    """
    if buffer_type == "recency_weighted":
        return RecencyWeightedBuffer(
            max_windows=max_windows,
            recent_window_fraction=recent_window_fraction,
            decay_factor=decay_factor,
        )

    raise ValueError(f"Unknown buffer type: {buffer_type}")
