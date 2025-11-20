"""IPC coordinator for async trainer using multiprocessing primitives.

Provides efficient inter-process communication using Events and shared state
instead of filesystem polling.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Shared state for training coordination."""

    epoch_counter: multiprocessing.Value  # Shared integer
    last_heartbeat: multiprocessing.Value  # Shared float (timestamp)


class IPCCoordinator:
    """Coordinates training pause/resume using multiprocessing primitives.

    Replaces filesystem-based polling with instant event signaling.

    Architecture:
    - Main process signals pause via pause_requested event
    - Training process acknowledges via pause_acknowledged event
    - Main process signals resume via resume_requested event
    - Training process acknowledges via resume_acknowledged event
    """

    def __init__(self) -> None:
        """Initialize IPC coordinator with shared events."""
        # Pause coordination
        self.pause_requested = multiprocessing.Event()
        self.pause_acknowledged = multiprocessing.Event()

        # Resume coordination
        self.resume_requested = multiprocessing.Event()
        self.resume_acknowledged = multiprocessing.Event()

        # Shared state
        self.epoch_counter = multiprocessing.Value("i", 0)  # Signed int
        self.last_heartbeat = multiprocessing.Value("d", 0.0)  # Double (timestamp)

    def request_pause(self, timeout: float = 30.0) -> bool:
        """Request training to pause and wait for acknowledgment.

        Args:
            timeout: Maximum time to wait for acknowledgment (seconds)

        Returns:
            True if training acknowledged pause, False if timeout
        """
        logger.info("Requesting training pause...")
        self.pause_requested.set()

        # Wait for training to acknowledge
        if self.pause_acknowledged.wait(timeout=timeout):
            logger.info("Training pause acknowledged")
            return True
        else:
            logger.error("Training pause acknowledgment timeout after %.1fs", timeout)
            return False

    def acknowledge_pause(self) -> None:
        """Training process acknowledges pause request."""
        logger.info("Acknowledging pause request")
        self.pause_acknowledged.set()

    def wait_for_pause_request(self, check_interval: float = 0.1) -> bool:
        """Training process waits for pause request (non-blocking check).

        Args:
            check_interval: Not used (kept for API compatibility)

        Returns:
            True if pause requested, False otherwise
        """
        return self.pause_requested.is_set()

    def request_resume(self, timeout: float = 30.0) -> bool:
        """Request training to resume and wait for acknowledgment.

        Args:
            timeout: Maximum time to wait for acknowledgment (seconds)

        Returns:
            True if training acknowledged resume, False if timeout
        """
        logger.info("Requesting training resume...")

        # Clear pause flags
        self.pause_requested.clear()
        self.pause_acknowledged.clear()

        # Signal resume
        self.resume_requested.set()

        # Wait for training to acknowledge
        if self.resume_acknowledged.wait(timeout=timeout):
            logger.info("Training resume acknowledged")
            self.resume_acknowledged.clear()
            return True
        else:
            logger.error("Training resume acknowledgment timeout after %.1fs", timeout)
            return False

    def acknowledge_resume(self) -> None:
        """Training process acknowledges resume request."""
        logger.info("Acknowledging resume request")
        self.resume_acknowledged.set()

    def wait_for_resume_request(self, timeout: float | None = None) -> bool:
        """Training process waits for resume signal.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if resume requested, False if timeout
        """
        return self.resume_requested.wait(timeout=timeout)

    def update_heartbeat(self) -> None:
        """Update training heartbeat timestamp."""
        with self.last_heartbeat.get_lock():
            self.last_heartbeat.value = time.time()

    def get_heartbeat_age(self) -> float:
        """Get age of training heartbeat in seconds.

        Returns:
            Age in seconds, or infinity if never set
        """
        with self.last_heartbeat.get_lock():
            heartbeat_time = self.last_heartbeat.value

        if heartbeat_time == 0.0:
            return float("inf")

        return time.time() - heartbeat_time

    def get_epoch_counter(self) -> int:
        """Get current epoch counter.

        Returns:
            Current epoch number
        """
        with self.epoch_counter.get_lock():
            return self.epoch_counter.value

    def increment_epoch_counter(self) -> int:
        """Increment and return epoch counter.

        Returns:
            New epoch number
        """
        with self.epoch_counter.get_lock():
            self.epoch_counter.value += 1
            return self.epoch_counter.value
