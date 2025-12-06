"""Inter-process communication primitives for async trainer.

Provides a unified interface for coordination between:
- Orchestrator (main process)
- Training process (GPU owner)
- Upload worker process

Design:
- IPCChannels: Dataclass holding all shared primitives
- Helper methods encapsulate common IPC patterns
- Filesystem markers kept as crash recovery backup
"""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IPCChannels:
    """Shared IPC primitives for inter-process communication.

    Passed as a single object to child processes for coordination.
    All primitives are process-safe (multiprocessing types).

    Attributes:
        stop: Signal to shutdown all processes
        pause_requested: Orchestrator requests training to pause
        pause_confirmed: Training confirms it has paused
        heartbeat: Atomic timestamp for liveness monitoring
        snapshot_queue: Messages from training to upload worker
    """

    # Lifecycle
    stop: multiprocessing.Event = field(default_factory=multiprocessing.Event)

    # Pause coordination (orchestrator ↔ training)
    pause_requested: multiprocessing.Event = field(default_factory=multiprocessing.Event)
    pause_confirmed: multiprocessing.Event = field(default_factory=multiprocessing.Event)

    # Heartbeat monitoring (training → orchestrator)
    # Using Any type because multiprocessing.Value is not generic
    heartbeat: Any = field(default=None)

    # Snapshot coordination (training → upload worker)
    snapshot_queue: Any = field(default=None)

    def __post_init__(self) -> None:
        """Initialize Value and Queue if not provided."""
        if self.heartbeat is None:
            self.heartbeat = multiprocessing.Value("d", 0.0)
        if self.snapshot_queue is None:
            self.snapshot_queue = multiprocessing.Queue()

    # ────────────────────────────────────────────────────────────────────────────
    # Heartbeat Methods
    # ────────────────────────────────────────────────────────────────────────────

    def update_heartbeat(self) -> None:
        """Update heartbeat timestamp atomically.

        Called by training process to signal liveness.
        """
        with self.heartbeat.get_lock():
            self.heartbeat.value = time.time()

    def get_heartbeat_age(self) -> float:
        """Get seconds since last heartbeat.

        Returns:
            Age in seconds, or infinity if no heartbeat yet
        """
        with self.heartbeat.get_lock():
            ts = self.heartbeat.value
        if ts == 0.0:
            return float("inf")
        return time.time() - ts

    # ────────────────────────────────────────────────────────────────────────────
    # Pause Coordination Methods
    # ────────────────────────────────────────────────────────────────────────────

    def request_pause(self) -> None:
        """Signal training process to pause for evaluation."""
        self.pause_confirmed.clear()  # Clear stale confirmation
        self.pause_requested.set()

    def clear_pause(self) -> None:
        """Signal training process to resume after evaluation."""
        self.pause_requested.clear()

    def confirm_pause(self) -> None:
        """Training process confirms it has paused."""
        self.pause_confirmed.set()

    def clear_pause_confirmed(self) -> None:
        """Clear pause confirmation for next cycle."""
        self.pause_confirmed.clear()

    def is_pause_requested(self) -> bool:
        """Check if pause has been requested."""
        return self.pause_requested.is_set()

    def wait_for_pause_confirmation(self, timeout: float) -> bool:
        """Wait for training to confirm pause.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if confirmed within timeout, False otherwise
        """
        return self.pause_confirmed.wait(timeout)

    # ────────────────────────────────────────────────────────────────────────────
    # Snapshot Queue Methods
    # ────────────────────────────────────────────────────────────────────────────

    def queue_snapshot(
        self,
        path: str,
        metadata: dict[str, Any],
        window: int,
    ) -> None:
        """Queue snapshot notification for upload worker.

        Args:
            path: Path to snapshot directory
            metadata: Snapshot metadata dict
            window: Current window number
        """
        self.snapshot_queue.put(
            {
                "type": "snapshot_ready",
                "path": path,
                "metadata": metadata,
                "window": window,
            }
        )


def create_ipc_channels() -> IPCChannels:
    """Factory function to create IPCChannels with all primitives initialized.

    Returns:
        Fully initialized IPCChannels instance
    """
    return IPCChannels()
