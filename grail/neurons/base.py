"""Base neuron lifecycle for long-running GRAIL processes.

Provides:
- Signal-safe start/stop handling (SIGINT, SIGTERM)
- A shared stop_event to coordinate shutdown
- Shared subtensor management with automatic reconnection
- Window calculation utilities
- Optional window change signaling for future event-driven orchestration

Subclasses should override `run()` and may push background tasks/threads into
the provided registries for coordinated teardown.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
from typing import Optional

import bittensor as bt

from ..infrastructure.network import create_subtensor
from ..shared.constants import WINDOW_LENGTH

logger = logging.getLogger(__name__)


class BaseNeuron:
    """Shared lifecycle for miner, validator, and trainer neurons.

    Subclasses should override `run()` and should not block the event loop
    on network/IO operations. Use asyncio with timeouts for reliability.
    """

    def __init__(self) -> None:
        # Cooperative shutdown signal for all async logic
        self.stop_event: asyncio.Event = asyncio.Event()

        # Background bookkeeping for structured teardown
        self._bg_tasks: set[asyncio.Task] = set()
        self._threads: list[threading.Thread] = []

        # Shared subtensor instance (lazy-initialized)
        self._subtensor: bt.subtensor | None = None

        # Optional window-tracking primitives (set by subclasses if needed)
        self.window_changed: Optional[asyncio.Event] = None  # noqa: UP045
        self._notify_loop: Optional[asyncio.AbstractEventLoop] = None  # noqa: UP045
        self.current_block: int = 0
        self.current_window: int = 0

    async def main(self) -> None:
        """Install signal handlers, initialize optional events, and run."""
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)

        # Initialize optional window signaling. Subclasses may call
        # `self.window_changed.set()` from background listeners.
        self.window_changed = asyncio.Event()
        self._notify_loop = loop

        try:
            await self.run()
        finally:
            # Ensure graceful shutdown even if run() completes normally
            if not self.stop_event.is_set():
                await self._shutdown(signal.SIGTERM)

    async def run(self) -> None:
        """Entry point for subclasses to implement their main loop."""
        raise NotImplementedError

    async def get_subtensor(self) -> bt.subtensor:
        """Get or create the shared subtensor instance.

        This method lazy-initializes a single subtensor connection and caches
        it for reuse. Subclasses should call this instead of managing their own
        subtensor instances.

        Returns:
            Initialized async subtensor instance

        Example:
            subtensor = await self.get_subtensor()
            current_block = await subtensor.get_current_block()
        """
        if self._subtensor is None:
            logger.info("Making Bittensor connection...")
            self._subtensor = await create_subtensor()
            logger.info("Connected to Bittensor")
        return self._subtensor

    def reset_subtensor(self) -> None:
        """Clear the cached subtensor instance.

        Call this when a subtensor operation fails to force reconnection
        on the next `get_subtensor()` call.

        Example:
            try:
                block = await subtensor.get_current_block()
            except Exception:
                self.reset_subtensor()  # Force reconnect next time
                raise
        """
        self._subtensor = None

    @staticmethod
    def calculate_window(block: int) -> int:
        """Calculate the window start block for a given block number.

        Args:
            block: Current block number

        Returns:
            Window start block (aligned to WINDOW_LENGTH)

        Example:
            window = self.calculate_window(12345)  # Returns 12000 if WINDOW_LENGTH=1000
        """
        return (block // WINDOW_LENGTH) * WINDOW_LENGTH

    async def wait_until_window(self, target_window: int) -> None:
        """Block until `current_window` >= target_window or shutdown.

        Uses the optional `window_changed` event if set; otherwise, polls
        at a low cadence. Subclasses are responsible for updating
        `current_window`.
        """
        evt = self.window_changed
        # Clear a stale signal so a prior set() doesn't wake immediately
        if evt is not None and evt.is_set():
            evt.clear()

        while not self.stop_event.is_set():
            if self.current_window >= target_window:
                return
            if evt is None:
                await asyncio.sleep(0.5)
            else:
                await evt.wait()
                evt.clear()

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register handlers to trigger cooperative shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._shutdown(s)),  # type: ignore[misc]
            )

    async def _shutdown(self, sig: signal.Signals) -> None:
        """Set stop flag, cancel tasks, and join threads.

        This function is idempotent and safe to call multiple times.
        """
        try:
            logger.warning("Shutting down due to signal: %s", sig.name)
        except Exception:
            pass

        if self.stop_event.is_set():
            # Already shutting down
            return
        self.stop_event.set()

        # Cancel and await background tasks
        for task in list(self._bg_tasks):
            task.cancel()
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)

        # Join helper threads
        for th in self._threads:
            try:
                th.join(timeout=2)
            except Exception:
                pass

        # Allow log buffers to flush
        await asyncio.sleep(0.1)
