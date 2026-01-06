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
import os
import signal
import threading
import time
from collections.abc import Callable

import bittensor as bt

from ..infrastructure.network import create_subtensor
from ..logging_utils import dump_asyncio_stacks
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

        # Watchdog state
        self._last_heartbeat: float = time.monotonic()

        # Registered cleanup callbacks to run during shutdown
        self._shutdown_callbacks: list[Callable[[], None]] = []

        # Optional window-tracking primitives (set by subclasses)
        self.window_changed: asyncio.Event | None = None  # noqa: UP045
        self._notify_loop: asyncio.AbstractEventLoop | None = None  # noqa: UP045
        self.current_block: int = 0
        self.current_window: int = 0

        # Track shutdown source for clearer logging
        self._shutdown_source: str | None = None

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
                self._shutdown_source = "normal_return"
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
            self._subtensor = await create_subtensor()  # type: ignore[assignment]
            logger.info("Connected to Bittensor")
        return self._subtensor  # type: ignore[return-value]

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

    def heartbeat(self) -> None:
        """Record liveness progress for the watchdog."""
        self._last_heartbeat = time.monotonic()

    def register_shutdown_callback(self, fn: Callable[[], None]) -> None:
        """Register a callable to run during cooperative shutdown."""
        self._shutdown_callbacks.append(fn)

    def start_watchdog(self, timeout_seconds: int = 600, grace_seconds: int = 10) -> None:
        """Start a background watchdog task on prolonged inactivity.

        Args:
            timeout_seconds: Max allowed time without heartbeat
            grace_seconds: Time for cooperative shutdown before hard-exit
        """
        task = asyncio.create_task(self._watchdog(timeout_seconds, grace_seconds))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _watchdog(self, timeout_seconds: int, grace_seconds: int) -> None:
        """Monitor heartbeats and terminate on stall."""
        # Sleep in smaller chunks to remain responsive
        sleep_step = max(1, timeout_seconds // 3)
        while True:
            await asyncio.sleep(sleep_step)
            elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > timeout_seconds:
                try:
                    logger.error(
                        "[WATCHDOG] No progress for %ss (>%ss). Initiating shutdown...",
                        f"{elapsed:.0f}",
                        timeout_seconds,
                    )
                except Exception:
                    pass

                # Emit compact asyncio task snapshot once before shutdown
                try:
                    await dump_asyncio_stacks(label="WATCHDOG")
                except Exception:
                    pass

                # Attempt cooperative shutdown first
                try:
                    self._shutdown_source = "watchdog"
                    loop = asyncio.get_running_loop()
                    loop.call_soon(asyncio.create_task, self._shutdown(signal.SIGTERM))
                except Exception:
                    pass

                # Give a short grace period, then hard-exit
                try:
                    await asyncio.sleep(max(1, grace_seconds))
                finally:
                    os._exit(1)

    @staticmethod
    def calculate_window(block: int) -> int:
        """Calculate the window start block for a given block number.

        Args:
            block: Current block number

        Returns:
            Window start block (aligned to WINDOW_LENGTH)

        Example:
            # Returns 12000 if WINDOW_LENGTH=1000
            window = self.calculate_window(12345)
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
            handler = self._make_shutdown_handler(sig)
            loop.add_signal_handler(sig, handler)

    def _make_shutdown_handler(self, sig: signal.Signals) -> Callable[[], None]:
        def _handler() -> None:
            # Log receipt of OS signal then schedule async teardown
            try:
                logger.warning("Received OS signal: %s", sig.name)
            except Exception:
                pass
            self._shutdown_source = "os_signal"
            asyncio.create_task(self._shutdown(sig))

        return _handler

    async def _shutdown(self, sig: signal.Signals) -> None:
        """Set stop flag, cancel tasks, and join threads.

        This function is idempotent and safe to call multiple times.
        """
        try:
            reason = self._shutdown_source or "unknown"
            if reason == "os_signal":
                logger.warning("Shutting down due to OS signal: %s", sig.name)
            elif reason == "watchdog":
                logger.warning("Shutting down due to watchdog timeout")
            elif reason == "normal_return":
                logger.info("Cooperative shutdown (reason=normal_return)")
            else:
                logger.warning("Shutting down (reason=%s, signal=%s)", reason, sig.name)
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

        # Invoke registered cleanup callbacks (best-effort)
        for cb in list(self._shutdown_callbacks):
            try:
                cb()
            except Exception:
                pass

        # Allow log buffers to flush
        await asyncio.sleep(0.1)
        # Reset for next lifecycle start if reused
        self._shutdown_source = None
