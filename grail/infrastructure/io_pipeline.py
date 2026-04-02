"""Pipelined I/O scheduler for overlapping checkpoint, upload, and chain operations.

Implements three key optimizations:
1. Checkpoint prefetching: downloads the NEXT window's checkpoint while the
   current window generates rollouts. Saves ~1-5s of blocking download time.
2. Upload pipelining: uploads the current window's rollouts as a background
   task while the next window's generation begins. Saves ~5s per window.
3. Chain state prefetching: keeps a cached, auto-refreshing view of
   current_block and block_hash so generation loops avoid blocking RPC calls.

Thread safety: all public methods are async-safe. The class manages its own
background tasks and cancels them cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                         Checkpoint Prefetcher                               #
# --------------------------------------------------------------------------- #


@dataclass
class PrefetchResult:
    """Result of a checkpoint prefetch operation."""

    window: int
    success: bool
    path: Path | None = None
    error: str | None = None
    elapsed_sec: float = 0.0


class CheckpointPrefetcher:
    """Prefetch the next window's checkpoint while current window generates.

    Usage:
        prefetcher = CheckpointPrefetcher(checkpoint_manager)

        # At start of window N generation:
        prefetcher.prefetch(window_n + WINDOW_LENGTH)

        # At start of window N+1, before checkpoint load:
        result = await prefetcher.get_result(window_n + WINDOW_LENGTH)
        if result and result.success:
            # Checkpoint already cached, skip download
            ...
    """

    def __init__(self, checkpoint_manager: Any) -> None:
        self._manager = checkpoint_manager
        self._pending: dict[int, asyncio.Task[PrefetchResult]] = {}
        self._results: dict[int, PrefetchResult] = {}

    def prefetch(self, window: int) -> None:
        """Start prefetching checkpoint for the given window (non-blocking).

        If a prefetch for this window is already in progress or completed,
        this is a no-op.
        """
        if window in self._pending or window in self._results:
            return

        task = asyncio.create_task(
            self._do_prefetch(window),
            name=f"prefetch-ckpt-{window}",
        )
        self._pending[window] = task

        # Clean up completed tasks from previous windows
        stale = [w for w in self._results if w < window - 1]
        for w in stale:
            del self._results[w]

        logger.info("Checkpoint prefetch started for window %d", window)

    async def get_result(self, window: int, timeout: float = 30.0) -> PrefetchResult | None:
        """Get prefetch result, waiting up to timeout seconds if still in progress.

        Returns None if no prefetch was started for this window.
        """
        # Already completed
        if window in self._results:
            result = self._results.pop(window)
            self._pending.pop(window, None)
            return result

        # Still in progress
        task = self._pending.get(window)
        if task is None:
            return None

        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self._pending.pop(window, None)
            return result
        except TimeoutError:
            logger.warning(
                "Checkpoint prefetch for window %d timed out after %.1fs", window, timeout
            )
            return None
        except Exception as exc:
            logger.warning("Checkpoint prefetch for window %d failed: %s", window, exc)
            self._pending.pop(window, None)
            return PrefetchResult(window=window, success=False, error=str(exc))

    async def _do_prefetch(self, window: int) -> PrefetchResult:
        """Internal: download checkpoint for window into local cache."""
        t0 = time.monotonic()
        try:
            path = await self._manager.get_checkpoint(window)
            elapsed = time.monotonic() - t0
            if path is not None:
                result = PrefetchResult(window=window, success=True, path=path, elapsed_sec=elapsed)
                logger.info(
                    "Checkpoint prefetch completed: window=%d path=%s elapsed=%.2fs",
                    window,
                    path,
                    elapsed,
                )
            else:
                result = PrefetchResult(
                    window=window,
                    success=False,
                    error="checkpoint not ready",
                    elapsed_sec=elapsed,
                )
                logger.debug(
                    "Checkpoint prefetch: window %d not ready yet (%.2fs)", window, elapsed
                )
            self._results[window] = result
            return result
        except Exception as exc:
            elapsed = time.monotonic() - t0
            result = PrefetchResult(
                window=window, success=False, error=str(exc), elapsed_sec=elapsed
            )
            self._results[window] = result
            logger.warning(
                "Checkpoint prefetch failed: window=%d error=%s elapsed=%.2fs",
                window,
                exc,
                elapsed,
            )
            return result

    async def cancel_all(self) -> None:
        """Cancel all pending prefetch tasks."""
        for window, task in self._pending.items():
            if not task.done():
                task.cancel()
                logger.debug("Cancelled prefetch for window %d", window)
        self._pending.clear()
        self._results.clear()


# --------------------------------------------------------------------------- #
#                         Upload Pipeline                                     #
# --------------------------------------------------------------------------- #


@dataclass
class UploadResult:
    """Result of a background upload operation."""

    window: int
    success: bool
    elapsed_sec: float = 0.0
    error: str | None = None


class UploadPipeline:
    """Background upload that overlaps with the next window's generation.

    Usage:
        pipeline = UploadPipeline()

        # After generation completes:
        pipeline.submit(window, upload_coro)

        # Before next upload or at shutdown:
        result = await pipeline.await_previous()
    """

    def __init__(self) -> None:
        self._current_task: asyncio.Task[UploadResult] | None = None
        self._current_window: int | None = None

    def submit(
        self,
        window: int,
        upload_fn: Any,
        *args: Any,
    ) -> None:
        """Submit an upload to run in the background.

        If a previous upload is still running, it will NOT be cancelled.
        The caller should await_previous() before submitting if they need
        to ensure sequential uploads.

        Args:
            window: Window number for logging.
            upload_fn: Async callable to execute.
            *args: Arguments to pass to upload_fn.
        """
        self._current_window = window
        self._current_task = asyncio.create_task(
            self._do_upload(window, upload_fn, *args),
            name=f"upload-window-{window}",
        )
        logger.info("Background upload started for window %d", window)

    async def await_previous(self, timeout: float = 120.0) -> UploadResult | None:
        """Wait for the previous background upload to complete.

        Returns None if no upload was pending.
        """
        if self._current_task is None:
            return None

        try:
            result = await asyncio.wait_for(self._current_task, timeout=timeout)
            self._current_task = None
            return result
        except TimeoutError:
            logger.error(
                "Background upload for window %d timed out after %.1fs",
                self._current_window,
                timeout,
            )
            self._current_task = None
            return UploadResult(
                window=self._current_window or -1,
                success=False,
                error="timeout",
                elapsed_sec=timeout,
            )
        except Exception as exc:
            logger.error("Background upload failed: %s", exc)
            self._current_task = None
            return UploadResult(
                window=self._current_window or -1,
                success=False,
                error=str(exc),
            )

    @property
    def is_busy(self) -> bool:
        """True if an upload is currently in progress."""
        return self._current_task is not None and not self._current_task.done()

    async def _do_upload(self, window: int, upload_fn: Any, *args: Any) -> UploadResult:
        """Internal: execute the upload and capture result."""
        t0 = time.monotonic()
        try:
            await upload_fn(*args)
            elapsed = time.monotonic() - t0
            logger.info("Background upload completed: window=%d elapsed=%.2fs", window, elapsed)
            return UploadResult(window=window, success=True, elapsed_sec=elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "Background upload failed: window=%d error=%s elapsed=%.2fs",
                window,
                exc,
                elapsed,
            )
            return UploadResult(window=window, success=False, error=str(exc), elapsed_sec=elapsed)

    async def cancel(self) -> None:
        """Cancel any pending upload."""
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
        self._current_task = None


# --------------------------------------------------------------------------- #
#                       Chain State Cache                                     #
# --------------------------------------------------------------------------- #


@dataclass
class ChainSnapshot:
    """Cached chain state for non-blocking access."""

    block: int = 0
    block_hash: str = ""
    timestamp: float = 0.0
    age_sec: float = 0.0


class ChainStateCache:
    """Background-refreshing cache for chain state (block number, block hash).

    Eliminates blocking RPC calls from the generation hot loop. The cache
    refreshes every `refresh_interval` seconds in the background.

    Usage:
        cache = ChainStateCache(subtensor)
        await cache.start()

        # Non-blocking reads from generation loop:
        snap = cache.current
        if snap.block > 0:
            # Use cached block number
            ...

        await cache.stop()
    """

    def __init__(self, subtensor: Any, refresh_interval: float = 6.0) -> None:
        self._subtensor = subtensor
        self._refresh_interval = refresh_interval
        self._snapshot = ChainSnapshot()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def current(self) -> ChainSnapshot:
        """Get the latest cached chain snapshot (non-blocking)."""
        snap = self._snapshot
        snap.age_sec = time.monotonic() - snap.timestamp if snap.timestamp > 0 else float("inf")
        return snap

    async def start(self) -> None:
        """Start background refresh loop."""
        # Do one immediate refresh before starting the loop
        await self._refresh()
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="chain-state-cache")
        logger.info("ChainStateCache started (refresh every %.1fs)", self._refresh_interval)

    async def stop(self) -> None:
        """Stop background refresh loop."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        logger.info("ChainStateCache stopped")

    async def force_refresh(self) -> ChainSnapshot:
        """Force an immediate refresh and return the new snapshot."""
        await self._refresh()
        return self._snapshot

    async def _loop(self) -> None:
        """Background refresh loop."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._refresh_interval)
                if self._stop_event.is_set():
                    break
                await self._refresh()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("ChainStateCache refresh error: %s", exc)
                await asyncio.sleep(2.0)

    async def _refresh(self) -> None:
        """Fetch current block and hash from chain."""
        try:
            block = await self._subtensor.get_current_block()  # type: ignore[misc]
            block_hash = await self._subtensor.get_block_hash(block)  # type: ignore[misc]
            self._snapshot = ChainSnapshot(
                block=int(block),
                block_hash=str(block_hash) if block_hash else "",
                timestamp=time.monotonic(),
            )
        except Exception as exc:
            logger.debug("ChainStateCache: failed to refresh: %s", exc)


# --------------------------------------------------------------------------- #
#                       Async Drand Fetcher                                   #
# --------------------------------------------------------------------------- #


async def fetch_drand_async(round_id: int | None = None) -> dict[str, Any]:
    """Fetch drand beacon using httpx async client.

    Replaces the synchronous requests-based implementation for use in
    async contexts without needing asyncio.to_thread().
    """
    import httpx

    from ..infrastructure.drand import (
        DRAND_CHAIN_HASH,
        DRAND_URLS,
        get_mock_beacon,
    )

    round_part = "latest" if round_id is None else str(round_id)
    endpoint = f"/{DRAND_CHAIN_HASH}/public/{round_part}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in DRAND_URLS:
            try:
                full_url = f"{url}{endpoint}"
                response = await client.get(full_url)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        "Drand async fetch: round=%s randomness=%s...",
                        data["round"],
                        data["randomness"][:8],
                    )
                    return {
                        "round": data["round"],
                        "randomness": data["randomness"],
                        "signature": data.get("signature", ""),
                        "previous_signature": data.get("previous_signature", ""),
                    }
            except Exception as exc:
                logger.debug("Drand async fetch failed from %s: %s", url, exc)
                continue

    logger.warning("All drand URLs failed, using mock beacon")
    return get_mock_beacon()
