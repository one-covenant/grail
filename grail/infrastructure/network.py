from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, ClassVar

import bittensor as bt

from ..logging_utils import await_with_stall_log

logger = logging.getLogger(__name__)


class ResilientSubtensor:
    """
    Wrapper around bittensor subtensor with circuit breaker, caching, and auto-restart.

    This prevents the application from hanging indefinitely when blockchain RPC
    calls fail or timeout. Protected methods will retry with exponential backoff.
    - Circuit breaker: Stops calls temporarily after repeated failures
    - Metagraph cache: Returns last good metagraph when calls timeout
    - Auto-restart: Recreates subtensor connection after extended failures
    """

    # Methods that should be protected with timeout and retry logic
    PROTECTED_METHODS: ClassVar[set[str]] = {
        "get_current_block",
        "get_block_hash",
        "metagraph",
        "get_commitment",
    }

    def __init__(
        self,
        subtensor: bt.subtensor,
        timeout: float = 10.0,
        retries: int = 3,
        backoff_base: float = 10.0,
    ):
        """
        Initialize resilient subtensor wrapper.

        Args:
            subtensor: The underlying bittensor subtensor instance
            timeout: Timeout in seconds for each attempt (default: 15s)
            retries: Number of retry attempts (default: 3)
            backoff_base: Base multiplier for exponential backoff (default: 5s)
        """
        object.__setattr__(self, "_subtensor", subtensor)
        object.__setattr__(self, "_timeout", timeout)
        object.__setattr__(self, "_retries", retries)
        object.__setattr__(self, "_backoff_base", backoff_base)
        # Circuit breaker state
        object.__setattr__(self, "_failure_count", 0)
        object.__setattr__(self, "_circuit_open_until", 0.0)
        object.__setattr__(self, "_circuit_threshold", 2)  # Open after 2 consecutive call failures
        object.__setattr__(self, "_circuit_timeout", 30.0)  # Stay open for 30s
        # Metagraph cache
        object.__setattr__(self, "_metagraph_cache", {})

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        circuit_open_until = object.__getattribute__(self, "_circuit_open_until")
        return time.time() < circuit_open_until

    def _get_cached_metagraph(self, netuid: int) -> Any | None:
        """Get cached metagraph for a given netuid."""
        metagraph_cache = object.__getattribute__(self, "_metagraph_cache")
        return metagraph_cache.get(netuid)

    def _cache_metagraph(self, netuid: int, metagraph: Any) -> None:
        """Cache metagraph for a given netuid."""
        metagraph_cache = object.__getattribute__(self, "_metagraph_cache")
        metagraph_cache[netuid] = metagraph

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker failure count on success."""
        object.__setattr__(self, "_failure_count", 0)

    def _increment_failure_count(self) -> int:
        """Increment and return failure count."""
        failure_count = object.__getattribute__(self, "_failure_count") + 1
        object.__setattr__(self, "_failure_count", failure_count)
        return failure_count

    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker for cooldown period."""
        circuit_timeout = object.__getattribute__(self, "_circuit_timeout")
        failure_count = object.__getattribute__(self, "_failure_count")
        object.__setattr__(self, "_circuit_open_until", time.time() + circuit_timeout)
        logger.error(
            "🔴 Circuit breaker opened after %d failures, cooling down for %ds",
            failure_count,
            circuit_timeout,
        )

    def _should_open_circuit(self) -> bool:
        """Check if circuit breaker threshold is reached."""
        failure_count = object.__getattribute__(self, "_failure_count")
        circuit_threshold = object.__getattribute__(self, "_circuit_threshold")
        return failure_count >= circuit_threshold

    async def _restart_subtensor(self) -> None:
        """Restart subtensor connection."""
        logger.warning("🔄 Restarting subtensor connection...")
        subtensor = object.__getattribute__(self, "_subtensor")
        network = (
            subtensor.network
            if hasattr(subtensor, "network")
            else os.getenv("BT_NETWORK", "finney")
        )
        new_subtensor = bt.async_subtensor(network=network)
        await new_subtensor.initialize()
        object.__setattr__(self, "_subtensor", new_subtensor)
        self._reset_circuit_breaker()
        logger.info("✅ Subtensor connection restarted")

    async def _handle_circuit_open(self, method_name: str, args: tuple) -> Any:
        """Handle method call when circuit breaker is open."""
        if method_name == "metagraph" and args:
            cached = self._get_cached_metagraph(args[0])
            if cached:
                logger.warning("⚡ Circuit open, returning cached metagraph for netuid %s", args[0])
                return cached
        logger.warning("⚡ Circuit breaker open, skipping %s call", method_name)
        raise TimeoutError(f"Circuit breaker open for {method_name}")

    async def _attempt_call(self, method: Any, args: tuple, kwargs: dict, timeout: float) -> Any:
        """Attempt a single method call with timeout."""
        return await asyncio.wait_for(method(*args, **kwargs), timeout=timeout)

    def _handle_success(self, method_name: str, args: tuple, result: Any, retry: int) -> None:
        """Handle successful method call."""
        self._reset_circuit_breaker()
        if method_name == "metagraph" and args:
            self._cache_metagraph(args[0], result)
        if retry > 0:
            logger.info("✅ %s() succeeded on attempt %d", method_name, retry + 1)

    async def _handle_retry_backoff(
        self,
        reason: str,
        method_name: str,
        retry: int,
        retries: int,
        backoff_base: float,
    ) -> None:
        """Handle retry backoff for timeout or cancellation events."""
        if retry < retries - 1:
            wait_time = backoff_base * (2**retry)
            logger.error(
                "⏱️ %s in %s (attempt %d/%d), retrying in %ds",
                reason,
                method_name,
                retry + 1,
                retries,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    def _handle_all_retries_failed(self, method_name: str, args: tuple, retries: int) -> Any:
        """Handle case when all retries are exhausted.

        Note: This counts complete call failures (after all retries), not individual
        retry attempts. Circuit opens after threshold consecutive call failures.
        """
        self._increment_failure_count()

        if self._should_open_circuit():
            self._open_circuit_breaker()
            asyncio.create_task(self._restart_subtensor())

        # Try to return cached metagraph as last resort
        if method_name == "metagraph" and args:
            cached = self._get_cached_metagraph(args[0])
            if cached:
                logger.warning("⚠️ Returning stale cached metagraph for netuid %s", args[0])
                return cached

        logger.error("❌ %s failed after %d attempts", method_name, retries)
        raise TimeoutError(f"{method_name} failed after {retries} attempts")

    async def _call_with_retry(
        self, method_name: str, method: Any, args: tuple, kwargs: dict
    ) -> Any:
        """Execute method with retry logic and circuit breaker protection."""
        # Check circuit breaker
        if self._is_circuit_open():
            return await self._handle_circuit_open(method_name, args)

        timeout = object.__getattribute__(self, "_timeout")
        retries = object.__getattribute__(self, "_retries")
        backoff_base = object.__getattribute__(self, "_backoff_base")

        # Retry loop
        for retry in range(retries):
            try:
                result = await self._attempt_call(method, args, kwargs, timeout)
                self._handle_success(method_name, args, result, retry)
                return result
            except asyncio.TimeoutError:
                await self._handle_retry_backoff(
                    "Timeout",
                    method_name,
                    retry,
                    retries,
                    backoff_base,
                )
            except asyncio.CancelledError:
                # If our task is actually being cancelled, propagate.
                task = asyncio.current_task()
                cancelling_count = 0
                if task is not None and hasattr(task, "cancelling"):
                    try:
                        # Python 3.11+: number of cancellation requests
                        cancelling_count = task.cancelling()  # type: ignore[attr-defined]
                    except Exception:
                        cancelling_count = 0
                if cancelling_count > 0:
                    raise
                # Otherwise, treat as transient cancellation from underlying client
                await self._handle_retry_backoff(
                    "Cancellation",
                    method_name,
                    retry,
                    retries,
                    backoff_base,
                )

        # All retries exhausted
        return self._handle_all_retries_failed(method_name, args, retries)

    def __getattr__(self, name: str) -> Any:
        """Intercept attribute access to wrap protected methods."""
        attr = getattr(object.__getattribute__(self, "_subtensor"), name)

        # Only wrap methods we want to protect
        if name not in self.PROTECTED_METHODS or not callable(attr):
            return attr

        # Check if it's an async method
        if not asyncio.iscoroutinefunction(attr):
            return attr

        # Return a wrapped version with retry logic
        async def wrapped_method(*args: Any, **kwargs: Any) -> Any:
            return await self._call_with_retry(name, attr, args, kwargs)

        return wrapped_method

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to underlying subtensor."""
        setattr(object.__getattribute__(self, "_subtensor"), name, value)

    def __repr__(self) -> str:
        """String representation of the resilient subtensor."""
        subtensor = object.__getattribute__(self, "_subtensor")
        timeout = object.__getattribute__(self, "_timeout")
        retries = object.__getattribute__(self, "_retries")
        return f"ResilientSubtensor(subtensor={subtensor}, timeout={timeout}s, retries={retries})"


def _resolve_network() -> tuple[str, str | None]:
    """
    Resolve network selection from environment variables.

    Priority:
    - BT_CHAIN_ENDPOINT: if set, use custom endpoint (overrides BT_NETWORK)
    - BT_NETWORK: named network ('finney', 'test', 'local'), defaults to 'finney'

    Returns:
        (network, chain_endpoint): network name and optional custom endpoint
    """
    network = os.getenv("BT_NETWORK", "finney")
    chain_endpoint = os.getenv("BT_CHAIN_ENDPOINT")  # No default - None if unset
    return network, chain_endpoint


async def create_subtensor(*, resilient: bool = True) -> bt.subtensor | ResilientSubtensor:
    """
    Create and initialize an async subtensor instance using env configuration.

    - If BT_CHAIN_ENDPOINT is set, connect to custom endpoint directly
    - Otherwise, use BT_NETWORK ('finney', 'test', 'local') - defaults to 'finney'

    The Bittensor SDK resolves named networks to official endpoints automatically.

    Args:
        resilient: If True, wrap subtensor with ResilientSubtensor for automatic
                   timeout and retry logic (default: True, recommended for production)

    Environment Variables:
        BT_NETWORK: Network name ('finney', 'test', 'local')
        BT_CHAIN_ENDPOINT: Custom WebSocket endpoint URL
        BT_CALL_TIMEOUT: Timeout in seconds for blockchain calls (default: 15.0)
        BT_CALL_RETRIES: Number of retry attempts (default: 3)
        BT_CALL_BACKOFF: Base backoff multiplier in seconds (default: 5.0)

    Returns:
        Initialized subtensor instance (optionally wrapped with resilience layer)
    """
    network, chain_endpoint = _resolve_network()

    if chain_endpoint:
        # Custom endpoint specified (e.g., local node, custom remote)
        logger.info(
            "Connecting to custom Bittensor endpoint: %s (BT_NETWORK=%s ignored)",
            chain_endpoint,
            network,
        )
        subtensor = bt.async_subtensor(network=chain_endpoint)
    else:
        # Use named network - SDK resolves to official endpoint
        label = {
            "finney": "mainnet",
            "test": "testnet",
            "local": "local",
        }.get(network, "custom")

        if network not in {"finney", "test", "local"}:
            logger.warning(
                "Unknown BT_NETWORK='%s', defaulting to 'finney'. "
                "Valid options: finney, test, local",
                network,
            )
            network = "finney"
            label = "mainnet"

        logger.info("Connecting to Bittensor %s (network=%s)", label, network)
        subtensor = bt.async_subtensor(network=network)

    await await_with_stall_log(
        subtensor.initialize(),
        label="subtensor.initialize",
        threshold_seconds=120.0,
        log=logger,
    )

    if resilient:
        # Wrap with resilience layer for production use
        timeout = float(os.getenv("BT_CALL_TIMEOUT", "15.0"))
        retries = int(os.getenv("BT_CALL_RETRIES", "3"))
        backoff = float(os.getenv("BT_CALL_BACKOFF", "5.0"))

        logger.info(
            "Wrapping subtensor with resilience layer (timeout=%ds, retries=%d, backoff=%ds)",
            timeout,
            retries,
            backoff,
        )
        return ResilientSubtensor(subtensor, timeout=timeout, retries=retries, backoff_base=backoff)

    return subtensor
