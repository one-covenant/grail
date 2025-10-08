from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, ClassVar

import bittensor as bt

logger = logging.getLogger(__name__)


class ResilientSubtensor:
    """
    Wrapper around bittensor subtensor that automatically adds timeout and retry
    logic to blockchain calls.

    This prevents the application from hanging indefinitely when blockchain RPC
    calls fail or timeout. Protected methods will retry with exponential backoff.
    """

    # Methods that should be protected with timeout and retry logic
    PROTECTED_METHODS: ClassVar[set[str]] = {
        "get_current_block",
        "get_block_hash",
        "metagraph",
        "commit",
        "get_commitment",
    }

    def __init__(
        self,
        subtensor: bt.subtensor,
        timeout: float = 15.0,
        retries: int = 3,
        backoff_base: float = 5.0,
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
            timeout = object.__getattribute__(self, "_timeout")
            retries = object.__getattribute__(self, "_retries")
            backoff_base = object.__getattribute__(self, "_backoff_base")

            for retry in range(retries):
                try:
                    result = await asyncio.wait_for(
                        attr(*args, **kwargs),
                        timeout=timeout,
                    )
                    if retry > 0:
                        logger.info("✅ %s() succeeded on attempt %d", name, retry + 1)
                    return result
                except asyncio.TimeoutError:
                    wait_time = backoff_base * (2**retry)
                    args_str = ", ".join(str(a) for a in args[:2])  # Show first 2 args
                    logger.error(
                        "⏱️ Timeout in %s(%s) (attempt %d/%d)",
                        name,
                        args_str,
                        retry + 1,
                        retries,
                    )
                    if retry < retries - 1:
                        logger.info("Retrying in %ds...", wait_time)
                        await asyncio.sleep(wait_time)

            # All retries exhausted
            args_str = ", ".join(str(a) for a in args[:2])
            total_time = timeout * retries + sum(backoff_base * (2**i) for i in range(retries - 1))
            logger.error(
                "❌ BLOCKCHAIN CALL FAILED: %s(%s) timed out after "
                "%d attempts (total ~%.0fs). This indicates network issues "
                "or blockchain node problems.",
                name,
                args_str,
                retries,
                total_time,
            )
            error_msg = f"{name} failed after {retries} attempts - blockchain connection issue"
            raise TimeoutError(error_msg)

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

    await subtensor.initialize()

    if resilient:
        # Wrap with resilience layer for production use
        timeout = float(os.getenv("BT_CALL_TIMEOUT", "5.0"))
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
