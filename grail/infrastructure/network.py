import logging
import os
from typing import Optional

import bittensor as bt

logger = logging.getLogger(__name__)


def _resolve_network() -> tuple[str, Optional[str]]:
    """
    Resolve network selection from environment variables.

    Priority:
    - BT_NETWORK / BT_CHAIN_ENDPOINT (preferred names)
    - GRAIL_NETWORK / GRAIL_CHAIN_ENDPOINT (legacy fallback)

    Returns:
        (network, chain_endpoint)
    """
    network = os.getenv("BT_NETWORK", "finney")
    chain_endpoint = os.getenv("BT_CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
    return network, chain_endpoint


async def create_subtensor() -> bt.subtensor:
    """
    Create and initialize an async subtensor instance using env configuration.

    - If BT_CHAIN_ENDPOINT/GRAIL_CHAIN_ENDPOINT is set, construct a config and pass via config.
    - Else, use BT_NETWORK/GRAIL_NETWORK (e.g., 'finney' for mainnet, 'test' for public testnet).
    """
    network, chain_endpoint = _resolve_network()
    logger.debug(f"Creating subtensor (network={network}, endpoint={chain_endpoint})")

    label = (
        "public testnet" if network == "test" else ("mainnet" if network == "finney" else "custom")
    )
    if chain_endpoint:
        # Pass the chain endpoint directly as the network parameter
        # This preserves the hostname (e.g., ws://alice:9944) in Docker environments
        logger.info(
            f"Connecting to Bittensor custom endpoint: {chain_endpoint} (BT_NETWORK={network}, {label})"
        )
        subtensor = bt.async_subtensor(network=chain_endpoint)
    else:
        # Supported labels in this codebase: 'finney' (mainnet), 'test' (public testnet)
        if network not in {"finney", "test"}:
            logger.warning(f"Unknown BT_NETWORK='{network}', defaulting to 'finney'")
            network = "finney"
            label = "mainnet"
        logger.info(f"Connecting to Bittensor {label} (BT_NETWORK={network})")
        subtensor = bt.async_subtensor(network=network)

    await subtensor.initialize()
    return subtensor
