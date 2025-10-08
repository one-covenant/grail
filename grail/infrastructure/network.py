import logging
import os

import bittensor as bt

logger = logging.getLogger(__name__)


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


async def create_subtensor() -> bt.subtensor:
    """
    Create and initialize an async subtensor instance using env configuration.

    - If BT_CHAIN_ENDPOINT is set, connect to custom endpoint directly
    - Otherwise, use BT_NETWORK ('finney', 'test', 'local') - defaults to 'finney'

    The Bittensor SDK resolves named networks to official endpoints automatically.
    """
    network, chain_endpoint = _resolve_network()

    if chain_endpoint:
        # Custom endpoint specified (e.g., local node, custom remote)
        logger.info(
            f"Connecting to custom Bittensor endpoint: {chain_endpoint} "
            f"(BT_NETWORK={network} ignored)"
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
                f"Unknown BT_NETWORK='{network}', defaulting to 'finney'. "
                "Valid options: finney, test, local"
            )
            network = "finney"
            label = "mainnet"

        logger.info(f"Connecting to Bittensor {label} (network={network})")
        subtensor = bt.async_subtensor(network=network)

    await subtensor.initialize()
    return subtensor
