"""Drand distributed randomness beacon integration for GRAIL."""

import logging
import os
from typing import Any, Optional, TypedDict

import requests

logger = logging.getLogger(__name__)


class ChainConfig(TypedDict):
    hash: str
    genesis_time: int
    period: int
    description: str


# Drand chain configurations
DRAND_CHAINS: dict[str, ChainConfig] = {
    "quicknet": {
        "hash": "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
        "genesis_time": 1692803367,
        "period": 3,  # 3 seconds per round
        "description": "Fast 3-second randomness (recommended)",
    },
    "mainnet": {
        "hash": "84b2234fb34e835dccd048255d7ad3194b81af7d978c3bf157e3469592ae4e02",
        "genesis_time": 1595431050,
        "period": 30,  # 30 seconds per round
        "description": "Original 30-second chain",
    },
}

# Default to quicknet for faster randomness
DEFAULT_CHAIN = "quicknet"

# Drand API endpoints
DRAND_URLS = [
    "https://api.drand.sh",
    "https://drand.cloudflare.com",
    "https://api.drand.secureweb3.com:6875",
]

# Current chain configuration (can be changed via set_chain)
_current_chain: str = DEFAULT_CHAIN
DRAND_CHAIN_HASH: str = DRAND_CHAINS[_current_chain]["hash"]
DRAND_GENESIS_TIME: int = DRAND_CHAINS[_current_chain]["genesis_time"]
DRAND_PERIOD: int = DRAND_CHAINS[_current_chain]["period"]

# Global counter for mock beacons
BEACON_COUNTER = 0


def set_chain(chain_name: str) -> None:
    """
    Switch to a different drand chain.

    Args:
        chain_name: Name of the chain ('quicknet' or 'mainnet')

    Raises:
        ValueError: If chain_name is not recognized
    """
    global _current_chain, DRAND_CHAIN_HASH, DRAND_GENESIS_TIME, DRAND_PERIOD

    if chain_name not in DRAND_CHAINS:
        raise ValueError(
            f"Unknown chain '{chain_name}'. Available chains: {list(DRAND_CHAINS.keys())}"
        )

    _current_chain = chain_name
    DRAND_CHAIN_HASH = DRAND_CHAINS[_current_chain]["hash"]
    DRAND_GENESIS_TIME = DRAND_CHAINS[_current_chain]["genesis_time"]
    DRAND_PERIOD = DRAND_CHAINS[_current_chain]["period"]

    logger.info(
        f"Switched to drand chain '{chain_name}': {DRAND_CHAINS[chain_name]['description']}"
    )


def get_current_chain() -> dict[str, Any]:
    """Get information about the currently selected chain."""
    return {"name": _current_chain, **DRAND_CHAINS[_current_chain]}


def get_drand_beacon(round_id: Optional[int] = None, use_fallback: bool = True) -> dict:
    """
    Fetch randomness from drand network.

    Args:
        round_id: Specific round to fetch, or None for latest
        use_fallback: If True, falls back to mock beacon on failure

    Returns:
        Dictionary with 'round' and 'randomness' keys
    """
    # Drand HTTP API path: /<chain-hash>/public/<latest|round>
    round_part = "latest" if round_id is None else str(round_id)
    endpoint = f"/{DRAND_CHAIN_HASH}/public/{round_part}"

    # Try each drand URL
    for url in DRAND_URLS:
        try:
            full_url = f"{url}{endpoint}"
            logger.debug(f"[Drand-{_current_chain}] Fetching from {full_url}")
            response = requests.get(full_url, timeout=10)  # Increased timeout
            if response.status_code == 200:
                data = response.json()
                msg_prefix = f"[Drand-{_current_chain}] Success!"
                msg_detail = f" round={data['round']}, randomness={data['randomness'][:8]}…"
                logger.info(msg_prefix + msg_detail)
                return {
                    "round": data["round"],
                    "randomness": data["randomness"],
                    "signature": data.get("signature", ""),
                    "previous_signature": data.get("previous_signature", ""),
                }
            else:
                try:
                    err_text = response.text[:200]
                except Exception:
                    err_text = "<no body>"
                logger.debug(
                    f"[Drand] Got status {response.status_code} from {url} for {endpoint} "
                    f"body={err_text}"
                )
        except Exception as e:
            logger.debug(f"[Drand] Failed to fetch from {url}: {e}")
            continue

    # Fallback to mock beacon if all URLs fail
    if use_fallback:
        logger.warning("[Drand] All URLs failed, using mock beacon")
        return get_mock_beacon()
    else:
        raise Exception("Failed to fetch from any drand URL")


def get_mock_beacon() -> dict:
    """Fallback mock beacon for testing/development."""
    global BEACON_COUNTER
    BEACON_COUNTER += 1
    rnd = os.urandom(32).hex()
    logger.debug(f"[MockBeacon] round={BEACON_COUNTER}, randomness={rnd[:8]}…")
    return {"round": BEACON_COUNTER, "randomness": rnd}


def get_beacon(round_id: str = "latest", use_drand: bool = True) -> dict:
    """
    Get randomness beacon (drand by default, with fallback).

    Args:
        round_id: "latest" or specific round number
        use_drand: If True, use drand network; if False, use mock
    """
    if not use_drand:
        return get_mock_beacon()

    try:
        if round_id == "latest":
            return get_drand_beacon(None)
        else:
            return get_drand_beacon(int(round_id))
    except Exception:
        # Fallback to mock on any error
        return get_mock_beacon()


def get_round_at_time(timestamp: int) -> int:
    """Calculate drand round number for a given timestamp."""
    elapsed = timestamp - DRAND_GENESIS_TIME
    return int((elapsed // DRAND_PERIOD) + 1)
