"""Drand distributed randomness beacon integration for GRAIL."""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Drand configuration - using League of Entropy mainnet
DRAND_URLS = [
    "https://api.drand.sh",
    "https://drand.cloudflare.com",
    "https://api.drand.secureweb3.com:6875"
]
DRAND_CHAIN_HASH = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"  # quicknet chain
DRAND_GENESIS_TIME = 1692803367  # quicknet genesis
DRAND_PERIOD = 3  # 3 seconds per round for quicknet

# Global counter for mock beacons
BEACON_COUNTER = 0

def get_drand_beacon(round_id: Optional[int] = None, use_fallback: bool = True) -> dict:
    """
    Fetch randomness from drand network.
    
    Args:
        round_id: Specific round to fetch, or None for latest
        use_fallback: If True, falls back to mock beacon on failure
    
    Returns:
        Dictionary with 'round' and 'randomness' keys
    """
    endpoint = f"/{DRAND_CHAIN_HASH}/public/{'latest' if round_id is None else round_id}"
    
    # Try each drand URL
    for url in DRAND_URLS:
        try:
            full_url = f"{url}{endpoint}"
            logger.debug(f"[Drand] Fetching from {full_url}")
            response = requests.get(full_url, timeout=10)  # Increased timeout
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[Drand] Success! round={data['round']}, randomness={data['randomness'][:8]}…")
                return {
                    "round": data["round"],
                    "randomness": data["randomness"],
                    "signature": data.get("signature", ""),
                    "previous_signature": data.get("previous_signature", "")
                }
            else:
                logger.debug(f"[Drand] Got status {response.status_code} from {url}")
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
    except:
        # Fallback to mock on any error
        return get_mock_beacon()

def get_round_at_time(timestamp: int) -> int:
    """Calculate drand round number for a given timestamp."""
    elapsed = timestamp - DRAND_GENESIS_TIME
    return (elapsed // DRAND_PERIOD) + 1

def verify_drand_signature(beacon: dict) -> bool:
    """
    Verify drand beacon signature (requires additional crypto libraries).
    For now, returns True - implement BLS verification if needed.
    """
    # TODO: Implement BLS signature verification
    # This requires py_ecc or similar library for BLS12-381
    return True