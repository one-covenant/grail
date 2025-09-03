#!/usr/bin/env python3
"""
GRAIL Shared Constants

Centralized configuration constants used across the GRAIL codebase.
"""

import os
import json
import subprocess
from typing import Optional

# ──────────────────────────  NETWORK & BLOCKCHAIN  ─────────────────────────────

def get_netuid(network: str = "finney") -> int:
    """
    Get NETUID from environment variable or auto-discover grail subnet.
    
    Args:
        network: Network to search ('finney' for mainnet, 'test' for testnet)
        
    Returns:
        NETUID for grail subnet
        
    Environment Variables:
        GRAIL_NETUID: Override netuid (e.g., "81" for mainnet, "165" for testnet)
        GRAIL_NETWORK: Network to use ('finney' or 'test')
    """
    # First check for explicit override
    netuid_override = os.getenv("GRAIL_NETUID")
    if netuid_override is not None:
        return int(netuid_override)
    
    # Auto-discover by subnet name
    discovered_netuid = discover_grail_netuid(network)
    if discovered_netuid is not None:
        return discovered_netuid
    
    # Fallback to mainnet default
    return 81


def discover_grail_netuid(network: str = "finney") -> Optional[int]:
    """
    Auto-discover grail subnet NETUID by name using btcli.
    
    Args:
        network: 'finney' for mainnet, 'test' for testnet
        
    Returns:
        NETUID if found, None otherwise
    """
    try:
        # Try to find btcli in the venv first, then fallback to PATH
        import sys
        import os
        
        btcli_path = "btcli"
        if hasattr(sys, 'prefix'):
            venv_btcli = os.path.join(sys.prefix, "bin", "btcli")
            if os.path.exists(venv_btcli):
                btcli_path = venv_btcli
        
        cmd = [btcli_path, "subnets", "list", "--subtensor.network", network, "--json-output"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return None
            
        subnets = json.loads(result.stdout)
        
        # Search for subnet with name "grail" (case-insensitive)
        for subnet in subnets:
            if subnet.get("name", "").lower() == "grail":
                return int(subnet["netuid"])
                
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
            json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        pass
    
    return None


# Get the network from environment (default to mainnet)
NETWORK = os.getenv("GRAIL_NETWORK", "finney")
NETUID = get_netuid(NETWORK)
WINDOW_LENGTH = 20  # Generate inferences every 20 blocks (increased for model downloads)

# ──────────────────────────  MODEL CONFIGURATION  ─────────────────────────────

MODEL_NAME = "google/gemma-3-1b-it"
LAYER_INDEX = -1

# ──────────────────────────  LOGGING  ─────────────────────────────

TRACE = 5

# ──────────────────────────  GRAIL CRYPTOGRAPHIC CONSTANTS  ─────────────────────────────

PRIME_Q = 2_147_483_647
CHALLENGE_K = 16
TOLERANCE = 3
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# ──────────────────────────  TERMINATION VALIDATION HPs  ─────────────────────────────

DEFAULT_MAX_NEW_TOKENS = 20  # Must match rollout generator default
MIN_EOS_PROBABILITY = 0.1  # Minimum probability for valid EOS termination
SANITY_CHECK_DRIFT_THRESHOLD = 0.1  # Max acceptable drift between miner/validator

# ──────────────────────────  TOKEN SAMPLING DIST CHECK HPs  ─────────────────────────────

SAMPLING_MIN_STEPS = 8
SAMPLING_LOW_P = 0.10
SAMPLING_HIGH_P = 0.90
SAMPLING_LOW_FRAC_MIN = 0.20
SAMPLING_HIGH_FRAC_MIN = 0.50
SAMPLING_MID_FRAC_MAX = 0.40
SAMPLING_BC_THRESHOLD = 0.58

# ──────────────────────────  VALIDATOR-SPECIFIC CONSTANTS  ─────────────────────────────

# Superlinear weighting exponent:
# For p > 1, w_i ∝ s_i^p amplifies differences and penalizes sybil splitting:
# splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 1.5
