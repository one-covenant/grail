#!/usr/bin/env python3
"""
GRAIL Shared Constants

Centralized configuration constants used across the GRAIL codebase.
"""

import os

# ──────────────────────────  NETWORK & BLOCKCHAIN  ─────────────────────────────


def _read_netuid() -> int:
    """Read NETUID from environment with sensible defaults.

    Supports both NETUID and legacy GRAIL_NETUID.
    """
    for key in ("NETUID", "GRAIL_NETUID"):
        v = os.getenv(key)
        if v and str(v).strip():
            return int(v)
    return 81


# Get the network from environment (default to mainnet)
NETWORK = os.getenv("BT_NETWORK") or os.getenv("GRAIL_NETWORK") or "finney"
NETUID = _read_netuid()
WINDOW_LENGTH = 20  # Generate inferences every 20 blocks (increased for model downloads)

# ──────────────────────────  MODEL CONFIGURATION  ─────────────────────────────

MODEL_NAME = os.getenv("GRAIL_MODEL_NAME", "google/gemma-3-1b-it")
LAYER_INDEX = -1

# ──────────────────────────  LOGGING  ─────────────────────────────

TRACE = 5

# ──────────────────────────  GRAIL CRYPTOGRAPHIC CONSTANTS  ─────────────────────────────

PRIME_Q = 2_147_483_647
CHALLENGE_K = 16
TOLERANCE = 3
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# ──────────────────────────  TERMINATION VALIDATION HPs  ─────────────────────────────

# Parse once as int; environment can override via GRAIL_MAX_NEW_TOKENS
try:
    DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GRAIL_MAX_NEW_TOKENS", "256"))
except Exception:
    DEFAULT_MAX_NEW_TOKENS = 256

# Must match rollout generator default
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

# Minimal sampling shape hyperparameters (median gate for unimodal-low)
SAMPLING_MEDIAN_LOW_MAX = 0.20

# ──────────────────────────  VALIDATOR-SPECIFIC CONSTANTS  ─────────────────────────────

# Superlinear weighting exponent:
# For p > 1, w_i ∝ s_i^p amplifies differences and penalizes sybil splitting:
# splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 1.5
