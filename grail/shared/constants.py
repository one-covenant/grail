#!/usr/bin/env python3
"""
GRAIL Shared Constants

Centralized configuration constants used across the GRAIL codebase.
"""

import os

# ────────────────  NETWORK & BLOCKCHAIN  ────────────────

NETWORK = os.getenv("BT_NETWORK", "finney")
NETUID = int(os.getenv("NETUID", 81))
WINDOW_LENGTH = 30

# ────────────────  MODEL CONFIGURATION  ────────────────

MODEL_NAME = os.getenv("GRAIL_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
LAYER_INDEX = -1

# ────────────────  LOGGING  ────────────────

TRACE = 5

# ────────────────  GRAIL CRYPTOGRAPHIC CONSTANTS  ────────────────

PRIME_Q = 2_147_483_647
CHALLENGE_K = 16
TOLERANCE = 3
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Integer-sketch configuration (for GPU-agnostic verification)
# Default algorithm id. Set to "dotfp_v0" to use legacy float-projection.
SKETCH_ALGO_ID = os.getenv("GRAIL_SKETCH_ALGO_ID", "intdot_v1")

# Integer sketch parameters (can be tuned via env)
INT_SKETCH_S = int(os.getenv("GRAIL_INT_SKETCH_S", "128"))
INT_SKETCH_R = int(os.getenv("GRAIL_INT_SKETCH_R", "8191"))
INT_SKETCH_H = int(os.getenv("GRAIL_INT_SKETCH_H", "32767"))
# Recommended starting tolerance for integer sketch
INT_SKETCH_TOLERANCE = int(os.getenv("GRAIL_INT_SKETCH_TOLERANCE", "300000"))

# Allow validators to accept legacy (dotfp_v0) commits during migration
ACCEPT_LEGACY_ALGO = os.getenv("GRAIL_ACCEPT_LEGACY_ALGO", "true").lower() in {"1", "true", "yes"}

# ────────────────  TERMINATION VALIDATION HPs  ────────────────

MAX_NEW_TOKENS = int(os.getenv("GRAIL_MAX_NEW_TOKENS", "1024"))

# Must match rollout generator default
MIN_EOS_PROBABILITY = 0.1  # Minimum probability for valid EOS termination
# Max acceptable drift between miner/validator
SANITY_CHECK_DRIFT_THRESHOLD = 0.1

# ────────────────  TOKEN SAMPLING DIST CHECK HPs  ────────────────

SAMPLING_MIN_STEPS = 30
SAMPLING_LOW_P = 0.10
SAMPLING_HIGH_P = 0.90
SAMPLING_LOW_FRAC_MIN = 0.20
SAMPLING_HIGH_FRAC_MIN = 0.50
SAMPLING_MID_FRAC_MAX = 0.40
# NOTE: this parameter so far hasn't been a good indicator of bimodality
SAMPLING_BC_THRESHOLD = 0.58

# Minimal sampling shape hyperparameters (median gate for unimodal-low)
SAMPLING_MEDIAN_LOW_MAX = 0.30
# NOTE: this parameter so far has been a good indicator of bimodality
SAMPLING_LOW_Q10_MAX = 0.15

# Extra sanity gates for sampling shape checks
SAMPLING_MIN_TOKEN_PROB = 1e-5
SAMPLING_INITIAL_WINDOW_STEPS = 35

# ────────────────  VALIDATOR-SPECIFIC CONSTANTS  ────────────────

# Superlinear weighting exponent:
# For p > 1, w_i ∝ s_i^p amplifies differences and penalizes sybil splitting:
# splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 4.0

# ────────────────  ROLLOUTS PER PROBLEM  ────────────────

ROLLOUTS_PER_PROBLEM = int(os.getenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4"))

# ────────────────  EMISSION BURN MECHANISM  ────────────────

GRAIL_BURN_UID = 0
GRAIL_BURN_PERCENTAGE = 80.0

# ────────────────  MINER SAMPLING (VALIDATION COST CONTROL)  ────────────────

# Enable/disable miner-level subsampling per window.
MINER_SAMPLING_ENABLED = True

# Fraction of active miners (those with a window file) to validate per window.
# Applied after MINER_SAMPLE_MIN and before MINER_SAMPLE_MAX.
MINER_SAMPLE_RATE = 1.0

# Minimum number of active miners to validate each window (floor).
MINER_SAMPLE_MIN = 1

# Optional cap on miners validated per window. Set to None to disable.
MINER_SAMPLE_MAX = 35
