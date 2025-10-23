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

# ────────────────  TIMING & VALIDATION  ────────────────

# Bittensor block time (target average)
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (±seconds)
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds)
NETWORK_UPLOAD_LATENCY = 30

# Upload timeout for object storage operations (seconds)
UPLOAD_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT", "180"))

# Grace period for upload deadline validation
# = block variance + upload latency
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (prevents gaming)
# Validators use drand from this many seconds AFTER upload deadline
DRAND_FUTURE_BUFFER = 30

# Trainer checkpoint upload buffer (blocks before window ends to force publish)
TRAINER_UPLOAD_BUFFER_BLOCKS = 10  # ~60 seconds at 12s/block

# Time needed to upload the READY marker file to object storage
READY_MARKER_UPLOAD_BLOCKS = 1  # ~12 seconds at 12s/block

# ────────────────  MODEL CONFIGURATION  ────────────────

LAYER_INDEX = -1

# Trainer hyperparameters (env configurable)
TRAINER_LR = float(os.getenv("GRAIL_TRAINER_LR", "1e-6"))
TRAINER_EPOCHS = int(os.getenv("GRAIL_TRAINER_EPOCHS", "2"))
TRAINER_BATCH_SIZE = int(os.getenv("GRAIL_TRAINER_BATCH_SIZE", "4"))
TRAINER_MAX_LENGTH = int(os.getenv("GRAIL_TRAINER_MAX_LENGTH", "1024"))
TRAINER_GRAD_CLIP = float(os.getenv("GRAIL_TRAINER_GRAD_CLIP", "0.5"))
TRAINER_WARMUP_STEPS = int(os.getenv("GRAIL_TRAINER_WARMUP_STEPS", "10"))
TRAINER_KL_COEF = float(os.getenv("GRAIL_TRAINER_KL_COEF", "0.02"))
TRAINER_ENTROPY_COEF = float(os.getenv("GRAIL_TRAINER_ENTROPY_COEF", "0.001"))
TRAINER_ADV_CLIP_PERCENTILE = float(os.getenv("GRAIL_TRAINER_ADV_CLIP_PERCENTILE", "99.0"))
TRAINER_GROUP_ADV_SUM_TOL = float(os.getenv("GRAIL_TRAINER_GROUP_ADV_SUM_TOL", "0.01"))
TRAINER_GRAD_ACCUM_STEPS = int(os.getenv("GRAIL_TRAINER_GRAD_ACCUM_STEPS", "2"))

# Importance sampling and PPO-style clipping
TRAINER_USE_IS = os.getenv("GRAIL_TRAINER_USE_IS", "1") == "1"
TRAINER_PPO_CLIP_EPS = float(os.getenv("GRAIL_TRAINER_PPO_CLIP_EPS", "0.2"))

# Log-ratio clamp for numerical stability when exponentiating to ratios
TRAINER_LOGRATIO_CLAMP = float(os.getenv("GRAIL_TRAINER_LOGRATIO_CLAMP", "5.0"))

# Adaptive KL settings
TRAINER_ADAPTIVE_KL = os.getenv("GRAIL_TRAINER_ADAPTIVE_KL", "1") == "1"
TRAINER_KL_TARGET = float(os.getenv("GRAIL_TRAINER_KL_TARGET", "0.04"))
TRAINER_KL_ADAPT_RATE = float(os.getenv("GRAIL_TRAINER_KL_ADAPT_RATE", "1.5"))
TRAINER_KL_MIN = float(os.getenv("GRAIL_TRAINER_KL_MIN", "0.001"))
TRAINER_KL_MAX = float(os.getenv("GRAIL_TRAINER_KL_MAX", "0.2"))

# Trainer miner trust filtering (weight-based)
TRAINER_MIN_AGGREGATE_WEIGHT = float(os.getenv("GRAIL_TRAINER_MIN_AGGREGATE_WEIGHT", "0.01"))
TRAINER_MIN_TRUSTED_MINERS = int(os.getenv("GRAIL_TRAINER_MIN_TRUSTED_MINERS", "1"))

# Checkpoint retention controls
CHECKPOINT_RETENTION_LIMIT = int(os.getenv("GRAIL_CHECKPOINT_RETENTION_LIMIT", "3"))
CHECKPOINT_MILESTONE_INTERVAL = int(os.getenv("GRAIL_CHECKPOINT_MILESTONE_INTERVAL", "100"))

# Trainer identity used for checkpoint publication
TRAINER_UID = 0

# ────────────────  LOGGING  ────────────────

TRACE = 5

# ────────────────  GRAIL CRYPTOGRAPHIC CONSTANTS  ────────────────

PRIME_Q = 2_147_483_647
CHALLENGE_K = 16
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# ────────────────  TERMINATION VALIDATION HPs  ────────────────

MAX_NEW_TOKENS = 512

# Must match rollout generator default
MIN_EOS_PROBABILITY = 0.02  # Minimum probability for valid EOS termination

# Max acceptable drift between miner/validator
SANITY_CHECK_DRIFT_THRESHOLD = 0.1

# ────────────────  TOKEN SAMPLING DIST CHECK HPs  ────────────────

SAMPLING_MIN_STEPS = 30
SAMPLING_LOW_P = 0.10
SAMPLING_HIGH_P = 0.90
SAMPLING_LOW_FRAC_MIN = 0.20
SAMPLING_HIGH_FRAC_MIN = 0.50
SAMPLING_MID_FRAC_MAX = 0.40

# Minimal sampling shape hyperparameters (median gate for unimodal-low)
SAMPLING_MEDIAN_LOW_MAX = 0.30

# NOTE: this parameter so far has been a good indicator of bimodality
SAMPLING_LOW_Q10_MAX = 0.025

# Extra sanity gates for sampling shape checks
SAMPLING_MIN_TOKEN_PROB = 1e-5
SAMPLING_INITIAL_WINDOW_STEPS = 50

# ────────────────  VALIDATOR-SPECIFIC CONSTANTS  ────────────────

# Superlinear weighting exponent:
# For p > 1, w_i ∝ s_i^p amplifies differences and penalizes sybil splitting:
# splitting into k identities yields k^(1-p) * s^p < s^p.
SUPERLINEAR_EXPONENT = 4.0

# Reward comparison tolerances (used by RewardValidator)
REWARD_REL_TOL = 0.02
REWARD_ABS_TOL = 1e-6

# ────────────────  ROLLOUTS PER PROBLEM  ────────────────

ROLLOUTS_PER_PROBLEM = 16

# ────────────────  ENVIRONMENT CONFIGURATION  ────────────────

# Current environment ID (validators use this constant, never trust miner data)
CURRENT_ENV_ID = "gsm8k"

# ────────────────  EMISSION BURN MECHANISM  ────────────────

GRAIL_BURN_UID = 0
GRAIL_BURN_PERCENTAGE = 80.0

# ────────────────  MINER SAMPLING (VALIDATION COST CONTROL)  ────────────────

# Enable/disable miner-level subsampling per window.
MINER_SAMPLING_ENABLED = True

# Fraction of active miners (those with a window file) to validate per window.
# Applied after MINER_SAMPLE_MIN and before MINER_SAMPLE_MAX.
MINER_SAMPLE_RATE = 0.25

# Minimum number of active miners to validate each window (floor).
MINER_SAMPLE_MIN = 2

# Optional cap on miners validated per window. Set to None to disable.
MINER_SAMPLE_MAX = 35

# Number of windows to look back for failure-based exclusion from sampling.
# Miners with failures in the last N windows are excluded from selection.
FAILURE_LOOKBACK_WINDOWS = 14

# ────────────────  GRAIL PROOF VERIFICATION  ────────────────

# Top-K activation selection (focus on stable, important features)
# UPDATED: Reduced from 256 to 32 for higher sensitivity to training changes
PROOF_TOPK = 32

# Logarithmic bucketing parameters
PROOF_NUM_BUCKETS = 16  # Buckets per sign

# Small bounded coefficients for sketch robustness
PROOF_COEFF_RANGE = 127  # r ∈ [-127, 127]

# Multi-check tolerances (calibrate empirically via cross-framework tests)
# Sketch: modular distance on dot product
# UPDATED: Reduced from 1000 to 50 for tighter verification
PROOF_SKETCH_TOLERANCE = 50

# Rank: minimum matches required in top-5 ordering
# UPDATED: Increased from 4 to 5 for stricter rank matching
PROOF_MIN_RANK_MATCHES = 5

# Histogram: L1 distance on bucket distribution
# UPDATED: Reduced from 50 to 10 for tighter distribution matching
PROOF_HISTOGRAM_TOLERANCE = 10

# Adaptive tolerance: position importance decay rate
PROOF_POSITION_IMPORTANCE_DECAY = 100.0

# GRAIL proof version
GRAIL_PROOF_VERSION = "v1"


# ────────────────  CHECKPOINT MOD10  ────────────────

# Only for testing purposes; going to be removed later on
GRAIL_CHECKPOINT_MOD10 = False
