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
UPLOAD_TIMEOUT = float(os.getenv("UPLOAD_TIMEOUT", "90"))

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

# ────────────────  Trainer MODEL CONFIGURATION  ────────────────

# Trainer hyperparameters (env configurable)
TRAINER_LR = float(os.getenv("GRAIL_TRAINER_LR", "1e-6"))
TRAINER_EPOCHS = int(os.getenv("GRAIL_TRAINER_EPOCHS", "1"))
TRAINER_BATCH_SIZE = int(os.getenv("GRAIL_TRAINER_BATCH_SIZE", "16"))
TRAINER_MAX_LENGTH = int(os.getenv("GRAIL_TRAINER_MAX_LENGTH", "3072"))
TRAINER_GRAD_CLIP = float(os.getenv("GRAIL_TRAINER_GRAD_CLIP", "0.5"))
TRAINER_WARMUP_STEPS = int(os.getenv("GRAIL_TRAINER_WARMUP_STEPS", "10"))
TRAINER_KL_COEF = float(os.getenv("GRAIL_TRAINER_KL_COEF", "0.02"))
TRAINER_ENTROPY_COEF = float(os.getenv("GRAIL_TRAINER_ENTROPY_COEF", "0.001"))
TRAINER_ADV_CLIP_PERCENTILE = float(os.getenv("GRAIL_TRAINER_ADV_CLIP_PERCENTILE", "99.0"))
TRAINER_GROUP_ADV_SUM_TOL = float(os.getenv("GRAIL_TRAINER_GROUP_ADV_SUM_TOL", "0.01"))
TRAINER_GRAD_ACCUM_STEPS = int(os.getenv("GRAIL_TRAINER_GRAD_ACCUM_STEPS", "8"))

# Importance sampling and PPO-style clipping
TRAINER_USE_IS = os.getenv("GRAIL_TRAINER_USE_IS", "1") == "1"
TRAINER_PPO_CLIP_EPS = float(os.getenv("GRAIL_TRAINER_PPO_CLIP_EPS", "0.2"))
# Asymmetric upper bound for PPO clipping (DAPO-style). If unset, defaults to 0.28.
TRAINER_PPO_CLIP_EPS_UPPER = float(os.getenv("GRAIL_TRAINER_PPO_CLIP_EPS_UPPER", "0.28"))
# Hard ceiling for importance sampling ratios to prevent instability (DCPO-style safety)
TRAINER_IS_RATIO_MAX = float(os.getenv("GRAIL_TRAINER_IS_RATIO_MAX", "10.0"))

# Log-ratio clamp for numerical stability when exponentiating to ratios
TRAINER_LOGRATIO_CLAMP = float(os.getenv("GRAIL_TRAINER_LOGRATIO_CLAMP", "5.0"))

# Adaptive KL settings
TRAINER_ADAPTIVE_KL = os.getenv("GRAIL_TRAINER_ADAPTIVE_KL", "1") == "1"
TRAINER_KL_TARGET = float(os.getenv("GRAIL_TRAINER_KL_TARGET", "0.04"))
TRAINER_KL_ADAPT_RATE = float(os.getenv("GRAIL_TRAINER_KL_ADAPT_RATE", "1.5"))
TRAINER_KL_MIN = float(os.getenv("GRAIL_TRAINER_KL_MIN", "0.001"))
TRAINER_KL_MAX = float(os.getenv("GRAIL_TRAINER_KL_MAX", "0.2"))

# Flash Attention for training optimization
TRAINER_USE_FLASH_ATTENTION = os.getenv("GRAIL_TRAINER_USE_FLASH_ATTENTION", "1") == "1"

# Gradient checkpointing for training memory efficiency
# Reduces activation memory by recomputing on backward pass (~20-30% memory reduction)
# Trade-off: ~10-15% slower training (recomputation cost), but enables larger batches/longer sequences
TRAINER_USE_GRADIENT_CHECKPOINTING = (
    os.getenv("GRAIL_TRAINER_USE_GRADIENT_CHECKPOINTING", "1") == "1"
)

# Trainer miner trust filtering (weight-based)
TRAINER_MIN_AGGREGATE_WEIGHT = float(os.getenv("GRAIL_TRAINER_MIN_AGGREGATE_WEIGHT", "0.01"))
TRAINER_MIN_TRUSTED_MINERS = int(os.getenv("GRAIL_TRAINER_MIN_TRUSTED_MINERS", "1"))

# ────────────────  GRPO DATA FILTERING (TWO-STAGE)  ────────────────
# Stage 1 (fast structural/cheap filters) happens before Stage 2.
# Defaults are conservative so behavior is unchanged unless configured.
GRPO_MAX_GROUPS = int(os.getenv("GRAIL_GRPO_MAX_GROUPS", "32"))
GRPO_MAX_COMPLETION_TOKENS = int(os.getenv("GRAIL_GRPO_MAX_COMPLETION_TOKENS", "2048"))

# GRPO loss aggregation variant
# Options: 'grpo' (per-sequence), 'bnpo' (global token avg),
#          'dapo' (distributed token norm), 'dr_grpo' (fixed denominator)
GRPO_VARIANT = os.getenv("GRAIL_GRPO_VARIANT", "dapo")

# Importance sampling level for policy gradient computation
# Options: 'sequence' (one ratio per sequence), 'token' (per-token ratios)
# 'token': More fine-grained clipping, inspired by HuggingFace TRL GRPO
IMPORTANCE_SAMPLING_LEVEL = os.getenv("GRAIL_IMPORTANCE_SAMPLING_LEVEL", "sequence")

# Stage 2 (refinement): quality/efficiency-oriented filters
# Require at least this fraction of successes within a group (0 disables)
GRPO_MIN_SUCCESS_FRACTION = float(os.getenv("GRAIL_GRPO_MIN_SUCCESS_FRACTION", "0.0"))
# Minimum mean reward/token across a group's rollouts (<=0 disables)
GRPO_MIN_REWARD_PER_TOKEN = float(os.getenv("GRAIL_GRPO_MIN_REWARD_PER_TOKEN", "0.0"))
# Drop lowest X quantile of groups by reward/token (0 disables)
GRPO_REWARD_PER_TOKEN_DROP_QUANTILE = float(
    os.getenv("GRAIL_GRPO_REWARD_PER_TOKEN_DROP_QUANTILE", "0.0")
)

# Group ranking: combined efficiency score weights (must sum to 1.0)
# Higher reward_per_token weight prioritizes token efficiency (GFPO)
# Higher advantage_variance weight prioritizes learning signal strength
GRPO_RANKING_REWARD_WEIGHT = float(os.getenv("GRAIL_GRPO_RANKING_REWARD_WEIGHT", "0.7"))
GRPO_RANKING_VARIANCE_WEIGHT = float(os.getenv("GRAIL_GRPO_RANKING_VARIANCE_WEIGHT", "0.3"))

# Checkpoint retention controls
CHECKPOINT_MILESTONE_INTERVAL = int(os.getenv("GRAIL_CHECKPOINT_MILESTONE_INTERVAL", "0"))

# R2 retention limits (used by checkpoint_publisher for trainer uploads)
# BASE: complete model weights (~14GB); DELTA: sparse diffs that depend on a BASE
# DELTA_RETENTION should be > DELTA_BASE_INTERVAL to prevent orphaned deltas
BASE_CHECKPOINT_RETENTION_LIMIT = int(os.getenv("GRAIL_BASE_CHECKPOINT_RETENTION_LIMIT", "3"))
DELTA_CHECKPOINT_RETENTION_LIMIT = int(os.getenv("GRAIL_DELTA_CHECKPOINT_RETENTION_LIMIT", "15"))

# Trainer identity used for checkpoint publication
TRAINER_UID = 0

# ────────────────  LOGGING  ────────────────

TRACE = 5

# ────────────────  GRAIL CRYPTOGRAPHIC CONSTANTS  ────────────────

PRIME_Q = 2_147_483_647
CHALLENGE_K = 16
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}
LAYER_INDEX = -1

# ────────────────  TERMINATION VALIDATION HPs  ────────────────

MAX_NEW_TOKENS = 2048

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
CURRENT_ENV_ID = "mbpp"

# ────────────────  EMISSION BURN MECHANISM  ────────────────

GRAIL_BURN_UID = 0
GRAIL_BURN_PERCENTAGE = 80.0

# ────────────────  UNIQUE ROLLOUTS CAP  ────────────────

# Maximum unique rollouts per miner that count toward weight allocation.
# Miners are rewarded proportionally to how close they are to this cap.
# This cap covers the full 12-window rolling period (5120 per window × 12 windows).
UNIQUE_ROLLOUTS_CAP = 61440

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

# Minimum file size for valid rollout window files (bytes).
# Empty window files are ~170 bytes (just JSON structure with empty array).
# Files below this threshold are considered to have 0 rollouts and are filtered.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200

# ────────────────  GRAIL PROOF VERIFICATION  ────────────────

# Top-K activation selection (focus on stable, important features)
# UPDATED: Reduced from 256 to 32 for higher sensitivity to training changes
PROOF_TOPK = 32

# Logarithmic bucketing parameters
PROOF_NUM_BUCKETS = 16  # Buckets per sign

# Small bounded coefficients for sketch robustness
PROOF_COEFF_RANGE = 127  # r ∈ [-127, 127]

# Sketch tolerance: modular distance on dot product
# Calibrated empirically via cross-framework tests
PROOF_SKETCH_TOLERANCE = 50

# Adaptive tolerance: position importance decay rate
PROOF_POSITION_IMPORTANCE_DECAY = 100.0

# GRAIL proof version
GRAIL_PROOF_VERSION = "v1"

# ────────────────  PARAMETER CHANGE TRACKING  ────────────────

# Measure parameter changes every N optimizer steps (0 disables tracking)
PARAM_CHANGE_MEASURE_INTERVAL = int(os.getenv("GRAIL_PARAM_CHANGE_MEASURE_INTERVAL", "0"))

# Primary threshold for classifying a parameter as "changed"
# With small LR (1e-6), weight updates can be very small (1e-9 to 1e-6)
# Use 1e-10 as floor to catch all meaningful changes
PARAM_CHANGE_THRESHOLD = float(os.getenv("GRAIL_PARAM_CHANGE_THRESHOLD", "0.0"))

# Enable per-layer sparsity breakdown
PARAM_CHANGE_TRACK_PER_LAYER = os.getenv("GRAIL_PARAM_CHANGE_TRACK_PER_LAYER", "1") == "1"

# Enable per-component breakdown (q_proj, v_proj, gate_proj, etc.)
PARAM_CHANGE_TRACK_COMPONENTS = os.getenv("GRAIL_PARAM_CHANGE_TRACK_COMPONENTS", "1") == "1"

# Enable sign flip tracking (detects oscillation/instability)
PARAM_CHANGE_TRACK_SIGN_FLIPS = os.getenv("GRAIL_PARAM_CHANGE_TRACK_SIGN_FLIPS", "1") == "1"

# Epsilon for relative delta computation to avoid division by zero
PARAM_CHANGE_RELATIVE_EPS = float(os.getenv("GRAIL_PARAM_CHANGE_RELATIVE_EPS", "1e-10"))

# ────────────────  SPARSE QUALITY ANALYSIS  ────────────────

# Enable/disable sparse quality analysis (runs at same interval as param tracking)
SPARSE_QUALITY_ENABLED = os.getenv("GRAIL_SPARSE_QUALITY_ENABLED", "0") == "1"

# ────────────────  CHECKPOINT PATH CONFIGURATION  ────────────────

# R2 bucket prefix for all checkpoints
CHECKPOINT_PREFIX = "grail/checkpoints/"

# Subdirectory names for checkpoint types
# At anchor windows, both DELTA and FULL coexist under:
#   checkpoint-{window}/DELTA/  (sparse delta for caught-up consumers)
#   checkpoint-{window}/FULL/   (full weights for new joiners)
CHECKPOINT_SUBDIR_DELTA = "DELTA"
CHECKPOINT_SUBDIR_FULL = "FULL"

# Checkpoint type identifiers (used in metadata.json)
CHECKPOINT_TYPE_DELTA = "DELTA"
CHECKPOINT_TYPE_FULL = "FULL"


# ────────────────  DELTA CHECKPOINT CONFIGURATION  ────────────────

# Upload FULL checkpoint every N windows (deltas for intermediate windows)
# Set to 1 to disable delta uploads (all FULL checkpoints)
DELTA_BASE_INTERVAL = int(os.getenv("GRAIL_DELTA_BASE_INTERVAL", "10"))

# Threshold for sparse delta (0 = keep all non-zero deltas)
DELTA_THRESHOLD = float(os.getenv("GRAIL_DELTA_THRESHOLD", "0.0"))

# Enable/disable delta checkpoint uploads (fallback to full if disabled)
DELTA_CHECKPOINT_ENABLED = os.getenv("GRAIL_DELTA_CHECKPOINT_ENABLED", "1") == "1"

# ──────────────── INVARIANT VALIDATION ────────────────────────────────────────
# Ensure DELTA_CHECKPOINT_RETENTION_LIMIT >= DELTA_BASE_INTERVAL
# This guarantees that retained deltas always have their base checkpoint available.
# If violated, deltas can become orphaned when their base is deleted.
assert DELTA_CHECKPOINT_RETENTION_LIMIT >= DELTA_BASE_INTERVAL, (
    f"DELTA_CHECKPOINT_RETENTION_LIMIT ({DELTA_CHECKPOINT_RETENTION_LIMIT}) must be >= "
    f"DELTA_BASE_INTERVAL ({DELTA_BASE_INTERVAL}) to prevent orphaned delta checkpoints. "
    f"Deltas are created every DELTA_BASE_INTERVAL windows, so we must keep at least that many "
    f"to ensure validators can reconstruct recent checkpoints."
)

# ────────────────  ASYNC TRAINING CONFIGURATION  ────────────────

# Upload worker poll interval (seconds between snapshot checks)
SNAPSHOT_POLL_INTERVAL_SECONDS = int(os.getenv("GRAIL_SNAPSHOT_POLL_INTERVAL", "30"))

# Training heartbeat timeout for liveness monitoring (15 minutes)
TRAINING_HEARTBEAT_TIMEOUT_SECONDS = int(os.getenv("GRAIL_TRAINING_HEARTBEAT_TIMEOUT", "900"))

# Upload retry configuration
UPLOAD_RETRY_MAX_ATTEMPTS = int(os.getenv("GRAIL_UPLOAD_RETRY_MAX_ATTEMPTS", "3"))
UPLOAD_RETRY_BACKOFF_BASE = int(os.getenv("GRAIL_UPLOAD_RETRY_BACKOFF_BASE", "60"))

# ────────────────  HELPER FUNCTIONS  ────────────────


def is_kl_enabled() -> bool:
    """Check if KL divergence is enabled based on TRAINER_KL_COEF.

    Returns:
        True if KL coefficient is greater than zero, False otherwise.
    """
    return float(TRAINER_KL_COEF) > 0.0


# ────────────────  CHECKPOINT MOD10  ────────────────

# Only for testing purposes; going to be removed later on
GRAIL_CHECKPOINT_MOD10 = False
