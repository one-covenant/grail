"""GRAIL Protocol Constants.

Immutable values that all network participants must agree on. Miner-validator
proof verification, reward computation, and incentive scoring depend on these.

Rules:
- No os.getenv() overrides. These are compile-time constants.
- Changes require bumping GRAIL_PROOF_VERSION and coordinated deployment.
- Each constant has a comment explaining what breaks if you change it.
"""

# ────────────────  GRAIL PROOF VERSION  ────────────────

# Protocol version tag embedded in every rollout. Validators reject mismatched versions.
GRAIL_PROOF_VERSION = "v5"

# ────────────────  CRYPTOGRAPHIC CONSTANTS  ────────────────

# Mersenne prime for modular sketch arithmetic.
# Changing breaks: all proof verification (sketch mod changes).
PRIME_Q = 2_147_483_647

# Number of random challenge positions per completion.
# Changing breaks: proof generation/verification position count mismatch.
CHALLENGE_K = 32

# PRF domain labels for different randomness derivations.
# Changing breaks: index derivation, sketch computation.
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}

# Transformer layer index for hidden state extraction (-1 = last layer).
# Changing breaks: proof hidden state extraction layer mismatch.
LAYER_INDEX = -1

# Batch size for proof computation (log-softmax / GRAIL commitments).
# Fixed constant: changing causes subtle numerical divergence between
# miner and validator due to padding-induced floating-point differences.
PROOF_BATCH_SIZE = 16

# Top-K activation selection for sketch computation.
# Changing breaks: sketch bucket population differs between miner/validator.
PROOF_TOPK = 16

# Logarithmic bucketing: buckets per sign (16 total = 8 positive + 8 negative).
# Changing breaks: sketch vector dimensionality mismatch.
PROOF_NUM_BUCKETS = 8

# Bounded coefficient range for sketch robustness: r in [-127, 127].
# Changing breaks: r_vec generation, sketch magnitude.
PROOF_COEFF_RANGE = 127

# Sketch tolerance at position 0. Covers cross-GPU drift across 6 models
# (Qwen2.5, Qwen3, Llama-3.2) and 3 GPU architectures (B200, A100, L40).
# Empirical max diff across ~300M positions = 3979. Base of 6000 gives ~50%
# headroom (matching TOPLOC's empirical safety margin) while staying below
# the LSH "two-bucket-shift" knee (~8000) to preserve distillation resistance.
# Forgery prob at base: 10^-167. Changing breaks valid proofs.
PROOF_SKETCH_TOLERANCE_BASE = 6000

# Sketch tolerance sqrt growth factor per position.
# FP divergence in causal attention grows as O(sqrt(P)) from different
# reduction orders across SDPA/FA2 implementations, torch versions, and GPUs.
# tolerance(P) = base + growth * sqrt(P). At pos 8192: tol=6452, forgery=10^-166.
# Changing breaks: tolerance envelope too tight (false rejections) or too loose (false accepts).
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

# Attention implementation forced across all model loading paths.
# FA2 is padding-invariant (uses flash_attn_varlen_func with unpadding), preventing
# sketch divergence when batch sizes vary between miner and validator.
# Changing breaks: proof verification (different FP accumulation order).
ATTN_IMPLEMENTATION = "flash_attention_2"

# ────────────────  TIMING (CONSENSUS)  ────────────────

# Blocks per window. All roles use this to determine window boundaries.
# Changing breaks: miner/validator window alignment.
WINDOW_LENGTH = 30

# Bittensor block time target average (seconds).
# Changing breaks: timing budget calculations across all roles.
BLOCK_TIME_SECONDS = 12

# Typical variance in block production time (seconds).
# Changing breaks: upload deadline grace period calculation.
BLOCK_TIME_VARIANCE = 3

# Network latency allowance for file uploads (seconds).
# Changing breaks: upload deadline enforcement.
NETWORK_UPLOAD_LATENCY = 30

# Grace period = block variance + upload latency.
# Used by validators to allow late uploads within tolerance.
UPLOAD_GRACE_PERIOD = BLOCK_TIME_VARIANCE + NETWORK_UPLOAD_LATENCY

# Buffer for future drand beacon (seconds after deadline, prevents gaming).
# Changing breaks: drand-based randomness derivation timing.
DRAND_FUTURE_BUFFER = 30

# ────────────────  ROLLOUT GENERATION  ────────────────

# GRPO group size: rollouts generated per problem.
# Changing breaks: advantage computation, group-level validation.
ROLLOUTS_PER_PROBLEM = 16

# Network-wide PROTOCOL CAP on completion length: validators reject any
# rollout whose completion exceeds this. The trainer's per-checkpoint
# ``generation_params.max_tokens`` is the actual operating limit and must
# be <= this value; the miner drives its backend with
# ``min(metadata.max_tokens, MAX_NEW_TOKENS_PROTOCOL_CAP)``.
# This is NOT a default; it is an immutable upper bound that miner and
# validator agree on at code level. Changing it would require a
# coordinated network upgrade and a GRAIL_PROOF_VERSION bump.
MAX_NEW_TOKENS_PROTOCOL_CAP = 8192

# Minimum EOS probability for valid termination.
# Changing breaks: termination validation (TerminationValidator).
MIN_EOS_PROBABILITY = 0.02

# Max acceptable drift between miner/validator logprobs.
# Changing breaks: logprob sanity check threshold.
SANITY_CHECK_DRIFT_THRESHOLD = 0.1

# ────────────────  REWARD PARAMETERS  ────────────────

# Sigmoid formulation: R = sigmoid(1{correct} + min(speedup, clip) - delta)
# Changing breaks: reward computation mismatch between miner and validator.
SIGMOID_DELTA = 1.8
SPEEDUP_CLIP = 3.0
SIGMOID_KERNEL_WEIGHT = 0.80

# Reward comparison tolerances (used by RewardValidator).
# Changing breaks: valid rewards rejected or invalid rewards accepted.
REWARD_REL_TOL = 0.02
REWARD_ABS_TOL = 1e-6

# ────────────────  ECONOMIC / INCENTIVE  ────────────────

# Superlinear weighting exponent for sybil resistance.
# w_i proportional to s_i^p. Splitting into k identities yields k^(1-p) * s^p < s^p.
# Changing breaks: weight computation, incentive economics.
SUPERLINEAR_EXPONENT = 4.0

# Maximum unique rollouts per miner per window that count toward weight allocation.
# The effective period cap = this value x rolling_windows.
# Changing breaks: weight normalization, cap enforcement.
UNIQUE_ROLLOUTS_CAP = 5000
UNIQUE_ROLLOUTS_CAP_ENABLED = True

# Emission burn: percentage burned to UID 0, remainder distributed to miners.
# Changing breaks: weight distribution economics.
GRAIL_BURN_UID = 0
GRAIL_BURN_PERCENTAGE = 90.0

# Trainer identity UID for checkpoint publication and data filtering.
# Changing breaks: checkpoint discovery, trust list filtering.
TRAINER_UID = 0

# ────────────────  VALIDATION RULES  ────────────────

# Miner sampling parameters (validator-local policy, but all validators must agree).
# Changing breaks: sampling fairness, validation coverage consistency.
MINER_SAMPLING_ENABLED = True
MINER_SAMPLE_RATE = 0.25
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

# Failure lookback for exclusion from sampling.
# Changing breaks: failure gating duration.
FAILURE_LOOKBACK_WINDOWS = 14

# File size bounds for valid rollout window files.
# Changing breaks: file acceptance/rejection thresholds.
MIN_ROLLOUT_FILE_SIZE_BYTES = 200
MAX_ROLLOUT_FILE_SIZE_BYTES = 350 * 1024 * 1024  # 350 MB

# ────────────────  TOKEN SAMPLING DISTRIBUTION CHECK  ────────────────

# Heuristic parameters for the DistributionValidator (soft check).
# Changing breaks: distribution shape detection thresholds.
SAMPLING_MIN_STEPS = 30
SAMPLING_LOW_P = 0.10
SAMPLING_HIGH_P = 0.90
SAMPLING_LOW_FRAC_MIN = 0.20
SAMPLING_HIGH_FRAC_MIN = 0.50
SAMPLING_MID_FRAC_MAX = 0.40
SAMPLING_MEDIAN_LOW_MAX = 0.30
SAMPLING_LOW_Q10_MAX = 0.025
SAMPLING_MIN_TOKEN_PROB = 1e-5
SAMPLING_INITIAL_WINDOW_STEPS = 50

# ────────────────  ENVIRONMENT  ────────────────

# Current environment ID. Validators use this constant, never trust miner data.
# Changing breaks: environment selection mismatch between roles.
CURRENT_ENV_ID = "triton_kernel"

# ────────────────  STORAGE PATHS  ────────────────

# R2 bucket prefix for all checkpoints. Must match across trainer/miner/validator.
CHECKPOINT_PREFIX = "grail/checkpoints/"

# Subdirectory names for checkpoint types.
CHECKPOINT_SUBDIR_DELTA = "DELTA"
CHECKPOINT_SUBDIR_FULL = "FULL"

# Checkpoint type identifiers (used in metadata.json).
CHECKPOINT_TYPE_DELTA = "DELTA"
CHECKPOINT_TYPE_FULL = "FULL"

# ────────────────  TRUST LIST  ────────────────

# Key prefix for trust list files in R2.
TRUST_LIST_KEY_PREFIX = "grail/trust/trust_list_"
TRUST_LIST_VERSION = 1
