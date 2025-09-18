#!/usr/bin/env python3
"""
GRAIL – Guaranteed Rollout Authenticity via Inference Ledger
Modified for SAT problem generation and RL rollouts
"""

import asyncio
import hashlib
import logging
import os
import random
import struct
from typing import Any, Optional, Union, cast

import bittensor as bt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from .environments import generate_sat_problem
from .environments.sat import create_sat_prompt
from .mining.rollout_generator import REASONING_START, SYSTEM_PROMPT
from .monitoring import get_monitoring_manager
from .shared.chat_templates import build_qwen_chat_template
from .shared.constants import (
    CHALLENGE_K,
    LAYER_INDEX,
    MAX_NEW_TOKENS,
    MIN_EOS_PROBABILITY,
    MODEL_NAME,
    PRIME_Q,
    RNG_LABEL,
    SAMPLING_BC_THRESHOLD,
    SAMPLING_HIGH_P,
    SAMPLING_LOW_P,
    SAMPLING_LOW_Q10_MAX,
    SAMPLING_MEDIAN_LOW_MAX,
    SAMPLING_MIN_STEPS,
    SANITY_CHECK_DRIFT_THRESHOLD,
    TOLERANCE,
)
from .shared.hf_compat import resolve_max_context_length, resolve_vocab_size

# Enable CUDA debugging for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# Use the same logger as the main module
logger = logging.getLogger(__name__)

# ──────────────────────────  CONFIGURATION  ─────────────────────────────
#  Constants are now imported from .shared.constants

COMMIT_DOMAIN = b"grail-commit-v1"


def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    """
    Pseudorandom function using SHA-256 in counter mode for arbitrary output length.
    Args:
        label: Domain separation label
        *parts: Variable number of byte strings to include in PRF input
        out_bytes: Number of output bytes required

    Returns:
        Deterministic pseudorandom bytes of length out_bytes

    Raises:
        ValueError: If out_bytes is negative or too large
        TypeError: If inputs are not bytes
    """
    # Input validation
    if out_bytes < 0:
        raise ValueError(f"out_bytes must be non-negative, got {out_bytes}")
    if out_bytes > 2**16:  # Reasonable upper limit (64KB)
        raise ValueError(f"out_bytes too large: {out_bytes} (max 65536)")
    if out_bytes == 0:
        return b""

    if not isinstance(label, bytes):
        raise TypeError(f"label must be bytes, got {type(label).__name__}")
    for i, part in enumerate(parts):
        if not isinstance(part, bytes):
            raise TypeError(f"parts[{i}] must be bytes, got {type(part).__name__}")

    # Use SHAKE256 for variable-length output if available (more efficient)
    try:
        import hashlib

        if hasattr(hashlib, "shake_256"):
            # SHAKE256 is designed for variable-length output
            shake = hashlib.shake_256()
            shake.update(label)
            shake.update(b"||")
            for part in parts[:-1] if parts else []:
                shake.update(part)
                shake.update(b"||")
            if parts:
                shake.update(parts[-1])
            return shake.digest(out_bytes)
    except Exception:
        pass  # Fall back to SHA256 method
    # Original SHA256-based expansion with optimization
    # Pre-calculate how many hash outputs we need
    hash_size = 32  # SHA256 output size
    num_blocks = (out_bytes + hash_size - 1) // hash_size
    # Build input once
    if parts:
        input_data = label + b"||" + b"||".join(parts)
    else:
        input_data = label

    # Use counter mode for expansion (more standard than chaining)
    output = bytearray(num_blocks * hash_size)

    for i in range(num_blocks):
        # Include counter in each block for better security properties
        block_input = input_data + i.to_bytes(4, "big")
        block_hash = hashlib.sha256(block_input).digest()
        output[i * hash_size : (i + 1) * hash_size] = block_hash

    return bytes(output[:out_bytes])


def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:
    # Add cache attribute to function
    if not hasattr(r_vec_from_randomness, "_cache"):
        # Initialize a simple dict cache; attribute added dynamically
        r_vec_from_randomness._cache = {}  # type: ignore[attr-defined]
    """
    Generate random projection vector from drand randomness.

    Takes drand randomness (32 bytes hex) and expands it deterministically
    into a d_model-dimensional vector using a PRF. This ensures everyone
    with the same drand value generates the same projection vector.

    Args:
        rand_hex: Hex string of drand randomness (typically from drand beacon)
        d_model: Model hidden dimension size

    Returns:
        Random projection vector of shape (d_model,) with int32 values

    Raises:
        ValueError: If rand_hex is invalid or d_model is invalid

    Note:
        Uses big-endian byte order for cross-platform consistency
    """
    # Input validation
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if d_model > 100000:  # Reasonable upper limit
        raise ValueError(f"d_model too large: {d_model} (max 100000)")
    if not rand_hex:
        raise ValueError("rand_hex cannot be empty")
    # Normalize hex string more robustly
    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string after cleaning: '{rand_hex}'")

    # Pad with leading zero if odd length
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex

    # Cache key for memoization (avoid recomputing for same inputs)
    cache_key = (clean_hex, d_model)

    # Check if we've already computed this (useful for repeated calls)
    cache: dict[tuple[str, int], torch.Tensor] = cast(
        dict[tuple[str, int], torch.Tensor],
        getattr(r_vec_from_randomness, "_cache", {}),
    )
    if cache_key in cache:
        logger.debug(f"[SketchVec] Using cached vector for d_model={d_model}")
        return cache[cache_key].clone()

    try:
        # Use PRF to expand drand randomness into d_model random integers
        # Using 4 bytes per integer for int32 range
        raw = prf(
            RNG_LABEL["sketch"],
            bytes.fromhex(clean_hex),
            out_bytes=4 * d_model,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    # Use numpy for more efficient unpacking if available
    try:
        import numpy as np

        # More efficient for large d_model
        ints_array = np.frombuffer(raw, dtype=">i4").astype(np.int32, copy=False)
        tensor = torch.from_numpy(ints_array.copy())  # copy to ensure ownership
    except ImportError as e:
        logger.error(f"Error unpacking ints_array: {e}")
        # Fallback to struct.unpack
        ints = struct.unpack(">" + "i" * d_model, raw)
        tensor = torch.tensor(ints, dtype=torch.int32)

    # Optionally normalize to unit variance (commented out to maintain compatibility)
    # tensor = tensor.float()
    # tensor = tensor / tensor.std()

    # Cache the result (limit cache size to prevent memory issues)
    if len(cache) < 100:
        cache[cache_key] = tensor.clone()
        r_vec_from_randomness._cache = cache

    logger.debug(
        f"[SketchVec] Generated vector with shape={tensor.shape}, "
        f"first 4 values: {tensor[:4].tolist()}"
    )
    return tensor


def indices_from_root(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    """
    Generate deterministic indices for proof verification.
    Args:
        tokens: List of token IDs from the model output
        rand_hex: Randomness hex string (from drand/block hash)
        seq_len: Sequence length to sample from
        k: Number of indices to select

    Returns:
        Sorted list of k indices sampled deterministically

    Raises:
        ValueError: If rand_hex is invalid or k > seq_len
    """
    # Validate inputs early
    if k > seq_len:
        raise ValueError(f"Cannot sample {k} indices from sequence of length {seq_len}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not tokens:
        raise ValueError("tokens list cannot be empty")

    # Efficient token bytes conversion using bytearray
    tokens_bytes = b"".join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()

    # Normalize hex string more robustly
    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string: '{rand_hex}'")

    # Validate hex string before conversion
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex  # Pad with leading zero if odd length

    try:
        material = prf(
            RNG_LABEL["open"],
            tokens_hash,
            bytes.fromhex(clean_hex),
            out_bytes=32,
        )
    except ValueError as e:
        raise ValueError(
            f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}"
        ) from e

    # Use deterministic sampling with seed
    rnd = random.Random(material)

    # For small k relative to seq_len, use sample (more efficient)
    # For large k, use shuffle and slice (avoids rejection sampling overhead)
    if k < seq_len * 0.1:  # If selecting less than 10% of indices
        idxs = sorted(rnd.sample(range(seq_len), k))
    else:
        # More efficient for large k
        all_indices = list(range(seq_len))
        rnd.shuffle(all_indices)
        idxs = sorted(all_indices[:k])

    logger.debug(
        f"[Indices] selected {len(idxs)} indices from seq_len={seq_len}: {idxs[:5]}..."
        if len(idxs) > 5
        else f"[Indices] selected {idxs}"
    )
    return idxs


# ─────────────────────────────  UTILITIES  ─────────────────────────────


def int_to_bytes(i: int) -> bytes:
    return struct.pack(">I", i & 0xFFFFFFFF)


def dot_mod_q(hidden: torch.Tensor, r_vec: torch.Tensor) -> int:
    # Ensure both tensors are on the same device
    device = hidden.device
    r_vec = r_vec.to(device)

    # Scale and convert to float for computation (avoid int64 issues on CUDA)
    scaled = torch.round(hidden * 1024)
    prod = torch.dot(scaled, r_vec.float())

    # Convert to int and apply modulo
    return int(prod.item()) % PRIME_Q


def sign_s_vals(s_vals: list[int], wallet: bt.wallet) -> bytes:
    """
    Sign the s_vals list using Bittensor wallet's cryptographic signature.

    Args:
        s_vals: List of s_vals to sign
        wallet: Bittensor wallet object (bt.wallet) with signing capability

    Returns:
        Signature bytes from Ed25519 signing

    Raises:
        TypeError: If wallet doesn't have signing capability
    """
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError(f"Wallet must be a bt.wallet with hotkey.sign() method, got {type(wallet)}")

    s_vals_bytes = b"".join(int_to_bytes(val) for val in s_vals)
    # Use Bittensor wallet's sign method (Ed25519 signature)
    signature: bytes = wallet.hotkey.sign(s_vals_bytes)
    logger.debug(f"[Signature] signed {len(s_vals)} s_vals with Bittensor wallet signature")
    return signature


def verify_s_vals_signature(s_vals: list[int], signature: bytes, wallet_address: str) -> bool:
    """
    Verify the signature of s_vals list using Bittensor wallet's public key.

    Args:
        s_vals: List of s_vals to verify
        signature: Signature to verify
        wallet_address: SS58 wallet address for public key verification

    Returns:
        True if signature is valid

    Raises:
        TypeError: If wallet_address is not a string
    """
    if not isinstance(wallet_address, str):
        raise TypeError(f"wallet_address must be a string SS58 address, got {type(wallet_address)}")

    s_vals_bytes = b"".join(int_to_bytes(val) for val in s_vals)

    try:
        import bittensor as bt

        # Create a keypair from the SS58 address (public key only)
        # Bittensor uses substrate under the hood but provides a cleaner interface
        keypair = bt.Keypair(ss58_address=wallet_address)
        # Verify signature using public key cryptography
        verified: bool = keypair.verify(data=s_vals_bytes, signature=signature)
        if not verified:
            logger.debug(f"Signature verification failed for {wallet_address[:8]}...")
        return verified
    except Exception as e:
        logger.warning(f"Signature verification error for {wallet_address[:8]}...: {e}")
        return False


def hash_s_vals(s_vals: list[int]) -> bytes:
    """Compute hash of s_vals for integrity checking."""
    s_vals_bytes = b"".join(int_to_bytes(val) for val in s_vals)
    return hashlib.sha256(s_vals_bytes).digest()


def verify_tokens(tokens: list[int], model_config: Union[PretrainedConfig, Any]) -> bool:
    """
    Verify token list validity for model processing.

    Args:
        tokens: List of token IDs to verify
        model_config: Model configuration object with vocab_size and max sequence length attributes

    Returns:
        True if tokens are valid, False otherwise
    """
    # Check empty tokens
    if not tokens:
        logger.warning("Empty token list in commit")
        return False

    # Validate token IDs (best-effort if vocab size available)
    vocab_size = resolve_vocab_size(model_config)
    if vocab_size is not None:
        if not _validate_token_ids(tokens, vocab_size):
            return False
    else:
        logger.debug("Model config lacks vocab_size; skipping token-id bounds check")

    # Validate sequence length
    if not _validate_sequence_length(tokens, model_config):
        return False

    return True


def _validate_token_ids(tokens: list[int], vocab_size: int) -> bool:
    """Check that all token IDs are within vocabulary bounds."""
    invalid_tokens = [t for t in tokens if not isinstance(t, int) or t < 0 or t >= vocab_size]
    if invalid_tokens:
        logger.warning(
            f"Invalid token IDs found in verification: {invalid_tokens[:10]}... "
            f"(vocab_size={vocab_size})"
        )
        return False
    return True


def _validate_sequence_length(
    tokens: list[int], model_config: Union[PretrainedConfig, Any]
) -> bool:
    """Check that token sequence doesn't exceed model's max length."""
    max_length = resolve_max_context_length(model_config)

    if len(tokens) > max_length:
        logger.warning(f"Token sequence ({len(tokens)}) exceeds model max length ({max_length})")
        return False
    return True


def hash_tokens(tokens: list[int]) -> bytes:
    """Compute hash of tokens for integrity checking."""
    tokens_bytes = b"".join(int_to_bytes(t) for t in tokens)
    return hashlib.sha256(tokens_bytes).digest()


def build_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    s_vals: list[int],
) -> bytes:
    """Build domain-separated commit binding to be signed.

    Format: SHA256(COMMIT_DOMAIN || len(x)||x for each x in
    [tokens_hash, rand_bytes, model_name_bytes, layer_index_be, s_vals_hash]).
    """

    def _len_bytes(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    rand_clean = randomness_hex.strip().replace("0x", "").replace("0X", "")
    if len(rand_clean) % 2 != 0:
        rand_clean = "0" + rand_clean
    rand_bytes = bytes.fromhex(rand_clean)
    tokens_h = hash_tokens(tokens)
    svals_h = hash_s_vals(s_vals)
    model_b = (model_name or "").encode("utf-8")
    layer_b = int(layer_index).to_bytes(4, "big", signed=True)

    h = hashlib.sha256()
    h.update(COMMIT_DOMAIN)
    for part in (tokens_h, rand_bytes, model_b, layer_b, svals_h):
        h.update(_len_bytes(part))
        h.update(part)
    return h.digest()


def sign_commit_binding(
    tokens: list[int],
    randomness_hex: str,
    model_name: str,
    layer_index: int,
    s_vals: list[int],
    wallet: bt.wallet,
) -> bytes:
    """Sign the commit-binding message with wallet hotkey."""
    if not hasattr(wallet, "hotkey") or not hasattr(wallet.hotkey, "sign"):
        raise TypeError("Wallet must provide hotkey.sign()")
    msg = build_commit_binding(tokens, randomness_hex, model_name, layer_index, s_vals)
    return wallet.hotkey.sign(msg)  # type: ignore


def verify_commit_signature(commit: dict, wallet_address: str) -> bool:
    """Verify commit signature binding tokens, randomness, model, layer, and s_vals."""
    try:
        tokens = commit["tokens"]
        s_vals = commit["s_vals"]
        beacon = commit.get("beacon", commit.get("round_R", {}))
        randomness = beacon["randomness"]
        model_info = commit.get("model", {})
        model_name = model_info.get("name", "")
        layer_index = int(model_info.get("layer_index"))
        sig = bytes.fromhex(commit["signature"])
    except Exception:
        return False
    msg = build_commit_binding(tokens, randomness, model_name, layer_index, s_vals)
    try:
        keypair = bt.Keypair(ss58_address=wallet_address)
        return keypair.verify(data=msg, signature=sig)  # type: ignore
    except Exception:
        return False


# ─────────────────────────────  PROVER  ────────────────────────────────


class Prover:
    def __init__(self, model_name: str = MODEL_NAME, wallet: bt.wallet = None) -> None:
        """
        Initialize Prover with model and Bittensor wallet for secure signatures.

        Args:
            model_name: Name of the model to load
            wallet: Bittensor wallet object (bt.wallet) for cryptographic signatures

        Raises:
            ValueError: If wallet is not provided
        """
        if wallet is None:
            raise ValueError("Prover requires a bt.wallet for secure signatures")

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad_token is properly set to prevent model confusion between padding and content
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )
        self.wallet = wallet
        self._state: dict = {}
        logger.debug("Prover initialized with wallet for secure signatures")

    def open(self, randomness_hex: str, k: int = CHALLENGE_K) -> dict:
        # Use provided randomness instead of generating beacon
        beacon_R1 = {"round": 2, "randomness": randomness_hex}
        # Use tokens instead of s_vals for index derivation
        idxs = indices_from_root(self._state["tokens"], randomness_hex, self._state["seq_len"], k)
        return {"round_R1": beacon_R1, "indices": idxs}


# ─────────────────────────────  VERIFIER  ──────────────────────────────


class Verifier:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad_token is properly set to prevent model confusion between padding and content
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )

        # Install the same Qwen-style chat template used by miners
        try:
            self.tokenizer.chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        except Exception:
            # Non-fatal: fallback to tokenizer default template
            logger.debug("Unable to set custom chat_template; using default.")

        # Cache from most recent proof verification forward pass
        self._last_tokens_hash: Optional[str] = None
        self._last_step_logits: Optional[torch.Tensor] = None
        # Current prover wallet for namespacing verbose metrics
        self._current_wallet: Optional[str] = None

    def verify_rollout(
        self,
        commit: dict,
        proof_pkg: dict,
        prover_address: str,
        *,
        challenge_randomness: str,
        min_k: int = CHALLENGE_K,
        log_identity: Optional[str] = None,
    ) -> tuple[bool, dict[str, bool]]:
        """
        Verify SAT rollout with GRAIL proof.

        Returns (is_valid, checks) where:
        - is_valid: overall verdict (unchanged semantics w.r.t. prior bool return)
        - checks: per-check booleans with concise names
            - tokens_valid: commit tokens are well-formed and within model bounds
            - proof_valid: GRAIL proof validates model identity (incl. commit binding)
            - sat_problem_valid: SAT instance matches deterministic regeneration from seed
            - prompt_valid: canonical prompt prefix exactly matches tokenized commit prefix
            - termination_valid: generation terminated correctly (max length or EOS with prob)
            - token_distribution_valid: stochastic sampling-shape heuristic passed
            - solution_valid: if success is claimed, assignment solves the instance
        """
        monitor = get_monitoring_manager()
        # Record current logging identity for namespaced debug metrics
        self._current_wallet = log_identity if log_identity is not None else prover_address

        # Initialize checks map with conservative defaults
        checks: dict[str, bool] = {
            "tokens_valid": False,
            "proof_valid": False,
            "sat_problem_valid": False,
            "prompt_valid": False,
            "termination_valid": False,
            # Heuristic: default to True when insufficient data
            "token_distribution_valid": True,
            # If no success is claimed, consider solution check vacuously true
            "solution_valid": True,
        }

        # First validate tokens before any further processing
        tokens = commit.get("tokens", [])
        if not verify_tokens(tokens, self.model.config):
            logger.debug("Token validation failed")
            if monitor:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(monitor.log_counter("grail.token_validation_failures"))
                except RuntimeError:
                    pass
            return False, checks
        checks["tokens_valid"] = True

        # Then verify the GRAIL proof - this proves the model identity
        if not self.verify(
            commit,
            proof_pkg,
            prover_address,
            challenge_randomness=challenge_randomness,
            min_k=min_k,
        ):
            logger.debug("GRAIL proof failed - model identity not verified")
            if monitor:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            monitor.log_counter("validation/grail/verification_failures")
                        )
                except RuntimeError:
                    pass
            return False, checks
        checks["proof_valid"] = True

        # Verify SAT problem was generated correctly from seed
        sat_data = commit.get("sat_problem", {})
        if not sat_data:
            logger.debug("No SAT problem data in commit")
            return False, checks

        expected_problem = self._verify_sat_problem(sat_data)
        if expected_problem is None:
            return False, checks
        checks["sat_problem_valid"] = True

        # Verify the canonical prompt prefix matches exactly
        prompt_ok = self._verify_prompt_prefix(commit, expected_problem)
        checks["prompt_valid"] = bool(prompt_ok)
        if not prompt_ok:
            logger.debug("Prompt prefix mismatch against canonical rendering")
            return False, checks

        # Enforce termination check before solution validation
        if not self._passes_termination_check(commit):
            logger.debug(
                "Termination check failed: sequence neither reached max context length "
                "nor ended with EOS token having probability >= 0.1"
            )
            return False, checks
        checks["termination_valid"] = True

        # Verify the solution if claimed successful
        # TODO: This is actually exploitable and the logic should improve later
        rollout = commit.get("rollout", {})
        if rollout.get("success", False):
            # Check that the assignment actually solves the problem
            assignment = rollout.get("assignment", [])
            if not expected_problem.check_solution(assignment):
                logger.debug("Claimed solution doesn't actually solve SAT problem")
                checks["solution_valid"] = False
                return False, checks
            checks["solution_valid"] = True

        logger.debug("SAT rollout verification successful - model identity confirmed")

        # Token sampling distribution shape check
        ok_sampling, sampling_stats = self._token_sampling_shape_check(commit)
        checks["token_distribution_valid"] = bool(ok_sampling)
        if not ok_sampling:
            logger.debug(f"Token sampling shape check failed: {sampling_stats}")
            return False, checks

        logger.debug(f"Token sampling shape check passed: {sampling_stats}")

        if monitor:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        monitor.log_counter("validation/grail/verification_successes")
                    )
            except RuntimeError:
                pass
        return True, checks

    def _verify_sat_problem(self, sat_data: dict) -> Optional[Any]:
        """Regenerate SAT problem deterministically and compare fields.

        Returns the expected problem on success, None on mismatch.
        """
        try:
            difficulty = sat_data.get("difficulty", 0.5)
            seed = sat_data["seed"]
            logger.debug(
                f"Regenerating SAT problem from seed '{seed}' with difficulty {difficulty}"
            )
            expected_problem = generate_sat_problem(seed, difficulty)
            if expected_problem.num_vars != sat_data.get(
                "num_vars"
            ) or expected_problem.clauses != sat_data.get("clauses"):
                logger.debug("SAT problem doesn't match seed generation:")
                logger.debug(
                    f"  Expected: {expected_problem.num_vars} vars, "
                    f"{len(expected_problem.clauses)} clauses"
                )
                logger.debug(
                    f"  Got: {sat_data.get('num_vars')} vars, "
                    f"{len(sat_data.get('clauses', []))} clauses"
                )
                logger.debug(f"  Seed: {seed}, Difficulty: {difficulty}")
                return None
            return expected_problem
        except Exception as e:
            logger.debug(f"SAT problem verification error: {e}")
            return None

    def _verify_prompt_prefix(self, commit: dict, problem: Any) -> bool:
        """Recreate canonical prompt (system+user via chat template) and
        enforce that commit tokens start with its exact tokenized prefix.
        """
        # TODO: this chat template and prompting approach should be properly versioned,
        # modularized and get deduplicated in the newer versions
        try:
            # Build user prompt exactly as in SATRolloutGenerator.create_prompt
            user_prompt = create_sat_prompt(problem)

            # Render canonical prompt text through template
            messages = [{"role": "user", "content": user_prompt}]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: emulate template minimalistically
                eos = self.tokenizer.eos_token or ""
                rendered = f"{SYSTEM_PROMPT}{eos}{user_prompt}{REASONING_START}"

            # Tokenize canonical prompt text to ids
            ids = (
                self.tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
                .input_ids[0]
                .tolist()
            )

            canonical_prompt_tokens = ids

            # Extract commit tokens and claimed lengths
            commit_tokens = commit.get("tokens", [])
            if not isinstance(commit_tokens, list) or not commit_tokens:
                return False
            rollout_meta = commit.get("rollout", {})
            claimed_pl = int(rollout_meta.get("prompt_length", 0) or 0)
            claimed_cl = int(rollout_meta.get("completion_length", 0) or 0)

            # Check 1: Prompt length must match canonical prompt length
            if claimed_pl != len(canonical_prompt_tokens):
                logger.debug(
                    "Prompt length mismatch: claimed=%s expected=%s",
                    claimed_pl,
                    len(canonical_prompt_tokens),
                )
                return False

            # Check 2: Prompt length + completion length must equal total tokens
            if claimed_pl + claimed_cl != len(commit_tokens):
                logger.debug(
                    "Token count mismatch: prompt_length(%s) + completion_length(%s) = %s "
                    "but total tokens = %s",
                    claimed_pl,
                    claimed_cl,
                    claimed_pl + claimed_cl,
                    len(commit_tokens),
                )
                return False

            # Check 3: Prefix tokens must match canonical prompt exactly
            prefix_ok = commit_tokens[:claimed_pl] == canonical_prompt_tokens
            return bool(prefix_ok)
        except Exception as e:
            logger.debug(f"Prompt prefix verification error: {e}")
            return False

    def verify(
        self,
        commit: dict,
        proof_pkg: dict,
        prover_address: str,
        *,
        challenge_randomness: str,
        min_k: int = CHALLENGE_K,
    ) -> bool:
        """
        Verify just the GRAIL proof portion using public key cryptography.

        Args:
            commit: Commitment data with s_vals and signature
            proof_pkg: Proof package with revealed information
            prover_address: SS58 wallet address for public key verification
            challenge_randomness: Use this randomness for index selection (verifier-chosen)
            min_k: Enforce a verifier-side minimum number of opened indices

        Returns:
            True if proof is valid, False otherwise
        """

        # Basic input validation
        try:
            tokens = commit["tokens"]
            s_vals = commit["s_vals"]
        except Exception:
            logger.debug("Missing tokens or s_vals in commit")
            return False

        if not isinstance(tokens, list) or not tokens:
            logger.debug("Invalid or empty tokens list")
            return False
        if not isinstance(s_vals, list) or not s_vals:
            logger.debug("Invalid or empty s_vals list")
            return False
        if len(tokens) != len(s_vals):
            logger.debug("tokens and s_vals length mismatch")
            return False
        # Token bounds
        vocab = resolve_vocab_size(self.model.config)
        if any((t < 0 or t >= vocab) for t in tokens):
            logger.debug("Token out of range for model vocab")
            return False
        # Enforce minimum coverage
        seq_len = len(tokens)
        if seq_len < int(min_k):
            logger.debug(f"Sequence too short for minimum challenge: len={seq_len}, min_k={min_k}")
            return False

        # Verify commit signature binding
        if not verify_commit_signature(commit, prover_address):
            logger.debug("Commit signature verification failed")
            return False

        # Check model/layer binding and re-derive sketch vector
        beacon = commit.get("beacon", commit.get("round_R", {}))
        model_info = commit.get("model", {})
        expected_model_name = getattr(self.model, "name_or_path", MODEL_NAME)
        if model_info.get("name") != expected_model_name:
            logger.debug("Model name mismatch in commit")
            return False
        try:
            layer_index_claim = int(model_info.get("layer_index"))
        except Exception:
            return False
        if layer_index_claim != LAYER_INDEX:
            logger.debug("Layer index mismatch in commit")
            return False
        r_vec = r_vec_from_randomness(beacon["randomness"], self.model.config.hidden_size)

        # Determine number of indices to open (enforce minimum)
        provided_indices = proof_pkg.get("indices", [])
        try:
            provided_k = len(provided_indices)
        except Exception:
            provided_k = 0
        k = max(int(min_k), int(provided_k)) if provided_k else int(min_k)

        # Use verifier-supplied randomness exclusively (no prover fallback)
        open_rand = challenge_randomness

        # Re-derive indices deterministically from tokens and chosen randomness
        try:
            idxs_exp = indices_from_root(tokens, open_rand, seq_len, k)
        except Exception as e:
            logger.debug(f"Index derivation failed: {e}")
            return False

        # Ignore prover-supplied indices entirely when verifier controls the challenge
        # (kept for backward compatibility of packet shape, but not used for verification)

        # Recompute hidden states
        full_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        try:
            with torch.inference_mode():
                outs = self.model(full_ids, output_hidden_states=True)
        except RuntimeError as e:
            logger.error(f"CUDA/Runtime error during model inference in verification: {e}")
            logger.error(
                f"Token count: {len(tokens)}, Token range: min={min(tokens)}, max={max(tokens)}"
            )
            _vsz = resolve_vocab_size(self.model.config)
            logger.error(f"Model vocab size: {_vsz if _vsz is not None else 'unknown'}")
            return False

        h_layer = outs.hidden_states[LAYER_INDEX][0]

        # Cache single-step logits corresponding to last generated token
        try:
            seq_len = outs.logits.size(1)
            if seq_len >= 2:
                # Logits at position t predict token at t+1
                step_logits = outs.logits[0, seq_len - 2, :].detach().to("cpu")
                # Hash current tokens to associate cache entries
                tokens_bytes = b"".join(int_to_bytes(t) for t in commit["tokens"])
                self._last_tokens_hash = hashlib.sha256(tokens_bytes).hexdigest()
                self._last_step_logits = step_logits
            else:
                self._last_tokens_hash = None
                self._last_step_logits = None
        except Exception as e:
            logger.debug(f"Unable to cache step logits for termination check: {e}")
            self._last_tokens_hash = None
            self._last_step_logits = None

        # Check each opened index (tolerance check only now)
        for i in idxs_exp:
            # Guard against malformed s_vals length
            if i >= len(s_vals):
                logger.debug("Index out of range for s_vals")
                return False
            committed_s_val = s_vals[i]

            # Sketch‐value check with proper modular distance
            local = dot_mod_q(h_layer[i], r_vec)
            logger.debug(f"[SketchCheck] idx={i}, committed={committed_s_val}, local={local}")

            # Calculate minimum distance considering modular arithmetic
            diff = abs(local - committed_s_val)
            mod_diff = min(diff, PRIME_Q - diff)  # Handle wraparound

            if mod_diff > TOLERANCE:
                logger.debug(
                    f"Sketch mismatch at index {i} ({local} vs {committed_s_val}, diff={mod_diff})"
                )
                return False

        logger.debug("GRAIL proof verification successful")
        return True

    def _collect_chosen_token_probs(self, commit: dict) -> Optional[list[float]]:
        """Collect validator probabilities of chosen tokens over completion tokens.

        Uses a single forward pass. For token at index t, predictive logits are at t-1.
        """
        try:
            tokens = commit.get("tokens", [])
            rollout = commit.get("rollout", {})
            if not isinstance(tokens, list) or len(tokens) < 2:
                return None

            prompt_len = int(rollout.get("prompt_length", 0) or 0)
            completion_len = int(rollout.get("completion_length", 0) or 0)

            if completion_len <= 0:
                # Fallback: assume everything after first token is completion
                prompt_len = max(1, prompt_len)
                completion_len = max(0, len(tokens) - prompt_len)

            if prompt_len <= 0 or completion_len <= 0:
                return None

            full_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
            full_ids = full_ids.unsqueeze(0)
            with torch.inference_mode():
                outs = self.model(full_ids)
            logits = outs.logits[0]  # [T, V]

            start_t = max(1, prompt_len)
            end_t = min(start_t + completion_len, logits.size(0))

            probs_list: list[float] = []
            for t in range(start_t, end_t):
                step_logits = logits[t - 1]
                step_probs = torch.softmax(step_logits, dim=-1)
                tok_id = int(tokens[t])
                probs_list.append(float(step_probs[tok_id].item()))
            return probs_list
        except Exception as e:
            logger.debug(f"Failed to collect chosen-token probs: {e}")
            return None

    def _bimodality_metrics(self, probs: list[float]) -> dict:
        """Compute fractions at extremes and bimodality coefficient from moments."""
        try:
            try:
                import numpy as np
            except Exception:
                np = None  # type: ignore

            x_list = probs
            if np is not None:
                x = np.asarray(x_list, dtype=np.float64)
                n = float(x.size)
                low_frac = float((x <= SAMPLING_LOW_P).mean())
                high_frac = float((x >= SAMPLING_HIGH_P).mean())
                mid_frac = max(0.0, 1.0 - low_frac - high_frac)
                m = float(x.mean())
                # Robust location stats
                med = float(np.median(x))
                try:
                    q10 = float(np.quantile(x, 0.10))
                except Exception:
                    q10 = float(np.percentile(x, 10))
                d = x - m
                s2 = float((d * d).mean())
                if s2 <= 1e-12:
                    skew = 0.0
                    kurt = 3.0
                else:
                    m3 = float((d**3).mean())
                    m4 = float((d**4).mean())
                    skew = m3 / (s2**1.5 + 1e-12)
                    kurt = m4 / (s2**2 + 1e-12)
            else:
                # Torch fallback without numpy
                tx = torch.tensor(x_list, dtype=torch.float64)
                n = float(tx.numel())
                low_frac = float((tx <= SAMPLING_LOW_P).to(torch.float64).mean().item())
                high_frac = float((tx >= SAMPLING_HIGH_P).to(torch.float64).mean().item())
                mid_frac = max(0.0, 1.0 - low_frac - high_frac)
                m = float(tx.mean().item())
                # Robust location stats
                med = float(tx.median().item())
                try:
                    q10 = float(torch.quantile(tx, torch.tensor(0.10, dtype=torch.float64)).item())
                except Exception:
                    sorted_tx = torch.sort(tx).values
                    idx = max(
                        0,
                        min(
                            sorted_tx.numel() - 1,
                            int(0.10 * (sorted_tx.numel() - 1)),
                        ),
                    )
                    q10 = float(sorted_tx[idx].item())
                d = tx - m
                s2_t = (d * d).mean()
                s2 = float(s2_t.item())
                if s2 <= 1e-12:
                    skew = 0.0
                    kurt = 3.0
                else:
                    m3 = float((d.pow(3).mean()).item())
                    m4 = float((d.pow(4).mean()).item())
                    skew = m3 / (s2**1.5 + 1e-12)
                    kurt = m4 / (s2**2 + 1e-12)

            bc = (skew * skew + 1.0) / max(kurt, 1e-6)
            return {
                "n": n,
                "mean": m,
                "median": med,
                "q10": q10,
                "low_frac": low_frac,
                "high_frac": high_frac,
                "mid_frac": mid_frac,
                "skew": skew,
                "kurtosis": kurt,
                "bc": bc,
            }
        except Exception as e:
            logger.debug(f"Failed to compute bimodality metrics: {e}")
            return {
                "n": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "q10": 0.0,
                "low_frac": 0.0,
                "high_frac": 0.0,
                "mid_frac": 0.0,
                "skew": 0.0,
                "kurtosis": 3.0,
                "bc": 0.0,
            }

    def _token_sampling_shape_check(self, commit: dict) -> tuple[bool, dict]:
        """Return (ok, stats) where ok=False indicates suspicious distribution"""
        probs = self._collect_chosen_token_probs(commit)
        n = 0 if probs is None else len(probs)
        if probs is None or n < SAMPLING_MIN_STEPS:
            return True, {"n": float(n), "reason": "insufficient"}

        metrics = self._bimodality_metrics(probs)

        # Simplified decision: unimodal-low via median; bimodal via BC gated by q10
        suspicious_unimodal_low = metrics.get("median", 1.0) <= SAMPLING_MEDIAN_LOW_MAX
        low_q10 = metrics.get("q10", 1.0) <= SAMPLING_LOW_Q10_MAX
        suspicious_bimodal = low_q10 and (metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD)
        suspicious = suspicious_unimodal_low or suspicious_bimodal

        # Verbose-mode monitoring: log histogram of chosen-token probabilities
        self._log_verbose_monitoring_metrics(probs, metrics)

        return (not suspicious), metrics

    def _log_verbose_monitoring_metrics(self, probs: list[float], metrics: dict) -> None:
        """Log verbose monitoring metrics for debugging and threshold tuning.

        Args:
            probs: List of chosen token probabilities
            metrics: Dictionary containing computed bimodality metrics
        """
        try:
            monitor = get_monitoring_manager()
            if not monitor:
                return

            root_level = logger.getEffectiveLevel()

            if root_level > logging.DEBUG:
                return

            try:
                loop = asyncio.get_event_loop()

                if not loop.is_running():
                    return

                # Namespace metrics per-miner using uid-first layout in W&B
                wallet_ns = (
                    f"{self._current_wallet}/sampling_shape_check"
                    if getattr(self, "_current_wallet", None)
                    else "sampling_shape_check"
                )

                # Log the histogram using the more reliable approach
                asyncio.create_task(monitor.log_histogram(f"{wallet_ns}/hist", probs))
                # Summary statistics under the sampling_shape_check prefix for easier navigation
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/mean", metrics["mean"]))
                asyncio.create_task(
                    monitor.log_gauge(f"{wallet_ns}/median", metrics.get("median", 0.0))
                )
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/q10", metrics.get("q10", 0.0)))
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/low_frac", metrics["low_frac"]))
                asyncio.create_task(
                    monitor.log_gauge(f"{wallet_ns}/high_frac", metrics["high_frac"])
                )
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/mid_frac", metrics["mid_frac"]))
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/bc", metrics["bc"]))
                asyncio.create_task(monitor.log_gauge(f"{wallet_ns}/n", metrics["n"]))
                logger.debug("-------- log_verbose_monitoring_metrics finished--------")
            except RuntimeError:
                logger.debug("-------- log_verbose_monitoring_metrics failed--------")
                pass
        except Exception:
            pass

    def _check_max_length_termination(self, commit: dict) -> bool:
        """Check if completion reached the configured generation max length."""
        expected_max_new = int(MAX_NEW_TOKENS)

        rollout = commit.get("rollout", {})
        completion_length = rollout.get("completion_length")

        if isinstance(completion_length, int) and completion_length >= expected_max_new:
            logger.debug(
                f"Termination via max length: completion_length={completion_length} "
                f">= expected_max_new={expected_max_new}"
            )
            return True
        return False

    def _check_eos_termination(self, commit: dict, step_logits: torch.Tensor) -> bool:
        """Check if sequence terminated with sufficient EOS probability."""
        tokens = commit.get("tokens", [])
        if not tokens:
            return False

        eos_id = self.tokenizer.eos_token_id
        last_token = tokens[-1]

        if last_token != eos_id:
            return False

        # Compute EOS probability from validator's logits
        probs = torch.softmax(step_logits, dim=-1)
        p_eos = float(probs[eos_id].item())

        if p_eos >= MIN_EOS_PROBABILITY:
            logger.debug(f"Termination via EOS with p={p_eos:.4f} >= {MIN_EOS_PROBABILITY}")
            return True

        logger.debug(f"EOS probability too low: p={p_eos:.4f} < {MIN_EOS_PROBABILITY}")
        return False

    def _run_sanity_check(self, commit: dict, step_logits: torch.Tensor) -> None:
        """Compare miner vs validator logprobs (logging only, does not affect validation)."""
        try:
            tokens = commit.get("tokens", [])
            rollout = commit.get("rollout", {})
            miner_logprobs = rollout.get("token_logprobs")

            if not isinstance(miner_logprobs, list) or len(miner_logprobs) == 0:
                return

            last_token = tokens[-1]
            probs = torch.softmax(step_logits, dim=-1)
            p_last = float(probs[last_token].item())
            miner_p_last = float(torch.exp(torch.tensor(miner_logprobs[-1])).item())
            drift = abs(p_last - miner_p_last)

            if drift > SANITY_CHECK_DRIFT_THRESHOLD:
                logger.debug(
                    f"Logprob drift detected: validator={p_last:.4f} vs miner={miner_p_last:.4f} "
                    f"(drift={drift:.4f} > {SANITY_CHECK_DRIFT_THRESHOLD})"
                )
        except Exception as e:
            logger.debug(f"Sanity check failed: {e}")

    def _get_step_logits(self, tokens: list[int]) -> Optional[torch.Tensor]:
        """Get logits for the last generation step, using cache when possible."""
        # Try to use cached logits from the proof verification pass
        try:
            tokens_bytes = b"".join(int_to_bytes(t) for t in tokens)
            cur_hash = hashlib.sha256(tokens_bytes).hexdigest()
            if self._last_tokens_hash == cur_hash and self._last_step_logits is not None:
                return self._last_step_logits
        except Exception:
            pass

        # Cache miss: run minimal forward pass
        try:
            full_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.inference_mode():
                outs = self.model(full_ids)
            if outs.logits.size(1) < 2:
                logger.debug("Not enough timesteps to compute last-step logits")
                return None
            return outs.logits[0, outs.logits.size(1) - 2, :].detach().to("cpu")  # type: ignore
        except Exception as e:
            logger.debug(f"Failed to compute step logits: {e}")
            return None

    def _passes_termination_check(self, commit: dict) -> bool:
        """
        Orchestrate termination validation with proper separation of concerns.

        Returns True if either:
        1. Completion reached max generation length, OR
        2. Sequence ended with EOS token having sufficient probability
        """
        try:
            tokens = commit.get("tokens", [])
            if not tokens:
                return False

            # Check max length termination first (most efficient)
            if self._check_max_length_termination(commit):
                return True

            # For EOS termination, we need logits
            step_logits = self._get_step_logits(tokens)
            if step_logits is None:
                logger.debug("Cannot verify EOS termination: no logits available")
                return False

            # Check EOS termination
            if self._check_eos_termination(commit, step_logits):
                # Run sanity check when EOS termination succeeds
                self._run_sanity_check(commit, step_logits)
                return True

            logger.debug("Termination check failed: neither max length nor valid EOS termination")
            return False

        except Exception as e:
            logger.debug(f"Termination check error: {e}")
            return False
