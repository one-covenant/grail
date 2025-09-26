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
from .environments.sat import SATParser, create_sat_prompt
from .logging_utils import MinerPrefixFilter
from .mining.rollout_generator import (
    REASONING_START,
    SYSTEM_PROMPT,
)
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
    SAMPLING_INITIAL_WINDOW_STEPS,
    SAMPLING_LOW_P,
    SAMPLING_LOW_Q10_MAX,
    SAMPLING_MEDIAN_LOW_MAX,
    SAMPLING_MIN_STEPS,
    SAMPLING_MIN_TOKEN_PROB,
    SANITY_CHECK_DRIFT_THRESHOLD,
    TOLERANCE,
)
from .shared.hf_compat import resolve_max_context_length, resolve_vocab_size

# Enable CUDA debugging for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Use the same logger as the main module
logger = logging.getLogger(__name__)
logger.addFilter(MinerPrefixFilter())


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
        r_vec_from_randomness._cache = {}
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
        logger.debug(f"Using cached sketch vector for d_model={d_model}")
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
        f"Generated sketch vector with shape={tensor.shape}, first 4 values: {tensor[:4].tolist()}"
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
        f"Selected {len(idxs)} indices from seq_len={seq_len}: {idxs[:5]}..."
        if len(idxs) > 5
        else f"Selected indices: {idxs}"
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
    logger.debug(f"Signed {len(s_vals)} s_vals with Bittensor wallet signature")
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
            logger.debug("Signature verification failed")
        return verified
    except Exception as e:
        logger.warning(f"Signature verification error: {e}")
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


def derive_canonical_sat(
    wallet_addr: str, window_hash: str, problem_index: int
) -> tuple[str, float]:
    """Derive canonical SAT seed and difficulty for miner/window/problem index.

    The seed binds problems to the miner hotkey and the window block hash.
    The difficulty is sampled ~uniformly in [0.3, 0.9] from a PRF of the
    same material to eliminate miner control while keeping a broad spread.
    """
    try:
        idx = int(problem_index)
    except Exception:
        idx = 0
    material = f"{wallet_addr}:{window_hash}:{idx}".encode()
    seed = hashlib.sha256(b"seed|" + material).hexdigest()
    diff_digest = hashlib.sha256(b"diff|" + material).digest()
    u = int.from_bytes(diff_digest[:8], "big") / float(1 << 64)
    difficulty = 0.3 + 0.6 * u
    return seed, float(difficulty)


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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
        self.model = self.model.to(self.device).eval()
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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
        self.model = self.model.to(self.device).eval()

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

    def _uid(self) -> str:
        try:
            return (
                str(self._current_wallet) if getattr(self, "_current_wallet", None) else "unknown"
            )
        except Exception:
            return "unknown"

    def _user_friendly_reason(self, event: str, fields: dict[str, object]) -> Optional[str]:
        """Return a short, user-facing explanation for failed checks.

        The goal is to help miners quickly diagnose why a check failed
        without reading validator internals. This only returns text when
        status=="fail"; otherwise returns None.
        """
        try:
            status = str(fields.get("status", ""))
            if status != "fail":
                return None

            reason = str(fields.get("reason", ""))

            # tokens_valid: verify_tokens() returned False. Root causes typically include:
            # - Using a different tokenizer/model pair (IDs out of vocab)
            # - Sequence exceeds model context window
            if event == "tokens_valid":
                return (
                    "Tokens are invalid for this model. This usually means the tokenizer/model "
                    "used for generation differs from the validator's, or the sequence exceeds the "
                    "model's context window. Ensure you tokenize and generate with the exact same "
                    "base model and tokenizer."
                )

            # proof_valid covers the cryptographic/model-binding side
            if event == "proof_valid":
                mapping: dict[str, str] = {
                    "missing_fields": (
                        "Commit is missing required fields (tokens/s_vals). Provide a complete commit."
                    ),
                    "invalid_s_vals": (
                        "s_vals list is empty or malformed. It must contain one value per token."
                    ),
                    "length_mismatch": (
                        "Length of s_vals does not match tokens. Ensure both arrays are aligned."
                    ),
                    "too_short": (
                        "Sequence is shorter than the required number of challenged indices (min_k)."
                    ),
                    "commit_signature_invalid": (
                        "Signed commit binding did not verify. Sign the exact commit payload with "
                        "your wallet hotkey and avoid modifying it afterward."
                    ),
                    "model_mismatch": (
                        "Commit binds to a different model than the validator expects. Generate "
                        "with the expected base model and include its exact name in the commit."
                    ),
                    "layer_mismatch": (
                        "Commit binds to a different layer index than expected. Use the required layer."
                    ),
                    "index_derivation_failed": (
                        "Failed to deterministically derive open indices from tokens and randomness. "
                        "Check randomness format (hex) and ensure the same tokens are used."
                    ),
                    "svals_index_oob": (
                        "An opened index exceeds s_vals length. This indicates a mismatch between "
                        "tokens and s_vals or tampering."
                    ),
                    "sketch_mismatch": (
                        "Hidden-state sketch does not match at challenged positions (beyond tolerance). "
                        "This typically means tokens were produced with a different base model/weights "
                        "or layer, or the model binding/randomness was altered."
                    ),
                    "inference_error": (
                        "Validator failed to run the model on provided tokens (e.g., invalid token IDs "
                        "for this model). Ensure tokens are valid for the declared model."
                    ),
                }
                return mapping.get(reason, "GRAIL proof failed. Check model binding and signature.")

            # sat_problem_valid ensures the SAT instance matches deterministic generation
            if event == "sat_problem_valid":
                if reason == "missing":
                    return "Commit lacks SAT problem metadata. Include seed/difficulty and clauses."
                if reason == "mismatch":
                    return (
                        "SAT instance does not match deterministic regeneration from the provided "
                        "seed/difficulty. Ensure the problem was not changed."
                    )
                return "SAT problem verification failed."

            # prompt_valid enforces canonical prompt prefix equivalence
            if event == "prompt_valid":
                if reason == "prompt_len_mismatch":
                    return (
                        "Claimed prompt length does not match the canonical prompt for this "
                        "SAT seed/template. Use the same chat template/system prompt without extra prefixes."
                    )
                if reason == "token_count_mismatch":
                    return (
                        "prompt_length + completion_length does not equal total tokens. Ensure the "
                        "reported lengths match the actual tokenization."
                    )
                if reason == "prefix_mismatch":
                    return (
                        "Token prefix does not match the canonical prompt. Likely a different chat "
                        "template, system prompt, or an added prefix."
                    )
                return "Canonical prompt prefix mismatch."

            # termination_valid enforces either max-length or confident EOS
            if event == "termination_valid":
                if reason in {"not_max_len_or_eos", "neither_maxlen_nor_eos"}:
                    return (
                        f"Sequence neither reached max_new_tokens ({int(MAX_NEW_TOKENS)}) nor ended "
                        f"with EOS having probability >= {float(MIN_EOS_PROBABILITY)}."
                    )
                if reason == "over_max_new_tokens":
                    return (
                        f"Completion length exceeds max_new_tokens ({int(MAX_NEW_TOKENS)}). "
                        "Reduce the number of generated tokens."
                    )
                if reason == "no_logits":
                    return (
                        "Validator could not compute logits to verify EOS probability. Provide the full "
                        "sequence and avoid truncation that prevents last-step logits."
                    )
                if reason == "no_tokens":
                    return "No tokens provided to validate termination."
                if reason == "completion_length_not_int":
                    return (
                        "Completion length is not an integer. Ensure the rollout metadata includes "
                        "a valid completion_length field."
                    )
                return "Termination condition not satisfied."

            # token_distribution_valid detects suspicious chosen-token probability shapes
            if event == "token_distribution_valid":
                return (
                    "Chosen-token probability distribution looks inconsistent with sampling from the "
                    "expected base model (e.g., suspicious bimodality or extremely low probabilities). "
                    "This often means tokens were generated by a different model/sampling setup, or via "
                    "prefill/prompt-composition tricks. Generate with the expected base model and a "
                    "consistent sampling configuration."
                )

            # solution_valid enforces assignment structure/consistency and satisfaction
            if event == "solution_valid":
                mapping = {
                    "parse_failed": (
                        "Could not parse a variable assignment from the completion. Output the expected format."
                    ),
                    "invalid_structure": (
                        "Claimed assignment has the wrong length/structure. It must be a boolean list "
                        "with length equal to num_vars."
                    ),
                    "assignment_mismatch": (
                        "Claimed assignment does not match what is encoded in the tokens."
                    ),
                    "not_satisfied": "The assignment does not satisfy the SAT instance.",
                }
                return mapping.get(reason, "Solution verification failed.")

            # Fallback
            return None
        except Exception:
            return None

    def _debug(self, event: str, **fields: object) -> None:
        try:
            # Attach concise user-facing description when a check fails
            desc = self._user_friendly_reason(event, dict(fields))
            if desc:
                # Avoid mutating the caller's mapping by building a new one
                extended_fields = dict(fields)
                extended_fields["desc"] = desc
            else:
                extended_fields = fields
            kv = " ".join(f"{k}={v}" for k, v in extended_fields.items())
            logger.debug(f"event={event} {kv}".rstrip())
        except Exception:
            logger.debug(f"event={event}")

    def _spawn_monitor_task(self, coro, tag: str) -> None:
        """Fire-and-forget a monitoring coroutine safely.

        Uses the currently running loop; if none, it no-ops. Any exception
        raised by the task is logged to avoid silent failures.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        try:
            task = loop.create_task(coro)
        except Exception as e:
            logger.debug(f"Failed to schedule monitor task tag={tag}: {e}")
            return

        def _done(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except Exception:
                exc = None
            if exc is not None:
                logger.debug(f"Monitor task failed tag={tag}: {exc}")

        try:
            task.add_done_callback(_done)
        except Exception:
            pass

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
                self._spawn_monitor_task(
                    monitor.log_counter("grail.token_validation_failures"),
                    tag="grail.token_validation_failures",
                )
            self._debug("tokens_valid", status="fail")
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
                self._spawn_monitor_task(
                    monitor.log_counter("validation/grail/verification_failures"),
                    tag="validation/grail/verification_failures",
                )
            self._debug("proof_valid", status="fail")
            return False, checks
        checks["proof_valid"] = True

        # Verify SAT problem was generated correctly from seed
        sat_data = commit.get("sat_problem", {})
        if not sat_data:
            logger.debug("No SAT problem data in commit")
            self._debug("sat_problem_valid", status="fail", reason="missing")
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
            self._debug("prompt_valid", status="fail", reason="mismatch")
            return False, checks

        # Enforce termination check before solution validation
        if not self._passes_termination_check(commit):
            logger.debug(
                "Termination check failed: sequence neither reached max context length "
                "nor ended with EOS token having probability >= 0.1"
            )
            self._debug("termination_valid", status="fail", reason="not_max_len_or_eos")
            return False, checks
        checks["termination_valid"] = True

        # Verify the solution if claimed successful
        rollout = commit.get("rollout", {})
        if rollout.get("success", False):
            solution_valid = self._validate_solution(commit, rollout, expected_problem, checks)
            if not solution_valid:
                return False, checks

        # Token sampling distribution shape check
        ok_sampling, sampling_stats = self._token_sampling_shape_check(commit)
        checks["token_distribution_valid"] = bool(ok_sampling)
        if not ok_sampling:
            logger.debug(f"Token sampling shape check failed: {sampling_stats}")
            try:
                self._debug(
                    "token_distribution_valid",
                    status="fail",
                    **sampling_stats,
                )
            except Exception:
                self._debug("token_distribution_valid", status="fail")
            return False, checks

        if monitor:
            self._spawn_monitor_task(
                monitor.log_counter("validation/grail/verification_successes"),
                tag="validation/grail/verification_successes",
            )
        return True, checks

    def _verify_sat_problem(self, sat_data: dict) -> Optional[Any]:
        """Regenerate SAT problem deterministically and compare fields.

        Returns the expected problem on success, None on mismatch.
        """
        try:
            difficulty = sat_data.get("difficulty", 0.5)
            seed = sat_data["seed"]
            logger.debug(f"SAT regeneration from seed '{seed}' with difficulty {difficulty}")
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
                self._debug(
                    "sat_problem_valid",
                    status="fail",
                    reason="mismatch",
                    seed=seed,
                    difficulty=float(difficulty),
                    exp_vars=int(expected_problem.num_vars),
                    got_vars=int(sat_data.get("num_vars", -1)),
                    exp_clauses=int(len(expected_problem.clauses)),
                    got_clauses=int(len(sat_data.get("clauses", []))),
                )
                return None
            return expected_problem
        except Exception as e:
            logger.debug(f"SAT verification error: {e}")
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
                    f"Prompt length mismatch: claimed={claimed_pl} expected={len(canonical_prompt_tokens)}"
                )
                self._debug(
                    "prompt_valid",
                    status="fail",
                    reason="prompt_len_mismatch",
                    claimed_pl=int(claimed_pl),
                    expected_pl=int(len(canonical_prompt_tokens)),
                )
                return False

            # Check 2: Prompt length + completion length must equal total tokens
            if claimed_pl + claimed_cl != len(commit_tokens):
                logger.debug(
                    f"Token count mismatch: prompt_length({claimed_pl}) + completion_length({claimed_cl}) = {claimed_pl + claimed_cl} "
                    f"but total tokens = {len(commit_tokens)}"
                )
                self._debug(
                    "prompt_valid",
                    status="fail",
                    reason="token_count_mismatch",
                    claimed_pl=int(claimed_pl),
                    claimed_cl=int(claimed_cl),
                    total_tokens=int(len(commit_tokens)),
                )
                return False

            # Check 3: Prefix tokens must match canonical prompt exactly
            prefix_ok = commit_tokens[:claimed_pl] == canonical_prompt_tokens
            if not prefix_ok:
                self._debug(
                    "prompt_valid",
                    status="fail",
                    reason="prefix_mismatch",
                    claimed_pl=int(claimed_pl),
                    total_tokens=int(len(commit_tokens)),
                )
            return bool(prefix_ok)
        except Exception as e:
            logger.debug(f"Prompt verification error: {e}")
            return False

    def _decode_completion_text(self, commit: dict) -> Optional[str]:
        """Decode completion text from tokens using prompt/completion lengths.

        Returns None if decoding is not possible.
        """
        try:
            tokens = commit.get("tokens", [])
            if not isinstance(tokens, list) or not tokens:
                return None
            rollout_meta = commit.get("rollout", {})
            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
            completion_len = int(rollout_meta.get("completion_length", 0) or 0)
            if completion_len > 0 and prompt_len >= 0:
                completion_ids = tokens[prompt_len : prompt_len + completion_len]
            else:
                completion_ids = tokens[prompt_len:]
            if not completion_ids:
                return None
            text = str(self.tokenizer.decode(completion_ids, skip_special_tokens=False))
            return text
        except Exception as e:
            logger.debug(f"Failed to decode completion text: {e}")
            return None

    def _extract_assignment_from_tokens(self, commit: dict, problem: Any) -> Optional[list[bool]]:
        """Extract assignment by decoding completion tokens and using SATParser."""
        try:
            text = self._decode_completion_text(commit)
            if text is None:
                return None
            parser = SATParser()
            parsed = parser.parse(text, problem)
            if not isinstance(parsed, dict):
                return None
            values_any = parsed.get("assignment", [])
            try:
                values_any = values_any[: problem.num_vars]
            except Exception:
                return None
            return [bool(x) for x in values_any]
        except Exception as e:
            logger.debug(f"Failed to extract assignment via SATParser: {e}")
            return None

    def _validate_solution(
        self,
        commit: dict,
        rollout: dict,
        expected_problem: Any,
        checks: dict[str, bool],
    ) -> bool:
        """
        Validate the solution assignment for a claimed successful SAT rollout.

        Args:
            commit: The commit data containing tokens
            rollout: The rollout data containing claimed assignment
            expected_problem: The expected SAT problem instance
            checks: Dictionary to update with validation results

        Returns:
            True if solution is valid, False otherwise
        """
        # Extract assignment from tokens (via SATParser) and compare
        token_assignment = self._extract_assignment_from_tokens(commit, expected_problem)
        claimed_assignment = rollout.get("assignment", [])

        if token_assignment is None:
            logger.debug("Missing assignment parsed from tokens for claimed success")
            checks["solution_valid"] = False
            self._debug("solution_valid", status="fail", reason="parse_failed")
            return False

        # Enforce strict structure for claimed assignment
        if (
            not isinstance(claimed_assignment, list)
            or len(claimed_assignment) != expected_problem.num_vars
            or not all(isinstance(v, bool) for v in claimed_assignment)
        ):
            logger.debug("Invalid claimed assignment structure in commit payload")
            checks["solution_valid"] = False
            self._debug(
                "solution_valid",
                status="fail",
                reason="invalid_structure",
                claimed_len=int(
                    len(claimed_assignment) if isinstance(claimed_assignment, list) else -1
                ),
                expected_len=int(expected_problem.num_vars),
            )
            return False

        if token_assignment != claimed_assignment:
            logger.debug("Assignment mismatch between tokens and commit payload")
            checks["solution_valid"] = False
            self._debug("solution_valid", status="fail", reason="assignment_mismatch")
            return False

        # Finally, check that the assignment actually solves the problem
        if not expected_problem.check_solution(token_assignment):
            logger.debug("Token-derived assignment does not solve SAT problem")
            checks["solution_valid"] = False
            self._debug("solution_valid", status="fail", reason="not_satisfied")
            return False

        checks["solution_valid"] = True
        return True

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
            self._debug("proof_valid", status="fail", reason="missing_fields")
            return False

        if not isinstance(s_vals, list) or not s_vals:
            logger.debug("Invalid or empty s_vals list")
            self._debug("proof_valid", status="fail", reason="invalid_s_vals")
            return False
        if len(tokens) != len(s_vals):
            logger.debug("tokens and s_vals length mismatch")
            self._debug("proof_valid", status="fail", reason="length_mismatch")
            return False

        # Enforce minimum coverage
        seq_len = len(tokens)
        if seq_len < int(min_k):
            logger.debug(f"Sequence too short for minimum challenge: len={seq_len}, min_k={min_k}")
            self._debug(
                "proof_valid",
                status="fail",
                reason="too_short",
                seq_len=int(seq_len),
                min_k=int(min_k),
            )
            return False

        # Verify commit signature binding
        if not verify_commit_signature(commit, prover_address):
            logger.debug("Commit signature verification failed")
            self._debug("proof_valid", status="fail", reason="commit_signature_invalid")
            return False

        # Check model/layer binding and re-derive sketch vector
        beacon = commit.get("beacon", commit.get("round_R", {}))
        model_info = commit.get("model", {})
        expected_model_name = getattr(self.model, "name_or_path", MODEL_NAME)
        if model_info.get("name") != expected_model_name:
            logger.debug("Model name mismatch in commit")
            self._debug(
                "proof_valid",
                status="fail",
                reason="model_mismatch",
                commit_model=model_info.get("name", ""),
            )
            return False
        try:
            layer_index_claim = int(model_info.get("layer_index"))
        except Exception:
            self._debug(
                "proof_valid",
                status="fail",
                reason="layer_mismatch",
                commit_layer=model_info.get("layer_index"),
            )
            return False
        if layer_index_claim != LAYER_INDEX:
            logger.debug("Layer index mismatch in commit")
            self._debug(
                "proof_valid",
                status="fail",
                reason="layer_mismatch",
                commit_layer=int(model_info.get("layer_index")),
            )
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
            self._debug("proof_valid", status="fail", reason="index_derivation_failed")
            return False

        # Ignore prover-supplied indices entirely when verifier controls the challenge
        # (kept for backward compatibility of packet shape, but not used for verification)

        # Recompute hidden states
        full_ids = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        try:
            with torch.inference_mode():
                outs = self.model(full_ids, output_hidden_states=True)
        except RuntimeError as e:
            logger.error(f"CUDA/Runtime error during verification: {e}")
            logger.error(
                f"Token count: {len(tokens)}, Token range: min={min(tokens)}, max={max(tokens)}"
            )
            _vsz = resolve_vocab_size(self.model.config)
            logger.error(f"Model vocab size: {_vsz if _vsz is not None else 'unknown'}")
            self._debug("proof_valid", status="fail", reason="inference_error")
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
                self._debug(
                    "proof_valid",
                    status="fail",
                    reason="svals_index_oob",
                    i=int(i),
                    svals_len=int(len(s_vals)),
                )
                return False
            committed_s_val = s_vals[i]

            # Sketch‐value check with proper modular distance
            local = dot_mod_q(h_layer[i], r_vec)

            # Calculate minimum distance considering modular arithmetic
            diff = abs(local - committed_s_val)
            mod_diff = min(diff, PRIME_Q - diff)  # Handle wraparound

            if mod_diff > TOLERANCE:
                logger.debug(
                    f"Sketch mismatch at index {i} ({local} vs {committed_s_val}, diff={mod_diff})"
                )
                self._debug(
                    "proof_valid",
                    status="fail",
                    reason="sketch_mismatch",
                    idx=int(i),
                    local=int(local),
                    committed=int(committed_s_val),
                    mod_diff=int(mod_diff),
                    tolerance=int(TOLERANCE),
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

        # considering that we're focusing on reasoning, we're always going to have more tokens than SAMPLING_MIN_STEPS
        if probs is None or n < SAMPLING_MIN_STEPS:
            return False, {"n": float(n), "reason": "insufficient"}

        # capture exploits if another model is used for token generation but the prefill has happened using the main model
        metrics = self._bimodality_metrics(probs)

        # Existing decisions: unimodal-low via median; bimodal via BC gated by q10
        suspicious_unimodal_low = metrics.get("median", 1.0) <= SAMPLING_MEDIAN_LOW_MAX
        low_q10 = metrics.get("q10", 1.0) <= SAMPLING_LOW_Q10_MAX
        suspicious_bimodal = low_q10 and (metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD)

        # Sanity gate: forbid extremely low chosen-token probability anywhere
        # This captures prefixes like "Check if <SOLUTION>..." where early tokens are near-0 prob.
        min_p = float(min(probs))
        low_prob_violation = min_p <= SAMPLING_MIN_TOKEN_PROB

        # Sometimes people end up putting some prefix for their completion such as
        # Check if <SOLUTION>{answer}</SOLUTION> satisfies all clauses:
        # and the the dist still stays unimoadal but the probs of some of the initial tokens
        # becomes too low like some even get the value of 0 somtimes. To capture these exploits
        # we can do a sanity check to make sure there's no tokens with value lower than 1e-5 (a constant value)
        # and also look at the probability of the initial 40 tokens (another constant) and check dist for bimodality the same way
        k = min(int(SAMPLING_INITIAL_WINDOW_STEPS), len(probs))
        initial_metrics = self._bimodality_metrics(probs[:k]) if k > 0 else {}
        low_q10_init = initial_metrics.get("q10", 1.0) <= SAMPLING_LOW_Q10_MAX
        suspicious_bimodal_initial = low_q10_init and (
            initial_metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD
        )

        suspicious = (
            suspicious_unimodal_low
            or suspicious_bimodal
            or low_prob_violation
            or suspicious_bimodal_initial
        )

        # Include brief context in metrics for debugging/telemetry
        metrics["min_p"] = min_p
        metrics["low_prob_init"] = bool(low_prob_violation)
        if k > 0:
            metrics["initial_q10"] = float(initial_metrics.get("q10", 0.0))
            metrics["initial_bc"] = float(initial_metrics.get("bc", 0.0))

        # Verbose-mode monitoring: log histogram of chosen-token probabilities (debug mode only)
        if logger.isEnabledFor(logging.DEBUG):
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

        # Accept termination via max-length only when it exactly hits the limit
        if completion_length == expected_max_new:
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
                self._debug("termination_valid", status="fail", reason="no_tokens")
                return False

            # Hard cap: if completion exceeds MAX_NEW_TOKENS, reject immediately
            expected_max_new = int(MAX_NEW_TOKENS)
            rollout = commit.get("rollout", {})
            completion_length = rollout.get("completion_length")

            if not isinstance(completion_length, int):
                logger.debug("Completion length is not an integer")
                self._debug("termination_valid", status="fail", reason="completion_length_not_int")
                return False

            if isinstance(completion_length, int) and completion_length > expected_max_new:
                logger.debug("Termination rejected due to over max_new_tokens")
                logger.debug(
                    f"completion_length={completion_length}, max_new_tokens={expected_max_new}"
                )
                self._debug("termination_valid", status="fail", reason="over_max_new_tokens")
                return False

            # Check max length termination first (most efficient)
            if self._check_max_length_termination(commit):
                return True

            # For EOS termination, we need logits
            step_logits = self._get_step_logits(tokens)
            if step_logits is None:
                logger.debug("Cannot verify EOS termination: no logits available")
                self._debug("termination_valid", status="fail", reason="no_logits")
                return False

            # Check EOS termination
            if self._check_eos_termination(commit, step_logits):
                # Run sanity check when EOS termination succeeds
                self._run_sanity_check(commit, step_logits)
                return True

            logger.debug("Termination check failed: neither max length nor valid EOS termination")
            self._debug("termination_valid", status="fail", reason="neither_maxlen_nor_eos")
            return False

        except Exception as e:
            logger.debug(f"Termination check error: {e}")
            return False
