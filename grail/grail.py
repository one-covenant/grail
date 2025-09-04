#!/usr/bin/env python3
"""
GRAIL – Guaranteed Rollout Authenticity via Inference Ledger
Modified for SAT problem generation and RL rollouts
"""

import hashlib
import logging
import os
import random
import struct
from typing import List, Optional

import bittensor as bt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .environments import SATProblem, generate_sat_problem, SATRolloutGenerator
from .shared.constants import (
    PRIME_Q,
    CHALLENGE_K,
    TOLERANCE,
    MODEL_NAME,
    LAYER_INDEX,
    RNG_LABEL,
    DEFAULT_MAX_NEW_TOKENS,
    MIN_EOS_PROBABILITY,
    SANITY_CHECK_DRIFT_THRESHOLD,
    SAMPLING_MIN_STEPS,
    SAMPLING_LOW_P,
    SAMPLING_HIGH_P,
    SAMPLING_BC_THRESHOLD,
    SAMPLING_MEDIAN_LOW_MAX,
)
from .monitoring import get_monitoring_manager

# Enable CUDA debugging for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


# Use the same logger as the main module
logger = logging.getLogger("grail")

# ──────────────────────────  CONFIGURATION  ─────────────────────────────
#  Constants are now imported from .shared.constants


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
        r_vec_from_randomness._cache: dict = {}
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
    if hasattr(r_vec_from_randomness, "_cache"):
        if cache_key in r_vec_from_randomness._cache:
            logger.debug(f"[SketchVec] Using cached vector for d_model={d_model}")
            return r_vec_from_randomness._cache[cache_key].clone()
    else:
        r_vec_from_randomness._cache = {}

    try:
        # Use PRF to expand drand randomness into d_model random integers
        # Using 4 bytes per integer for int32 range
        raw = prf(RNG_LABEL["sketch"], bytes.fromhex(clean_hex), out_bytes=4 * d_model)
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
    if len(r_vec_from_randomness._cache) < 100:
        r_vec_from_randomness._cache[cache_key] = tensor.clone()

    logger.debug(
        f"[SketchVec] Generated vector with shape={tensor.shape}, first 4 values: {tensor[:4].tolist()}"
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
        material = prf(RNG_LABEL["open"], tokens_hash, bytes.fromhex(clean_hex), out_bytes=32)
    except ValueError as e:
        raise ValueError(f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}")

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

# REMOVED: derive_secret_key_from_hotkey was insecure and has been removed
# Use wallet.hotkey.sign() for actual cryptographic signatures


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
    signature = wallet.hotkey.sign(s_vals_bytes)
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
        verified = keypair.verify(data=s_vals_bytes, signature=signature)
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


# ─────────────────────────────  PROVER  ────────────────────────────────


class Prover:
    def __init__(self, model_name: str = MODEL_NAME, wallet: bt.wallet = None) -> None:
        """
        Initialize Prover with model and Bittensor wallet for secure signatures.

        Args:
            model_name: Name of the model to load
            wallet: Bittensor wallet object (bt.wallet) for cryptographic signatures (required)

        Raises:
            ValueError: If wallet is not provided
        """
        if wallet is None:
            raise ValueError("Prover requires a bt.wallet for secure signatures")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad_token is properly set to prevent model confusion between padding and content tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )
        self.wallet = wallet
        logger.debug("Prover initialized with wallet for secure signatures")

    def commit(self, tokens: list[int], randomness_hex: str) -> dict:
        """Generate GRAIL commitment for tokens with monitoring."""
        monitor = get_monitoring_manager()
        """
        Generate GRAIL commitment for given tokens.

        This is the core GRAIL proof generation - it takes any tokens
        and produces a cryptographic commitment that proves these tokens
        were processed by this specific model.

        Args:
            tokens: List of token IDs to commit to
            randomness_hex: Hex string of randomness (typically from drand)

        Returns:
            Dictionary with GRAIL commitment (s_vals, signature, etc.)
        """
        # Validate tokens before processing
        if not tokens:
            logger.warning("Empty token list provided to commit")
            return {
                "beacon": {"round": 1, "randomness": randomness_hex},
                "tokens": [],
                "s_vals": [],
                "signature": b"".hex(),
            }

        # Check for invalid token IDs
        vocab_size = self.model.config.vocab_size
        invalid_tokens = [t for t in tokens if t < 0 or t >= vocab_size]
        if invalid_tokens:
            logger.error(
                f"Invalid token IDs found: {invalid_tokens[:10]}... (vocab_size={vocab_size})"
            )
            raise ValueError(
                f"Token IDs must be in range [0, {vocab_size}), found: {min(invalid_tokens)}-{max(invalid_tokens)}"
            )

        # Check sequence length
        max_length = self.model.config.max_position_embeddings
        if len(tokens) > max_length:
            logger.warning(
                f"Token sequence ({len(tokens)}) exceeds model max length ({max_length}), truncating"
            )
            tokens = tokens[:max_length]

        # Set up randomness for sketch computation
        self.beacon_R = {"round": 1, "randomness": randomness_hex}
        self.r_vec = r_vec_from_randomness(randomness_hex, self.model.config.hidden_size)

        # Compute sketch values from model hidden states
        s_vals = []
        try:
            with torch.no_grad():
                # Ensure correct dtype for token tensor
                token_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

                # Add shape logging for debugging
                logger.debug(
                    f"Token tensor shape: {token_tensor.shape}, dtype: {token_tensor.dtype}"
                )

                outputs = self.model(token_tensor, output_hidden_states=True)

                # Validate LAYER_INDEX
                num_layers = len(outputs.hidden_states)
                if LAYER_INDEX >= num_layers or LAYER_INDEX < -num_layers:
                    raise ValueError(
                        f"LAYER_INDEX {LAYER_INDEX} out of bounds for model with {num_layers} layers"
                    )

                h_layer = outputs.hidden_states[LAYER_INDEX][0]
                logger.debug(f"Hidden layer shape: {h_layer.shape}")

                for pos in range(len(tokens)):
                    if pos < h_layer.size(0):
                        s_val = dot_mod_q(h_layer[pos], self.r_vec)
                        s_vals.append(s_val)
                    else:
                        logger.warning(
                            f"Position {pos} exceeds hidden layer size {h_layer.size(0)}"
                        )

        except RuntimeError as e:
            logger.error(f"CUDA/Runtime error during model inference: {e}")
            logger.error(
                f"Tokens sample: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}"
            )
            logger.error(f"Token range: min={min(tokens)}, max={max(tokens)}")
            raise

        logger.debug(f"[Commit] Generated {len(s_vals)} sketch values for {len(tokens)} tokens")

        # Sign the s_vals for integrity
        signature = sign_s_vals(s_vals, self.wallet)

        # Store state for open() method
        self._state = {
            "tokens": tokens,
            "s_vals": s_vals,
            "seq_len": len(tokens),
            "signature": signature,
        }
        
        # Log GRAIL proof generation metrics
        if monitor:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(monitor.log_counter("grail.proofs_generated"))
                    asyncio.create_task(monitor.log_histogram("grail.token_count", len(tokens)))
                    asyncio.create_task(monitor.log_histogram("grail.s_vals_count", len(s_vals)))
            except RuntimeError:
                # No event loop running, skip monitoring
                pass

        return {
            "beacon": self.beacon_R,
            "tokens": tokens,
            "s_vals": s_vals,
            "signature": signature.hex(),
        }

    def commit_rollout(
        self, sat_problem: SATProblem, randomness_hex: str, difficulty: float = 0.5
    ) -> dict:
        """
        Generate SAT rollout and create GRAIL proof.

        This combines:
        1. SAT rollout generation (delegated to SATRolloutGenerator)
        2. GRAIL proof generation (commit method)

        Returns:
            Dictionary with both rollout data and GRAIL proof
        """
        # Use SATRolloutGenerator for environment-specific logic
        sat_generator = SATRolloutGenerator(self.model, self.tokenizer, self.device)
        rollout_data = sat_generator.generate_rollout(sat_problem)

        # Create GRAIL commitment for the generated tokens
        commit_data = self.commit(rollout_data["tokens"], randomness_hex)

        # Combine rollout and proof data
        return {
            "sat_problem": {
                "seed": sat_problem.seed,
                "num_vars": sat_problem.num_vars,
                "clauses": sat_problem.clauses,
                "difficulty": difficulty,
            },
            "rollout": {
                "trajectory": rollout_data["trajectory"],
                "total_reward": rollout_data["total_reward"],
                "success": rollout_data["success"],
                "satisfied_clauses": rollout_data["satisfied_clauses"],
                "assignment": rollout_data["assignment"],
            },
            "tokens": commit_data["tokens"],
            "s_vals": commit_data["s_vals"],
            "signature": commit_data["signature"],
            "beacon": commit_data["beacon"],
        }

    def open(self, randomness_hex: str, k: int = CHALLENGE_K) -> dict:
        # Use provided randomness instead of generating beacon
        beacon_R1 = {"round": 2, "randomness": randomness_hex}
        # Use tokens instead of s_vals for index derivation
        idxs = indices_from_root(self._state["tokens"], randomness_hex, self._state["seq_len"], k)
        return {"round_R1": beacon_R1, "indices": idxs}


# ─────────────────────────────  VERIFIER  ──────────────────────────────


class Verifier:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad_token is properly set to prevent model confusion between padding and content tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )

        # Cache from most recent proof verification forward pass
        self._last_tokens_hash: Optional[str] = None
        self._last_step_logits: Optional[torch.Tensor] = None

    def verify_rollout(self, commit: dict, proof_pkg: dict, prover_address: str) -> bool:
        """
        Verify SAT rollout with GRAIL proof.

        This verification ensures:
        1. The GRAIL proof is valid (sketch values match the model)
        2. The SAT problem matches deterministic generation from seed
        3. The solution (if successful) actually solves the problem

        The GRAIL proof cryptographically proves that:
        - The SAT problem text was processed by the claimed model
        - The solution trajectory was generated by that same model
        - The model that generated this cannot be substituted
        """
        monitor = get_monitoring_manager()
        # First verify the GRAIL proof - this proves the model identity
        if not self.verify(commit, proof_pkg, prover_address):
            logger.debug("GRAIL proof failed - model identity not verified")
            if monitor:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(monitor.log_counter("grail.verification_failures"))
                except RuntimeError:
                    pass
            return False

        # Verify SAT problem was generated correctly from seed
        sat_data = commit.get("sat_problem", {})
        if not sat_data:
            logger.debug("No SAT problem data in commit")
            return False

        # Regenerate SAT problem from seed and verify it matches
        # Use the difficulty from the commit data, defaulting to 0.5 if not present
        difficulty = sat_data.get("difficulty", 0.5)
        logger.debug(
            f"Regenerating SAT problem from seed '{sat_data['seed']}' with difficulty {difficulty}"
        )
        expected_problem = generate_sat_problem(sat_data["seed"], difficulty)
        if (
            expected_problem.num_vars != sat_data["num_vars"]
            or expected_problem.clauses != sat_data["clauses"]
        ):
            logger.debug("SAT problem doesn't match seed generation:")
            logger.debug(
                f"  Expected: {expected_problem.num_vars} vars, {len(expected_problem.clauses)} clauses"
            )
            logger.debug(f"  Got: {sat_data['num_vars']} vars, {len(sat_data['clauses'])} clauses")
            logger.debug(f"  Seed: {sat_data['seed']}, Difficulty: {difficulty}")
            return False

        # Enforce termination check before solution validation
        if not self._passes_termination_check(commit):
            logger.debug(
                "Termination check failed: sequence neither reached max context length "
                "nor ended with EOS token having probability >= 0.1"
            )
            return False

        # Token sampling distribution shape check
        ok_sampling, sampling_stats = self._token_sampling_shape_check(commit)
        if not ok_sampling:
            logger.debug(
                f"Token sampling shape check failed: {sampling_stats}"
            )
            return False
        logger.debug(f"Token sampling shape check passed: {sampling_stats}")

        # Verify the solution if claimed successful
        rollout = commit.get("rollout", {})
        if rollout.get("success", False):
            # Check that the assignment actually solves the problem
            assignment = rollout.get("assignment", [])
            if not expected_problem.check_solution(assignment):
                logger.debug("Claimed solution doesn't actually solve SAT problem")
                return False

        logger.debug("SAT rollout verification successful - model identity confirmed")
        if monitor:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(monitor.log_counter("grail.verification_successes"))
            except RuntimeError:
                pass
        return True

    def verify(self, commit: dict, proof_pkg: dict, prover_address: str) -> bool:
        """
        Verify just the GRAIL proof portion using public key cryptography.

        Args:
            commit: Commitment data with s_vals and signature
            proof_pkg: Proof package with revealed information
            prover_address: SS58 wallet address for public key verification

        Returns:
            True if proof is valid, False otherwise
        """
        # Verify s_vals signature for integrity
        signature = bytes.fromhex(commit["signature"])
        if not verify_s_vals_signature(commit["s_vals"], signature, prover_address):
            logger.debug("s_vals signature verification failed")
            return False

        # Re-derive sketch vector
        beacon = commit.get("beacon", commit.get("round_R", {}))
        r_vec = r_vec_from_randomness(beacon["randomness"], self.model.config.hidden_size)

        # Re-derive and compare indices using tokens (not s_vals)
        proof_beacon = proof_pkg.get("beacon", proof_pkg.get("round_R1", {}))
        idxs_exp = indices_from_root(
            commit["tokens"],
            proof_beacon["randomness"],
            len(commit["tokens"]),
            len(proof_pkg["indices"]),
        )
        if idxs_exp != proof_pkg["indices"]:
            logger.debug("Index-selection mismatch")
            return False

        # Recompute hidden states
        full_ids = torch.tensor(commit["tokens"], dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            outs = self.model(full_ids, output_hidden_states=True)
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
            committed_s_val = commit["s_vals"][i]

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
                np = None

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
                    m3 = float((d ** 3).mean())
                    m4 = float((d ** 4).mean())
                    skew = m3 / (s2 ** 1.5 + 1e-12)
                    kurt = m4 / (s2 ** 2 + 1e-12)
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
                    idx = max(0, min(sorted_tx.numel() - 1, int(0.10 * (sorted_tx.numel() - 1))))
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
                    skew = m3 / (s2 ** 1.5 + 1e-12)
                    kurt = m4 / (s2 ** 2 + 1e-12)

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
        allow_bc = metrics.get("q10", 1.0) <= SAMPLING_MEDIAN_LOW_MAX
        suspicious_bimodal = allow_bc and (metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD)
        suspicious = suspicious_unimodal_low or suspicious_bimodal

        # Verbose-mode monitoring: log histogram of chosen-token probabilities to help tune thresholds
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
                
            import logging as _logging
            root_level = _logging.getLogger().getEffectiveLevel()
            if root_level > _logging.DEBUG:
                return
                
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_event_loop()
                if not loop.is_running():
                    return
                    
                # Log the histogram using the more reliable approach
                _asyncio.create_task(
                    monitor.log_histogram("sampling_shape_check.hist", probs)
                )
                # Summary statistics under the sampling_shape_check prefix for easier navigation
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.mean", metrics["mean"])
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.median", metrics.get("median", 0.0))
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.q10", metrics.get("q10", 0.0))
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.low_frac", metrics["low_frac"])
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.high_frac", metrics["high_frac"])
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.mid_frac", metrics["mid_frac"])
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.bc", metrics["bc"])
                )
                _asyncio.create_task(
                    monitor.log_gauge("sampling_shape_check.n", metrics["n"])
                )
            except RuntimeError:
                pass
        except Exception:
            pass

    def _check_max_length_termination(self, commit: dict) -> bool:
        """Check if completion reached the configured generation max length."""
        try:
            expected_max_new = int(os.getenv("GRAIL_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
        except Exception:
            expected_max_new = DEFAULT_MAX_NEW_TOKENS

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

    def _get_step_logits(self, tokens: List[int]) -> Optional[torch.Tensor]:
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
            return outs.logits[0, outs.logits.size(1) - 2, :].detach().to("cpu")
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
