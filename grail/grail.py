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
from typing import List, Tuple, Dict, Optional

import bittensor as bt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .drand import get_beacon, get_drand_beacon, get_round_at_time
from .environments import SATProblem, generate_sat_problem, SATRolloutGenerator
from .rollout import RolloutGenerator

# Enable CUDA debugging for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# Use the same logger as the main module
logger = logging.getLogger("grail")

# ──────────────────────────  CONFIGURATION  ─────────────────────────────


PRIME_Q      = 2_147_483_647
CHALLENGE_K  = 16
TOLERANCE    = 3

MODEL_NAME   = "sshleifer/tiny-gpt2"
LAYER_INDEX  = -1
RNG_LABEL    = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}



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
        return b''
    
    if not isinstance(label, bytes):
        raise TypeError(f"label must be bytes, got {type(label).__name__}")
    for i, part in enumerate(parts):
        if not isinstance(part, bytes):
            raise TypeError(f"parts[{i}] must be bytes, got {type(part).__name__}")
    
    # Use SHAKE256 for variable-length output if available (more efficient)
    try:
        import hashlib
        if hasattr(hashlib, 'shake_256'):
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
        block_input = input_data + i.to_bytes(4, 'big')
        block_hash = hashlib.sha256(block_input).digest()
        output[i * hash_size:(i + 1) * hash_size] = block_hash
    
    return bytes(output[:out_bytes])

def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:
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
        clean_hex = '0' + clean_hex
    
    # Cache key for memoization (avoid recomputing for same inputs)
    cache_key = (clean_hex, d_model)
    
    # Check if we've already computed this (useful for repeated calls)
    if hasattr(r_vec_from_randomness, '_cache'):
        if cache_key in r_vec_from_randomness._cache:
            logger.debug(f"[SketchVec] Using cached vector for d_model={d_model}")
            return r_vec_from_randomness._cache[cache_key].clone()
    else:
        r_vec_from_randomness._cache = {}
    
    try:
        # Use PRF to expand drand randomness into d_model random integers
        # Using 4 bytes per integer for int32 range
        raw = prf(RNG_LABEL["sketch"], bytes.fromhex(clean_hex), out_bytes=4*d_model)
    except ValueError as e:
        raise ValueError(f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}") from e
    
    # Use numpy for more efficient unpacking if available
    try:
        import numpy as np
        # More efficient for large d_model
        ints_array = np.frombuffer(raw, dtype='>i4').astype(np.int32, copy=False)
        tensor = torch.from_numpy(ints_array.copy())  # copy to ensure ownership
    except ImportError as e:
        logger.error(f"Error unpacking ints_array: {e}")
        # Fallback to struct.unpack
        ints = struct.unpack(">" + "i"*d_model, raw)
        tensor = torch.tensor(ints, dtype=torch.int32)
    
    # Optionally normalize to unit variance (commented out to maintain compatibility)
    # tensor = tensor.float()
    # tensor = tensor / tensor.std()
    
    # Cache the result (limit cache size to prevent memory issues)
    if len(r_vec_from_randomness._cache) < 100:
        r_vec_from_randomness._cache[cache_key] = tensor.clone()
    
    logger.debug(f"[SketchVec] Generated vector with shape={tensor.shape}, first 4 values: {tensor[:4].tolist()}")
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
    tokens_bytes = b''.join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()
    
    # Normalize hex string more robustly
    clean_hex = rand_hex.strip().replace("0x", "").replace("0X", "")
    if not clean_hex:
        raise ValueError(f"Empty randomness hex string: '{rand_hex}'")
    
    # Validate hex string before conversion
    if len(clean_hex) % 2 != 0:
        clean_hex = '0' + clean_hex  # Pad with leading zero if odd length
    
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
    
    logger.debug(f"[Indices] selected {len(idxs)} indices from seq_len={seq_len}: {idxs[:5]}..." if len(idxs) > 5 else f"[Indices] selected {idxs}")
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
    if not hasattr(wallet, 'hotkey') or not hasattr(wallet.hotkey, 'sign'):
        raise TypeError(f"Wallet must be a bt.wallet with hotkey.sign() method, got {type(wallet)}")
    
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
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
    
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    
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
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    return hashlib.sha256(s_vals_bytes).digest()


# ─────────────────────────────  PROVER  ────────────────────────────────

class Prover:
    def __init__(self, model_name=MODEL_NAME, wallet: bt.wallet = None):
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
        
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad_token is properly set to prevent model confusion between padding and content tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )
        self.wallet = wallet
        logger.debug("Prover initialized with wallet for secure signatures")

    
    def commit(self, tokens: list[int], randomness_hex: str) -> dict:
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
                "signature": b"".hex()
            }
        
        # Check for invalid token IDs
        vocab_size = self.model.config.vocab_size
        invalid_tokens = [t for t in tokens if t < 0 or t >= vocab_size]
        if invalid_tokens:
            logger.error(f"Invalid token IDs found: {invalid_tokens[:10]}... (vocab_size={vocab_size})")
            raise ValueError(f"Token IDs must be in range [0, {vocab_size}), found: {min(invalid_tokens)}-{max(invalid_tokens)}")
        
        # Check sequence length
        max_length = self.model.config.max_position_embeddings
        if len(tokens) > max_length:
            logger.warning(f"Token sequence ({len(tokens)}) exceeds model max length ({max_length}), truncating")
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
                logger.debug(f"Token tensor shape: {token_tensor.shape}, dtype: {token_tensor.dtype}")
                
                outputs = self.model(token_tensor, output_hidden_states=True)
                
                # Validate LAYER_INDEX
                num_layers = len(outputs.hidden_states)
                if LAYER_INDEX >= num_layers or LAYER_INDEX < -num_layers:
                    raise ValueError(f"LAYER_INDEX {LAYER_INDEX} out of bounds for model with {num_layers} layers")
                
                h_layer = outputs.hidden_states[LAYER_INDEX][0]
                logger.debug(f"Hidden layer shape: {h_layer.shape}")
                
                for pos in range(len(tokens)):
                    if pos < h_layer.size(0):
                        s_val = dot_mod_q(h_layer[pos], self.r_vec)
                        s_vals.append(s_val)
                    else:
                        logger.warning(f"Position {pos} exceeds hidden layer size {h_layer.size(0)}")
                        
        except RuntimeError as e:
            logger.error(f"CUDA/Runtime error during model inference: {e}")
            logger.error(f"Tokens sample: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}")
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
            "signature": signature
        }
        
        return {
            "beacon": self.beacon_R,
            "tokens": tokens,
            "s_vals": s_vals,
            "signature": signature.hex()
        }
    
    def commit_rollout(self, sat_problem: SATProblem, randomness_hex: str, difficulty: float = 0.5) -> dict:
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
                "difficulty": difficulty
            },
            "rollout": {
                "trajectory": rollout_data["trajectory"],
                "total_reward": rollout_data["total_reward"],
                "success": rollout_data["success"],
                "satisfied_clauses": rollout_data["satisfied_clauses"],
                "assignment": rollout_data["assignment"]
            },
            "tokens": commit_data["tokens"],
            "s_vals": commit_data["s_vals"],
            "signature": commit_data["signature"],
            "beacon": commit_data["beacon"]
        }

    def open(self, randomness_hex: str, k: int = CHALLENGE_K) -> dict:
        # Use provided randomness instead of generating beacon
        beacon_R1 = {"round": 2, "randomness": randomness_hex}
        # Use tokens instead of s_vals for index derivation
        idxs = indices_from_root(self._state["tokens"],
                                randomness_hex,
                                self._state["seq_len"],
                                k)
        return {"round_R1": beacon_R1, "indices": idxs}

# ─────────────────────────────  VERIFIER  ──────────────────────────────

class Verifier:
    def __init__(self, model_name=MODEL_NAME):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad_token is properly set to prevent model confusion between padding and content tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )

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
        # First verify the GRAIL proof - this proves the model identity
        if not self.verify(commit, proof_pkg, prover_address):
            logger.debug("GRAIL proof failed - model identity not verified")
            return False
        
        # Verify SAT problem was generated correctly from seed
        sat_data = commit.get("sat_problem", {})
        if not sat_data:
            logger.debug("No SAT problem data in commit")
            return False
        
        # Regenerate SAT problem from seed and verify it matches
        # Use the difficulty from the commit data, defaulting to 0.5 if not present
        difficulty = sat_data.get("difficulty", 0.5)
        logger.debug(f"Regenerating SAT problem from seed '{sat_data['seed']}' with difficulty {difficulty}")
        expected_problem = generate_sat_problem(sat_data["seed"], difficulty)
        if expected_problem.num_vars != sat_data["num_vars"] or \
           expected_problem.clauses != sat_data["clauses"]:
            logger.debug(f"SAT problem doesn't match seed generation:")
            logger.debug(f"  Expected: {expected_problem.num_vars} vars, {len(expected_problem.clauses)} clauses")
            logger.debug(f"  Got: {sat_data['num_vars']} vars, {len(sat_data['clauses'])} clauses")
            logger.debug(f"  Seed: {sat_data['seed']}, Difficulty: {difficulty}")
            return False
        
        # The GRAIL proof has already verified that these tokens (including the SAT problem)
        # were processed by the claimed model - no other model could produce matching sketches
        
        # Verify the solution if claimed successful
        rollout = commit.get("rollout", {})
        if rollout.get("success", False):
            # Check that the assignment actually solves the problem
            assignment = rollout.get("assignment", [])
            if not expected_problem.check_solution(assignment):
                logger.debug("Claimed solution doesn't actually solve SAT problem")
                return False
        
        logger.debug("SAT rollout verification successful - model identity confirmed")
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
        r_vec = r_vec_from_randomness(
            beacon["randomness"],
            self.model.config.hidden_size
        )

        # Re-derive and compare indices using tokens (not s_vals)
        proof_beacon = proof_pkg.get("beacon", proof_pkg.get("round_R1", {}))
        idxs_exp = indices_from_root(
            commit["tokens"],
            proof_beacon["randomness"],
            len(commit["tokens"]),
            len(proof_pkg["indices"])
        )
        if idxs_exp != proof_pkg["indices"]:
            logger.debug("Index-selection mismatch")
            return False

        # Recompute hidden states
        full_ids = torch.tensor(commit["tokens"], dtype=torch.long,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            outs = self.model(full_ids, output_hidden_states=True)
        h_layer = outs.hidden_states[LAYER_INDEX][0]

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
                logger.debug(f"Sketch mismatch at index {i} ({local} vs {committed_s_val}, diff={mod_diff})")
                return False

        logger.debug("GRAIL proof verification successful")
        return True
    