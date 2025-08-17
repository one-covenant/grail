#!/usr/bin/env python3
"""
GRAIL – Guaranteed Rollout Authenticity via Inference Ledger
Modified for SAT problem generation and RL rollouts
"""

import io
import os
import hmac
import time
import torch
import random
import struct
import logging
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable CUDA debugging for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from .environments import SATProblem, SATEnvironment, generate_sat_problem, SATRolloutGenerator
from .rollout import RolloutGenerator
from .drand import get_drand_beacon, get_beacon, get_round_at_time

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
    h = hashlib.sha256(label + b"||" + b"||".join(parts)).digest()
    while len(h) < out_bytes:
        h += hashlib.sha256(h).digest()
    return h[:out_bytes]

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
        Random projection vector of shape (d_model,)
    """
    # Remove 0x prefix if present and ensure we have valid hex
    clean_hex = rand_hex.replace("0x", "").replace("0X", "")
    try:
        # Use PRF to expand drand randomness into d_model random integers
        raw = prf(RNG_LABEL["sketch"], bytes.fromhex(clean_hex), out_bytes=4*d_model)
    except ValueError as e:
        raise ValueError(f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}") from e
    ints = struct.unpack(">" + "i"*d_model, raw)
    logger.debug(f"[SketchVec] first 4 ints: {ints[:4]}")
    return torch.tensor(ints, dtype=torch.int32)

def indices_from_root(tokens: list[int], rand_hex: str, seq_len: int, k: int) -> list[int]:
    # Use tokens hash instead of s_vals hash for index derivation
    # This ensures indices remain stable even when s_vals change within tolerance
    tokens_bytes = b''.join(int_to_bytes(token) for token in tokens)
    tokens_hash = hashlib.sha256(tokens_bytes).digest()
    # Remove 0x prefix if present and ensure we have valid hex
    clean_hex = rand_hex.replace("0x", "").replace("0X", "")
    try:
        material = prf(RNG_LABEL["open"], tokens_hash, bytes.fromhex(clean_hex), out_bytes=32)
    except ValueError as e:
        raise ValueError(f"Invalid hex string for randomness: '{rand_hex}' -> '{clean_hex}': {e}")
    rnd = random.Random(material)
    idxs = sorted(rnd.sample(range(seq_len), k))
    logger.debug(f"[Indices] selected {idxs}")
    return idxs

# ─────────────────────────────  UTILITIES  ─────────────────────────────

def derive_secret_key_from_hotkey(hotkey_address: str) -> bytes:
    """
    Derive a deterministic secret key from a hotkey address.
    
    This ensures that:
    1. Each miner has a unique secret key based on their hotkey
    2. The validator can derive the same key to verify signatures
    3. The key is deterministic and reproducible
    
    Args:
        hotkey_address: The miner's hotkey address (e.g., ss58 format)
    
    Returns:
        32-byte secret key for HMAC signing
    """
    return hashlib.sha256(f"grail_secret_{hotkey_address}".encode()).digest()

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

def sign_s_vals(s_vals: list[int], secret_key: bytes) -> bytes:
    """Sign the s_vals list for integrity protection."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    signature = hmac.new(secret_key, s_vals_bytes, hashlib.sha256).digest()
    logger.debug(f"[Signature] signed {len(s_vals)} s_vals")
    return signature

def verify_s_vals_signature(s_vals: list[int], signature: bytes, secret_key: bytes) -> bool:
    """Verify the signature of s_vals list."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    expected_sig = hmac.new(secret_key, s_vals_bytes, hashlib.sha256).digest()
    return hmac.compare_digest(signature, expected_sig)

def hash_s_vals(s_vals: list[int]) -> bytes:
    """Compute hash of s_vals for integrity checking."""
    s_vals_bytes = b''.join(int_to_bytes(val) for val in s_vals)
    return hashlib.sha256(s_vals_bytes).digest()


# ─────────────────────────────  PROVER  ────────────────────────────────

class Prover:
    def __init__(self, model_name=MODEL_NAME, secret_key: Optional[bytes] = None):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )
        # Secret key for signing s_vals - MUST be set deterministically before use
        # In production, this is derived from the miner's hotkey address
        if secret_key is not None:
            self.secret_key = secret_key
        else:
            # This should never be used - the miner must set a deterministic key
            raise ValueError("Prover requires a deterministic secret_key. Use derive_secret_key(hotkey_address)")

    
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
        signature = sign_s_vals(s_vals, self.secret_key)
        
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
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name, use_safetensors=True)
            .to(self.device)
            .eval()
        )

    def verify_rollout(self, commit: dict, proof_pkg: dict, prover_secret_key: bytes) -> bool:
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
        if not self.verify(commit, proof_pkg, prover_secret_key):
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
    
    def verify(self, commit: dict, proof_pkg: dict, prover_secret_key: bytes) -> bool:
        """Verify just the GRAIL proof portion."""
        # Verify s_vals signature for integrity
        signature = bytes.fromhex(commit["signature"])
        if not verify_s_vals_signature(commit["s_vals"], signature, prover_secret_key):
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
    