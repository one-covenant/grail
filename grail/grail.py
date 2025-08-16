#!/usr/bin/env python3
"""
GRAIL – Guaranteed Rollout Authenticity via Inference Ledger
Modified for SAT problem generation and RL rollouts
"""

import os
import io
import struct
import hashlib
import random
import hmac
import torch
import numpy as np
import logging
import requests
import time
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import SAT environment classes
from .environments import SATProblem, SATEnvironment, generate_sat_problem

# Use the same logger as the main module
logger = logging.getLogger("grail")

# ──────────────────────────  CONFIGURATION  ─────────────────────────────

BEACON_COUNTER = 0

PRIME_Q      = 2_147_483_647
CHALLENGE_K  = 16
TOLERANCE    = 3

MODEL_NAME   = "sshleifer/tiny-gpt2"
LAYER_INDEX  = -1
RNG_LABEL    = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}


# ────────────────────  DRAND BEACON HELPERS  ──────────────────────────────

# Drand configuration - using League of Entropy mainnet
DRAND_URLS = [
    "https://api.drand.sh",
    "https://drand.cloudflare.com",
    "https://api.drand.secureweb3.com:6875"
]
DRAND_CHAIN_HASH = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"  # quicknet chain
DRAND_GENESIS_TIME = 1692803367  # quicknet genesis
DRAND_PERIOD = 3  # 3 seconds per round for quicknet

def get_drand_beacon(round_id: Optional[int] = None, use_fallback: bool = True) -> dict:
    """
    Fetch randomness from drand network.
    
    Args:
        round_id: Specific round to fetch, or None for latest
        use_fallback: If True, falls back to mock beacon on failure
    
    Returns:
        Dictionary with 'round' and 'randomness' keys
    """
    endpoint = f"/{DRAND_CHAIN_HASH}/public/{'latest' if round_id is None else round_id}"
    
    # Try each drand URL
    for url in DRAND_URLS:
        try:
            full_url = f"{url}{endpoint}"
            logger.debug(f"[Drand] Fetching from {full_url}")
            response = requests.get(full_url, timeout=10)  # Increased timeout
            if response.status_code == 200:
                data = response.json()
                logger.info(f"[Drand] Success! round={data['round']}, randomness={data['randomness'][:8]}…")
                return {
                    "round": data["round"],
                    "randomness": data["randomness"],
                    "signature": data.get("signature", ""),
                    "previous_signature": data.get("previous_signature", "")
                }
            else:
                logger.debug(f"[Drand] Got status {response.status_code} from {url}")
        except Exception as e:
            logger.debug(f"[Drand] Failed to fetch from {url}: {e}")
            continue
    
    # Fallback to mock beacon if all URLs fail
    if use_fallback:
        logger.warning("[Drand] All URLs failed, using mock beacon")
        return get_mock_beacon()
    else:
        raise Exception("Failed to fetch from any drand URL")

def get_mock_beacon() -> dict:
    """Fallback mock beacon for testing/development."""
    global BEACON_COUNTER
    BEACON_COUNTER += 1
    rnd = os.urandom(32).hex()
    logger.debug(f"[MockBeacon] round={BEACON_COUNTER}, randomness={rnd[:8]}…")
    return {"round": BEACON_COUNTER, "randomness": rnd}

def get_beacon(round_id: str = "latest", use_drand: bool = True) -> dict:
    """
    Get randomness beacon (drand by default, with fallback).
    
    Args:
        round_id: "latest" or specific round number
        use_drand: If True, use drand network; if False, use mock
    """
    if not use_drand:
        return get_mock_beacon()
    
    try:
        if round_id == "latest":
            return get_drand_beacon(None)
        else:
            return get_drand_beacon(int(round_id))
    except:
        # Fallback to mock on any error
        return get_mock_beacon()

def get_round_at_time(timestamp: int) -> int:
    """Calculate drand round number for a given timestamp."""
    elapsed = timestamp - DRAND_GENESIS_TIME
    return (elapsed // DRAND_PERIOD) + 1

def verify_drand_signature(beacon: dict) -> bool:
    """
    Verify drand beacon signature (requires additional crypto libraries).
    For now, returns True - implement BLS verification if needed.
    """
    # TODO: Implement BLS signature verification
    # This requires py_ecc or similar library for BLS12-381
    return True

def prf(label: bytes, *parts: bytes, out_bytes: int) -> bytes:
    h = hashlib.sha256(label + b"||" + b"||".join(parts)).digest()
    while len(h) < out_bytes:
        h += hashlib.sha256(h).digest()
    return h[:out_bytes]

def r_vec_from_randomness(rand_hex: str, d_model: int) -> torch.Tensor:
    # Remove 0x prefix if present and ensure we have valid hex
    clean_hex = rand_hex.replace("0x", "").replace("0X", "")
    try:
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
    def __init__(self, model_name=MODEL_NAME):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = (
            AutoModelForCausalLM
            .from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        # Generate a secret key for signing (in practice this would be securely managed)
        self.secret_key = os.urandom(32)

    def commit_rollout(self, sat_problem: SATProblem, randomness_hex: str, difficulty: float = 0.5) -> dict:
        """
        Generate RL rollout for SAT problem with GRAIL proof.
        
        The GRAIL proof ensures:
        1. The SAT problem text was processed by this specific model
        2. The solution trajectory was generated by this model's decisions
        3. All tokens (problem + solution) produce correct sketch values
        
        This cryptographically proves the rollout came from the claimed model.
        """
        # Use provided randomness
        self.beacon_R = {"round": 1, "randomness": randomness_hex}
        self.r_vec = r_vec_from_randomness(randomness_hex, self.model.config.hidden_size)
        
        # Initialize environment
        env = SATEnvironment(sat_problem)
        state = env.reset()
        
        # Execute rollout based on LLM decisions
        trajectory = []
        total_reward = 0
        done = False
        all_tokens = []
        
        while not done:
            # Create prompt for current state
            prompt = self._create_state_prompt(sat_problem, env, trajectory)
            
            # Get LLM's decision for current variable
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                try:
                    gen = self.model.generate(
                        input_ids,
                        max_new_tokens=20,  # Short response for single decision
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        top_p=0.95,  # Nucleus sampling for stability
                        top_k=50,    # Limit vocabulary for stability
                        repetition_penalty=1.1,  # Avoid repetition
                        bad_words_ids=None,  # No bad words
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError as e:
                    if "inf" in str(e) or "nan" in str(e):
                        # Fallback to greedy decoding if sampling fails
                        logger.debug(f"Sampling failed, using greedy decoding: {e}")
                        gen = self.model.generate(
                            input_ids,
                            max_new_tokens=20,
                            do_sample=False,  # Greedy decoding
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        raise
            
            # Parse action from generated text
            generated_text = self.tokenizer.decode(gen[0][len(input_ids[0]):], skip_special_tokens=True)
            action = self._parse_action(generated_text, env.current_var)
            
            # Store tokens for GRAIL proof
            all_tokens.extend(gen[0].tolist())
            
            # Take action in environment
            state, reward, done, info = env.step(action)
            trajectory.append((env.current_var - 1, action, reward))
            total_reward += reward
        
        # Compute sketch values for proof
        s_vals = []
        hiddens = []
        
        # Get hidden states for all generated tokens
        if all_tokens:
            with torch.no_grad():
                outputs = self.model(torch.tensor([all_tokens]).to(self.device), output_hidden_states=True)
                all_hidden = outputs.hidden_states
                
                for pos in range(len(all_tokens)):
                    if pos < len(all_hidden[LAYER_INDEX][0]):
                        h = all_hidden[LAYER_INDEX][0, pos, :]
                        hiddens.append(h)
                        s_val = dot_mod_q(h, self.r_vec)
                        s_vals.append(s_val)
        
        self.s_vals = s_vals
        self.hiddens = hiddens
        self.tokens = all_tokens
        
        # Sign the s_vals for integrity
        signature = sign_s_vals(s_vals, self.secret_key)
        
        # Store state for open() method
        self._state = {
            "tokens": all_tokens,
            "s_vals": s_vals,
            "seq_len": len(all_tokens),
            "signature": signature
        }
        
        return {
            "sat_problem": {
                "seed": sat_problem.seed,
                "num_vars": sat_problem.num_vars,
                "clauses": sat_problem.clauses,
                "difficulty": difficulty
            },
            "rollout": {
                "trajectory": trajectory,
                "total_reward": total_reward,
                "success": info.get("success", False),
                "satisfied_clauses": info.get("satisfied_clauses", 0),
                "assignment": env.assignment
            },
            "tokens": all_tokens,
            "s_vals": s_vals,
            "signature": signature.hex(),
            "beacon": self.beacon_R
        }
    
    def _create_state_prompt(self, sat_problem: SATProblem, env: SATEnvironment, trajectory: list) -> str:
        """Create a prompt for the LLM to decide the next variable assignment."""
        prompt = f"SAT Problem:\n{sat_problem.to_text()}\n"
        
        if trajectory:
            prompt += "\nAssignments so far:\n"
            for var, action, _ in trajectory:
                prompt += f"  x{var+1} = {action}\n"
        
        prompt += f"\nCurrent state: {env.count_satisfied_clauses()}/{len(sat_problem.clauses)} clauses satisfied\n"
        prompt += f"Next variable: x{env.current_var+1}\n"
        prompt += "Should x{} be 0 (false) or 1 (true)? Consider which value satisfies more clauses.\n".format(env.current_var+1)
        prompt += "Answer with just '0' or '1':"
        
        return prompt
    
    def _parse_action(self, text: str, var_idx: int) -> int:
        """Parse the action from LLM output."""
        text = text.strip().lower()
        
        # Look for explicit 0 or 1
        if '1' in text or 'true' in text:
            return 1
        elif '0' in text or 'false' in text:
            return 0
        
        # Check for yes/no style answers
        if 'yes' in text:
            return 1
        elif 'no' in text:
            return 0
        
        # Default to trying true first (can be randomized)
        return 1 if var_idx % 2 == 0 else 0
    
    def commit(self, prompt: str, randomness_hex: str, max_new_tokens: int = 32) -> dict:
        """Original commit method for backward compatibility."""
        # Use provided randomness instead of generating beacon
        self.beacon_R = {"round": 1, "randomness": randomness_hex}
        self.r_vec    = r_vec_from_randomness(randomness_hex,
                                              self.model.config.hidden_size)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True
            )
        tokens = gen.sequences[0].tolist()
        logger.debug(f"[Commit] tokens length = {len(tokens)}")

        with torch.no_grad():
            outs = self.model(gen.sequences, output_hidden_states=True)
        h_layer = outs.hidden_states[LAYER_INDEX][0]

        s_vals = [dot_mod_q(h_layer[t], self.r_vec) for t in range(h_layer.size(0))]
        logger.debug(f"[Commit] first 8 s_vals = {s_vals[:8]}")

        # Sign the s_vals for integrity
        signature = sign_s_vals(s_vals, self.secret_key)

        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        model_hash = hashlib.sha256(buf.getvalue()).hexdigest()

        self._state = {
            "tokens":  tokens,
            "s_vals":  s_vals,
            "seq_len": len(tokens),
            "signature": signature
        }

        return {
            "round_R":     self.beacon_R,
            "tokens":      tokens,
            "s_vals":      s_vals,
            "signature":   signature.hex(),
            "model_hash":  model_hash,
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
            .from_pretrained(model_name)
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
    