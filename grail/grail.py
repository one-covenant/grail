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
from typing import List, Tuple, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# SAT Problem Configuration
MIN_VARS = 3
MAX_VARS = 20
MIN_CLAUSES = 5
MAX_CLAUSES = 50
CLAUSE_LENGTH = 3  # 3-SAT problems

# ────────────────────  MOCK BEACON HELPERS  ──────────────────────────────

def get_beacon(round_id: str = "latest") -> dict:
    global BEACON_COUNTER
    BEACON_COUNTER += 1
    rnd = os.urandom(32).hex()
    logger.debug(f"[Beacon] round={BEACON_COUNTER}, randomness={rnd[:8]}…")
    return {"round": BEACON_COUNTER, "randomness": rnd}

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

# ─────────────────────────  SAT PROBLEM GENERATION  ────────────────────────

class SATProblem:
    """Represents a SAT problem instance."""
    
    def __init__(self, num_vars: int, clauses: List[List[int]], seed: str):
        self.num_vars = num_vars
        self.clauses = clauses
        self.seed = seed
        self.solution = None
        
    def to_text(self) -> str:
        """Convert SAT problem to text format for LLM processing."""
        text = f"SAT Problem (seed: {self.seed[:8]}...):\n"
        text += f"Variables: {self.num_vars}\n"
        text += "Clauses:\n"
        for i, clause in enumerate(self.clauses):
            clause_str = " OR ".join([f"{'NOT ' if lit < 0 else ''}x{abs(lit)}" for lit in clause])
            text += f"  ({clause_str})\n"
        return text
    
    def check_solution(self, assignment: List[bool]) -> bool:
        """Check if assignment satisfies all clauses."""
        if len(assignment) != self.num_vars:
            return False
        
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if lit > 0 and assignment[var_idx]:
                    satisfied = True
                    break
                elif lit < 0 and not assignment[var_idx]:
                    satisfied = True
                    break
            if not satisfied:
                return False
        return True

def generate_sat_problem(seed: str, difficulty: float = 0.5) -> SATProblem:
    """Generate a SAT problem from seed with controlled difficulty."""
    # Use seed for deterministic generation
    rng = random.Random(hashlib.sha256(seed.encode()).digest())
    
    # Scale problem size based on difficulty
    num_vars = rng.randint(
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty * 0.5),
        MIN_VARS + int((MAX_VARS - MIN_VARS) * difficulty)
    )
    num_clauses = rng.randint(
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty * 0.5),
        MIN_CLAUSES + int((MAX_CLAUSES - MIN_CLAUSES) * difficulty)
    )
    
    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = rng.sample(range(1, num_vars + 1), min(CLAUSE_LENGTH, num_vars))
        for var in vars_in_clause:
            # Randomly negate
            if rng.random() < 0.5:
                clause.append(-var)
            else:
                clause.append(var)
        clauses.append(clause)
    
    return SATProblem(num_vars, clauses, seed)

# ─────────────────────────  RL ENVIRONMENT  ────────────────────────────

class SATEnvironment:
    """RL environment for solving SAT problems."""
    
    def __init__(self, problem: SATProblem):
        self.problem = problem
        self.assignment = [False] * problem.num_vars
        self.current_var = 0
        self.steps = 0
        self.max_steps = problem.num_vars * 2
        self.trajectory = []
        
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.assignment = [False] * self.problem.num_vars
        self.current_var = 0
        self.steps = 0
        self.trajectory = []
        return self.get_state()
    
    def get_state(self) -> Dict:
        """Get current state as dict for LLM processing."""
        return {
            "problem": self.problem.to_text(),
            "current_assignment": self.assignment.copy(),
            "current_var": self.current_var,
            "steps": self.steps,
            "satisfied_clauses": self.count_satisfied_clauses()
        }
    
    def count_satisfied_clauses(self) -> int:
        """Count how many clauses are currently satisfied."""
        count = 0
        for clause in self.problem.clauses:
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx < len(self.assignment):
                    if (lit > 0 and self.assignment[var_idx]) or \
                       (lit < 0 and not self.assignment[var_idx]):
                        count += 1
                        break
        return count
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Take action (0=false, 1=true for current variable)."""
        if self.current_var >= self.problem.num_vars:
            return self.get_state(), 0, True, {"success": False}
        
        # Record action in trajectory
        self.trajectory.append((self.current_var, action))
        
        # Apply action
        self.assignment[self.current_var] = bool(action)
        self.current_var += 1
        self.steps += 1
        
        # Calculate reward
        satisfied = self.count_satisfied_clauses()
        total_clauses = len(self.problem.clauses)
        
        # Check if done
        done = False
        success = False
        
        if self.current_var >= self.problem.num_vars:
            # All variables assigned
            done = True
            success = self.problem.check_solution(self.assignment)
            if success:
                reward = 10.0  # Big reward for solving
            else:
                reward = satisfied / total_clauses - 1.0  # Partial credit
        elif self.steps >= self.max_steps:
            done = True
            reward = -1.0  # Penalty for timeout
        else:
            # Intermediate reward based on progress
            reward = (satisfied / total_clauses) * 0.1
        
        info = {
            "success": success,
            "satisfied_clauses": satisfied,
            "total_clauses": total_clauses
        }
        
        return self.get_state(), reward, done, info
    
    def render_trajectory(self) -> str:
        """Render trajectory as text for LLM."""
        text = "Solution trajectory:\n"
        for var, val in self.trajectory:
            text += f"  x{var+1} = {val}\n"
        return text

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

    def commit_rollout(self, sat_problem: SATProblem, randomness_hex: str) -> dict:
        """Generate RL rollout for SAT problem with GRAIL proof."""
        # Use provided randomness
        self.beacon_R = {"round": 1, "randomness": randomness_hex}
        self.r_vec = r_vec_from_randomness(randomness_hex, self.model.config.hidden_size)
        
        # Initialize environment
        env = SATEnvironment(sat_problem)
        state = env.reset()
        
        # Convert SAT problem to prompt for LLM
        prompt = f"Solve this SAT problem step by step:\n{sat_problem.to_text()}\nSolution:"
        
        # Generate solution using LLM
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids,
                max_new_tokens=sat_problem.num_vars * 10,  # Enough tokens for solution
                use_cache=True,
                return_dict_in_generate=True
            )
        tokens = gen.sequences[0].tolist()
        
        # Execute rollout based on generated tokens
        trajectory = []
        total_reward = 0
        done = False
        
        # Parse actions from generated text (simplified - in practice would be more sophisticated)
        generated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        while not done:
            # Use model to decide action (simplified - extract from generated text)
            action = 0 if env.current_var % 2 == 0 else 1  # Placeholder logic
            
            # In a real implementation, we'd parse the LLM output to get actions
            # For now, using a simple heuristic
            if "true" in generated_text.lower() and f"x{env.current_var+1}" in generated_text:
                action = 1
            
            state, reward, done, info = env.step(action)
            trajectory.append((env.current_var - 1, action, reward))
            total_reward += reward
        
        # Compute sketch values for proof
        s_vals = []
        hiddens = []
        
        # Get hidden states for the generated sequence
        with torch.no_grad():
            outputs = self.model(torch.tensor([tokens]).to(self.device), output_hidden_states=True)
            all_hidden = outputs.hidden_states
            
            for pos in range(len(tokens)):
                if pos < len(all_hidden[LAYER_INDEX][0]):
                    h = all_hidden[LAYER_INDEX][0, pos, :]
                    hiddens.append(h)
                    s_val = dot_mod_q(h, self.r_vec)
                    s_vals.append(s_val)
        
        self.s_vals = s_vals
        self.hiddens = hiddens
        self.tokens = tokens
        
        # Sign the s_vals for integrity
        signature = sign_s_vals(s_vals, self.secret_key)
        
        return {
            "sat_problem": {
                "seed": sat_problem.seed,
                "num_vars": sat_problem.num_vars,
                "clauses": sat_problem.clauses
            },
            "rollout": {
                "trajectory": trajectory,
                "total_reward": total_reward,
                "success": info.get("success", False),
                "satisfied_clauses": info.get("satisfied_clauses", 0),
                "assignment": env.assignment
            },
            "tokens": tokens,
            "s_vals": s_vals,
            "signature": signature.hex(),
            "beacon": self.beacon_R
        }
    
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
        """Verify SAT rollout with GRAIL proof."""
        # First verify the GRAIL proof
        if not self.verify_grail_proof(commit, proof_pkg, prover_secret_key):
            return False
        
        # Verify SAT problem was generated correctly from seed
        sat_data = commit.get("sat_problem", {})
        if not sat_data:
            logger.debug("No SAT problem data in commit")
            return False
        
        # Regenerate SAT problem from seed and verify it matches
        expected_problem = generate_sat_problem(sat_data["seed"])
        if expected_problem.num_vars != sat_data["num_vars"] or \
           expected_problem.clauses != sat_data["clauses"]:
            logger.debug("SAT problem doesn't match seed generation")
            return False
        
        # Verify the solution if claimed successful
        rollout = commit.get("rollout", {})
        if rollout.get("success", False):
            # Check that the assignment actually solves the problem
            assignment = rollout.get("assignment", [])
            if not expected_problem.check_solution(assignment):
                logger.debug("Claimed solution doesn't actually solve SAT problem")
                return False
        
        logger.debug("SAT rollout verification successful")
        return True
    
    def verify_grail_proof(self, commit: dict, proof_pkg: dict, prover_secret_key: bytes) -> bool:
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
    
    def verify(self, commit: dict, proof_pkg: dict, prover_secret_key: bytes) -> bool:
        # Verify s_vals signature for integrity
        signature = bytes.fromhex(commit["signature"])
        if not verify_s_vals_signature(commit["s_vals"], signature, prover_secret_key):
            logger.debug("s_vals signature verification failed")
            return False

        # Re-derive sketch vector
        r_vec = r_vec_from_randomness(
            commit["round_R"]["randomness"],
            self.model.config.hidden_size
        )

        # Re-derive and compare indices using tokens (not s_vals)
        idxs_exp = indices_from_root(
            commit["tokens"],
            proof_pkg["round_R1"]["randomness"],
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

        logger.debug("Verification successful")
        return True