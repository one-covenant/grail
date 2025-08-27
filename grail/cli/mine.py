#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import typer
import random
import asyncio
import logging
import hashlib
import traceback
import math
import bittensor as bt
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file, save, load
from safetensors import safe_open
# TODO(v2): Re-enable training imports
# from trl import PPOTrainer, PPOConfig
# TODO(v2): Re-enable for training
# from accelerate import Accelerator

__version__ = "0.0.0"

from ..grail import Prover, Verifier
from . import console
from ..drand import get_drand_beacon, get_round_at_time
from ..environments import (
    # New reward system
    Parser, RewardVector, SATParser, 
    create_sat_reward_vector,
    # Existing classes
    SATProblem, generate_sat_problem, SATRolloutGenerator
)
from ..rollout import RolloutGenerator
from ..comms import (
    upload_file_chunked, download_file_chunked, file_exists, list_bucket_files,
    get_file, sink_window_inferences, 
    # TODO(v2): Re-enable model state management for training
    # save_model_state, load_model_state, model_state_exists,
    upload_valid_rollouts, get_valid_rollouts,
    # NEW: Hugging Face dataset upload
    upload_to_huggingface, download_from_huggingface, login_huggingface, PROTOCOL_VERSION
)

__all__ = [
    # Core classes
    "Prover", "Verifier", 
    # New reward system
    "Parser", "RewardVector", "SATParser",
    "create_sat_reward_vector",
    # Existing SAT classes
    "SATProblem", "generate_sat_problem", "SATRolloutGenerator",
    # Entry points
    "main"
]

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
NETUID = 81
WINDOW_LENGTH = 20  # Generate inferences every 20 blocks (increased for model downloads)
TRACE  = 5
logging.addLevelName(TRACE, "TRACE")

# Model configuration
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama 1B model

# --------------------------------------------------------------------------- #
#                               Logging                                       #
# --------------------------------------------------------------------------- #
def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)
logging.Logger.trace = _trace
logger = logging.getLogger("grail")

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>")
        raise typer.Exit(code=1)
    return v or default

# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None
async def get_subtensor():
    global SUBTENSOR
    if SUBTENSOR == None:
        logger.trace("Making Bittensor connection...")
        SUBTENSOR = bt.async_subtensor()
        await SUBTENSOR.initialize()
        logger.trace("Connected")
    return SUBTENSOR


# S3/R2 communication functions are now imported from comms.py

# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #
def generate_prompt(hotkey_address: str, block_hash: str, nonce: int) -> str:
    """Generate prompt in the required format"""
    return f"Hey my name is {hotkey_address} it is currently {block_hash} days since friday and my fav number is {nonce}, tell me a story about these three facts"

def parse_filename(filename: str) -> Tuple[str, int, int]:
    """Parse filename to extract wallet, block, nonce"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    parts = basename.split('-')
    if len(parts) >= 3:
        wallet = parts[0]
        block = int(parts[1])
        nonce = int(parts[2])
        return wallet, block, nonce
    return None, None, None

def parse_window_filename(filename: str) -> Tuple[str, int]:
    """Parse window filename to extract wallet and window_start"""
    # Remove prefix and extension
    basename = filename.split('/')[-1].replace('.json', '')
    # Format: {wallet}-window-{window_start}
    parts = basename.split('-')
    if len(parts) >= 3 and parts[1] == 'window':
        wallet = parts[0]
        window_start = int(parts[2])
        return wallet, window_start
    return None, None

def sign_rollout(rollout_data: dict, wallet: bt.wallet) -> dict:
    """Sign a SAT rollout using the wallet hotkey"""
    # Create challenge string from key rollout data
    sat_seed = rollout_data.get('sat_seed', '')
    block_hash = rollout_data.get('block_hash', '')
    nonce = rollout_data.get('nonce', '')
    challenge = f"{sat_seed}{block_hash}{nonce}"
    rollout_data['challenge'] = challenge
    rollout_data['hotkey'] = wallet.hotkey.ss58_address
    rollout_data['signature'] = wallet.hotkey.sign(data=challenge).hex()
    return rollout_data

def verify_rollout_signature(rollout_data: dict) -> bool:
    """Verify the signature of a rollout"""
    try:
        challenge = rollout_data.get('challenge')
        hotkey = rollout_data.get('hotkey')
        signature = rollout_data.get('signature')
        
        if not all([challenge, hotkey, signature]):
            return False
            
        keypair = bt.Keypair(ss58_address=hotkey)
        return keypair.verify(data=challenge, signature=bytes.fromhex(signature))
    except Exception:
        return False

# REMOVED: derive_secret_key was insecure and has been removed
# The GRAIL proof system now uses wallet signatures for security

# Global storage for miner state
miner_inference_counts = defaultdict(list)  # track inferences per block for weight calculation

# --------------------------------------------------------------------------- #
#                               TRAINER                                       #
# --------------------------------------------------------------------------- #
# TODO(v2): Re-enable Trainer class with improved architecture
# - Async training that doesn't block mining
# - Optional local fine-tuning by miners  
# - Federated learning approach
# - Model checkpointing and versioning
'''
class Trainer:
    def __init__(self, model_name=LLAMA_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            use_safetensors=True
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare for training
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)
    
    def _check_model_health(self) -> bool:
        """Check if model has NaN or Inf parameters."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"NaN/Inf detected in parameter: {name}")
                return True
        return False
        
    async def train_window(self, hotkey: str, window: int) -> bool:
        """
        Train model on SAT rollouts from previous window using GRPO and upload for future window.
        
        IMPORTANT: The trainer only receives rollouts that have already been:
        1. Verified by validators using verify_rollout() 
        2. Confirmed to have valid GRAIL proofs (model identity verified)
        3. Checked for SAT problem/solution correctness
        
        This ensures we only train on legitimate model-generated rollouts.
        """
        
        # Download valid rollouts from the previous window  
        # These have already been verified by validators
        valid_rollouts = await get_valid_rollouts(window - WINDOW_LENGTH)
        
        if not valid_rollouts:
            logger.warning(f"No valid rollouts found for window {window - WINDOW_LENGTH}")
            # Still upload base model state if no training data
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
            
        logger.info(f"🎓 Training on {len(valid_rollouts)} SAT rollouts from window {window - WINDOW_LENGTH}")
        
        # Prepare training data for GRPO
        texts = []
        rewards = []
        trajectories = []  # Store trajectories for analysis
        successful_count = 0
        unique_solutions = set()  # Track unique successful solutions
        
        for rollout in valid_rollouts:
            try:
                # Extract SAT problem and rollout data
                commit = rollout.get('commit', {})
                tokens = commit.get('tokens', [])
                rollout_data = commit.get('rollout', {})
                sat_problem = commit.get('sat_problem', {})
                
                if not tokens or not rollout_data:
                    continue
                
                # Decode the full sequence (SAT problem + solution attempt)
                full_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                texts.append(full_text)
                
                # Calculate reward based on SAT solving performance
                # GRPO rewards: higher for successful solutions, partial credit for progress
                trajectory = rollout_data.get('trajectory', [])
                assignment = rollout_data.get('assignment', [])
                
                if rollout_data.get('success', False):
                    # High reward for successful solution
                    reward = 1.0
                    successful_count += 1
                    
                    # Track unique solutions for bonus rewards
                    solution_hash = hashlib.sha256(str(assignment).encode()).hexdigest()
                    if solution_hash not in unique_solutions:
                        unique_solutions.add(solution_hash)
                        reward += 0.5  # Bonus for finding unique solution
                        logger.debug(f"Found unique solution #{len(unique_solutions)}")
                else:
                    # Partial reward based on satisfied clauses
                    satisfied = rollout_data.get('satisfied_clauses', 0)
                    total = len(sat_problem.get('clauses', [1]))  # Avoid division by zero
                    reward = -0.5 + (satisfied / total) * 0.5  # Range: [-0.5, 0]
                
                # Add trajectory reward (bonus for efficiency)
                if trajectory and rollout_data.get('success', False):
                    # Bonus for solving quickly
                    efficiency_bonus = max(0, 0.2 * (1 - len(trajectory) / (sat_problem.get('num_vars', 10) * 2)))
                    reward += efficiency_bonus
                
                rewards.append(reward)
                trajectories.append(trajectory)
                
            except Exception as e:
                logger.debug(f"Skipping invalid SAT rollout: {e}")
                continue
        
        if not texts:
            logger.warning("No valid training texts extracted")
            # Still upload base model state
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
            
        logger.info(f"📚 Training on {len(texts)} SAT rollouts ({successful_count} successful, {len(unique_solutions)} unique)")
        logger.info(f"📊 Average reward: {sum(rewards)/len(rewards):.3f}, Max: {max(rewards):.3f}")
        
        # GRPO-style training: reinforce successful trajectories
        try:
            # Even lower learning rate for stability
            base_lr = 2e-6  # Reduced from 5e-6
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=base_lr,
                weight_decay=0.01,  # Add weight decay for regularization
                eps=1e-8  # Numerical stability
            )
            
            # Learning rate scheduler for warmup
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of base_lr
                total_iters=10  # Warmup over 10 steps
            )
            
            for epoch in range(2):  # Two epochs for better learning
                total_loss = 0
                batch_size = min(4, len(texts))  # Small batch size
                
                # Check model health before training
                if self._check_model_health():
                    logger.warning("Model has NaN/Inf parameters before training, skipping training")
                    break
                
                # Sort by rewards to prioritize learning from successful rollouts
                sorted_indices = sorted(range(len(texts)), key=lambda i: rewards[i], reverse=True)
                
                for batch_idx in range(0, len(sorted_indices), batch_size):
                    batch_indices = sorted_indices[batch_idx:batch_idx+batch_size]
                    batch_texts = [texts[i] for i in batch_indices]
                    batch_rewards = [rewards[i] for i in batch_indices]
                    
                    # Tokenize batch with explicit attention mask
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_attention_mask=True
                    )
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # GRPO reward weighting: emphasize high-reward trajectories
                    # Normalize rewards to [0, 1] range for this batch
                    min_reward = min(batch_rewards)
                    max_reward = max(batch_rewards)
                    if max_reward > min_reward:
                        normalized_rewards = [(r - min_reward) / (max_reward - min_reward) for r in batch_rewards]
                    else:
                        normalized_rewards = [0.5] * len(batch_rewards)
                    
                    # Apply reward-weighted loss
                    avg_normalized_reward = sum(normalized_rewards) / len(normalized_rewards)
                    reward_weight = 0.5 + avg_normalized_reward  # Range: [0.5, 1.5]
                    weighted_loss = loss * reward_weight
                    
                    # Backward pass
                    optimizer.zero_grad()
                    self.accelerator.backward(weighted_loss)
                    
                    # More aggressive gradient clipping for stability
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Reduced from 1.0
                    
                    # Check for gradient explosion
                    if grad_norm > 10.0:
                        logger.warning(f"Large gradient norm detected: {grad_norm:.2f}, skipping batch")
                        continue
                    
                    # Check for NaN/Inf gradients
                    has_nan_grad = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        logger.warning("NaN/Inf gradients detected, skipping batch")
                        continue
                    
                    optimizer.step()
                    scheduler.step()  # Update learning rate
                    
                    # Check model health after update
                    if self._check_model_health():
                        logger.error("Model became unhealthy during training, stopping")
                        break
                    
                    total_loss += weighted_loss.item()
                    
                avg_loss = total_loss / (len(texts) // batch_size + 1)
                logger.info(f"Epoch {epoch+1} completed - avg loss: {avg_loss:.4f}")
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Still try to upload base model
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success
        
        # Upload trained model state for future window (window + WINDOW_LENGTH)
        future_window = window + WINDOW_LENGTH
        logger.info(f"💾 Uploading trained model for future window {future_window}")
        success = await save_model_state(self.model, hotkey, future_window)
        
        if success:
            logger.info(f"✅ Successfully trained and uploaded model for window {future_window}")
        else:
            logger.error(f"❌ Failed to upload trained model for window {future_window}")
            
        return success
'''  # End of commented Trainer class

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("mine")(mine)

# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()
async def watchdog(timeout: int = 300):
    global HEARTBEAT
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)
            
# --------------------------------------------------------------------------- #
#                               MINER                                         #
# --------------------------------------------------------------------------- #
def mine(
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Use drand for randomness (default: True)",
        show_default=True,
    ),
) -> None:
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize model and prover
    logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model: {LLAMA_MODEL}")
    # Use wallet for secure GRAIL proof signatures
    prover = Prover(model_name=LLAMA_MODEL, wallet=wallet)
    
    async def _run():
        subtensor = None
        last_window_start = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()                
                current_block = await subtensor.get_current_block()
                
                # Calculate current window start (blocks divisible by WINDOW_LENGTH)
                window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                
                # Only process if we're in a new window
                if window_start <= last_window_start:
                    await asyncio.sleep(2)  # Wait for new window
                    continue
                
                # TODO(v2): Re-enable model state management for training
                # Check if model state exists for current window, wait if not
                # model_available = await model_state_exists(wallet.hotkey.ss58_address, window_start)
                # if not model_available:
                #     logger.info(f"⏳ Waiting for model state for window {window_start}...")
                #     await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                #     continue
                
                # Load model state for current window
                # logger.info(f"📥 Loading model state for window {window_start}")
                # try:
                #     success = await load_model_state(prover.model, wallet.hotkey.ss58_address, window_start)
                #     if success:
                #         logger.info(f"✅ Loaded model state for window {window_start}")
                #         # Update prover with new model state
                #         prover.model.eval()
                #     else:
                #         logger.warning(f"⚠️ Failed to load model state for window {window_start}, using base model")
                # except Exception as e:
                #     logger.warning(f"Error loading model state: {e}, using base model")
                #     pass
                
                # v1: Use base model directly without waiting
                logger.info(f"🚀 Using base model for window {window_start}")
                
                logger.info(f"🔥 Starting inference generation for window {window_start}-{window_start + WINDOW_LENGTH - 1}")
                
                # Check if we're already past this window
                current_check = await subtensor.get_current_block()
                if current_check > window_start + WINDOW_LENGTH - 2:
                    logger.warning(f"Window {window_start} nearly over (current block: {current_check}), waiting for next window")
                    last_window_start = window_start
                    await asyncio.sleep(5)
                    continue
                
                window_block_hash = await subtensor.get_block_hash(window_start)
                
                # Get drand randomness for this window if enabled
                if use_drand:
                    try:
                        drand_round = get_round_at_time(int(time.time()))
                        drand_beacon = get_drand_beacon(drand_round)
                        logger.info(f"🎲 Using drand randomness from round {drand_beacon['round']}")
                        # Combine drand with block hash for window randomness
                        combined_randomness = hashlib.sha256(
                            (window_block_hash + drand_beacon['randomness']).encode()
                        ).hexdigest()
                    except Exception as e:
                        logger.warning(f"Failed to get drand, using block hash only: {e}")
                        combined_randomness = window_block_hash
                else:
                    combined_randomness = window_block_hash
                
                # Generate as many inferences as possible during this window
                inferences = []
                start_time = time.time()
                inference_count = 0
                
                # Generate inferences until the window closes
                problem_count = 0
                while True:
                    current_block = await subtensor.get_current_block()
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                    
                    # Stop if we've moved to the next window
                    if current_window > window_start:
                        logger.info(f"Window {window_start} has ended, moving to next window")
                        break
                    
                    # Check if we're getting close to window end (leave time for upload)
                    blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
                    if blocks_remaining <= 2:  # Leave last 2 blocks for upload
                        logger.info(f"Approaching window end (blocks remaining: {blocks_remaining}), stopping generation")
                        break
                    
                    try:
                        problem_count += 1
                        inference_count += 1  # For logging
                        logger.info(f"⚡ Generating GRPO rollouts for problem {problem_count} (block {current_block}/{window_start + WINDOW_LENGTH - 1})...")
                        
                        # Clean up GPU memory periodically  
                        if inference_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                        
                        # Generate unique base nonce for problem group
                        base_nonce = random.randint(1000, 9999)
                        
                        # Generate SAT problem from seed (using base nonce)
                        sat_seed_base = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{base_nonce}"
                        difficulty = min(0.9, 0.3 + (inference_count * 0.01))  # Gradually increase difficulty
                        sat_problem = generate_sat_problem(sat_seed_base, difficulty)
                        logger.debug(f"Generated SAT problem: {sat_problem.num_vars} vars, {len(sat_problem.clauses)} clauses")

                        # Generate GRPO rollouts (4 per problem)
                        sat_generator = SATRolloutGenerator(
                            prover.model,
                            prover.tokenizer,
                            prover.device,
                            rollouts_per_problem=4  # GRPO standard
                        )

                        # Generate rollouts with GRAIL proofs
                        logger.debug(f"Generating GRPO rollouts with randomness: {combined_randomness[:16]}...")
                        grpo_rollouts = sat_generator.generate_grpo_rollouts(
                            sat_problem,
                            combined_randomness,
                            wallet  # Use wallet for secure signatures
                        )
                        
                        # Log GRPO statistics
                        successful_count = sum(1 for r in grpo_rollouts if r.success)
                        mean_reward = sum(r.reward for r in grpo_rollouts) / len(grpo_rollouts) if grpo_rollouts else 0
                        logger.info(f"GRPO batch: {successful_count}/{len(grpo_rollouts)} successful, mean reward: {mean_reward:.3f}")
                        
                        # Log progress every 10 problems
                        if problem_count % 10 == 0:
                            elapsed = time.time() - start_time
                            rollouts_per_sec = len(inferences) / elapsed if elapsed > 0 else 0
                            logger.info(f"📊 Progress: {len(inferences)} rollouts from {problem_count} problems in {elapsed:.1f}s ({rollouts_per_sec:.1f} rollouts/sec)")
                        
                        # Package rollouts for submission
                        for rollout_idx, rollout in enumerate(grpo_rollouts):
                            # Create unique nonce for this rollout while maintaining group association
                            rollout_nonce = base_nonce * 10 + rollout_idx  # e.g., 1234 -> 12340, 12341, 12342, 12343
                            rollout_sat_seed = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{rollout_nonce}"
                            
                            # Set up prover state for open() method
                            prover._state = {
                                "tokens": rollout.tokens,
                                "s_vals": rollout.s_vals,
                                "seq_len": len(rollout.tokens),
                                "signature": rollout.signature
                            }
                            proof_data = prover.open(combined_randomness)
                            
                            rollout_data = {
                                "window_start": window_start,
                                "block": current_block,
                                "nonce": rollout_nonce,  # Unique nonce per rollout
                                "sat_seed": rollout_sat_seed,  # Unique seed that validator can verify
                                "difficulty": difficulty,
                                "block_hash": window_block_hash,
                                "randomness": combined_randomness,
                                "use_drand": use_drand,
                                "rollout_group": base_nonce,  # Group identifier for GRPO rollouts
                                "rollout_index": rollout_idx,
                                "total_in_group": len(grpo_rollouts),
                                "commit": {
                                    "tokens": rollout.tokens,
                                    "s_vals": rollout.s_vals,
                                    "signature": rollout.signature.hex(),
                                    "beacon": rollout.beacon,
                                    "sat_problem": {
                                        "seed": sat_seed_base,  # Use base seed for GRPO group
                                        "num_vars": sat_problem.num_vars,
                                        "clauses": sat_problem.clauses,
                                        "difficulty": difficulty
                                    },
                                    "rollout": {
                                        "trajectory": rollout.trajectory,
                                        "total_reward": rollout.reward,
                                        "advantage": rollout.advantage,  # GRPO advantage
                                        "success": rollout.success,
                                        "token_logprobs": rollout.token_logprobs,  # For training
                                        "prompt_length": rollout.prompt_length,
                                        "completion_length": rollout.completion_length,
                                        # For backward compatibility
                                        "satisfied_clauses": len([c for c in sat_problem.clauses if any(
                                            (lit > 0 and rollout.trajectory[0][1][abs(lit)-1] if len(rollout.trajectory) > 0 and isinstance(rollout.trajectory[0][1], list) and abs(lit)-1 < len(rollout.trajectory[0][1]) else False) or
                                            (lit < 0 and not rollout.trajectory[0][1][abs(lit)-1] if len(rollout.trajectory) > 0 and isinstance(rollout.trajectory[0][1], list) and abs(lit)-1 < len(rollout.trajectory[0][1]) else False)
                                            for lit in c
                                        )]) if rollout.trajectory else 0,
                                        "assignment": rollout.trajectory[0][1] if rollout.trajectory and isinstance(rollout.trajectory[0][1], list) else []
                                    }
                                },
                                "proof": proof_data,
                                "timestamp": time.time()
                            }
                            
                            # Sign each rollout
                            rollout_data = sign_rollout(rollout_data, wallet)
                            inferences.append(rollout_data)
                        
                        # Tiny delay to yield control
                        await asyncio.sleep(0.01)
                        
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.error(f"CUDA error at inference {inference_count}: {e}")
                            logger.error(f"SAT problem: vars={sat_problem.num_vars if 'sat_problem' in locals() else 'N/A'}, "
                                       f"clauses={len(sat_problem.clauses) if 'sat_problem' in locals() else 'N/A'}")
                            logger.error(f"Difficulty: {difficulty if 'difficulty' in locals() else 'N/A'}")
                            # Try to recover by clearing cache
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Skip this inference and continue
                            continue
                        else:
                            raise
                    except Exception as e:
                        logger.warning(f"Failed to generate inference {inference_count}: {e}")
                        continue
                
                
                elapsed_time = time.time() - start_time
                logger.info(f"🎯 Generated {len(inferences)} rollouts in {elapsed_time:.1f}s for window {window_start}")
                
                if inferences:
                    logger.info(f"📤 Uploading {len(inferences)} rollouts to R2 for window {window_start}...")
                    try:
                        # Upload all inferences as a single window file
                        await sink_window_inferences(wallet, window_start, inferences)
                        logger.info(f"✅ Successfully uploaded window {window_start} with {len(inferences)} rollouts")
                    except Exception as e:
                        logger.error(f"❌ Failed to upload window {window_start}: {e}")
                        logger.error(traceback.format_exc())
                else:
                    logger.warning(f"No inferences generated for window {window_start}")
                
                last_window_start = window_start
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in miner loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
                
    async def _main():
        await asyncio.gather(
            _run(),
            watchdog()
        )
    asyncio.run(_main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    mine()