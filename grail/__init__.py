#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import click
import random
import asyncio
import logging
import hashlib
import traceback
import bittensor as bt
from dotenv import load_dotenv
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file, save, load
from safetensors import safe_open
from trl import PPOTrainer, PPOConfig
from accelerate import Accelerator

__version__ = "0.0.0"

from .grail import Prover, Verifier
from .drand import get_drand_beacon, get_round_at_time
from .environments import SATProblem, SATEnvironment, generate_sat_problem
from .comms import (
    upload_file_chunked, download_file_chunked, file_exists, list_bucket_files,
    get_file, sink_window_inferences, save_model_state, load_model_state,
    model_state_exists, upload_valid_rollouts, get_valid_rollouts
)

__all__ = ["Prover", "Verifier", "SATProblem", "SATEnvironment", "generate_sat_problem", "main", "cli"]

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
def setup_logging(verbosity: int):
    level = TRACE if verbosity >= 3 else logging.DEBUG if verbosity == 2 else logging.INFO if verbosity == 1 else logging.CRITICAL + 1
    for noisy in ["websockets", "bittensor", "bittensor-cli", "btdecode", "asyncio", "aiobotocore.regions", "botocore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    
    # GRAIL debug details only visible with -vv or higher
    if verbosity < 2:
        logging.getLogger("grail").setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #
load_dotenv(override=True)
def get_conf(key, default=None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        click.echo(f"{key} not set.\nRun:\n\taf set {key} <value>", err=True)
        sys.exit(1)
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

def derive_secret_key(hotkey_address: str) -> bytes:
    """Derive deterministic secret key from hotkey for verification"""
    return hashlib.sha256(f"grail_secret_{hotkey_address}".encode()).digest()

# Global storage for miner state
miner_inference_counts = defaultdict(list)  # track inferences per block for weight calculation

# --------------------------------------------------------------------------- #
#                               TRAINER                                       #
# --------------------------------------------------------------------------- #
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
            device_map="auto" if torch.cuda.is_available() else None
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
                    
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
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

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
@click.group()
@click.option('-v', '--verbose', count=True, help='Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)')
def cli(verbose):
    """GRAIL CLI"""
    setup_logging(verbose)

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
@cli.command("mine")
@click.option('--use-drand/--no-drand', default=True, help='Use drand for randomness (default: True)')
def mine(use_drand):    
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize model and prover
    logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model: {LLAMA_MODEL}")
    prover = Prover(model_name=LLAMA_MODEL)
    # Set deterministic secret key based on hotkey
    prover.secret_key = derive_secret_key(wallet.hotkey.ss58_address)
    
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
                
                # Check if model state exists for current window, wait if not
                model_available = await model_state_exists(wallet.hotkey.ss58_address, window_start)
                if not model_available:
                    logger.info(f"⏳ Waiting for model state for window {window_start}...")
                    await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                    continue
                
                # Load model state for current window
                logger.info(f"📥 Loading model state for window {window_start}")
                try:
                    success = await load_model_state(prover.model, wallet.hotkey.ss58_address, window_start)
                    if success:
                        logger.info(f"✅ Loaded model state for window {window_start}")
                        # Update prover with new model state
                        prover.model.eval()
                    else:
                        logger.warning(f"⚠️ Failed to load model state for window {window_start}, using base model")
                except Exception as e:
                    logger.warning(f"Error loading model state: {e}, using base model")
                    pass
                
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
                while True:
                    current_block = await subtensor.get_current_block()
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                    
                    # Stop if we've moved to the next window
                    if current_window > window_start:
                        logger.info(f"Window {window_start} has ended, moving to next window")
                        break
                    
                    try:
                        inference_count += 1
                        logger.info(f"⚡ Generating SAT rollout {inference_count}...")
                        
                        # Generate unique seed for SAT problem
                        nonce = random.randint(1000, 9999)
                        sat_seed = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{nonce}"
                        
                        # Generate SAT problem from seed
                        difficulty = min(0.9, 0.3 + (inference_count * 0.01))  # Gradually increase difficulty
                        sat_problem = generate_sat_problem(sat_seed, difficulty)
                        logger.debug(f"Generated SAT problem: {sat_problem.num_vars} vars, {len(sat_problem.clauses)} clauses")
                        
                        # Generate rollout with GRAIL proof using combined randomness
                        logger.debug(f"Generating rollout with randomness: {combined_randomness[:16]}...")
                        commit_data = prover.commit_rollout(sat_problem, combined_randomness, difficulty)
                        logger.debug(f"Rollout complete: success={commit_data['rollout']['success']}")
                        
                        proof_data = prover.open(combined_randomness)
                        logger.debug(f"Proof generated with {len(proof_data['indices'])} indices")
                        
                        # Prepare rollout data
                        rollout_data = {
                            "window_start": window_start,
                            "block": current_block,
                            "nonce": nonce,
                            "sat_seed": sat_seed,
                            "difficulty": difficulty,
                            "block_hash": window_block_hash,
                            "randomness": combined_randomness,
                            "use_drand": use_drand,
                            "commit": commit_data,
                            "proof": proof_data,
                            "timestamp": time.time()
                        }
                        
                        # Sign the rollout
                        rollout_data = sign_rollout(rollout_data, wallet)
                        
                        # Log successful rollouts
                        if commit_data["rollout"]["success"]:
                            logger.info(f"✅ Successfully solved SAT problem (vars={sat_problem.num_vars}, clauses={len(sat_problem.clauses)})")
                        
                        inferences.append(rollout_data)
                        
                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate inference {inference_count}: {e}")
                        continue
                
                
                elapsed_time = time.time() - start_time
                logger.info(f"🎯 Generated {len(inferences)} inferences in {elapsed_time:.1f}s for window {window_start}")
                
                if inferences:
                    # Upload all inferences as a single window file
                    await sink_window_inferences(wallet, window_start, inferences)
                    logger.info(f"📤 Uploaded window {window_start} with {len(inferences)} inferences")
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
                
    async def main():
        await asyncio.gather(
            _run(),
            watchdog()
        )
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #
@cli.command("validate")
@click.option('--use-drand/--no-drand', default=True, help='Verify drand randomness (default: True)')
@click.option('--test-mode/--no-test-mode', default=True, help='Test mode: validate own files (default: True)')
def validate(use_drand, test_mode):
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize verifier
    logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model for validation: {LLAMA_MODEL}")
    verifier = Verifier(model_name=LLAMA_MODEL)
    
    # Storage for inference counts per miner
    inference_counts = defaultdict(lambda: defaultdict(int))  # {hotkey: {window: count}}
    
    async def _run():
        subtensor = None
        last_processed_window = -1
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()

                meta = await subtensor.metagraph(NETUID)
                current_block = await subtensor.get_current_block()
                
                # Calculate current and previous windows
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                # Process the previous complete window
                target_window = current_window - WINDOW_LENGTH
                
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(5)  # Wait for new window
                    continue
                
                # Check if model state exists for target window, wait if not
                model_available = await model_state_exists(wallet.hotkey.ss58_address, target_window)
                if not model_available:
                    logger.info(f"⏳ Waiting for model state for window {target_window}...")
                    await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                    continue
                
                logger.info(f"🔍 Processing window {target_window}-{target_window + WINDOW_LENGTH - 1}")
                
                # Load model state for target window
                logger.info(f"📥 Loading model state for window {target_window}")
                try:
                    success = await load_model_state(verifier.model, wallet.hotkey.ss58_address, target_window)
                    if success:
                        logger.info(f"✅ Loaded model state for window {target_window}")
                        verifier.model.eval()
                    else:
                        logger.warning(f"⚠️ Failed to load model state for window {target_window}, using base model")
                except Exception as e:
                    logger.warning(f"Error loading model state: {e}, using base model")
                    pass
                
                # Get block hash for the window start
                target_window_hash = await subtensor.get_block_hash(target_window)
                
                # For testing: just use the validator's own hotkey (same as miner in local testing)
                # In production, this would iterate through meta.hotkeys
                test_mode = True  # Set to False for production
                
                if test_mode:
                    # Use the wallet's own hotkey for testing
                    hotkeys_to_check = [wallet.hotkey.ss58_address]
                    logger.info(f"🧪 TEST MODE: Checking files for own hotkey {wallet.hotkey.ss58_address} in window {target_window}")
                else:
                    # Use metagraph hotkeys for production
                    hotkeys_to_check = meta.hotkeys
                    logger.info(f"Checking files for {len(meta.hotkeys)} active hotkeys in window {target_window}")
                
                # Download and process files
                total_valid_rollouts = 0
                window_inference_counts = defaultdict(int)
                files_found = 0
                all_valid_rollouts = []  # Store all valid rollouts for uploading
                
                for wallet_addr in hotkeys_to_check:
                    try:
                        # Construct expected filename for this hotkey and window
                        filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"
                        
                        # Check if file exists before downloading
                        exists = await file_exists(filename)
                        if not exists:
                            logger.debug(f"No file found for {wallet_addr} at {filename}")
                            continue
                        
                        files_found += 1
                        logger.info(f"📁 Found file for hotkey {wallet_addr}")
                        
                        window_data = await get_file(filename)
                        if not window_data:
                            logger.warning(f"Could not download {filename}")
                            continue
                        
                        file_wallet_addr = window_data.get("wallet")
                        window_start = window_data.get("window_start")
                        inferences = window_data.get("inferences", [])
                        
                        # Basic window validation
                        if file_wallet_addr != wallet_addr:
                            logger.warning(f"Wallet mismatch in {filename}: expected {wallet_addr}, got {file_wallet_addr}")
                            continue
                        
                        if window_start != target_window:
                            logger.warning(f"Window mismatch in {filename}: expected {target_window}, got {window_start}")
                            continue
                        
                        # Verify all rollouts in the window
                        valid_count = 0
                        successful_rollouts = 0
                        unique_solutions = set()  # Track unique successful solutions
                        nonces_seen = set()
                        
                        for inference in inferences:
                            try:
                                # Check required fields for SAT rollouts
                                required_fields = ["window_start", "nonce", "sat_seed", "block_hash", "commit", "proof", "challenge", "hotkey", "signature"]
                                if not all(field in inference for field in required_fields):
                                    logger.debug(f"Missing required fields in inference from {wallet_addr}")
                                    continue
                                
                                # Check window consistency
                                if inference["window_start"] != target_window:
                                    logger.debug(f"Window mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check block hash matches
                                if inference["block_hash"] != target_window_hash:
                                    logger.debug(f"Block hash mismatch in inference from {wallet_addr}")
                                    continue
                                
                                # Check nonce uniqueness within window
                                nonce = inference["nonce"]
                                if nonce in nonces_seen:
                                    logger.debug(f"Duplicate nonce {nonce} in window from {wallet_addr}")
                                    continue
                                nonces_seen.add(nonce)
                                
                                # Verify signature
                                if not verify_rollout_signature(inference):
                                    logger.debug(f"Invalid signature for inference from {wallet_addr}")
                                    continue
                                
                                # Verify SAT seed format
                                expected_seed = f"{wallet_addr}-{target_window_hash}-{nonce}"
                                if inference.get("sat_seed") != expected_seed:
                                    logger.debug(f"Invalid SAT seed in inference from {wallet_addr}")
                                    continue
                                
                                # Verify GRAIL proof and SAT rollout
                                # We must verify ALL rollouts to ensure model identity
                                try:
                                    logger.debug(f"Verifying SAT rollout from {wallet_addr}")
                                    prover_secret_key = derive_secret_key(wallet_addr)
                                    is_valid = verifier.verify_rollout(inference["commit"], inference["proof"], prover_secret_key)
                                    if not is_valid:
                                        logger.warning(f"SAT rollout verification failed for {wallet_addr} - skipping")
                                        continue
                                except Exception as e:
                                    logger.warning(f"Rollout verification error for {wallet_addr}: {e}")
                                    continue
                                
                                valid_count += 1
                                
                                # Track successful unique solutions
                                rollout = inference.get("commit", {}).get("rollout", {})
                                if rollout.get("success", False):
                                    successful_rollouts += 1
                                    # Create hash of solution for uniqueness
                                    assignment = rollout.get("assignment", [])
                                    solution_hash = hashlib.sha256(str(assignment).encode()).hexdigest()
                                    unique_solutions.add(solution_hash)
                                
                                # Add to collection of all valid rollouts
                                all_valid_rollouts.append(inference)
                                
                            except Exception as e:
                                logger.debug(f"Error processing inference from {wallet_addr}: {e}")
                                continue
                        
                        # Store metrics for this miner
                        window_inference_counts[wallet_addr] = {
                            "valid": valid_count,
                            "successful": successful_rollouts,
                            "unique": len(unique_solutions)
                        }
                        total_valid_rollouts += valid_count
                        
                        logger.info(f"✅ {wallet_addr}: {valid_count} valid, {successful_rollouts} successful, {len(unique_solutions)} unique solutions")
                        
                    except Exception as e:
                        logger.warning(f"Error processing window file {filename}: {e}")
                        continue
                
                logger.info(f"📁 Found {files_found} window files from {len(meta.hotkeys)} active hotkeys")
                logger.info(f"🏁 Total valid rollouts in window {target_window}: {total_valid_rollouts}")
                
                # Upload all valid rollouts for training
                if all_valid_rollouts:
                    upload_success = await upload_valid_rollouts(target_window, all_valid_rollouts)
                    if upload_success:
                        logger.info(f"📤 Uploaded {len(all_valid_rollouts)} valid rollouts for training")
                    else:
                        logger.warning(f"⚠️ Failed to upload valid rollouts for training")
                
                # Update global inference counts for weight calculation
                for hotkey, metrics in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = metrics
                
                # Compute weights based on unique successful rollouts
                weights = []
                for uid, hotkey in enumerate(meta.hotkeys):
                    # Calculate score over last 3 windows
                    recent_windows = range(max(0, target_window - 2*WINDOW_LENGTH), target_window + 1, WINDOW_LENGTH)
                    
                    total_unique = 0
                    total_successful = 0
                    total_valid = 0
                    
                    for w in recent_windows:
                        metrics = inference_counts[hotkey].get(w, {})
                        if isinstance(metrics, dict):
                            total_unique += metrics.get("unique", 0)
                            total_successful += metrics.get("successful", 0)
                            total_valid += metrics.get("valid", 0)
                        else:
                            # Backward compatibility
                            total_valid += metrics if isinstance(metrics, (int, float)) else 0
                    
                    # Scoring formula: prioritize unique solutions, then successful, then valid
                    # Weight = 0.6 * unique_ratio + 0.3 * success_ratio + 0.1 * valid_ratio
                    unique_score = min(1.0, total_unique / 10.0) if total_unique > 0 else 0
                    success_score = min(1.0, total_successful / 20.0) if total_successful > 0 else 0
                    valid_score = min(1.0, total_valid / 50.0) if total_valid > 0 else 0
                    
                    weight = 0.6 * unique_score + 0.3 * success_score + 0.1 * valid_score
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [0.0] * len(meta.hotkeys)
                
                # Log non-zero weights
                non_zero_weights = [(meta.hotkeys[i], weights[i]) for i in range(len(weights)) if weights[i] > 0]
                if non_zero_weights:
                    logger.info(f"⚖️  Setting weights for {len(non_zero_weights)} miners")
                    for hotkey, weight in non_zero_weights[:5]:  # Show top 5
                        logger.info(f"   {hotkey}: {weight:.4f}")
                else:
                    logger.info("⚖️  No miners received weights this window")
                
                # Set weights on network
                await subtensor.set_weights(
                    wallet=wallet,
                    netuid=NETUID,
                    uids=meta.uids,
                    weights=weights,
                    wait_for_inclusion=False
                )
                
                last_processed_window = target_window
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue
            
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                               TRAINER CLI                                   #
# --------------------------------------------------------------------------- #
@cli.command("train")
def train():
    """Run the training process"""
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize trainer
    logger.info(f"Initializing trainer with model: {LLAMA_MODEL}")
    trainer = Trainer(model_name=LLAMA_MODEL)
    
    async def _run():
        subtensor = None
        last_processed_window = -1
        
        # Upload initial base model state on startup
        logger.info("🏁 Uploading initial base model state...")
        current_block = 0
        if subtensor is None:
            subtensor = await get_subtensor()
            current_block = await subtensor.get_current_block()
        
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        initial_window = current_window + WINDOW_LENGTH
        
        # Upload base model for the next window
        success = await save_model_state(trainer.model, wallet.hotkey.ss58_address, initial_window)
        if success:
            logger.info(f"✅ Uploaded initial model state for window {initial_window}")
        else:
            logger.error("❌ Failed to upload initial model state")
            return
        
        while True:
            try:
                global HEARTBEAT; HEARTBEAT = time.monotonic()
                if subtensor is None: 
                    subtensor = await get_subtensor()
                    
                current_block = await subtensor.get_current_block()
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                
                # Process previous complete window for training
                target_window = current_window - WINDOW_LENGTH
                
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(10)  # Wait for new window
                    continue
                
                logger.info(f"🎓 Processing training for window {target_window}")
                
                # Train on previous window's valid inferences and upload for future window
                success = await trainer.train_window(wallet.hotkey.ss58_address, target_window)
                
                if success:
                    logger.info(f"✅ Completed training cycle for window {target_window}")
                else:
                    logger.warning(f"⚠️ Training cycle had issues for window {target_window}")
                
                last_processed_window = target_window
                
            except asyncio.CancelledError: 
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in trainer loop: {e}. Continuing...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(30)  # Wait before retrying
                continue
    
    async def main():
        await asyncio.gather(
            _run(),
            watchdog(timeout=(60 * 15))  # 15 minute timeout for training
        )
    
    asyncio.run(main())

# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main():
    """Main entry point for the CLI"""
    cli()