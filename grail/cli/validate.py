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
    app.command("validate")(validate)

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
#                               Validator                                     #
# --------------------------------------------------------------------------- #
def validate(
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Verify drand randomness (default: True)",
        show_default=True,
    ),
    test_mode: bool = typer.Option(
        True,
        "--test-mode/--no-test-mode",
        help="Test mode: validate own files (default: True)",
        show_default=True,
    ),
) -> None:
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey  = get_conf("BT_WALLET_HOT", "default")
    wallet  = bt.wallet(name=coldkey, hotkey=hotkey)
    
    # Initialize verifier
    logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model for validation: {LLAMA_MODEL}")
    verifier = Verifier(model_name=LLAMA_MODEL)
    
    # Login to Hugging Face for dataset uploads
    logger.info("🤗 Logging into Hugging Face for dataset uploads...")
    login_huggingface()
    
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
                
                # TODO(v2): Re-enable model state management for training
                # Check if model state exists for target window, wait if not
                # model_available = await model_state_exists(wallet.hotkey.ss58_address, target_window)
                # if not model_available:
                #     logger.info(f"⏳ Waiting for model state for window {target_window}...")
                #     await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                #     continue
                
                logger.info(f"🔍 Processing window {target_window}-{target_window + WINDOW_LENGTH - 1}")
                
                # Load model state for target window
                # logger.info(f"📥 Loading model state for window {target_window}")
                # try:
                #     success = await load_model_state(verifier.model, wallet.hotkey.ss58_address, target_window)
                #     if success:
                #         logger.info(f"✅ Loaded model state for window {target_window}")
                #         verifier.model.eval()
                #     else:
                #         logger.warning(f"⚠️ Failed to load model state for window {target_window}, using base model")
                # except Exception as e:
                #     logger.warning(f"Error loading model state: {e}, using base model")
                #     pass
                
                # v1: Use base model directly without waiting
                logger.info(f"🚀 Using base model for verification")
                
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
                        
                        # Spot check configuration
                        MIN_SAMPLES_PER_MINER = 3      # Minimum rollouts to check
                        MAX_SAMPLES_PER_MINER = 20     # Maximum rollouts to check  
                        SAMPLE_RATE = 0.1               # Check 10% of rollouts
                        FAILURE_THRESHOLD = 0.3         # Stop if >30% failures
                        BATCH_SIZE = 5                  # Check in batches for early stopping
                        
                        # Calculate sample size based on total inferences
                        total_inferences = len(inferences)
                        
                        # For GRPO, we need to check complete groups
                        # First, identify all groups in the inferences
                        groups_map = defaultdict(list)
                        for idx, inf in enumerate(inferences):
                            group_id = inf.get("rollout_group")
                            if group_id is not None:
                                groups_map[group_id].append(idx)
                            else:
                                # Non-GRPO rollout, treat as individual
                                groups_map[f"single_{idx}"] = [idx]
                        
                        # Decide whether to spot check or verify all
                        if total_inferences <= MAX_SAMPLES_PER_MINER:
                            # If few enough rollouts, check them all
                            indices_to_check = list(range(total_inferences))
                            use_spot_check = False
                            logger.info(f"🔍 Verifying all {total_inferences} rollouts from {wallet_addr}")
                        else:
                            # For spot checking, sample complete GRPO groups
                            use_spot_check = True
                            indices_to_check = []
                            
                            # Calculate how many groups to check
                            num_groups = len(groups_map)
                            groups_to_check = max(1, min(num_groups, int(num_groups * SAMPLE_RATE)))
                            
                            # Randomly select groups
                            selected_groups = random.sample(list(groups_map.keys()), groups_to_check)
                            
                            # Add all rollouts from selected groups
                            for group_id in selected_groups:
                                indices_to_check.extend(groups_map[group_id])
                            
                            indices_to_check.sort()  # Process in order for better cache locality
                            logger.info(f"📊 Spot checking {len(indices_to_check)}/{total_inferences} rollouts from {groups_to_check}/{num_groups} groups ({SAMPLE_RATE*100:.0f}% of groups)")
                        
                        valid_count = 0
                        checked_count = 0
                        successful_rollouts = 0
                        unique_solutions = set()  # Track unique successful solutions
                        nonces_seen = set()
                        rollout_groups = defaultdict(list)  # Track GRPO groups
                        
                        # Progressive verification with early stopping
                        should_stop = False
                        batch_failures = 0
                        
                        for idx, inference_idx in enumerate(indices_to_check):
                            # Early stopping check every BATCH_SIZE verifications
                            if checked_count > 0 and checked_count % BATCH_SIZE == 0:
                                failure_rate = (checked_count - valid_count) / checked_count
                                if failure_rate > FAILURE_THRESHOLD and checked_count >= MIN_SAMPLES_PER_MINER:
                                    logger.warning(f"⚠️ Early stopping for {wallet_addr}: {failure_rate:.1%} failure rate after {checked_count} checks")
                                    should_stop = True
                                    break
                            
                            inference = inferences[inference_idx]
                            checked_count += 1
                            
                            # Track GRPO groups
                            rollout_group = inference.get("rollout_group")
                            if rollout_group:
                                rollout_groups[rollout_group].append(inference)
                            
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
                                    logger.debug(f"Invalid SAT seed in inference from {wallet_addr}: expected {expected_seed}, got {inference.get('sat_seed')}")
                                    continue
                                
                                # Verify GRAIL proof and SAT rollout
                                # We must verify ALL rollouts to ensure model identity
                                try:
                                    logger.debug(f"Verifying SAT rollout from {wallet_addr}")
                                    
                                    # For GRPO rollouts, we need to modify the commit data to use the base problem
                                    commit_data = inference["commit"]
                                    rollout_group = inference.get("rollout_group")
                                    if rollout_group:
                                        # This is a GRPO rollout - regenerate base problem for verification
                                        base_sat_seed = f"{wallet_addr}-{target_window_hash}-{rollout_group}"
                                        base_problem = generate_sat_problem(base_sat_seed, inference.get("difficulty", 0.5))
                                        # Update commit data with base problem for verification
                                        commit_data["sat_problem"]["seed"] = base_sat_seed
                                        # The verifier will regenerate the problem from this seed
                                    
                                    # Use wallet address for signature verification (public key verification)
                                    is_valid = verifier.verify_rollout(commit_data, inference["proof"], wallet_addr)
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
                        
                        # Verify GRPO groups after processing checked inferences
                        grpo_valid_groups = 0
                        grpo_invalid_groups = 0
                        grpo_incomplete_groups = 0
                        
                        for group_id, group_rollouts in rollout_groups.items():
                            # Skip single rollout "groups" (non-GRPO)
                            if str(group_id).startswith("single_"):
                                continue
                                
                            # Verify group has multiple rollouts (GRPO requirement)
                            if len(group_rollouts) < 2:
                                logger.debug(f"GRPO group {group_id} has only {len(group_rollouts)} rollouts in checked sample, may be incomplete due to spot-checking")
                                grpo_incomplete_groups += 1
                                continue
                            
                            # Check if this looks like a complete group (should have 4 rollouts for GRPO)
                            expected_group_size = 4  # Standard GRPO uses 4 rollouts per problem
                            if len(group_rollouts) != expected_group_size:
                                logger.debug(f"GRPO group {group_id} has {len(group_rollouts)} rollouts, expected {expected_group_size}")
                            
                            # Verify advantages sum to ~0 (GRPO property)
                            advantages = []
                            for r in group_rollouts:
                                adv = r.get("commit", {}).get("rollout", {}).get("advantage", 0.0)
                                advantages.append(adv)
                            
                            advantage_sum = sum(advantages)
                            if abs(advantage_sum) > 0.01:  # Allow small floating point errors
                                logger.debug(f"GRPO group {group_id} advantages don't sum to 0: {advantage_sum} (advantages: {advantages})")
                                grpo_invalid_groups += 1
                                continue
                            
                            # Verify all rollouts in group have same base problem
                            # They should all have the same rollout_group and same base sat_problem seed
                            base_seeds = []
                            for r in group_rollouts:
                                sat_problem = r.get("commit", {}).get("sat_problem", {})
                                base_seeds.append(sat_problem.get("seed"))
                            
                            if len(set(base_seeds)) != 1:
                                logger.debug(f"GRPO group {group_id} has different base problems: {set(base_seeds)}")
                                grpo_invalid_groups += 1
                                continue
                            
                            logger.debug(f"✅ GRPO group {group_id} verified: {len(group_rollouts)} rollouts, advantages sum to {advantage_sum:.6f}")
                            grpo_valid_groups += 1
                        
                        if rollout_groups:
                            # Only report on groups we actually checked
                            total_groups_checked = grpo_valid_groups + grpo_invalid_groups + grpo_incomplete_groups
                            if total_groups_checked > 0:
                                logger.info(f"GRPO groups checked: {grpo_valid_groups} valid, {grpo_invalid_groups} invalid, {grpo_incomplete_groups} incomplete (spot-check artifact)")
                        
                        # Calculate estimated total valid rollouts based on sampling
                        if should_stop:
                            # If we stopped early due to failures, assume 0 valid rollouts
                            estimated_valid = 0
                        else:
                            # Extrapolate from sample to estimate total
                            sample_pass_rate = valid_count / checked_count if checked_count > 0 else 0
                            estimated_valid = int(total_inferences * sample_pass_rate)
                        
                        # Store metrics for this miner
                        window_inference_counts[wallet_addr] = {
                            "valid": valid_count,
                            "checked": checked_count,
                            "total": total_inferences,
                            "estimated_valid": estimated_valid,
                            "successful": successful_rollouts,
                            "unique": len(unique_solutions)
                        }
                        total_valid_rollouts += estimated_valid  # Use estimated for rewards
                        
                        logger.info(f"✅ {wallet_addr}: {valid_count}/{checked_count} checked, ~{estimated_valid}/{total_inferences} estimated valid, {successful_rollouts} successful, {len(unique_solutions)} unique")
                        
                    except Exception as e:
                        logger.warning(f"Error processing window file {filename}: {e}")
                        continue
                
                logger.info(f"📁 Found {files_found} window files from {len(meta.hotkeys)} active hotkeys")
                logger.info(f"🏁 Total valid rollouts in window {target_window}: {total_valid_rollouts}")
                
                # Upload all valid rollouts for training and to Hugging Face
                if all_valid_rollouts:
                    # Upload to S3/R2 for immediate access
                    upload_success = await upload_valid_rollouts(target_window, all_valid_rollouts)
                    if upload_success:
                        logger.info(f"📤 Uploaded {len(all_valid_rollouts)} valid rollouts for training")
                    else:
                        logger.warning(f"⚠️ Failed to upload valid rollouts for training")
                    
                    # NEW: Upload to Hugging Face dataset for community access
                    try:
                        hf_success = await upload_to_huggingface(all_valid_rollouts, target_window, PROTOCOL_VERSION)
                        if hf_success:
                            logger.info(f"🤗 Uploaded {len(all_valid_rollouts)} rollouts to Hugging Face dataset")
                        else:
                            logger.debug("Failed to upload to Hugging Face (may need HF_TOKEN)")
                    except Exception as e:
                        logger.debug(f"Hugging Face upload error: {e}")
                
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
            
    async def _main():
        await asyncio.gather(
            _run(),
            watchdog(timeout = (60 * 10))
        )
    asyncio.run(_main())

# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    validate()