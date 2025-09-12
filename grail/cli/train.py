#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import hashlib
import logging
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Optional, cast

import bittensor as bt
import torch
import typer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.comms import get_valid_rollouts, save_model_state
from ..infrastructure.network import create_subtensor
from ..monitoring import get_monitoring_manager
from ..monitoring.config import MonitoringConfig
from ..shared.constants import MODEL_NAME, WINDOW_LENGTH
from . import console

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
logger = logging.getLogger("grail")

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #


def get_conf(key: str, default: Any = None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>")
        raise typer.Exit(code=1)
    return v or default


# --------------------------------------------------------------------------- #
#                               Subtensor                                     #
# --------------------------------------------------------------------------- #
SUBTENSOR = None


async def get_subtensor() -> bt.subtensor:
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.info("Making Bittensor connection...")
        SUBTENSOR = await create_subtensor()
        logger.info("Connected")
    return SUBTENSOR


# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #


def parse_filename(
    filename: str,
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse filename to extract wallet, block, nonce"""
    # Remove prefix and extension
    basename = filename.split("/")[-1].replace(".json", "")
    parts = basename.split("-")
    if len(parts) >= 3:
        wallet = parts[0]
        block = int(parts[1])
        nonce = int(parts[2])
        return wallet, block, nonce
    return None, None, None


def parse_window_filename(
    filename: str,
) -> tuple[Optional[str], Optional[int]]:
    """Parse window filename to extract wallet and window_start"""
    # Remove prefix and extension
    basename = filename.split("/")[-1].replace(".json", "")
    # Format: {wallet}-window-{window_start}
    parts = basename.split("-")
    if len(parts) >= 3 and parts[1] == "window":
        wallet = parts[0]
        window_start = int(parts[2])
        return wallet, window_start
    return None, None


def sign_rollout(rollout_data: dict, wallet: bt.wallet) -> dict:
    """Sign a SAT rollout using the wallet hotkey"""
    # Create challenge string from key rollout data
    sat_seed = rollout_data.get("sat_seed", "")
    block_hash = rollout_data.get("block_hash", "")
    nonce = rollout_data.get("nonce", "")
    challenge = f"{sat_seed}{block_hash}{nonce}"
    rollout_data["challenge"] = challenge
    rollout_data["hotkey"] = wallet.hotkey.ss58_address
    rollout_data["signature"] = wallet.hotkey.sign(data=challenge).hex()
    return rollout_data


def verify_rollout_signature(rollout_data: dict) -> bool:
    """Verify the signature of a rollout"""
    try:
        challenge = rollout_data.get("challenge")
        hotkey = rollout_data.get("hotkey")
        signature = rollout_data.get("signature")

        if not all([challenge, hotkey, signature]):
            return False

        keypair = bt.Keypair(ss58_address=hotkey)
        return keypair.verify(data=challenge, signature=bytes.fromhex(signature))  # type: ignore
    except Exception:
        return False


# Global storage for miner state
miner_inference_counts: defaultdict[str, list] = defaultdict(
    list
)  # track inferences per block for weight calculation

# --------------------------------------------------------------------------- #
#                               TRAINER                                       #
# --------------------------------------------------------------------------- #
# TODO(v2): Re-enable Trainer class with improved architecture
# - Async training that doesn't block mining
# - Optional local fine-tuning by miners
# - Federated learning approach
# - Model checkpointing and versioning


class Trainer:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()

        # Load base model and tokenizer
        logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else None,
            use_safetensors=True,
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
        monitor = get_monitoring_manager()

        # Download valid rollouts from the previous window
        # These have already been verified by validators
        valid_rollouts = await get_valid_rollouts(window - WINDOW_LENGTH)

        if not valid_rollouts:
            logger.warning(f"No valid rollouts found for window {window - WINDOW_LENGTH}")
            # Still upload base model state if no training data
            success = await save_model_state(
                cast(AutoModelForCausalLM, self.model), hotkey, window + WINDOW_LENGTH
            )
            return success

        logger.info(
            f"🎓 Training on {len(valid_rollouts)} SAT rollouts from window {window - WINDOW_LENGTH}"
        )

        # Prepare training data for GRPO
        texts = []
        rewards = []
        trajectories = []  # Store trajectories for analysis
        successful_count = 0
        unique_solutions = set()  # Track unique successful solutions

        for rollout in valid_rollouts:
            try:
                # Extract SAT problem and rollout data
                commit = rollout.get("commit", {})
                tokens = commit.get("tokens", [])
                rollout_data = commit.get("rollout", {})
                sat_problem = commit.get("sat_problem", {})

                if not tokens or not rollout_data:
                    continue

                # Decode the full sequence (SAT problem + solution attempt)
                full_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                texts.append(full_text)

                # Calculate reward based on SAT solving performance
                # GRPO rewards: higher for successful solutions, partial credit for progress
                trajectory = rollout_data.get("trajectory", [])
                assignment = rollout_data.get("assignment", [])

                if rollout_data.get("success", False):
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
                    satisfied = rollout_data.get("satisfied_clauses", 0)
                    total = len(sat_problem.get("clauses", [1]))  # Avoid division by zero
                    reward = -0.5 + (satisfied / total) * 0.5  # Range: [-0.5, 0]

                # Add trajectory reward (bonus for efficiency)
                if trajectory and rollout_data.get("success", False):
                    # Bonus for solving quickly
                    efficiency_bonus = max(
                        0,
                        0.2 * (1 - len(trajectory) / (sat_problem.get("num_vars", 10) * 2)),
                    )
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

        logger.info(
            f"📚 Training on {len(texts)} SAT rollouts "
            f"({successful_count} successful, {len(unique_solutions)} unique)"
        )
        logger.info(
            f"📊 Average reward: {sum(rewards) / len(rewards):.3f}, Max: {max(rewards):.3f}"
        )

        # GRPO-style training: reinforce successful trajectories
        try:
            # Even lower learning rate for stability
            base_lr = 2e-6  # Reduced from 5e-6
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=base_lr,
                weight_decay=0.01,  # Add weight decay for regularization
                eps=1e-8,  # Numerical stability
            )

            # Learning rate scheduler for warmup
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # Start at 10% of base_lr
                total_iters=10,  # Warmup over 10 steps
            )

            for epoch in range(2):  # Two epochs for better learning
                total_loss = 0
                batch_size = min(4, len(texts))  # Small batch size

                # Check model health before training
                if self._check_model_health():
                    logger.warning(
                        "Model has NaN/Inf parameters before training, skipping training"
                    )
                    break

                # Sort by rewards to prioritize learning from successful rollouts
                sorted_indices = sorted(range(len(texts)), key=lambda i: rewards[i], reverse=True)

                for batch_idx in range(0, len(sorted_indices), batch_size):
                    batch_indices = sorted_indices[batch_idx : batch_idx + batch_size]
                    batch_texts = [texts[i] for i in batch_indices]
                    batch_rewards = [rewards[i] for i in batch_indices]

                    # Tokenize batch with explicit attention mask
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_attention_mask=True,
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
                        normalized_rewards = [
                            (r - min_reward) / (max_reward - min_reward) for r in batch_rewards
                        ]
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=0.5
                    )  # Reduced from 1.0

                    # Check for gradient explosion
                    if grad_norm > 10.0:
                        logger.warning(
                            f"Large gradient norm detected: {grad_norm:.2f}, skipping batch"
                        )
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
                logger.info(f"Epoch {epoch + 1} completed - avg loss: {avg_loss:.4f}")

                # Log training metrics
                if monitor:
                    await monitor.log_gauge("training.epoch_loss", avg_loss)
                    await monitor.log_counter("training.epochs_completed")
                    await monitor.log_gauge("training.gradient_norm", float(grad_norm))
                    await monitor.log_gauge("training.successful_solutions", successful_count)
                    await monitor.log_gauge("training.total_rollouts", len(texts))
                    if len(texts) > 0:
                        success_rate = successful_count / len(texts)
                        await monitor.log_gauge("training.training_success_rate", success_rate)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Still try to upload base model
            success = await save_model_state(self.model, hotkey, window + WINDOW_LENGTH)
            return success

        # Upload trained model state for future window (window + WINDOW_LENGTH)
        future_window = window + WINDOW_LENGTH
        logger.info(f"💾 Uploading trained model for future window {future_window}")
        success = await save_model_state(self.model, hotkey, future_window)  # type: ignore

        if success:
            logger.info(f"✅ Successfully trained and uploaded model for window {future_window}")
        else:
            logger.error(f"❌ Failed to upload trained model for window {future_window}")

        return success


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("train")(train)


# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()


async def watchdog(timeout: int = 600) -> None:
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            os._exit(1)


# --------------------------------------------------------------------------- #
#                               TRAINER CLI                                   #
# --------------------------------------------------------------------------- #
# TODO(v2): Re-enable train command with improved architecture


def train() -> None:
    """Run the training process"""
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Initialize trainer
    logger.info(f"Initializing trainer with model: {MODEL_NAME}")
    trainer = Trainer(model_name=MODEL_NAME)

    async def _run() -> None:
        subtensor = None
        last_processed_window = -1

        # Initialize monitoring for training operations
        monitor = get_monitoring_manager()
        if monitor:
            # Start a training run with wallet-specific configuration
            training_config = MonitoringConfig.for_training(wallet.name)
            run_id = await monitor.start_run(
                f"training_{wallet.name}",
                training_config.get("hyperparameters", {}),
            )
            logger.info(f"Started monitoring run: {run_id}")

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
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
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
                if monitor:
                    with monitor.timer("training.window_duration"):
                        success = await trainer.train_window(
                            wallet.hotkey.ss58_address, target_window
                        )
                else:
                    success = await trainer.train_window(wallet.hotkey.ss58_address, target_window)

                if success:
                    logger.info(f"✅ Completed training cycle for window {target_window}")
                    if monitor:
                        await monitor.log_counter("training.successful_windows")
                else:
                    logger.warning(f"⚠️ Training cycle had issues for window {target_window}")
                    if monitor:
                        await monitor.log_counter("training.failed_windows")

                last_processed_window = target_window

            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in trainer loop: {e}. Continuing...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(30)  # Wait before retrying
                continue

    async def _main() -> None:
        await asyncio.gather(_run(), watchdog(timeout=(60 * 15)))  # 15 minute timeout for training

    asyncio.run(_main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    train()
