#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import hashlib
import logging
import os
import random
import time
import traceback

import bittensor as bt
import torch
import typer
from typing import Any, Optional, Tuple

# TODO(v2): Re-enable training imports
# from trl import PPOTrainer, PPOConfig
# TODO(v2): Re-enable for training
# from accelerate import Accelerator


from ..grail import Prover
from . import console
from ..infrastructure.drand import get_drand_beacon
from ..infrastructure.network import create_subtensor
from ..environments import generate_sat_problem, SATRolloutGenerator
from ..infrastructure.comms import sink_window_inferences
from ..infrastructure.credentials import load_r2_credentials
from ..infrastructure.chain import GrailChainManager
from ..shared.constants import WINDOW_LENGTH, MODEL_NAME
from ..monitoring import get_monitoring_manager
from ..monitoring.config import MonitoringConfig


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
SUBTENSOR: Optional[bt.subtensor] = None


async def get_subtensor() -> bt.subtensor:
    global SUBTENSOR
    if SUBTENSOR is None:
        logger.info("Making Bittensor connection...")
        SUBTENSOR = await create_subtensor()
        logger.info("Connected")
    return SUBTENSOR


# S3/R2 communication functions are now imported from comms.py


# --------------------------------------------------------------------------- #
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #
def generate_prompt(hotkey_address: str, block_hash: str, nonce: int) -> str:
    """Generate prompt in the required format"""
    return (
        f"Hey my name is {hotkey_address} it is currently {block_hash} days since friday "
        f"and my fav number is {nonce}, tell me a story about these three facts"
    )


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
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


def parse_window_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
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

        if not isinstance(signature, str):
            return False

        keypair = bt.Keypair(ss58_address=hotkey)
        result = keypair.verify(data=challenge, signature=bytes.fromhex(signature))
        return bool(result)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("mine")(mine)


# --------------------------------------------------------------------------- #
#                               Watchdog                                      #
# --------------------------------------------------------------------------- #
HEARTBEAT = time.monotonic()


async def watchdog(timeout: int = 600) -> None:
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
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Initialize model and prover
    logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model: {MODEL_NAME}")
    # Use wallet for secure GRAIL proof signatures
    prover = Prover(model_name=MODEL_NAME, wallet=wallet)

    async def _run() -> None:
        subtensor = None
        last_window_start = -1
        
        # Load R2 credentials
        try:
            credentials = load_r2_credentials()
            logger.info("✅ Loaded R2 credentials")
        except Exception as e:
            logger.error(f"Failed to load R2 credentials: {e}")
            raise
        
        # Initialize chain manager for credential commitments
        # Create a simple config object with just netuid
        from types import SimpleNamespace
        config = SimpleNamespace(netuid=int(get_conf("BT_NETUID", get_conf("NETUID", 200))))
        chain_manager = GrailChainManager(config, wallet, credentials)
        await chain_manager.initialize()
        logger.info("✅ Initialized chain manager and committed read credentials")
        
        # Initialize monitoring for mining operations
        monitor = get_monitoring_manager()
        if monitor:
            # Start a mining run with wallet-specific configuration
            mining_config = MonitoringConfig.for_mining(wallet.name)
            run_id = await monitor.start_run(f"mining_{wallet.name}", mining_config.get("hyperparameters", {}))
            logger.info(f"Started monitoring run: {run_id}")

        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
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

                logger.info(
                    f"🔥 Starting inference generation for window {window_start}-{window_start + WINDOW_LENGTH - 1}"
                )

                # Check if we're already past this window
                current_check = await subtensor.get_current_block()
                if current_check > window_start + WINDOW_LENGTH - 2:
                    logger.warning(
                        f"Window {window_start} nearly over (current block: {current_check}), waiting for next window"
                    )
                    last_window_start = window_start
                    await asyncio.sleep(5)
                    continue

                window_block_hash = await subtensor.get_block_hash(window_start)

                # Get drand randomness for this window if enabled
                if use_drand:
                    try:
                        drand_beacon = get_drand_beacon(None)
                        logger.info(
                            f"🎲 Using drand randomness from round {drand_beacon['round']}"
                        )
                        # Combine drand with block hash for window randomness
                        combined_randomness = hashlib.sha256(
                            (window_block_hash + drand_beacon["randomness"]).encode()
                        ).hexdigest()
                    except Exception as e:
                        logger.warning(f"Failed to get drand, using block hash only: {e}")
                        combined_randomness = window_block_hash
                else:
                    combined_randomness = window_block_hash

                # Generate as many inferences as possible during this window
                inferences: list = []
                start_time = time.time()
                inference_count = 0
                
                # Track mining metrics
                successful_rollouts = 0
                failed_rollouts = 0
                total_reward = 0.0
                # Debug text logging limits (only used when -vv)
                TEXT_LOG_LIMIT_PER_WINDOW = 5
                text_logs_emitted = 0

                # Generate inferences until the window closes
                problem_count = 0
                while True:
                    current_block = await subtensor.get_current_block()
                    # Refresh watchdog heartbeat during long-running inner loop
                    HEARTBEAT = time.monotonic()
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

                    # Stop if we've moved to the next window
                    if current_window > window_start:
                        logger.info(f"Window {window_start} has ended, moving to next window")
                        break

                    # Check if we're getting close to window end (leave time for upload)
                    blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
                    if blocks_remaining <= 2:  # Leave last 2 blocks for upload
                        logger.info(
                            f"Approaching window end (blocks remaining: {blocks_remaining}), stopping generation"
                        )
                        break

                    try:
                        problem_count += 1
                        inference_count += 1  # For logging
                        logger.info(
                            f"⚡ Generating GRPO rollouts for problem {problem_count} "
                            f"(block {current_block}/{window_start + WINDOW_LENGTH - 1})..."
                        )

                        # Clean up GPU memory periodically
                        if inference_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug(
                                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
                            )

                        # Generate unique base nonce for problem group
                        base_nonce = random.randint(1000, 9999)

                        # Generate SAT problem from seed (using base nonce)
                        sat_seed_base = (
                            f"{wallet.hotkey.ss58_address}-{window_block_hash}-{base_nonce}"
                        )
                        difficulty = min(
                            0.9, 0.3 + (inference_count * 0.01)
                        )  # Gradually increase difficulty
                        sat_problem = generate_sat_problem(sat_seed_base, difficulty)
                        logger.debug(
                            f"Generated SAT problem: {sat_problem.num_vars} vars, {len(sat_problem.clauses)} clauses"
                        )

                        # Generate GRPO rollouts (4 per problem)
                        sat_generator = SATRolloutGenerator(
                            prover.model,
                            prover.tokenizer,
                            prover.device,
                            rollouts_per_problem=4,  # GRPO standard
                        )

                        # Generate rollouts with GRAIL proofs
                        logger.debug(
                            f"Generating GRPO rollouts with randomness: {combined_randomness[:16]}..."
                        )
                        # Refresh heartbeat before potentially long generation
                        HEARTBEAT = time.monotonic()
                        grpo_rollouts = sat_generator.generate_grpo_rollouts(
                            sat_problem,
                            combined_randomness,
                            wallet,  # Use wallet for secure signatures
                        )
                        # Refresh after generation completes
                        HEARTBEAT = time.monotonic()

                        # Debug-only: sample and log a small subset of generated texts
                        if (
                            logger.isEnabledFor(logging.DEBUG)
                            and text_logs_emitted < TEXT_LOG_LIMIT_PER_WINDOW
                            and grpo_rollouts
                        ):
                            try:
                                sample = grpo_rollouts[0]
                                prompt_len = int(getattr(sample, "prompt_length", 0) or 0)
                                completion_len = int(getattr(sample, "completion_length", 0) or 0)
                                if completion_len > 0 and prompt_len >= 0:
                                    completion_ids = sample.tokens[prompt_len: prompt_len + completion_len]
                                else:
                                    completion_ids = sample.tokens[prompt_len:]
                                sample_text = prover.tokenizer.decode(
                                    completion_ids, skip_special_tokens=False
                                )
                                sample_nonce = base_nonce * 10
                                logger.debug(
                                    "TEXT[mine] window=%s group=%s nonce=%s reward=%.3f adv=%.3f "
                                    "success=%s text=%s",
                                    window_start,
                                    base_nonce,
                                    sample_nonce,
                                    float(sample.reward),
                                    float(sample.advantage),
                                    bool(sample.success),
                                    sample_text,
                                )
                                if monitor:
                                    await monitor.log_artifact(
                                        "mining/sample_text",
                                        {
                                            "window": window_start,
                                            "group": base_nonce,
                                            "nonce": sample_nonce,
                                            "reward": float(sample.reward),
                                            "advantage": float(sample.advantage),
                                            "success": bool(sample.success),
                                            "text": sample_text,
                                        },
                                        "text",
                                    )
                                text_logs_emitted += 1
                            except Exception:
                                # Do not disrupt mining on debug logging errors
                                pass

                        # Log GRPO statistics
                        successful_count = sum(1 for r in grpo_rollouts if r.success)
                        mean_reward = (
                            sum(r.reward for r in grpo_rollouts) / len(grpo_rollouts)
                            if grpo_rollouts
                            else 0
                        )
                        logger.info(
                            f"GRPO batch: {successful_count}/{len(grpo_rollouts)} successful, mean reward: {mean_reward:.3f}"
                        )

                        # Log progress every 10 problems
                        if problem_count % 10 == 0:
                            elapsed = time.time() - start_time
                            rollouts_per_sec = len(inferences) / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"📊 Progress: {len(inferences)} rollouts from {problem_count} problems "
                                f"in {elapsed:.1f}s ({rollouts_per_sec:.1f} rollouts/sec)"
                            )
                            
                            # Log mining metrics
                            if monitor:
                                await monitor.log_gauge("mining.rollouts_generated", len(inferences))
                                await monitor.log_gauge("mining.problems_processed", problem_count)
                                await monitor.log_gauge("mining.rollouts_per_second", rollouts_per_sec)
                                if successful_rollouts + failed_rollouts > 0:
                                    success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
                                    await monitor.log_gauge("mining.success_rate", success_rate)

                        # Package rollouts for submission
                        for rollout_idx, rollout in enumerate(grpo_rollouts):
                            # Create unique nonce for this rollout while maintaining group association
                            rollout_nonce = (
                                base_nonce * 10 + rollout_idx
                            )  # e.g., 1234 -> 12340, 12341, 12342, 12343
                            rollout_sat_seed = (
                                f"{wallet.hotkey.ss58_address}-{window_block_hash}-{rollout_nonce}"
                            )

                            # Set up prover state for open() method
                            prover._state = {
                                "tokens": rollout.tokens,
                                "s_vals": rollout.s_vals,
                                "seq_len": len(rollout.tokens),
                                "signature": rollout.signature,
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
                                        "difficulty": difficulty,
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
                                        "satisfied_clauses": (
                                            len(
                                                [
                                                    c
                                                    for c in sat_problem.clauses
                                                    if any(
                                                        (
                                                            lit > 0
                                                            and rollout.trajectory[0][1][
                                                                abs(lit) - 1
                                                            ]
                                                            if len(rollout.trajectory) > 0
                                                            and isinstance(
                                                                rollout.trajectory[0][1], list
                                                            )
                                                            and abs(lit) - 1
                                                            < len(rollout.trajectory[0][1])
                                                            else False
                                                        )
                                                        or (
                                                            lit < 0
                                                            and not rollout.trajectory[0][1][
                                                                abs(lit) - 1
                                                            ]
                                                            if len(rollout.trajectory) > 0
                                                            and isinstance(
                                                                rollout.trajectory[0][1], list
                                                            )
                                                            and abs(lit) - 1
                                                            < len(rollout.trajectory[0][1])
                                                            else False
                                                        )
                                                        for lit in c
                                                    )
                                                ]
                                            )
                                            if rollout.trajectory
                                            else 0
                                        ),
                                        "assignment": (
                                            rollout.trajectory[0][1]
                                            if rollout.trajectory
                                            and isinstance(rollout.trajectory[0][1], list)
                                            else []
                                        ),
                                    },
                                },
                                "proof": proof_data,
                                "timestamp": time.time(),
                            }

                            # Sign each rollout
                            rollout_data = sign_rollout(rollout_data, wallet)
                            inferences.append(rollout_data)
                            
                            # Track rollout metrics
                            if rollout.success:
                                successful_rollouts += 1
                                total_reward += rollout.reward
                                if monitor:
                                    await monitor.log_counter("mining.successful_rollouts")
                                    await monitor.log_histogram("mining.reward_distribution", rollout.reward)
                            else:
                                failed_rollouts += 1
                                if monitor:
                                    await monitor.log_counter("mining.failed_rollouts")

                        # Tiny delay to yield control
                        await asyncio.sleep(0.01)
                        HEARTBEAT = time.monotonic()

                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.error(f"CUDA error at inference {inference_count}: {e}")
                            logger.error(
                                f"SAT problem: vars={sat_problem.num_vars if 'sat_problem' in locals() else 'N/A'}, "
                                f"clauses={len(sat_problem.clauses) if 'sat_problem' in locals() else 'N/A'}"
                            )
                            logger.error(
                                f"Difficulty: {difficulty if 'difficulty' in locals() else 'N/A'}"
                            )
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
                logger.info(
                    f"🎯 Generated {len(inferences)} rollouts in {elapsed_time:.1f}s for window {window_start}"
                )
                
                # Log final window metrics
                if monitor:
                    await monitor.log_counter("mining.windows_completed")
                    await monitor.log_gauge("mining.window_duration", elapsed_time)
                    await monitor.log_gauge("mining.total_rollouts_in_window", len(inferences))
                    if successful_rollouts + failed_rollouts > 0:
                        final_success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
                        await monitor.log_gauge("mining.final_success_rate", final_success_rate)
                    if successful_rollouts > 0:
                        avg_reward = total_reward / successful_rollouts
                        await monitor.log_gauge("mining.average_reward", avg_reward)

                if inferences:
                    logger.info(
                        f"📤 Uploading {len(inferences)} rollouts to R2 for window {window_start}..."
                    )
                    try:
                        if monitor:
                            with monitor.timer("mining.upload_duration"):
                                await sink_window_inferences(wallet, window_start, inferences, credentials)
                        else:
                            await sink_window_inferences(wallet, window_start, inferences, credentials)
                            
                        logger.info(
                            f"✅ Successfully uploaded window {window_start} with {len(inferences)} rollouts"
                        )
                        HEARTBEAT = time.monotonic()
                        if monitor:
                            await monitor.log_counter("mining.successful_uploads")
                            await monitor.log_gauge("mining.uploaded_rollouts", len(inferences))
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to upload window {window_start}: {e}")
                        logger.error(traceback.format_exc())
                        if monitor:
                            await monitor.log_counter("mining.failed_uploads")
                else:
                    logger.warning(f"No inferences generated for window {window_start}")
                    if monitor:
                        await monitor.log_counter("mining.empty_windows")

                last_window_start = window_start

            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in miner loop: {e}. Continuing ...")
                subtensor = None  # Force reconnection on next iteration
                await asyncio.sleep(10)  # Wait before retrying
                continue

    async def _main() -> None:
        await asyncio.gather(_run(), watchdog())

    asyncio.run(_main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    mine()
