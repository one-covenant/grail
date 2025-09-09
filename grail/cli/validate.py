#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import os
import time
import typer
import random
import asyncio
import logging
import hashlib
import traceback
import math
import bittensor as bt
from collections import defaultdict
from typing import Any, Tuple, Optional, DefaultDict

# TODO(v2): Re-enable training imports
# from trl import PPOTrainer, PPOConfig
# TODO(v2): Re-enable for training
# from accelerate import Accelerator


from ..grail import Verifier
from . import console

from ..environments import generate_sat_problem, create_sat_reward_vector
from ..infrastructure.network import create_subtensor
from ..infrastructure.comms import (
    file_exists,
    get_file,
    upload_valid_rollouts,
    upload_to_huggingface,
    login_huggingface,
    PROTOCOL_VERSION,
)
from ..infrastructure.credentials import load_r2_credentials
from ..infrastructure.chain import GrailChainManager
from ..shared.constants import NETUID, WINDOW_LENGTH, MODEL_NAME, SUPERLINEAR_EXPONENT
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
SUBTENSOR = None


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


# REMOVED: derive_secret_key was insecure and has been removed
# The GRAIL proof system now uses wallet signatures for security

# Global storage for miner state
miner_inference_counts: DefaultDict[str, list] = defaultdict(
    list
)  # track inferences per block for weight calculation


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("validate")(validate)


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
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    # Initialize verifier
    logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
    logger.info(f"Loading base model for validation: {MODEL_NAME}")
    verifier = Verifier(model_name=MODEL_NAME)

    # Login to Hugging Face for dataset uploads
    logger.info("🤗 Logging into Hugging Face for dataset uploads...")
    login_huggingface()

    # Storage for inference counts per miner
    inference_counts: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )  # {hotkey: {window: count}}

    # Declarative SAT reward bounds (composed from per-function bounds)
    try:
        _sat_rv = create_sat_reward_vector()
        SAT_REWARD_LOW, SAT_REWARD_HIGH = _sat_rv.reward_bounds()
    except Exception:
        # Fallback to permissive bounds if vector is unavailable
        SAT_REWARD_LOW, SAT_REWARD_HIGH = float("-inf"), float("inf")

    async def _run() -> None:
        subtensor = None
        
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
        config = SimpleNamespace(netuid=NETUID)
        chain_manager = GrailChainManager(config, wallet, credentials)
        await chain_manager.initialize()
        logger.info("✅ Initialized chain manager and committed read credentials")
        
        # Initialize monitoring for validation operations
        monitor = get_monitoring_manager()
        if monitor:
            # Start a validation run with wallet-specific configuration
            validation_config = MonitoringConfig.for_validation(wallet.name)
            run_id = await monitor.start_run(f"validation_{wallet.name}", validation_config.get("hyperparameters", {}))
            logger.info(f"Started monitoring run: {run_id}")
        last_processed_window = -1

        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
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
                    logger.debug(f"Waiting for new window {target_window}")
                    continue

                # TODO(v2): Re-enable model state management for training
                # Check if model state exists for target window, wait if not
                # model_available = await model_state_exists(wallet.hotkey.ss58_address, target_window)
                # if not model_available:
                #     logger.info(f"⏳ Waiting for model state for window {target_window}...")
                #     await asyncio.sleep(5)  # Wait for model to be uploaded by trainer
                #     continue

                logger.info(
                    f"🔍 Processing window {target_window}-{target_window + WINDOW_LENGTH - 1}"
                )
                
                # Set block context for monitoring
                if monitor:
                    monitor.set_block_context(current_block, target_window)

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
                logger.info("🚀 Using base model for verification")

                # Get block hash for the window start
                target_window_hash = await subtensor.get_block_hash(target_window)

                # For testing: just use the validator's own hotkey (same as miner in local testing)
                # In production, this would iterate through meta.hotkeys
                # Use the test_mode parameter passed to the function instead of hardcoding

                if test_mode:
                    # Use the wallet's own hotkey for testing
                    hotkeys_to_check = [wallet.hotkey.ss58_address]
                    logger.info(
                        f"🧪 TEST MODE: Checking files for own hotkey {wallet.hotkey.ss58_address} in window {target_window}"
                    )
                else:
                    # Use metagraph hotkeys for production
                    hotkeys_to_check = meta.hotkeys
                    logger.info(
                        f"Checking files for {len(meta.hotkeys)} active hotkeys in window {target_window}"
                    )
                    logger.info(f"Active hotkeys: {meta.hotkeys}")

                # Download and process files
                total_valid_rollouts = 0
                window_inference_counts: DefaultDict[str, int] = defaultdict(int)
                files_found = 0
                all_valid_rollouts = []  # Store all valid rollouts for uploading
                # Debug text logging limits (only used when -vv)
                TEXT_LOG_LIMIT_PER_WALLET = 5
                text_logs_emitted_by_wallet: DefaultDict[str, int] = defaultdict(int)
                
                # Track validation metrics
                total_rollouts_processed = 0
                invalid_signatures = 0
                invalid_proofs = 0
                processing_errors = 0

                for wallet_addr in hotkeys_to_check:
                    try:
                        # Construct expected filename for this hotkey and window
                        filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"

                        # Get miner's read credentials from chain
                        miner_bucket = chain_manager.get_bucket_for_hotkey(wallet_addr)
                        
                        # Check if file exists before downloading
                        # Use miner's read credentials if available, otherwise use our own
                        exists = await file_exists(filename, credentials=miner_bucket if miner_bucket else credentials, use_write=False)
                        if not exists:
                            logger.debug(f"No file found for {wallet_addr} at {filename}")
                            continue

                        files_found += 1
                        logger.info(f"📁 Found file for hotkey {wallet_addr}")

                        # Download using appropriate credentials
                        window_data = await get_file(filename, credentials=miner_bucket if miner_bucket else credentials, use_write=False)
                        if not window_data:
                            logger.warning(f"Could not download {filename}")
                            continue

                        file_wallet_addr = window_data.get("wallet")
                        window_start = window_data.get("window_start")
                        inferences = window_data.get("inferences", [])

                        # Basic window validation
                        if file_wallet_addr != wallet_addr:
                            logger.warning(
                                f"Wallet mismatch in {filename}: expected {wallet_addr}, got {file_wallet_addr}"
                            )
                            continue

                        if window_start != target_window:
                            logger.warning(
                                f"Window mismatch in {filename}: expected {target_window}, got {window_start}"
                            )
                            continue

                        # Spot check configuration
                        MIN_SAMPLES_PER_MINER = 3  # Minimum rollouts to check
                        MAX_SAMPLES_PER_MINER = 20  # Maximum rollouts to check
                        SAMPLE_RATE = 0.1  # Check 10% of rollouts
                        FAILURE_THRESHOLD = 0.3  # Stop if >30% failures
                        BATCH_SIZE = 5  # Check in batches for early stopping

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
                            logger.info(
                                f"🔍 Verifying all {total_inferences} rollouts from {wallet_addr}"
                            )
                        else:
                            # For spot checking, sample complete GRPO groups
                            use_spot_check = True
                            indices_to_check = []

                            # Calculate how many groups to check
                            num_groups = len(groups_map)
                            groups_to_check = max(1, min(num_groups, int(num_groups * SAMPLE_RATE)))

                            # Randomly select groups
                            selected_groups = random.sample(
                                list(groups_map.keys()), groups_to_check
                            )

                            # Add all rollouts from selected groups
                            for group_id in selected_groups:
                                indices_to_check.extend(groups_map[group_id])

                            indices_to_check.sort()  # Process in order for better cache locality
                            logger.info(
                                f"📊 Spot checking {len(indices_to_check)}/{total_inferences} rollouts from {groups_to_check}/{num_groups} groups ({SAMPLE_RATE*100:.0f}% of groups)"
                            )

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
                                if (
                                    failure_rate > FAILURE_THRESHOLD
                                    and checked_count >= MIN_SAMPLES_PER_MINER
                                ):
                                    logger.warning(
                                        f"⚠️ Early stopping for {wallet_addr}: {failure_rate:.1%} failure rate after {checked_count} checks"
                                    )
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
                                required_fields = [
                                    "window_start",
                                    "nonce",
                                    "sat_seed",
                                    "block_hash",
                                    "commit",
                                    "proof",
                                    "challenge",
                                    "hotkey",
                                    "signature",
                                ]
                                if not all(field in inference for field in required_fields):
                                    logger.debug(
                                        f"Missing required fields in inference from {wallet_addr}"
                                    )
                                    continue

                                # Check window consistency
                                if inference["window_start"] != target_window:
                                    logger.debug(f"Window mismatch in inference from {wallet_addr}")
                                    continue

                                # Check block hash matches
                                if inference["block_hash"] != target_window_hash:
                                    logger.debug(
                                        f"Block hash mismatch in inference from {wallet_addr}"
                                    )
                                    continue

                                # Check nonce uniqueness within window
                                nonce = inference["nonce"]
                                if nonce in nonces_seen:
                                    logger.debug(
                                        f"Duplicate nonce {nonce} in window from {wallet_addr}"
                                    )
                                    continue
                                nonces_seen.add(nonce)

                                # Verify signature
                                if not verify_rollout_signature(inference):
                                    logger.debug(
                                        f"Invalid signature for inference from {wallet_addr}"
                                    )
                                    invalid_signatures += 1
                                    continue

                                # Verify SAT seed format
                                expected_seed = f"{wallet_addr}-{target_window_hash}-{nonce}"
                                if inference.get("sat_seed") != expected_seed:
                                    logger.debug(
                                        f"Invalid SAT seed in inference from {wallet_addr}: expected {expected_seed}, got {inference.get('sat_seed')}"
                                    )
                                    continue

                                # Verify GRAIL proof and SAT rollout
                                # We must verify ALL rollouts to ensure model identity
                                try:
                                    logger.debug(f"Verifying SAT rollout from {wallet_addr}")

                                    # For GRPO rollouts, we need to modify the commit data to use the base problem
                                    commit_data = inference["commit"]
                                    # Reward bounds check (pre-proof; fast filter)
                                    try:
                                        rollout_meta = commit_data.get("rollout", {})
                                        total_reward = rollout_meta.get("total_reward", None)
                                        if not isinstance(total_reward, (int, float)):
                                            logger.debug("Missing or invalid total_reward; skipping inference")
                                            continue
                                        if (
                                            total_reward < SAT_REWARD_LOW
                                            or total_reward > SAT_REWARD_HIGH
                                        ):
                                            logger.debug(
                                                "total_reward %.4f outside bounds [%.4f, %.4f]",
                                                total_reward,
                                                SAT_REWARD_LOW,
                                                SAT_REWARD_HIGH,
                                            )
                                            continue
                                    except Exception:
                                        pass
                                    rollout_group = inference.get("rollout_group")
                                    if rollout_group:
                                        # This is a GRPO rollout - regenerate base problem for verification
                                        base_sat_seed = (
                                            f"{wallet_addr}-{target_window_hash}-{rollout_group}"
                                        )
                                        base_problem = generate_sat_problem(
                                            base_sat_seed, inference.get("difficulty", 0.5)
                                        )
                                        # Update commit data with base problem for verification
                                        commit_data["sat_problem"]["seed"] = base_sat_seed
                                        # The verifier will regenerate the problem from this seed

                                    # Get challenge randomness from the inference data
                                    challenge_randomness = inference.get("randomness", "")
                                    
                                    # Use wallet address for signature verification (public key verification)
                                    if monitor:
                                        with monitor.timer("validation.rollout_verification"):
                                            is_valid = verifier.verify_rollout(
                                                commit_data, inference["proof"], wallet_addr,
                                                challenge_randomness=challenge_randomness
                                            )
                                    else:
                                        is_valid = verifier.verify_rollout(
                                            commit_data, inference["proof"], wallet_addr,
                                            challenge_randomness=challenge_randomness
                                        )
                                    
                                    total_rollouts_processed += 1
                                    if not is_valid:
                                        logger.warning(
                                            f"SAT rollout verification failed for {wallet_addr} - skipping"
                                        )
                                        invalid_proofs += 1
                                        continue
                                except Exception as e:
                                    logger.warning(
                                        f"Rollout verification error for {wallet_addr}: {e}"
                                    )
                                    continue

                                valid_count += 1

                                # Debug-only: log a small subset of validated texts with rewards
                                if logger.isEnabledFor(logging.DEBUG) and text_logs_emitted_by_wallet[wallet_addr] < TEXT_LOG_LIMIT_PER_WALLET:
                                    try:
                                        tokens = commit_data.get("tokens", [])
                                        if isinstance(tokens, list) and tokens:
                                            rollout_meta = commit_data.get("rollout", {})
                                            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
                                            completion_len = int(rollout_meta.get("completion_length", 0) or 0)

                                            # Select only the generated completion tokens
                                            if completion_len > 0 and prompt_len >= 0:
                                                completion_ids = tokens[prompt_len: prompt_len + completion_len]
                                            else:
                                                completion_ids = tokens[prompt_len:]

                                            text = verifier.tokenizer.decode(
                                                completion_ids, skip_special_tokens=False
                                            )
                                            reward_val = rollout_meta.get("total_reward", float("nan"))
                                            adv_val = rollout_meta.get("advantage", float("nan"))
                                            success_val = rollout_meta.get("success", False)
                                            logger.debug(
                                                "TEXT[validate] window=%s wallet=%s nonce=%s reward=%.3f adv=%.3f "
                                                "success=%s text=%s",
                                                target_window,
                                                wallet_addr,
                                                nonce,
                                                float(reward_val),
                                                float(adv_val),
                                                bool(success_val),
                                                text,
                                            )
                                            if monitor:
                                                await monitor.log_artifact(
                                                    f"validation/{wallet_addr}/sample_text",
                                                    {
                                                        "window": target_window,
                                                        "wallet": wallet_addr,
                                                        "nonce": nonce,
                                                        "reward": float(reward_val),
                                                        "advantage": float(adv_val),
                                                        "success": bool(success_val),
                                                        "text": text,
                                                    },
                                                    "text",
                                                )
                                            text_logs_emitted_by_wallet[wallet_addr] += 1
                                    except Exception:
                                        pass

                                # Track successful unique solutions
                                rollout = inference.get("commit", {}).get("rollout", {})
                                if rollout.get("success", False):
                                    successful_rollouts += 1
                                    # Create hash of solution for uniqueness
                                    assignment = rollout.get("assignment", [])
                                    solution_hash = hashlib.sha256(
                                        str(assignment).encode()
                                    ).hexdigest()
                                    unique_solutions.add(solution_hash)

                                # Add to collection of all valid rollouts
                                all_valid_rollouts.append(inference)

                            except Exception as e:
                                logger.debug(f"Error processing inference from {wallet_addr}: {e}")
                                processing_errors += 1
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
                                logger.debug(
                                    f"GRPO group {group_id} has only {len(group_rollouts)} rollouts in checked sample, may be incomplete due to spot-checking"
                                )
                                grpo_incomplete_groups += 1
                                continue

                            # Check if this looks like a complete group (should have 4 rollouts for GRPO)
                            expected_group_size = 4  # Standard GRPO uses 4 rollouts per problem
                            if len(group_rollouts) != expected_group_size:
                                logger.debug(
                                    f"GRPO group {group_id} has {len(group_rollouts)} rollouts, expected {expected_group_size}"
                                )

                            # Verify advantages sum to ~0 (GRPO property)
                            # NOTE: this need to change later on when we want to change the algo
                            advantages = []
                            for r in group_rollouts:
                                adv = r.get("commit", {}).get("rollout", {}).get("advantage", 0.0)
                                advantages.append(adv)

                            advantage_sum = sum(advantages)
                            if abs(advantage_sum) > 0.01:  # Allow small floating point errors
                                logger.debug(
                                    f"GRPO group {group_id} advantages don't sum to 0: {advantage_sum} (advantages: {advantages})"
                                )
                                grpo_invalid_groups += 1
                                continue

                            # Verify all rollouts in group have same base problem
                            # They should all have the same rollout_group and same base sat_problem seed
                            base_seeds = []
                            for r in group_rollouts:
                                sat_problem = r.get("commit", {}).get("sat_problem", {})
                                base_seeds.append(sat_problem.get("seed"))

                            if len(set(base_seeds)) != 1:
                                logger.debug(
                                    f"GRPO group {group_id} has different base problems: {set(base_seeds)}"
                                )
                                grpo_invalid_groups += 1
                                continue

                            logger.debug(
                                f"✅ GRPO group {group_id} verified: {len(group_rollouts)} rollouts, advantages sum to {advantage_sum:.6f}"
                            )
                            grpo_valid_groups += 1

                        if rollout_groups:
                            # Only report on groups we actually checked
                            total_groups_checked = (
                                grpo_valid_groups + grpo_invalid_groups + grpo_incomplete_groups
                            )
                            if total_groups_checked > 0:
                                logger.info(
                                    f"GRPO groups checked: {grpo_valid_groups} valid, {grpo_invalid_groups} invalid, {grpo_incomplete_groups} incomplete (spot-check artifact)"
                                )

                        # Calculate estimated total valid rollouts based on sampling
                        if should_stop:
                            # If we stopped early due to failures, assume 0 valid rollouts
                            estimated_valid = 0
                        else:
                            # Extrapolate from sample to estimate total
                            sample_pass_rate = (
                                valid_count / checked_count if checked_count > 0 else 0
                            )
                            estimated_valid = int(total_inferences * sample_pass_rate)

                        # Store metrics for this miner
                        window_inference_counts[wallet_addr] = {
                            "valid": valid_count,
                            "checked": checked_count,
                            "total": total_inferences,
                            "estimated_valid": estimated_valid,
                            "successful": successful_rollouts,
                            "unique": len(unique_solutions),
                        }
                        total_valid_rollouts += estimated_valid  # Use estimated for rewards

                        logger.info(
                            f"✅ {wallet_addr}: {valid_count}/{checked_count} checked, ~{estimated_valid}/{total_inferences} estimated valid, {successful_rollouts} successful, {len(unique_solutions)} unique"
                        )

                    except Exception as e:
                        logger.warning(f"Error processing window file {filename}: {e}")
                        continue

                logger.info(
                    f"📁 Found {files_found} window files from {len(meta.hotkeys)} active hotkeys"
                )
                logger.info(
                    f"🏁 Total valid rollouts in window {target_window}: {total_valid_rollouts}"
                )
                
                # Log validation metrics with block context already set
                if monitor:
                    await monitor.log_counter("validation.windows_processed")
                    await monitor.log_gauge("validation.total_rollouts_processed", total_rollouts_processed)
                    await monitor.log_gauge("validation.valid_rollouts", total_valid_rollouts)
                    await monitor.log_gauge("validation.invalid_signatures", invalid_signatures)
                    await monitor.log_gauge("validation.invalid_proofs", invalid_proofs)
                    await monitor.log_gauge("validation.processing_errors", processing_errors)
                    await monitor.log_gauge("validation.files_found", files_found)
                    
                    if total_rollouts_processed > 0:
                        validation_success_rate = total_valid_rollouts / total_rollouts_processed
                        await monitor.log_gauge("validation.success_rate", validation_success_rate)

                # Upload all valid rollouts for training and to Hugging Face
                if all_valid_rollouts:
                    # Upload to S3/R2 for immediate access
                    upload_success = await upload_valid_rollouts(target_window, all_valid_rollouts, credentials)
                    if upload_success:
                        logger.info(
                            f"📤 Uploaded {len(all_valid_rollouts)} valid rollouts for training"
                        )
                    else:
                        logger.warning("⚠️ Failed to upload valid rollouts for training")

                    # NEW: Upload to Hugging Face dataset for community access
                    try:
                        hf_success = await upload_to_huggingface(
                            all_valid_rollouts, target_window, PROTOCOL_VERSION
                        )
                        if hf_success:
                            logger.info(
                                "🤗 Uploaded {} rollouts to Hugging Face dataset".format(
                                    len(all_valid_rollouts)
                                )
                            )
                        else:
                            logger.debug("Failed to upload to Hugging Face (may need HF_TOKEN)")
                    except Exception as e:
                        logger.debug(f"Hugging Face upload error: {e}")

                # Update global inference counts for weight calculation
                for hotkey, metrics in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = metrics

                # Compute weights based on unique successful rollouts
                raw_scores = []
                for uid, hotkey in enumerate(meta.hotkeys):
                    # Calculate score over last 3 windows
                    recent_windows = range(
                        max(0, target_window - 2 * WINDOW_LENGTH), target_window + 1, WINDOW_LENGTH
                    )

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
                    # Base performance score in [0, 1]
                    unique_score = min(1.0, total_unique / 10.0) if total_unique > 0 else 0
                    success_score = min(1.0, total_successful / 20.0) if total_successful > 0 else 0
                    valid_score = min(1.0, total_valid / 50.0) if total_valid > 0 else 0

                    base_score = 0.6 * unique_score + 0.0 * success_score + 0.4 * valid_score
                    base_score = max(0.0, min(1.0, base_score))

                    # Apply superlinear curve: emphasizes higher performers and penalizes splitting
                    superlinear_score = base_score**SUPERLINEAR_EXPONENT
                    raw_scores.append(superlinear_score)

                # Normalize weights
                denom = math.fsum(raw_scores)
                if denom > 0.0:
                    weights = [score / denom for score in raw_scores]
                else:
                    weights = [0.0] * len(meta.hotkeys)

                # Log non-zero weights
                non_zero_weights = [
                    (meta.hotkeys[i], weights[i]) for i in range(len(weights)) if weights[i] > 0
                ]
                if non_zero_weights:
                    logger.info(f"⚖️  Setting weights for {len(non_zero_weights)} miners")
                    for hotkey, weight in non_zero_weights[:5]:  # Show top 5
                        logger.info(f"   {hotkey}: {weight:.4f}")
                else:
                    logger.info("⚖️  No miners received weights this window")
                
                # Log weight distribution metrics
                if monitor:
                    await monitor.log_gauge("validation.miners_with_weights", len(non_zero_weights))
                    await monitor.log_gauge("validation.total_miners", len(meta.hotkeys))
                    if weights:
                        max_weight = max(weights)
                        avg_weight = sum(weights) / len(weights)
                        await monitor.log_gauge("validation.max_weight", max_weight)
                        await monitor.log_gauge("validation.average_weight", avg_weight)

                # Set weights on network
                await subtensor.set_weights(
                    wallet=wallet,
                    netuid=NETUID,
                    uids=meta.uids,
                    weights=weights,
                    wait_for_inclusion=False,
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

    async def _main() -> None:
        await asyncio.gather(_run(), watchdog(timeout=(60 * 10)))

    asyncio.run(_main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    validate()
