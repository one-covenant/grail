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
import json
import traceback
import math
import bittensor as bt
from collections import defaultdict
from typing import Any, Tuple, Optional, DefaultDict, Dict, List

from grail.infrastructure.drand import get_round_at_time, get_drand_beacon
from types import SimpleNamespace

from ..grail import Verifier
from . import console

from ..environments import create_sat_reward_vector
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
from ..shared.constants import NETUID, WINDOW_LENGTH, MODEL_NAME, SUPERLINEAR_EXPONENT, ROLLOUTS_PER_PROBLEM
from ..monitoring import get_monitoring_manager
from ..monitoring.config import MonitoringConfig

# --------------------------------------------------------------------------- #
#                  Future Training/Checkpoint Integration (commented)        #
# --------------------------------------------------------------------------- #
#
# The following helpers outline the intended interfaces for training-time
# model checkpoint management. Keep these commented until training is enabled.
# They are structured to minimize coupling with the validation flow and allow
# re-use across CLI modules.
#
# async def _model_state_exists(hotkey: str, window_start: int) -> bool:
#     """Return True if a model checkpoint exists for hotkey at window_start."""
#     # Implementation will likely check object storage (e.g., R2/S3) for
#     # model artifacts following a consistent naming scheme.
#     # Example path: f"models/{hotkey}/{window_start}/model.safetensors"
#     # Include timeouts and retries.
#     raise NotImplementedError
#
# async def _load_model_state(model: Any, hotkey: str, window_start: int) -> bool:
#     """Load weights into `model` for the given hotkey and window_start."""
#     # Use safetensors to load weights to avoid format drift and improve safety.
#     # Example:
#     #   path = await _download_model_artifact(hotkey, window_start)
#     #   state = safetensors.torch.load_file(path)
#     #   model.load_state_dict(state, strict=False)
#     # Return True on success, False otherwise.
#     raise NotImplementedError


# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #

logger = logging.getLogger("grail")

# --------------------------------------------------------------------------- #
#                       Styling & configuration constants                     #
# --------------------------------------------------------------------------- #
# Sampling and validation parameters. Keep these centralized to avoid magic
# numbers scattered through validation logic and to make tuning straightforward.
MAX_SAMPLES_PER_MINER = 20                # If <= this many rollouts, check all
SAMPLE_RATE = 0.1                          # Fraction of GRPO groups to spot-check
STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.05  # Soft-failure fraction to gate wallet
REWARD_REL_TOL = 0.02                      # Relative tolerance on reward bounds
REWARD_ABS_TOL = 1e-6                      # Absolute tolerance on reward bounds
GRPO_ADV_SUM_TOLERANCE = 0.01              # Sum of advantages should be ~0
DEBUG_TEXT_LOG_LIMIT_PER_WALLET = 5        # Max sample texts logged per wallet
SOFT_CHECK_KEY = "token_distribution_valid"  # Soft heuristic key from verifier
HARD_CHECK_KEYS = (                         # Hard checks required for validity
    "tokens_valid",
    "proof_valid",
    "sat_problem_valid",
    "termination_valid",
    "solution_valid",
)

# Submit weights to chain at most once per this many blocks
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 360

# Number of windows to include when computing rolling weights
WEIGHT_ROLLING_WINDOWS = 12

# Number of miners to log in detail on submission
#TODO: reduce this later
TOP_K_WEIGHTS_LOGGED = 256

# Required fields for SAT rollouts validation
REQUIRED_ROLLOUT_FIELDS = [
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

# --------------------------------------------------------------------------- #
#                             Utility helpers                                 #
# --------------------------------------------------------------------------- #


def get_conf(key: str, default: Any = None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(
            f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>"
        )
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
    filename: str
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
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
    filename: str
) -> Tuple[Optional[str], Optional[int]]:
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
        result = keypair.verify(
            data=challenge, signature=bytes.fromhex(signature)
        )
        return bool(result)
    except Exception:
        return False


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
            logging.error(
                f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process."
            )
            os._exit(1)


# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #

def validate(
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Include drand in challenge randomness (default: True)",
        show_default=True,
    ),
    test_mode: bool = typer.Option(
        False,
        "--test-mode/--no-test-mode",
        help="Test mode: validate only own files (default: False)",
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

    # Declarative SAT reward bounds (composed from per-function bounds)
    SAT_REWARD_LOW, SAT_REWARD_HIGH = _get_sat_reward_bounds()

    # Run service
    asyncio.run(
        _run_validation_service(
            wallet=wallet,
            verifier=verifier,
            sat_reward_low=SAT_REWARD_LOW,
            sat_reward_high=SAT_REWARD_HIGH,
            use_drand=use_drand,
            test_mode=test_mode,
        )
    )


# ----------------------------- Refactored Helpers ---------------------------- #


def _get_sat_reward_bounds() -> Tuple[float, float]:
    """Return SAT reward bounds or permissive defaults on failure."""
    try:
        _sat_rv = create_sat_reward_vector()
        low, high = _sat_rv.reward_bounds()
        return float(low), float(high)
    except Exception:
        return float("-inf"), float("inf")


async def _run_validation_service(
    wallet: bt.wallet,
    verifier: "Verifier",
    sat_reward_low: float,
    sat_reward_high: float,
    use_drand: bool,
    test_mode: bool,
) -> None:
    """Run validation service: main loop + watchdog.

    Args:
        wallet: Bittensor wallet used for signing and network ops.
        verifier: GRAIL verifier instance with model/tokenizer loaded.
        sat_reward_low: Lower bound for SAT rollout reward sanity checks.
        sat_reward_high: Upper bound for SAT rollout reward sanity checks.
        use_drand: Whether to incorporate drand randomness in challenges.
        test_mode: If True, validate only the local hotkey (developer testing).

    This function manages lifecycle and orchestration. It delegates window/work
    processing to helpers and keeps the outer loop small and readable.
    """
    async def _validation_loop() -> None:
        subtensor = None
        credentials, chain_manager = await _initialize_credentials_and_chain(wallet)
        monitor = await _initialize_monitor(wallet)
        # Rolling window metrics per hotkey, keyed by window_start -> metric dict
        # Metrics include: valid, checked, total, estimated_valid, successful, unique
        inference_counts: DefaultDict[
            str, DefaultDict[int, Dict[str, int]]
        ] = defaultdict(lambda: defaultdict(dict))
        last_processed_window = -1
        last_weights_interval_submitted = -1
        
        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
                
                if subtensor is None:
                    subtensor = await get_subtensor()
                
                meta = await subtensor.metagraph(NETUID)
                # Build hotkey -> uid mapping for per-miner logging namespaces
                uid_by_hotkey = {hk: uid for hk, uid in zip(meta.hotkeys, meta.uids)}
                current_block = await subtensor.get_current_block()
                target_window = _determine_target_window(current_block)
                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(5)
                    logger.debug(f"Waiting for new window {target_window}")
                    continue
                
                if monitor:
                    monitor.set_block_context(current_block, target_window)
                # ------------------------------------------------------------------ #
                # Training integration (commented for now):
                #
                # If training is enabled, validators may want to load the miner's
                # model state for the target window before verification.
                # This keeps validation consistent with the model used to produce
                # rollouts.
                #
                # Example flow:
                # - Wait for model checkpoint to exist for this window
                # - Load model state into verifier.model
                # - Set model to eval mode
                #
                # Uncomment when training modules are wired:
                #
                # available = await _model_state_exists(
                #     hotkey=wallet.hotkey.ss58_address,
                #     window_start=target_window,
                # )
                # if available:
                #     loaded = await _load_model_state(
                #         model=verifier.model,
                #         hotkey=wallet.hotkey.ss58_address,
                #         window_start=target_window,
                #     )
                #     if loaded:
                #         logger.info(
                #             "✅ Loaded model state for window %s", target_window
                #         )
                #         verifier.model.eval()
                #     else:
                #         logger.warning(
                #             "⚠️ Failed to load model state; using base model"
                #         )
                # else:
                #     logger.debug(
                #         "No model checkpoint available for window %s", target_window
                #     )
                # Using base model directly without waiting for state
                logger.info("🚀 Using base model for verification")
                
                target_window_hash = await subtensor.get_block_hash(target_window)
                # Single per-window randomness value reused across all checks
                window_rand = _compute_window_randomness(
                    target_window_hash, use_drand
                )
                hotkeys_to_check = _determine_hotkeys_to_check(
                    test_mode, wallet, meta
                )
                (
                    window_inference_counts,
                    total_valid_rollouts,
                    total_rollouts_processed,
                    invalid_signatures,
                    invalid_proofs,
                    processing_errors,
                    files_found,
                    all_valid_rollouts,
                ) = await _process_window(
                    hotkeys_to_check=hotkeys_to_check,
                    target_window=target_window,
                    target_window_hash=target_window_hash,
                    window_rand=window_rand,
                    wallet=wallet,
                    verifier=verifier,
                    credentials=credentials,
                    chain_manager=chain_manager,
                    monitor=monitor,
                    uid_by_hotkey=uid_by_hotkey,
                    sat_reward_low=sat_reward_low,
                    sat_reward_high=sat_reward_high,
                )
                
                # Log summary
                logger.info(
                    f"📁 Found {files_found} window files from "
                    f"{len(meta.hotkeys)} active hotkeys"
                )
                logger.info(
                    f"🏁 Total valid rollouts in window {target_window}: "
                    f"{total_valid_rollouts}"
                )
                
                # Monitoring metrics
                if monitor:
                    await monitor.log_counter("validation.windows_processed")
                    await monitor.log_gauge(
                        "validation.total_rollouts_processed", total_rollouts_processed
                    )
                    await monitor.log_gauge(
                        "validation.valid_rollouts", total_valid_rollouts
                    )
                    await monitor.log_gauge(
                        "validation.invalid_signatures", invalid_signatures
                    )
                    await monitor.log_gauge(
                        "validation.invalid_proofs", invalid_proofs
                    )
                    await monitor.log_gauge(
                        "validation.processing_errors", processing_errors
                    )
                    await monitor.log_gauge("validation.files_found", files_found)
                    if total_rollouts_processed > 0:
                        success_rate = (
                            total_valid_rollouts / total_rollouts_processed
                        )
                        await monitor.log_gauge(
                            "validation.success_rate", success_rate
                        )
                
                # Uploads
                if all_valid_rollouts:
                    await _upload_rollouts(
                        target_window=target_window,
                        all_valid_rollouts=all_valid_rollouts,
                        credentials=credentials,
                    )
                
                # Update inference counts
                for hotkey, metrics in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = metrics
                
                # Compute and set weights
                weights, non_zero_weights = _compute_weights(
                    meta_hotkeys=meta.hotkeys,
                    inference_counts=inference_counts,
                    target_window=target_window,
                )
                if non_zero_weights:
                    logger.info(
                        f"⚖️  Setting weights for {len(non_zero_weights)} miners"
                    )
                    for hotkey, weight in non_zero_weights[:5]:
                        logger.info(f"   {hotkey}: {weight:.4f}")
                else:
                    logger.info("⚖️  No miners received weights this window")
                
                if monitor:
                    await monitor.log_gauge(
                        "validation.miners_with_weights", len(non_zero_weights)
                    )
                    await monitor.log_gauge(
                        "validation.total_miners", len(meta.hotkeys)
                    )
                    if weights:
                        max_weight = max(weights)
                        avg_weight = sum(weights) / len(weights)
                        await monitor.log_gauge(
                            "validation.max_weight", max_weight
                        )
                        await monitor.log_gauge(
                            "validation.average_weight", avg_weight
                        )
                # Global submission context (per loop)
                if monitor:
                    await monitor.log_gauge(
                        "weights/submission/interval_blocks",
                        WEIGHT_SUBMISSION_INTERVAL_BLOCKS,
                    )
                    await monitor.log_gauge(
                        "weights/submission/current_block",
                        current_block,
                    )
                    await monitor.log_gauge(
                        "weights/submission/target_window",
                        target_window,
                    )
                    await monitor.log_gauge(
                        "weights/config/rolling_windows",
                        WEIGHT_ROLLING_WINDOWS,
                    )
                    await monitor.log_gauge(
                        "weights/config/superlinear_exponent",
                        SUPERLINEAR_EXPONENT,
                    )
                logger.debug(
                    "Weights context prepared: interval_blocks=%s block=%s window=%s rolling=%s superlinear=%s",
                    WEIGHT_SUBMISSION_INTERVAL_BLOCKS,
                    current_block,
                    target_window,
                    WEIGHT_ROLLING_WINDOWS,
                    SUPERLINEAR_EXPONENT,
                )
                
                # Throttle on-chain weight submissions to once per 360-block interval
                current_interval = int(
                    current_block // WEIGHT_SUBMISSION_INTERVAL_BLOCKS
                )
                if current_interval != last_weights_interval_submitted:
                    if not non_zero_weights:
                        logger.debug(
                            "Skipping weight submission: all weights are zero"
                        )
                    else:
                        # Precompute top miners for per-miner logging on submission
                        top_miners = _build_top_miners(
                            meta.hotkeys, meta.uids, weights, TOP_K_WEIGHTS_LOGGED
                        )
                        await subtensor.set_weights(
                            wallet=wallet,
                            netuid=NETUID,
                            uids=meta.uids,
                            weights=weights,
                            wait_for_inclusion=False,
                        )
                        last_weights_interval_submitted = current_interval
                        if monitor:
                            await monitor.log_gauge(
                                "weights/submission/submitted", 1.0
                            )
                            await monitor.log_gauge(
                                "weights/submission/interval_index",
                                current_interval,
                            )
                        logger.info(
                            "Submitted weights: interval=%s block=%s miners_with_weights=%s total_miners=%s rolling=%s superlinear=%s",
                            current_interval,
                            current_block,
                            len(non_zero_weights),
                            len(meta.hotkeys),
                            WEIGHT_ROLLING_WINDOWS,
                            SUPERLINEAR_EXPONENT,
                        )
                        # Log detailed per-miner metrics for top-K on submission
                        if monitor:
                            for hk, uid, w in top_miners:
                                tu, ts, tev, b, s = _aggregate_weight_inputs(
                                    hk, inference_counts, target_window
                                )
                                uid_ns = f"{uid}/weights"
                                await monitor.log_gauge(f"{uid_ns}/weight", w)
                                await monitor.log_gauge(f"{uid_ns}/base_score", b)
                                await monitor.log_gauge(
                                    f"{uid_ns}/superlinear_score", s
                                )
                                await monitor.log_gauge(
                                    f"{uid_ns}/inputs/unique_rolling", tu
                                )
                                await monitor.log_gauge(
                                    f"{uid_ns}/inputs/successful_rolling", ts
                                )
                                await monitor.log_gauge(
                                    f"{uid_ns}/inputs/estimated_valid_rolling",
                                    tev,
                                )
                        for hk, uid, w in top_miners:
                            tu, ts, tev, b, s = _aggregate_weight_inputs(
                                hk, inference_counts, target_window
                            )
                            logger.info(
                                "Weight uid=%s hotkey=%s weight=%.6f base=%.6f unique_rolling=%d successful_rolling=%d estimated_valid_rolling=%d",
                                uid,
                                hk,
                                w,
                                b,
                                tu,
                                ts,
                                tev,
                            )
                        logger.info(
                            "Submitted weights (interval %s, block %s)",
                            current_interval,
                            current_block,
                        )
                else:
                    if monitor:
                        await monitor.log_gauge(
                            "weights/submission/submitted", 0.0
                        )
                        await monitor.log_gauge(
                            "weights/submission/interval_index", current_interval
                        )
                    logger.debug(
                        "Skipping weight submission (360-block throttle). "
                        "interval=%s, last_submitted=%s",
                        current_interval,
                        last_weights_interval_submitted,
                    )
                last_processed_window = target_window
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                traceback.print_exc()
                logger.info(f"Error in validator loop: {e}. Continuing ...")
                subtensor = None
                await asyncio.sleep(10)
                continue

    await asyncio.gather(
        _validation_loop(), watchdog(timeout=(60 * 10))
    )


async def _initialize_credentials_and_chain(
    wallet: bt.wallet
) -> Tuple[Any, GrailChainManager]:
    """Load storage credentials and initialize chain manager.

    Returns:
        Tuple (credentials, chain_manager): object storage credentials for R2/S3
        and an initialized `GrailChainManager` that has committed read creds.
    """
    try:
        credentials = load_r2_credentials()
        logger.info("✅ Loaded R2 credentials")
    except Exception as e:
        logger.error(f"Failed to load R2 credentials: {e}")
        raise
    
    config = SimpleNamespace(netuid=NETUID)
    chain_manager = GrailChainManager(config, wallet, credentials)
    await chain_manager.initialize()
    logger.info(
        "✅ Initialized chain manager and committed read credentials"
    )
    
    return credentials, chain_manager


async def _initialize_monitor(wallet: bt.wallet) -> Any:
    """Initialize monitoring run for validation, if configured.

    Returns:
        Monitoring client or None if monitoring is disabled.
    """
    monitor = get_monitoring_manager()
    if monitor:
        validation_config = MonitoringConfig.for_validation(wallet.name)
        run_id = await monitor.start_run(
            f"validation_{wallet.name}",
            validation_config.get("hyperparameters", {})
        )
        logger.info(f"Started monitoring run: {run_id}")
    
    return monitor


def _determine_target_window(current_block: int) -> int:
    """Compute the last fully completed window start from current block."""
    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
    return current_window - WINDOW_LENGTH


def _compute_window_randomness(
    target_window_hash: str, use_drand: bool
) -> str:
    """Derive deterministic per-window randomness, optionally including drand."""
    if use_drand:
        try:
            drand_round = get_round_at_time(int(time.time()))
            drand_beacon = get_drand_beacon(drand_round)
            return hashlib.sha256(
                (target_window_hash + drand_beacon["randomness"]).encode()
            ).hexdigest()
        except Exception:
            return hashlib.sha256(
                target_window_hash.encode()
            ).hexdigest()
    
    return hashlib.sha256(
        target_window_hash.encode()
    ).hexdigest()


def _determine_hotkeys_to_check(
    test_mode: bool, wallet: bt.wallet, meta: Any
) -> List[str]:
    """Choose which hotkeys to validate based on test/prod mode."""
    if test_mode:
        logger.info(
            f"🧪 TEST MODE: Checking files for own hotkey "
            f"{wallet.hotkey.ss58_address}"
        )
        return [wallet.hotkey.ss58_address]
    
    logger.info(f"Checking files for {len(meta.hotkeys)} active hotkeys")
    logger.info(f"Active hotkeys: {meta.hotkeys}")
    return list(meta.hotkeys)


def _aggregate_weight_inputs(
    hotkey: str,
    inference_counts: DefaultDict[str, DefaultDict[int, Dict[str, int]]],
    target_window: int,
) -> Tuple[int, int, int, float, float]:
    """Aggregate rolling inputs and derive base/superlinear scores.

    Returns: (unique_sum, successful_sum, estimated_valid_sum,
              base_score, superlinear_score)
    """
    recent_windows = range(
        max(0, target_window - (WEIGHT_ROLLING_WINDOWS - 1) * WINDOW_LENGTH),
        target_window + 1,
        WINDOW_LENGTH,
    )
    total_unique = 0
    total_successful = 0
    total_estimated_valid = 0
    for w in recent_windows:
        metrics = inference_counts[hotkey].get(w, {})
        total_unique += int(metrics.get("unique", 0))
        total_successful += int(metrics.get("successful", 0))
        total_estimated_valid += int(metrics.get("estimated_valid", 0))
    unique_score = (
        min(1.0, total_unique / 10.0) if total_unique > 0 else 0.0
    )
    # NOTE: at this stage we only give weights to unique scores
    base_score = max(0.0, min(1.0, 1.0 * unique_score + 0.0 * 0.0 + 0.0 * 0.0))
    superlinear_score = base_score**SUPERLINEAR_EXPONENT
    return (
        total_unique,
        total_successful,
        total_estimated_valid,
        base_score,
        superlinear_score,
    )


def _build_top_miners(
    hotkeys: List[str], uids: List[int], weights: List[float], k: int
) -> List[Tuple[str, int, float]]:
    """Return top-k (hotkey, uid, weight) sorted by weight desc."""
    pairs = [
        (hk, uid, float(weights[i]))
        for i, (hk, uid) in enumerate(zip(hotkeys, uids))
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


async def _process_window(
    hotkeys_to_check: List[str],
    target_window: int,
    target_window_hash: str,
    window_rand: str,
    wallet: bt.wallet,
    verifier: "Verifier",
    credentials: Any,
    chain_manager: GrailChainManager,
    monitor: Any,
    uid_by_hotkey: Dict[str, int],
    sat_reward_low: float,
    sat_reward_high: float,
) -> Tuple[Dict[str, Dict[str, int]], int, int, int, int, int, int, List[dict]]:
    """Process a window across hotkeys and aggregate metrics/results."""
    total_valid_rollouts = 0
    window_inference_counts: Dict[str, Dict[str, int]] = {}
    files_found = 0
    all_valid_rollouts: List[dict] = []
    # Limit how many sample texts we log per wallet (debug noise control)
    text_logs_emitted_by_wallet: DefaultDict[str, int] = defaultdict(int)
    
    total_rollouts_processed = 0
    invalid_signatures = 0
    invalid_proofs = 0
    processing_errors = 0
    
    for wallet_addr in hotkeys_to_check:
        try:
            (
                found_file,
                metrics,
                published_rollouts,
                processed_counts,
            ) = await _process_wallet_window(
                wallet_addr=wallet_addr,
                target_window=target_window,
                target_window_hash=target_window_hash,
                window_rand=window_rand,
                wallet=wallet,
                verifier=verifier,
                credentials=credentials,
                chain_manager=chain_manager,
                monitor=monitor,
                uid_by_hotkey=uid_by_hotkey,
                text_logs_emitted_by_wallet=text_logs_emitted_by_wallet,
                text_log_limit=DEBUG_TEXT_LOG_LIMIT_PER_WALLET,
                sat_reward_low=sat_reward_low,
                sat_reward_high=sat_reward_high,
            )
            if found_file:
                files_found += 1
            if metrics is not None:
                window_inference_counts[wallet_addr] = metrics
            if published_rollouts:
                all_valid_rollouts.extend(published_rollouts)
            (
                pr_total,
                pr_invalid_sig,
                pr_invalid_proof,
                pr_processing_err
            ) = processed_counts
            total_rollouts_processed += pr_total
            invalid_signatures += pr_invalid_sig
            invalid_proofs += pr_invalid_proof
            processing_errors += pr_processing_err
        except Exception as e:
            logger.warning(f"Error processing wallet {wallet_addr}: {e}")
            continue
    
    for metrics in window_inference_counts.values():
        total_valid_rollouts += metrics.get("estimated_valid", 0)
    
    return (
        window_inference_counts,
        total_valid_rollouts,
        total_rollouts_processed,
        invalid_signatures,
        invalid_proofs,
        processing_errors,
        files_found,
        all_valid_rollouts,
    )


async def _process_wallet_window(
    wallet_addr: str,
    target_window: int,
    target_window_hash: str,
    window_rand: str,
    wallet: bt.wallet,
    verifier: "Verifier",
    credentials: Any,
    chain_manager: GrailChainManager,
    monitor: Any,
    uid_by_hotkey: Dict[str, int],
    text_logs_emitted_by_wallet: DefaultDict[str, int],
    text_log_limit: int,
    sat_reward_low: float,
    sat_reward_high: float,
) -> Tuple[bool, Optional[Dict[str, int]], List[dict], Tuple[int, int, int, int]]:
    """Validate a single wallet window file and return metrics and rollouts."""
    filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"
    miner_bucket = chain_manager.get_bucket_for_hotkey(wallet_addr)
    exists = await file_exists(
        filename,
        credentials=miner_bucket if miner_bucket else credentials,
        use_write=False,
    )
    if not exists:
        logger.debug(f"No file found for {wallet_addr} at {filename}")
        return False, None, [], (0, 0, 0, 0)
    
    logger.info(f"📁 Found file for hotkey {wallet_addr}")
    # Resolve miner UID (fallback to wallet address string)
    uid_str = str(uid_by_hotkey.get(wallet_addr, wallet_addr))
    window_data = await get_file(
        filename,
        credentials=miner_bucket if miner_bucket else credentials,
        use_write=False
    )
    if not window_data:
        logger.warning(f"Could not download {filename}")
        return True, None, [], (0, 0, 0, 0)
    
    file_wallet_addr = window_data.get("wallet")
    window_start = window_data.get("window_start")
    inferences = window_data.get("inferences", [])
    if file_wallet_addr != wallet_addr:
        logger.warning(
            f"Wallet mismatch in {filename}: expected {wallet_addr}, "
            f"got {file_wallet_addr}"
        )
        return True, None, [], (0, 0, 0, 0)
    if window_start != target_window:
        logger.warning(
            f"Window mismatch in {filename}: expected {target_window}, "
            f"got {window_start}"
        )
        return True, None, [], (0, 0, 0, 0)
    
    total_inferences = len(inferences)
    groups_map = defaultdict(list)
    for idx, inf in enumerate(inferences):
        group_id = inf.get("rollout_group")
        if group_id is not None:
            groups_map[group_id].append(idx)
        else:
            groups_map[f"single_{idx}"] = [idx]
    
    # Determine whether to check all rollouts or sample GRPO groups.
    # Sampling is deterministic per wallet+window via a seeded RNG to keep
    # validators consistent and discourage gaming.
    if total_inferences <= MAX_SAMPLES_PER_MINER:
        indices_to_check = list(range(total_inferences))
        logger.info(
            f"🔍 Verifying all {total_inferences} rollouts from {wallet_addr}"
        )
    else:
        indices_to_check = []
        num_groups = len(groups_map)
        
        # Choose how many GRPO groups to spot-check (at least 1, at most all).
        groups_to_check = max(
            1, min(num_groups, int(num_groups * SAMPLE_RATE))
        )
        
        # Derive a deterministic RNG seed from miner wallet,
        # window randomness, and validator hotkey.
        # Selection is stable per validator and window, and
        # harder to game via order-dependent behavior.
        # We take the first 8 bytes of the SHA-256 to get
        # a 64-bit integer seed.
        seed_material = (
            f"{wallet_addr}:{window_rand}:{wallet.hotkey.ss58_address}"
        ).encode()
        seed_int = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8], "big"
        )
        rng = random.Random(seed_int)

        def _group_digest(gidxs: list[int]) -> str:
            # Compute a stable digest per group from the canonical
            # JSON of each rollout's commit. Indices are sorted so
            # the digest is independent of original order.
            # Using sort_keys and compact separators yields a
            # canonical JSON, so identical content produces the
            # same digest across validators.
            dig = hashlib.sha256()
            for i in sorted(gidxs):
                commit_json = json.dumps(
                    inferences[i].get("commit", {}),
                    sort_keys=True,
                    separators=(",", ":")
                )
                dig.update(hashlib.sha256(commit_json.encode()).digest())
            return dig.hexdigest()
            
        # Canonicalize group ordering by sorting ids by their
        # content-derived digest. This avoids dict insertion
        # order or arbitrary ids affecting RNG sampling.
        group_keys = sorted(
            list(groups_map.keys()),
            key=lambda gid: _group_digest(groups_map[gid])
        )
        # Deterministically sample groups without replacement using the seeded RNG.
        selected_groups = rng.sample(group_keys, groups_to_check)
        for group_id in selected_groups:
            indices_to_check.extend(groups_map[group_id])
        # Sort indices to make the per-rollout verification order deterministic.
        indices_to_check.sort()
        logger.info(
            f"📊 Spot checking {len(indices_to_check)}/{total_inferences} "
            f"rollouts from {groups_to_check}/{num_groups} groups "
            f"({SAMPLE_RATE*100:.0f}% of groups)"
        )
    
    # Per-wallet counters for metrics and gating
    valid_count = 0
    checked_count = 0
    successful_rollouts = 0
    unique_solutions = set()
    nonces_seen = set()
    rollout_groups = defaultdict(list)
    wallet_rollouts_buffer = []
    soft_failures = 0
    hard_failure = False
    soft_gate_triggered = False
    total_planned_checks = len(indices_to_check)
    
    # Compute soft failure threshold for wallet gating
    soft_fail_cutoff = max(
        1,
        math.ceil(
            STOCHASTIC_CHECK_FAILURE_THRESHOLD * max(1, total_planned_checks)
        )
    )
    
    # Per-wallet processing counters
    pr_total = 0
    pr_invalid_sig = 0
    pr_invalid_proof = 0
    pr_processing_err = 0
    
    for _, inference_idx in enumerate(indices_to_check):
        inference = inferences[inference_idx]
        checked_count += 1
        rollout_group = inference.get("rollout_group")
        if rollout_group:
            rollout_groups[rollout_group].append(inference)
        try:
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
                hard_failure = True
                logger.warning(
                    f"Missing required fields in inference from {wallet_addr}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            if inference["window_start"] != target_window:
                hard_failure = True
                logger.warning(
                    f"Window mismatch in inference from {wallet_addr}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            if inference["block_hash"] != target_window_hash:
                hard_failure = True
                logger.warning(
                    f"Block hash mismatch in inference from {wallet_addr}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            nonce = inference["nonce"]
            if nonce in nonces_seen:
                hard_failure = True
                logger.warning(
                    f"Duplicate nonce {nonce} in window from {wallet_addr}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            nonces_seen.add(nonce)
            if not verify_rollout_signature(inference):
                pr_invalid_sig += 1
                hard_failure = True
                logger.warning(
                    f"Invalid signature for {wallet_addr}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            expected_seed = f"{wallet_addr}-{target_window_hash}-{nonce}"
            if inference.get("sat_seed") != expected_seed:
                hard_failure = True
                logger.warning(
                    f"Invalid SAT seed in inference from {wallet_addr}: "
                    f"expected {expected_seed}, got {inference.get('sat_seed')}; "
                    f"invalidating wallet for window {target_window}"
                )
                break
            try:
                commit_data = inference["commit"]
                try:
                    rollout_meta = commit_data.get("rollout", {})
                    total_reward = rollout_meta.get("total_reward", None)
                    if not isinstance(total_reward, (int, float)):
                        logger.debug(
                            "Missing or invalid total_reward; skipping inference"
                        )
                        continue
                    low = float(sat_reward_low)
                    high = float(sat_reward_high)
                    tr = float(total_reward)
                    lo = (
                        float("-inf")
                        if low == float("-inf")
                        else low - max(
                            abs(low) * REWARD_REL_TOL, REWARD_ABS_TOL
                        )
                    )
                    hi = (
                        float("inf")
                        if high == float("inf")
                        else high + max(
                            abs(high) * REWARD_REL_TOL, REWARD_ABS_TOL
                        )
                    )
                    if not (lo <= tr <= hi):
                        hard_failure = True
                        logger.warning(
                            f"Reward {tr:.6f} outside tolerant bounds "
                            f"[{lo:.6f}, {hi:.6f}] (base=[{low:.6f}, {high:.6f}]); "
                            f"invalidating wallet for window {target_window}"
                        )
                        break
                except Exception:
                    pass
                
                if rollout_group:
                    base_sat_seed = (
                        f"{wallet_addr}-{target_window_hash}-{rollout_group}"
                    )
                    commit_data.setdefault("sat_problem", {})["seed"] = base_sat_seed
                
                challenge_rand = window_rand
                if monitor:
                    with monitor.timer("validation.rollout_verification"):
                        is_valid, checks = verifier.verify_rollout(
                            commit_data,
                            inference["proof"],
                            wallet_addr,
                            challenge_randomness=challenge_rand,
                            log_identity=uid_str,
                        )
                else:
                    is_valid, checks = verifier.verify_rollout(
                        commit_data,
                        inference["proof"],
                        wallet_addr,
                        challenge_randomness=challenge_rand,
                        log_identity=uid_str,
                    )
                
                pr_total += 1
                # Hard checks are cryptographic/proof constraints; any failure rejects wallet
                hard_valid = all(
                    checks.get(k, False) for k in HARD_CHECK_KEYS
                )
                soft_valid = checks.get(SOFT_CHECK_KEY, True)
                if not hard_valid:
                    pr_invalid_proof += 1
                    hard_failure = True
                    logger.warning(
                        f"Hard verification failed for {wallet_addr}; "
                        f"invalidating wallet for window {target_window}"
                    )
                    break
                if not soft_valid:
                    soft_failures += 1
                    if soft_failures >= soft_fail_cutoff:
                        soft_gate_triggered = True
                        logger.warning(
                            f"Soft-check failures threshold reached "
                            f"({soft_failures}/{total_planned_checks}) for {wallet_addr}; "
                            f"invalidating wallet for window {target_window}"
                        )
                        break
            except Exception as e:
                logger.warning(
                    f"Rollout verification error for {wallet_addr}: {e}"
                )
                continue
            
            valid_count += 1
            if (logger.isEnabledFor(logging.DEBUG) and
                text_logs_emitted_by_wallet[wallet_addr] < text_log_limit):
                try:
                    tokens = commit_data.get("tokens", [])
                    if isinstance(tokens, list) and tokens:
                        rollout_meta = commit_data.get("rollout", {})
                        prompt_len = int(
                            rollout_meta.get("prompt_length", 0) or 0
                        )
                        completion_len = int(
                            rollout_meta.get("completion_length", 0) or 0
                        )
                        if completion_len > 0 and prompt_len >= 0:
                            completion_ids = tokens[
                                prompt_len: prompt_len + completion_len
                            ]
                        else:
                            completion_ids = tokens[prompt_len:]
                        text = verifier.tokenizer.decode(
                            completion_ids, skip_special_tokens=False
                        )
                        reward_val = rollout_meta.get(
                            "total_reward", float("nan")
                        )
                        adv_val = rollout_meta.get(
                            "advantage", float("nan")
                        )
                        success_val = rollout_meta.get("success", False)
                        logger.debug(
                            "TEXT[validate] window=%s wallet=%s nonce=%s "
                            "reward=%.3f adv=%.3f success=%s text=%s",
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
                                f"validation/{uid_str}/sample_text",
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
            
            rollout_meta = inference.get("commit", {}).get("rollout", {})
            if rollout_meta.get("success", False):
                successful_rollouts += 1
                assignment = rollout_meta.get("assignment", [])
                solution_hash = hashlib.sha256(
                    str(assignment).encode()
                ).hexdigest()
                unique_solutions.add(solution_hash)
            wallet_rollouts_buffer.append(inference)
        except Exception as e:
            logger.debug(
                f"Error processing inference from {wallet_addr}: {e}"
            )
            pr_processing_err += 1
            continue
    
    # Handle wallet rejection due to hard or soft failures
    if hard_failure or soft_gate_triggered:
        metrics = {
            "valid": 0,
            "checked": checked_count,
            "total": total_inferences,
            "estimated_valid": 0,
            "successful": 0,
            "unique": 0,
        }
        logger.info(
            f"❌ Wallet {wallet_addr} rejected for window {target_window} "
            f"(hard_failure={hard_failure}, "
            f"soft_failures={soft_failures}/{total_planned_checks})"
        )
        return True, metrics, [], (
            pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err
        )
    
    # Verify GRPO groups after processing checked inferences
    # This ensures grouped rollouts follow GRPO constraints (shared base problem,
    # advantages sum to zero, etc.) even when spot-checking
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
                f"GRPO group {group_id} has only {len(group_rollouts)} "
                f"rollouts in checked sample, may be incomplete due to "
                f"spot-checking"
            )
            grpo_incomplete_groups += 1
            continue
        
        # Check if this looks like a complete group (should have 4 rollouts for GRPO)
        expected_group_size = ROLLOUTS_PER_PROBLEM
        if len(group_rollouts) != expected_group_size:
            logger.debug(
                f"GRPO group {group_id} has {len(group_rollouts)} rollouts, "
                f"expected {expected_group_size}"
            )
        advantages = []
        for r in group_rollouts:
            adv = r.get("commit", {}).get("rollout", {}).get(
                "advantage", 0.0
            )
            advantages.append(adv)
        advantage_sum = sum(advantages)
        if abs(advantage_sum) > GRPO_ADV_SUM_TOLERANCE:
            logger.debug(
                f"GRPO group {group_id} advantages don't sum to 0: "
                f"{advantage_sum} (advantages: {advantages})"
            )
            grpo_invalid_groups += 1
            continue
        base_seeds = []
        for r in group_rollouts:
            sat_problem = r.get("commit", {}).get("sat_problem", {})
            base_seeds.append(sat_problem.get("seed"))
        if len(set(base_seeds)) != 1:
            logger.debug(
                f"GRPO group {group_id} has different base problems: "
                f"{set(base_seeds)}"
            )
            grpo_invalid_groups += 1
            continue
        logger.debug(
            f"✅ GRPO group {group_id} verified: {len(group_rollouts)} "
            f"rollouts, advantages sum to {advantage_sum:.6f}"
        )
        grpo_valid_groups += 1
    if rollout_groups:
        total_groups_checked = (
            grpo_valid_groups + grpo_invalid_groups + grpo_incomplete_groups
        )
        if total_groups_checked > 0:
            logger.info(
                f"GRPO groups checked: {grpo_valid_groups} valid, "
                f"{grpo_invalid_groups} invalid, {grpo_incomplete_groups} "
                f"incomplete (spot-check artifact)"
            )
    
    # Calculate estimated total valid rollouts based on sampling
    # Extrapolate from sample to estimate total for weight computation
    sample_pass_rate = (
        (valid_count / checked_count) if checked_count > 0 else 0
    )
    estimated_valid = int(total_inferences * sample_pass_rate)
    
    # Store metrics for this miner
    metrics = {
        "valid": valid_count,
        "checked": checked_count,
        "total": total_inferences,
        "estimated_valid": estimated_valid,
        "successful": successful_rollouts,
        "unique": len(unique_solutions),
    }
    if wallet_rollouts_buffer:
        logger.info(
            f"✅ {wallet_addr}: {valid_count}/{checked_count} checked, "
            f"~{estimated_valid}/{total_inferences} estimated valid, "
            f"{successful_rollouts} successful, "
            f"{len(unique_solutions)} unique"
        )
    
    return True, metrics, wallet_rollouts_buffer, (
        pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err
    )


def _compute_weights(
    meta_hotkeys: List[str],
    inference_counts: DefaultDict[
        str, DefaultDict[int, Dict[str, int]]
    ],
    target_window: int,
) -> Tuple[List[float], List[Tuple[str, float]]]:
    """Compute normalized weights over the last N windows.

    Args:
        meta_hotkeys: Hotkeys in metagraph order.
        inference_counts: hotkey -> window_start -> metrics dict.
        target_window: Right edge (inclusive) of the range.

    Method:
        Sum metrics over the last WEIGHT_ROLLING_WINDOWS windows
        (step=WINDOW_LENGTH; missing windows -> 0). Convert to capped
        scores in [0, 1]; only 'unique' is currently weighted. Apply
        x**SUPERLINEAR_EXPONENT, then L1-normalize across miners. If the
        sum is zero, return an all-zero vector.

    Returns:
        (weights, non_zero_weights)
        weights: floats aligned to meta_hotkeys; sums to 1.0 or all zeros.
        non_zero_weights: (hotkey, weight) pairs where weight > 0.0.
    """
    EMPTY_METRICS: Dict[str, int] = {}
    raw_scores = []
    
    for _, hotkey in enumerate(meta_hotkeys):
        # Calculate score over last 12 windows
        recent_windows = range(
            max(0, target_window - (WEIGHT_ROLLING_WINDOWS - 1) * WINDOW_LENGTH),
            target_window + 1,
            WINDOW_LENGTH
        )
        total_unique = 0
        total_successful = 0
        total_estimated_valid = 0
        
        for w in recent_windows:
            metrics = inference_counts[hotkey].get(w, EMPTY_METRICS)
            total_unique += metrics.get("unique", 0)
            total_successful += metrics.get("successful", 0)
            total_estimated_valid += metrics.get("estimated_valid", 0)
        
        # Scoring formula: prioritize unique solutions, then successful, then valid
        # Base performance score in [0, 1]
        unique_score = (
            min(1.0, total_unique / 10.0) if total_unique > 0 else 0
        )
        success_score = (
            min(1.0, total_successful / 20.0) if total_successful > 0 else 0
        )
        valid_score = (
            min(1.0, total_estimated_valid / 50.0)
            if total_estimated_valid > 0 else 0
        )
        # NOTE: at this stage we only give weights to unique scores
        base_score = (
            1.0 * unique_score + 0.0 * success_score + 0.0 * valid_score
        )
        base_score = max(0.0, min(1.0, base_score))
        
        # Apply superlinear curve: emphasizes higher performers and penalizes splitting
        superlinear_score = base_score**SUPERLINEAR_EXPONENT
        raw_scores.append(superlinear_score)
    
    # Normalize weights
    denom = math.fsum(raw_scores)
    weights = (
        [score / denom for score in raw_scores]
        if denom > 0.0 else [0.0] * len(meta_hotkeys)
    )
    non_zero_weights = [
        (meta_hotkeys[i], weights[i])
        for i in range(len(weights)) if weights[i] > 0
    ]
    
    return weights, non_zero_weights


async def _upload_rollouts(
    target_window: int, all_valid_rollouts: List[dict], credentials: Any
) -> None:
    """Upload validated rollouts to object storage and Hugging Face."""
    upload_success = await upload_valid_rollouts(
        target_window, all_valid_rollouts, credentials
    )
    if upload_success:
        logger.info(
            f"📤 Uploaded {len(all_valid_rollouts)} valid rollouts for training"
        )
    else:
        logger.warning("⚠️ Failed to upload valid rollouts for training")
    
    # Upload to Hugging Face dataset for community access
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
            logger.debug(
                "Failed to upload to Hugging Face (may need HF_TOKEN)"
            )
    except Exception as e:
        logger.debug(f"Hugging Face upload error: {e}")


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    validate()
