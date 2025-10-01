#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import atexit
import contextlib
import faulthandler
import hashlib
import json
import logging
import math
import os
import random
import signal
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from types import SimpleNamespace, TracebackType
from typing import Any, Optional

import bittensor as bt
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from grail.infrastructure.drand import get_drand_beacon, get_round_at_time

from ..environments import create_sat_reward_vector
from ..grail import derive_canonical_sat
from ..infrastructure.chain import GrailChainManager
from ..infrastructure.comms import (
    file_exists,
    get_file,
    upload_to_huggingface,
    upload_valid_rollouts,
)
from ..infrastructure.credentials import load_r2_credentials
from ..infrastructure.network import create_subtensor
from ..logging_utils import MinerPrefixFilter, miner_log_context
from ..monitoring import get_monitoring_manager
from ..monitoring.config import MonitoringConfig
from ..scoring.weights import WeightComputer
from ..shared.constants import (
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    MINER_SAMPLING_ENABLED,
    NETUID,
    ROLLOUTS_PER_PROBLEM,
    SUPERLINEAR_EXPONENT,
    WINDOW_LENGTH,
)
from ..shared.subnet import get_own_uid_on_subnet
from ..validation import (
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_TRACKER,
    COPYCAT_WINDOW_THRESHOLD,
    CopycatViolation,
    compute_completion_digest,
)
from ..validation.pipeline import ValidationPipeline
from . import console

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
logger.addFilter(MinerPrefixFilter())


# --------------------------------------------------------------------------- #
#                           Crash Diagnostics                                 #
# --------------------------------------------------------------------------- #


def _flush_all_logs() -> None:
    """Best-effort flush of all logging handlers and stdio."""
    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            try:
                h.flush()
            except Exception:
                pass
        for h in list(logger.handlers):
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _install_crash_diagnostics() -> None:
    """Enable faulthandler and global exception logging for silent crashes."""
    # Dump Python tracebacks on fatal signals and C-level faults
    try:
        faulthandler.enable(all_threads=True)
        # Register common termination signals to dump tracebacks before exit
        for sig in (
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGABRT", None),
            getattr(signal, "SIGSEGV", None),
        ):
            if sig is not None:
                try:
                    faulthandler.register(sig, chain=True)
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure unhandled exceptions get logged
    def _excepthook(
        exc_type: type[BaseException], exc: BaseException, tb: Optional[TracebackType]
    ) -> None:
        try:
            if exc_type is KeyboardInterrupt:
                # Let standard handling occur for Ctrl-C
                return sys.__excepthook__(exc_type, exc, tb)
            logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            _flush_all_logs()

    try:
        sys.excepthook = _excepthook
    except Exception:
        pass

    # Flush logs on normal interpreter exit
    try:
        atexit.register(_flush_all_logs)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#                       Styling & configuration constants                     #
# --------------------------------------------------------------------------- #
# Sampling and validation parameters. Keep these centralized to avoid magic
# numbers scattered through validation logic and to make tuning straightforward.
MAX_SAMPLES_PER_MINER_THRESHOLD = 20  # If <= this many rollouts, check all
MAX_SAMPLES_PER_MINER = 40  # If > this many rollouts, sample GRPO groups
SAMPLE_RATE = 0.1  # Fraction of GRPO groups to spot-check
STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.26  # Soft-failure fraction to gate wallet
REWARD_REL_TOL = 0.02  # Relative tolerance on reward bounds
REWARD_ABS_TOL = 1e-6  # Absolute tolerance on reward bounds
GRPO_ADV_SUM_TOLERANCE = 0.01  # Sum of advantages should be ~0
DEBUG_TEXT_LOG_LIMIT_PER_WALLET = 5  # Max sample texts logged per wallet
SOFT_CHECK_KEY = "token_distribution_valid"  # Soft heuristic (reduces score, doesn't reject)

HARD_CHECK_KEYS = (  # Hard checks required for validity (fail any = reject)
    "schema_valid",  # Schema/structure validation
    "tokens_valid",  # Token vocab/length validation
    "proof_valid",  # GRAIL cryptographic proof
    "sat_problem_valid",  # SAT problem regeneration
    "prompt_valid",  # Canonical prompt matching
    "termination_valid",  # Max length or confident EOS
    "solution_valid",  # Assignment correctness (if success claimed)
)

# Per-wallet per-window failure flag for gating across rolling windows
FAILURE_FLAG_KEY = "had_failure"

# Submit weights to chain at most once per this many blocks
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 360

# Number of windows to include when computing rolling weights
WEIGHT_ROLLING_WINDOWS = int(WEIGHT_SUBMISSION_INTERVAL_BLOCKS / WINDOW_LENGTH)

# Number of miners to log in detail on submission
# TODO: reduce this later
TOP_K_WEIGHTS_LOGGED = 256

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


def parse_filename(filename: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
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


def parse_window_filename(filename: str) -> tuple[Optional[str], Optional[int]]:
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


# ----------------------------- Sampling Helpers ----------------------------- #


# NOTE: If this function experiences performance issues or timeouts,
# consider reducing the concurrency parameter (default: 16)
async def _list_active_hotkeys_for_window(
    meta_hotkeys: list[str],
    window_start: int,
    chain_manager: "GrailChainManager",
    default_credentials: Any,
    uid_by_hotkey: Optional[dict[str, int]] = None,
    concurrency: int = 16,
) -> list[str]:
    """Return hotkeys with an available window file for the given window.

    Active miners are those that uploaded `grail/windows/{hotkey}-window-{window_start}.json`.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _check(hotkey: str) -> tuple[str, bool]:
        filename = f"grail/windows/{hotkey}-window-{window_start}.json"
        bucket = chain_manager.get_bucket_for_hotkey(hotkey)
        uid = uid_by_hotkey.get(hotkey) if uid_by_hotkey else None
        miner_id = f"uid={uid}" if uid is not None else f"hotkey={hotkey[:12]}..."

        # Immediately skip miners without a committed bucket (no fallback)
        if bucket is None:
            return hotkey, False

        async with semaphore:
            import time

            start_time = time.time()
            try:
                from ..infrastructure.comms import file_exists

                exists = await asyncio.wait_for(
                    file_exists(
                        filename,
                        credentials=bucket,
                        use_write=False,
                    ),
                    timeout=6.0,
                )
                elapsed = time.time() - start_time
                return hotkey, bool(exists)
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.debug(
                    "Window file check TIMEOUT for %s window=%s after %.2fs",
                    miner_id,
                    window_start,
                    elapsed,
                )
                return hotkey, False
            except Exception:
                elapsed = time.time() - start_time
                return hotkey, False

    results = await asyncio.gather(*(_check(hk) for hk in meta_hotkeys))
    return [hk for hk, ok in results if ok]


def _compute_sample_size(active_count: int) -> int:
    if active_count <= 0:
        return 0
    rate_k = int(math.ceil(active_count * float(MINER_SAMPLE_RATE)))
    k = max(int(MINER_SAMPLE_MIN), rate_k)
    if MINER_SAMPLE_MAX is not None:
        try:
            k = min(k, int(MINER_SAMPLE_MAX))
        except Exception:
            pass
    return min(k, active_count)


def _select_miners_for_window(
    active_hotkeys: list[str],
    target_window_hash: str,
    selection_counts: dict[str, int],
) -> list[str]:
    k = _compute_sample_size(len(active_hotkeys))
    if k == 0:
        return []

    def _tie_break(hk: str) -> int:
        dig = hashlib.sha256(f"{target_window_hash}:{hk}".encode()).digest()
        return int.from_bytes(dig[:8], "big")

    ranked = sorted(
        active_hotkeys, key=lambda hk: (int(selection_counts.get(hk, 0)), _tie_break(hk))
    )
    return ranked[:k]


def _update_rolling(
    history: "deque[set[str]]",
    counts: dict[str, int],
    new_set: set[str],
    horizon: int,
) -> None:
    if len(history) >= horizon:
        old_set = history.popleft()
        for hk in old_set:
            counts[hk] = max(0, int(counts.get(hk, 0)) - 1)
    history.append(new_set)
    for hk in new_set:
        counts[hk] = int(counts.get(hk, 0)) + 1


def _compute_count_stats(values: list[int]) -> dict[str, float]:
    try:
        n = len(values)
        if n == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0, "var": 0.0}
        v_min = float(min(values))
        v_max = float(max(values))
        mean_v = float(math.fsum(values) / n)
        var_v = float(math.fsum((float(x) - mean_v) * (float(x) - mean_v) for x in values) / n)
        return {"min": v_min, "mean": mean_v, "max": v_max, "var": var_v}
    except Exception:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "var": 0.0}


# Global storage for miner state
miner_inference_counts: defaultdict[str, list] = defaultdict(
    list
)  # track inferences per block for weight calculation


# --------------------------------------------------------------------------- #
#                        Unique Rollout Helpers                               #
# --------------------------------------------------------------------------- #


def _update_unique_rollouts(
    unique_rollouts: set[str],
    commit_data: dict,
    rollout_meta: dict,
) -> set[str]:
    """
    Update the set of unique rollouts by hashing completion token IDs.

    Args:
        unique_rollouts: Set to add unique rollout hashes to
        commit_data: Commit data containing tokens
        rollout_meta: Rollout metadata containing length information

    Returns:
        Updated unique_rollouts set
    """
    try:
        tokens = commit_data.get("tokens", [])
        if not tokens:
            return unique_rollouts

        prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)

        # Extract completion token IDs
        completion_ids = tokens[prompt_len:]

        # Hash the completion token IDs
        digest_input = json.dumps(
            completion_ids, separators=(",", ":"), ensure_ascii=False
        ).encode()
        rollout_hash = hashlib.sha256(digest_input).hexdigest()

        if rollout_hash not in unique_rollouts:
            unique_rollouts.add(rollout_hash)

    except Exception as e:
        # Fallback: hash full token list if slicing fails
        logger.debug(f"Completion token slicing failed ({e}), using fallback hash method")
        try:
            digest_input = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode()
            rollout_hash = hashlib.sha256(digest_input).hexdigest()
            if rollout_hash not in unique_rollouts:
                unique_rollouts.add(rollout_hash)
        except Exception:
            logger.warning("Failed to hash tokens for unique rollout tracking")

    return unique_rollouts


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
    while True:
        await asyncio.sleep(timeout // 3)
        elapsed = time.monotonic() - HEARTBEAT
        if elapsed > timeout:
            logging.error(f"[WATCHDOG] Process stalled {elapsed:.0f}s — exiting process.")
            # Best-effort flush so the final error is not lost
            try:
                _flush_all_logs()
                # Give a moment for logs to ship in containerized envs
                time.sleep(0.1)
            except Exception:
                pass
            # Hard exit to avoid being stuck indefinitely
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
    # Install crash diagnostics early to catch silent failures
    _install_crash_diagnostics()
    from ..neurons import ValidatorNeuron

    asyncio.run(ValidatorNeuron(use_drand=use_drand, test_mode=test_mode).main())


# ----------------------------- Refactored Helpers ---------------------------- #


def _flush_all_logs() -> None:
    """Flush all logging handlers to ensure messages are written before exit."""
    import logging

    for handler in logging.root.handlers:
        handler.flush()
    # Also flush any handlers on the current logger
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        handler.flush()


def _get_sat_reward_bounds() -> tuple[float, float]:
    """Return SAT reward bounds or permissive defaults on failure."""
    try:
        _sat_rv = create_sat_reward_vector()
        low, high = _sat_rv.reward_bounds()
        return float(low), float(high)
    except Exception:
        return float("-inf"), float("inf")


async def _run_validation_service(
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sat_pipeline: ValidationPipeline,
    weight_computer: WeightComputer,
    sat_reward_low: float,
    sat_reward_high: float,
    use_drand: bool,
    test_mode: bool,
) -> None:
    """Run validation service: main loop + watchdog.

    Args:
        wallet: Bittensor wallet used for signing and network ops.
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        sat_pipeline: SAT validation pipeline.
        weight_computer: Weight computer instance.
        sat_reward_low: Lower bound for SAT rollout reward sanity checks.
        sat_reward_high: Upper bound for SAT rollout reward sanity checks.
        use_drand: Whether to incorporate drand randomness in challenges.
        test_mode: If True, validate only the local hotkey (developer testing).

    This function manages lifecycle and orchestration. It delegates window/work
    processing to helpers and keeps the outer loop small and readable.
    """

    # Install an asyncio exception handler to log any task errors
    try:
        loop = asyncio.get_running_loop()

        def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
            msg = context.get("message") or "Asyncio exception in task"
            exc = context.get("exception")
            if exc is not None:
                logger.error(msg, exc_info=exc)
            else:
                logger.error(msg)
            _flush_all_logs()

        loop.set_exception_handler(_asyncio_exception_handler)
    except Exception:
        pass

    async def _validation_loop() -> None:
        subtensor = None
        credentials, chain_manager = await _initialize_credentials_and_chain(wallet)
        monitor = await _initialize_monitor(wallet)
        # Rolling window metrics per hotkey, keyed by window_start -> metric dict
        # Metrics include: valid, checked, total, estimated_valid, successful, unique
        inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        last_processed_window = -1
        last_weights_interval_submitted = -1
        last_copycat_interval_id = -1
        # Rolling histories (12-window horizon) for selection coverage and availability
        selection_history: deque[set[str]] = deque(maxlen=WEIGHT_ROLLING_WINDOWS)
        availability_history: deque[set[str]] = deque(maxlen=WEIGHT_ROLLING_WINDOWS)
        selection_counts: dict[str, int] = {}
        availability_counts: dict[str, int] = {}

        while True:
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()

                if subtensor is None:
                    subtensor = await get_subtensor()

                meta = await subtensor.metagraph(NETUID)
                # Build hotkey -> uid mapping for per-miner logging namespaces
                uid_by_hotkey = dict(zip(meta.hotkeys, meta.uids))
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

                # Reset copycat tracker at the start of each submission interval,
                # using window-derived interval ids to avoid mid-loop resets.
                copycat_interval_id = int(target_window // WEIGHT_SUBMISSION_INTERVAL_BLOCKS)
                if copycat_interval_id != last_copycat_interval_id:
                    COPYCAT_TRACKER.reset_interval(copycat_interval_id)
                    last_copycat_interval_id = copycat_interval_id

                target_window_hash = await subtensor.get_block_hash(target_window)
                # Single per-window randomness value reused across all checks
                window_rand = _compute_window_randomness(target_window_hash, use_drand)

                # Discover active miners (those with a window file in storage)
                timer_ctx = (
                    monitor.timer("validation/hotkeys_discovery")
                    if monitor
                    else contextlib.nullcontext()
                )
                with timer_ctx:
                    active_hotkeys = await _list_active_hotkeys_for_window(
                        meta.hotkeys, target_window, chain_manager, credentials, uid_by_hotkey
                    )

                logger.info(
                    f"🔍 Found {len(active_hotkeys)}/{len(meta.hotkeys)} active miners for window {target_window}"
                )

                # Update availability (windows_with_file) rolling window
                _update_rolling(
                    availability_history,
                    availability_counts,
                    set(active_hotkeys),
                    WEIGHT_ROLLING_WINDOWS,
                )
                # Determine subset to validate this window
                if test_mode:
                    hotkeys_to_check = [wallet.hotkey.ss58_address]
                elif MINER_SAMPLING_ENABLED:
                    hotkeys_to_check = _select_miners_for_window(
                        active_hotkeys, target_window_hash, selection_counts
                    )
                else:
                    hotkeys_to_check = active_hotkeys

                # Update selection coverage rolling window
                _update_rolling(
                    selection_history,
                    selection_counts,
                    set(hotkeys_to_check),
                    WEIGHT_ROLLING_WINDOWS,
                )

                # Log sampling and availability metrics
                if monitor:
                    try:
                        await monitor.log_gauge("miner_sampling/miners_total", len(meta.hotkeys))
                        await monitor.log_gauge("miner_sampling/miners_active", len(active_hotkeys))
                        await monitor.log_gauge(
                            "miner_sampling/miners_selected", len(hotkeys_to_check)
                        )
                        eff_rate = (
                            (len(hotkeys_to_check) / len(active_hotkeys)) if active_hotkeys else 0.0
                        )
                        await monitor.log_gauge("miner_sampling/check_rate", eff_rate)

                        av_values = [int(availability_counts.get(hk, 0)) for hk in meta.hotkeys]
                        av_stats = _compute_count_stats(av_values)
                        await monitor.log_gauge(
                            f"miner_availability/min_windows_active_{WEIGHT_ROLLING_WINDOWS}w",
                            av_stats["min"],
                        )
                        await monitor.log_gauge(
                            f"miner_availability/avg_windows_active_{WEIGHT_ROLLING_WINDOWS}w",
                            av_stats["mean"],
                        )
                        await monitor.log_gauge(
                            f"miner_availability/max_windows_active_{WEIGHT_ROLLING_WINDOWS}w",
                            av_stats["max"],
                        )
                    except Exception:
                        pass

                logger.info(
                    "Sampling: total=%s active=%s selected=%s rate=%.3f",
                    len(meta.hotkeys),
                    len(active_hotkeys),
                    len(hotkeys_to_check),
                    (len(hotkeys_to_check) / len(active_hotkeys)) if active_hotkeys else 0.0,
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
                    model=model,
                    tokenizer=tokenizer,
                    sat_pipeline=sat_pipeline,
                    credentials=credentials,
                    chain_manager=chain_manager,
                    monitor=monitor,
                    subtensor=subtensor,
                    uid_by_hotkey=uid_by_hotkey,
                    sat_reward_low=sat_reward_low,
                    sat_reward_high=sat_reward_high,
                )

                # Log summary
                logger.info(
                    f"📁 Found {files_found} window files from {len(meta.uids)} registered UIDs"
                )
                logger.info(
                    f"🏁 Total valid rollouts in window {target_window}: {total_valid_rollouts}"
                )

                # Log rollout statistics per miner
                rollout_counts = [m.get("total", 0) for m in window_inference_counts.values()]
                if rollout_counts:
                    min_rollouts = min(rollout_counts)
                    max_rollouts = max(rollout_counts)
                    avg_rollouts = sum(rollout_counts) / len(rollout_counts)
                    logger.info(
                        f"📊 Rollouts per miner - min: {min_rollouts}, "
                        f"avg: {avg_rollouts:.1f}, max: {max_rollouts}"
                    )

                # Aggregate had_failure across wallets for visibility
                failed_wallets = sum(
                    1
                    for m in window_inference_counts.values()
                    if int(m.get(FAILURE_FLAG_KEY, 0)) == 1
                )
                logger.info(f"🚫 UIDs gated this window: {failed_wallets}")

                # Monitoring metrics
                if monitor:
                    await monitor.log_counter("validation/windows_processed")
                    await monitor.log_gauge(
                        "validation/total_rollouts_processed", total_rollouts_processed
                    )
                    await monitor.log_gauge("validation/valid_rollouts", total_valid_rollouts)
                    await monitor.log_gauge("validation/invalid_signatures", invalid_signatures)
                    await monitor.log_gauge("validation/invalid_proofs", invalid_proofs)
                    await monitor.log_gauge("validation/processing_errors", processing_errors)
                    await monitor.log_gauge("validation/files_found", files_found)
                    await monitor.log_gauge("validation/failed_wallets_window", failed_wallets)
                    # Aggregate prompt prefix validity across wallets for this window
                    try:
                        agg_prompt_valid = sum(
                            int(m.get("prompt_valid", 0)) for m in window_inference_counts.values()
                        )
                        agg_prompt_mismatch = sum(
                            int(m.get("prompt_mismatch", 0))
                            for m in window_inference_counts.values()
                        )
                        await monitor.log_gauge("validation/prompt_valid", float(agg_prompt_valid))
                        await monitor.log_gauge(
                            "validation/prompt_mismatches", float(agg_prompt_mismatch)
                        )
                    except Exception:
                        pass
                    if rollout_counts:
                        await monitor.log_gauge("validation/rollouts_per_miner/min", min_rollouts)
                        await monitor.log_gauge("validation/rollouts_per_miner/avg", avg_rollouts)
                        await monitor.log_gauge("validation/rollouts_per_miner/max", max_rollouts)
                    if total_rollouts_processed > 0:
                        success_rate = total_valid_rollouts / total_rollouts_processed
                        await monitor.log_gauge("validation/success_rate", success_rate)

                # Uploads
                if all_valid_rollouts:
                    await _upload_rollouts(
                        target_window=target_window,
                        all_valid_rollouts=all_valid_rollouts,
                        credentials=credentials,
                    )

                # Update inference counts (must occur before score-based logging)
                for hotkey, metrics in window_inference_counts.items():
                    inference_counts[hotkey][target_window] = metrics

                # Top-miner logs (use already-computed metrics; scores include current window)
                if rollout_counts:
                    await _log_top_miners_by_rollout_count(
                        window_inference_counts, uid_by_hotkey, target_window, monitor
                    )
                    # TODO: the unique identifier logic needs to improve before we uncomment this
                    # await _log_top_miners_by_unique_rollouts(
                    #     window_inference_counts, uid_by_hotkey, target_window, monitor
                    # )
                    await _log_top_miners_by_score(
                        inference_counts, uid_by_hotkey, target_window, monitor
                    )

                # Compute active miner UIDs over the last rolling windows (includes failures)
                active_uids = _get_active_uids(
                    meta.hotkeys, list(meta.uids), inference_counts, target_window
                )
                active_miners = len(active_uids)
                logger.info(
                    f"👷 Active miners (last {WEIGHT_ROLLING_WINDOWS} windows): {active_miners}"
                )
                logger.info(
                    f"👷 Active miner UIDs (last {WEIGHT_ROLLING_WINDOWS} windows): {active_uids}"
                )
                if monitor:
                    await monitor.log_gauge(
                        f"validation/active_miners_past_{WEIGHT_ROLLING_WINDOWS}w", active_miners
                    )
                    await monitor.log_artifact(
                        f"validation/active_miners_uids_past_{WEIGHT_ROLLING_WINDOWS}w",
                        {"window": target_window, "text": ",".join(str(u) for u in active_uids)},
                        "text",
                    )

                # Compute and set weights using weight computer
                weights, non_zero_weights = weight_computer.compute_weights(
                    meta_hotkeys=meta.hotkeys,
                    meta_uids=list(meta.uids),
                    inference_counts=inference_counts,
                    target_window=target_window,
                )
                if non_zero_weights:
                    logger.info(f"⚖️  Setting weights for {len(non_zero_weights)} miners")
                    logger.info(f"Displaying weights for first 5 miners: {non_zero_weights[:5]}")
                    for hotkey, weight in non_zero_weights[:5]:
                        uid = uid_by_hotkey.get(hotkey, None)
                        display = uid if uid is not None else hotkey
                        with miner_log_context(display, target_window):
                            logger.info(f"weight: {weight:.4f} - top 5 miners")
                else:
                    logger.info("⚖️  No miners received weights this window")

                if monitor:
                    await monitor.log_gauge("validation/miners_with_weights", len(non_zero_weights))
                    await monitor.log_gauge("validation/total_registered_miners", len(meta.hotkeys))
                    if weights:
                        max_weight = max(weights)
                        avg_weight = sum(weights) / len(weights)
                        await monitor.log_gauge("validation/max_weight", max_weight)
                        await monitor.log_gauge("validation/average_weight", avg_weight)
                        # Additional distribution stats (top 5 minimal set)
                        stats = _compute_weight_stats(weights)
                        await monitor.log_gauge("weights/stats/min", stats["min"])
                        await monitor.log_gauge("weights/stats/mean", stats["mean"])
                        await monitor.log_gauge("weights/stats/max", stats["max"])
                        await monitor.log_gauge("weights/stats/gini", stats["gini"])
                # Concise distribution summary
                if weights:
                    stats = _compute_weight_stats(weights)
                    logger.info(
                        "weights: min=%.6f mean=%.6f max=%.6f gini=%.6f",
                        stats["min"],
                        stats["mean"],
                        stats["max"],
                        stats["gini"],
                    )
                # Global submission context (per loop)
                if monitor:
                    await monitor.log_gauge(
                        "weights/config/rolling_windows",
                        WEIGHT_ROLLING_WINDOWS,
                    )
                    await monitor.log_gauge(
                        "weights/config/superlinear_exponent",
                        SUPERLINEAR_EXPONENT,
                    )
                logger.debug(
                    "Weights context prepared: interval_blocks=%s block=%s "
                    "window=%s rolling=%s superlinear=%s",
                    WEIGHT_SUBMISSION_INTERVAL_BLOCKS,
                    current_block,
                    target_window,
                    WEIGHT_ROLLING_WINDOWS,
                    SUPERLINEAR_EXPONENT,
                )

                # Throttle on-chain weight submissions to once per 360-block interval
                current_interval = int(current_block // WEIGHT_SUBMISSION_INTERVAL_BLOCKS)
                is_a_weight_submission_block = current_interval != last_weights_interval_submitted
                if is_a_weight_submission_block:
                    # Precompute top miners for per-miner logging on submission
                    top_miners = _build_top_miners(
                        meta.hotkeys, meta.uids, weights, TOP_K_WEIGHTS_LOGGED
                    )
                    logger.debug(
                        "set_weights args: wallet=%s netuid=%s uids=%s weights=%s wait_for_inclusion=%s",
                        wallet.hotkey.ss58_address if wallet and wallet.hotkey else "None",
                        NETUID,
                        meta.uids,
                        weights,
                        False,
                    )
                    await subtensor.set_weights(
                        wallet=wallet,
                        netuid=NETUID,
                        uids=meta.uids,
                        weights=weights,
                        wait_for_inclusion=False,
                    )
                    last_weights_interval_submitted = current_interval

                    # Log successful miners during weight submission
                    submission_successful_uids = [
                        uid_by_hotkey.get(hk)
                        for hk, _ in non_zero_weights
                        if uid_by_hotkey.get(hk) is not None
                    ]

                    if monitor:
                        await monitor.log_gauge("weights/submission/submitted", 1.0)
                        # Log successful miners at submission time
                        await monitor.log_gauge(
                            "weights/submission/successful_miners_count",
                            len(submission_successful_uids),
                        )
                        if submission_successful_uids:
                            # Create a list of UID data for better visualization
                            uid_weight_map = {}
                            for hk, w in non_zero_weights:
                                uid = uid_by_hotkey.get(hk)
                                if uid is not None:
                                    uid_weight_map[uid] = w

                            # Log summary with window prominently displayed
                            logger.info(
                                f"🏆 Window {target_window} - Successful miners: {len(submission_successful_uids)} UIDs"
                            )

                            # Log detailed UID information
                            uid_details = []
                            for uid in sorted(submission_successful_uids):
                                weight = uid_weight_map.get(uid, 0.0)
                                uid_details.append(f"UID:{uid} (weight:{weight:.4f})")

                            logger.info(f"Window {target_window} UIDs: {', '.join(uid_details)}")

                            # Log all UIDs in a single row
                            await monitor.log_artifact(
                                "weights/submission/successful_miners",
                                {
                                    "window": target_window,
                                    "text": ",".join(
                                        str(uid) for uid in sorted(submission_successful_uids)
                                    ),
                                },
                                "text",
                            )

                    logger.info(
                        "Submitted weights: interval=%s block=%s window=%s "
                        "miners_with_weights=%s total_miners=%s rolling=%s superlinear=%s",
                        current_interval,
                        current_block,
                        target_window,
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
                            await monitor.log_gauge(f"{uid_ns}/superlinear_score", s)
                            await monitor.log_gauge(f"{uid_ns}/inputs/unique_rolling", tu)
                            await monitor.log_gauge(f"{uid_ns}/inputs/successful_rolling", ts)
                            await monitor.log_gauge(
                                f"{uid_ns}/inputs/estimated_valid_rolling",
                                tev,
                            )

                    # Log top 5 best
                    LOG_TOP_MINERS = 5
                    for hk, uid, w in top_miners[:LOG_TOP_MINERS]:
                        tu, ts, tev, b, s = _aggregate_weight_inputs(
                            hk, inference_counts, target_window
                        )
                        logger.info(
                            "Weight uid=%s hotkey=%s weight=%.6f base=%.6f "
                            "unique_rolling=%d successful_rolling=%d "
                            "estimated_valid_rolling=%d",
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
                        await monitor.log_gauge("weights/submission/submitted", 0.0)
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

    results = await asyncio.gather(
        _validation_loop(),
        watchdog(timeout=(60 * 10)),
        return_exceptions=True,
    )
    # Log any surfaced exceptions from gathered tasks (should be rare)
    for res in results:
        if isinstance(res, Exception):
            logger.error("Background task error", exc_info=res)
            _flush_all_logs()


async def _initialize_credentials_and_chain(wallet: bt.wallet) -> tuple[Any, GrailChainManager]:
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
    logger.info("✅ Initialized chain manager and committed read credentials")

    return credentials, chain_manager


async def _initialize_monitor(wallet: bt.wallet) -> Any:
    """Initialize monitoring run for validation, if configured.

    Returns:
        Monitoring client or None if monitoring is disabled.
    """
    monitor = get_monitoring_manager()
    if monitor:
        validation_config = MonitoringConfig.for_validation(wallet.name)
        try:
            subtensor = await get_subtensor()
        except Exception:
            subtensor = None
        uid = None
        if subtensor is not None:
            uid = await get_own_uid_on_subnet(subtensor, 81, wallet.hotkey.ss58_address)
        run_name = f"validator-{uid}" if uid is not None else f"validation_{wallet.name}"
        run_id = await monitor.start_run(run_name, validation_config.get("hyperparameters", {}))
        logger.info(f"Started monitoring run: {run_id} (name={run_name})")

    return monitor


def _determine_target_window(current_block: int) -> int:
    """Compute the last fully completed window start from current block."""
    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
    return current_window - WINDOW_LENGTH


def _compute_window_randomness(target_window_hash: str, use_drand: bool) -> str:
    """Derive deterministic per-window randomness, optionally including drand."""
    if use_drand:
        try:
            drand_round = get_round_at_time(int(time.time()))
            drand_beacon = get_drand_beacon(drand_round)
            return hashlib.sha256(
                (target_window_hash + drand_beacon["randomness"]).encode()
            ).hexdigest()
        except Exception:
            return hashlib.sha256(target_window_hash.encode()).hexdigest()

    return hashlib.sha256(target_window_hash.encode()).hexdigest()


def _determine_hotkeys_to_check(test_mode: bool, wallet: bt.wallet, meta: Any) -> list[str]:
    """Choose which hotkeys to validate based on test/prod mode."""
    uid_by_hotkey = dict(zip(meta.hotkeys, meta.uids))
    if test_mode:
        own_uid = uid_by_hotkey.get(wallet.hotkey.ss58_address)
        msg_id = own_uid if own_uid is not None else wallet.hotkey.ss58_address
        logger.info(f"🧪 TEST MODE: Checking files for own uid {msg_id}")
        return [wallet.hotkey.ss58_address]

    logger.info(f"Checking files for {len(meta.uids)} registered UIDs")
    logger.info(f"UIDs to Check: {list(meta.uids)}")
    return list(meta.hotkeys)


def _aggregate_weight_inputs(
    hotkey: str,
    inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]],
    target_window: int,
) -> tuple[int, int, int, float, float]:
    """Aggregate rolling inputs and derive base/superlinear scores.

    Returns: (unique_sum, successful_sum, estimated_valid_sum,
              base_score, superlinear_score)
    """
    recent_windows = range(
        max(0, target_window - (WEIGHT_ROLLING_WINDOWS - 1) * WINDOW_LENGTH),
        target_window + 1,
        WINDOW_LENGTH,
    )
    total_estimated_unique = 0
    total_estimated_successful = 0
    total_estimated_valid = 0
    for w in recent_windows:
        metrics = inference_counts[hotkey].get(w, {})
        total_estimated_unique += int(metrics.get("estimated_unique", 0))
        total_estimated_successful += int(metrics.get("estimated_successful", 0))
        total_estimated_valid += int(metrics.get("estimated_valid", 0))
    # Unbounded unique score to match _compute_weights logic
    unique_score = total_estimated_unique
    # NOTE: at this stage we only give weights to unique scores
    base_score = max(0.0, 1.0 * unique_score + 0.0 * 0.0 + 0.0 * 0.0)
    superlinear_score = base_score**SUPERLINEAR_EXPONENT
    return (
        total_estimated_unique,
        total_estimated_successful,
        total_estimated_valid,
        base_score,
        superlinear_score,
    )


def _build_top_miners(
    hotkeys: list[str], uids: list[int], weights: list[float], k: int
) -> list[tuple[str, int, float]]:
    """Return top-k (hotkey, uid, weight) sorted by weight desc."""
    pairs = [(hk, uid, float(weights[i])) for i, (hk, uid) in enumerate(zip(hotkeys, uids))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def _compute_weight_stats(weights: list[float]) -> dict[str, float]:
    """Compute minimal weight distribution stats relevant for validators.

    Returns keys: min, mean, max, gini.
    """
    try:
        n = len(weights)
        if n == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0, "gini": 0.0}
        w_min = float(min(weights))
        w_max = float(max(weights))
        w_mean = float(math.fsum(weights) / n)
        total = math.fsum(weights)
        if total <= 0.0:
            gini = 0.0
        else:
            # Gini using sorted weights: (2*sum(i*w_i))/(n*sum) - (n+1)/n
            sorted_w = sorted(float(w) for w in weights)
            cumulative_indexed_sum = 0.0
            for i, w in enumerate(sorted_w, start=1):
                cumulative_indexed_sum += i * w
            gini = (2.0 * cumulative_indexed_sum) / (n * total) - (n + 1.0) / n
            if gini < 0.0:
                gini = 0.0
        return {"min": w_min, "mean": w_mean, "max": w_max, "gini": float(gini)}
    except Exception:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "gini": 0.0}


def _get_active_uids(
    meta_hotkeys: list[str],
    meta_uids: list[int],
    inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]],
    target_window: int,
) -> list[int]:
    """Return UIDs of miners active in the last WEIGHT_ROLLING_WINDOWS windows."""
    recent_windows = range(
        max(0, target_window - (WEIGHT_ROLLING_WINDOWS - 1) * WINDOW_LENGTH),
        target_window + 1,
        WINDOW_LENGTH,
    )
    active_uids: list[int] = []
    for hotkey, uid in zip(meta_hotkeys, meta_uids):
        hk_windows = inference_counts[hotkey]
        for w in recent_windows:
            metrics = hk_windows.get(w)
            if metrics and (
                int(metrics.get("total", 0)) > 0 or int(metrics.get(FAILURE_FLAG_KEY, 0)) == 1
            ):
                active_uids.append(int(uid))
                break
    return active_uids


async def _process_window(
    hotkeys_to_check: list[str],
    target_window: int,
    target_window_hash: str,
    window_rand: str,
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sat_pipeline: ValidationPipeline,
    credentials: Any,
    chain_manager: GrailChainManager,
    monitor: Any,
    subtensor: bt.subtensor,
    uid_by_hotkey: dict[str, int],
    sat_reward_low: float,
    sat_reward_high: float,
) -> tuple[dict[str, dict[str, int]], int, int, int, int, int, int, list[dict]]:
    """Process a window across hotkeys and aggregate metrics/results.

    Rationale:
    - Validates selected miners' window files, performs per-rollout checks,
      aggregates per-miner metrics, and ingests rollout digests into the
      copycat tracker (pairwise, window + interval scopes). Cheaters are gated
      (set FAILURE_FLAG_KEY) and excluded from uploads.

    Returns
    -------
    window_inference_counts: dict[str, dict[str, int]]
        Per-miner metrics for this window (valid/checked/total/estimated_valid/unique/...)
        plus FAILURE_FLAG_KEY when gated.
    total_valid_rollouts: int
        Sum of estimated_valid across miners (after any copycat gating).
    total_rollouts_processed: int
        Count of rollouts the verifier attempted to process (per-rollout checks).
    invalid_signatures: int
        Number of per-rollout signature failures observed.
    invalid_proofs: int
        Number of per-rollout hard-proof failures observed.
    processing_errors: int
        Number of exceptions during per-rollout verification.
    files_found: int
        Number of miner window files found this window.
    all_valid_rollouts: list[dict]
        Rollouts that passed for upload (pruned of any cheater miners).
    """
    # Window timing (seconds and blocks)
    window_t0 = time.monotonic()
    try:
        block_beg = await subtensor.get_current_block()
    except Exception:
        block_beg = None
    total_valid_rollouts = 0
    window_inference_counts: dict[str, dict[str, int]] = {}
    miner_rollout_counters: dict[str, tuple[Counter[str], int]] = {}
    files_found = 0
    all_valid_rollouts: list[dict] = []
    # Limit how many sample texts we log per wallet (debug noise control)
    text_logs_emitted_by_wallet: defaultdict[str, int] = defaultdict(int)

    total_rollouts_processed = 0
    invalid_signatures = 0
    invalid_proofs = 0
    processing_errors = 0

    miner_seconds_list: list[float] = []
    miner_blocks_list: list[int] = []

    for wallet_addr in hotkeys_to_check:
        try:
            # Refresh watchdog heartbeat on each miner to avoid false positives
            try:
                global HEARTBEAT
                HEARTBEAT = time.monotonic()
            except Exception:
                pass
            # Per-miner timing (seconds and blocks) measured around the call
            t0 = time.monotonic()
            try:
                b0 = await subtensor.get_current_block()
            except Exception:
                b0 = None

            with miner_log_context(uid_by_hotkey[wallet_addr], target_window):
                (
                    found_file,
                    metrics,
                    published_rollouts,
                    processed_counts,
                    digest_counter,
                    total_rollouts,
                ) = await _process_wallet_window(
                    wallet_addr=wallet_addr,
                    target_window=target_window,
                    target_window_hash=target_window_hash,
                    window_rand=window_rand,
                    wallet=wallet,
                    model=model,
                    tokenizer=tokenizer,
                    sat_pipeline=sat_pipeline,
                    credentials=credentials,
                    chain_manager=chain_manager,
                    monitor=monitor,
                    uid_by_hotkey=uid_by_hotkey,
                    text_logs_emitted_by_wallet=text_logs_emitted_by_wallet,
                    text_log_limit=DEBUG_TEXT_LOG_LIMIT_PER_WALLET,
                    sat_reward_low=sat_reward_low,
                    sat_reward_high=sat_reward_high,
                )
            t1 = time.monotonic()
            try:
                b1 = await subtensor.get_current_block()
            except Exception:
                b1 = None
            sec = t1 - t0
            blk = (b1 - b0) if (b0 is not None and b1 is not None) else 0
            miner_seconds_list.append(float(sec))
            miner_blocks_list.append(int(blk))
            if found_file:
                files_found += 1
            if metrics is not None:
                window_inference_counts[wallet_addr] = metrics
            if published_rollouts:
                all_valid_rollouts.extend(published_rollouts)
            if digest_counter is not None:
                miner_rollout_counters[wallet_addr] = (digest_counter, total_rollouts)
            (pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err) = processed_counts
            total_rollouts_processed += pr_total
            invalid_signatures += pr_invalid_sig
            invalid_proofs += pr_invalid_proof
            processing_errors += pr_processing_err
        except Exception as e:
            uid_str = str(uid_by_hotkey.get(wallet_addr, wallet_addr))
            with miner_log_context(uid_str, target_window):
                logger.warning(f"Error processing: {e}")
            continue

    for metrics in window_inference_counts.values():
        total_valid_rollouts += metrics.get("estimated_valid", 0)

    # Copycat detection ingestion and gating
    window_cheaters: set[str] = set()
    window_violation_details: list[CopycatViolation] = []
    interval_cheaters: set[str] = set()
    interval_violation_details: list[CopycatViolation] = []
    if miner_rollout_counters:
        timer_ctx = (
            monitor.timer("validation/copycat_detector") if monitor else contextlib.nullcontext()
        )
        with timer_ctx:
            (
                window_cheaters,
                window_violation_details,
                interval_cheaters,
                interval_violation_details,
                window_all_pairs,
                interval_all_pairs,
            ) = COPYCAT_TRACKER.ingest_window(target_window, miner_rollout_counters)

    if monitor:
        try:
            await monitor.log_gauge(
                "validation/copycat/window_cheaters", float(len(window_cheaters))
            )
            await monitor.log_gauge(
                "validation/copycat/interval_cheaters", float(len(interval_cheaters))
            )
        except Exception:
            pass

    violation_map: defaultdict[str, list[CopycatViolation]] = defaultdict(list)
    for violation in window_violation_details + interval_violation_details:
        violation_map[violation.miner_a].append(violation)
        violation_map[violation.miner_b].append(violation)

    if violation_map:
        for violation in window_violation_details + interval_violation_details:
            logger.warning(
                "Copycat overlap detected: miners %s & %s shared=%d denom=%d ratio=%.3f threshold=%.2f scope=%s window=%d",
                violation.miner_a,
                violation.miner_b,
                violation.shared,
                violation.denominator,
                violation.ratio,
                violation.threshold,
                violation.scope,
                violation.window_start,
            )

    cheaters_detected = window_cheaters.union(interval_cheaters)

    # Log pairwise overlap ratios to monitoring for each miner to track trends.
    # We log the maximum ratio over pairs for each miner at both window and interval scopes,
    # so dashboards can visualize proximity to thresholds over time.
    if monitor:
        try:
            logger.info("Window all pairs logging started")
            logger.info("Interval all pairs: %s", interval_all_pairs)
            logger.info("Window all pairs: %s", window_all_pairs)

            # Build per-miner max ratios
            window_max_ratio: defaultdict[str, float] = defaultdict(float)
            for v in window_all_pairs:
                window_max_ratio[v.miner_a] = max(window_max_ratio[v.miner_a], v.ratio)
                window_max_ratio[v.miner_b] = max(window_max_ratio[v.miner_b], v.ratio)

            interval_max_ratio: defaultdict[str, float] = defaultdict(float)
            for v in interval_all_pairs:
                interval_max_ratio[v.miner_a] = max(interval_max_ratio[v.miner_a], v.ratio)
                interval_max_ratio[v.miner_b] = max(interval_max_ratio[v.miner_b], v.ratio)

            # Emit gauges per miner namespace (uid/hotkey-aware)
            for miner_hk in set(list(window_max_ratio.keys()) + list(interval_max_ratio.keys())):
                uid_str = str(uid_by_hotkey.get(miner_hk, miner_hk))
                wr = float(window_max_ratio.get(miner_hk, 0.0))
                ir = float(interval_max_ratio.get(miner_hk, 0.0))
                await monitor.log_gauge(f"{uid_str}/copycat/window_max_ratio", wr)
                await monitor.log_gauge(f"{uid_str}/copycat/interval_max_ratio", ir)
                # Also log proximity to thresholds (ratio / threshold)
                await monitor.log_gauge(
                    f"{uid_str}/copycat/window_proximity",
                    wr / COPYCAT_WINDOW_THRESHOLD if COPYCAT_WINDOW_THRESHOLD > 0 else 0.0,
                )
                await monitor.log_gauge(
                    f"{uid_str}/copycat/interval_proximity",
                    ir / COPYCAT_INTERVAL_THRESHOLD if COPYCAT_INTERVAL_THRESHOLD > 0 else 0.0,
                )
        except Exception:
            pass

    for cheater in cheaters_detected:
        metrics = window_inference_counts.get(cheater)
        if metrics is None:
            metrics = {
                "valid": 0,
                "checked": 0,
                "total": 0,
                "estimated_valid": 0,
                "estimated_successful": 0,
                "estimated_unique": 0,
                "successful": 0,
                "unique": 0,
                "prompt_valid": 0,
                "prompt_mismatch": 0,
            }
            window_inference_counts[cheater] = metrics
        else:
            metrics["valid"] = 0
            metrics["estimated_valid"] = 0
            metrics["estimated_successful"] = 0
            metrics["estimated_unique"] = 0
            metrics["successful"] = 0
            metrics["unique"] = 0
        metrics[FAILURE_FLAG_KEY] = 1
        uid_str = str(uid_by_hotkey.get(cheater, cheater))
        scopes = {violation.scope for violation in violation_map.get(cheater, [])}
        ratios = ", ".join(f"{violation.ratio:.3f}" for violation in violation_map.get(cheater, []))
        with miner_log_context(uid_str, target_window):
            logger.warning(
                "Copycat gating applied (scopes=%s ratios=%s)",
                ",".join(sorted(scopes)) if scopes else "unknown",
                ratios or "n/a",
            )

    if cheaters_detected and all_valid_rollouts:
        all_valid_rollouts = [
            rollout
            for rollout in all_valid_rollouts
            if rollout.get("hotkey") not in cheaters_detected
        ]

    # Window timing end
    window_t1 = time.monotonic()
    try:
        block_end = await subtensor.get_current_block()
    except Exception:
        block_end = None
    window_seconds = window_t1 - window_t0
    window_blocks = (
        (block_end - block_beg) if (block_beg is not None and block_end is not None) else 0
    )

    # Aggregate per-miner timing stats
    def _stat(xs: list[float]) -> tuple[float, float, float]:
        if not xs:
            return 0.0, 0.0, 0.0
        return float(min(xs)), float(sum(xs) / len(xs)), float(max(xs))

    sec_min, sec_mean, sec_max = _stat(miner_seconds_list)
    blk_min, blk_mean, blk_max = _stat([float(x) for x in miner_blocks_list])

    # TODO: these monitoring measures need to be optimized later on
    if monitor:
        try:
            await monitor.log_gauge("window_timing/window_seconds", float(window_seconds))
            await monitor.log_gauge("window_timing/window_blocks", float(window_blocks))
            await monitor.log_gauge("window_timing/miner_seconds/min", float(sec_min))
            await monitor.log_gauge("window_timing/miner_seconds/mean", float(sec_mean))
            await monitor.log_gauge("window_timing/miner_seconds/max", float(sec_max))
            await monitor.log_gauge("window_timing/miner_blocks/min", float(blk_min))
            await monitor.log_gauge("window_timing/miner_blocks/mean", float(blk_mean))
            await monitor.log_gauge("window_timing/miner_blocks/max", float(blk_max))
        except Exception:
            pass

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


async def _handle_wallet_hard_failure(
    uid_str: str,
    target_window: int,
    hard_failure: bool,
    soft_failures: int,
    total_planned_checks: int,
    checked_count: int,
    total_inferences: int,
    prompt_mismatch_count: int,
    pr_total: int,
    pr_invalid_sig: int,
    pr_invalid_proof: int,
    pr_processing_err: int,
    monitor: Any,
) -> tuple[bool, dict[str, int], list[dict], tuple[int, int, int, int]]:
    """Handle wallet rejection due to hard or soft failures."""
    metrics = {
        "valid": 0,
        "checked": checked_count,
        "total": total_inferences,
        "estimated_valid": 0,
        "successful": 0,
        "estimated_successful": 0,
        "unique": 0,
        "estimated_unique": 0,
        "prompt_valid": 0,
        "prompt_mismatch": prompt_mismatch_count,
        FAILURE_FLAG_KEY: 1,
    }
    logger.info(
        f"❌ Rejected "
        f"(hard_failure={hard_failure}, "
        f"soft_failures={soft_failures}/{total_planned_checks})"
    )
    if monitor:
        await monitor.log_gauge(f"{uid_str}/had_failure", 1.0)
        try:
            await monitor.log_gauge(f"{uid_str}/prompt_valid", float(0))
            await monitor.log_gauge(f"{uid_str}/prompt_mismatch", float(prompt_mismatch_count))
        except Exception:
            pass
    return (
        True,
        metrics,
        [],
        (pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err),
    )


async def _process_wallet_window(
    wallet_addr: str,
    target_window: int,
    target_window_hash: str,
    window_rand: str,
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sat_pipeline: ValidationPipeline,
    credentials: Any,
    chain_manager: GrailChainManager,
    monitor: Any,
    uid_by_hotkey: dict[str, int],
    text_logs_emitted_by_wallet: defaultdict[str, int],
    text_log_limit: int,
    sat_reward_low: float,
    sat_reward_high: float,
) -> tuple[
    bool,
    Optional[dict[str, int]],
    list[dict],
    tuple[int, int, int, int],
    Optional[Counter[str]],
    int,
]:
    """Validate a single wallet window file and return metrics and rollouts.

    Returns
    -------
    found_file: bool
        True if the miner's window file was found and parsed.
    metrics: Optional[dict[str, int]]
        Per-miner window metrics or None if file invalid/mismatch.
    wallet_rollouts_buffer: list[dict]
        Rollouts that passed checks for this miner (used for uploads/logging).
    processed_counts: tuple[int, int, int, int]
        (pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err) tallies.
    digest_counter: Optional[Counter[str]]
        Completion digest multiset over ALL rollouts in the file (not only
        validated ones), present only if the miner did not trigger hard/soft
        failure gates; None otherwise.
    total_inferences: int
        Total rollouts present in the window file (for denominator context).
    """
    filename = f"grail/windows/{wallet_addr}-window-{target_window}.json"
    miner_bucket = chain_manager.get_bucket_for_hotkey(wallet_addr)
    # Resolve miner UID (fallback to wallet address string)
    uid_str = str(uid_by_hotkey.get(wallet_addr, wallet_addr))

    exists = await file_exists(
        filename,
        credentials=miner_bucket if miner_bucket else credentials,
        use_write=False,
    )
    if not exists:
        logger.debug(f"No file found at {filename}")
        return False, None, [], (0, 0, 0, 0), None, 0

    logger.info("📁 Found file")
    window_data = await get_file(
        filename, credentials=miner_bucket if miner_bucket else credentials, use_write=False
    )
    if not window_data:
        logger.warning(f"Could not download {filename}")
        return True, None, [], (0, 0, 0, 0), None, 0

    file_wallet_addr = window_data.get("wallet")
    window_start = window_data.get("window_start")
    inferences = window_data.get("inferences", [])
    if file_wallet_addr != wallet_addr:
        got_uid = uid_by_hotkey.get(file_wallet_addr)
        got_id = got_uid if got_uid is not None else "unknown"
        logger.warning(f"UID mismatch in {filename}: expected {uid_str}, got {got_id}")
        return True, None, [], (0, 0, 0, 0), None, 0
    if window_start != target_window:
        logger.warning(
            f"Window mismatch in {filename}: expected {target_window}, got {window_start}"
        )
        return True, None, [], (0, 0, 0, 0), None, 0

    # Continue with the rest of the function inside the context
    total_inferences = len(inferences)
    groups_map = defaultdict(list)
    # Build group membership and assign canonical indices in file order (simple, deterministic)
    group_index_by_id: dict[str, int] = {}
    for idx, inf in enumerate(inferences):
        raw_gid = inf.get("rollout_group")
        if raw_gid is not None:  # Only process inferences with rollout_group
            group_id = str(raw_gid)
            groups_map[group_id].append(idx)
            if group_id not in group_index_by_id:
                group_index_by_id[group_id] = len(group_index_by_id)  # 0, 1, 2, ...

    # Determine whether to check all rollouts or sample GRPO groups.
    # Sampling is deterministic per wallet+window via a seeded RNG to keep
    # validators consistent and discourage gaming.
    if total_inferences <= MAX_SAMPLES_PER_MINER_THRESHOLD:
        indices_to_check = list(range(total_inferences))
        logger.info(f"🔍 Verifying all {total_inferences} rollouts")
    else:
        indices_to_check = []
        num_groups = len(groups_map)

        # Choose how many GRPO groups to spot-check (at least 1, at most all).
        groups_to_check = max(1, min(num_groups, int(num_groups * SAMPLE_RATE)))

        groups_to_check = min(groups_to_check, MAX_SAMPLES_PER_MINER // ROLLOUTS_PER_PROBLEM)

        # Derive a deterministic RNG seed from miner wallet,
        # window randomness, and validator hotkey.
        # Selection is stable per validator and window, and
        # harder to game via order-dependent behavior.
        # We take the first 8 bytes of the SHA-256 to get
        # a 64-bit integer seed.
        seed_material = (f"{wallet_addr}:{window_rand}:{wallet.hotkey.ss58_address}").encode()
        seed_int = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
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
                    inferences[i].get("commit", {}), sort_keys=True, separators=(",", ":")
                )
                dig.update(hashlib.sha256(commit_json.encode()).digest())
            return dig.hexdigest()

        # Canonicalize group ordering by sorting ids by their
        # content-derived digest. This avoids dict insertion
        # order or arbitrary ids affecting RNG sampling.
        group_keys = sorted(groups_map.keys(), key=lambda gid: _group_digest(groups_map[gid]))
        # Deterministically sample groups without replacement using the seeded RNG.
        selected_groups = rng.sample(group_keys, groups_to_check)
        for group_id in selected_groups:
            indices_to_check.extend(groups_map[group_id])
        # Sort indices to make the per-rollout verification order deterministic.
        indices_to_check.sort()
        logger.info(
            f"📊 Spot checking {len(indices_to_check)}/{total_inferences} "
            f"rollouts from {groups_to_check}/{num_groups} groups!"
        )

    # Per-wallet counters for metrics and gating
    valid_count = 0
    checked_count = 0
    successful_rollouts = 0
    unique_rollouts: set[str] = set()
    nonces_seen = set()
    rollout_groups = defaultdict(list)
    wallet_rollouts_buffer = []
    soft_failures = 0
    hard_failure = False
    soft_gate_triggered = False
    total_planned_checks = len(indices_to_check)
    # Prompt-prefix validity metrics
    prompt_valid_count = 0
    prompt_mismatch_count = 0

    # Compute soft failure threshold for wallet gating
    soft_fail_cutoff = max(
        1, math.ceil(STOCHASTIC_CHECK_FAILURE_THRESHOLD * max(1, total_planned_checks))
    )

    # Per-wallet processing counters
    pr_total = 0
    pr_invalid_sig = 0
    pr_invalid_proof = 0
    pr_processing_err = 0

    for _, inference_idx in enumerate(indices_to_check):
        # Refresh watchdog heartbeat on each rollout
        try:
            global HEARTBEAT
            HEARTBEAT = time.monotonic()
        except Exception:
            pass
        inference = inferences[inference_idx]
        checked_count += 1

        try:
            # Window consistency check
            if inference["window_start"] != target_window:
                hard_failure = True
                logger.warning("Window mismatch in inference; invalidating uid")
                break
            if inference["block_hash"] != target_window_hash:
                hard_failure = True
                logger.warning("Block hash mismatch in inference; invalidating uid")
                break
            nonce = inference["nonce"]
            if nonce in nonces_seen:
                hard_failure = True
                logger.warning(f"Duplicate nonce {nonce} in window; invalidating uid")
                break
            nonces_seen.add(nonce)
            if not verify_rollout_signature(inference):
                pr_invalid_sig += 1
                hard_failure = True
                logger.warning("Invalid signature; invalidating uid")
                break
            expected_seed = f"{wallet_addr}-{target_window_hash}-{nonce}"
            if inference.get("sat_seed") != expected_seed:
                hard_failure = True
                logger.warning("Invalid SAT seed in inference; invalidating uid")
                break
            try:
                commit_data = inference["commit"]
                try:
                    rollout_meta = commit_data.get("rollout", {})
                    total_reward = rollout_meta.get("total_reward", None)
                    if not isinstance(total_reward, (int, float)):
                        logger.debug("Missing or invalid total_reward; skipping inference")
                        continue
                    low = float(sat_reward_low)
                    high = float(sat_reward_high)
                    tr = float(total_reward)
                    lo = (
                        float("-inf")
                        if low == float("-inf")
                        else low - max(abs(low) * REWARD_REL_TOL, REWARD_ABS_TOL)
                    )
                    hi = (
                        float("inf")
                        if high == float("inf")
                        else high + max(abs(high) * REWARD_REL_TOL, REWARD_ABS_TOL)
                    )
                    if not (lo <= tr <= hi):
                        hard_failure = True
                        logger.warning(
                            f"Reward {tr:.6f} outside tolerant bounds "
                            f"[{lo:.6f}, {hi:.6f}] (base=[{low:.6f}, {high:.6f}]); "
                            f"invalidating uid"
                        )
                        break
                except Exception:
                    pass

                # rollout_group is guaranteed to be non-None due to hard failure check above
                rollout_group = str(inference.get("rollout_group"))
                rollout_groups[rollout_group].append(inference)

                # Derive canonical SAT seed/difficulty based on deterministic file-order index
                gid = rollout_group
                idx = group_index_by_id.get(gid, 0)
                can_seed, can_diff = derive_canonical_sat(wallet_addr, target_window_hash, idx)
                satp = commit_data.setdefault("sat_problem", {})
                satp["seed"] = can_seed
                satp["difficulty"] = can_diff

                challenge_rand = window_rand

                # Use validation pipeline
                from grail.validation.context import ValidationContext

                ctx = ValidationContext(
                    commit=commit_data,
                    prover_address=wallet_addr,
                    challenge_randomness=challenge_rand,
                    model=model,
                    tokenizer=tokenizer,
                    device=model.device,
                )

                if monitor:
                    with monitor.timer("validation/rollout_verification"):
                        is_valid, checks = sat_pipeline.validate(ctx)
                else:
                    is_valid, checks = sat_pipeline.validate(ctx)

                pr_total += 1
                # Hard checks are cryptographic/proof constraints; any failure rejects wallet
                hard_valid = all(checks.get(k, False) for k in HARD_CHECK_KEYS)
                soft_valid = checks.get(SOFT_CHECK_KEY, True)

                # Log specific check failures for debugging
                if not hard_valid:
                    # Find the first hard check that failed (due to early return pattern)
                    failed_hard_check = None
                    for k in HARD_CHECK_KEYS:
                        if not checks.get(k, False):
                            failed_hard_check = k
                            break
                    if failed_hard_check:
                        logger.debug(f"CHECK_FAILURE type=hard failed_check={failed_hard_check}")

                if not soft_valid:
                    logger.debug(f"CHECK_FAILURE type=soft failed_check={SOFT_CHECK_KEY}")

                # Track prompt validity as a metric (not gating)
                try:
                    if (
                        bool(checks.get("tokens_valid"))
                        and bool(checks.get("proof_valid"))
                        and bool(checks.get("sat_problem_valid"))
                    ):
                        if bool(checks.get("prompt_valid")):
                            prompt_valid_count += 1
                        else:
                            prompt_mismatch_count += 1
                except Exception:
                    pass
                if not hard_valid:
                    pr_invalid_proof += 1
                    hard_failure = True
                    logger.warning("Hard verification failed; invalidating uid")
                    break
                if not soft_valid:
                    soft_failures += 1
                    # Log soft trigger progression and record per-miner metrics
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"soft_failures={soft_failures}/{total_planned_checks}! "
                            f"If it exceeds {soft_fail_cutoff} it will fail the wallet "
                            "for this window"
                        )
                    if monitor:
                        try:
                            await monitor.log_gauge(
                                f"{uid_str}/soft_failures", float(soft_failures)
                            )
                            rate_checked = float(soft_failures) / float(
                                max(1, total_planned_checks)
                            )
                            await monitor.log_gauge(f"{uid_str}/soft_failure_rate", rate_checked)
                        except Exception:
                            pass
                    if soft_failures >= soft_fail_cutoff:
                        soft_gate_triggered = True
                        logger.warning(
                            f"Soft-check failures threshold reached "
                            f"({soft_failures}/{total_planned_checks}); "
                            f"invalidating uid"
                        )
                        break
            except Exception as e:
                logger.warning(f"Rollout verification error: {e}")
                continue

            valid_count += 1
            if (
                logger.isEnabledFor(logging.DEBUG)
                and text_logs_emitted_by_wallet[wallet_addr] < text_log_limit
            ):
                try:
                    tokens = commit_data.get("tokens", [])
                    if isinstance(tokens, list) and tokens:
                        rollout_meta = commit_data.get("rollout", {})
                        prompt_len = int(rollout_meta.get("prompt_length"))
                        completion_len = int(rollout_meta.get("completion_length", 0) or 0)
                        if completion_len > 0 and prompt_len >= 0:
                            completion_ids = tokens[prompt_len : prompt_len + completion_len]
                        else:
                            completion_ids = tokens[prompt_len:]
                        problem_text = tokenizer.decode(
                            tokens[:prompt_len], skip_special_tokens=False
                        )
                        text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                        reward_val = rollout_meta.get("total_reward", float("nan"))
                        adv_val = rollout_meta.get("advantage", float("nan"))
                        success_val = rollout_meta.get("success", False)
                        logger.debug(
                            f"TEXT[validate] nonce={nonce} "
                            f"reward={float(reward_val):.3f} adv={float(adv_val):.3f} success={bool(success_val)} text={text}"
                        )
                        if monitor:
                            await monitor.log_artifact(
                                f"{uid_str}/validation/sample_text",
                                {
                                    "window": target_window,
                                    "group": uid_str,
                                    "nonce": nonce,
                                    "reward": float(reward_val),
                                    "advantage": float(adv_val),
                                    "success": bool(success_val),
                                    "text": f"Problem:\n{problem_text}\n\nCompletion:\n{text}",
                                },
                                "text",
                            )
                        text_logs_emitted_by_wallet[wallet_addr] += 1
                except Exception:
                    pass

            if rollout_meta.get("success", False):
                successful_rollouts += 1

            # Count unique rollouts by hashing completion token IDs
            unique_rollouts = _update_unique_rollouts(unique_rollouts, commit_data, rollout_meta)

            wallet_rollouts_buffer.append(inference)
        except Exception as e:
            logger.debug(f"Error processing inference: {e}")
            pr_processing_err += 1
            continue

    # Handle wallet rejection due to hard or soft failures
    if hard_failure or soft_gate_triggered:
        failure_result = await _handle_wallet_hard_failure(
            uid_str,
            target_window,
            hard_failure,
            soft_failures,
            total_planned_checks,
            checked_count,
            total_inferences,
            prompt_mismatch_count,
            pr_total,
            pr_invalid_sig,
            pr_invalid_proof,
            pr_processing_err,
            monitor,
        )
        return failure_result + (None, total_inferences)

    # Verify GRPO groups - hard requirement validation
    for group_id, group_rollouts in rollout_groups.items():
        # Check if this looks like a complete group (should have 4 rollouts for GRPO)
        expected_group_size = ROLLOUTS_PER_PROBLEM
        if len(group_rollouts) != expected_group_size:
            hard_failure = True
            logger.warning(
                f"GRPO group {group_id} has {len(group_rollouts)} rollouts, "
                f"expected {expected_group_size}; "
                f"invalidating uid"
            )
            break

        # Verify advantages sum to approximately zero
        advantages = []
        for r in group_rollouts:
            adv = r.get("commit", {}).get("rollout", {}).get("advantage", 0.0)
            advantages.append(adv)
        advantage_sum = sum(advantages)

        if abs(advantage_sum) > GRPO_ADV_SUM_TOLERANCE:
            hard_failure = True
            logger.warning(
                f"GRPO group {group_id} advantages don't sum to 0: "
                f"{advantage_sum} (tolerance: {GRPO_ADV_SUM_TOLERANCE}); "
                f"invalidating uid"
            )
            break

        # Verify all rollouts share the same base SAT problem
        base_seeds = []
        for r in group_rollouts:
            sat_problem = r.get("commit", {}).get("sat_problem", {})
            base_seeds.append(sat_problem.get("seed"))

        if len(set(base_seeds)) != 1:
            hard_failure = True
            logger.warning(
                f"GRPO group {group_id} has different base problems: {set(base_seeds)}; "
                f"invalidating uid"
            )
            break

    # Check for hard failure after GRPO validation
    if hard_failure:
        failure_result = await _handle_wallet_hard_failure(
            uid_str,
            target_window,
            hard_failure,
            soft_failures,
            total_planned_checks,
            checked_count,
            total_inferences,
            prompt_mismatch_count,
            pr_total,
            pr_invalid_sig,
            pr_invalid_proof,
            pr_processing_err,
            monitor,
        )
        return failure_result + (None, total_inferences)

    # Extrapolate from sample to estimate total for weight computation
    # These estimates are going to be used in logging in wandb and grafana later on too
    sample_pass_rate = (valid_count / checked_count) if checked_count > 0 else 0
    estimated_valid = int(total_inferences * sample_pass_rate)

    unique_sample_pass_rate = (len(unique_rollouts) / checked_count) if checked_count > 0 else 0
    estimated_unique = int(total_inferences * unique_sample_pass_rate)

    success_rate = (successful_rollouts / checked_count) if checked_count > 0 else 0
    estimated_successful = int(total_inferences * success_rate)

    # Store metrics for this miner
    metrics = {
        "valid": valid_count,
        "checked": checked_count,
        "total": total_inferences,
        "estimated_valid": estimated_valid,
        "successful": successful_rollouts,
        "estimated_successful": estimated_successful,
        "unique": len(unique_rollouts),
        "estimated_unique": estimated_unique,
        "prompt_valid": prompt_valid_count,
        "prompt_mismatch": prompt_mismatch_count,
        FAILURE_FLAG_KEY: 0,
    }

    if wallet_rollouts_buffer:
        logger.info(
            f"✅ {valid_count}/{checked_count} checked, "
            f"~{estimated_valid}/{total_inferences} estimated valid, "
            f"{successful_rollouts}/{total_inferences} successful, "
            f"{estimated_successful}/{total_inferences} estimated successful, "
            f"{len(unique_rollouts)}/{total_inferences} unique, "
            f"{estimated_unique}/{total_inferences} estimated unique"
        )

    if monitor:
        await monitor.log_gauge(f"{uid_str}/had_failure", 0.0)
        try:
            await monitor.log_gauge(f"{uid_str}/prompt_valid", float(prompt_valid_count))
            await monitor.log_gauge(f"{uid_str}/prompt_mismatch", float(prompt_mismatch_count))
        except Exception:
            pass

    # Build digest counter over ALL rollouts in the file (not just validated),
    # as long as the miner has not failed soft/hard checks for this window.
    rollout_digest_counter: Counter[str] = Counter()
    for inf in inferences:
        commit_data = inf.get("commit", {})
        rollout_meta = commit_data.get("rollout", {})
        dig = compute_completion_digest(commit_data, rollout_meta)
        if dig is not None:
            rollout_digest_counter[dig] += 1

    return (
        True,
        metrics,
        wallet_rollouts_buffer,
        (pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err),
        rollout_digest_counter,
        total_inferences,
    )


async def _upload_rollouts(
    target_window: int, all_valid_rollouts: list[dict], credentials: Any
) -> None:
    """Upload validated rollouts to object storage and Hugging Face."""
    upload_success = await upload_valid_rollouts(target_window, all_valid_rollouts, credentials)
    if upload_success:
        logger.info(f"📤 Uploaded {len(all_valid_rollouts)} valid rollouts for training")
    else:
        logger.warning("⚠️ Failed to upload valid rollouts for training")

    # Upload to Hugging Face dataset for community access
    try:
        hf_success = await upload_to_huggingface(all_valid_rollouts, target_window)
        if hf_success:
            logger.info(f"🤗 Uploaded {len(all_valid_rollouts)} rollouts to Hugging Face dataset")
        else:
            logger.debug("Failed to upload to Hugging Face (may need HF_TOKEN)")
    except Exception as e:
        logger.debug(f"Hugging Face upload error: {e}")


# --------------------------------------------------------------------------- #
#                          Miner Logging Helpers                              #
# --------------------------------------------------------------------------- #


async def _log_top_miners_by_rollout_count(
    window_inference_counts: dict[str, dict[str, Any]],
    uid_by_hotkey: dict[str, str],
    target_window: int,
    monitor: Optional[Any],
    top_n: int = 5,
) -> None:
    """Log top miners by total rollout count."""
    # Create list of (hotkey, uid, rollout_count) for top performers
    miner_rollout_data = []
    for hotkey, metrics in window_inference_counts.items():
        uid = uid_by_hotkey.get(hotkey, hotkey)
        rollout_count = metrics.get("total", 0)
        miner_rollout_data.append((hotkey, uid, rollout_count))

    # Sort by rollout count (descending) and get top N
    top_miners = sorted(miner_rollout_data, key=lambda x: x[2], reverse=True)[:top_n]

    if top_miners:
        logger.info(f"🏆 Top {min(len(top_miners), top_n)} miners by rollout count:")
        for i, (_hotkey, uid, count) in enumerate(top_miners, 1):
            logger.info(f"  {i}. UID {uid}: {count} rollouts")

        # Log to monitoring as text
        if monitor:
            text_lines = []
            for rank, (_hotkey, uid, count) in enumerate(top_miners, 1):
                text_lines.append(f"{rank}. UID {uid}: {count} rollouts")

            await monitor.log_artifact(
                "validation/top_miners_by_rollout_count",
                {"window": target_window, "text": "\n".join(text_lines)},
                "text",
            )


async def _log_top_miners_by_unique_rollouts(
    window_inference_counts: dict[str, dict[str, Any]],
    uid_by_hotkey: dict[str, str],
    target_window: int,
    monitor: Optional[Any],
    top_n: int = 5,
) -> None:
    """Log top miners by unique rollouts count."""
    # Create list of (hotkey, uid, unique_count) for top performers
    miner_unique_data = []
    for hotkey, metrics in window_inference_counts.items():
        uid = uid_by_hotkey.get(hotkey, hotkey)
        unique_count = metrics.get("estimated_unique", 0)
        miner_unique_data.append((hotkey, uid, unique_count))

    # Sort by unique count (descending) and get top N
    top_miners = sorted(miner_unique_data, key=lambda x: x[2], reverse=True)[:top_n]

    if top_miners:
        logger.info(f"🌟 Top {min(len(top_miners), top_n)} miners by unique rollouts:")
        for i, (_hotkey, uid, count) in enumerate(top_miners, 1):
            logger.info(f"  {i}. UID {uid}: {count} unique rollouts")

        # Log to monitoring as text
        if monitor:
            text_lines = []
            for rank, (_hotkey, uid, count) in enumerate(top_miners, 1):
                text_lines.append(f"{rank}. UID {uid}: {count} unique rollouts")

            await monitor.log_artifact(
                "validation/top_miners_by_unique_rollouts",
                {"window": target_window, "text": "\n".join(text_lines)},
                "text",
            )


async def _log_top_miners_by_score(
    inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]],
    uid_by_hotkey: dict[str, str],
    target_window: int,
    monitor: Optional[Any],
    top_n: int = 5,
) -> None:
    """Log top miners by highest score (base_score and superlinear_score)."""
    # Create list of (hotkey, uid, base_score, superlinear_score)
    # for top performers
    miner_score_data = []
    for hotkey in inference_counts.keys():
        uid = uid_by_hotkey.get(hotkey, hotkey)
        try:
            # Calculate scores using the same logic as _aggregate_weight_inputs
            (
                total_unique,
                total_successful,
                total_estimated_valid,
                base_score,
                superlinear_score,
            ) = _aggregate_weight_inputs(hotkey, inference_counts, target_window)
            miner_score_data.append((hotkey, uid, base_score, superlinear_score))
        except Exception as e:
            logger.debug(f"[MINER hotkey={hotkey}] Error calculating score: {e}")
            continue

    # Sort by superlinear_score (descending) and get top N
    top_miners = sorted(miner_score_data, key=lambda x: x[3], reverse=True)[:top_n]

    if top_miners:
        logger.info(f"🎯 Top {min(len(top_miners), top_n)} miners by score:")
        for i, (_hotkey, uid, base_score, superlinear_score) in enumerate(top_miners, 1):
            logger.info(
                f"  {i}. UID {uid}: base={base_score:.2f}, superlinear={superlinear_score:.2f}"
            )

        # Log to monitoring as text
        if monitor:
            text_lines = []
            for rank, (_hotkey, uid, base_score, superlinear_score) in enumerate(top_miners, 1):
                text_lines.append(
                    f"{rank}. UID {uid}: base={base_score:.2f}, superlinear={superlinear_score:.2f}"
                )

            await monitor.log_artifact(
                "validation/top_miners_by_score",
                {"window": target_window, "text": "\n".join(text_lines)},
                "text",
            )


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    validate()
