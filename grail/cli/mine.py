#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import bittensor as bt
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..environments.factory import create_env
from ..environments.loop import AgentEnvLoop
from ..grail import derive_env_seed
from ..infrastructure.comms import sink_window_inferences
from ..infrastructure.drand import get_drand_beacon
from ..shared.constants import (
    BLOCK_TIME_SECONDS,
    CHALLENGE_K,
    LAYER_INDEX,
    ROLLOUTS_PER_PROBLEM,
    WINDOW_LENGTH,
)
from . import console

# --------------------------------------------------------------------------- #
#                       Constants & global singletons                         #
# --------------------------------------------------------------------------- #
logger = logging.getLogger("grail")


# --------------------------------------------------------------------------- #
#                       Styling & configuration constants                     #
# --------------------------------------------------------------------------- #
# Mining timing and safety parameters. Centralized for easy tuning and clarity.
EMA_ALPHA = 0.2  # Exponential moving average smoothing

MINER_SAFETY_BLOCKS = int(  # Safety margin blocks before window end
    os.getenv("GRAIL_MINER_SAFETY_BLOCKS", "3")
)
DEBUG_TEXT_LOG_LIMIT_PER_WINDOW = 5  # Max sample texts logged per window

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
#                        Helper Functions                                     #
# --------------------------------------------------------------------------- #


def parse_filename(
    filename: str,
) -> tuple[str | None, int | None, int | None]:
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
) -> tuple[str | None, int | None]:
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


def sign_rollout(rollout_data: dict[str, Any], wallet: bt.wallet) -> dict[str, Any]:
    """Sign a rollout using the wallet hotkey (env-agnostic)."""
    # Create challenge string from key rollout data
    episode_seed = rollout_data.get("episode_seed", rollout_data.get("sat_seed", ""))
    block_hash = rollout_data.get("block_hash", "")
    nonce = rollout_data.get("nonce", "")

    # Validate input types
    if not isinstance(block_hash, str):
        raise ValueError(f"block_hash must be str, got {type(block_hash).__name__}")
    if not isinstance(nonce, (int, str)):
        raise ValueError(f"nonce must be int or str, got {type(nonce).__name__}")

    # Use delimiter to prevent collision attacks
    challenge = f"{episode_seed}|{block_hash}|{nonce}"
    rollout_data["challenge"] = challenge
    rollout_data["hotkey"] = wallet.hotkey.ss58_address
    # Encode challenge to bytes before signing (explicit UTF-8)
    signature = wallet.hotkey.sign(data=challenge.encode("utf-8")).hex()
    rollout_data["signature"] = signature
    return rollout_data


# --------------------------------------------------------------------------- #
#                         Time & window utilities                             #
# --------------------------------------------------------------------------- #


def calculate_window_start(block_number: int) -> int:
    return (block_number // WINDOW_LENGTH) * WINDOW_LENGTH


@dataclass
class MiningTimers:
    """Tracks time estimates and exponential moving averages (EMAs).

    We keep EMAs of block time, generation time, and upload time to make
    conservative, adaptive decisions about whether there's enough time left
    in the current window to safely generate and upload another batch.
    """

    block_time_ema_s: float = float(BLOCK_TIME_SECONDS)
    gen_time_ema_s: float | None = None
    upload_time_ema_s: float | None = None
    last_block_num: int | None = None
    last_block_ts: float | None = None

    def update_block_time_ema(self, current_block: int) -> None:
        """Update the EMA for block time using observed block deltas.

        Uses the time elapsed between the last seen block and the current block
        to update an EMA of the chain's average block time.
        """
        now_ts = time.time()
        if self.last_block_num is not None and self.last_block_ts is not None:
            dn = current_block - self.last_block_num
            if dn > 0:
                dt = now_ts - self.last_block_ts
                if dt > 0.0:
                    sample_bt = dt / dn
                    self.block_time_ema_s = (
                        EMA_ALPHA * sample_bt + (1.0 - EMA_ALPHA) * self.block_time_ema_s
                    )
        self.last_block_num = current_block
        self.last_block_ts = now_ts

    def blocks_needed_for_next_gen(self) -> int:
        """Estimate how many blocks we need to finish a gen+upload safely.

        Combines gen time EMA, upload time EMA, and a safety margin (in blocks)
        to convert projected seconds into blocks remaining in the window.
        """
        est_gen_s = (
            self.gen_time_ema_s if self.gen_time_ema_s is not None else 6.0 * self.block_time_ema_s
        )
        est_upload_s = (
            self.upload_time_ema_s
            if self.upload_time_ema_s is not None
            else 1.0 * self.block_time_ema_s
        )
        safety_s = float(MINER_SAFETY_BLOCKS) * self.block_time_ema_s
        total_s = est_gen_s + est_upload_s + safety_s
        needed = max(1, math.ceil(total_s / max(0.001, self.block_time_ema_s)))
        # Never claim we need the entire window — always attempt at least one
        # generation at the start of a new window so the EMA can recover.
        return min(needed, WINDOW_LENGTH - 1)

    def update_gen_time_ema(self, duration_s: float) -> None:
        self.gen_time_ema_s = (
            duration_s
            if self.gen_time_ema_s is None
            else EMA_ALPHA * duration_s + (1.0 - EMA_ALPHA) * self.gen_time_ema_s
        )

    def update_upload_time_ema(self, duration_s: float) -> None:
        self.upload_time_ema_s = (
            duration_s
            if self.upload_time_ema_s is None
            else EMA_ALPHA * duration_s + (1.0 - EMA_ALPHA) * self.upload_time_ema_s
        )


async def has_time_for_next_generation(
    subtensor: bt.subtensor, timers: MiningTimers, window_start: int
) -> bool:
    """Return True if there is enough time left to run one more gen+upload.

    Args:
        subtensor: Bittensor subtensor client for chain reads.
        timers: Moving averages and block-time state.
        window_start: Start block number of the current window.

    Returns:
        True if blocks remaining > conservative estimate of blocks required.
    """
    current_check = await subtensor.get_current_block()  # type: ignore[await-not-coroutine]
    timers.update_block_time_ema(current_check)
    blocks_remaining = (window_start + WINDOW_LENGTH) - current_check
    needed_blocks = timers.blocks_needed_for_next_gen()
    if blocks_remaining <= needed_blocks:
        logger.warning(
            "Window %s nearly over (block %s); need %s blocks to safely "
            "finish next generation+upload.",
            window_start,
            current_check,
            needed_blocks,
        )
        return False
    return True


async def get_window_randomness(
    subtensor: bt.subtensor, window_start: int, use_drand: bool
) -> tuple[str, str]:
    """Compute randomness for the window using block hash and optional drand.

    We prefer mixing the window's block hash with the drand beacon when
    available to avoid miner-controlled randomness. Falls back to block hash.

    Returns:
        (window_block_hash, combined_randomness)
    """
    window_block_hash = await subtensor.get_block_hash(window_start)  # type: ignore[await-not-coroutine]
    if not use_drand:
        return window_block_hash, window_block_hash

    try:
        # Run drand HTTP request in thread pool to avoid blocking event loop
        drand_beacon = await asyncio.to_thread(get_drand_beacon, None)
        logger.info("🎲 Using drand randomness from round %s", drand_beacon["round"])
        combined_randomness = hashlib.sha256(
            (window_block_hash + drand_beacon["randomness"]).encode()
        ).hexdigest()
        return window_block_hash, combined_randomness
    except Exception as e:
        logger.warning("Failed to get drand, using block hash only: %s", e)
        return window_block_hash, window_block_hash


async def maybe_log_debug_sample(
    tokenizer: AutoTokenizer,
    sample: Any,
    window_start: int,
    base_nonce: int,
    monitor: Any | None,
    text_logs_emitted: int,
    text_log_limit: int,
) -> int:
    """Emit a single decoded sample for debugging, rate-limited per window.

    Args:
        tokenizer: Tokenizer for decoding tokens to text
        sample: Rollout sample to log
        window_start: Window start block
        base_nonce: Base nonce for the rollout group
        monitor: Optional monitoring client
        text_logs_emitted: Current count of emitted logs
        text_log_limit: Maximum logs to emit

    Returns:
        Updated text_logs_emitted counter
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return text_logs_emitted
    if text_logs_emitted >= text_log_limit:
        return text_logs_emitted
    if not sample:
        return text_logs_emitted

    try:
        prompt_len = int(getattr(sample, "prompt_length", 0) or 0)
        completion_len = int(getattr(sample, "completion_length", 0) or 0)
        sample_text = tokenizer.decode(sample.tokens, skip_special_tokens=False)  # type: ignore[attr-defined]
        sample_nonce = base_nonce * ROLLOUTS_PER_PROBLEM
        logger.debug(
            (
                "TEXT[mine] window=%s group=%s nonce=%s reward=%.3f "
                "adv=%.3f success=%s text=%s prompt_len=%d completion_len=%d"
            ),
            window_start,
            base_nonce,
            sample_nonce,
            float(sample.reward),
            float(sample.advantage),
            bool(sample.success),
            sample_text,
            prompt_len,
            completion_len,
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
        return text_logs_emitted + 1
    except Exception:
        return text_logs_emitted


def extract_assignment_from_rollout(rollout: Any) -> list[bool]:
    """Extract boolean assignment from rollout trajectory if available."""
    if rollout.trajectory and isinstance(rollout.trajectory[0][1], list):
        return rollout.trajectory[0][1]
    return []


def count_satisfied_clauses(sat_problem: Any, assignment: list[bool]) -> int:
    """Count how many SAT clauses are satisfied by a boolean assignment."""
    if not assignment:
        return 0
    satisfied = 0
    for clause in sat_problem.clauses:
        clause_satisfied = False
        for lit in clause:
            idx = abs(lit) - 1
            if idx < 0 or idx >= len(assignment):
                continue
            value = assignment[idx]
            if (lit > 0 and value) or (lit < 0 and not value):
                clause_satisfied = True
                break
        if clause_satisfied:
            satisfied += 1
    return satisfied


async def log_generation_timing(
    subtensor: bt.subtensor,
    timers: MiningTimers,
    window_start: int,
    generation_duration: float,
    rollout_count: int,
    monitor: Any | None,
) -> bool:
    """Log generation timing metrics and check if generation finished safely.

    Args:
        subtensor: Bittensor subtensor client for block queries
        timers: Mining timers with EMA estimates
        window_start: Start block of current window
        generation_duration: Time taken for rollout generation
        rollout_count: Number of rollouts generated
        monitor: Optional monitoring client

    Returns:
        True if generation finished with safe buffer for upload, False otherwise
    """
    post_gen_block = await subtensor.get_current_block()  # type: ignore[await-not-coroutine]
    blocks_remaining = (window_start + WINDOW_LENGTH) - post_gen_block
    time_remaining_s = blocks_remaining * timers.block_time_ema_s
    needed_blocks_for_upload = max(
        1, math.ceil((timers.upload_time_ema_s or 0.0) / max(0.001, timers.block_time_ema_s))
    )

    generation_safe = blocks_remaining > needed_blocks_for_upload

    logger.info(
        "Generation timing: %d blocks remaining, %.1fs left, upload needs %d blocks - %s",
        blocks_remaining,
        time_remaining_s,
        needed_blocks_for_upload,
        "✅ SAFE" if generation_safe else "⚠️ TIGHT",
    )

    if monitor:
        await monitor.log_gauge(
            "profiling/generation_finished_safely",
            1.0 if generation_safe else 0.0,
        )

    return generation_safe


def package_rollout_data(
    model: AutoModelForCausalLM,
    wallet: bt.wallet,
    rollout: Any,
    base_nonce: int,
    rollout_idx: int,
    total_in_group: int,
    window_start: int,
    current_block: int,
    window_block_hash: str,
    combined_randomness: str,
    use_drand: bool,
    checkpoint_window: int | None,
) -> dict:
    """Assemble the full on-chain/off-chain payload for a single rollout.

    This binds model outputs (tokens, commitments) to the randomness, model name,
    and layer via a commit-binding signature, and includes proof metadata
    required by validators.

    Args:
        model: Loaded model (for name_or_path)
        wallet: Miner wallet for signing
        rollout: Generated rollout with tokens/commitments/trajectory
        base_nonce: Base nonce for the group
        rollout_idx: Index within the group
        total_in_group: Total rollouts in group
        window_start: Window start block
        current_block: Current block
        window_block_hash: Window block hash
        combined_randomness: Challenge randomness
        use_drand: Whether drand was used
        checkpoint_window: The checkpoint window used for this rollout

    Returns:
        Signed dictionary ready to upload for validation
    """
    rollout_nonce = base_nonce * ROLLOUTS_PER_PROBLEM + rollout_idx

    # Sign commit binding (tokens, randomness, model, layer, commitments)
    from ..protocol.signatures import sign_commit_binding

    logger.debug("Signing commit binding for rollout %s", rollout_idx)
    commit_sig = sign_commit_binding(
        tokens=rollout.tokens,
        randomness_hex=combined_randomness,
        model_name=model.name_or_path,  # type: ignore[attr-defined]
        layer_index=LAYER_INDEX,
        commitments=rollout.commitments,
        wallet=wallet,
    )

    assignment = extract_assignment_from_rollout(rollout)
    # satisfied_clauses retained for backward-compat field; set to 0 in env-agnostic mode
    satisfied_clauses = 0

    payload = {
        "window_start": window_start,
        "block": current_block,
        "nonce": rollout_nonce,
        "block_hash": window_block_hash,
        "randomness": combined_randomness,
        "use_drand": use_drand,
        "rollout_group": base_nonce,
        "rollout_index": rollout_idx,
        "total_in_group": total_in_group,
        "checkpoint_window": checkpoint_window,  # Explicit checkpoint used
        "commit": {
            "tokens": rollout.tokens,
            "commitments": rollout.commitments,
            "proof_version": rollout.proof_version,
            "model": {
                "name": model.name_or_path,  # type: ignore[attr-defined]
                "layer_index": LAYER_INDEX,
            },
            "signature": commit_sig.hex(),
            "beacon": rollout.beacon,
            "rollout": {
                "trajectory": rollout.trajectory,
                "total_reward": rollout.reward,
                "advantage": rollout.advantage,
                "success": rollout.success,
                "token_logprobs": rollout.token_logprobs,
                "prompt_length": rollout.prompt_length,
                "completion_length": rollout.completion_length,
                "satisfied_clauses": satisfied_clauses,
                "assignment": assignment,
            },
        },
        "timestamp": time.time(),
    }

    return sign_rollout(payload, wallet)


async def upload_inferences_with_metrics(
    wallet: bt.wallet,
    window_start: int,
    inferences: list[dict],
    credentials: Any,
    monitor: Any | None,
) -> float:
    """Upload window payload to object storage and return elapsed seconds.

    Args:
        wallet: Miner wallet for authentication.
        window_start: Start block of the window being uploaded.
        inferences: List of rollout data to upload.
        credentials: Object storage credentials.
        monitor: Optional monitoring client for timing metrics.

    Returns:
        Upload duration in seconds.
    """
    upload_start = time.time()
    if monitor:
        with monitor.timer("profiling/upload"):
            await sink_window_inferences(
                wallet,
                window_start,
                inferences,
                credentials,
            )
    else:
        await sink_window_inferences(
            wallet,
            window_start,
            inferences,
            credentials,
        )
    return time.time() - upload_start


async def _pipelined_generation_loop(
    pipeline_engine: Any,
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subtensor: bt.subtensor,
    window_start: int,
    window_block_hash: str,
    combined_randomness: str,
    timers: MiningTimers,
    monitor: Any | None,
    use_drand: bool,
    checkpoint_window: int | None,
    env_id: str | None = None,
    env_params: dict[str, Any] | None = None,
) -> list[dict]:
    """Generate rollouts using the 3-GPU pipeline engine.

    Uses PipelinedMiningEngine for overlapped generation (GPU 0),
    proof computation (GPU 1), and kernel eval (GPU 2).

    Mirrors the time-budgeting and packaging logic of
    ``generate_rollouts_for_window`` but delegates generation to the pipeline.
    """
    from ..environments.factory import create_env
    from ..grail import derive_env_seed
    from ..shared.constants import CHALLENGE_K, ROLLOUTS_PER_PROBLEM

    inferences: list[dict] = []
    start_time = time.time()
    problem_count = 0
    text_logs_emitted = 0

    while True:
        current_block = await subtensor.get_current_block()  # type: ignore[await-not-coroutine]
        timers.update_block_time_ema(current_block)
        current_window = calculate_window_start(current_block)
        if current_window > window_start:
            break

        blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
        needed_blocks = timers.blocks_needed_for_next_gen()
        if blocks_remaining <= needed_blocks:
            logger.info(
                "Pipeline: stopping — %s blocks remain, need %s",
                blocks_remaining,
                needed_blocks,
            )
            break

        try:
            gen_start = time.time()
            problem_count += 1

            problem_index = max(0, problem_count - 1)
            seed_int = derive_env_seed(wallet.hotkey.ss58_address, window_block_hash, problem_index)
            base_nonce = problem_index

            def _env_factory() -> Any:
                return create_env(env_id=env_id, env_params=env_params)

            grpo_rollouts = await pipeline_engine.generate_grpo_group(
                _env_factory,
                ROLLOUTS_PER_PROBLEM,
                combined_randomness,
                wallet,
                seed=seed_int,
            )

            if grpo_rollouts:
                text_logs_emitted = await maybe_log_debug_sample(
                    tokenizer,
                    grpo_rollouts[0],
                    window_start,
                    base_nonce,
                    monitor,
                    text_logs_emitted,
                    DEBUG_TEXT_LOG_LIMIT_PER_WINDOW,
                )

            # Completion length gate
            short_rollouts = [
                (i, r.completion_length)
                for i, r in enumerate(grpo_rollouts)
                if r.completion_length < CHALLENGE_K
            ]
            if short_rollouts:
                logger.warning(
                    "Pipeline: dropping group %d — %d/%d rollouts too short",
                    base_nonce,
                    len(short_rollouts),
                    len(grpo_rollouts),
                )
                timers.update_gen_time_ema(time.time() - gen_start)
                continue

            for rollout_idx, rollout in enumerate(grpo_rollouts):
                rollout_data = package_rollout_data(
                    model,
                    wallet,
                    rollout,
                    base_nonce,
                    rollout_idx,
                    len(grpo_rollouts),
                    window_start,
                    current_block,
                    window_block_hash,
                    combined_randomness,
                    use_drand,
                    checkpoint_window,
                )
                inferences.append(rollout_data)

            gen_duration = time.time() - gen_start
            # Cap outliers: don't let a single slow group poison the EMA
            # beyond the window length (WINDOW_LENGTH * BLOCK_TIME_SECONDS)
            max_gen_s = float(WINDOW_LENGTH * BLOCK_TIME_SECONDS)
            timers.update_gen_time_ema(min(gen_duration, max_gen_s))
            await asyncio.sleep(0.01)

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error("Pipeline CUDA error: %s", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.warning("Pipeline generation failed: %s", e)
            continue

    elapsed_time = time.time() - start_time
    logger.info(
        "Pipeline: generated %s rollouts in %.1fs for window %s",
        len(inferences),
        elapsed_time,
        window_start,
    )
    return inferences


async def generate_rollouts_for_window(
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subtensor: bt.subtensor,
    window_start: int,
    window_block_hash: str,
    combined_randomness: str,
    timers: MiningTimers,
    monitor: Any | None,
    use_drand: bool,
    checkpoint_window: int | None,
    env_id: str | None = None,
    env_params: dict[str, Any] | None = None,
    generation_params: dict[str, Any] | None = None,
) -> list[dict]:
    """Generate as many GRPO rollouts as safely possible within a window.

    Core loop responsibilities:
      - Respect time budget using EMAs (stop before window end)
      - Periodically clear CUDA cache to reduce fragmentation
      - Track and log per-window metrics
      - Package each rollout with commit-binding signatures and proofs

    Args:
        wallet: Miner wallet for signing and authentication.
        model: Loaded model instance.
        tokenizer: Loaded tokenizer instance.
        subtensor: Bittensor client for chain reads.
        window_start: Start block of the current window.
        window_block_hash: Block hash at window start.
        combined_randomness: Per-window randomness for challenges.
        timers: EMA-based timing estimates for safety.
        monitor: Optional monitoring client for metrics.
        use_drand: Whether drand was used in randomness generation.
        checkpoint_window: The checkpoint window used for this generation
        env_id: Environment identifier from checkpoint metadata
        env_params: Environment parameters from checkpoint metadata
        generation_params: Generation parameters from checkpoint metadata

    Returns:
        List of signed rollout data ready for upload.
    """
    # Window generation state and metrics
    inferences: list[dict] = []
    start_time = time.time()
    inference_count = 0  # Total number of problems attempted in this window
    successful_rollouts = 0
    failed_rollouts = 0
    total_reward = 0.0
    # Avoid flooding logs in debug mode
    text_logs_emitted = 0  # Running count of emitted debug texts
    problem_count = 0

    device = model.device  # type: ignore[attr-defined]
    # Batch size for parallel rollout generation (tune per node for memory/throughput)
    batch_size = int(os.getenv("GRAIL_GENERATION_BATCH_SIZE", "2"))
    if batch_size > ROLLOUTS_PER_PROBLEM:
        logger.warning(
            "GRAIL_GENERATION_BATCH_SIZE=%d exceeds ROLLOUTS_PER_PROBLEM=%d; capping at %d",
            batch_size,
            ROLLOUTS_PER_PROBLEM,
            ROLLOUTS_PER_PROBLEM,
        )
        batch_size = ROLLOUTS_PER_PROBLEM
    # Create AgentEnvLoop with generation parameters from checkpoint metadata (if available)
    # Validate and clamp generation params to safe ranges
    generation_params = generation_params or {}

    def clamp(value: float | int, min_val: float | int, max_val: float | int) -> float | int:
        """Clamp value to [min_val, max_val] range."""
        return max(min_val, min(value, max_val))

    max_tokens = clamp(generation_params["max_tokens"], 1, 16384)
    temperature = clamp(generation_params.get("temperature", 0.7), 0.01, 2.0)
    top_p = clamp(generation_params.get("top_p", 0.95), 0.0, 1.0)
    top_k = clamp(generation_params.get("top_k", 50), 0, 1000)
    repetition_penalty = clamp(generation_params.get("repetition_penalty", 1.1), 1.0, 2.0)

    loop = AgentEnvLoop(
        model,
        tokenizer,
        device,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k) if top_k > 0 else None,
        repetition_penalty=float(repetition_penalty),
    )
    if batch_size > 1:
        logger.info("Using batch_size=%d for parallel rollout generation", batch_size)

    while True:
        current_block = await subtensor.get_current_block()  # type: ignore[await-not-coroutine]
        timers.update_block_time_ema(current_block)
        current_window = calculate_window_start(current_block)
        if current_window > window_start:
            logger.info("Window %s has ended, moving to next window", window_start)
            break

        blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
        needed_blocks = timers.blocks_needed_for_next_gen()
        if blocks_remaining <= needed_blocks:
            logger.info(
                (
                    "Stopping generation: %s blocks remain, need %s "
                    "(gen≈%.1fs, upload≈%.1fs, block≈%.2fs)"
                ),
                blocks_remaining,
                needed_blocks,
                (timers.gen_time_ema_s or 0.0),
                (timers.upload_time_ema_s or 0.0),
                timers.block_time_ema_s,
            )
            break

        try:
            gen_start = time.time()
            problem_count += 1
            inference_count += 1

            logger.info(
                "⚡ Generating GRPO rollouts for problem %s (block %s/%s)...",
                problem_count,
                current_block,
                window_start + WINDOW_LENGTH - 1,
            )

            # Periodically reclaim free memory — helpful for long runs
            if inference_count % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(
                    "GPU memory allocated: %s MB",
                    f"{torch.cuda.memory_allocated() / 1024**2:.2f}",
                )

            # Deterministically derive environment seed from miner+window+index
            problem_index = max(0, problem_count - 1)
            seed_int = derive_env_seed(wallet.hotkey.ss58_address, window_block_hash, problem_index)
            # Use deterministic problem index as rollout_group identifier
            base_nonce = problem_index
            logger.debug(
                ("MINER SEED DERIVATION: hotkey=%s window_hash=%s problem_index=%d -> seed=%d"),
                wallet.hotkey.ss58_address[:12],
                window_block_hash[:12],
                problem_index,
                seed_int,
            )

            # Generate GRPO rollouts using AgentEnvLoop
            # Factory uses cached task source automatically (no manual instantiation needed)
            # Use environment configuration from checkpoint metadata if available
            def _env_factory():
                return create_env(env_id=env_id, env_params=env_params)

            # Time the rollout generation for both logging and monitoring
            rollout_gen_start = time.time()
            if monitor:
                with monitor.timer("profiling/rollout_generation"):
                    grpo_rollouts = await asyncio.to_thread(
                        loop.run_grpo_group,
                        _env_factory,
                        ROLLOUTS_PER_PROBLEM,
                        combined_randomness,
                        wallet,
                        batch_size=batch_size,
                        seed=seed_int,
                    )
            else:
                grpo_rollouts = await asyncio.to_thread(
                    loop.run_grpo_group,
                    _env_factory,
                    ROLLOUTS_PER_PROBLEM,
                    combined_randomness,
                    wallet,
                    batch_size=batch_size,
                    seed=seed_int,
                )
            rollout_gen_duration = time.time() - rollout_gen_start

            if grpo_rollouts:
                text_logs_emitted = await maybe_log_debug_sample(
                    tokenizer,
                    grpo_rollouts[0],
                    window_start,
                    base_nonce,
                    monitor,
                    text_logs_emitted,
                    DEBUG_TEXT_LOG_LIMIT_PER_WINDOW,
                )

            successful_count = sum(1 for r in grpo_rollouts if r.success)
            mean_reward = (
                sum(r.reward for r in grpo_rollouts) / len(grpo_rollouts) if grpo_rollouts else 0
            )
            logger.info(
                "GRPO batch: %s/%s successful, mean reward: %.3f, generation time: %.2fs",
                successful_count,
                len(grpo_rollouts),
                mean_reward,
                rollout_gen_duration,
            )

            # Check generation timing and log metrics
            await log_generation_timing(
                subtensor, timers, window_start, rollout_gen_duration, len(grpo_rollouts), monitor
            )

            if problem_count % 2 == 0:
                elapsed = time.time() - start_time
                rollouts_per_sec = (len(inferences) / elapsed) if elapsed > 0 else 0
                logger.info(
                    ("📊 Progress: %s rollouts from %s problems in %.1fs (%.1f rollouts/sec)"),
                    len(inferences),
                    problem_count,
                    elapsed,
                    rollouts_per_sec,
                )
                if monitor:
                    await monitor.log_gauge("mining/rollouts_generated", len(inferences))
                    await monitor.log_gauge("mining/problems_processed", problem_count)
                    await monitor.log_gauge("mining/rollouts_per_second", rollouts_per_sec)
                    if successful_rollouts + failed_rollouts > 0:
                        success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
                        await monitor.log_gauge("mining/success_rate", success_rate)

            # ─────────────────────────────────────────────────────────────────────
            # COMPLETION LENGTH GATE: Drop entire group if any rollout is too short
            # ─────────────────────────────────────────────────────────────────────
            # Validators require at least CHALLENGE_K tokens in the completion region
            # to perform cryptographic verification (sketch checks at k=32 positions).
            # If any rollout in the group has completion_length < CHALLENGE_K, the
            # validator will reject that rollout. Rather than waste bandwidth uploading
            # a partially valid group, we drop the entire group preemptively.
            short_rollouts = [
                (i, r.completion_length)
                for i, r in enumerate(grpo_rollouts)
                if r.completion_length < CHALLENGE_K
            ]
            if short_rollouts:
                short_details = ", ".join(f"idx={i}:len={length}" for i, length in short_rollouts)
                logger.warning(
                    "Dropping group %d: %d/%d rollouts have completion < %d tokens (%s)",
                    base_nonce,
                    len(short_rollouts),
                    len(grpo_rollouts),
                    CHALLENGE_K,
                    short_details,
                )
                # Skip packaging this group entirely; continue to next problem
                timers.update_gen_time_ema(time.time() - gen_start)
                continue

            # Package each rollout with signatures and proofs for validation
            for rollout_idx, rollout in enumerate(grpo_rollouts):
                rollout_data = package_rollout_data(
                    model,
                    wallet,
                    rollout,
                    base_nonce,
                    rollout_idx,
                    len(grpo_rollouts),
                    window_start,
                    current_block,
                    window_block_hash,
                    combined_randomness,
                    use_drand,
                    checkpoint_window,
                )
                inferences.append(rollout_data)

                if rollout.success:
                    successful_rollouts += 1
                    total_reward += rollout.reward
                    if monitor:
                        await monitor.log_counter("mining/successful_rollouts")
                        await monitor.log_histogram("mining/reward_distribution", rollout.reward)
                else:
                    failed_rollouts += 1
                    if monitor:
                        await monitor.log_counter("mining/failed_rollouts")

            timers.update_gen_time_ema(time.time() - gen_start)
            await asyncio.sleep(0.01)

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error("CUDA error at inference %s: %s", inference_count, e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.warning("Failed to generate inference %s: %s", inference_count, e)
            continue

    elapsed_time = time.time() - start_time
    avg_gen_time = timers.gen_time_ema_s or 0.0

    logger.info(
        "🎯 Generated %s rollouts in %.1fs for window %s (avg gen time: %.2fs/problem)",
        len(inferences),
        elapsed_time,
        window_start,
        avg_gen_time,
    )
    if monitor:
        await monitor.log_counter("mining/windows_completed")
        await monitor.log_gauge(
            "profiling/window_duration",
            elapsed_time,
        )
        await monitor.log_gauge("mining/total_rollouts_in_window", len(inferences))
        await monitor.log_gauge(
            "profiling/average_generation_time",
            avg_gen_time,
        )
        if successful_rollouts + failed_rollouts > 0:
            final_success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
            await monitor.log_gauge("mining/final_success_rate", final_success_rate)
        if successful_rollouts > 0:
            avg_reward = total_reward / successful_rollouts
            await monitor.log_gauge("mining/average_reward", avg_reward)

    return inferences


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def register(app: typer.Typer) -> None:
    app.command("mine")(mine)


# (Watchdog removed; handled by BaseNeuron in MinerNeuron)


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
    """Mine GRPO rollouts for SAT problems using GRAIL proofs.

    Stage 2: delegate to MinerNeuron lifecycle to keep behavior identical
    while standardizing the long-running process management.
    """
    from ..neurons import MinerNeuron

    asyncio.run(MinerNeuron(use_drand=use_drand).main())


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #
def main() -> None:
    mine()
