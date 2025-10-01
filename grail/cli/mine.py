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
from typing import Any, Optional

import bittensor as bt
import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..environments import SATRolloutGenerator, generate_sat_problem
from ..grail import derive_canonical_sat
from ..infrastructure.comms import sink_window_inferences
from ..infrastructure.drand import get_drand_beacon
from ..infrastructure.network import create_subtensor
from ..shared.constants import LAYER_INDEX, ROLLOUTS_PER_PROBLEM, WINDOW_LENGTH
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
DEFAULT_BLOCK_TIME_S = 12.0  # Bittensor block time in seconds
MINER_SAFETY_BLOCKS = int(  # Safety margin blocks before window end
    os.getenv("GRAIL_MINER_SAFETY_BLOCKS", "1")
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

        if not isinstance(signature, str):
            return False

        keypair = bt.Keypair(ss58_address=hotkey)
        result = keypair.verify(data=challenge, signature=bytes.fromhex(signature))
        return bool(result)
    except Exception:
        return False


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

    block_time_ema_s: float = DEFAULT_BLOCK_TIME_S
    gen_time_ema_s: Optional[float] = None
    upload_time_ema_s: Optional[float] = None
    last_block_num: Optional[int] = None
    last_block_ts: Optional[float] = None

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
        return max(1, math.ceil(total_s / max(0.001, self.block_time_ema_s)))

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
    current_check = await subtensor.get_current_block()
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
    window_block_hash = await subtensor.get_block_hash(window_start)
    if not use_drand:
        return window_block_hash, window_block_hash

    try:
        drand_beacon = get_drand_beacon(None)
        logger.info("🎲 Using drand randomness from round %s", drand_beacon["round"])
        combined_randomness = hashlib.sha256(
            (window_block_hash + drand_beacon["randomness"]).encode()
        ).hexdigest()
        return window_block_hash, combined_randomness
    except Exception as e:
        logger.warning("Failed to get drand, using block hash only: %s", e)
        return window_block_hash, window_block_hash


def compute_difficulty(inference_count: int) -> float:
    """Compute SAT problem difficulty based on inference count.

    Difficulty increases with more attempts to prevent easy problems
    from being solved repeatedly. Capped at 0.9 for reasonable solve times.
    """
    return min(0.9, 0.3 + (inference_count * 0.01))


def generate_sat_problem_for_group(
    wallet: bt.wallet, window_block_hash: str, base_nonce: int, difficulty: float
) -> tuple[Any, str]:
    """Create a SAT problem and the base seed shared by a GRPO rollout group."""
    sat_seed_base = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{base_nonce}"
    sat_problem = generate_sat_problem(sat_seed_base, difficulty)
    return sat_problem, sat_seed_base


def create_rollout_generator(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device
) -> SATRolloutGenerator:
    """Create a GRPO rollout generator configured for the given model.

    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        device: Torch device (cuda/cpu)

    Returns:
        SATRolloutGenerator instance
    """
    return SATRolloutGenerator(
        model,
        tokenizer,
        device,
        rollouts_per_problem=ROLLOUTS_PER_PROBLEM,
    )


async def maybe_log_debug_sample(
    tokenizer: AutoTokenizer,
    sample: Any,
    window_start: int,
    base_nonce: int,
    monitor: Optional[Any],
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
        if completion_len > 0 and prompt_len >= 0:
            completion_ids = sample.tokens[prompt_len : prompt_len + completion_len]
        else:
            completion_ids = sample.tokens[prompt_len:]
        sample_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        sample_nonce = base_nonce * 10
        logger.debug(
            "TEXT[mine] window=%s group=%s nonce=%s reward=%.3f adv=%.3f success=%s text=%s",
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


def package_rollout_data(
    model: AutoModelForCausalLM,
    wallet: bt.wallet,
    rollout: Any,
    base_nonce: int,
    rollout_idx: int,
    total_in_group: int,
    sat_problem: Any,
    sat_seed_base: str,
    window_start: int,
    current_block: int,
    window_block_hash: str,
    combined_randomness: str,
    difficulty: float,
    use_drand: bool,
) -> dict:
    """Assemble the full on-chain/off-chain payload for a single rollout.

    This binds model outputs (tokens, commitments) to the randomness, model name,
    and layer via a commit-binding signature, and includes proof and SAT
    metadata required by validators.

    Args:
        model: Loaded model (for name_or_path)
        wallet: Miner wallet for signing
        rollout: Generated rollout with tokens/commitments/trajectory
        base_nonce: Base nonce for the group
        rollout_idx: Index within the group
        total_in_group: Total rollouts in group
        sat_problem: SAT problem instance
        sat_seed_base: SAT seed
        window_start: Window start block
        current_block: Current block
        window_block_hash: Window block hash
        combined_randomness: Challenge randomness
        difficulty: SAT difficulty
        use_drand: Whether drand was used

    Returns:
        Signed dictionary ready to upload for validation
    """
    rollout_nonce = base_nonce * 10 + rollout_idx
    rollout_sat_seed = f"{wallet.hotkey.ss58_address}-{window_block_hash}-{rollout_nonce}"

    # Sign commit binding (tokens, randomness, model, layer, commitments)
    from ..protocol.signatures import sign_commit_binding

    commit_sig = sign_commit_binding(
        tokens=rollout.tokens,
        randomness_hex=combined_randomness,
        model_name=model.name_or_path,
        layer_index=LAYER_INDEX,
        commitments=rollout.commitments,
        wallet=wallet,
    )

    assignment = extract_assignment_from_rollout(rollout)
    satisfied_clauses = count_satisfied_clauses(sat_problem, assignment)

    payload = {
        "window_start": window_start,
        "block": current_block,
        "nonce": rollout_nonce,
        "sat_seed": rollout_sat_seed,
        "difficulty": difficulty,
        "block_hash": window_block_hash,
        "randomness": combined_randomness,
        "use_drand": use_drand,
        "rollout_group": base_nonce,
        "rollout_index": rollout_idx,
        "total_in_group": total_in_group,
        "commit": {
            "tokens": rollout.tokens,
            "commitments": rollout.commitments,
            "proof_version": rollout.proof_version,
            "model": {
                "name": model.name_or_path,
                "layer_index": LAYER_INDEX,
            },
            "signature": commit_sig.hex(),
            "beacon": rollout.beacon,
            "sat_problem": {
                "seed": sat_seed_base,
                "num_vars": sat_problem.num_vars,
                "clauses": sat_problem.clauses,
                "difficulty": difficulty,
            },
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
    monitor: Optional[Any],
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
        with monitor.timer("mining.upload_duration"):
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


async def generate_rollouts_for_window(
    wallet: bt.wallet,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subtensor: bt.subtensor,
    window_start: int,
    window_block_hash: str,
    combined_randomness: str,
    timers: MiningTimers,
    monitor: Optional[Any],
    use_drand: bool,
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

    device = model.device
    generator = create_rollout_generator(model, tokenizer, device)

    while True:
        current_block = await subtensor.get_current_block()
        timers.update_block_time_ema(current_block)
        global HEARTBEAT
        HEARTBEAT = time.monotonic()
        current_window = calculate_window_start(current_block)
        if current_window > window_start:
            logger.info("Window %s has ended, moving to next window", window_start)
            break

        blocks_remaining = (window_start + WINDOW_LENGTH) - current_block
        needed_blocks = timers.blocks_needed_for_next_gen()
        if blocks_remaining <= needed_blocks:
            logger.info(
                "Stopping generation: %s blocks remain, need %s "
                "(gen≈%.1fs, upload≈%.1fs, block≈%.2fs)",
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

            # Deterministically derive seed and difficulty from miner+window+index
            problem_index = max(0, problem_count - 1)
            seed, difficulty = derive_canonical_sat(
                wallet.hotkey.ss58_address, window_block_hash, problem_index
            )
            sat_problem = generate_sat_problem(seed, difficulty)
            # Use deterministic problem index as rollout_group identifier
            base_nonce = problem_index
            sat_seed_base = seed
            logger.debug(
                "Generated SAT problem: %s vars, %s clauses",
                sat_problem.num_vars,
                len(sat_problem.clauses),
            )

            HEARTBEAT = time.monotonic()
            grpo_rollouts = generator.generate_grpo_rollouts(
                sat_problem, combined_randomness, wallet
            )
            HEARTBEAT = time.monotonic()

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
                "GRPO batch: %s/%s successful, mean reward: %.3f",
                successful_count,
                len(grpo_rollouts),
                mean_reward,
            )

            if problem_count % 2 == 0:
                elapsed = time.time() - start_time
                rollouts_per_sec = (len(inferences) / elapsed) if elapsed > 0 else 0
                logger.info(
                    "📊 Progress: %s rollouts from %s problems in %.1fs (%.1f rollouts/sec)",
                    len(inferences),
                    problem_count,
                    elapsed,
                    rollouts_per_sec,
                )
                if monitor:
                    await monitor.log_gauge("mining.rollouts_generated", len(inferences))
                    await monitor.log_gauge("mining.problems_processed", problem_count)
                    await monitor.log_gauge("mining.rollouts_per_second", rollouts_per_sec)
                    if successful_rollouts + failed_rollouts > 0:
                        success_rate = successful_rollouts / (successful_rollouts + failed_rollouts)
                        await monitor.log_gauge("mining.success_rate", success_rate)

            # Package each rollout with signatures and proofs for validation
            for rollout_idx, rollout in enumerate(grpo_rollouts):
                rollout_data = package_rollout_data(
                    model,
                    wallet,
                    rollout,
                    base_nonce,
                    rollout_idx,
                    len(grpo_rollouts),
                    sat_problem,
                    sat_seed_base,
                    window_start,
                    current_block,
                    window_block_hash,
                    combined_randomness,
                    difficulty,
                    use_drand,
                )
                inferences.append(rollout_data)

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

            timers.update_gen_time_ema(time.time() - gen_start)
            await asyncio.sleep(0.01)
            HEARTBEAT = time.monotonic()

        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error("CUDA error at inference %s: %s", inference_count, e)
                logger.error(
                    "SAT problem: vars=%s, clauses=%s",
                    sat_problem.num_vars if "sat_problem" in locals() else "N/A",
                    len(sat_problem.clauses) if "sat_problem" in locals() else "N/A",
                )
                logger.error(
                    "Difficulty: %s",
                    difficulty if "difficulty" in locals() else "N/A",
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
        except Exception as e:
            logger.warning("Failed to generate inference %s: %s", inference_count, e)
            continue

    elapsed_time = time.time() - start_time
    logger.info(
        "🎯 Generated %s rollouts in %.1fs for window %s",
        len(inferences),
        elapsed_time,
        window_start,
    )
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

    return inferences


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
