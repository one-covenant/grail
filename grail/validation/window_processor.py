"""Window processing orchestrator for GRAIL validation.

Coordinates miner validation, copycat detection, and result aggregation for a single window.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.chain import GrailChainManager
from ..logging_utils import miner_log_context
from .copycat_service import CopycatService
from .miner_validator import MinerValidator
from .types import MinerResults, WindowResults

logger = logging.getLogger(__name__)


class WindowProcessor:
    """Orchestrates validation for a single window.

    Responsibilities:
    1. Validate each selected miner's submissions
    2. Aggregate metrics across miners
    3. Detect copycat cheaters
    4. Apply gating to cheaters
    5. Return structured results

    Design:
    - Delegates miner validation to MinerValidator
    - Delegates copycat detection to CopycatService
    - Coordinates timing and aggregation
    - Returns structured WindowResults
    """

    def __init__(
        self,
        miner_validator: MinerValidator,
        copycat_service: CopycatService,
    ):
        """Initialize window processor.

        Args:
            miner_validator: Validator for individual miner submissions
            copycat_service: Service for copycat detection and gating
        """
        self._miner_validator = miner_validator
        self._copycat_service = copycat_service

    async def process_window(
        self,
        window: int,
        window_hash: str,
        window_rand: str,
        miners_to_check: list[str],
        validator_wallet: Any,  # bt.wallet
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        credentials: Any,
        chain_manager: GrailChainManager,
        monitor: Any | None,
        uid_by_hotkey: dict[str, int],
        subtensor: Any,  # bt.subtensor
        heartbeat_callback: Any = None,
    ) -> WindowResults:
        """Process a complete validation window.

        Args:
            window: Window start block number
            window_hash: Block hash at window start
            window_rand: Combined randomness for the window
            miners_to_check: List of miner hotkeys to validate
            validator_wallet: Validator's wallet
            model: Language model for validation
            tokenizer: Tokenizer for decoding
            credentials: Default R2 credentials
            chain_manager: Chain manager for miner buckets
            monitor: Optional monitoring client
            uid_by_hotkey: Mapping of hotkey to UID
            subtensor: Subtensor instance for block queries
            heartbeat_callback: Optional watchdog heartbeat callback

        Returns:
            WindowResults with all aggregated metrics and rollouts
        """
        # Window timing
        window_t0 = time.monotonic()
        try:
            block_beg = await subtensor.get_current_block()
        except Exception:
            block_beg = None

        # Initialize aggregation structures
        window_metrics: dict[str, dict[str, int]] = {}
        miner_rollout_counters: dict[str, tuple[Counter[str], int]] = {}
        all_valid_rollouts: list[dict] = []
        text_logs_emitted: dict[str, int] = {}

        # Aggregated counters
        total_rollouts_processed = 0
        invalid_signatures = 0
        invalid_proofs = 0
        processing_errors = 0
        files_found = 0

        # Per-miner timing lists
        miner_seconds_list: list[float] = []
        miner_blocks_list: list[int] = []

        # Process each miner
        for miner_hotkey in miners_to_check:
            # Update heartbeat
            if heartbeat_callback:
                try:
                    heartbeat_callback()
                except Exception:
                    pass

            # Per-miner timing
            t0 = time.monotonic()
            try:
                b0 = await subtensor.get_current_block()
            except Exception:
                b0 = None

            try:
                # Validate miner (use .get() to avoid KeyError if hotkey not in map)
                uid = uid_by_hotkey.get(miner_hotkey, miner_hotkey)
                with miner_log_context(uid, window):
                    result: MinerResults = await self._miner_validator.validate_miner(
                        miner_hotkey=miner_hotkey,
                        window=window,
                        window_hash=window_hash,
                        window_rand=window_rand,
                        validator_wallet=validator_wallet,
                        model=model,
                        tokenizer=tokenizer,
                        credentials=credentials,
                        chain_manager=chain_manager,
                        monitor=monitor,
                        uid_by_hotkey=uid_by_hotkey,
                        text_logs_emitted=text_logs_emitted,
                        heartbeat_callback=heartbeat_callback,
                    )

                # Record timing
                t1 = time.monotonic()
                try:
                    b1 = await subtensor.get_current_block()
                except Exception:
                    b1 = None

                sec = t1 - t0
                blk = (b1 - b0) if (b0 is not None and b1 is not None) else 0
                miner_seconds_list.append(float(sec))
                miner_blocks_list.append(int(blk))

                # Aggregate results
                if result.found_file:
                    files_found += 1

                if result.metrics is not None:
                    window_metrics[miner_hotkey] = result.metrics

                if result.rollouts:
                    all_valid_rollouts.extend(result.rollouts)

                if result.digest_counter is not None:
                    miner_rollout_counters[miner_hotkey] = (
                        result.digest_counter,
                        result.total_inferences_in_file,
                    )

                # Aggregate processing counts
                (pr_total, pr_invalid_sig, pr_invalid_proof, pr_processing_err) = (
                    result.processed_counts
                )
                total_rollouts_processed += pr_total
                invalid_signatures += pr_invalid_sig
                invalid_proofs += pr_invalid_proof
                processing_errors += pr_processing_err

            except Exception as e:
                uid_str = str(uid_by_hotkey.get(miner_hotkey, miner_hotkey))
                logger.warning(f"[window={window} uid={uid_str}] Error processing miner: {e}")
                continue

        # Compute total valid rollouts
        total_valid_rollouts = sum(
            metrics.get("estimated_valid", 0) for metrics in window_metrics.values()
        )

        # Copycat detection and gating
        (
            window_cheaters,
            interval_cheaters,
            violations,
        ) = await self._copycat_service.detect_cheaters(
            window=window,
            miner_rollout_counters=miner_rollout_counters,
            uid_by_hotkey=uid_by_hotkey,
            monitor=monitor,
        )

        # Combine cheater sets
        cheaters_detected = window_cheaters.union(interval_cheaters)

        # Apply gating to cheaters
        self._copycat_service.apply_gating(
            cheaters=cheaters_detected,
            violations=violations,
            window_metrics=window_metrics,
            uid_by_hotkey=uid_by_hotkey,
            window=window,
        )

        # Filter cheater rollouts
        all_valid_rollouts = self._copycat_service.filter_cheater_rollouts(
            rollouts=all_valid_rollouts,
            cheaters=cheaters_detected,
        )

        # Recompute total_valid_rollouts after gating
        total_valid_rollouts = sum(
            metrics.get("estimated_valid", 0) for metrics in window_metrics.values()
        )

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

        # Log window summary
        logger.info(
            f"Window {window} complete: {total_valid_rollouts} valid rollouts, "
            f"{files_found}/{len(miners_to_check)} files found, "
            f"{len(cheaters_detected)} cheaters gated, "
            f"{window_seconds:.1f}s ({window_blocks} blocks)"
        )

        # Log timing stats if available
        if miner_seconds_list:
            avg_sec = sum(miner_seconds_list) / len(miner_seconds_list)
            logger.info(f"Avg miner processing: {avg_sec:.2f}s")

        # Return structured results
        return WindowResults(
            window_start=window,
            window_block_hash=window_hash,
            window_randomness=window_rand,
            miner_results={},  # Not needed in current implementation
            total_valid_rollouts=total_valid_rollouts,
            total_rollouts_processed=total_rollouts_processed,
            invalid_signatures=invalid_signatures,
            invalid_proofs=invalid_proofs,
            processing_errors=processing_errors,
            files_found=files_found,
            all_valid_rollouts_for_upload=all_valid_rollouts,
            window_cheaters=window_cheaters,
            interval_cheaters=interval_cheaters,
            violation_details=violations,
            window_metrics=window_metrics,
            window_timing_seconds=window_seconds,
            window_timing_blocks=window_blocks,
            miner_timing_seconds=miner_seconds_list,
            miner_timing_blocks=miner_blocks_list,
        )
