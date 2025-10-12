"""Shared types for validation service.

Defines data structures for validation results, metrics, and context.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class MinerResults:
    """Results from validating a single miner's window submission.

    Attributes:
        hotkey: Miner's hotkey address
        uid: Miner's UID (None if not found in metagraph)
        found_file: Whether the miner's window file was found
        metrics: Per-miner metrics (valid, checked, total, estimated_valid, etc.)
        rollouts: Valid rollouts that passed all checks
        processed_counts: Tuple of (total, invalid_sig, invalid_proof, processing_err)
        digest_counter: Counter of rollout completion digests (for copycat detection)
        total_inferences_in_file: Total number of rollouts in the window file
    """

    hotkey: str
    uid: int | None
    found_file: bool
    metrics: dict[str, int] | None
    rollouts: list[dict]
    processed_counts: tuple[int, int, int, int]
    digest_counter: Counter[str] | None
    total_inferences_in_file: int


@dataclass
class WindowResults:
    """Aggregated results from processing a validation window.

    Attributes:
        window_start: Window start block number
        window_block_hash: Block hash at window start
        window_randomness: Combined randomness for the window
        miner_results: Per-miner MinerResults (optional, can be empty)
        total_valid_rollouts: Sum of estimated_valid across all miners
        total_rollouts_processed: Total rollouts processed
        invalid_signatures: Count of signature validation failures
        invalid_proofs: Count of proof validation failures
        processing_errors: Count of processing exceptions
        files_found: Number of miner window files found
        all_valid_rollouts_for_upload: All valid rollouts (post-gating)
        window_cheaters: Miners gated at window scope
        interval_cheaters: Miners gated at interval scope
        violation_details: All copycat violations detected
        window_metrics: Per-miner metrics dict {hotkey: {metric_name: value}}
        window_timing_seconds: Total time for window processing
        window_timing_blocks: Total blocks elapsed during processing
        miner_timing_seconds: List of per-miner processing times
        miner_timing_blocks: List of per-miner block counts
    """

    window_start: int
    window_block_hash: str
    window_randomness: str
    miner_results: dict[str, Any]  # Can be populated or empty
    total_valid_rollouts: int
    total_rollouts_processed: int
    invalid_signatures: int
    invalid_proofs: int
    processing_errors: int
    files_found: int
    all_valid_rollouts_for_upload: list[dict]
    window_cheaters: set[str]
    interval_cheaters: set[str]
    violation_details: list[Any]  # CopycatViolation
    window_metrics: dict[str, dict[str, int]]
    window_timing_seconds: float
    window_timing_blocks: int
    miner_timing_seconds: list[float]
    miner_timing_blocks: list[int]


@dataclass
class ValidationContext:
    """Shared context for validation operations.

    Contains all dependencies and state needed for validation operations.
    Avoids passing many individual parameters through call chains.

    Attributes:
        wallet: Validator wallet for signing
        model: Language model for validation
        tokenizer: Tokenizer for decoding
        sat_pipeline: SAT validation pipeline
        credentials: Object storage credentials
        chain_manager: Chain manager for miner bucket credentials
        monitor: Optional monitoring client
        subtensor: Async subtensor instance
        uid_by_hotkey: Mapping of hotkey to UID for logging
        sat_reward_low: Lower bound for SAT reward sanity checks
        sat_reward_high: Upper bound for SAT reward sanity checks
    """

    wallet: Any  # bt.wallet
    model: Any  # AutoModelForCausalLM
    tokenizer: Any  # AutoTokenizer
    sat_pipeline: Any  # ValidationPipeline
    credentials: Any
    chain_manager: Any  # GrailChainManager
    monitor: Any | None
    subtensor: Any  # bt.subtensor
    uid_by_hotkey: dict[str, int]
    sat_reward_low: float
    sat_reward_high: float
