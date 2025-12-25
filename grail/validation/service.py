"""Core validation service for GRAIL protocol.

Orchestrates the validation loop, window processing, and weight submission.
Separated from CLI concerns for better testability and maintainability.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import bittensor as bt
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.chain import GrailChainManager
from ..infrastructure.checkpoint_consumer import CheckpointManager
from ..infrastructure.credentials import BucketCredentials
from ..logging_utils import dump_asyncio_stacks
from ..model.provider import (
    clear_model_and_tokenizer,
    get_model,
    get_tokenizer,
)
from ..scoring.weights import WeightComputer
from ..shared.constants import (
    FAILURE_LOOKBACK_WINDOWS,
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    MINER_SAMPLING_ENABLED,
    TRAINER_UID,
    WINDOW_LENGTH,
)
from ..shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)

# Imports retained only where used
from .copycat_service import COPYCAT_SERVICE
from .miner_validator import MinerValidator
from .pipeline import ValidationPipeline
from .sampling import MinerSampler
from .window_processor import WindowProcessor

logger = logging.getLogger(__name__)

# Weight submission constants
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 360
WEIGHT_ROLLING_WINDOWS = int(WEIGHT_SUBMISSION_INTERVAL_BLOCKS / WINDOW_LENGTH)
DEADLINE_SLACK_SECONDS = -4.0


class ValidationService:
    """Core validation orchestration service.

    Handles the main validation loop, window processing, and weight submission.
    Maintains no global state - all state is instance-specific for testability.

    This service coordinates:
    - Checkpoint management and model loading
    - Chain manager initialization for miner credentials
    - Window discovery and processing
    - Weight computation and submission
    - Monitoring and metrics

    Design:
    - Single async subtensor passed from ValidatorNeuron (via BaseNeuron)
    - Chain manager with worker process for commitment fetching
    - Clear async boundaries with timeouts
    - Dependency injection for all external resources
    """

    def __init__(
        self,
        wallet: bt.wallet,
        netuid: int,
        validation_pipeline: ValidationPipeline,
        weight_computer: WeightComputer,
        credentials: BucketCredentials,
        checkpoint_manager: CheckpointManager,
        monitor: Any | None = None,
    ):
        """Initialize validation service.

        Args:
            wallet: Validator wallet for signing transactions
            netuid: Network UID for the subnet
            validation_pipeline: Validation pipeline for rollout verification
            weight_computer: Weight computation engine
            credentials: Object storage credentials for rollout access
            checkpoint_manager: Checkpoint manager for model downloads
            monitor: Optional monitoring client for metrics
        """
        self._wallet = wallet
        self._netuid = netuid
        self._validation_pipeline = validation_pipeline
        self._weight_computer = weight_computer
        self._credentials = credentials
        self._checkpoint_manager = checkpoint_manager
        self._monitor = monitor

        # Initialized during setup
        self._chain_manager: GrailChainManager | None = None
        self._subtensor: bt.subtensor | None = None
        self._metagraph: Any = None
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

        # Service components (lazy-init)
        self._miner_sampler: MinerSampler | None = None
        self._miner_validator: MinerValidator | None = None
        self._window_processor: WindowProcessor | None = None

        # Validation state
        self._last_processed_window: int = -1
        self._last_weights_interval_submitted: int = -1
        self._last_copycat_interval_id: int = -1
        self._windows_processed_since_start: int = 0
        self._current_checkpoint_id: str | None = None
        self._current_checkpoint_window: int | None = None  # Track window for fast path

        # Rolling histories for miner selection and availability
        self._selection_history: deque[set[str]] = deque(maxlen=WEIGHT_ROLLING_WINDOWS)
        self._availability_history: deque[set[str]] = deque(maxlen=WEIGHT_ROLLING_WINDOWS)
        self._selection_counts: dict[str, int] = {}
        self._availability_counts: dict[str, int] = {}

        # Failure tracking for exclusion from sampling
        self._failure_history: deque[set[str]] = deque(maxlen=FAILURE_LOOKBACK_WINDOWS)
        self._failure_counts: dict[str, int] = {}

        # Rolling window metrics per hotkey: window_start -> metric dict
        self._inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Window wait tracking for clean logging
        self._window_wait_tracker = WindowWaitTracker(log_interval_secs=120)

        logger.info(f"Initialized ValidationService for netuid {netuid}")

    async def run_validation_loop(
        self,
        subtensor: bt.subtensor,
        use_drand: bool,
        test_mode: bool,
        heartbeat_callback: Any | None = None,
    ) -> None:
        """Run the main validation loop.

        This is the entry point for the validation service. It:
        1. Initializes chain manager and loads initial checkpoint
        2. Enters the main loop processing windows
        3. Computes and submits weights at configured intervals
        4. Handles errors and reconnections gracefully

        Args:
            subtensor: Async subtensor instance from ValidatorNeuron
            use_drand: Whether to use drand for challenge randomness
            test_mode: If True, validate only own wallet (for testing)
            heartbeat_callback: Optional callback to update heartbeat timestamp

        Note:
            This function runs indefinitely until interrupted.
            The caller should run it with a watchdog for liveness monitoring.
        """
        self._subtensor = subtensor

        # Initialize chain manager and metagraph
        await self._initialize_chain_manager()

        # Initialize service components
        self._initialize_components()

        logger.info(f"Starting validation loop (use_drand={use_drand}, test_mode={test_mode})")

        # Main validation loop
        while True:
            try:
                # Update heartbeat if callback provided
                if heartbeat_callback:
                    heartbeat_callback()

                # Get current state
                meta = await self._subtensor.metagraph(self._netuid)
                uid_by_hotkey = dict(zip(meta.hotkeys, meta.uids, strict=True))

                # Update chain manager's metagraph to keep hotkey->UID lookups fresh
                # This is critical because miners register/deregister and UIDs shift
                if self._chain_manager:
                    self._chain_manager.metagraph = meta

                current_block = await self._subtensor.get_current_block()
                # Validate the last fully completed window, not the in-progress one
                target_window = self._compute_target_validation_window(current_block)

                # Skip if already processed
                if target_window <= self._last_processed_window or target_window < 0:
                    if self._window_wait_tracker.should_log_initial():
                        log_window_wait_initial(
                            current_block=current_block,
                            last_processed_window=self._last_processed_window,
                            window_length=WINDOW_LENGTH,
                        )
                    elif self._window_wait_tracker.should_log_periodic():
                        next_window = calculate_next_window(
                            self._last_processed_window, WINDOW_LENGTH
                        )
                        log_window_wait_periodic(
                            next_window=next_window,
                            elapsed_seconds=self._window_wait_tracker.get_elapsed_seconds(),
                        )

                    await asyncio.sleep(5)
                    continue

                # Window is available - reset wait tracker for next time
                self._window_wait_tracker.reset()

                # Set monitoring context (use block_number for x-axis)
                if self._monitor:
                    self._monitor.set_block_context(current_block, None)

                # Load checkpoint for this window
                checkpoint_loaded = await self._load_checkpoint_for_window(target_window)
                if not checkpoint_loaded:
                    # No checkpoint available - skip this window, wait for next
                    logger.warning(
                        "No checkpoint for window %s, waiting for next window", target_window
                    )
                    self._last_processed_window = target_window
                    continue

                # Cleanup old checkpoints
                try:
                    await self._checkpoint_manager.cleanup_local(target_window)
                except Exception:
                    logger.debug("Checkpoint cache cleanup failed", exc_info=True)

                # Reset copycat tracker at interval boundaries
                copycat_interval_id = int(target_window // WEIGHT_SUBMISSION_INTERVAL_BLOCKS)
                if copycat_interval_id != self._last_copycat_interval_id:
                    COPYCAT_SERVICE.reset_interval(copycat_interval_id)
                    self._last_copycat_interval_id = copycat_interval_id

                # Process window
                await self._process_window(
                    target_window=target_window,
                    meta=meta,
                    uid_by_hotkey=uid_by_hotkey,
                    use_drand=use_drand,
                    test_mode=test_mode,
                    heartbeat_callback=heartbeat_callback,
                )

                # Update weight submission state
                self._windows_processed_since_start += 1

                # Submit weights if interval reached
                await self._submit_weights_if_ready(current_block, target_window, meta)

                # Update state
                self._last_processed_window = target_window

            except asyncio.CancelledError:
                # Log full traceback and attempt an asyncio task snapshot for diagnostics
                logger.warning(
                    "Validation loop cancelled",
                    exc_info=True,
                )
                try:
                    await dump_asyncio_stacks(label="CANCEL")
                except Exception:
                    logger.debug(
                        "Failed to dump asyncio stacks during cancellation",
                        exc_info=True,
                    )
                break
            except Exception as e:
                logger.error(f"Error in validation loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _initialize_chain_manager(self) -> None:
        """Initialize chain manager with subtensor and metagraph."""
        if self._subtensor is None:
            raise RuntimeError("Subtensor must be set before initializing chain manager")

        # Get metagraph for the subnet
        self._metagraph = await self._subtensor.metagraph(self._netuid)
        logger.info(f"Loaded metagraph with {len(self._metagraph.hotkeys)} neurons")

        # Initialize chain manager with injected dependencies
        config = SimpleNamespace(netuid=self._netuid)
        self._chain_manager = GrailChainManager(
            config,
            self._wallet,
            self._metagraph,
            self._subtensor,
            self._credentials,
        )

        # Initialize and commit credentials
        await self._chain_manager.initialize()
        logger.info("Initialized chain manager and committed read credentials")

    def _initialize_components(self) -> None:
        """Initialize service components."""
        # Miner sampler
        self._miner_sampler = MinerSampler(
            sample_rate=MINER_SAMPLE_RATE,
            sample_min=MINER_SAMPLE_MIN,
            sample_max=MINER_SAMPLE_MAX,
            concurrency=8,
        )

        # Miner validator
        self._miner_validator = MinerValidator(
            pipeline=self._validation_pipeline,
            text_log_limit=5,
        )

        # Window processor
        self._window_processor = WindowProcessor(
            miner_validator=self._miner_validator,
            copycat_service=COPYCAT_SERVICE,
        )

        logger.info("Initialized service components")

    async def _load_checkpoint_for_window(self, target_window: int) -> bool:
        """Load model/tokenizer checkpoint for validation window.

        Uses unified load_or_update_model for fast path (in-place delta) and
        slow path (full disk load) with automatic fallback.

        Args:
            target_window: Target window to validate

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        # Get trainer's bucket for checkpoints (one-time setup)
        if self._chain_manager and self._current_checkpoint_id is None:
            trainer_bucket = self._chain_manager.get_bucket(TRAINER_UID)
            if trainer_bucket:
                logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")
                self._checkpoint_manager.credentials = trainer_bucket
            else:
                logger.warning(
                    f"⚠️ Trainer UID {TRAINER_UID} bucket not found, using local credentials"
                )

        # Unified checkpoint loading (fast path + slow path with fallback)
        timer_ctx = (
            self._monitor.timer("profiling/checkpoint_load")
            if self._monitor
            else contextlib.nullcontext()
        )
        with timer_ctx:
            result, checkpoint_path = await self._checkpoint_manager.load_or_update_model(
                target_window, self._model, self._current_checkpoint_window
            )

        if not result.success:
            logger.warning(
                f"No checkpoint available for window {target_window}, skipping validation"
            )
            return False

        # Fast path: model already updated in-place
        if result.is_fast_path:
            self._current_checkpoint_window = result.window
            self._current_checkpoint_id = f"inplace-{result.window}"
            logger.info(f"⚡ Validator model updated in-place to window {result.window}")
            return True

        # Slow path: load from disk
        if checkpoint_path is not None:
            try:
                logger.info(
                    f"🚀 Loading checkpoint for validation window {target_window} "
                    f"from {checkpoint_path}"
                )
                self._model, self._tokenizer = clear_model_and_tokenizer(
                    self._model, self._tokenizer
                )
                self._model = get_model(str(checkpoint_path), device=None, eval_mode=True)
                self._tokenizer = get_tokenizer(str(checkpoint_path))
                self._current_checkpoint_id = str(checkpoint_path)
                self._current_checkpoint_window = result.window

                self._log_tokenizer_info(checkpoint_path)
                return True
            except Exception:
                logger.exception(f"Failed to load checkpoint for window {target_window}")
                return False

        return True

    def _log_tokenizer_info(self, checkpoint_path: Any) -> None:
        """Log tokenizer version information for debugging."""
        try:
            import tokenizers  # type: ignore
            import transformers

            logger.info(
                "VALIDATOR TOKENIZER INFO: transformers=%s, tokenizers=%s, name_or_path=%s, checkpoint=%s",
                transformers.__version__,
                tokenizers.__version__,
                getattr(self._tokenizer, "name_or_path", "unknown"),
                str(checkpoint_path),
            )
        except Exception as e:
            logger.debug("Failed to log tokenizer version info: %s", e)

    async def _process_window(
        self,
        target_window: int,
        meta: Any,
        uid_by_hotkey: dict[str, int],
        use_drand: bool,
        test_mode: bool,
        heartbeat_callback: Any | None,
    ) -> None:
        """Process a single validation window.

        Args:
            target_window: Window start block
            meta: Metagraph instance
            uid_by_hotkey: Mapping of hotkey to UID
            use_drand: Use drand for randomness
            test_mode: Test mode flag
            heartbeat_callback: Optional heartbeat callback
        """
        # Get window block hash and randomness
        if self._subtensor is None:
            raise RuntimeError("Subtensor not initialized")
        target_window_hash = await self._subtensor.get_block_hash(target_window)
        window_rand = await self._compute_window_randomness(target_window_hash, use_drand)

        # Compute upload deadline timestamp (start of the validator window)
        deadline_ts: float | None = None
        try:
            if self._chain_manager is not None:
                deadline_block = target_window + WINDOW_LENGTH
                deadline_ts = await self._chain_manager.get_block_timestamp(deadline_block)
                if deadline_ts is None:
                    deadline_ts = await self._chain_manager.estimate_block_timestamp(deadline_block)
                if deadline_ts is not None:
                    deadline_ts += DEADLINE_SLACK_SECONDS
            readable_ts: str = (
                datetime.fromtimestamp(deadline_ts, tz=timezone.utc).isoformat()
                if deadline_ts is not None
                else "None"
            )
            logger.info(f"Deadline timestamp: {readable_ts} and deadline block: {deadline_block}")
        except Exception:
            logger.debug("Failed to compute deadline timestamp", exc_info=True)

        # Discover active miners
        timer_ctx = (
            self._monitor.timer("profiling/hotkeys_discovery")
            if self._monitor
            else contextlib.nullcontext()
        )
        with timer_ctx:
            if self._miner_sampler is None:
                raise RuntimeError("MinerSampler not initialized")
            active_hotkeys = await self._miner_sampler.discover_active_miners(
                meta_hotkeys=list(meta.hotkeys),
                window=target_window,
                chain_manager=self._chain_manager,
                uid_by_hotkey=uid_by_hotkey,
                heartbeat_callback=heartbeat_callback,
                deadline_ts=deadline_ts,
            )

        logger.info(
            f"🔍 Found {len(active_hotkeys)}/{len(meta.hotkeys)} active miners "
            f"for window {target_window}"
        )

        # Update availability history
        self._update_rolling(
            self._availability_history,
            self._availability_counts,
            set(active_hotkeys),
        )

        # Exclude miners with recent failures
        eligible_hotkeys = self._filter_hotkeys_without_failures(active_hotkeys)

        # Determine subset to validate
        if test_mode:
            hotkeys_to_check = [self._wallet.hotkey.ss58_address]
        elif MINER_SAMPLING_ENABLED:
            if self._miner_sampler is None:
                raise RuntimeError("MinerSampler not initialized")
            hotkeys_to_check = self._miner_sampler.select_miners_for_validation(
                active_hotkeys=eligible_hotkeys,
                window_hash=target_window_hash,
                selection_counts=self._selection_counts,
            )
        else:
            hotkeys_to_check = eligible_hotkeys

        # Update selection history
        self._update_rolling(
            self._selection_history,
            self._selection_counts,
            set(hotkeys_to_check),
        )

        # Log sampling metrics
        if self._monitor:
            await self._log_sampling_metrics(
                total=len(meta.hotkeys),
                active=len(active_hotkeys),
                eligible=len(eligible_hotkeys),
                selected=len(hotkeys_to_check),
            )

        # Process window (validate all miners) - time the entire operation
        if self._window_processor is None:
            raise RuntimeError("WindowProcessor not initialized")
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")
        if self._chain_manager is None:
            raise RuntimeError("Chain manager not initialized")

        timer_ctx = (
            self._monitor.timer("profiling/window_processing")
            if self._monitor
            else contextlib.nullcontext()
        )
        with timer_ctx:
            window_results = await self._window_processor.process_window(
                window=target_window,
                window_hash=target_window_hash,
                window_rand=window_rand,
                miners_to_check=hotkeys_to_check,
                validator_wallet=self._wallet,
                model=self._model,
                tokenizer=self._tokenizer,
                credentials=self._credentials,
                chain_manager=self._chain_manager,
                monitor=self._monitor,
                uid_by_hotkey=uid_by_hotkey,
                subtensor=self._subtensor,
                heartbeat_callback=heartbeat_callback,
                deadline_ts=deadline_ts,
            )

        # Update inference counts for weight computation
        for hotkey, metrics in window_results.window_metrics.items():
            self._inference_counts[hotkey][target_window] = metrics

        # Update failure history
        failed_hotkeys = {
            hotkey
            for hotkey, metrics in window_results.window_metrics.items()
            if metrics.get("had_failure", 0) > 0
        }
        self._update_rolling(
            self._failure_history,
            self._failure_counts,
            failed_hotkeys,
        )

        logger.info(
            f"Window {target_window} complete: "
            f"{window_results.total_valid_rollouts} valid rollouts, "
            f"{window_results.files_found}/{len(hotkeys_to_check)} files found"
        )

        # Log aggregated window metrics to monitoring (W&B)
        if self._monitor:
            try:
                await self._monitor.log_gauge(
                    "validation/active_miners/window_valid_rollouts",
                    window_results.total_valid_rollouts,
                )
                await self._monitor.log_gauge(
                    "validation/active_miners/window_files_found",
                    window_results.files_found,
                )
                await self._monitor.log_gauge(
                    "validation/active_miners/window_invalid_signatures",
                    window_results.invalid_signatures,
                )
                await self._monitor.log_gauge(
                    "validation/active_miners/window_invalid_proofs",
                    window_results.invalid_proofs,
                )
                await self._monitor.log_gauge(
                    "validation/active_miners/window_processing_errors",
                    window_results.processing_errors,
                )

                await self._monitor.log_gauge(
                    "profiling/window_blocks",
                    int(window_results.window_timing_blocks),
                )
            except Exception:
                logger.debug("Failed to log aggregated window metrics", exc_info=True)

    async def _submit_weights_if_ready(
        self, current_block: int, target_window: int, meta: Any
    ) -> None:
        """Submit weights if interval has been reached.

        Args:
            current_block: Current block number
            meta: Metagraph instance
            target_window: Target window to submit weights for
        """
        current_interval = int(current_block // WEIGHT_SUBMISSION_INTERVAL_BLOCKS)

        # Check if we should submit
        if current_interval <= self._last_weights_interval_submitted:
            return

        # Check if we have enough history
        if self._windows_processed_since_start < WEIGHT_ROLLING_WINDOWS:
            logger.info(
                f"Not enough windows for weight submission "
                f"({self._windows_processed_since_start}/{WEIGHT_ROLLING_WINDOWS})"
            )
            self._last_weights_interval_submitted = current_interval
            return

        # Aggregate metrics over rolling window
        logger.info(f"Computing weights over rolling {WEIGHT_ROLLING_WINDOWS}-window history")

        # Collect metrics from last N windows
        aggregated: dict[str, dict[str, int]] = {}
        for hotkey, window_metrics_dict in self._inference_counts.items():
            # Sum metrics across all windows for this hotkey
            aggregated[hotkey] = {}
            for _window_start, metrics in window_metrics_dict.items():
                for key, value in metrics.items():
                    aggregated[hotkey][key] = aggregated[hotkey].get(key, 0) + value

        # Compute weights (with timing)
        timer_ctx = (
            self._monitor.timer("profiling/weights_computation")
            if self._monitor
            else contextlib.nullcontext()
        )
        with timer_ctx:
            weights, non_zero_weights = self._weight_computer.compute_weights(
                meta_hotkeys=list(meta.hotkeys),
                meta_uids=list(meta.uids),
                inference_counts=self._inference_counts,
                target_window=target_window,
                availability_counts=self._availability_counts,
            )

        # Submit to chain (with timing)
        timer_ctx = (
            self._monitor.timer("profiling/weights_submission")
            if self._monitor
            else contextlib.nullcontext()
        )
        with timer_ctx:
            try:
                if self._subtensor is None:
                    raise RuntimeError("Subtensor not initialized")
                await self._subtensor.set_weights(
                    wallet=self._wallet,
                    netuid=self._netuid,
                    uids=list(meta.uids),
                    weights=weights,
                    wait_for_inclusion=False,
                )
                logger.info(f"✅ Submitted weights for interval {current_interval}")
                self._last_weights_interval_submitted = current_interval
            except Exception as e:
                logger.error(f"Failed to submit weights: {e}")

        # Log top miners by weight to monitoring (W&B)
        if self._monitor and non_zero_weights:
            # Build UID mapping and select top-K by weight
            uid_by_hotkey = dict(zip(meta.hotkeys, meta.uids, strict=True))
            top_k = 5
            top_miners = sorted(non_zero_weights, key=lambda x: float(x[1]), reverse=True)[:top_k]

            # Prepare human-readable lines for top miners
            lines: list[str] = []
            for rank, (hk, w) in enumerate(top_miners, start=1):
                uid = uid_by_hotkey.get(hk)
                ident = str(uid) if uid is not None else hk[:12]
                lines.append(f"{rank}. UID {ident}: weight={float(w):.6f}")

            await self._monitor.log_gauge(
                "weights/submission/successful_miners_count",
                len(non_zero_weights),
            )
            await self._monitor.log_artifact(
                "weights/submission/top_miners",
                {
                    "interval": current_interval,
                    "block": current_block,
                    "text": "\n".join(lines),
                },
                "text",
            )

            # Log active miners artifact
            active_miners_lines: list[str] = []
            for hk, _weight in non_zero_weights:
                uid = uid_by_hotkey.get(hk)
                ident = str(uid) if uid is not None else hk[:12]
                active_miners_lines.append(f"UID {ident}")

            await self._monitor.log_artifact(
                "weights/submission/active_miners",
                {
                    "interval": current_interval,
                    "block": current_block,
                    "count": len(non_zero_weights),
                    "text": "\n".join(active_miners_lines),
                },
                "text",
            )

            # Compute rollout statistics for top miners
            top_miner_hotkeys = [hk for hk, _w in top_miners]
            rollout_stats = {
                "total_rollouts": [],
                "unique_rollouts": [],
                "successful_rollouts": [],
            }

            for hk in top_miner_hotkeys:
                # Aggregate metrics across all windows for this hotkey
                total_rollouts = 0
                total_unique = 0
                total_successful = 0

                for _window_start, metrics in self._inference_counts[hk].items():
                    total_rollouts += metrics.get("total", 0)
                    total_unique += metrics.get("estimated_unique", 0)
                    total_successful += metrics.get("estimated_successful", 0)

                rollout_stats["total_rollouts"].append(total_rollouts)
                rollout_stats["unique_rollouts"].append(total_unique)
                rollout_stats["successful_rollouts"].append(total_successful)

            # Calculate averages
            avg_total = (
                sum(rollout_stats["total_rollouts"]) / len(rollout_stats["total_rollouts"])
                if rollout_stats["total_rollouts"]
                else 0.0
            )
            avg_unique = (
                sum(rollout_stats["unique_rollouts"]) / len(rollout_stats["unique_rollouts"])
                if rollout_stats["unique_rollouts"]
                else 0.0
            )
            avg_successful = (
                sum(rollout_stats["successful_rollouts"])
                / len(rollout_stats["successful_rollouts"])
                if rollout_stats["successful_rollouts"]
                else 0.0
            )

            # Prepare detailed rollout lines for each top miner
            total_rollout_lines: list[str] = []
            unique_rollout_lines: list[str] = []
            successful_rollout_lines: list[str] = []

            for i, hk in enumerate(top_miner_hotkeys):
                uid = uid_by_hotkey.get(hk)
                ident = str(uid) if uid is not None else hk[:12]
                total_rollout_lines.append(f"{ident}: {rollout_stats['total_rollouts'][i]}")
                unique_rollout_lines.append(f"{ident}: {rollout_stats['unique_rollouts'][i]}")
                successful_rollout_lines.append(
                    f"{ident}: {rollout_stats['successful_rollouts'][i]}"
                )

            # Log average rollouts artifact
            await self._monitor.log_artifact(
                "weights/submission/top_miners_avg_rollouts",
                {
                    "interval": current_interval,
                    "block": current_block,
                    "top_k": top_k,
                    "avg_total_rollouts": avg_total,
                    "text": (
                        f"Average total rollouts for top {top_k} miners: {avg_total:.1f}\n\n"
                        + "\n".join(total_rollout_lines)
                    ),
                },
                "text",
            )

            # Log average unique rollouts artifact
            await self._monitor.log_artifact(
                "weights/submission/top_miners_avg_unique",
                {
                    "interval": current_interval,
                    "block": current_block,
                    "top_k": top_k,
                    "avg_unique_rollouts": avg_unique,
                    "text": (
                        f"Average unique rollouts for top {top_k} miners: {avg_unique:.1f}\n\n"
                        + "\n".join(unique_rollout_lines)
                    ),
                },
                "text",
            )

            # Log average successful rollouts artifact
            await self._monitor.log_artifact(
                "weights/submission/top_miners_avg_successful",
                {
                    "interval": current_interval,
                    "block": current_block,
                    "top_k": top_k,
                    "avg_successful_rollouts": avg_successful,
                    "text": (
                        f"Average successful rollouts for top {top_k} miners: {avg_successful:.1f}\n\n"
                        + "\n".join(successful_rollout_lines)
                    ),
                },
                "text",
            )

            # Log aggregate rollout statistics across all evaluated miners
            await self._log_aggregate_rollout_stats(non_zero_weights)

            # Compute per-UID rollout statistics over the rolling window (12 windows)
            # For all miners with non-zero weight
            await self._log_per_uid_rollout_stats(non_zero_weights, uid_by_hotkey)

    def _compute_target_validation_window(self, current_block: int) -> int:
        """Compute the target window for validation based on current block.

        Validates the last fully completed window, not the in-progress one.
        This ensures we have complete data for the window being validated.

        Args:
            current_block: Current blockchain block number

        Returns:
            Target window start block for validation
        """
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        return current_window - WINDOW_LENGTH

    def _filter_hotkeys_without_failures(self, active_hotkeys: list[str]) -> list[str]:
        """Filter hotkeys to exclude those with recent failures.

        Args:
            active_hotkeys: List of active miner hotkeys

        Returns:
            List of hotkeys without recent failures (failure_count == 0)
        """
        return [hk for hk in active_hotkeys if self._failure_counts.get(hk, 0) == 0]

    async def _compute_window_randomness(self, target_window_hash: str, use_drand: bool) -> str:
        """Derive deterministic per-window randomness with optional drand.

        This must match the logic miners use to ensure prompt validation
        succeeds.

        Args:
            target_window_hash: Block hash at window start
            use_drand: Whether to incorporate drand beacon randomness

        Returns:
            SHA-256 hex digest of combined randomness
        """
        if use_drand:
            try:
                from ..infrastructure.drand import get_drand_beacon

                # Run drand HTTP request in thread pool to avoid blocking
                drand_beacon = await asyncio.to_thread(get_drand_beacon, None)
                logger.info(
                    "🎲 Using drand randomness from round %s",
                    drand_beacon["round"],
                )
                combined_randomness = hashlib.sha256(
                    (target_window_hash + drand_beacon["randomness"]).encode()
                ).hexdigest()
                return combined_randomness
            except Exception as e:
                logger.warning("Failed to get drand, using block hash: %s", e)
                return hashlib.sha256(target_window_hash.encode()).hexdigest()

        return hashlib.sha256(target_window_hash.encode()).hexdigest()

    def _update_rolling(
        self,
        history: deque[set[str]],
        counts: dict[str, int],
        current_set: set[str],
    ) -> None:
        """Update rolling history and counts.

        Args:
            history: Rolling deque of sets
            counts: Counter dict to update
            current_set: Current set to add to history
        """
        # If we're at max length, decrement counts for oldest window
        if len(history) == history.maxlen:
            oldest = history[0]
            for hotkey in oldest:
                counts[hotkey] = max(0, counts.get(hotkey, 0) - 1)

        # Add current window
        history.append(current_set)
        for hotkey in current_set:
            counts[hotkey] = counts.get(hotkey, 0) + 1

    async def _log_sampling_metrics(
        self,
        total: int,
        active: int,
        eligible: int,
        selected: int,
    ) -> None:
        """Log miner sampling metrics.

        Args:
            total: Total miners in metagraph
            active: Active miners with files
            eligible: Eligible miners (not recently failed)
            selected: Selected miners for validation
        """
        if not self._monitor:
            return

        await self._monitor.log_gauge("miner_sampling/miners_total", total)
        await self._monitor.log_gauge("miner_sampling/miners_active", active)
        await self._monitor.log_gauge("miner_sampling/miners_eligible", eligible)
        await self._monitor.log_gauge("miner_sampling/miners_excluded_failures", active - eligible)
        await self._monitor.log_gauge("miner_sampling/miners_selected", selected)

        eff_rate = (selected / active) if active else 0.0
        await self._monitor.log_gauge("miner_sampling/check_rate", eff_rate)

    async def _log_aggregate_rollout_stats(self, non_zero_weights: list[tuple[str, float]]) -> None:
        """Log aggregate rollout statistics across all evaluated miners.

        Computes and logs 9 aggregate metrics under weights/aggregate_stats/:
        - total_rollouts_avg/min/max
        - unique_rollouts_avg/min/max
        - successful_rollouts_avg/min/max

        These show overall statistics across all miners over the 12-window rolling period.

        Args:
            non_zero_weights: List of (hotkey, weight) tuples for miners with non-zero weight
        """
        if not self._monitor:
            return

        # Collect per-miner aggregate rollouts
        miner_total_rollouts: list[int] = []
        miner_unique_rollouts: list[int] = []
        miner_successful_rollouts: list[int] = []

        for hk, _weight in non_zero_weights:
            # Aggregate metrics across all windows for this hotkey
            total_rollouts = 0
            total_unique = 0
            total_successful = 0

            for _window_start, metrics in self._inference_counts[hk].items():
                total_rollouts += metrics.get("total", 0)
                total_unique += metrics.get("estimated_unique", 0)
                total_successful += metrics.get("estimated_successful", 0)

            miner_total_rollouts.append(total_rollouts)
            miner_unique_rollouts.append(total_unique)
            miner_successful_rollouts.append(total_successful)

        # Compute aggregate statistics for total rollouts
        if miner_total_rollouts:
            total_avg = sum(miner_total_rollouts) / len(miner_total_rollouts)
            total_min = min(miner_total_rollouts)
            total_max = max(miner_total_rollouts)

            await self._monitor.log_gauge("weights/aggregate_stats/total_rollouts_avg", total_avg)
            await self._monitor.log_gauge(
                "weights/aggregate_stats/total_rollouts_min", float(total_min)
            )
            await self._monitor.log_gauge(
                "weights/aggregate_stats/total_rollouts_max", float(total_max)
            )

        # Compute aggregate statistics for unique rollouts
        if miner_unique_rollouts:
            unique_avg = sum(miner_unique_rollouts) / len(miner_unique_rollouts)
            unique_min = min(miner_unique_rollouts)
            unique_max = max(miner_unique_rollouts)

            await self._monitor.log_gauge("weights/aggregate_stats/unique_rollouts_avg", unique_avg)
            await self._monitor.log_gauge(
                "weights/aggregate_stats/unique_rollouts_min", float(unique_min)
            )
            await self._monitor.log_gauge(
                "weights/aggregate_stats/unique_rollouts_max", float(unique_max)
            )

        # Compute aggregate statistics for successful rollouts
        if miner_successful_rollouts:
            successful_avg = sum(miner_successful_rollouts) / len(miner_successful_rollouts)
            successful_min = min(miner_successful_rollouts)
            successful_max = max(miner_successful_rollouts)

            await self._monitor.log_gauge(
                "weights/aggregate_stats/successful_rollouts_avg", successful_avg
            )
            await self._monitor.log_gauge(
                "weights/aggregate_stats/successful_rollouts_min", float(successful_min)
            )
            await self._monitor.log_gauge(
                "weights/aggregate_stats/successful_rollouts_max", float(successful_max)
            )

    async def _log_per_uid_rollout_stats(
        self, non_zero_weights: list[tuple[str, float]], uid_by_hotkey: dict[str, int]
    ) -> None:
        """Log per-UID rollout statistics over the rolling window.

        For each miner with non-zero weight, computes and logs:
        - total_rollouts_avg/min/max
        - unique_rollouts_avg/min/max
        - successful_rollouts_avg/min/max
        - windows_count

        Metrics are logged under {uid_str}/ namespace for per-UID tracking.

        Args:
            non_zero_weights: List of (hotkey, weight) tuples for miners with non-zero weight
            uid_by_hotkey: Mapping of hotkey to UID
        """
        if not self._monitor:
            return

        for hk, _weight in non_zero_weights:
            uid = uid_by_hotkey.get(hk)
            if uid is None:
                continue  # Skip if UID not found

            uid_str = str(uid)

            # Collect rollout counts per window for this miner
            window_total_rollouts: list[int] = []
            window_unique_rollouts: list[int] = []
            window_successful_rollouts: list[int] = []

            for _window_start, metrics in self._inference_counts[hk].items():
                window_total_rollouts.append(metrics.get("total", 0))
                window_unique_rollouts.append(metrics.get("estimated_unique", 0))
                window_successful_rollouts.append(metrics.get("estimated_successful", 0))

            # Compute statistics if we have data
            if window_total_rollouts:
                # Total rollouts statistics
                total_avg = sum(window_total_rollouts) / len(window_total_rollouts)
                total_min = min(window_total_rollouts)
                total_max = max(window_total_rollouts)

                await self._monitor.log_gauge(f"{uid_str}/total_rollouts_avg", total_avg)
                await self._monitor.log_gauge(f"{uid_str}/total_rollouts_min", float(total_min))
                await self._monitor.log_gauge(f"{uid_str}/total_rollouts_max", float(total_max))

            if window_unique_rollouts:
                # Unique rollouts statistics
                unique_avg = sum(window_unique_rollouts) / len(window_unique_rollouts)
                unique_min = min(window_unique_rollouts)
                unique_max = max(window_unique_rollouts)

                await self._monitor.log_gauge(f"{uid_str}/unique_rollouts_avg", unique_avg)
                await self._monitor.log_gauge(f"{uid_str}/unique_rollouts_min", float(unique_min))
                await self._monitor.log_gauge(f"{uid_str}/unique_rollouts_max", float(unique_max))

            if window_successful_rollouts:
                # Successful rollouts statistics
                successful_avg = sum(window_successful_rollouts) / len(window_successful_rollouts)
                successful_min = min(window_successful_rollouts)
                successful_max = max(window_successful_rollouts)

                await self._monitor.log_gauge(
                    f"{uid_str}/successful_rollouts_avg",
                    successful_avg,
                )
                await self._monitor.log_gauge(
                    f"{uid_str}/successful_rollouts_min",
                    float(successful_min),
                )
                await self._monitor.log_gauge(
                    f"{uid_str}/successful_rollouts_max",
                    float(successful_max),
                )

            # Log window count for this miner (useful for context)
            await self._monitor.log_gauge(
                f"{uid_str}/windows_count",
                float(len(window_total_rollouts)),
            )

    def cleanup(self) -> None:
        """Clean up resources.

        Stops background tasks like the chain manager worker process.
        Call this before shutdown.
        """
        # Make idempotent: ensure we only run once even if called from multiple paths
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True

        if self._chain_manager:
            try:
                self._chain_manager.stop()
                logger.info("Stopped chain manager")
            except Exception as e:
                logger.warning(f"Error stopping chain manager: {e}")
