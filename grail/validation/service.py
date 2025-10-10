"""Core validation service for GRAIL protocol.

Orchestrates the validation loop, window processing, and weight submission.
Separated from CLI concerns for better testability and maintainability.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Any

import bittensor as bt
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.chain import GrailChainManager
from ..infrastructure.checkpoints import CheckpointManager
from ..infrastructure.credentials import BucketCredentials
from ..mining.rollout_generator import REASONING_START, SYSTEM_PROMPT
from ..model.provider import (
    clear_model_and_tokenizer,
    get_model,
    get_tokenizer,
)
from ..scoring.weights import WeightComputer
from ..shared.chat_templates import build_qwen_chat_template
from ..shared.constants import (
    FAILURE_LOOKBACK_WINDOWS,
    MINER_SAMPLE_MAX,
    MINER_SAMPLE_MIN,
    MINER_SAMPLE_RATE,
    MINER_SAMPLING_ENABLED,
    TRAINER_UID,
    WINDOW_LENGTH,
)
from .copycat_service import COPYCAT_SERVICE
from .miner_validator import MinerValidator
from .pipeline import ValidationPipeline
from .sampling import MinerSampler
from .window_processor import WindowProcessor

logger = logging.getLogger(__name__)

# Weight submission constants
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 360
WEIGHT_ROLLING_WINDOWS = int(WEIGHT_SUBMISSION_INTERVAL_BLOCKS / WINDOW_LENGTH)


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
        sat_pipeline: ValidationPipeline,
        weight_computer: WeightComputer,
        credentials: BucketCredentials,
        checkpoint_manager: CheckpointManager,
        sat_reward_bounds: tuple[float, float],
        monitor: Any | None = None,
    ):
        """Initialize validation service.

        Args:
            wallet: Validator wallet for signing transactions
            netuid: Network UID for the subnet
            sat_pipeline: SAT validation pipeline for rollout verification
            weight_computer: Weight computation engine
            credentials: Object storage credentials for rollout access
            checkpoint_manager: Checkpoint manager for model downloads
            sat_reward_bounds: (low, high) bounds for reward validation
            monitor: Optional monitoring client for metrics
        """
        self._wallet = wallet
        self._netuid = netuid
        self._sat_pipeline = sat_pipeline
        self._weight_computer = weight_computer
        self._credentials = credentials
        self._checkpoint_manager = checkpoint_manager
        self._sat_reward_bounds = sat_reward_bounds
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
                    await asyncio.sleep(5)
                    logger.debug(f"Waiting for new window {target_window}")
                    continue

                # Set monitoring context
                if self._monitor:
                    self._monitor.set_block_context(current_block, target_window)

                # Load checkpoint for this window
                checkpoint_loaded = await self._load_checkpoint_for_window(target_window)
                if not checkpoint_loaded:
                    await asyncio.sleep(30)
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

                # Submit weights if interval reached
                await self._submit_weights_if_ready(current_block, meta)

                # Update state
                self._last_processed_window = target_window
                self._windows_processed_since_start += 1

            except asyncio.CancelledError:
                logger.info("Validation loop cancelled")
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
        sat_reward_low, sat_reward_high = self._sat_reward_bounds
        self._miner_validator = MinerValidator(
            sat_pipeline=self._sat_pipeline,
            sat_reward_bounds=(sat_reward_low, sat_reward_high),
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

        Args:
            target_window: Target window to validate

        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        checkpoint_window = target_window - WINDOW_LENGTH

        # Get trainer's bucket for checkpoints
        if self._chain_manager and self._current_checkpoint_id is None:
            trainer_bucket = self._chain_manager.get_bucket(TRAINER_UID)
            if trainer_bucket:
                logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")
                self._checkpoint_manager.credentials = trainer_bucket
            else:
                logger.warning(
                    f"⚠️ Trainer UID {TRAINER_UID} bucket not found, using local credentials"
                )

        # Try to get checkpoint
        checkpoint_path = None
        try:
            timer_ctx = (
                self._monitor.timer("validation/checkpoint_download")
                if self._monitor
                else contextlib.nullcontext()
            )
            with timer_ctx:
                if checkpoint_window >= 0:
                    checkpoint_path = await self._checkpoint_manager.get_checkpoint(
                        checkpoint_window
                    )
        except Exception:
            logger.warning(
                f"Failed to resolve checkpoint path for target_window={target_window} "
                f"(ckpt={checkpoint_window})"
            )

        if not checkpoint_path:
            logger.warning(f"No checkpoint available for window {target_window}, skipping")
            return False

        # Load if new checkpoint or models not loaded
        if (
            str(checkpoint_path) != self._current_checkpoint_id
            or self._model is None
            or self._tokenizer is None
        ):
            try:
                logger.info(
                    f"🚀 Loading checkpoint for validation window {target_window} "
                    f"from {checkpoint_path}"
                )
                # Pre-load cleanup to prevent VRAM growth
                self._model, self._tokenizer = clear_model_and_tokenizer(
                    self._model, self._tokenizer
                )
                chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
                self._model = get_model(str(checkpoint_path), device=None, eval_mode=True)
                self._tokenizer = get_tokenizer(str(checkpoint_path), chat_template=chat_template)
                self._current_checkpoint_id = str(checkpoint_path)
                return True
            except Exception:
                logger.exception(f"Failed to load checkpoint for window {target_window}")
                return False

        return True

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
        window_rand = target_window_hash  # TODO: Integrate drand if use_drand

        # Discover active miners
        timer_ctx = (
            self._monitor.timer("validation/hotkeys_discovery")
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

        # Process window (validate all miners)
        if self._window_processor is None:
            raise RuntimeError("WindowProcessor not initialized")
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")
        if self._chain_manager is None:
            raise RuntimeError("Chain manager not initialized")
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

    async def _submit_weights_if_ready(self, current_block: int, meta: Any) -> None:
        """Submit weights if interval has been reached.

        Args:
            current_block: Current block number
            meta: Metagraph instance
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

        # Compute weights
        weights, non_zero_weights = self._weight_computer.compute_weights(
            meta_hotkeys=list(meta.hotkeys),
            meta_uids=list(meta.uids),
            inference_counts=self._inference_counts,
            target_window=current_block,
            availability_counts=self._availability_counts,
        )

        # Submit to chain
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

    def cleanup(self) -> None:
        """Clean up resources.

        Stops background tasks like the chain manager worker process.
        Call this before shutdown.
        """
        if self._chain_manager:
            try:
                self._chain_manager.stop()
                logger.info("Stopped chain manager")
            except Exception as e:
                logger.warning(f"Error stopping chain manager: {e}")
