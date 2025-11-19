"""Continuous training process for async trainer.

Runs as separate process that:
1. Trains continuously without window boundaries
2. Saves model snapshots after each epoch
3. Monitors PAUSE_TRAINING flag for evaluation coordination
4. Updates heartbeat for liveness monitoring
"""

from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

import bittensor as bt
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.network import create_subtensor
from grail.model.train_loading import ModelLoadSpec, load_training_artifacts
from grail.monitoring import get_monitoring_manager, initialize_monitoring
from grail.shared.constants import NETUID, WINDOW_LENGTH, is_kl_enabled
from grail.trainer.algorithms import GRPOAlgorithm, TrainingAlgorithm
from grail.trainer.algorithms.grpo import load_grpo_groups
from grail.trainer.config import TrainingConfig
from grail.trainer.snapshot_manager import SnapshotManager
from grail.trainer.trust import get_trusted_miner_hotkeys

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────

# Optimizer hyperparameters
OPTIMIZER_BETAS = (0.9, 0.999)
OPTIMIZER_WEIGHT_DECAY = 0.1

# Scheduler hyperparameters
TOTAL_TRAINING_WINDOWS = 1000
WARMUP_FRACTION = 0.05
SCHEDULER_ETA_MIN = 1e-7

# Training loop constants
PAUSE_CHECK_INTERVAL_SECONDS = 5
NO_MINERS_SLEEP_SECONDS = 60
NO_DATA_SLEEP_SECONDS = 60
EPOCH_FAILURE_SLEEP_SECONDS = 30
LOOP_ERROR_SLEEP_SECONDS = 30
INITIAL_CHECKPOINT_WAIT_INTERVAL_SECONDS = 30

# Trust computation timeout
TRUST_COMPUTATION_TIMEOUT_SECONDS = 5.0

# Replay buffer constants
INITIAL_LAST_LOADED_WINDOW = -1


# ────────────────────────────────────────────────────────────────────────────────
# Training Service
# ────────────────────────────────────────────────────────────────────────────────


class TrainingService:
    """Service for continuous model training with snapshot management.

    Encapsulates all training logic including:
    - Resource initialization (models, optimizer, scheduler, subtensor, chain manager)
    - Initial checkpoint upload and synchronization
    - Continuous training loop with pause/resume support
    - Snapshot management and heartbeat monitoring
    """

    def __init__(
        self,
        config: TrainingConfig,
        snapshot_manager: SnapshotManager,
        credentials: Any,
        wallet: bt.wallet,
        monitor: Any | None = None,
        *,
        train_spec: ModelLoadSpec,
        ref_spec: ModelLoadSpec,
    ) -> None:
        """Initialize training service.

        Args:
            config: Training configuration
            snapshot_manager: Snapshot manager for IPC
            credentials: R2 credentials for loading rollouts
            wallet: Wallet for trust computation
            monitor: Optional monitoring manager
            train_spec: Specification for loading the training model/tokenizer
            ref_spec: Specification for loading the reference model
        """
        self.config = config
        self.snapshot_manager = snapshot_manager
        self.credentials = credentials
        self.wallet = wallet
        self.monitor = monitor
        self.train_spec = train_spec
        self.ref_spec = ref_spec

        # Async resources (initialized in run())
        self.subtensor: bt.subtensor | None = None
        self.chain_manager: GrailChainManager | None = None
        self.train_model: Any = None
        self.ref_model: Any = None
        self.tokenizer: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.algorithm: TrainingAlgorithm | None = None

        # Training state
        self.epoch_counter: int = 0
        self.last_loaded_window: int = INITIAL_LAST_LOADED_WINDOW
        self.groups: list[Any] = []

    async def run(
        self,
        stop_event: multiprocessing.Event,
    ) -> None:
        """Run continuous training service.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Training service starting")

        # Initialize all resources
        await self._initialize_resources()

        # Upload initial checkpoint and wait for miners
        await self._upload_initial_checkpoint(stop_event)

        # Wait 1 window for miners to download
        await self._wait_for_miners(stop_event)

        # Run continuous training loop
        await self._training_loop(stop_event)

        logger.info("Training service exiting (epochs completed: %d)", self.epoch_counter)


    async def _wait_for_miners(self, stop_event: multiprocessing.Event) -> None:
        """Wait for miners to download initial checkpoint.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Waiting for miners to download initial checkpoint...")
        # Wait 0 window for miners to download
        # TODO: Later on when we use extra process/GPUs, some waiting should be added here.
        WAIT_WINDOW_LENGTH = 0.0  # Wait 0 window for miners to download
        
        # Get current block first before using it
        current_block = await self.subtensor.get_current_block()
        target_block = current_block + WAIT_WINDOW_LENGTH
        logger.info("Waiting %s window for miners (until block %s)", WAIT_WINDOW_LENGTH, target_block)

        while not stop_event.is_set():
            current_block = await self.subtensor.get_current_block()
            if current_block >= target_block:
                logger.info("Wait complete, starting continuous training")
                break
            await asyncio.sleep(INITIAL_CHECKPOINT_WAIT_INTERVAL_SECONDS)

        logger.info("Waiting for miners to download initial checkpoint complete")

    async def _initialize_resources(self) -> None:
        """Initialize all async and GPU resources.
        """
        logger.info(
            "Loading training artifacts (train_spec=%s, ref_spec=%s)",
            self.train_spec,
            self.ref_spec,
        )
        checkpoint_manager = CheckpointManager(
            cache_root=default_checkpoint_cache_root(),
            credentials=self.credentials,
        )
        load_ref_model = is_kl_enabled()
        train_model, ref_model, tokenizer = await load_training_artifacts(
            self.train_spec,
            self.ref_spec,
            checkpoint_manager,
            load_ref_model=load_ref_model,
        )
        self.train_model = train_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        logger.info(
            "Training artifacts loaded (train=%s, ref=%s, ref_enabled=%s)",
            getattr(self.train_model, "name_or_path", "unknown"),
            getattr(self.ref_model, "name_or_path", "disabled"),
            load_ref_model,
        )

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create algorithm
        self.algorithm = GRPOAlgorithm(config=self.config)

        # Create subtensor (must be inside async context)
        logger.info("Creating subtensor connection...")
        self.subtensor = await create_subtensor(resilient=True)

        # Initialize chain manager
        logger.info("Initializing chain manager...")
        metagraph = await self.subtensor.metagraph(NETUID)
        chain_config = SimpleNamespace(netuid=NETUID)
        self.chain_manager = GrailChainManager(
            chain_config,
            self.wallet,
            metagraph,
            self.subtensor,
            self.credentials,
        )
        await self.chain_manager.initialize()
        logger.info("Chain manager initialized")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with configured hyperparameters.

        Returns:
            Configured optimizer instance
        """
        return torch.optim.AdamW(
            self.train_model.parameters(),
            lr=self.config.lr,
            betas=OPTIMIZER_BETAS,
            weight_decay=OPTIMIZER_WEIGHT_DECAY,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup and cosine annealing.

        Returns:
            Configured scheduler instance
        """
        warmup_steps = max(1, int(WARMUP_FRACTION * TOTAL_TRAINING_WINDOWS))

        def lr_lambda_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda_warmup)
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=TOTAL_TRAINING_WINDOWS - warmup_steps,
            eta_min=SCHEDULER_ETA_MIN,
        )

        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    async def _upload_initial_checkpoint(self, stop_event: multiprocessing.Event) -> None:
        """Upload initial checkpoint and wait for miners to download.

        Args:
            stop_event: Event to signal shutdown
        """
        current_block = await self.subtensor.get_current_block()
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

        logger.info("Saving initial checkpoint for window %s (upload handled by worker)", current_window)

        # Prepare training config for metadata
        training_config = {
            "lr": self.config.lr,
            "seed": current_window,
        }

        # Save initial snapshot
        self.snapshot_manager.save_snapshot_atomic(
            self.train_model,
            self.tokenizer,
            {
                "epoch": 0,
                "timestamp": time.time(),
                "status": "initial_upload",
                "training_config": training_config,
                "parent_window": max(0, current_window - WINDOW_LENGTH),
            },
        )

    async def _training_loop(self, stop_event: multiprocessing.Event) -> None:
        """Main continuous training loop.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Training loop starting (PID=%d)", multiprocessing.current_process().pid)

        # Prepare models with Accelerator
        accelerator = Accelerator(mixed_precision="no")
        train_model, ref_model, optimizer = self._prepare_models_with_accelerator(accelerator)

        logger.info("Models and optimizer prepared with Accelerator")

        while not stop_event.is_set():
            try:
                # Update heartbeat
                self.snapshot_manager.set_training_heartbeat()

                # Check PAUSE_TRAINING flag
                if self.snapshot_manager.check_pause_flag():
                    train_model, ref_model, optimizer = await self._handle_pause(
                        train_model,
                        ref_model,
                        optimizer,
                        accelerator,
                        stop_event,
                    )
                    if stop_event.is_set():
                        break
                    continue

                # Get current window
                current_window = await self._get_current_window()

                # Seed RNGs for reproducibility
                _seed_all(current_window + self.epoch_counter)

                # Get trusted miners
                trusted_hotkeys = await self._get_trusted_miners()
                if not trusted_hotkeys:
                    await asyncio.sleep(NO_MINERS_SLEEP_SECONDS)
                    continue

                # Load GRPO groups (Replay Buffer)
                await self._load_grpo_groups(current_window, trusted_hotkeys)

                if not self.groups:
                    logger.warning("No data available in replay buffer, sleeping %ds", NO_DATA_SLEEP_SECONDS)
                    await asyncio.sleep(NO_DATA_SLEEP_SECONDS)
                    continue

                # Train one epoch
                metrics = await self._train_epoch(train_model, ref_model, accelerator, current_window)

                # Save snapshot
                self._save_snapshot(train_model, accelerator, current_window, metrics)

                # Update heartbeat
                self.snapshot_manager.set_training_heartbeat()

            except asyncio.CancelledError:
                logger.info("Training process received CancelledError, exiting")
                break
            except Exception as exc:
                logger.exception("Training loop error: %s", exc)
                await asyncio.sleep(LOOP_ERROR_SLEEP_SECONDS)

    def _prepare_models_with_accelerator(
        self,
        accelerator: Accelerator,
    ) -> tuple[Any, Any | None, torch.optim.Optimizer]:
        """Prepare models and optimizer with Accelerator for distributed training.

        Args:
            accelerator: Accelerator instance

        Returns:
            Tuple of (train_model, ref_model, optimizer)
        """
        kl_enabled = is_kl_enabled()

        if kl_enabled and self.ref_model is not None:
            train_model, ref_model, optimizer = accelerator.prepare(
                self.train_model,
                self.ref_model,
                self.optimizer,
            )
            if hasattr(ref_model, "eval"):
                ref_model.eval()
        else:
            train_model, optimizer = accelerator.prepare(
                self.train_model,
                self.optimizer,
            )
            ref_model = None

        return train_model, ref_model, optimizer

    async def _handle_pause(
        self,
        train_model: Any,
        ref_model: Any | None,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        stop_event: multiprocessing.Event,
    ) -> tuple[Any, Any | None, torch.optim.Optimizer]:
        """Handle PAUSE_TRAINING flag by moving models to CPU and waiting.

        Args:
            train_model: Training model
            ref_model: Reference model (or None)
            optimizer: Optimizer
            accelerator: Accelerator instance
            stop_event: Event to signal shutdown

        Returns:
            Tuple of (train_model, ref_model, optimizer) after resuming
        """
        logger.info("PAUSE_TRAINING flag detected, freeing GPU for evaluation...")

        # Unwrap and move models to CPU
        unwrapped_train = accelerator.unwrap_model(train_model)
        unwrapped_ref = accelerator.unwrap_model(ref_model) if ref_model else None

        logger.info("Moving models to CPU to free GPU memory...")
        train_model_cpu = unwrapped_train.cpu()
        ref_model_cpu = unwrapped_ref.cpu() if unwrapped_ref else None

        # Clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb, _ = torch.cuda.mem_get_info()
            logger.info("GPU memory freed: %.2f GB available", free_gb / (1024**3))

        # Save current state before pausing
        self.snapshot_manager.save_snapshot_atomic(
            train_model_cpu,
            self.tokenizer,
            {
                "epoch": self.epoch_counter,
                "timestamp": time.time(),
                "status": "paused_for_evaluation",
            },
        )
        logger.info("Saved snapshot before pause (models on CPU)")

        # Wait for pause flag to be cleared
        while self.snapshot_manager.check_pause_flag() and not stop_event.is_set():
            await asyncio.sleep(PAUSE_CHECK_INTERVAL_SECONDS)

        if stop_event.is_set():
            return train_model_cpu, ref_model_cpu, optimizer

        # Move models back to GPU
        logger.info("PAUSE_TRAINING cleared, moving models back to GPU...")
        train_model_gpu = train_model_cpu.cuda()
        ref_model_gpu = ref_model_cpu.cuda() if ref_model_cpu else None

        # Re-prepare with accelerator
        if ref_model_gpu:
            train_model, ref_model, optimizer = accelerator.prepare(
                train_model_gpu, ref_model_gpu, optimizer
            )
            ref_model.eval()
        else:
            train_model, optimizer = accelerator.prepare(train_model_gpu, optimizer)
            ref_model = None

        logger.info("Models moved back to GPU, resuming training")
        return train_model, ref_model, optimizer

    async def _get_current_window(self) -> int:
        """Get current window number from subtensor.

        Returns:
            Current window number
        """
        try:
            current_block = await self.subtensor.get_current_block()
            return (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        except Exception as exc:
            logger.warning("Failed to get current block: %s", exc)
            return self.epoch_counter * WINDOW_LENGTH

    async def _get_trusted_miners(self) -> list[str]:
        """Get trusted miner hotkeys from metagraph with fallback to cache.

        Returns:
            List of trusted miner hotkeys (may be cached if blockchain unavailable)
        """
        try:
            metagraph = await self.subtensor.metagraph(NETUID)
            trusted_hotkeys = await get_trusted_miner_hotkeys(
                metagraph,
                self.config.min_aggregate_weight,
                self.config.min_trusted_miners,
                timeout=TRUST_COMPUTATION_TIMEOUT_SECONDS,
                subtensor=self.subtensor,
            )

            if not trusted_hotkeys:
                logger.warning("No trusted miners found")

            return trusted_hotkeys

        except Exception as exc:
            logger.error("Failed to get trusted miners: %s", exc)
            return []

    async def _load_grpo_groups(self, current_window: int, trusted_hotkeys: list[str]) -> None:
        """Load GRPO groups from trusted miners (Replay Buffer Logic).

        Args:
            current_window: Current window number
            trusted_hotkeys: List of trusted miner hotkeys
        """
        # Train on the PREVIOUS (completed) window to ensure consistent data
        target_data_window = current_window - WINDOW_LENGTH

        # Only load if window changed and is valid
        if target_data_window == self.last_loaded_window or target_data_window < 0:
            return

        try:
            metagraph = await self.subtensor.metagraph(NETUID)
            uid_by_hotkey = dict(zip(metagraph.hotkeys, metagraph.uids, strict=True))

            logger.info(
                "Loading GRPO groups for window %s (previously loaded: %s)",
                target_data_window,
                self.last_loaded_window,
            )

            new_groups = await load_grpo_groups(
                target_data_window,
                self.config.group_adv_sum_tolerance,
                trusted_hotkeys,
                self.credentials,
                self.chain_manager,
                uid_by_hotkey,
                self.config,
                self.monitor,
                self.tokenizer.eos_token_id,
            )

            if new_groups:
                logger.info(
                    "Loaded %d GRPO groups for window %s (updating replay buffer)",
                    len(new_groups),
                    target_data_window,
                )
                self.groups = new_groups
                self.last_loaded_window = target_data_window
            else:
                logger.warning(
                    "No valid GRPO groups found for window %s, retaining old groups (%d)",
                    target_data_window,
                    len(self.groups),
                )

        except Exception as exc:
            logger.error("Failed to load GRPO groups: %s", exc)
            # Don't clear existing groups on failure

    async def _train_epoch(
        self,
        train_model: Any,
        ref_model: Any | None,
        accelerator: Accelerator,
        current_window: int,
    ) -> dict[str, Any]:
        """Train one epoch on current groups.

        Args:
            train_model: Training model
            ref_model: Reference model (or None)
            accelerator: Accelerator instance
            current_window: Current window number

        Returns:
            Training metrics dictionary
        """
        try:
            epoch_start = time.time()
            logger.info("Training epoch %d with %d groups", self.epoch_counter + 1, len(self.groups))

            metrics = await self.algorithm.train_epoch(
                train_model,
                ref_model,
                self.tokenizer,
                self.groups,
                self.optimizer,
                accelerator,
                self.monitor,
                current_window,
                self.config,
            )

            # Update scheduler
            self.scheduler.step()

            epoch_duration = time.time() - epoch_start
            logger.info(
                "Epoch %d complete in %.1fs: loss=%.4f reward_mean=%.4f",
                self.epoch_counter + 1,
                epoch_duration,
                metrics.get("loss_total", 0.0),
                metrics.get("reward_mean", 0.0),
            )

            self.epoch_counter += 1
            return metrics

        except Exception as exc:
            logger.exception("Training epoch failed: %s", exc)
            await asyncio.sleep(EPOCH_FAILURE_SLEEP_SECONDS)
            return {}

    def _save_snapshot(
        self,
        train_model: Any,
        accelerator: Accelerator,
        current_window: int,
        metrics: dict[str, Any],
    ) -> None:
        """Save model snapshot atomically.

        Args:
            train_model: Training model
            accelerator: Accelerator instance
            current_window: Current window number
            metrics: Training metrics
        """
        try:
            unwrapped_train = accelerator.unwrap_model(train_model)

            self.snapshot_manager.save_snapshot_atomic(
                unwrapped_train,
                self.tokenizer,
                {
                    "epoch": self.epoch_counter,
                    "timestamp": time.time(),
                    "window": current_window,
                    "parent_window": current_window - WINDOW_LENGTH,
                    "metrics": metrics,
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
                },
            )

            logger.info("Snapshot saved for epoch %d", self.epoch_counter)

        except Exception as exc:
            logger.error("Failed to save snapshot: %s", exc)
            # Continue training even if snapshot fails


# ────────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────────


def _seed_all(seed: int) -> None:
    """Seed all RNGs for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_child_process_logging() -> None:
    """Configure logging for child process.

    Child processes need handlers configured even though file descriptors
    are inherited from parent (redirected to train.log).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [training_process] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Force immediate flush
    sys.stdout.flush()
    sys.stderr.flush()


def _reconstruct_wallet(wallet_args: dict[str, str]) -> bt.wallet:
    """Reconstruct wallet from serialized arguments.

    Args:
        wallet_args: Wallet configuration (name, hotkey, path)

    Returns:
        Reconstructed wallet instance
    """
    return bt.wallet(**wallet_args)


def _initialize_monitoring(monitor_config: dict[str, Any]) -> Any | None:
    """Initialize monitoring in child process.

    Args:
        monitor_config: Monitoring configuration

    Returns:
        Monitoring manager instance or None if initialization fails
    """
    if not monitor_config:
        return None

    try:
        backend_type = monitor_config.get("backend_type", "wandb")
        init_config = {k: v for k, v in monitor_config.items() if k != "backend_type"}

        if "run_id" in init_config:
            logger.info(
                "Resuming W&B run %s in training process for multi-process logging",
                init_config["run_id"],
            )

        initialize_monitoring(backend_type=backend_type, **init_config)
        monitor = get_monitoring_manager()
        logger.info("Monitoring initialized in training process")
        return monitor

    except Exception as exc:
        logger.warning("Failed to initialize monitoring in child process: %s", exc)
        return None


# ────────────────────────────────────────────────────────────────────────────────
# Process Entry Point
# ────────────────────────────────────────────────────────────────────────────────


def run_training_process(
    train_spec: ModelLoadSpec,
    ref_spec: ModelLoadSpec,
    config: TrainingConfig,
    snapshot_manager: SnapshotManager,
    credentials: Any,
    wallet_args: dict[str, str],
    monitor_config: dict[str, Any],
    stop_event: multiprocessing.Event,
) -> None:
    """Entry point for training process.

    Loads models from specs (resolved inside child process) to avoid memory duplication.
    Main process never initializes CUDA, so no fork issues.
    Creates its own subtensor and chain manager connections.

    Args:
        train_spec: Training model load specification
        ref_spec: Reference model load specification
        config: Training configuration
        snapshot_manager: Snapshot manager for IPC
        credentials: R2 credentials
        wallet_args: Wallet configuration (name, hotkey, path)
        monitor_config: Monitoring configuration
        stop_event: Event to signal shutdown
    """
    # Configure logging for child process
    _configure_child_process_logging()

    logger.info("Training process starting (PID=%d)", multiprocessing.current_process().pid)
    sys.stdout.flush()

    # Reconstruct objects from primitives
    try:
        logger.info("Reconstructing wallet and services in child process...")
        wallet = _reconstruct_wallet(wallet_args)
        monitor = _initialize_monitoring(monitor_config)

    except Exception as exc:
        logger.exception("Failed to reconstruct services in training process: %s", exc)
        return

    # Run training service
    try:
        service = TrainingService(
            config=config,
            snapshot_manager=snapshot_manager,
            credentials=credentials,
            wallet=wallet,
            monitor=monitor,
            train_spec=train_spec,
            ref_spec=ref_spec,
        )

        asyncio.run(service.run(stop_event))

    except KeyboardInterrupt:
        logger.info("Training process interrupted")
    except Exception as exc:
        logger.exception("Training process crashed: %s", exc)
    finally:
        logger.info("Training process exiting")
