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
import json
import logging
import multiprocessing
import os
import random
import sys
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from grail.monitoring import initialize_subprocess_monitoring
from grail.shared.constants import NETUID, TOTAL_TRAINING_WINDOWS, WINDOW_LENGTH, is_kl_enabled
from grail.trainer.algorithms import GRPOAlgorithm, TrainingAlgorithm
from grail.trainer.algorithms.grpo import load_grpo_groups
from grail.trainer.config import TrainingConfig
from grail.trainer.ipc import IPCChannels
from grail.trainer.replay_buffer import ReplayBuffer, create_replay_buffer
from grail.trainer.snapshot_manager import SnapshotManager
from grail.trainer.trust import get_trust_list_from_validator, get_trusted_miner_hotkeys

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────

# Optimizer hyperparameters
OPTIMIZER_BETAS = (0.9, 0.95)
OPTIMIZER_WEIGHT_DECAY = 0.1

# Scheduler hyperparameters
WARMUP_FRACTION = float(os.getenv("GRAIL_WARMUP_FRACTION", "0.05"))
SCHEDULER_ETA_MIN = float(os.getenv("GRAIL_SCHEDULER_ETA_MIN", "1e-7"))
# LR scheduler type: "constant" (warmup then constant) or "cosine" (warmup then cosine decay)
LR_SCHEDULER_TYPE = os.getenv("GRAIL_LR_SCHEDULER_TYPE", "constant").strip().lower()

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
        monitor_config: dict[str, Any] | None = None,
        *,
        train_spec: ModelLoadSpec,
        ref_spec: ModelLoadSpec,
        ipc: IPCChannels | None = None,
        test_mode: bool = False,
    ) -> None:
        """Initialize training service.

        Args:
            config: Training configuration
            snapshot_manager: Snapshot manager for IPC
            credentials: R2 credentials for loading rollouts
            wallet: Wallet for trust computation
            monitor_config: Optional monitoring configuration (will be initialized in async context)
            train_spec: Specification for loading the training model/tokenizer
            ref_spec: Specification for loading the reference model
            ipc: IPC channels for coordination (None = use filesystem fallback)
            test_mode: Test mode flag for training on TRAINER_UID data only
        """
        self.config = config
        self.snapshot_manager = snapshot_manager
        self.credentials = credentials
        self.wallet = wallet
        self.monitor_config = monitor_config
        self.monitor: Any | None = None
        self.train_spec = train_spec
        self.ref_spec = ref_spec
        self.test_mode = test_mode

        # IPC channels for all inter-process communication (with filesystem fallback)
        self._ipc = ipc

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
        self._full_checkpoint_saved: bool = (
            False  # Track if full checkpoint was saved at TOTAL_TRAINING_WINDOWS
        )
        self._epochs_this_window: int = 0  # Track epochs within current window for wandb logging
        self._current_training_window: int = (
            INITIAL_LAST_LOADED_WINDOW  # Current window being trained
        )
        self._windows_completed: int = (
            0  # Track total windows completed for HuggingFace upload trigger
        )
        self._background_upload_task: asyncio.Task[bool] | None = (
            None  # Keep reference to prevent GC
        )

        # Initialize replay buffer
        if config.replay_buffer_enabled:
            self.replay_buffer: ReplayBuffer | None = create_replay_buffer(
                buffer_type="recency_weighted",
                max_windows=config.replay_buffer_max_windows,
                recent_window_fraction=config.replay_buffer_recent_fraction,
                decay_factor=config.replay_buffer_decay_factor,
            )
            logger.info(
                "Replay buffer initialized (max_windows=%d, recent_fraction=%.2f, decay_factor=%.2f)",
                config.replay_buffer_max_windows,
                config.replay_buffer_recent_fraction,
                config.replay_buffer_decay_factor,
            )
        else:
            self.replay_buffer = None
            logger.info("Replay buffer disabled, using single-window training")

    def _update_heartbeat(self) -> None:
        """Update heartbeat timestamp via IPC channels (primary) or filesystem (fallback).

        Uses atomic Value update for reliable IPC without file I/O overhead.
        Also updates filesystem for crash recovery.
        """
        if self._ipc is not None:
            self._ipc.update_heartbeat()
        # Also update filesystem for crash recovery / backward compat
        self.snapshot_manager.set_training_heartbeat()

    async def run(
        self,
        stop_event: Event,
    ) -> None:
        """Run continuous training service.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Training service starting")

        # Initialize all resources (models, chain manager, etc.)
        await self._initialize_resources()

        # Initialize monitoring AFTER heavy resources are loaded
        await self._initialize_monitoring()

        # Upload initial checkpoint and wait for miners
        await self._upload_initial_checkpoint(stop_event)

        # Wait 1 window for miners to download
        await self._wait_for_miners(stop_event)

        # Run continuous training loop
        await self._training_loop(stop_event)

        logger.info("Training service exiting (epochs completed: %d)", self.epoch_counter)

    async def _wait_for_miners(self, stop_event: Event) -> None:
        """Wait for miners to download initial checkpoint.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Waiting for miners to download initial checkpoint...")
        # Wait 0 window for miners to download
        # TODO: Later on when we use extra process/GPUs, some waiting should be added here.
        WAIT_WINDOW_LENGTH = 0.0  # Wait 0 window for miners to download

        # Get current block first before using it
        current_block = await self.subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        if self.monitor:
            self.monitor.set_block_context(current_block, current_window)
        target_block = current_block + WAIT_WINDOW_LENGTH
        logger.info(
            "Waiting %s window for miners (until block %s)", WAIT_WINDOW_LENGTH, target_block
        )

        while not stop_event.is_set():
            current_block = await self.subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
            current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
            if self.monitor:
                self.monitor.set_block_context(current_block, current_window)
            if current_block >= target_block:
                logger.info("Wait complete, starting continuous training")
                break
            await asyncio.sleep(INITIAL_CHECKPOINT_WAIT_INTERVAL_SECONDS)

        logger.info("Waiting for miners to download initial checkpoint complete")

    async def _initialize_resources(self) -> None:
        """Initialize all async and GPU resources."""
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

        # Enable gradient checkpointing on the raw model before accelerator wrapping (if configured)
        if self.config.use_gradient_checkpointing:
            if hasattr(self.train_model, "gradient_checkpointing_enable"):
                try:
                    self.train_model.gradient_checkpointing_enable()
                    logger.info(
                        "✅ Gradient checkpointing enabled for train_model (pre-accelerator)"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to enable gradient checkpointing: %s", exc)
            else:
                logger.debug("Model does not support gradient_checkpointing_enable()")
        else:
            logger.info(
                "Gradient checkpointing disabled (via GRAIL_TRAINER_USE_GRADIENT_CHECKPOINTING=0)"
            )
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
        self.subtensor = await create_subtensor(resilient=True)  # type: ignore[assignment]  # ResilientSubtensor compatible

        # Initialize chain manager
        logger.info("Initializing chain manager...")
        assert self.subtensor is not None
        metagraph = await self.subtensor.metagraph(NETUID)  # type: ignore[misc]  # bittensor async stub
        chain_config = SimpleNamespace(netuid=NETUID)
        self.chain_manager = GrailChainManager(
            chain_config,
            self.wallet,
            metagraph,
            self.subtensor,  # type: ignore[arg-type]  # ResilientSubtensor compatible
            self.credentials,
        )
        await self.chain_manager.initialize()
        logger.info("Chain manager initialized")

    async def _initialize_monitoring(self) -> None:
        """Initialize monitoring after heavy resources are loaded.

        This is called AFTER models and chain manager are initialized to avoid
        resource contention during WandB connection.

        Uses the shared `initialize_subprocess_monitoring` helper for consistent
        behavior across all subprocesses (training, upload worker, etc.).
        """

        async def get_block_context() -> tuple[int, int]:
            """Get current block and window for monitoring context."""
            current_block = await self.subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
            current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
            return current_block, current_window

        self.monitor = await initialize_subprocess_monitoring(
            self.monitor_config,
            process_name="training_process",
            test_connection=True,
            get_block_context=get_block_context,
        )

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
        """Create learning rate scheduler with warmup phase.

        Supports two scheduler types via GRAIL_LR_SCHEDULER_TYPE:
        - "constant": Warmup then constant LR (recommended for RL/GRPO)
        - "cosine": Warmup then cosine annealing decay (common for SFT)

        Returns:
            Configured scheduler instance
        """
        assert self.optimizer is not None, "Optimizer must be initialized before creating scheduler"
        warmup_steps = max(1, int(WARMUP_FRACTION * TOTAL_TRAINING_WINDOWS))

        # Warmup lambda: linear ramp from 0 to 1 over warmup_steps, then constant 1.0
        def lr_lambda_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda_warmup)

        if LR_SCHEDULER_TYPE == "cosine":
            # Warmup + Cosine Annealing
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=TOTAL_TRAINING_WINDOWS - warmup_steps,
                eta_min=SCHEDULER_ETA_MIN,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            logger.info(
                "LR scheduler: warmup (%d steps) + cosine (T_max=%d, eta_min=%.2e)",
                warmup_steps,
                TOTAL_TRAINING_WINDOWS - warmup_steps,
                SCHEDULER_ETA_MIN,
            )
        else:
            # Warmup + Constant LR (default, recommended for GRPO)
            scheduler = warmup_scheduler
            logger.info("LR scheduler: warmup (%d steps) + constant", warmup_steps)

        return scheduler

    async def _upload_initial_checkpoint(self, stop_event: Event) -> None:
        """Save initial checkpoint and queue for upload worker.

        Args:
            stop_event: Event to signal shutdown
        """
        current_block = await self.subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
        if self.monitor:
            self.monitor.set_block_context(current_block, current_window)

        logger.info("Saving initial checkpoint for window %s", current_window)

        # Prepare snapshot metadata
        snapshot_metadata = {
            "epoch": 0,
            "timestamp": time.time(),
            "status": "initial_upload",
            "training_config": {
                "lr": self.config.lr,
                "seed": current_window,
            },
            "parent_window": max(0, current_window - WINDOW_LENGTH),
        }

        # Save initial snapshot
        self.snapshot_manager.save_snapshot_atomic(
            self.train_model,
            self.tokenizer,
            snapshot_metadata,
        )

        # Queue snapshot for upload worker
        snapshot_path = self.snapshot_manager.get_latest_snapshot_path()
        if self._ipc is not None and snapshot_path:
            self._ipc.queue_snapshot(str(snapshot_path), snapshot_metadata, current_window)
            logger.info("Initial checkpoint queued for upload worker (window=%s)", current_window)
        else:
            logger.warning(
                "Could not queue initial checkpoint: ipc=%s, snapshot_path=%s",
                self._ipc is not None,
                snapshot_path is not None,
            )

    async def _training_loop(self, stop_event: Event) -> None:
        """Main continuous training loop.

        Args:
            stop_event: Event to signal shutdown
        """
        logger.info("Training loop starting (PID=%d)", multiprocessing.current_process().pid)

        # Prepare models with Accelerator
        accelerator = Accelerator(mixed_precision="no")
        train_model, ref_model, optimizer = self._prepare_models_with_accelerator(accelerator)

        logger.info("Models and optimizer prepared with Accelerator")

        # Start event loop monitoring in background (only in DEBUG mode)
        if logger.isEnabledFor(logging.DEBUG):
            from grail.logging_utils import monitor_event_loop_lag

            asyncio.create_task(monitor_event_loop_lag(interval=5.0, threshold=1.0))
            logger.debug("Event loop lag monitoring started")

        while not stop_event.is_set():
            try:
                # Update heartbeat via shared Value (primary) + filesystem (backup)
                self._update_heartbeat()

                # Check for pause request via IPC
                pause_requested = self._ipc.is_pause_requested() if self._ipc else False
                if pause_requested:
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

                # Load GRPO groups only if window changed (skip trusted miners query if not needed)
                target_data_window = current_window - WINDOW_LENGTH
                if target_data_window != self.last_loaded_window and target_data_window >= 0:
                    # Log epochs from previous window before switching (if any training occurred)
                    if self._epochs_this_window > 0 and self._current_training_window >= 0:
                        await self._log_window_epochs(
                            self._current_training_window, self._epochs_this_window
                        )
                        self._windows_completed += 1
                        logger.info(
                            "Window %d complete: %d epochs trained (total windows: %d)",
                            self._current_training_window,
                            self._epochs_this_window,
                            self._windows_completed,
                        )

                        # Check if we've reached TOTAL_TRAINING_WINDOWS - save checkpoint immediately
                        if (
                            self._windows_completed >= TOTAL_TRAINING_WINDOWS
                            and not self._full_checkpoint_saved
                        ):
                            logger.info(
                                "Reached TOTAL_TRAINING_WINDOWS (%d windows), saving full checkpoint",
                                TOTAL_TRAINING_WINDOWS,
                            )
                            await self._save_full_checkpoint(
                                train_model, accelerator, current_window
                            )

                    # Reset counter for new window
                    self._epochs_this_window = 0
                    self._current_training_window = target_data_window

                    # Get trusted miners only when we need to fetch new data
                    trusted_hotkeys = await self._get_trusted_miners(target_data_window)
                    if not trusted_hotkeys:
                        await asyncio.sleep(NO_MINERS_SLEEP_SECONDS)
                        continue

                    # Load GRPO groups (Replay Buffer)
                    await self._load_grpo_groups(current_window, trusted_hotkeys)

                # Check if replay buffer has data
                if self.replay_buffer is not None:
                    stats = self.replay_buffer.get_stats()
                    if stats["total_groups"] == 0:
                        logger.warning(
                            "No data available in replay buffer, sleeping %ds",
                            NO_DATA_SLEEP_SECONDS,
                        )
                        await asyncio.sleep(NO_DATA_SLEEP_SECONDS)
                        continue
                else:
                    # Legacy mode: no replay buffer (should not happen with default config)
                    logger.warning("Replay buffer disabled, cannot train without groups")
                    await asyncio.sleep(NO_DATA_SLEEP_SECONDS)
                    continue

                # Train one epoch
                metrics = await self._train_epoch(
                    train_model, ref_model, accelerator, current_window
                )

                # Save snapshot
                self._save_snapshot(train_model, accelerator, current_window, metrics)

                # Update heartbeat via shared Value (primary) + filesystem (backup)
                self._update_heartbeat()

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
                assert ref_model is not None
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
        stop_event: Event,
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
        logger.info("🔄 STATE: pause_requested - freeing GPU for evaluation")

        # Unwrap models to access raw PyTorch objects
        # Note: optimizer from accelerator.prepare() is already unwrapped
        unwrapped_train = accelerator.unwrap_model(train_model)
        unwrapped_ref = accelerator.unwrap_model(ref_model) if ref_model else None

        logger.info("🔄 STATE: models_moving_to_cpu - starting GPU→CPU transfer")
        start_cpu_transfer = time.time()

        # Move models to CPU in thread pool to avoid blocking event loop
        train_model_cpu = await asyncio.to_thread(unwrapped_train.cpu)
        ref_model_cpu = await asyncio.to_thread(unwrapped_ref.cpu) if unwrapped_ref else None

        # Move optimizer state to CPU (momentum buffers, variance estimates, etc.)
        # This is critical for AdamW which stores per-parameter state on GPU
        def move_optimizer_to_cpu(opt: torch.optim.Optimizer) -> None:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()

        await asyncio.to_thread(move_optimizer_to_cpu, optimizer)

        cpu_transfer_duration = time.time() - start_cpu_transfer
        logger.info(
            "✅ Models and optimizer moved to CPU in %.3fs",
            cpu_transfer_duration,
        )

        # Clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb, _ = torch.cuda.mem_get_info()
            logger.info("GPU memory freed: %.2f GB available", free_gb / (1024**3))

        # Save snapshot BEFORE confirming pause to avoid disk I/O contention
        # If evaluation starts while snapshot is being saved, vLLM model loading
        # competes with the ~15GB write, causing startup timeouts (120s+)
        logger.info("🔄 STATE: saving_snapshot - saving model before evaluation starts")
        self.snapshot_manager.save_snapshot_atomic(
            train_model_cpu,
            self.tokenizer,
            {
                "epoch": self.epoch_counter,
                "timestamp": time.time(),
                "status": "paused_for_evaluation",
            },
        )
        logger.info("🔄 STATE: pause_snapshot_saved - snapshot saved during pause")

        # Signal pause confirmed via IPC AFTER snapshot save completes
        # This ensures vLLM can load the model without disk I/O contention
        if self._ipc is not None:
            self._ipc.confirm_pause()
            logger.info(
                "🔄 STATE: pause_confirmed - signaled orchestrator via IPC (snapshot saved, GPU freed)"
            )
        else:
            logger.info("🔄 STATE: models_on_cpu_waiting - GPU freed (filesystem mode)")

        # Close subtensor connection before long idle period (prevents 10s timeout)
        if self.subtensor:
            try:
                if hasattr(self.subtensor, "_subtensor"):
                    # Unwrap ResilientSubtensor to get underlying subtensor
                    inner = object.__getattribute__(self.subtensor, "_subtensor")
                    if hasattr(inner, "close"):
                        await inner.close()
                elif hasattr(self.subtensor, "close"):
                    close_result = self.subtensor.close()
                    # Handle both sync and async close methods
                    if asyncio.iscoroutine(close_result):
                        await close_result  # type: ignore[misc]
                logger.info("Closed subtensor connection during pause to prevent idle timeout")
            except Exception as e:
                logger.warning("Failed to close subtensor during pause: %s", e)
            self.subtensor = None

        # Wait for pause flag to be cleared via IPC (primary) or filesystem (fallback)
        while not stop_event.is_set():
            # Check if resume signal received
            if self._ipc is not None:
                pause_still_active = self._ipc.is_pause_requested()
            elif hasattr(self.snapshot_manager, "check_pause_flag"):
                pause_still_active = self.snapshot_manager.check_pause_flag()  # type: ignore[attr-defined]
            else:
                # No IPC and no pause flag method - exit pause loop
                logger.debug("No IPC or pause flag method available, exiting pause")
                pause_still_active = False
            if not pause_still_active:
                break
            await asyncio.sleep(PAUSE_CHECK_INTERVAL_SECONDS)

        if stop_event.is_set():
            return train_model_cpu, ref_model_cpu, optimizer

        # Clear the confirmed event for next cycle
        if self._ipc is not None:
            self._ipc.clear_pause_confirmed()

        logger.info("🔄 STATE: resume_requested - pause signal cleared")

        # Recreate subtensor connection after pause
        logger.info("Recreating subtensor connection after pause...")
        self.subtensor = await create_subtensor(resilient=True)  # type: ignore[assignment]
        logger.info("Subtensor connection recreated successfully")

        logger.info("🔄 STATE: models_moving_to_gpu - starting CPU→GPU transfer")
        start_gpu_transfer = time.time()

        # Move models back to GPU using thread pool to avoid blocking event loop
        train_model_gpu = await asyncio.to_thread(train_model_cpu.cuda)
        ref_model_gpu = await asyncio.to_thread(ref_model_cpu.cuda) if ref_model_cpu else None

        # Move optimizer state back to GPU
        def move_optimizer_to_gpu(opt: torch.optim.Optimizer) -> None:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        await asyncio.to_thread(move_optimizer_to_gpu, optimizer)

        gpu_transfer_duration = time.time() - start_gpu_transfer
        logger.info(
            "✅ Models and optimizer moved to GPU in %.3fs",
            gpu_transfer_duration,
        )

        # Re-prepare with accelerator (also blocking, use thread pool)
        logger.info("🔄 STATE: accelerator_preparing - wrapping models with accelerator")
        start_prepare = time.time()

        if ref_model_gpu:
            train_model, ref_model, prepared_optimizer = await asyncio.to_thread(
                accelerator.prepare,
                train_model_gpu,
                ref_model_gpu,
                optimizer,
            )
            assert ref_model is not None
            ref_model.eval()
        else:
            train_model, prepared_optimizer = await asyncio.to_thread(
                accelerator.prepare,
                train_model_gpu,
                optimizer,
            )
            ref_model = None

        prepare_duration = time.time() - start_prepare
        logger.info("✅ Accelerator prepare complete in %.3fs", prepare_duration)
        logger.info("🔄 STATE: training_resumed - models ready, resuming training")

        return train_model, ref_model, prepared_optimizer

    async def _get_current_window(self) -> int:
        """Get current window number from subtensor.

        Returns:
            Current window number
        """
        try:
            current_block = await self.subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
            current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
            if self.monitor:
                self.monitor.set_block_context(current_block, current_window)
            return current_window
        except Exception as exc:
            logger.warning("Failed to get current block: %s", exc)
            return self.epoch_counter * WINDOW_LENGTH

    async def _log_window_epochs(self, window: int, epochs: int) -> None:
        """Log the number of training epochs completed in a window to wandb.

        Args:
            window: Window number that was trained
            epochs: Number of epochs completed in that window
        """
        if self.monitor is None:
            return

        try:
            await self.monitor.log_gauge(
                "training/prefilter/per_epochs_per_window",
                float(epochs),
                tags={"window_number": str(window)},
            )
        except Exception as exc:
            logger.debug("Failed to log epochs_per_window metric: %s", exc)

    async def _get_trusted_miners(self, target_window: int) -> set[str]:
        """Get trusted miner hotkeys, preferring validator R2 trust list.

        Resolution order:
        1. Test mode → only TRAINER_UID's hotkey
        2. Primary → validator-published trust list from R2
        3. Fallback → on-chain incentive (Yuma Consensus)

        Args:
            target_window: The data window the trainer is loading

        Returns:
            Set of trusted miner hotkeys (may be empty if all methods fail)
        """
        try:
            metagraph = await self.subtensor.metagraph(NETUID)  # type: ignore[misc]  # bittensor async stub

            # Test mode: only use TRAINER_UID
            if self.test_mode:
                from grail.shared.constants import TRAINER_UID

                if TRAINER_UID < len(metagraph.hotkeys):
                    trainer_hotkey = metagraph.hotkeys[TRAINER_UID]
                    logger.info(
                        "Test mode: Using TRAINER_UID %d hotkey %s for training data",
                        TRAINER_UID,
                        trainer_hotkey[:12],
                    )
                    return {trainer_hotkey}
                else:
                    logger.warning(
                        "Test mode: TRAINER_UID %d not found in metagraph (size: %d)",
                        TRAINER_UID,
                        len(metagraph.hotkeys),
                    )
                    return set()

            # Primary: try validator-published trust list from R2
            if self.chain_manager is not None:
                try:
                    trust_list = await get_trust_list_from_validator(
                        metagraph, self.chain_manager, target_window
                    )
                    if trust_list:
                        logger.info(
                            "Using validator trust list: %d miners for window %d",
                            len(trust_list),
                            target_window,
                        )
                        return trust_list
                except Exception as exc:
                    logger.warning("Trust list lookup failed, falling back to on-chain: %s", exc)

            # Fallback: on-chain incentive (Yuma Consensus)
            logger.info("Falling back to on-chain incentive for trusted miners")
            trusted_hotkeys = await get_trusted_miner_hotkeys(
                metagraph,
                self.config.min_trusted_miners,
                timeout=TRUST_COMPUTATION_TIMEOUT_SECONDS,
            )

            if not trusted_hotkeys:
                logger.warning("No trusted miners found")

            return trusted_hotkeys

        except Exception as exc:
            logger.error("Failed to get trusted miners: %s", exc)
            return set()

    async def _load_grpo_groups(self, current_window: int, trusted_hotkeys: set[str]) -> None:
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
            metagraph = await self.subtensor.metagraph(NETUID)  # type: ignore[misc]  # bittensor async stub
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
                    "Loaded %d GRPO groups for window %s",
                    len(new_groups),
                    target_data_window,
                )

                # Add groups to replay buffer if enabled
                if self.replay_buffer is not None:
                    self.replay_buffer.add_window(target_data_window, new_groups)
                    stats = self.replay_buffer.get_stats()
                    logger.info(
                        "Replay buffer updated: %d windows, %d total groups (%.1f MB) [%s → %s]",
                        stats["windows"],
                        stats["total_groups"],
                        stats["memory_mb"],
                        stats["oldest_window"],
                        stats["newest_window"],
                    )

                    # Log per-window allocation preview
                    if logger.isEnabledFor(logging.DEBUG):
                        max_groups = self.config.replay_buffer_max_groups_per_epoch
                        seed = current_window + self.epoch_counter
                        sample_preview = self.replay_buffer.sample_groups(max_groups, seed)
                        logger.debug(
                            "Replay buffer sample preview: would sample %d groups with seed=%d",
                            len(sample_preview),
                            seed,
                        )

                self.last_loaded_window = target_data_window
            else:
                logger.warning(
                    "No valid GRPO groups found for window %s",
                    target_data_window,
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
        """Train one epoch by sampling groups from replay buffer.

        Args:
            train_model: Training model
            ref_model: Reference model (or None)
            accelerator: Accelerator instance
            current_window: Current window number

        Returns:
            Training metrics dictionary
        """
        try:
            # Sample groups from replay buffer
            if self.replay_buffer is not None:
                seed = current_window + self.epoch_counter
                groups = self.replay_buffer.sample_groups(
                    self.config.replay_buffer_max_groups_per_epoch,
                    seed,
                )

                if not groups:
                    logger.warning("Replay buffer returned no groups for training")
                    return {}

                logger.info(
                    "Training epoch %d with %d groups sampled from replay buffer (seed=%d)",
                    self.epoch_counter + 1,
                    len(groups),
                    seed,
                )
            else:
                logger.error("Replay buffer not initialized, cannot train")
                return {}

            epoch_start = time.time()
            assert self.optimizer is not None, "Optimizer must be initialized before training"
            assert self.algorithm is not None, "Algorithm must be initialized before training"

            metrics = await self.algorithm.train_epoch(
                train_model,
                ref_model,
                self.tokenizer,
                groups,
                self.optimizer,
                accelerator,
                self.monitor,
                current_window,
                self.config,
            )

            # Update scheduler
            if self.scheduler is not None:
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
            self._epochs_this_window += 1
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
        """Save model snapshot atomically and notify upload worker.

        Args:
            train_model: Training model
            accelerator: Accelerator instance
            current_window: Current window number
            metrics: Training metrics
        """
        try:
            unwrapped_train = accelerator.unwrap_model(train_model)

            snapshot_metadata = {
                "epoch": self.epoch_counter,
                "timestamp": time.time(),
                "window": current_window,
                "parent_window": current_window - WINDOW_LENGTH,
                "metrics": metrics,
                "lr": self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
            }

            self.snapshot_manager.save_snapshot_atomic(
                unwrapped_train,
                self.tokenizer,
                snapshot_metadata,
            )

            # Notify upload worker via IPC queue (primary) or filesystem marker (fallback)
            snapshot_path = self.snapshot_manager.get_latest_snapshot_path()
            if self._ipc is not None and snapshot_path:
                self._ipc.queue_snapshot(str(snapshot_path), snapshot_metadata, current_window)
                logger.debug("Snapshot message queued for upload worker")

            logger.info("Snapshot saved for epoch %d", self.epoch_counter)

        except Exception as exc:
            logger.error("Failed to save snapshot: %s", exc)
            # Continue training even if snapshot fails

    async def _save_full_checkpoint(
        self,
        train_model: Any,
        accelerator: Accelerator,
        current_window: int,
    ) -> None:
        """Save complete training checkpoint including optimizer and scheduler state.

        This saves everything needed to fully resume training:
        - Model weights (safetensors format)
        - Tokenizer
        - Optimizer state (momentum buffers, variance estimates)
        - Scheduler state
        - Training configuration and hyperparameters
        - Epoch counter and training state

        Called when _windows_completed reaches TOTAL_TRAINING_WINDOWS.

        Args:
            train_model: Training model (wrapped by accelerator)
            accelerator: Accelerator instance
            current_window: Current window number
        """
        checkpoint_dir = self.snapshot_manager.cache_root / "checkpoints"
        checkpoint_path = checkpoint_dir / f"checkpoint_window_{self._windows_completed}"

        try:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            # Unwrap model from accelerator
            unwrapped_train = accelerator.unwrap_model(train_model)

            # Save model and tokenizer
            logger.info(
                "Saving full checkpoint to %s (windows=%d, TOTAL_TRAINING_WINDOWS=%d)",
                checkpoint_path,
                self._windows_completed,
                TOTAL_TRAINING_WINDOWS,
            )
            unwrapped_train.save_pretrained(
                str(checkpoint_path),
                safe_serialization=True,
            )
            self.tokenizer.save_pretrained(str(checkpoint_path))

            # Save optimizer state
            assert self.optimizer is not None
            optimizer_path = checkpoint_path / "optimizer.pt"
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.info("Optimizer state saved to %s", optimizer_path)

            # Save scheduler state
            if self.scheduler is not None:
                scheduler_path = checkpoint_path / "scheduler.pt"
                torch.save(self.scheduler.state_dict(), scheduler_path)
                logger.info("Scheduler state saved to %s", scheduler_path)

            # Save complete training state and hyperparameters
            training_state = {
                "epoch": self.epoch_counter,
                "windows_completed": self._windows_completed,
                "last_loaded_window": self.last_loaded_window,
                "current_window": current_window,
                "timestamp": time.time(),
                "total_training_windows": TOTAL_TRAINING_WINDOWS,
                # Optimizer hyperparameters
                "optimizer_config": {
                    "type": "AdamW",
                    "lr": self.config.lr,
                    "betas": list(OPTIMIZER_BETAS),
                    "weight_decay": OPTIMIZER_WEIGHT_DECAY,
                },
                # Scheduler hyperparameters
                "scheduler_config": {
                    "type": "SequentialLR(LambdaLR+CosineAnnealingLR)",
                    "total_training_windows": TOTAL_TRAINING_WINDOWS,
                    "warmup_fraction": WARMUP_FRACTION,
                    "eta_min": SCHEDULER_ETA_MIN,
                    "current_lr": self.scheduler.get_last_lr()[0]
                    if self.scheduler
                    else self.config.lr,
                },
                # Training config
                "training_config": {
                    "lr": self.config.lr,
                    "batch_size": self.config.batch_size,
                    "grad_accum_steps": self.config.grad_accum_steps,
                    "grad_clip": self.config.grad_clip,
                    "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
                    "replay_buffer_enabled": self.config.replay_buffer_enabled,
                    "replay_buffer_max_windows": self.config.replay_buffer_max_windows,
                    "replay_buffer_max_groups_per_epoch": self.config.replay_buffer_max_groups_per_epoch,
                    "replay_buffer_recent_fraction": self.config.replay_buffer_recent_fraction,
                    "replay_buffer_decay_factor": self.config.replay_buffer_decay_factor,
                },
            }

            state_path = checkpoint_path / "training_state.json"
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(training_state, f, indent=2)

            logger.info(
                "Full checkpoint saved successfully: windows=%d, path=%s",
                self._windows_completed,
                checkpoint_path,
            )
            self._full_checkpoint_saved = True

            # Upload model to HuggingFace in background (non-blocking)
            # Training continues while upload happens asynchronously
            self._start_background_huggingface_upload(checkpoint_path)

        except Exception as exc:
            logger.error("Failed to save full checkpoint: %s", exc)
            # Continue training even if checkpoint save fails

    def _start_background_huggingface_upload(self, checkpoint_path: Path) -> None:
        """Start HuggingFace upload as a background task (non-blocking).

        Args:
            checkpoint_path: Path to the saved checkpoint directory
        """
        from grail.shared.constants import HF_TOKEN, HF_USERNAME

        if not HF_TOKEN or not HF_USERNAME:
            logger.warning(
                "HuggingFace credentials not configured. "
                "Set HF_TOKEN and HF_USERNAME in .env to enable model upload."
            )
            return

        # Generate repository name based on model and training info
        # Include timestamp to avoid overwriting prior uploads
        model_name = getattr(self.train_model, "name_or_path", "grail-model")
        # Clean model name for repo (replace / with -)
        clean_name = model_name.replace("/", "-").lower()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        repo_name = f"grail-trained-{clean_name}-{timestamp}"

        commit_message = (
            f"GRAIL trained model - {TOTAL_TRAINING_WINDOWS} windows, epoch {self.epoch_counter}"
        )

        logger.info(
            "Starting background HuggingFace upload: %s/%s (non-blocking)",
            HF_USERNAME,
            repo_name,
        )

        # Create background task for upload (store reference to prevent GC)
        self._background_upload_task = asyncio.create_task(
            self._upload_to_huggingface_async(checkpoint_path, repo_name, commit_message)
        )

        # Add callback to log completion/failure
        def _on_upload_done(t: asyncio.Task[bool]) -> None:
            try:
                success = t.result()
                if success:
                    logger.info(
                        "✅ Background HuggingFace upload completed: https://huggingface.co/%s/%s",
                        HF_USERNAME,
                        repo_name,
                    )
                else:
                    logger.error("Background HuggingFace upload failed for %s", repo_name)
            except Exception as exc:
                logger.error("Background HuggingFace upload error: %s", exc)

        self._background_upload_task.add_done_callback(_on_upload_done)

    async def _upload_to_huggingface_async(
        self,
        checkpoint_path: Path,
        repo_name: str,
        commit_message: str,
    ) -> bool:
        """Upload trained model to HuggingFace Hub (async worker).

        Args:
            checkpoint_path: Path to the saved checkpoint directory
            repo_name: Repository name (without username)
            commit_message: Commit message for the upload

        Returns:
            True if upload succeeded, False otherwise
        """
        from grail.infrastructure.comms import upload_model_to_huggingface

        return await upload_model_to_huggingface(
            model_path=checkpoint_path,
            repo_name=repo_name,
            commit_message=commit_message,
            private=False,
        )


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


def _configure_child_process_logging(verbosity: int) -> None:
    """Configure enhanced logging for training process.

    Args:
        verbosity: CLI verbosity level (0=silent, 1=INFO, >=2=DEBUG)

    Uses structured logging with process/thread IDs, timing, and correlation IDs.
    """
    from grail.logging_utils import configure_process_logging

    # Map verbosity to log level (same as parent CLI)
    log_level = logging.DEBUG if verbosity >= 2 else logging.INFO
    configure_process_logging("training", level=log_level, include_function=True)


def _reconstruct_wallet(wallet_args: dict[str, str]) -> bt.wallet:
    """Reconstruct wallet from serialized arguments.

    Args:
        wallet_args: Wallet configuration (name, hotkey, path)

    Returns:
        Reconstructed wallet instance
    """
    return bt.wallet(**wallet_args)


# Removed: _initialize_monitoring() is now a method of TrainingService class


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
    ipc: IPCChannels,
    verbosity: int = 1,
    test_mode: bool = False,
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
        ipc: IPC channels for coordination with orchestrator
        verbosity: CLI verbosity level (0=silent, 1=INFO, >=2=DEBUG)
        test_mode: Test mode flag for training on TRAINER_UID data only
    """
    # Configure logging for child process
    _configure_child_process_logging(verbosity)

    logger.info("Training process starting (PID=%d)", multiprocessing.current_process().pid)
    sys.stdout.flush()

    # Reconstruct objects from primitives
    try:
        logger.info("Reconstructing wallet in child process...")
        wallet = _reconstruct_wallet(wallet_args)

    except Exception as exc:
        logger.exception("Failed to reconstruct wallet in training process: %s", exc)
        return

    # Run training service
    try:
        service = TrainingService(
            config=config,
            snapshot_manager=snapshot_manager,
            credentials=credentials,
            wallet=wallet,
            monitor_config=monitor_config,
            train_spec=train_spec,
            ref_spec=ref_spec,
            ipc=ipc,
            test_mode=test_mode,
        )

        asyncio.run(service.run(ipc.stop))

    except KeyboardInterrupt:
        logger.info("Training process interrupted")
    except Exception as exc:
        logger.exception("Training process crashed: %s", exc)
    finally:
        logger.info("Training process exiting")
