"""Trainer neuron orchestrating async training, upload, and evaluation."""

from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import bittensor as bt
import numpy as np
import torch

from grail.environments.factory import create_env_factory
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.checkpoint_consumer import default_checkpoint_cache_root
from grail.model.provider import get_model, get_tokenizer
from grail.model.train_loading import ModelLoadSpec
from grail.shared.constants import (
    NETUID,
    SNAPSHOT_POLL_INTERVAL_SECONDS,
    TRAINER_USE_FLASH_ATTENTION,
    TRAINING_HEARTBEAT_TIMEOUT_SECONDS,
    WINDOW_LENGTH,
)
from grail.trainer.checkpoint_publisher import CheckpointPublisher
from grail.trainer.config import EvalConfig, TrainingConfig
from grail.trainer.eval_planner import EvaluationPlanner
from grail.trainer.evaluator import EvaluatorService
from grail.trainer.inference_server import create_inference_server
from grail.trainer.ipc import IPCChannels, create_ipc_channels
from grail.trainer.snapshot_manager import SnapshotManager
from grail.trainer.training_process import run_training_process
from grail.trainer.upload_worker import run_upload_worker

from .base import BaseNeuron

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────

WATCHDOG_TIMEOUT_SECONDS = 60 * 15  # 15 minutes
WATCHDOG_GRACE_SECONDS = 10

ORCHESTRATION_SLEEP_SECONDS = 60
ORCHESTRATION_ERROR_SLEEP_SECONDS = 30

PAUSE_CONFIRMATION_POLL_SECONDS = 2
PAUSE_CONFIRMATION_TIMEOUT_SECONDS = 300  # 5 minutes max wait for training to pause
PROCESS_JOIN_TIMEOUT_SECONDS = 30
PROCESS_TERMINATE_TIMEOUT_SECONDS = 10


# ────────────────────────────────────────────────────────────────────────────────
# Data Classes
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainerContext:
    """Resources required to run the trainer neuron.

    Attributes:
        wallet: Bittensor wallet for authentication
        credentials: R2/S3 credentials for storage
        checkpoint_publisher: Publisher for uploading checkpoints
        monitor: Monitoring manager (W&B, etc.)
        train_spec: Specification for loading training model
        ref_spec: Specification for loading reference model
        verbosity: CLI verbosity level for child process logging
        chain_manager: Chain manager for miner data (initialized later)
    """

    wallet: bt.wallet
    credentials: Any
    checkpoint_publisher: CheckpointPublisher | None
    monitor: Any | None
    train_spec: ModelLoadSpec
    ref_spec: ModelLoadSpec
    verbosity: int = 1
    chain_manager: Any | None = None


# ────────────────────────────────────────────────────────────────────────────────
# Trainer Neuron
# ────────────────────────────────────────────────────────────────────────────────


class TrainerNeuron(BaseNeuron):
    """Orchestrates async training, upload, and evaluation processes.

    Architecture:
    - Main process: Orchestrates child processes, runs evaluation
    - Training process: Loads models to GPU, trains continuously
    - Upload worker: Uploads snapshots to R2/S3 asynchronously

    The main process never uses GPU to avoid CUDA fork issues.
    """

    def __init__(self, context: TrainerContext) -> None:
        """Initialize trainer neuron.

        Args:
            context: Resources required for training
        """
        super().__init__()
        self._context = context

        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        multiprocessing.set_start_method("spawn", force=True)
        logger.info("Set multiprocessing start method to 'spawn' for CUDA compatibility")

        # Configuration
        self._train_cfg = TrainingConfig()
        self._eval_cfg = EvalConfig()

        # Snapshot management (IPC coordination)
        cache_root = default_checkpoint_cache_root() / "async_trainer"
        self._snapshot_manager = SnapshotManager(cache_root)

        # Child processes
        self._training_process: multiprocessing.Process | None = None
        self._upload_process: multiprocessing.Process | None = None

        # Unified IPC channels for all inter-process communication
        self._ipc: IPCChannels = create_ipc_channels()

        # Evaluation state
        self._eval_in_progress: bool = False
        self._eval_last_run_window_number: int | None = None
        self._windows_since_last_eval: int = 0
        self._eval_checkpoint_dir: str | None = None
        self._last_seen_window: int | None = None  # Track window changes for eval interval

    async def run(self) -> None:
        """Run trainer orchestration loop.

        Lifecycle:
        1. Start watchdog for liveness monitoring
        2. Initialize chain manager
        3. Request pause (if evaluation enabled) to prevent training before first eval
        4. Spawn training and upload worker processes
        5. Enter orchestration loop (first iteration runs initial evaluation)
        6. Gracefully shutdown on exit
        """
        self.start_watchdog(
            timeout_seconds=WATCHDOG_TIMEOUT_SECONDS,
            grace_seconds=WATCHDOG_GRACE_SECONDS,
        )

        await self._initialize_chain_manager()

        logger.info("Main process will not use GPU (training process owns GPU)")

        try:
            # Request pause BEFORE starting training to ensure initial evaluation runs first.
            # Training will see this flag when it enters its loop and confirm pause.
            # The orchestration loop's first iteration will then run evaluation immediately
            # since _coordinate_evaluation() will see pause already confirmed.
            if self._eval_cfg.enabled:
                self._ipc.request_pause()
                logger.info("Pause requested before training start (initial evaluation pending)")

            logger.info("Starting async training and upload worker processes...")
            self._start_training_process()
            self._start_upload_worker()

            logger.info("Entering orchestration loop...")
            await self._orchestration_loop()

        except asyncio.CancelledError:
            logger.info("Trainer received CancelledError, shutting down...")
        except Exception:
            logger.exception("Trainer run() failed")
        finally:
            await self._shutdown_processes()

        logger.info("Trainer exited")

    # ────────────────────────────────────────────────────────────────────────────
    # Process Management
    # ────────────────────────────────────────────────────────────────────────────

    def _start_training_process(self) -> None:
        """Spawn training process for continuous training."""
        wallet_args = self._serialize_wallet()
        monitor_config = self._prepare_monitor_config(subprocess_label="training_process")

        self._training_process = multiprocessing.Process(
            target=run_training_process,
            args=(
                self._context.train_spec,
                self._context.ref_spec,
                self._train_cfg,
                self._snapshot_manager,
                self._context.credentials,
                wallet_args,
                monitor_config,
                self._ipc,
                self._context.verbosity,
            ),
        )
        self._training_process.start()
        logger.info("Training process started (PID=%d)", self._training_process.pid)

    def _start_upload_worker(self) -> None:
        """Spawn upload worker process for async uploads."""
        wallet_args = self._serialize_wallet()
        monitor_config = self._prepare_monitor_config(subprocess_label="upload_worker")

        self._upload_process = multiprocessing.Process(
            target=run_upload_worker,
            args=(
                self._snapshot_manager,
                self._context.credentials,
                wallet_args,
                monitor_config,
                self._ipc,
                SNAPSHOT_POLL_INTERVAL_SECONDS,
                self._context.verbosity,
            ),
        )
        self._upload_process.start()
        logger.info("Upload worker started (PID=%d)", self._upload_process.pid)

    async def _shutdown_processes(self) -> None:
        """Gracefully shutdown child processes."""
        logger.info("Shutting down child processes...")

        self._ipc.stop.set()

        await self._shutdown_process(self._training_process, "Training")
        await self._shutdown_process(self._upload_process, "Upload")

        logger.info("Child processes shut down")

    async def _shutdown_process(
        self,
        process: multiprocessing.Process | None,
        name: str,
    ) -> None:
        """Shutdown a single process gracefully.

        Args:
            process: Process to shutdown
            name: Process name for logging
        """
        if not process:
            return

        if not process.pid:
            logger.warning("%s process was never started (no PID)", name)
            return

        process.join(timeout=PROCESS_JOIN_TIMEOUT_SECONDS)

        if process.is_alive():
            logger.warning("%s process didn't exit, terminating...", name)
            process.terminate()
            process.join(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)

    # ────────────────────────────────────────────────────────────────────────────
    # Orchestration Loop
    # ────────────────────────────────────────────────────────────────────────────

    async def _orchestration_loop(self) -> None:
        """Main orchestration loop: monitor processes and coordinate evaluation."""
        while not self.stop_event.is_set():
            try:
                self.heartbeat()

                await self._check_process_health()

                if await self._should_wait_for_initialization():
                    await asyncio.sleep(ORCHESTRATION_SLEEP_SECONDS)
                    continue

                # Skip window checks during evaluation to avoid event loop starvation
                if self._eval_in_progress:
                    logger.debug("Evaluation in progress, skipping window check")
                    await asyncio.sleep(ORCHESTRATION_SLEEP_SECONDS)
                    continue

                current_window = await self._get_current_window()

                # Only count window changes, not loop iterations
                # A window is ~6 min (30 blocks × 12s), loop runs every 60s
                window_changed = (
                    self._last_seen_window is None or current_window != self._last_seen_window
                )
                if window_changed:
                    self._last_seen_window = current_window
                    self._windows_since_last_eval += 1
                    logger.debug(
                        "Window changed to %d, windows_since_last_eval=%d",
                        current_window,
                        self._windows_since_last_eval,
                    )

                if self._should_run_evaluation():
                    logger.info("Evaluation due, coordinating with training process...")
                    await self._coordinate_evaluation(current_window)
                    self._windows_since_last_eval = 0

                await asyncio.sleep(ORCHESTRATION_SLEEP_SECONDS)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Orchestration loop error")
                await asyncio.sleep(ORCHESTRATION_ERROR_SLEEP_SECONDS)

    async def _should_wait_for_initialization(self) -> bool:
        """Check if training process is still initializing.

        Returns:
            True if should wait, False if ready
        """
        heartbeat_age = self._get_heartbeat_age()
        if heartbeat_age == float("inf"):
            logger.debug(
                "Training process still initializing (no heartbeat yet), skipping evaluation check"
            )
            return True
        return False

    async def _get_current_window(self) -> int:
        """Get current window number from subtensor.

        Returns:
            Current window number
        """
        subtensor = await self.get_subtensor()
        current_block = await subtensor.get_current_block()
        current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

        if self._context.monitor:
            self._context.monitor.set_block_context(current_block, current_window)

        return current_window

    async def _check_process_health(self) -> None:
        """Monitor training and upload process health."""
        if self._training_process and not self._training_process.is_alive():
            logger.error("Training process died - system should restart")

        if self._upload_process and not self._upload_process.is_alive():
            logger.error("Upload process died - system should restart")

        # Get heartbeat age from IPC (primary) or filesystem (fallback)
        heartbeat_age = self._get_heartbeat_age()
        if heartbeat_age > TRAINING_HEARTBEAT_TIMEOUT_SECONDS:
            logger.error(
                "Training heartbeat stale (%.1fs > %ds)",
                heartbeat_age,
                TRAINING_HEARTBEAT_TIMEOUT_SECONDS,
            )

    def _get_heartbeat_age(self) -> float:
        """Get training heartbeat age from IPC channels.

        Returns:
            Age in seconds, or infinity if no heartbeat received yet
        """
        age = self._ipc.get_heartbeat_age()
        if age == float("inf"):
            # No heartbeat yet - fall back to filesystem for backward compat
            return self._snapshot_manager.get_training_heartbeat_age()
        return age

    def _should_run_evaluation(self) -> bool:
        """Check if evaluation should run.

        Returns:
            True if evaluation is due
        """
        if not self._eval_cfg.enabled:
            return False

        is_first_eval = self._eval_last_run_window_number is None
        interval_reached = self._windows_since_last_eval >= self._eval_cfg.window_interval

        return is_first_eval or interval_reached

    # ────────────────────────────────────────────────────────────────────────────
    # Evaluation Coordination
    # ────────────────────────────────────────────────────────────────────────────

    async def _coordinate_evaluation(self, current_window: int) -> None:
        """Pause training, run evaluation, resume training.

        Uses IPC events for coordination between main and training processes.

        Args:
            current_window: Current window number
        """
        logger.info("🔄 STATE: evaluation_pause_requested (window=%d)", current_window)

        # Signal pause via IPC event
        self._ipc.request_pause()
        logger.info("Set pause_requested event, waiting for training to pause...")

        if not await self._wait_for_training_pause():
            logger.error("Training process failed to pause in time, skipping evaluation")
            self._ipc.clear_pause()
            self.reset_subtensor()  # Reset connection after idle wait period
            return

        logger.info("🔄 STATE: evaluation_starting - training paused, GPU freed")
        try:
            await self._maybe_run_evaluation(current_window)
            logger.info("🔄 STATE: evaluation_complete - clearing pause flag")
        finally:
            # Signal resume via IPC event
            self._ipc.clear_pause()
            # Reset main process subtensor connection after idle period during evaluation
            self.reset_subtensor()
            logger.info("🔄 STATE: evaluation_resume_signaled - training will resume")

    async def _wait_for_training_pause(self) -> bool:
        """Wait for training to confirm pause via IPC event.

        Uses multiprocessing.Event with native timeout support for reliable
        coordination. Falls back to checking process liveness and stop_event.

        Returns:
            True if training paused successfully, False if timeout/failure
        """
        start_wait = time.time()
        timeout = PAUSE_CONFIRMATION_TIMEOUT_SECONDS

        # Use Event.wait() with timeout in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()

        while True:
            # Check for shutdown request
            if self.stop_event.is_set():
                logger.info("Stop event set during pause wait, aborting")
                return False

            # Check if training process is still alive
            if self._training_process and not self._training_process.is_alive():
                logger.error("Training process died while waiting for pause")
                return False

            # Check for timeout
            elapsed = time.time() - start_wait
            if elapsed > timeout:
                logger.error(
                    "Timeout waiting for training to pause (%.1fs > %ds)",
                    elapsed,
                    timeout,
                )
                return False

            # Wait for pause confirmation with short timeout (non-blocking check)
            # Using run_in_executor to avoid blocking the asyncio event loop
            confirmed = await loop.run_in_executor(
                None,
                self._ipc.wait_for_pause_confirmation,
                PAUSE_CONFIRMATION_POLL_SECONDS,  # Short wait, then re-check conditions
            )

            if confirmed:
                logger.info(
                    "Confirmed training paused via IPC event (%.1fs elapsed)",
                    time.time() - start_wait,
                )
                return True

            logger.info(
                "Waiting for training to pause... %.1fs / %ds",
                elapsed,
                timeout,
            )

    async def _maybe_run_evaluation(self, current_window: int) -> bool:
        """Run evaluation if due.

        Args:
            current_window: Current window number

        Returns:
            True if evaluation executed successfully
        """
        logger.debug("_maybe_run_evaluation: enabled=%s", self._eval_cfg.enabled)

        if not self._eval_cfg.enabled:
            return False

        window_number = current_window // WINDOW_LENGTH
        is_first_eval = self._eval_last_run_window_number is None

        should_start = (
            self._eval_in_progress
            or is_first_eval
            or (
                self._windows_since_last_eval >= self._eval_cfg.window_interval
                and self._eval_last_run_window_number != window_number
            )
        )

        if not should_start:
            logger.debug("Evaluation not due yet")
            return False

        self._eval_in_progress = True
        logger.info("📊 Starting evaluation cycle (window_number=%d)", window_number)

        plan, env_factory = self._create_evaluation_plan(window_number)
        eval_start = time.time()

        try:
            metrics = await self._run_evaluation(plan, window_number, env_factory)

            logger.info("🧪 Evaluation metrics: %s", metrics)
            await self._log_evaluation_metrics(metrics, time.time() - eval_start)

            self._eval_last_run_window_number = window_number
            self._windows_since_last_eval = 0

            self.heartbeat()
            logger.info("✅ Evaluation cycle complete")
            return True

        except Exception:
            logger.exception("Evaluation failed")
            await self._log_evaluation_failure(time.time() - eval_start)
            return False

        finally:
            self._eval_in_progress = False
            self._eval_checkpoint_dir = None

    async def _run_evaluation(
        self,
        plan: Any,
        window_number: int,
        env_factory: Any,
    ) -> dict[str, float]:
        """Run evaluation with appropriate backend.

        Args:
            plan: Evaluation plan
            window_number: Current window number
            env_factory: Factory function for environments

        Returns:
            Evaluation metrics
        """
        should_start_server = self._eval_cfg.start_server and self._eval_cfg.backend in (
            "sglang",
            "vllm",
        )

        if should_start_server:
            return await self._run_server_evaluation(plan, window_number, env_factory)
        else:
            return await self._run_direct_evaluation(plan, window_number, env_factory)

    def _create_evaluation_plan(self, window_number: int) -> tuple[Any, Any]:
        """Create evaluation plan and environment factory.

        Args:
            window_number: Current window number

        Returns:
            Tuple of (plan, env_factory)
        """
        from grail.environments import get_or_create_task_source

        source = get_or_create_task_source("math", split=self._eval_cfg.split)
        env_factory = create_env_factory("math", task_source=source, split=self._eval_cfg.split)

        if self._eval_cfg.subset_size is not None:
            plan = self._create_subset_evaluation_plan(source, window_number)
        else:
            plan = self._create_full_evaluation_plan(source, window_number)

        return plan, env_factory

    def _create_subset_evaluation_plan(self, source: Any, window_number: int) -> Any:
        """Create evaluation plan for fixed subset.

        Args:
            source: Task source
            window_number: Current window number

        Returns:
            Evaluation plan
        """

        def generate_fixed_subset(cycle_index: int, subset_size: int) -> list[str]:
            all_ids = source.iter_ids()
            n_samples = min(subset_size, len(all_ids))
            rng = np.random.RandomState(seed=self._eval_cfg.seed_base)
            indices = rng.choice(len(all_ids), size=n_samples, replace=False)
            return [all_ids[i] for i in sorted(indices)]

        planner = EvaluationPlanner(
            replicates=self._eval_cfg.replicates,
            seed_base=self._eval_cfg.seed_base,
            generate_ids=generate_fixed_subset,
        )
        plan = planner.for_cycle(
            cycle_index=window_number,
            subset_size=self._eval_cfg.subset_size,
        )

        logger.info(
            "Using fixed subset: %d tasks (%.1f%%)",
            len(plan.ids),
            100 * len(plan.ids) / source.size(),
        )

        return plan

    def _create_full_evaluation_plan(self, source: Any, window_number: int) -> Any:
        """Create evaluation plan for full dataset.

        Args:
            source: Task source
            window_number: Current window number

        Returns:
            Evaluation plan
        """
        planner = EvaluationPlanner(
            replicates=self._eval_cfg.replicates,
            seed_base=self._eval_cfg.seed_base,
            enumerate_ids=source.iter_ids,
        )
        return planner.for_cycle(cycle_index=window_number)

    def _load_evaluation_resources(
        self,
        for_hf_backend: bool = False,
    ) -> tuple[str, Any, Any | None]:
        """Load tokenizer and optionally model from snapshot for evaluation.

        Args:
            for_hf_backend: If True, loads model to GPU for HF backend

        Returns:
            Tuple of (snapshot_path, tokenizer, model_or_none)

        Raises:
            RuntimeError: If snapshot not available
        """
        snapshot_path = self._snapshot_manager.get_latest_snapshot_path()
        if not snapshot_path:
            raise RuntimeError("No snapshot available for evaluation")

        logger.info("Loading tokenizer from snapshot: %s", snapshot_path)
        tokenizer = get_tokenizer(str(snapshot_path))

        model = None
        if for_hf_backend:
            logger.info("Loading model to GPU for HF backend evaluation...")
            model = get_model(
                str(snapshot_path),
                device="cuda",
                eval_mode=True,
                use_flash_attention=TRAINER_USE_FLASH_ATTENTION,
            )

        return str(snapshot_path), tokenizer, model

    async def _run_server_evaluation(
        self,
        plan: Any,
        window_number: int,
        env_factory: Any,
    ) -> dict[str, float]:
        """Run evaluation with managed vLLM/SGLang server.

        Args:
            plan: Evaluation plan
            window_number: Current window number
            env_factory: Factory function for environments

        Returns:
            Evaluation metrics
        """
        snapshot_path, tokenizer, _ = self._load_evaluation_resources(for_hf_backend=False)
        self._eval_checkpoint_dir = snapshot_path

        self._log_gpu_memory("before server")

        chat_template_path = self._get_chat_template_path(snapshot_path)

        server_manager = create_inference_server(
            backend=self._eval_cfg.backend,
            model_path=snapshot_path,
            eval_config=self._eval_cfg,
            model_name_override="async_trainer_snapshot",
            chat_template_path=chat_template_path,
        )

        async with server_manager as server:
            self.heartbeat()
            logger.info("Starting server process...")
            await server.start_server()
            logger.info("Server started at %s", server.base_url)
            self.heartbeat()

            evaluator = EvaluatorService(
                model=None,
                tokenizer=tokenizer,
                env_factory=env_factory,
                config=self._eval_cfg,
                monitor=self._context.monitor,
                device="cuda",
                server_base_url=server.base_url,
                server_model_name=server.model_name,
            )

            try:
                metrics = await self._run_evaluation_cycle(
                    plan=plan,
                    window_number=window_number,
                    env_factory=env_factory,
                    evaluator=evaluator,
                )
            finally:
                # Explicitly shutdown evaluator before server context exits
                # to ensure all resources are released before vLLM process is killed
                evaluator.shutdown()
                del tokenizer
                gc.collect()

        logger.info("Server shutdown complete")
        return metrics

    async def _run_direct_evaluation(
        self,
        plan: Any,
        window_number: int,
        env_factory: Any,
    ) -> dict[str, float]:
        """Run evaluation with HF backend or external server.

        Args:
            plan: Evaluation plan
            window_number: Current window number
            env_factory: Factory function for environments

        Returns:
            Evaluation metrics
        """
        backend_name = (self._eval_cfg.backend or "hf").lower()

        server_base_url = None
        if backend_name in ("vllm", "sglang"):
            server_base_url = f"http://{self._eval_cfg.server_host}:{self._eval_cfg.server_port}"
            logger.info("Using external server at %s", server_base_url)

        need_model = backend_name == "hf" or not server_base_url
        _, tokenizer, model = self._load_evaluation_resources(for_hf_backend=need_model)

        evaluator = EvaluatorService(
            model=model,
            tokenizer=tokenizer,
            env_factory=env_factory,
            config=self._eval_cfg,
            monitor=self._context.monitor,
            device="cuda",
            server_base_url=server_base_url,
            server_model_name=None,
        )

        try:
            return await self._run_evaluation_cycle(
                plan=plan,
                window_number=window_number,
                env_factory=env_factory,
                evaluator=evaluator,
            )
        finally:
            self._cleanup_evaluation_resources(evaluator, tokenizer, model)

    async def _run_evaluation_cycle(
        self,
        *,
        plan: Any,
        window_number: int,
        env_factory: Any,
        evaluator: EvaluatorService,
    ) -> dict[str, float]:
        """Run evaluation cycle with given plan and evaluator.

        Args:
            plan: Evaluation plan with task IDs and seeds
            window_number: Current window number for logging
            env_factory: Factory function to create evaluation environments
            evaluator: Pre-configured evaluator instance

        Returns:
            Dictionary of evaluation metrics
        """
        is_startup_eval = self._eval_last_run_window_number is None
        eval_reason = (
            "startup" if is_startup_eval else f"after {self._windows_since_last_eval} windows"
        )

        logger.info(
            "🧪 Starting evaluation: window=%s tasks=%s replicates=%s split=%s backend=%s (%s)",
            window_number,
            len(plan.ids),
            plan.replicates,
            self._eval_cfg.split,
            self._eval_cfg.backend,
            eval_reason,
        )

        return await evaluator.run_cycle(
            plan, start_offset=0, heartbeat=self.heartbeat, window_number=window_number
        )

    # ────────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ────────────────────────────────────────────────────────────────────────────

    def _serialize_wallet(self) -> dict[str, str]:
        """Serialize wallet for pickling to child process.

        Returns:
            Dictionary of wallet arguments
        """
        return {
            "name": self._context.wallet.name,
            "hotkey": self._context.wallet.hotkey_str,
            "path": self._context.wallet.path,
        }

    def _prepare_monitor_config(self, *, subprocess_label: str) -> dict[str, Any]:
        """Prepare monitoring config for child process.

        Returns:
            Dictionary of monitoring configuration
        """
        if not self._context.monitor:
            return {}

        # Copy from backend.config (not manager._config) to get run_name and updated settings
        # The backend.config is updated by start_run() with run_name, while manager._config
        # contains only the initial config from CLI initialization.
        if hasattr(self._context.monitor, "backend") and hasattr(
            self._context.monitor.backend, "config"
        ):
            monitor_config = self._context.monitor.backend.config.copy()
            # Add backend_type from backend class name (needed by subprocess)
            backend_class_name = self._context.monitor.backend.__class__.__name__
            if "WandB" in backend_class_name:
                monitor_config["backend_type"] = "wandb"
            elif "Null" in backend_class_name:
                monitor_config["backend_type"] = "null"

            # If using shared mode, subprocess is a worker (not primary)
            if monitor_config.get("wandb_shared_mode"):
                monitor_config["wandb_x_primary"] = False
                monitor_config["wandb_x_label"] = subprocess_label
                logger.debug(
                    "Subprocess %s will use WandB shared mode as worker",
                    subprocess_label,
                )
        else:
            # Fallback to manager config if backend doesn't have config
            monitor_config = self._context.monitor._config.copy()

        if hasattr(self._context.monitor, "backend") and hasattr(
            self._context.monitor.backend, "run"
        ):
            wandb_run = self._context.monitor.backend.run
            if wandb_run and hasattr(wandb_run, "id"):
                monitor_config["run_id"] = wandb_run.id
                logger.info(
                    "Passing W&B run ID %s to %s for multi-process logging",
                    wandb_run.id,
                    subprocess_label,
                )

        # Debug log to verify config contents
        logger.debug(
            "Monitor config for subprocess: backend_type=%s run_name=%s run_id=%s entity=%s project=%s",
            monitor_config.get("backend_type"),
            monitor_config.get("run_name"),
            monitor_config.get("run_id"),
            monitor_config.get("entity"),
            monitor_config.get("project"),
        )
        logger.debug("Full monitor config keys being passed: %s", list(monitor_config.keys()))

        # Warn if critical parameters are missing
        if not monitor_config.get("entity"):
            logger.warning("⚠️  entity not in monitor_config passed to subprocess!")
        if not monitor_config.get("project"):
            logger.warning("⚠️  project not in monitor_config passed to subprocess!")

        return monitor_config

    def _log_gpu_memory(self, context: str) -> None:
        """Log GPU memory usage.

        Args:
            context: Context string for logging
        """
        if torch.cuda.is_available():
            free_gb, total_gb = torch.cuda.mem_get_info()
            logger.info(
                "GPU memory %s: %.2f GB free / %.2f GB total",
                context,
                free_gb / (1024**3),
                total_gb / (1024**3),
            )

    def _get_chat_template_path(self, snapshot_path: str) -> str | None:
        """Get chat template path from snapshot.

        Args:
            snapshot_path: Path to snapshot directory

        Returns:
            Path to chat template or None if not found
        """
        chat_template_path = os.path.join(snapshot_path, "chat_template.jinja")
        if not os.path.isfile(chat_template_path):
            logger.warning("chat_template.jinja not found, server may use default")
            return None
        return chat_template_path

    def _cleanup_evaluation_resources(
        self,
        evaluator: EvaluatorService,
        tokenizer: Any,
        model: Any | None,
    ) -> None:
        """Cleanup evaluation resources and free GPU memory.

        Args:
            evaluator: Evaluator service to shutdown
            tokenizer: Tokenizer to delete
            model: Model to delete (or None)
        """
        logger.info("Cleaning up evaluator and freeing GPU memory...")
        evaluator.shutdown()
        del tokenizer
        if model:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _log_evaluation_metrics(
        self,
        metrics: dict[str, float],
        duration: float,
    ) -> None:
        """Log evaluation metrics to monitoring system.

        Args:
            metrics: Evaluation metrics
            duration: Evaluation duration in seconds
        """
        if not self._context.monitor:
            return

        await self._context.monitor.log_counter("eval/cycle_completed")
        for key, val in metrics.items():
            await self._context.monitor.log_gauge(f"eval/{key}", float(val))

        logger.info("🧪 Total evaluation time: %.2fs (setup + run + cleanup)", duration)
        await self._context.monitor.log_gauge("profiling/eval_total_time", duration)

    async def _log_evaluation_failure(self, duration: float) -> None:
        """Log evaluation failure metrics.

        Args:
            duration: Time spent before failure in seconds
        """
        logger.info("🧪 Evaluation failed after %.2fs", duration)
        if self._context.monitor:
            await self._context.monitor.log_gauge("profiling/eval_total_time_failed", duration)

    async def _initialize_chain_manager(self) -> None:
        """Initialize chain manager for miner data fetching."""
        try:
            subtensor = await self.get_subtensor()
            metagraph = await subtensor.metagraph(NETUID)

            config = SimpleNamespace(netuid=NETUID)
            chain_manager = GrailChainManager(
                config,
                self._context.wallet,
                metagraph,
                subtensor,
                self._context.credentials,
            )

            await chain_manager.initialize()
            self._context.chain_manager = chain_manager
            logger.info("Initialized chain manager for trainer lifetime")

            self.register_shutdown_callback(self._cleanup_chain_manager)

        except Exception as exc:
            logger.warning(
                "Failed to initialize chain manager: %s; will continue with default credentials",
                exc,
            )
            self._context.chain_manager = None

    def _cleanup_chain_manager(self) -> None:
        """Clean up chain manager on shutdown."""
        if self._context.chain_manager:
            try:
                self._context.chain_manager.stop()
                logger.info("Stopped chain manager")
            except Exception as exc:
                logger.warning("Error stopping chain manager: %s", exc)
