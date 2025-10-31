"""Trainer neuron orchestrating window selection and delegating training."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import bittensor as bt
import numpy as np
import torch

from grail.environments.gsm8k_env import GSM8KEnv
from grail.environments.providers import GSM8KTaskSource
from grail.infrastructure.chain import GrailChainManager
from grail.shared.constants import NETUID, READY_MARKER_UPLOAD_BLOCKS, WINDOW_LENGTH
from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)
from grail.trainer.checkpointing import finalize_checkpoint_ready
from grail.trainer.config import EvalConfig
from grail.trainer.eval_planner import EvaluationPlanner
from grail.trainer.evaluator import EvaluatorService
from grail.trainer.inference_server import create_inference_server
from grail.trainer.service import TrainerService

from .base import BaseNeuron

logger = logging.getLogger(__name__)


@dataclass
class TrainerContext:
    """Resources required to run the trainer neuron."""

    wallet: bt.wallet
    credentials: Any
    checkpoint_manager: Any | None
    monitor: Any | None
    train_model: Any
    ref_model: Any
    tokenizer: Any
    chain_manager: Any | None = None
    # Model source paths for reloading after evaluation
    train_model_path: str | None = None
    ref_model_path: str | None = None


class TrainerNeuron(BaseNeuron):
    """Runs training cycles by delegating to the TrainerService."""

    def __init__(self, context: TrainerContext) -> None:
        super().__init__()
        self._context = context
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._window_wait_tracker = WindowWaitTracker(log_interval_secs=120)
        self._wait_start_time: float | None = None
        self._last_wait_log: float = 0.0
        # Evaluation state
        self._eval_cfg = EvalConfig()
        self._eval_in_progress: bool = False
        self._eval_last_run_window_number: int | None = None
        self._windows_since_last_eval: int = 0  # Counter for windows processed since last eval

    async def run(self) -> None:
        # Start the built-in watchdog (15 minute timeout)
        self.start_watchdog(timeout_seconds=60 * 15, grace_seconds=10)

        # Initialize chain manager once for the lifetime of the trainer
        await self._initialize_chain_manager()

        # Initialize optimizer and scheduler once for the lifetime of the trainer
        self._initialize_training_parameters()

        last_processed_window = -1

        while not self.stop_event.is_set():
            try:
                # Update heartbeat from BaseNeuron
                self.heartbeat()

                # Use shared subtensor from base class
                subtensor = await self.get_subtensor()

                current_block = await subtensor.get_current_block()
                current_window = self.calculate_window(current_block)
                target_window = current_window - WINDOW_LENGTH

                logger.debug(
                    "Loop iteration: current_block=%d current_window=%d target_window=%d last_processed=%d",
                    current_block,
                    current_window,
                    target_window,
                    last_processed_window,
                )

                if target_window <= last_processed_window or target_window < 0:
                    logger.debug(
                        "Window not ready: target_window=%d <= last_processed=%d",
                        target_window,
                        last_processed_window,
                    )
                    await self._handle_wait_for_window(
                        target_window, current_block, last_processed_window
                    )
                    await asyncio.sleep(10)
                    continue

                # Window is available - reset wait tracker for next time
                self._window_wait_tracker.reset()

                # Periodic evaluation at startup and every configured interval
                logger.debug(
                    "Evaluation check: windows_since_last_eval=%d, interval=%d",
                    self._windows_since_last_eval,
                    self._eval_cfg.window_interval,
                )
                logger.debug(
                    "About to call _maybe_run_evaluation with current_window=%d",
                    current_window,
                )
                eval_result = await self._maybe_run_evaluation(current_window)
                logger.debug("_maybe_run_evaluation returned: %s", eval_result)

                if eval_result:
                    # Skip training when evaluation runs (may span multiple windows)
                    last_processed_window = target_window
                    logger.info(
                        "Evaluation executed, updated last_processed_window=%d, about to continue",
                        last_processed_window,
                    )
                    continue

                logger.info("🎓 Training window %s", target_window)
                # Train on target window (past window), not current window
                success = await self._train_window(target_window)

                if success:
                    logger.info("✅ Trained window %s", target_window)
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training/success")
                else:
                    logger.warning("⚠️ Training issue (w=%s)", target_window)
                    logger.warning("Retrying next window")
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training/failed")

                # Finalize the checkpoint if we are still in the current window
                # If not, we never finalize the checkpoint and the checkpoint is
                # going to be cleaned up later on.
                current_block = await subtensor.get_current_block()
                current_block = current_block + READY_MARKER_UPLOAD_BLOCKS
                try:
                    finalized = await finalize_checkpoint_ready(
                        current_block, current_window, self._context.credentials
                    )
                    if finalized:
                        logger.info("✅ Finalized READY markers for checkpoint(s): %s", finalized)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to finalize checkpoint READY markers: %s", exc)

                # Mark window as processed regardless of outcome
                last_processed_window = target_window
                self._wait_start_time = None
                self._last_wait_log = 0.0

                # Increment window counter for evaluation scheduling
                self._windows_since_last_eval += 1
                logger.debug(
                    "Training cycle complete: window=%d, windows_since_eval=%d",
                    target_window,
                    self._windows_since_last_eval,
                )

            except asyncio.CancelledError:  # pragma: no cover - coop shutdown
                logger.info("Trainer loop received CancelledError, breaking")
                break
            except Exception:
                logger.exception("Trainer loop error", exc_info=True)
                # Force reconnect on next iteration
                self.reset_subtensor()
                await asyncio.sleep(30)

        logger.info("Trainer loop exited (stop_event=%s)", self.stop_event.is_set())

    async def _maybe_run_evaluation(self, current_window: int) -> bool:
        """Run evaluation if due; return True if evaluation executed."""
        logger.debug("_maybe_run_evaluation: enabled=%s", self._eval_cfg.enabled)

        if not self._eval_cfg.enabled:
            logger.debug("Evaluation disabled, returning False")
            return False

        window_number = current_window // WINDOW_LENGTH

        # Use counter-based approach: evaluate on startup and every window_interval windows
        # This ensures consistent intervals regardless of startup window_number
        is_first_eval = self._eval_last_run_window_number is None

        should_start = (
            self._eval_in_progress
            or is_first_eval  # Always evaluate on startup
            or (
                self._windows_since_last_eval >= self._eval_cfg.window_interval
                and self._eval_last_run_window_number != window_number
            )
        )

        logger.debug(
            "_maybe_run_evaluation: is_first_eval=%s should_start=%s eval_in_progress=%s",
            is_first_eval,
            should_start,
            self._eval_in_progress,
        )

        if not should_start:
            logger.debug("Evaluation not due yet, returning False")
            return False

        # Mark progress
        self._eval_in_progress = True
        logger.info("📊 Starting evaluation cycle (window_number=%d)", window_number)

        # Build dataset-backed evaluation (GSM8K by default)
        source = GSM8KTaskSource(split=self._eval_cfg.split)

        def env_factory() -> GSM8KEnv:
            return GSM8KEnv(task_source=source)

        # Choose between full dataset or fixed subset evaluation
        if self._eval_cfg.subset_size is not None:
            # Fixed random subset: deterministic sampling for consistent cross-cycle comparison
            def generate_fixed_subset(cycle_index: int, subset_size: int) -> list[str]:
                """Generate same subset every cycle using seed_base for reproducibility.

                Args:
                    cycle_index: Current evaluation cycle (ignored for consistency)
                    subset_size: Number of tasks to sample

                Returns:
                    Deterministic subset of task IDs
                """
                all_ids = source.iter_ids()
                total = len(all_ids)
                n_samples = min(subset_size, total)

                # Use seed_base for reproducibility; ignore cycle_index for consistency
                rng = np.random.RandomState(seed=self._eval_cfg.seed_base)
                indices = rng.choice(total, size=n_samples, replace=False)

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
                "Using fixed random subset: %d tasks (%.1f%% of %d total)",
                len(plan.ids),
                100 * len(plan.ids) / source.size(),
                source.size(),
            )
        else:
            # Full dataset evaluation
            planner = EvaluationPlanner(
                replicates=self._eval_cfg.replicates,
                seed_base=self._eval_cfg.seed_base,
                enumerate_ids=source.iter_ids,
            )
            plan = planner.for_cycle(cycle_index=window_number)

        # Track total evaluation time (including setup/cleanup)
        import time as _time

        eval_start = _time.time()

        # Determine if we need to start a server
        should_start_server = self._eval_cfg.sglang_start_server and self._eval_cfg.backend in (
            "sglang",
            "vllm",
        )
        logger.info(
            "Evaluation config: backend=%s should_start_server=%s",
            self._eval_cfg.backend,
            should_start_server,
        )

        try:
            # Use context manager for clean server lifecycle management
            if should_start_server:
                logger.info("Starting inference server for evaluation...")
                # Create server manager (will save checkpoint + free its model ref in __aenter__)
                server_manager = create_inference_server(
                    backend=self._eval_cfg.backend,
                    model=self._context.train_model,
                    tokenizer=self._context.tokenizer,
                    eval_config=self._eval_cfg,
                )

                async with server_manager as server:
                    logger.info("Server context entered, freeing training VRAM...")
                    # At this point: checkpoint is saved, server_manager._model is freed
                    # But trainer models are still loaded - free them now
                    self._free_training_vram_for_eval()

                    # Update heartbeat before expensive server startup
                    logger.debug("Heartbeat before server startup")
                    self.heartbeat()

                    # Now start the server with maximum available GPU memory
                    logger.info("Starting server process...")
                    await server.start_server()
                    logger.info("Server started successfully at %s", server.base_url)

                    # Update heartbeat after server is ready
                    logger.debug("Heartbeat after server startup")
                    self.heartbeat()

                    # Server is running with clean GPU memory; create evaluator
                    logger.info("Running evaluation cycle with server backend...")
                    metrics = await self._run_evaluation_cycle(
                        plan=plan,
                        window_number=window_number,
                        env_factory=env_factory,
                        server_base_url=server.base_url,
                        server_model_name=server.model_name,
                    )
                # Context manager handles cleanup automatically
                logger.info("Server context exited, subprocess terminated and GPU memory freed")

                # Verify GPU memory is available before reload
                if torch.cuda.is_available():
                    try:
                        free_gb, total_gb = torch.cuda.mem_get_info()
                        logger.info(
                            "GPU memory after server shutdown: %.2f GB free / %.2f GB total",
                            free_gb / (1024**3),
                            total_gb / (1024**3),
                        )
                    except Exception:
                        pass

                # Reload training artifacts after server shutdown
                logger.info("About to reload training models after server cleanup...")
                self._reload_training_models()
                logger.info("Training models reloaded successfully")
            else:
                # Direct evaluation without server (HF backend or server already running)
                logger.info("Running evaluation cycle with HF backend...")
                metrics = await self._run_evaluation_cycle(
                    plan=plan,
                    window_number=window_number,
                    env_factory=env_factory,
                    server_base_url=None,
                    server_model_name=None,
                )

            # Log metrics
            logger.info("🧪 Evaluation metrics: %s", metrics)

            if self._context.monitor:
                await self._context.monitor.log_counter("eval/cycle_completed")
                for key, val in metrics.items():
                    await self._context.monitor.log_gauge(f"eval/{key}", float(val))

            self._eval_last_run_window_number = window_number
            self._windows_since_last_eval = 0  # Reset counter after successful evaluation

            # Log total evaluation time
            eval_total = _time.time() - eval_start
            logger.info(
                "🧪 Total evaluation time: %.2fs (setup + run + cleanup)",
                eval_total,
            )
            if self._context.monitor:
                await self._context.monitor.log_gauge("profiling/eval_total_time", eval_total)

            # Update heartbeat after evaluation completes
            logger.debug("Heartbeat after evaluation completes")
            self.heartbeat()

            logger.info("✅ Evaluation cycle complete, returning True")
            return True

        except Exception:
            logger.exception("Evaluation failed", exc_info=True)

            # Log time even on failure
            eval_total = _time.time() - eval_start
            logger.info("🧪 Evaluation failed after %.2fs", eval_total)
            if self._context.monitor:
                await self._context.monitor.log_gauge(
                    "profiling/eval_total_time_failed", eval_total
                )

            logger.warning("Evaluation failed, returning False")
            return False
        finally:
            logger.debug("_maybe_run_evaluation finally block: setting eval_in_progress=False")
            self._eval_in_progress = False

    # -------------- Eval orchestration helpers --------------
    def _free_training_vram_for_eval(self) -> None:
        """Drop references to training and ref models and clear CUDA cache.

        This must be called AFTER the server manager has saved the checkpoint
        but BEFORE creating the evaluator for server-backed generation.
        """
        try:
            self._context.train_model = None  # type: ignore[assignment]
            self._context.ref_model = None  # type: ignore[assignment]
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logger.debug("Freed training VRAM for server-backed evaluation")
        except Exception as exc:
            logger.debug("Issue freeing training VRAM: %s", exc)

    async def _run_evaluation_cycle(
        self,
        *,
        plan: Any,
        window_number: int,
        env_factory: Any,
        server_base_url: str | None,
        server_model_name: str | None,
    ) -> dict[str, float]:
        """Run evaluation cycle with given plan and optional server configuration.

        Args:
            plan: Evaluation plan with task IDs and seeds
            window_number: Current window number for logging
            env_factory: Factory function to create evaluation environments
            server_base_url: Optional server URL (for vLLM/SGLang)
            server_model_name: Optional model name for server API calls

        Returns:
            Dictionary of evaluation metrics
        """
        evaluator: EvaluatorService | None = None

        try:
            # Create evaluator with server configuration
            evaluator = EvaluatorService(
                model=self._context.train_model,
                tokenizer=self._context.tokenizer,
                env_factory=env_factory,
                config=self._eval_cfg,
                monitor=self._context.monitor,
                device="cuda",
                server_base_url=server_base_url,
                server_model_name=server_model_name,
            )

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

            metrics = await evaluator.run_cycle(plan, start_offset=0, heartbeat=self.heartbeat)
            return metrics

        finally:
            # Always cleanup evaluator to free GPU memory
            if evaluator is not None:
                logger.info("🧹 Shutting down evaluator...")
                evaluator.shutdown()
                logger.info("🧹 Evaluator shutdown complete")

    def _reload_training_models(self) -> None:
        """Reload training and reference models after evaluation server shutdown."""
        logger.info(
            "Reloading training models: train_model=%s ref_model=%s",
            self._context.train_model is not None,
            self._context.ref_model is not None,
        )

        if self._context.train_model is not None and self._context.ref_model is not None:
            logger.debug("Training models still loaded, skipping reload")
            return

        # Log GPU memory state before reload
        if torch.cuda.is_available():
            try:
                free_gb, total_gb = torch.cuda.mem_get_info()
                logger.info(
                    "Reloading training models: GPU %.2f GB free / %.2f GB total",
                    free_gb / (1024**3),
                    total_gb / (1024**3),
                )
            except Exception:
                pass

        try:
            from grail.model.provider import get_model, get_tokenizer

            # Reload training model from original path
            if self._context.train_model is None and self._context.train_model_path:
                logger.info("Reloading training model from: %s", self._context.train_model_path)
                self._context.train_model = get_model(
                    self._context.train_model_path, eval_mode=False
                )
                logger.info("✅ Reloaded training model: %s", self._context.train_model_path)
            else:
                logger.debug(
                    "Skipping training model reload: model_path=%s",
                    self._context.train_model_path,
                )

            # Reload reference model from original path
            if self._context.ref_model is None and self._context.ref_model_path:
                logger.info("Reloading reference model from: %s", self._context.ref_model_path)
                self._context.ref_model = get_model(self._context.ref_model_path, eval_mode=True)
                logger.info("✅ Reloaded reference model: %s", self._context.ref_model_path)
            else:
                logger.debug(
                    "Skipping reference model reload: model_path=%s",
                    self._context.ref_model_path,
                )

            # Reload tokenizer from training model path
            if self._context.train_model_path:
                logger.info("Reloading tokenizer from: %s", self._context.train_model_path)
                self._context.tokenizer = get_tokenizer(self._context.train_model_path)
                logger.info("✅ Reloaded tokenizer")
            else:
                logger.debug("Skipping tokenizer reload: no train_model_path")

            logger.info("✅ All training models reloaded successfully")

        except Exception as exc:
            logger.exception("Failed to reload training models: %s", exc)

    async def _handle_wait_for_window(
        self, target_window: int, current_block: int, last_processed_window: int
    ) -> None:
        """Display progress while waiting for the next training window."""
        if self._window_wait_tracker.should_log_initial():
            log_window_wait_initial(
                current_block=current_block,
                last_processed_window=last_processed_window,
                window_length=WINDOW_LENGTH,
            )
        elif self._window_wait_tracker.should_log_periodic():
            next_window = calculate_next_window(last_processed_window, WINDOW_LENGTH)
            log_window_wait_periodic(
                next_window=next_window,
                elapsed_seconds=self._window_wait_tracker.get_elapsed_seconds(),
            )

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

            # Register cleanup callback
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

    def _initialize_training_parameters(self) -> None:
        """Initialize optimizer and scheduler once for the trainer lifetime.

        These persist across windows to maintain training state and convergence.
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

        from grail.shared.constants import TRAINER_LR

        try:
            # Create optimizer for training model parameters with weight decay
            self._optimizer = torch.optim.AdamW(
                self._context.train_model.parameters(),
                lr=TRAINER_LR,
                betas=(0.9, 0.999),
                weight_decay=0.1,
            )

            # Calculate adaptive warmup as 5% of total training windows
            total_training_windows = 1000  # Typical training horizon
            warmup_steps = max(1, int(0.05 * total_training_windows))

            # Create warmup scheduler (linear increase from 0 to 1)
            def lr_lambda_warmup(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(self._optimizer, lr_lambda_warmup)

            # Create cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=total_training_windows - warmup_steps,
                eta_min=1e-7,
            )

            # Combine schedulers: warmup first, then cosine annealing
            self._scheduler = SequentialLR(
                self._optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

            logger.info(
                "Initialized training parameters: lr=%.2e, betas=(0.9, 0.999), weight_decay=0.1, warmup_steps=%d, total_windows=%d",
                TRAINER_LR,
                warmup_steps,
                total_training_windows,
            )
        except Exception as exc:
            logger.error("Failed to initialize training parameters: %s", exc, exc_info=True)
            raise

    async def _train_window(self, window: int) -> bool:
        logger.info("📖 _train_window called for window=%d", window)
        ctx = self._context

        # Verify models are loaded
        if ctx.train_model is None:
            logger.error("❌ train_model is None at start of _train_window!")
            return False
        if ctx.ref_model is None:
            logger.error("❌ ref_model is None at start of _train_window!")
            return False

        # Update heartbeat before long operation
        self.heartbeat()

        # Get subtensor for metagraph queries
        subtensor = await self.get_subtensor()

        service = TrainerService(
            wallet=ctx.wallet,
            credentials=ctx.credentials,
            checkpoint_manager=ctx.checkpoint_manager,
            monitor=ctx.monitor,
            train_model=ctx.train_model,
            ref_model=ctx.ref_model,
            tokenizer=ctx.tokenizer,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            chain_manager=ctx.chain_manager,
        )
        logger.info("TrainerService created, calling train_window")
        result = await service.train_window(window, subtensor)
        logger.info("TrainerService.train_window completed with result=%s", result)
        return result
