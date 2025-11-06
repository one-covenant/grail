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
from grail.shared.constants import NETUID, READY_MARKER_UPLOAD_BLOCKS, WINDOW_LENGTH, is_kl_enabled
from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)
from grail.trainer.algorithms import GRPOAlgorithm, TrainingAlgorithm
from grail.trainer.checkpointing import finalize_checkpoint_ready
from grail.trainer.config import EvalConfig
from grail.trainer.eval_planner import EvaluationPlanner
from grail.trainer.evaluator import EvaluatorService
from grail.trainer.inference_server import create_inference_server
from grail.trainer.service import TrainerService
from grail.trainer.training_state import (
    apply_training_state,
    load_training_state,
    save_training_state,
)

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
        # Persistent algorithm instance across windows (for KL controller, counters, etc.)
        self._algorithm: TrainingAlgorithm = GRPOAlgorithm()
        self._window_wait_tracker = WindowWaitTracker(log_interval_secs=120)
        self._wait_start_time: float | None = None
        self._last_wait_log: float = 0.0
        # Evaluation state
        self._eval_cfg = EvalConfig()
        self._eval_in_progress: bool = False
        self._eval_last_run_window_number: int | None = None
        self._windows_since_last_eval: int = 0  # Counter for windows processed since last eval
        self._eval_checkpoint_dir: str | None = None  # Saved checkpoint path for exact reload

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

                # Set monitoring context for eval metrics to use block_number as x-axis in wandb
                # This ensures eval/* metrics are plotted against block_number instead of global step
                if self._context.monitor:
                    self._context.monitor.set_block_context(current_block, None)

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

                # Step 1: Save exact training state before freeing GPU memory
                model_name = getattr(self._context.train_model, "name_or_path", "model")
                logger.info("Saving checkpoint before eval (models still in GPU memory)...")
                self._eval_checkpoint_dir = self._save_eval_checkpoint()
                logger.info("Checkpoint saved, now freeing GPU memory...")

                # Step 2: Free all GPU memory to maximize available memory for server
                self._free_training_vram_for_eval()

                # Step 3: Verify GPU memory is clean before server startup
                if torch.cuda.is_available():
                    try:
                        free_gb, total_gb = torch.cuda.mem_get_info()
                        logger.info(
                            "GPU memory after freeing models: %.2f GB free / %.2f GB total",
                            free_gb / (1024**3),
                            total_gb / (1024**3),
                        )
                    except Exception:
                        pass

                # Step 4: Create and start server with clean GPU
                # Construct path to chat_template.jinja saved by tokenizer.save_pretrained()
                import os

                chat_template_path = os.path.join(self._eval_checkpoint_dir, "chat_template.jinja")
                # Verify the file exists; if not, log warning and proceed without it
                if not os.path.isfile(chat_template_path):
                    logger.warning(
                        "chat_template.jinja not found at %s; server may use default template",
                        chat_template_path,
                    )
                    chat_template_path = None

                server_manager = create_inference_server(
                    backend=self._eval_cfg.backend,
                    model_path=self._eval_checkpoint_dir,
                    eval_config=self._eval_cfg,
                    model_name_override=model_name,
                    chat_template_path=chat_template_path,
                )

                async with server_manager as server:
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

                # Context manager handles server cleanup and waits for GPU memory release
                logger.info("Server context exited, subprocess terminated")

                # Verify GPU memory was released by server shutdown before reloading models
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
            # Always ensure models are reloaded and temp checkpoint cleaned up
            try:
                if self._eval_checkpoint_dir:
                    logger.info("Reloading training models from eval checkpoint...")
                    self._reload_training_models()
                    logger.info("Training models reloaded successfully")

                    self._cleanup_eval_checkpoint(self._eval_checkpoint_dir)
                    self._eval_checkpoint_dir = None
            finally:
                logger.debug("_maybe_run_evaluation finally block: setting eval_in_progress=False")
                self._eval_in_progress = False

    # -------------- Eval orchestration helpers --------------
    def _free_training_vram_for_eval(self) -> None:
        """Drop references to training and ref models and clear CUDA cache.

        Call this after saving the evaluation checkpoint and before starting
        any external inference server to maximize available GPU memory.
        """
        try:
            # Detach optimizer and scheduler so they can be rebuilt and reattached later
            self._optimizer = None
            self._scheduler = None

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

    def _save_eval_checkpoint(self) -> str:
        """Persist the current training model, reference model, and tokenizer to a temp directory.

        Both models are saved to ensure exact weight preservation and bit-identical reload.
        The training model is saved to the root; ref model to a 'ref_model' subdirectory.

        Returns:
            Absolute path to the saved checkpoint directory.
        """
        if self._context.train_model is None or self._context.tokenizer is None:
            raise RuntimeError("Cannot save eval checkpoint: train_model/tokenizer not loaded")
        kl_enabled: bool = is_kl_enabled()
        if kl_enabled and self._context.ref_model is None:
            raise RuntimeError("Cannot save eval checkpoint: ref_model not loaded (KL enabled)")

        import json
        import os
        import shutil
        import tempfile

        checkpoint_dir = tempfile.mkdtemp(prefix="grail_eval_ckpt_")
        try:
            # Ensure tokenizer has the correct Qwen chat template before saving
            # This is critical for vLLM/SGLang to apply the correct formatting
            try:
                from grail.shared.chat_templates import build_qwen_chat_template
                from grail.shared.prompt_constants import REASONING_START, SYSTEM_PROMPT

                expected_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
                current_template = getattr(self._context.tokenizer, "chat_template", None)

                if not current_template:
                    logger.warning(
                        "Tokenizer chat_template missing before eval checkpoint save; applying Qwen template"
                    )
                    self._context.tokenizer.chat_template = expected_template
                elif current_template != expected_template:
                    logger.warning(
                        "Tokenizer chat_template differs from expected before eval checkpoint save; applying Qwen template"
                    )
                    self._context.tokenizer.chat_template = expected_template
                else:
                    logger.debug("Tokenizer chat_template matches expected Qwen template")
            except Exception as exc:
                logger.warning(
                    "Failed to verify/apply chat template before checkpoint save: %s", exc
                )

            # Save training model to root of checkpoint directory
            self._context.train_model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )
            self._context.tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Saved training model to eval checkpoint: %s", checkpoint_dir)

            # Write metadata to preserve original model identifier for stable name_or_path
            try:
                train_meta = {
                    "model_name": self._context.train_model_path
                    or getattr(self._context.train_model, "name_or_path", "model"),
                }
                with open(
                    os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(train_meta, f)
            except Exception as e:
                logger.debug("Failed to write training metadata.json: %s", e)

            # Save reference model to subdirectory only when KL is enabled
            if kl_enabled and self._context.ref_model is not None:
                ref_model_dir = os.path.join(checkpoint_dir, "ref_model")
                os.makedirs(ref_model_dir, exist_ok=True)
                self._context.ref_model.save_pretrained(
                    ref_model_dir,
                    safe_serialization=True,
                )
                logger.info("Saved reference model to eval checkpoint: %s", ref_model_dir)

                # Write metadata for reference model as well
                try:
                    ref_meta = {
                        "model_name": self._context.ref_model_path
                        or getattr(self._context.ref_model, "name_or_path", "model"),
                    }
                    with open(
                        os.path.join(ref_model_dir, "metadata.json"), "w", encoding="utf-8"
                    ) as f:
                        json.dump(ref_meta, f)
                except Exception as e:
                    logger.debug("Failed to write reference metadata.json: %s", e)

            # Persist optimizer/scheduler and RNG states for seamless resume
            try:
                # Save state on rank 0 only in distributed settings, then barrier
                should_save = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    should_save = torch.distributed.get_rank() == 0
                if should_save:
                    save_training_state(
                        checkpoint_dir,
                        optimizer=self._optimizer,
                        scheduler=self._scheduler,
                    )
                    logger.info(
                        "Saved training state (optimizer/scheduler/RNG) to %s", checkpoint_dir
                    )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save training state: %s", exc)

            return checkpoint_dir
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save evaluation checkpoint: %s", exc)
            try:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            except Exception:
                pass
            raise

    def _cleanup_eval_checkpoint(self, path: str) -> None:
        """Remove a temporary evaluation checkpoint directory."""
        import shutil

        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Cleaned up eval checkpoint: %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Checkpoint cleanup failed: %s", exc)

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
        """Reload training and reference models after evaluation server shutdown.

        Prefers eval checkpoint (exact saved state) over original path to guarantee
        bit-identical reload including weights and tokenizer chat template.
        Both models are reloaded from the same checkpoint to ensure consistency.
        """
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
            import os

            from grail.model.provider import get_model, get_tokenizer

            # Prefer eval checkpoint (exact saved state) over original path
            train_reload_path = self._eval_checkpoint_dir or self._context.train_model_path
            kl_enabled: bool = is_kl_enabled()
            ref_reload_path = (
                os.path.join(self._eval_checkpoint_dir, "ref_model")
                if kl_enabled and self._eval_checkpoint_dir
                else (self._context.ref_model_path if kl_enabled else None)
            )

            # Reload training model
            if self._context.train_model is None and train_reload_path:
                reload_source = "eval checkpoint" if self._eval_checkpoint_dir else "original path"
                logger.info(
                    "Reloading training model from %s: %s", reload_source, train_reload_path
                )
                self._context.train_model = get_model(train_reload_path, eval_mode=False)
                logger.info("✅ Reloaded training model from %s", reload_source)
            else:
                logger.debug(
                    "Skipping training model reload: train_reload_path=%s", train_reload_path
                )

            # Reload reference model only when KL is enabled
            if kl_enabled and self._context.ref_model is None and ref_reload_path:
                reload_source = "eval checkpoint" if self._eval_checkpoint_dir else "original path"
                logger.info("Reloading reference model from %s: %s", reload_source, ref_reload_path)
                self._context.ref_model = get_model(ref_reload_path, eval_mode=True)
                logger.info("✅ Reloaded reference model from %s", reload_source)
            else:
                logger.debug(
                    "Skipping reference model reload (kl_enabled=%s, ref_reload_path=%s)",
                    kl_enabled,
                    ref_reload_path,
                )

            # Reload tokenizer from same path as training model (preserves chat template)
            if train_reload_path:
                reload_source = "eval checkpoint" if self._eval_checkpoint_dir else "original path"
                logger.info("Reloading tokenizer from %s: %s", reload_source, train_reload_path)
                self._context.tokenizer = get_tokenizer(train_reload_path)

                # Verify chat template is present
                has_template = hasattr(self._context.tokenizer, "chat_template") and bool(
                    self._context.tokenizer.chat_template
                )
                if has_template:
                    logger.info("✅ Reloaded tokenizer with chat template from %s", reload_source)
                else:
                    logger.warning(
                        "⚠️ Tokenizer reloaded but chat_template missing (source: %s)",
                        reload_source,
                    )
            else:
                logger.debug("Skipping tokenizer reload: no train_model_path")

            # Ensure optimizer and scheduler are attached to the reloaded model
            if self._optimizer is None or self._scheduler is None:
                self._initialize_training_parameters()

            # Restore optimizer/scheduler/RNG state if present in eval checkpoint
            if self._eval_checkpoint_dir:
                try:
                    # Ensure all ranks see the saved files
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.barrier()
                    opt_state, sched_state, rng_state = load_training_state(
                        self._eval_checkpoint_dir
                    )
                    apply_training_state(
                        optimizer=self._optimizer,
                        scheduler=self._scheduler,
                        optimizer_state=opt_state,
                        scheduler_state=sched_state,
                        rng_state=rng_state,
                    )
                    logger.info("✅ Restored optimizer/scheduler/RNG state from eval checkpoint")
                except Exception as exc:
                    logger.warning("Failed to restore training state: %s", exc)

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

        # Only require reference model when KL is enabled
        if ctx.ref_model is None and is_kl_enabled():
            logger.error("❌ ref_model is required when KL is enabled")
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
            algorithm=self._algorithm,
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
