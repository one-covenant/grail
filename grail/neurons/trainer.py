"""Trainer neuron orchestrating window selection and delegating training."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import tempfile
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

                if target_window <= last_processed_window or target_window < 0:
                    await self._handle_wait_for_window(
                        target_window, current_block, last_processed_window
                    )
                    await asyncio.sleep(10)
                    continue

                # Window is available - reset wait tracker for next time
                self._window_wait_tracker.reset()

                # Periodic evaluation at startup and every configured interval
                if await self._maybe_run_evaluation(current_window):
                    # Skip training when evaluation runs (may span multiple windows)
                    last_processed_window = target_window
                    continue

                return False #TODO: remove this

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

            except asyncio.CancelledError:  # pragma: no cover - coop shutdown
                break
            except Exception:
                logger.exception("Trainer loop error", exc_info=True)
                # Force reconnect on next iteration
                self.reset_subtensor()
                await asyncio.sleep(30)

    async def _maybe_run_evaluation(self, current_window: int) -> bool:
        """Run evaluation if due; return True if evaluation executed."""
        if not self._eval_cfg.enabled:
            return False

        window_number = current_window // WINDOW_LENGTH

        should_start = (
            self._eval_in_progress
            or self._eval_last_run_window_number is None
            or (
                window_number % max(1, self._eval_cfg.window_interval) == 0
                and self._eval_last_run_window_number != window_number
            )
        )

        if not should_start:
            return False

        # Mark progress
        self._eval_in_progress = True

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

        # Prepare backend resources (checkpoint, server, etc.)
        tmp_dir: str | None = None
        server_proc: subprocess.Popen[bytes] | None = None
        evaluator: EvaluatorService | None = None
        original_backend = self._eval_cfg.backend

        try:
            # Optionally start sgLang server with saved checkpoint
            if self._eval_cfg.backend == "sglang" and self._eval_cfg.sglang_start_server:
                tmp_dir, server_proc = await self._prepare_sglang_server()

            # Create evaluator with appropriate backend
            evaluator = EvaluatorService(
                model=self._context.train_model,
                tokenizer=self._context.tokenizer,
                env_factory=env_factory,
                config=self._eval_cfg,
                monitor=self._context.monitor,
                device="cuda",
            )

            logger.info(
                "🧪 Starting evaluation: window_number=%s ids=%s replicates=%s split=%s backend=%s",
                window_number,
                len(plan.ids),
                plan.replicates,
                self._eval_cfg.split,
                self._eval_cfg.backend,
            )

            logger.info("📊 Calling evaluator.run_cycle...")
            metrics = await evaluator.run_cycle(plan, start_offset=0, heartbeat=self.heartbeat)
            logger.info("📊 run_cycle completed, got metrics: %s", metrics)

            if self._context.monitor:
                logger.info("📊 Logging to monitor...")
                await self._context.monitor.log_counter("eval/cycle_completed")
                logger.info("📊 Logged counter, now logging gauges...")
                for key, val in metrics.items():
                    await self._context.monitor.log_gauge(f"eval/{key}", float(val))
                logger.info("📊 Monitor logging complete")

            logger.info("🧪 Evaluation metrics: %s", metrics)

            self._eval_last_run_window_number = window_number
            return True
        except Exception:
            logger.exception("Evaluation failed", exc_info=True)
            return False
        finally:
            # Cleanup order is critical for GPU memory management:
            # 1. Shutdown evaluation backend engines (SGLang/vLLM) to free GPU
            logger.info("🧹 Starting evaluator cleanup...")
            if evaluator is not None:
                logger.info("🧹 Calling evaluator.shutdown()...")
                evaluator.shutdown()
                logger.info("🧹 evaluator.shutdown() completed")

            # 2. Terminate server process if running (and wait for GPU memory release)
            self._terminate_process(server_proc)

            # 3. Reload training models now that GPU memory is free
            if tmp_dir is not None:
                self._reload_training_artifacts_if_needed(tmp_dir)
                self._cleanup_temp_dir(tmp_dir)

            # 4. Restore backend config
            self._eval_cfg.backend = original_backend
            self._eval_in_progress = False

    # -------------- Eval orchestration helpers --------------
    async def _prepare_sglang_server(self) -> tuple[str | None, subprocess.Popen[bytes] | None]:
        """Save checkpoint, free VRAM, and launch sgLang server.

        Returns (tmp_checkpoint_dir, server_process) tuple.
        Falls back to HF backend if any step fails.
        """
        tmp_dir = self._save_eval_checkpoint()
        if tmp_dir is None:
            logger.warning("Checkpoint save failed; falling back to HF backend for eval.")
            self._eval_cfg.backend = "hf"
            return None, None

        self._free_vram()

        # Choose model path for server: use saved checkpoint
        server_proc, bound_port = await self._launch_and_wait_sglang_server(
            model_path=tmp_dir,
            host=self._eval_cfg.sglang_host,
            port=self._eval_cfg.sglang_port,
            timeout_s=self._eval_cfg.sglang_server_timeout_s,
            trust_remote_code=self._eval_cfg.sglang_trust_remote_code,
        )

        # Check if server launched successfully
        if server_proc is not None and bound_port is not None:
            self._eval_cfg.sglang_port = bound_port
            return tmp_dir, server_proc

        # Server failed; fall back to HF
        logger.warning("sgLang server not ready; using HF backend for this evaluation.")
        self._eval_cfg.backend = "hf"
        return tmp_dir, None

    def _save_eval_checkpoint(self) -> str | None:
        """Save current training model/tokenizer to a temporary directory for eval.

        Returns path to the saved checkpoint, or None on failure.
        """
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="grail_eval_ckpt_")
            self._context.train_model.save_pretrained(tmp_dir, safe_serialization=True)
            self._context.tokenizer.save_pretrained(tmp_dir)
            logger.info("Saved eval checkpoint to %s", tmp_dir)
            return tmp_dir
        except Exception as exc:
            logger.warning("Failed to save eval checkpoint: %s", exc)
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

    def _free_vram(self) -> None:
        """Release GPU memory by dropping references and emptying CUDA cache."""
        try:
            self._context.train_model = None  # type: ignore[assignment]
            self._context.ref_model = None  # type: ignore[assignment]
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("VRAM free encountered issue: %s", exc)

    async def _launch_and_wait_sglang_server(
        self,
        *,
        model_path: str | None,
        host: str,
        port: int,
        timeout_s: float,
        trust_remote_code: bool,
    ) -> tuple[subprocess.Popen[bytes] | None, int | None]:
        """Launch sgLang server subprocess and wait until HTTP endpoint is ready.

        Configures server with SGLang best practices:
        - Memory-optimized settings for large models
        - Request rate limiting to prevent overload
        - Extended startup timeouts
        - Post-launch warmup to verify stability
        """
        bound_port = int(port)
        if bound_port <= 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, 0))
                bound_port = s.getsockname()[1]

        # SGLang server launch with optimized parameters
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path or "",
            "--host",
            host,
            "--port",
            str(bound_port),
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            # SGLang performance tuning per docs
            "--mem-fraction-static",
            "0.85",  # Reserve 85% VRAM for KV cache
            "--max-running-requests",
            "8",  # Limit concurrent batch processing
            "--schedule-policy",
            "fcfs",  # Fair scheduling
        ]
        if trust_remote_code:
            cmd.append("--trust-remote-code")

        proc: subprocess.Popen[bytes] | None = None
        try:
            # Launch with combined stdout/stderr to monitor for errors
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,  # Keep as bytes
            )
            logger.info(
                "Launched sgLang server (pid=%s) on %s:%s with optimized config",
                proc.pid if proc else None,
                host,
                bound_port,
            )
        except Exception as exc:
            logger.warning("Failed to launch sgLang server: %s", exc)
            return None, None

        # Poll readiness with extended timeout for model loading
        import time as _time

        import requests

        ready_url = f"http://{host}:{bound_port}/v1/models"
        t_deadline = _time.time() + float(timeout_s)
        last_err: str | None = None
        poll_count: int = 0

        while _time.time() < t_deadline:
            # Check if process is still alive
            if proc.poll() is not None:
                # Process exited unexpectedly
                logger.error(
                    "sgLang server process exited unexpectedly (return code: %s)",
                    proc.returncode,
                )
                return None, None

            try:
                r = requests.get(ready_url, timeout=3.0)
                if r.status_code == 200:
                    logger.info("sgLang server ready at %s:%s", host, bound_port)

                    # Optional: send warmup request to force KV cache initialization
                    try:
                        warmup_start = _time.time()
                        warmup_payload = {
                            "model": model_path or "model",
                            "prompt": "OK",
                            "max_tokens": 1,
                        }
                        requests.post(
                            f"{ready_url.replace('/v1/models', '')}/v1/completions",
                            json=warmup_payload,
                            timeout=30.0,
                        )
                        warmup_time = _time.time() - warmup_start
                        logger.info("Server warmup completed in %.2fs", warmup_time)
                    except Exception as warmup_err:
                        logger.warning("Warmup request failed (non-fatal): %s", warmup_err)

                    return proc, bound_port

                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = str(exc)

            poll_count += 1
            if poll_count % 6 == 0:  # Log every 3 seconds (0.5s * 6)
                elapsed = _time.time() - (t_deadline - timeout_s)
                logger.debug(
                    "Waiting for sgLang server (%.1fs elapsed): %s",
                    elapsed,
                    last_err,
                )

            await asyncio.sleep(0.5)

        logger.warning(
            "sgLang server readiness check failed after %.1fs: %s",
            timeout_s,
            last_err,
        )
        # If not ready, terminate and signal failure
        self._terminate_process(proc)
        return None, None

    def _terminate_process(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Terminate a subprocess gracefully, wait for GPU memory release."""
        if proc is None:
            return

        import time

        try:
            # Step 1: Graceful shutdown
            proc.terminate()
            try:
                proc.wait(timeout=10)
                logger.info("SGLang server terminated (pid=%s)", proc.pid)
            except Exception:
                logger.warning("SGLang server didn't exit gracefully, force killing")
                proc.kill()
                proc.wait(timeout=5)
        except Exception as e:
            logger.warning("Error terminating SGLang process: %s", e)

        # Step 2: Wait for GPU memory to actually be freed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            # Poll for GPU memory release (up to 30 seconds)
            max_wait = 30.0
            start = time.time()
            while time.time() - start < max_wait:
                try:
                    # Check if we have enough free memory for training model reload
                    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # GB
                    if free_mem > 25.0:  # Need at least 25 GB free for training model
                        logger.info("GPU memory freed: %.2f GB available", free_mem)
                        return
                    time.sleep(1.0)
                except Exception:
                    break

            # Log final state even if timeout
            try:
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                logger.warning(
                    "GPU memory may still be held after %.1fs: %.2f GB free",
                    max_wait,
                    free_mem,
                )
            except Exception:
                pass

    def _reload_training_artifacts_if_needed(self, tmp_dir: str | None) -> None:
        """Reload training and reference models after evaluation if they were freed."""
        # Check if models need reloading
        needs_reload = self._context.train_model is None or self._context.ref_model is None
        if not needs_reload:
            return

        # Log GPU memory state before reload
        if torch.cuda.is_available():
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                logger.info(
                    "Reloading training artifacts: GPU memory %.2f GB free / %.2f GB total",
                    free_mem / (1024**3),
                    total_mem / (1024**3),
                )
            except Exception:
                pass

        try:
            from grail.model.provider import get_model, get_tokenizer

            # Reload training model from saved checkpoint or original path
            if self._context.train_model is None:
                model_source = (
                    tmp_dir
                    if (tmp_dir and os.path.isdir(tmp_dir))
                    else self._context.train_model_path
                )
                if model_source:
                    self._context.train_model = get_model(model_source, eval_mode=False)
                    logger.info("Reloaded training model from %s", model_source)

            # Reload reference model from original path
            if self._context.ref_model is None and self._context.ref_model_path:
                self._context.ref_model = get_model(self._context.ref_model_path, eval_mode=True)
                logger.info("Reloaded reference model from %s", self._context.ref_model_path)

            # Reload tokenizer if needed
            if tmp_dir and os.path.isdir(tmp_dir):
                self._context.tokenizer = get_tokenizer(tmp_dir)
            elif self._context.train_model_path:
                self._context.tokenizer = get_tokenizer(self._context.train_model_path)

        except Exception as exc:
            logger.warning("Failed to reload training artifacts after eval: %s", exc)

    def _cleanup_temp_dir(self, tmp_dir: str | None) -> None:
        """Remove a temporary directory if provided."""
        if tmp_dir is None:
            return
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

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
        ctx = self._context
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
        return await service.train_window(window, subtensor)
