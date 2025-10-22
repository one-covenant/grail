"""Trainer service: train for a window and publish a checkpoint."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator

from grail.shared.constants import (
    NETUID,
    TRAINER_UPLOAD_BUFFER_BLOCKS,
    WINDOW_LENGTH,
)

from .algorithms import GRPOAlgorithm, TrainingAlgorithm
from .algorithms.grpo import load_grpo_groups
from .checkpointing import publish_checkpoint
from .config import TrainingConfig
from .trust import get_trusted_miner_hotkeys

if TYPE_CHECKING:
    import bittensor as bt

    from grail.infrastructure.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


class TrainerService:
    """Coordinates algorithm epochs and checkpoint publishing.

    Optimizer and scheduler are persistent across windows to maintain
    training state. Each window is a single training step.
    """

    def __init__(
        self,
        wallet: bt.wallet,
        credentials: Any,
        checkpoint_manager: CheckpointManager,
        monitor: Any | None,
        algorithm: TrainingAlgorithm | None = None,
        config: TrainingConfig | None = None,
        *,
        train_model: Any,
        ref_model: Any,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        chain_manager: Any | None = None,
    ) -> None:
        self.wallet = wallet
        self.credentials = credentials
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.algorithm = algorithm or GRPOAlgorithm()
        self.config = config or TrainingConfig()
        self.train_model = train_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.chain_manager = chain_manager
        self.accelerator = Accelerator(mixed_precision="no")

        # Persistent optimizer and scheduler across windows
        self.optimizer = optimizer
        self.scheduler = scheduler

    async def train_window(self, window: int, subtensor: Any) -> bool:
        # `window` is the target (past) window for training, not the current window
        self._seed_all(window)

        # Get metagraph for UID mapping
        metagraph = await subtensor.metagraph(NETUID)
        uid_by_hotkey = dict(zip(metagraph.hotkeys, metagraph.uids, strict=True))

        # Get trusted miners from chain weights
        trusted_hotkeys = await self._get_trusted_miners(subtensor)

        # TODO: this should be removed later on but right now we only trust uid 80 for testing
        trusted_hotkeys = {metagraph.hotkeys[80]}

        # Load data directly from trusted miners using shared chain manager
        groups = []
        training_skipped = False

        if not trusted_hotkeys:
            logger.warning(
                "No trusted miners for window %s; will publish unchanged checkpoint",
                window,
            )
            training_skipped = True
        else:
            groups = await load_grpo_groups(
                window,
                self.config.group_adv_sum_tolerance,
                trusted_hotkeys,
                self.credentials,
                self.chain_manager,
                uid_by_hotkey,
            )

            if not groups:
                logger.warning(
                    "No valid GRPO groups for window %s; will publish unchanged checkpoint",
                    window,
                )
                training_skipped = True

        # Use persistent models
        model = self.train_model
        ref_model = self.ref_model
        tokenizer = self.tokenizer

        # Calculate window timing
        current_window = window + WINDOW_LENGTH
        next_window = current_window + WINDOW_LENGTH
        deadline_block = next_window - TRAINER_UPLOAD_BUFFER_BLOCKS

        # Only train if we have valid data
        epochs_completed = 0
        metrics: dict[str, Any] = {}

        if not training_skipped:
            # Ensure optimizer and scheduler are initialized
            if self.optimizer is None or self.scheduler is None:
                raise RuntimeError(
                    "Optimizer and scheduler must be initialized before training windows"
                )

            # Prepare models and optimizer for distributed training
            model, ref_model, opt = self.accelerator.prepare(
                model,
                ref_model,
                self.optimizer,
            )

            if hasattr(ref_model, "eval"):
                ref_model.eval()

            training_start = time.monotonic()

            for epoch in range(self.config.epochs):
                # Check deadline before starting epoch
                try:
                    current_block = await subtensor.get_current_block()
                    if current_block >= deadline_block:
                        logger.warning(
                            "⏰ Upload deadline reached at block %s (target: %s); "
                            "stopping training after %s/%s epochs",
                            current_block,
                            deadline_block,
                            epoch,
                            self.config.epochs,
                        )
                        if self.monitor:
                            await self.monitor.log_counter("training.deadline_hit")
                            await self.monitor.log_gauge(
                                "training.epochs_completed_before_deadline", epoch
                            )
                        break
                except Exception as e:
                    logger.warning("Failed to check deadline: %s", e)

                logger.info("Epoch %s/%s", epoch + 1, self.config.epochs)
                metrics = await self.algorithm.train_epoch(
                    model,
                    ref_model,
                    tokenizer,
                    groups,
                    opt,
                    self.accelerator,
                    self.monitor,
                    window,
                    self.config,
                )

                self.scheduler.step()
                if self.monitor:
                    for key, value in metrics.items():
                        await self.monitor.log_gauge(f"training.{key}", value)
                    await self.monitor.log_gauge(
                        "training.lr",
                        self.scheduler.get_last_lr()[0],
                    )
                    await self.monitor.log_counter("training.epochs_completed")

                logger.info(
                    "Epoch %s metrics: loss=%.4f pg=%.4f kl=%.4f entropy=%.4f",
                    epoch + 1,
                    metrics.get("loss_total", 0.0),
                    metrics.get("loss_pg", 0.0),
                    metrics.get("loss_kl", 0.0),
                    metrics.get("loss_entropy", 0.0),
                )
                epochs_completed = epoch + 1

            # Log total training time
            training_duration = time.monotonic() - training_start
            logger.info(
                "Training completed in %.1fs for %s epochs", training_duration, epochs_completed
            )
            if self.monitor:
                await self.monitor.log_gauge("profiling/training_duration", training_duration)

            # Unwrap model for publishing
            unwrapped = self.accelerator.unwrap_model(model)
        else:
            # Training was skipped; use model as-is
            logger.info("Training skipped for window %s; publishing unchanged checkpoint", window)
            unwrapped = model

        # Publish checkpoint with deadline enforcement
        # Publish checkpoint for CURRENT window (where it will be used), not the past window trained on
        checkpoint_publish_window = current_window  # Use current_window, not target window

        if training_skipped:
            logger.info(
                "💾 Publishing unchanged checkpoint for window %s (training skipped)",
                checkpoint_publish_window,
            )
        else:
            logger.info(
                "💾 Publishing checkpoint for window %s (trained on window %s)",
                checkpoint_publish_window,
                window,
            )

        # Time the checkpoint publishing
        publish_start = time.monotonic()
        if self.monitor:
            with self.monitor.timer("profiling/checkpoint_publish"):
                success = await publish_checkpoint(
                    unwrapped,
                    tokenizer,
                    checkpoint_publish_window,
                    window,
                    self.wallet,
                    self.credentials,
                    self.checkpoint_manager,
                    seed=window,
                )
        else:
            success = await publish_checkpoint(
                unwrapped,
                tokenizer,
                checkpoint_publish_window,
                window,
                self.wallet,
                self.credentials,
                self.checkpoint_manager,
                seed=window,
            )

        publish_duration = time.monotonic() - publish_start
        if self.monitor:
            await self.monitor.log_gauge("profiling/checkpoint_publish_duration", publish_duration)

        # Verify checkpoint was published before deadline
        try:
            final_block = await subtensor.get_current_block()
            if final_block >= next_window:
                logger.warning(
                    "⚠️ Checkpoint published after start of next window! "
                    "Current block %s >= deadline %s",
                    final_block,
                    next_window,
                )
                if self.monitor:
                    await self.monitor.log_counter("training.checkpoint_published_late")
            else:
                logger.info("✅ Checkpoint published before deadline with margin")
                if self.monitor:
                    await self.monitor.log_counter("training.checkpoint_published_on_time")
        except Exception as e:
            logger.warning("Failed to verify checkpoint deadline: %s", e)

        if not success:
            logger.error(
                "Failed to publish checkpoint for window %s (trained on window %s)",
                checkpoint_publish_window,
                window,
            )

        return success

    async def _load_reference_model(self) -> Any:
        # Unused after refactor; kept for API compatibility.
        return self.ref_model

    async def _get_trusted_miners(self, subtensor: Any) -> set[str]:
        """Get trusted miner hotkeys from chain aggregation.

        Subtensor is wrapped with ResilientSubtensor, providing automatic
        timeout, retry, and circuit breaker protection.

        Returns empty set if no trusted miners can be determined.
        """
        try:
            # Fetch metagraph (protected by ResilientSubtensor)
            metagraph = await subtensor.metagraph(NETUID)

            # Compute trusted miners (quick local computation)
            trusted_hotkeys = await get_trusted_miner_hotkeys(
                metagraph,
                self.config.min_aggregate_weight,
                self.config.min_trusted_miners,
                timeout=5.0,  # Quick timeout for local computation
                subtensor=subtensor,  # For raw weights access
            )

            return trusted_hotkeys

        except Exception as exc:
            logger.error(
                "Failed to get trusted miners: %s; skipping",
                exc,
                exc_info=True,
            )
            return set()

    def _seed_all(self, seed: int) -> None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
