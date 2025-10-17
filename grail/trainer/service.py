"""Trainer service: train for a window and publish a checkpoint."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator

from grail.shared.constants import (
    NETUID,
    TRAINER_LR,
    TRAINER_WARMUP_STEPS,
)

from .algorithms import GRPOAlgorithm, TrainingAlgorithm
from .checkpointing import publish_checkpoint
from .config import TrainingConfig
from .data import load_grpo_groups
from .trust import get_trusted_miner_hotkeys

if TYPE_CHECKING:
    import bittensor as bt

    from grail.infrastructure.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


class TrainerService:
    """Coordinates model setup, algorithm epochs, and checkpoint publishing."""

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
        self.accelerator = Accelerator(mixed_precision="fp16")

    async def train_window(self, window: int, subtensor: Any) -> bool:
        # Deterministic seeding
        self._seed_all(window)

        # Get metagraph for UID mapping
        metagraph = await subtensor.metagraph(NETUID)
        uid_by_hotkey = dict(zip(metagraph.hotkeys, metagraph.uids, strict=True))

        # Get trusted miners from chain weights
        trusted_hotkeys = await self._get_trusted_miners(subtensor)

        # Skip training if no trusted miners available
        if not trusted_hotkeys:
            logger.warning(
                "No trusted miners for window %s; will wait for next window",
                window,
            )
            return False

        # Load data directly from trusted miners using shared chain manager
        groups = await load_grpo_groups(
            window,
            self.config,
            trusted_hotkeys,
            self.credentials,
            self.chain_manager,
            uid_by_hotkey,
        )

        if not groups:
            logger.warning(
                ("No valid GRPO groups for window %s; will wait for next window"),
                window,
            )
            return False

        # Models and tokenizer prepared once; optimizer per window
        model = self.train_model
        ref_model = self.ref_model
        tokenizer = self.tokenizer

        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINER_LR)
        model, ref_model, optimizer = self.accelerator.prepare(
            model,
            ref_model,
            optimizer,
        )
        if hasattr(ref_model, "eval"):
            ref_model.eval()

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=max(1, TRAINER_WARMUP_STEPS),
        )

        for epoch in range(self.config.epochs):
            logger.info("Epoch %s/%s", epoch + 1, self.config.epochs)
            metrics = await self.algorithm.train_epoch(
                model,
                ref_model,
                tokenizer,
                groups,
                optimizer,
                self.accelerator,
                self.monitor,
                window,
                self.config,
            )

            scheduler.step()
            if self.monitor:
                for key, value in metrics.items():
                    await self.monitor.log_gauge(f"training.{key}", value)
                await self.monitor.log_gauge(
                    "training.lr",
                    scheduler.get_last_lr()[0],
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

        # Publish checkpoint for the same window used for updates
        target_window = window
        logger.info("💾 Publishing checkpoint for window %s", target_window)

        unwrapped = self.accelerator.unwrap_model(model)
        success = await publish_checkpoint(
            unwrapped,
            tokenizer,
            target_window,
            window,
            self.wallet,
            self.credentials,
            self.checkpoint_manager,
            seed=window,
        )

        if not success:
            logger.error(
                "Failed to publish checkpoint",
            )
            logger.error("Window: %s", target_window)

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
