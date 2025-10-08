"""Trainer neuron orchestrating GRPO training and checkpoint publishing."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import bittensor as bt
import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infrastructure.checkpoints import CheckpointManager
from ..infrastructure.network import create_subtensor
from ..shared.constants import (
    MODEL_NAME,
    TRAINER_EPOCHS,
    TRAINER_GROUP_ADV_SUM_TOL,
    TRAINER_LR,
    TRAINER_WARMUP_STEPS,
    WINDOW_LENGTH,
)
from ..training.checkpointing import publish_checkpoint
from ..training.data import load_grpo_groups
from ..training.grpo import train_grpo_epoch
from .base import BaseNeuron

logger = logging.getLogger(__name__)


@dataclass
class TrainerContext:
    """Resources required to run the trainer neuron."""

    wallet: bt.wallet
    credentials: Any
    checkpoint_manager: CheckpointManager
    monitor: Any | None
    update_heartbeat: Callable[[], None]


class TrainerNeuron(BaseNeuron):
    """Runs GRPO training loops and publishes checkpoints."""

    def __init__(self, context: TrainerContext) -> None:
        super().__init__()
        self._context = context

    async def run(self) -> None:  # noqa: D401
        subtensor: bt.subtensor | None = None
        last_processed_window = -1

        while not self.stop_event.is_set():
            try:
                self._context.update_heartbeat()

                if subtensor is None:
                    logger.info("Making Bittensor connection...")
                    subtensor = await create_subtensor()
                    logger.info("Connected to subtensor")

                current_block = await subtensor.get_current_block()
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                target_window = current_window - WINDOW_LENGTH

                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(10)
                    continue

                logger.info("🎓 Processing training for window %s", target_window)
                success = await self._train_window(target_window)

                if success:
                    logger.info("✅ Completed training cycle for window %s", target_window)
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training.successful_windows")
                else:
                    logger.warning("⚠️ Training cycle had issues for window %s", target_window)
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training.failed_windows")

                last_processed_window = target_window

            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("Trainer loop error: %s", exc)
                subtensor = None
                await asyncio.sleep(30)

    async def _train_window(self, window: int) -> bool:
        ctx = self._context
        ctx.update_heartbeat()

        self._seed_all(window)

        groups = await load_grpo_groups(window, TRAINER_GROUP_ADV_SUM_TOL)
        if not groups:
            logger.warning("No valid GRPO groups for window %s", window)
            # TODO: publish previous stable checkpoint when data is missing
            return False

        total_rollouts = sum(len(group.rollouts) for group in groups)
        success_count = sum(1 for group in groups for rollout in group.rollouts if rollout.success)
        mean_reward = (
            sum(rollout.reward for group in groups for rollout in group.rollouts) / total_rollouts
            if total_rollouts
            else 0.0
        )

        logger.info(
            "📚 Training on %s groups (%s rollouts), %s successful, mean reward %.3f",
            len(groups),
            total_rollouts,
            success_count,
            mean_reward,
        )

        accelerator = Accelerator(mixed_precision="fp16")

        logger.info("Loading base model %s", MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ref_model = await self._load_reference_model(ctx.checkpoint_manager)

        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINER_LR)
        model, ref_model, optimizer = accelerator.prepare(model, ref_model, optimizer)
        ref_model.eval()

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=max(1, TRAINER_WARMUP_STEPS),
        )
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(TRAINER_EPOCHS):
            ctx.update_heartbeat()
            logger.info("Epoch %s/%s", epoch + 1, TRAINER_EPOCHS)

            metrics = await train_grpo_epoch(
                model,
                ref_model,
                tokenizer,
                groups,
                optimizer,
                scaler,
                accelerator,
                ctx.monitor,
                window,
            )

            scheduler.step()

            if ctx.monitor:
                for key, value in metrics.items():
                    await ctx.monitor.log_gauge(f"training.{key}", value)
                await ctx.monitor.log_gauge("training.lr", scheduler.get_last_lr()[0])
                await ctx.monitor.log_counter("training.epochs_completed")

            logger.info(
                "Epoch %s metrics: loss=%.4f pg=%.4f kl=%.4f entropy=%.4f",
                epoch + 1,
                metrics.get("loss_total", 0.0),
                metrics.get("loss_pg", 0.0),
                metrics.get("loss_kl", 0.0),
                metrics.get("loss_entropy", 0.0),
            )

        future_window = window + WINDOW_LENGTH
        logger.info("💾 Publishing checkpoint for window %s", future_window)

        unwrapped_model = accelerator.unwrap_model(model)
        success = await publish_checkpoint(
            unwrapped_model,
            tokenizer,
            future_window,
            window,
            ctx.wallet,
            ctx.credentials,
            ctx.checkpoint_manager,
            seed=window,
        )

        if not success:
            logger.error("Failed to publish checkpoint for window %s", future_window)

        return success

    async def _load_reference_model(self, checkpoint_manager: CheckpointManager):
        ref_model_path: str | None = None
        try:
            windows = await checkpoint_manager.list_remote_windows()
            if windows:
                latest_window = max(windows)
                checkpoint_path = await checkpoint_manager.get_checkpoint(latest_window)
                if checkpoint_path:
                    ref_model_path = str(checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to fetch reference checkpoint: %s", exc)

        if ref_model_path:
            logger.info("Loading reference model from %s", ref_model_path)
            return AutoModelForCausalLM.from_pretrained(
                ref_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
            )

        logger.info("Using base model as reference")
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
        )

    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
