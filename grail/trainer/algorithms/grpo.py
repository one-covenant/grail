"""GRPO algorithm implementation using the training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grail.training.grpo import train_grpo_epoch as _train_grpo_epoch

from .base import TrainingAlgorithm

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator

    from grail.trainer.config import TrainingConfig


class GRPOAlgorithm(TrainingAlgorithm):
    name: str = "grpo"

    async def train_epoch(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        groups: list[Any],
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        monitor: Any | None,
        window: int,
        config: TrainingConfig,
    ) -> dict[str, float]:
        # Delegate to existing implementation to stay DRY for now
        return await _train_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            accelerator,
            monitor,
            window,
        )
