"""Base classes for training algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from accelerate import Accelerator

    from grail.trainer.config import TrainingConfig


class TrainingAlgorithm:
    """Abstract base for training algorithms.

    We use a simple inheritance base instead of typing.Protocol to
    keep it concrete.
    """

    name: str = "base"

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
    ) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError
