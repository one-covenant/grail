"""Trainer-facing checkpointing facade wrapping training.checkpointing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grail.training.checkpointing import publish_checkpoint as _publish_checkpoint

if TYPE_CHECKING:
    import bittensor as bt

    from grail.infrastructure.checkpoints import CheckpointManager


async def publish_checkpoint(
    model: Any,
    tokenizer: Any,
    target_window: int,
    trained_on_window: int,
    wallet: bt.wallet,
    credentials: Any,
    checkpoint_manager: CheckpointManager,
    seed: int,
) -> bool:
    """Stable facade used by TrainerService; delegates to training layer."""
    return await _publish_checkpoint(
        model,
        tokenizer,
        target_window,
        trained_on_window,
        wallet,
        credentials,
        checkpoint_manager,
        seed,
    )
