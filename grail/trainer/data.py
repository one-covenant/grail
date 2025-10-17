"""Re-export of training data structures and loaders for trainer namespace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grail.training.data import GRPOGroup
from grail.training.data import load_grpo_groups as _load_grpo_groups

__all__ = ["GRPOGroup", "load_grpo_groups"]

if TYPE_CHECKING:
    from grail.infrastructure.chain import GrailChainManager
    from grail.shared.schemas import BucketCredentials

    from .config import TrainingConfig


async def load_grpo_groups(
    window: int,
    cfg: TrainingConfig,
    trusted_miner_hotkeys: set[str] | None = None,
    credentials: BucketCredentials | Any = None,
    chain_manager: GrailChainManager | None = None,
    uid_by_hotkey: dict[str, int] | None = None,
) -> list[GRPOGroup]:
    """Load validated GRPO groups directly from trusted miners.

    Args:
        window: Training window number
        cfg: Training configuration
        trusted_miner_hotkeys: Set of trusted miner hotkeys to load from
        credentials: R2 credentials for bucket access
        chain_manager: Chain manager for miner bucket discovery
        uid_by_hotkey: Mapping of hotkey to UID for readable logging

    Returns:
        List of valid GRPO groups
    """
    return await _load_grpo_groups(
        window,
        cfg.group_adv_sum_tolerance,
        trusted_miner_hotkeys,
        credentials,
        chain_manager,
        uid_by_hotkey,
    )
