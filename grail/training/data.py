"""GRPO training data structures and loaders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..infrastructure.miner_data import fetch_multiple_miners_data
from ..shared.constants import ROLLOUTS_PER_PROBLEM

if TYPE_CHECKING:
    from ..infrastructure.chain import GrailChainManager
    from ..shared.schemas import BucketCredentials

logger = logging.getLogger(__name__)


@dataclass
class GRPORollout:
    """Single rollout from a GRPO group."""

    tokens: list[int]
    prompt_length: int
    completion_length: int
    advantage: float
    reward: float
    success: bool
    nonce: int
    rollout_group: str
    token_logprobs: list[float] | None = None


@dataclass
class GRPOGroup:
    """Collection of rollouts associated with one SAT problem."""

    group_id: str
    rollouts: list[GRPORollout]

    def is_valid(self, advantage_tolerance: float) -> bool:
        """Validate group size and zero-sum advantage condition."""

        if len(self.rollouts) != ROLLOUTS_PER_PROBLEM:
            return False
        advantage_sum = sum(r.advantage for r in self.rollouts)
        return abs(advantage_sum) < advantage_tolerance


async def load_grpo_groups(
    window: int,
    advantage_tolerance: float,
    trusted_miner_hotkeys: set[str] | None = None,
    credentials: BucketCredentials | Any = None,
    chain_manager: GrailChainManager | None = None,
    uid_by_hotkey: dict[str, int] | None = None,
) -> list[GRPOGroup]:
    """Load and validate GRPO groups directly from trusted miners.

    Args:
        window: Training window number
        advantage_tolerance: Maximum allowed sum of advantages in a group
        trusted_miner_hotkeys: Set of trusted miner hotkeys to load from
        credentials: R2 credentials for bucket access
        chain_manager: Chain manager for miner bucket discovery
        uid_by_hotkey: Mapping of hotkey to UID for readable logging

    Returns:
        List of valid GRPO groups
    """
    # Require trusted miners and credentials for direct miner fetching
    if not trusted_miner_hotkeys:
        logger.warning(
            "No trusted miners for window %s; skipping data load",
            window,
        )
        return []

    if credentials is None:
        logger.error("Credentials required for loading miner data")
        return []

    # Build UID list for logging
    trusted_uids = []
    if uid_by_hotkey:
        trusted_uids = sorted(
            [uid_by_hotkey[hk] for hk in trusted_miner_hotkeys if hk in uid_by_hotkey]
        )

    # Fetch window data from all trusted miners in parallel
    logger.info(
        "Fetching data from %d trusted miners (UIDs=%s) for window %s",
        len(trusted_miner_hotkeys),
        trusted_uids if trusted_uids else "N/A",
        window,
    )

    miner_data = await fetch_multiple_miners_data(
        miner_hotkeys=trusted_miner_hotkeys,
        window=window,
        credentials=credentials,
        chain_manager=chain_manager,
        max_concurrent=10,
    )

    if not miner_data:
        logger.warning(
            "No data fetched from any trusted miner for window %s",
            window,
        )
        return []

    # Extract all rollouts from all miners
    raw_rollouts = []
    for miner_hotkey, window_data in miner_data.items():
        miner_uid = uid_by_hotkey.get(miner_hotkey) if uid_by_hotkey else None
        miner_ident = f"UID {miner_uid}" if miner_uid is not None else miner_hotkey[:12]

        if not isinstance(window_data, dict):
            logger.debug(
                "Invalid window data format from miner %s",
                miner_ident,
            )
            continue

        inferences = window_data.get("inferences", [])
        if not isinstance(inferences, list):
            logger.debug(
                "Invalid inferences format from miner %s",
                miner_ident,
            )
            continue

        # Tag each rollout with the miner hotkey
        for rollout in inferences:
            if isinstance(rollout, dict):
                rollout["hotkey"] = miner_hotkey
                raw_rollouts.append(rollout)

    logger.info(
        "Loaded %d raw rollouts from %d miners for window %s",
        len(raw_rollouts),
        len(miner_data),
        window,
    )

    grouped: dict[str, list[GRPORollout]] = {}
    for rollout_dict in raw_rollouts:
        group_id = str(rollout_dict.get("rollout_group", ""))
        if not group_id:
            continue

        commit = rollout_dict.get("commit", {})
        rollout_meta = commit.get("rollout", {})

        try:
            grouped.setdefault(group_id, []).append(
                GRPORollout(
                    tokens=list(commit.get("tokens", [])),
                    prompt_length=int(rollout_meta.get("prompt_length", 0)),
                    completion_length=int(rollout_meta.get("completion_length", 0) or 0),
                    advantage=float(rollout_meta.get("advantage", 0.0)),
                    reward=float(rollout_meta.get("total_reward", 0.0)),
                    success=bool(rollout_meta.get("success", False)),
                    nonce=int(rollout_dict.get("nonce", 0)),
                    rollout_group=group_id,
                    token_logprobs=(
                        list(rollout_meta.get("token_logprobs", []))
                        if isinstance(
                            rollout_meta.get("token_logprobs", None),
                            list,
                        )
                        else None
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to parse rollout for group %s: %s",
                group_id,
                exc,
            )

    groups: list[GRPOGroup] = [
        GRPOGroup(group_id, rollouts) for group_id, rollouts in grouped.items()
    ]

    valid_groups = [group for group in groups if group.is_valid(advantage_tolerance)]
    invalid_count = len(groups) - len(valid_groups)
    if invalid_count > 0:
        logger.warning(
            "Filtered out %s invalid GRPO groups for window %s",
            invalid_count,
            window,
        )

    logger.info(
        "Loaded %s valid GRPO groups for window %s",
        len(valid_groups),
        window,
    )
    return valid_groups
