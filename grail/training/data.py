"""GRPO training data structures and loaders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..infrastructure.comms import get_valid_rollouts
from ..shared.constants import ROLLOUTS_PER_PROBLEM

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
) -> list[GRPOGroup]:
    """Load and validate GRPO groups for a training window."""

    rollouts_data: Any = await get_valid_rollouts(window)
    if not rollouts_data:
        logger.warning("No valid rollouts found for window %s", window)
        return []

    if not isinstance(rollouts_data, dict):
        logger.warning("Invalid rollouts format for window %s", window)
        return []

    raw_rollouts = rollouts_data.get("rollouts", [])
    logger.info("Loaded %s raw rollouts for window %s", len(raw_rollouts), window)

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
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse rollout for group %s: %s", group_id, exc)

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

    logger.info("Loaded %s valid GRPO groups for window %s", len(valid_groups), window)
    return valid_groups
