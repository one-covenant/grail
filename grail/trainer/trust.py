"""Miner trust scoring based on validator weights."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from ..shared.constants import GRAIL_BURN_UID

logger = logging.getLogger(__name__)


async def get_trusted_miner_hotkeys(
    metagraph: Any,
    min_aggregate_weight: float,
    min_trusted_miners: int,
    timeout: float,
    subtensor: Any | None = None,
) -> set[str]:
    """Get hotkeys of miners trusted by validators via stake-weighted aggregation.

    Args:
        metagraph: Bittensor metagraph with validator weights
        min_aggregate_weight: Minimum aggregate weight threshold (0-1)
        min_trusted_miners: Minimum number of trusted miners to select
        timeout: Maximum time to spend computing trust scores

    Returns:
        Set of trusted miner hotkeys, or empty set if no trusted miners available
    """
    try:
        return await asyncio.wait_for(
            _compute_trusted_hotkeys(
                metagraph, min_aggregate_weight, min_trusted_miners, subtensor
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Timeout computing trusted miners after %.1fs; skipping training",
            timeout,
        )
        return set()
    except Exception as exc:
        logger.error(
            "Failed to compute trusted miners: %s; skipping training",
            exc,
            exc_info=True,
        )
        return set()


async def _get_raw_weights(
    metagraph: Any, subtensor: Any | None = None
) -> dict[int, dict[int, float]]:
    """Get raw weights from subtensor, parsing the tuple format."""

    # NOTE: metagraph.W is empty on recent Bittensor builds; we must read weights
    #       directly from the subtensor API and normalize them here.
    try:
        active_subtensor = subtensor
        if active_subtensor is None:
            if hasattr(metagraph, "subtensor") and metagraph.subtensor is not None:
                active_subtensor = metagraph.subtensor
            else:
                logger.error("No subtensor available for weights query")
                return {}

        raw_weights = await active_subtensor.weights(metagraph.netuid)

        parsed_weights: dict[int, dict[int, float]] = {}
        for validator_uid, weight_list in raw_weights:
            parsed_weights[validator_uid] = {
                miner_uid: weight_u16 / 65535.0 for miner_uid, weight_u16 in weight_list
            }

        logger.debug("Parsed weights for %d validators from raw chain data", len(parsed_weights))
        return parsed_weights

    except Exception as exc:
        logger.error("Failed to get raw weights: %s", exc, exc_info=True)
        return {}


async def _compute_trusted_hotkeys(
    metagraph: Any,
    min_aggregate_weight: float,
    min_trusted_miners: int,
    subtensor: Any | None = None,
) -> set[str]:
    """Compute stake-weighted trust scores and filter to trusted miners.

    Always selects at least min_trusted_miners (top N by trust).
    """

    # Identify active validators (those with non-zero validator_trust)
    validator_uids = [
        uid for uid in range(len(metagraph.uids)) if metagraph.validator_trust[uid] > 0
    ]

    if not validator_uids:
        logger.warning("No active validators found; skipping training window")
        return set()

    # Calculate total validator stake for normalization
    total_validator_stake = sum(metagraph.S[uid] for uid in validator_uids)
    if total_validator_stake == 0:
        logger.warning("Total validator stake is zero; skipping training window")
        return set()

    # Get raw weights from subtensor (workaround for metagraph.W being empty)
    raw_weights = await _get_raw_weights(metagraph, subtensor)
    if not raw_weights:
        logger.warning("No raw weights available; skipping training window")
        return set()

    # Aggregate miner trust: stake-weighted sum of validator weights
    miner_trust: dict[int, float] = defaultdict(float)

    for v_uid in validator_uids:
        validator_stake_fraction = metagraph.S[v_uid] / total_validator_stake

        # Check if this validator has set weights
        if v_uid not in raw_weights:
            logger.debug("Validator %d has no weights set", v_uid)
            continue

        validator_weights = raw_weights[v_uid]
        logger.debug("Validator %d has weights for %d miners", v_uid, len(validator_weights))

        for m_uid, weight in validator_weights.items():
            if weight > 0:
                # Accumulate stake-weighted trust
                miner_trust[m_uid] += float(weight) * validator_stake_fraction

    if not miner_trust:
        logger.warning("No miner trust scores computed; skipping training window")
        return set()

    # Exclude burn UID from trust scores (never train on burner data)
    if GRAIL_BURN_UID in miner_trust:
        logger.debug("Excluding burn UID %d from trusted miners", GRAIL_BURN_UID)
        del miner_trust[GRAIL_BURN_UID]

    if not miner_trust:
        logger.warning("No miner trust scores after excluding burn UID; skipping training window")
        return set()

    # Sort miners by trust score (highest first)
    sorted_miners = sorted(miner_trust.items(), key=lambda x: x[1], reverse=True)

    # Select top N miners by trust score
    top_n_miners = sorted_miners[:min_trusted_miners]
    trusted_uids = {uid for uid, _ in top_n_miners}

    trusted_hotkeys = {metagraph.hotkeys[uid] for uid in trusted_uids}

    # Log with UIDs for readability
    trusted_uid_list = sorted(trusted_uids)
    logger.info(
        "Selected %d/%d trusted miners: UIDs=%s (validators: %d, stake: %.2f)",
        len(trusted_hotkeys),
        len(metagraph.hotkeys),
        trusted_uid_list,
        len(validator_uids),
        total_validator_stake,
    )

    return trusted_hotkeys
