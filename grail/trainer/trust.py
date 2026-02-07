"""Miner trust scoring based on on-chain incentive (Yuma Consensus output).

This module selects trusted miners for training data by using the on-chain
incentive metric, which is the output of Bittensor's Yuma Consensus mechanism.

WHY ON-CHAIN INCENTIVE (vs raw validator weights):
==================================================

The previous approach computed trust by manually aggregating raw validator
weights with stake-weighting. This had a critical flaw: a single validator
with anomalous weights could dominate the trust ranking.

Example of the bug:
  - Validator 197 (10.5% stake) set weight=0.25 on UID 47
  - All other validators set weight=0.0014 on UID 73
  - Result: UID 47 selected despite having ZERO on-chain incentive
  - UID 73 (actual highest incentive miner) was ignored

WHY YUMA CONSENSUS IS BETTER:
=============================

Yuma Consensus is Bittensor's core mechanism for aggregating validator opinions
into a single, manipulation-resistant score. It provides:

1. STAKE-WEIGHTED VOTING: Validators vote proportionally to their stake,
   but with consensus requirements that prevent single-validator domination.

2. BOND MECHANISM: Validators form "bonds" with miners they consistently
   rate highly. This creates temporal smoothing and prevents sudden
   manipulation of scores.

3. CONSENSUS REQUIREMENT: Unlike simple stake-weighted averaging, Yuma
   penalizes validators who deviate from consensus. A single validator
   giving anomalously high weights gets diluted.

4. MANIPULATION RESISTANCE: To manipulate incentive, an attacker needs
   to either:
   - Control >50% of stake (expensive)
   - Convince multiple validators to collude (coordination cost)

   vs raw weights where ONE validator with 10% stake could dominate.

The on-chain incentive (metagraph.incentive) IS the output of Yuma Consensus.
It represents the network's agreed-upon quality ranking of miners. By using
it directly, we:

  - Leverage the consensus mechanism we're already paying for (validator emissions)
  - Avoid re-implementing a weaker version of consensus
  - Get manipulation resistance built into Bittensor's protocol
  - Simplify our code from ~170 lines to ~30 lines

TRADEOFF - UPDATE LAG:
======================

On-chain incentive updates every tempo (360 blocks ≈ 72 minutes) after
validators submit weights. This means:

  - Worst case: ~2.5 hours for a cheating miner's incentive to drop
  - During this window, trainer may use data from a miner being penalized

We accept this tradeoff because:
  1. Validators ARE the validation layer - duplicating their work is wasteful
  2. The chain's consensus is the source of truth for miner quality
  3. Training on one bad window won't catastrophically damage the model
  4. Trying to be faster than the chain adds complexity without clear benefit
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ..infrastructure.comms import get_file
from ..shared.constants import (
    GRAIL_BURN_UID,
    TRUST_LIST_KEY_PREFIX,
    TRUST_LIST_MAX_STALENESS_WINDOWS,
    TRUST_LIST_VERSION,
    WINDOW_LENGTH,
)

logger = logging.getLogger(__name__)


async def get_trusted_miner_hotkeys(
    metagraph: Any,
    min_trusted_miners: int,
    timeout: float,
) -> set[str]:
    """Get hotkeys of trusted miners using on-chain incentive (Yuma Consensus).

    This function selects the top miners by on-chain incentive, which is the
    output of Bittensor's Yuma Consensus mechanism. This is more robust than
    manually aggregating raw validator weights because Yuma includes consensus
    requirements that prevent single-validator manipulation.

    Args:
        metagraph: Bittensor metagraph with incentive data
        min_trusted_miners: Number of top miners to select
        timeout: Maximum time for the operation

    Returns:
        Set of trusted miner hotkeys, or empty set if no miners have incentive
    """
    try:
        return await asyncio.wait_for(
            _select_top_miners_by_incentive(metagraph, min_trusted_miners),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Timeout selecting trusted miners after %.1fs; skipping training",
            timeout,
        )
        return set()
    except Exception as exc:
        logger.error(
            "Failed to select trusted miners: %s; skipping training",
            exc,
            exc_info=True,
        )
        return set()


async def _select_top_miners_by_incentive(
    metagraph: Any,
    min_trusted_miners: int,
) -> set[str]:
    """Select top miners by on-chain incentive (Yuma Consensus output).

    The incentive metric is computed by Bittensor's Yuma Consensus, which
    aggregates all validator weights with stake-weighting AND consensus
    requirements. This makes it resistant to single-validator manipulation.
    """
    # Build list of (uid, incentive) for miners with non-zero incentive
    # Exclude the burn UID which receives most incentive but isn't a real miner
    miner_incentives: list[tuple[int, float]] = []

    for uid in range(len(metagraph.uids)):
        if uid == GRAIL_BURN_UID:
            continue

        incentive = float(metagraph.incentive[uid])
        if incentive > 0:
            miner_incentives.append((uid, incentive))

    if not miner_incentives:
        logger.warning(
            "No miners with non-zero incentive (excluding burn UID %d); skipping training window",
            GRAIL_BURN_UID,
        )
        return set()

    # Sort by incentive descending (highest first)
    sorted_miners = sorted(miner_incentives, key=lambda x: x[1], reverse=True)

    # Select top N miners
    top_n = sorted_miners[:min_trusted_miners]
    trusted_uids = [uid for uid, _ in top_n]
    trusted_hotkeys = {metagraph.hotkeys[uid] for uid in trusted_uids}

    # Log selection details
    logger.info(
        "Selected %d/%d trusted miners by incentive: UIDs=%s",
        len(trusted_hotkeys),
        len(metagraph.uids),
        trusted_uids,
    )

    # Log incentive values for transparency
    for uid, incentive in top_n:
        logger.debug(
            "  UID %d: incentive=%.6f, hotkey=%s",
            uid,
            incentive,
            metagraph.hotkeys[uid][:16] + "...",
        )

    return trusted_hotkeys


def _find_highest_stake_validator(metagraph: Any) -> int | None:
    """Find the UID of the highest-stake validator with a permit.

    Uses numpy masking on metagraph.S (stake tensor) filtered by
    metagraph.validator_permit (boolean tensor).

    Args:
        metagraph: Bittensor metagraph instance

    Returns:
        UID of highest-stake validator, or None if no validators have permits
    """
    permits = np.array(metagraph.validator_permit)
    if not permits.any():
        return None

    stakes = np.where(permits, np.array(metagraph.S), -1.0)
    uid = int(np.argmax(stakes))

    if stakes[uid] <= 0:
        return None

    return uid


async def get_trust_list_from_validator(
    metagraph: Any,
    chain_manager: Any,
    target_window: int,
) -> set[str] | None:
    """Download the validator-published trust list from R2.

    Finds the highest-stake validator, resolves its R2 bucket via
    chain_manager, and downloads the per-window trust list JSON.
    Falls back to older windows (up to TRUST_LIST_MAX_STALENESS_WINDOWS)
    if the exact window key is missing.

    Args:
        metagraph: Bittensor metagraph instance
        chain_manager: GrailChainManager for bucket lookups
        target_window: The window the trainer wants data for

    Returns:
        Set of eligible hotkeys, or None if unavailable (caller should fallback)
    """
    # Find the highest-stake validator
    validator_uid = _find_highest_stake_validator(metagraph)
    if validator_uid is None:
        logger.warning("No validator with permit found for trust list lookup")
        return None

    # Get the validator's R2 bucket
    validator_bucket = chain_manager.get_bucket(validator_uid)
    if validator_bucket is None:
        logger.warning(
            "Validator UID %d has no R2 bucket committed on chain",
            validator_uid,
        )
        return None

    # Try the target window, then progressively older windows
    for offset in range(TRUST_LIST_MAX_STALENESS_WINDOWS + 1):
        check_window = target_window - offset * WINDOW_LENGTH
        if check_window < 0:
            break

        key = f"{TRUST_LIST_KEY_PREFIX}{check_window}.json"
        trust_data = await get_file(key, credentials=validator_bucket)

        if trust_data is None:
            continue

        # Validate version
        if trust_data.get("version") != TRUST_LIST_VERSION:
            logger.warning(
                "Trust list version mismatch: got %s, expected %d",
                trust_data.get("version"),
                TRUST_LIST_VERSION,
            )
            return None

        eligible = trust_data.get("eligible_hotkeys", [])
        list_window = trust_data.get("window", check_window)

        logger.info(
            "Loaded trust list from validator UID %d (window %d, %d eligible miners, offset=%d)",
            validator_uid,
            list_window,
            len(eligible),
            offset,
        )

        if not eligible:
            logger.warning("Trust list from validator is empty, falling back")
            return None

        return set(eligible)

    logger.info(
        "No trust list found for windows %d..%d from validator UID %d",
        target_window,
        target_window - TRUST_LIST_MAX_STALENESS_WINDOWS * WINDOW_LENGTH,
        validator_uid,
    )
    return None
