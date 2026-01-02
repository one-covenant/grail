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

from ..shared.constants import GRAIL_BURN_UID

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
