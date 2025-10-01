"""Weight computation from miner metrics."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class WeightComputer:
    """Computes normalized weights from rolling window of miner metrics.

    Extracted from validate.py _compute_weights function.
    """

    def __init__(
        self,
        rolling_windows: int,
        window_length: int,
        superlinear_exponent: float,
        burn_uid: int | None,
        burn_percentage: float,
    ):
        self.rolling_windows = rolling_windows
        self.window_length = window_length
        self.superlinear_exponent = superlinear_exponent
        self.burn_uid = burn_uid
        self.burn_percentage = burn_percentage

    def compute_weights(
        self,
        meta_hotkeys: list[str],
        meta_uids: list[int],
        inference_counts: defaultdict[str, defaultdict[int, dict[str, int]]],
        target_window: int,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        """Compute normalized weights over rolling window.

        Args:
            meta_hotkeys: Hotkeys in metagraph order
            meta_uids: UIDs in metagraph order
            inference_counts: hotkey -> window -> metrics dict
            target_window: Current window

        Returns:
            (weights, non_zero_weights)
            weights: Normalized floats aligned to meta_hotkeys
            non_zero_weights: (hotkey, weight) pairs where weight > 0
        """
        # Calculate recent window range
        recent_windows = range(
            max(0, target_window - (self.rolling_windows - 1) * self.window_length),
            target_window + 1,
            self.window_length,
        )

        # Compute raw scores
        raw_scores = []
        for hotkey in meta_hotkeys:
            # Gate if any recent window had failure
            had_failure = False
            for w in recent_windows:
                metrics = inference_counts[hotkey].get(w, {})
                if metrics.get("had_failure", 0) == 1:
                    had_failure = True
                    break

            if had_failure:
                raw_scores.append(0.0)
                continue

            # Aggregate unique rollouts over rolling window
            total_unique = sum(
                inference_counts[hotkey].get(w, {}).get("estimated_unique", 0)
                for w in recent_windows
            )

            # Superlinear scoring (unbounded unique score)
            base_score = max(0.0, float(total_unique))
            superlinear_score = base_score**self.superlinear_exponent
            raw_scores.append(superlinear_score)

        # Normalize
        denom = math.fsum(raw_scores)
        if denom > 0.0:
            pre_burn_weights = [score / denom for score in raw_scores]
        else:
            pre_burn_weights = [0.0] * len(meta_hotkeys)

        pre_burn_nonzero_indices = [i for i, w in enumerate(pre_burn_weights) if w > 0.0]

        # Apply burn mechanism
        weights = self._apply_burn(
            pre_burn_weights,
            pre_burn_nonzero_indices,
            meta_uids,
            denom,
        )

        # Compose non-zero list
        burn_index = meta_uids.index(self.burn_uid) if self.burn_uid in meta_uids else None
        allowed_indices = set(pre_burn_nonzero_indices)
        if burn_index is not None:
            allowed_indices.add(burn_index)

        non_zero_weights = [
            (meta_hotkeys[i], weights[i]) for i in allowed_indices if weights[i] > 0.0
        ]

        return weights, non_zero_weights

    def _apply_burn(
        self,
        pre_burn_weights: list[float],
        pre_burn_nonzero_indices: list[int],
        meta_uids: list[int],
        denom: float,
    ) -> list[float]:
        """Apply burn mechanism to allocate percentage to burn UID."""
        if self.burn_uid is None or self.burn_percentage <= 0:
            raise ValueError("GRAIL_BURN_UID and GRAIL_BURN_PERCENTAGE must be set")

        burn_uid = int(self.burn_uid)
        burn_pct = max(0.0, min(100.0, self.burn_percentage))
        burn_fraction = burn_pct / 100.0

        if burn_uid not in meta_uids:
            logger.warning(f"Burn UID {burn_uid} not in metagraph; burn disabled")
            return pre_burn_weights

        burn_index = meta_uids.index(burn_uid)

        if denom <= 0.0:
            # No signal: allocate 100% to burn UID
            weights = [0.0] * len(pre_burn_weights)
            weights[burn_index] = 1.0
            return weights

        # Scale non-burn weights to remaining fraction
        remaining_fraction = 1.0 - burn_fraction
        non_burn_sum = sum(w for i, w in enumerate(pre_burn_weights) if i != burn_index)

        if non_burn_sum > 0:
            scale_factor = remaining_fraction / non_burn_sum
            weights = [
                w * scale_factor if i != burn_index else 0 for i, w in enumerate(pre_burn_weights)
            ]
        else:
            weights = [0.0] * len(pre_burn_weights)

        # Set burn UID to exact fraction
        weights[burn_index] = burn_fraction

        logger.info(f"🔥 Burn: {burn_pct:.1f}% to UID {burn_uid}")
        return weights
