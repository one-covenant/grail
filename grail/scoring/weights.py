"""Weight computation from miner metrics."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from ..shared.constants import UNIQUE_ROLLOUTS_CAP

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
        availability_counts: dict[str, int],
    ) -> tuple[list[float], list[tuple[str, float]]]:
        """Compute normalized weights over rolling window.

        Args:
            meta_hotkeys: Hotkeys in metagraph order
            meta_uids: UIDs in metagraph order
            inference_counts: hotkey -> window -> metrics dict
            target_window: Current window
            availability_counts: Count of windows each miner was active (had file)

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
            # Count windows where miner was active (submitted file)
            windows_active = availability_counts.get(hotkey, 0)

            # Gate: Must be active in at least 1 window
            if windows_active == 0:
                raw_scores.append(0.0)
                continue

            # Gate: Check for any failures in recent windows
            had_failure = False
            for w in recent_windows:
                metrics = inference_counts[hotkey].get(w, {})
                if metrics.get("had_failure", 0) == 1:
                    had_failure = True
                    break

            if had_failure:
                raw_scores.append(0.0)
                continue

            # Count windows where miner was actually checked (implicit: metrics exist)
            # If metrics exist for a window, the miner was selected and validated
            checked_windows = [w for w in recent_windows if w in inference_counts[hotkey]]
            windows_checked = len(checked_windows)

            # Gate: Must have been checked at least once to extrapolate
            if windows_checked == 0:
                raw_scores.append(0.0)
                continue

            # Sum estimated_unique over checked windows only
            total_unique_checked = sum(
                inference_counts[hotkey][w].get("estimated_unique", 0) for w in checked_windows
            )

            # Extrapolate to windows_active (fair normalization)
            # This ensures miners checked 2x vs 3x get equal weights for equal performance
            extrapolation_factor = windows_active / windows_checked
            total_unique_extrapolated = total_unique_checked * extrapolation_factor

            # Cap-proportional scoring: reward based on fraction of cap reached
            capped_unique = min(total_unique_extrapolated, UNIQUE_ROLLOUTS_CAP)

            # Apply superlinear scoring
            base_score = max(0.0, float(capped_unique))
            superlinear_score = base_score**self.superlinear_exponent
            raw_scores.append(superlinear_score)

        # Normalize against theoretical max (cap^exponent)
        # This ensures miners are rewarded proportionally to cap achievement
        max_score = UNIQUE_ROLLOUTS_CAP**self.superlinear_exponent
        cap_relative = [score / max_score for score in raw_scores]
        total_cap_relative = math.fsum(cap_relative)

        if total_cap_relative > 1.0:
            # Multiple strong miners: share proportionally, capped at 1.0 total
            pre_burn_weights = [cr / total_cap_relative for cr in cap_relative]
        else:
            # Underproduction: miners get their cap-relative share, rest burns
            pre_burn_weights = cap_relative

        pre_burn_nonzero_indices = [i for i, w in enumerate(pre_burn_weights) if w > 0.0]

        # Apply burn mechanism
        weights = self._apply_burn(
            pre_burn_weights,
            pre_burn_nonzero_indices,
            meta_uids,
            total_cap_relative,
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
        total_cap_relative: float,
    ) -> list[float]:
        """Apply burn mechanism to allocate percentage to burn UID.

        When miners underperform (total_cap_relative < 1.0), the "missing"
        weight goes to burn rather than being redistributed to miners.
        """
        if self.burn_uid is None or self.burn_percentage <= 0:
            raise ValueError("GRAIL_BURN_UID and GRAIL_BURN_PERCENTAGE must be set")

        burn_uid = int(self.burn_uid)
        burn_pct = max(0.0, min(100.0, self.burn_percentage))
        burn_fraction = burn_pct / 100.0
        remaining_fraction = 1.0 - burn_fraction

        if burn_uid not in meta_uids:
            logger.warning(f"Burn UID {burn_uid} not in metagraph; burn disabled")
            return pre_burn_weights

        burn_index = meta_uids.index(burn_uid)

        if total_cap_relative <= 0.0:
            # No signal: allocate 100% to burn UID
            weights = [0.0] * len(pre_burn_weights)
            weights[burn_index] = 1.0
            return weights

        # Compute effective miner allocation (capped by remaining_fraction)
        # If total_cap_relative < 1.0, miners get less than remaining_fraction
        effective_miner_fraction = min(remaining_fraction, remaining_fraction * total_cap_relative)

        # Scale pre_burn_weights to effective_miner_fraction
        pre_burn_sum = math.fsum(pre_burn_weights)
        if pre_burn_sum > 0:
            scale_factor = effective_miner_fraction / pre_burn_sum
            weights = [
                w * scale_factor if i != burn_index else 0.0 for i, w in enumerate(pre_burn_weights)
            ]
        else:
            weights = [0.0] * len(pre_burn_weights)

        # Burn gets base burn_fraction + any unallocated miner fraction
        weights[burn_index] = 1.0 - math.fsum(w for i, w in enumerate(weights) if i != burn_index)

        logger.info(f"🔥 Burn: {weights[burn_index] * 100:.1f}% to UID {burn_uid}")
        return weights
