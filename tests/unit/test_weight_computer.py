"""Unit tests for WeightComputer scoring logic."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from grail.scoring.weights import WeightComputer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BURN_UID = 0
BURN_PCT = 80.0
# Use exponent=1.0 where possible to keep arithmetic simple
DEFAULTS = {
    "rolling_windows": 1,
    "window_length": 30,
    "superlinear_exponent": 1.0,
    "burn_uid": BURN_UID,
    "burn_percentage": BURN_PCT,
}


def _make_counts(
    data: dict[str, int],
    window: int = 0,
) -> defaultdict[str, defaultdict[int, dict[str, int]]]:
    """Build inference_counts: hotkey -> {window: {estimated_unique: N}}."""
    counts: defaultdict[str, defaultdict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for hk, unique in data.items():
        counts[hk][window] = {"estimated_unique": unique}
    return counts


def _compute(
    hotkeys: list[str],
    uids: list[int],
    unique_per_miner: dict[str, int],
    availability: dict[str, int],
    *,
    cap_enabled: bool = True,
    superlinear_exponent: float = 1.0,
    window: int = 0,
) -> list[float]:
    wc = WeightComputer(
        **{**DEFAULTS, "cap_enabled": cap_enabled, "superlinear_exponent": superlinear_exponent}
    )
    weights, _ = wc.compute_weights(
        meta_hotkeys=hotkeys,
        meta_uids=uids,
        inference_counts=_make_counts(unique_per_miner, window),
        target_window=window,
        availability_counts=availability,
    )
    return weights


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightsAlwaysSumToOne:
    """Every valid output must be a proper distribution."""

    @pytest.mark.parametrize("cap_enabled", [True, False])
    def test_single_miner(self, cap_enabled: bool) -> None:
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        w = _compute(hotkeys, uids, {"m1": 5000}, {"m1": 1}, cap_enabled=cap_enabled)
        assert math.isclose(sum(w), 1.0, abs_tol=1e-9)

    @pytest.mark.parametrize("cap_enabled", [True, False])
    def test_two_miners(self, cap_enabled: bool) -> None:
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 3000, "m2": 6000},
            {"m1": 1, "m2": 1},
            cap_enabled=cap_enabled,
        )
        assert math.isclose(sum(w), 1.0, abs_tol=1e-9)


class TestCapBehavior:
    """Verify capping vs uncapping rollout scores."""

    def test_cap_limits_score(self) -> None:
        """A miner producing 2x the cap should score the same as exactly at cap."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        at_cap = _compute(hotkeys, uids, {"m1": 61440}, {"m1": 1}, cap_enabled=True)
        over_cap = _compute(hotkeys, uids, {"m1": 122880}, {"m1": 1}, cap_enabled=True)
        assert math.isclose(at_cap[1], over_cap[1], abs_tol=1e-9)

    def test_uncapped_rewards_beyond_cap(self) -> None:
        """Without cap, producing more rollouts should yield more weight."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 61440, "m2": 122880},
            {"m1": 1, "m2": 1},
            cap_enabled=False,
        )
        # m2 produced 2x -> gets 2x miner weight (with exponent=1)
        assert w[2] > w[1]
        assert math.isclose(w[2], w[1] * 2, rel_tol=1e-9)

    def test_capped_equal_at_cap(self) -> None:
        """With cap, two miners both above cap get equal weight."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 61440, "m2": 200000},
            {"m1": 1, "m2": 1},
            cap_enabled=True,
        )
        assert math.isclose(w[1], w[2], abs_tol=1e-9)

    def test_uncapped_over_cap_beats_at_cap(self) -> None:
        """Without cap, a miner above the old cap outweighs one at the cap."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 61440, "m2": 200000},
            {"m1": 1, "m2": 1},
            cap_enabled=False,
        )
        assert w[2] > w[1]

    def test_uncapped_no_score_clamp(self) -> None:
        """Without cap, a single miner over the old cap still gets full miner share."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        w_at = _compute(hotkeys, uids, {"m1": 61440}, {"m1": 1}, cap_enabled=False)
        w_over = _compute(hotkeys, uids, {"m1": 200000}, {"m1": 1}, cap_enabled=False)
        # Both are the sole miner so both get the same 20% miner share
        assert math.isclose(w_at[1], w_over[1], abs_tol=1e-9)
        assert math.isclose(w_at[1], 0.20, abs_tol=1e-9)


class TestBurnMechanism:
    """Verify burn UID gets the right share."""

    def test_burn_base_percentage_when_uncapped(self) -> None:
        """With cap disabled, burn should get exactly its configured percentage."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        w = _compute(hotkeys, uids, {"m1": 5000}, {"m1": 1}, cap_enabled=False)
        # total_cap_relative=1.0 when uncapped, so effective_miner = remaining * 1.0
        # burn gets exactly burn_fraction = 80%
        assert math.isclose(w[0], 0.80, abs_tol=1e-9)
        assert math.isclose(w[1], 0.20, abs_tol=1e-9)

    def test_underproduction_increases_burn_when_capped(self) -> None:
        """With cap enabled and low output, burn gets MORE than base percentage."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        # 6144 = 10% of cap -> total_cap_relative ~ 0.1
        w = _compute(hotkeys, uids, {"m1": 6144}, {"m1": 1}, cap_enabled=True)
        # Miner gets remaining_fraction * total_cap_relative = 0.2 * 0.1 = 0.02
        assert math.isclose(w[1], 0.02, abs_tol=1e-9)
        assert w[0] > 0.80  # Burn absorbs underproduction

    def test_no_miners_all_to_burn(self) -> None:
        """When no miner has output, burn UID gets 100%."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        w = _compute(hotkeys, uids, {}, {"m1": 0})
        assert math.isclose(w[0], 1.0, abs_tol=1e-9)
        assert w[1] == 0.0

    def test_uncapped_no_underproduction_burn(self) -> None:
        """With cap disabled, multiple miners still leave burn at exactly base percentage."""
        hotkeys = ["burn", "m1", "m2", "m3"]
        uids = [0, 1, 2, 3]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 100, "m2": 200, "m3": 300},
            {"m1": 1, "m2": 1, "m3": 1},
            cap_enabled=False,
        )
        assert math.isclose(w[0], 0.80, abs_tol=1e-9)

    @pytest.mark.parametrize("cap_enabled", [True, False])
    def test_no_output_all_to_burn(self, cap_enabled: bool) -> None:
        """Zero output sends 100% to burn regardless of cap setting."""
        hotkeys = ["burn", "m1"]
        uids = [0, 1]
        w = _compute(hotkeys, uids, {}, {"m1": 0}, cap_enabled=cap_enabled)
        assert math.isclose(w[0], 1.0, abs_tol=1e-9)


class TestGating:
    """Miners should be gated to zero for inactivity, failures, or no checks."""

    def test_inactive_miner_gets_zero(self) -> None:
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 5000},
            {"m1": 1, "m2": 0},  # m2 inactive
        )
        assert w[2] == 0.0
        assert w[1] > 0.0

    def test_failed_miner_gets_zero(self) -> None:
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        counts: defaultdict[str, defaultdict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        counts["m1"][0] = {"estimated_unique": 5000}
        counts["m2"][0] = {"estimated_unique": 5000, "had_failure": 1}

        wc = WeightComputer(**{**DEFAULTS, "cap_enabled": True})
        w, _ = wc.compute_weights(
            meta_hotkeys=hotkeys,
            meta_uids=uids,
            inference_counts=counts,
            target_window=0,
            availability_counts={"m1": 1, "m2": 1},
        )
        assert w[2] == 0.0
        assert w[1] > 0.0

    def test_unchecked_miner_gets_zero(self) -> None:
        """Miner active but with no inference_counts entry gets zero."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        # Only m1 has inference data; m2 is active but unchecked
        w = _compute(
            hotkeys,
            uids,
            {"m1": 5000},
            {"m1": 1, "m2": 1},
        )
        assert w[2] == 0.0


class TestSuperlinearExponent:
    """Superlinear exponent should amplify score differences."""

    def test_exponent_amplifies_gap(self) -> None:
        """With exponent > 1, a 2x output gap yields > 2x weight gap."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]
        w = _compute(
            hotkeys,
            uids,
            {"m1": 1000, "m2": 2000},
            {"m1": 1, "m2": 1},
            cap_enabled=False,
            superlinear_exponent=2.0,
        )
        # m2 has 2x rollouts -> raw score ratio = 4x with exponent 2
        # Weights are proportional: m2/(m1+m2) = 4/5, m1/(m1+m2) = 1/5
        miner_ratio = w[2] / w[1] if w[1] > 0 else float("inf")
        assert math.isclose(miner_ratio, 4.0, rel_tol=1e-6)


class TestExtrapolation:
    """Miners checked fewer times should be extrapolated fairly."""

    def test_extrapolation_equalizes(self) -> None:
        """Two miners with same per-window output but different check counts
        should get equal weight."""
        hotkeys = ["burn", "m1", "m2"]
        uids = [0, 1, 2]

        wc = WeightComputer(**{**DEFAULTS, "cap_enabled": False, "rolling_windows": 3})
        counts: defaultdict[str, defaultdict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # m1 checked in 2 windows, 1000 each -> total_checked=2000, active=3, extrapolated=3000
        counts["m1"][0] = {"estimated_unique": 1000}
        counts["m1"][30] = {"estimated_unique": 1000}
        # m2 checked in 1 window, 1000 -> total_checked=1000, active=3, extrapolated=3000
        counts["m2"][0] = {"estimated_unique": 1000}

        w, _ = wc.compute_weights(
            meta_hotkeys=hotkeys,
            meta_uids=uids,
            inference_counts=counts,
            target_window=60,
            availability_counts={"m1": 3, "m2": 3},
        )
        assert math.isclose(w[1], w[2], rel_tol=1e-9)
