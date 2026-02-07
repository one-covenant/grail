"""Tests for trust list reader functions (trainer side).

Covers:
- _find_highest_stake_validator: numpy-based validator selection
- get_trust_list_from_validator: R2 download, staleness, validation, fallback
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from grail.shared.constants import (
    TRUST_LIST_KEY_PREFIX,
    TRUST_LIST_MAX_STALENESS_WINDOWS,
    TRUST_LIST_VERSION,
    WINDOW_LENGTH,
)
from grail.trainer.trust import (
    _find_highest_stake_validator,
    get_trust_list_from_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metagraph(stakes: list[float], permits: list[bool]) -> SimpleNamespace:
    """Build a minimal metagraph-like object."""
    n = len(stakes)
    return SimpleNamespace(
        S=np.array(stakes, dtype=np.float64),
        validator_permit=np.array(permits, dtype=bool),
        hotkeys=[f"hotkey_{i}" for i in range(n)],
        uids=list(range(n)),
    )


# ---------------------------------------------------------------------------
# _find_highest_stake_validator
# ---------------------------------------------------------------------------

class TestFindHighestStakeValidator:
    def test_single_validator(self):
        meta = _make_metagraph(
            stakes=[0.0, 100.0, 50.0],
            permits=[False, True, False],
        )
        assert _find_highest_stake_validator(meta) == 1

    def test_multiple_validators_picks_highest_stake(self):
        meta = _make_metagraph(
            stakes=[10.0, 200.0, 300.0, 50.0],
            permits=[True, True, True, False],
        )
        assert _find_highest_stake_validator(meta) == 2

    def test_no_permits_returns_none(self):
        meta = _make_metagraph(
            stakes=[100.0, 200.0],
            permits=[False, False],
        )
        assert _find_highest_stake_validator(meta) is None

    def test_zero_stake_with_permit_returns_none(self):
        meta = _make_metagraph(
            stakes=[0.0, 0.0],
            permits=[True, True],
        )
        assert _find_highest_stake_validator(meta) is None


# ---------------------------------------------------------------------------
# get_trust_list_from_validator
# ---------------------------------------------------------------------------

def _trust_list_dict(window: int, eligible: list[str], version: int = TRUST_LIST_VERSION) -> dict:
    return {
        "version": version,
        "window": window,
        "timestamp": 1700000000.0,
        "validator_hotkey": "hotkey_1",
        "eligible_hotkeys": eligible,
        "active_count": len(eligible) + 1,
        "excluded_failure_count": 1,
    }


class TestGetTrustListFromValidator:
    @pytest.mark.asyncio
    async def test_valid_list_returns_set(self):
        meta = _make_metagraph(
            stakes=[0.0, 500.0, 10.0],
            permits=[False, True, False],
        )
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = MagicMock()

        target_window = 1000
        eligible = ["miner_a", "miner_b"]
        trust_data = _trust_list_dict(target_window, eligible)

        with patch("grail.trainer.trust.get_file", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = trust_data
            result = await get_trust_list_from_validator(meta, chain_manager, target_window)

        assert result == {"miner_a", "miner_b"}
        mock_get.assert_called_once_with(
            f"{TRUST_LIST_KEY_PREFIX}{target_window}.json",
            credentials=chain_manager.get_bucket.return_value,
        )

    @pytest.mark.asyncio
    async def test_missing_file_falls_back_to_older_window(self):
        meta = _make_metagraph(stakes=[0.0, 500.0], permits=[False, True])
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = MagicMock()

        target_window = 300
        older_window = target_window - WINDOW_LENGTH
        eligible = ["miner_x"]

        async def side_effect(key, credentials=None):
            if str(older_window) in key:
                return _trust_list_dict(older_window, eligible)
            return None

        with patch("grail.trainer.trust.get_file", new_callable=AsyncMock, side_effect=side_effect):
            result = await get_trust_list_from_validator(meta, chain_manager, target_window)

        assert result == {"miner_x"}

    @pytest.mark.asyncio
    async def test_all_windows_missing_returns_none(self):
        meta = _make_metagraph(stakes=[0.0, 500.0], permits=[False, True])
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = MagicMock()

        with patch("grail.trainer.trust.get_file", new_callable=AsyncMock, return_value=None):
            result = await get_trust_list_from_validator(meta, chain_manager, 3000)

        assert result is None

    @pytest.mark.asyncio
    async def test_wrong_version_returns_none(self):
        meta = _make_metagraph(stakes=[0.0, 500.0], permits=[False, True])
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = MagicMock()

        bad_data = _trust_list_dict(1000, ["miner_a"], version=999)

        with patch("grail.trainer.trust.get_file", new_callable=AsyncMock, return_value=bad_data):
            result = await get_trust_list_from_validator(meta, chain_manager, 1000)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_eligible_returns_none(self):
        meta = _make_metagraph(stakes=[0.0, 500.0], permits=[False, True])
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = MagicMock()

        empty_data = _trust_list_dict(1000, [])

        with patch("grail.trainer.trust.get_file", new_callable=AsyncMock, return_value=empty_data):
            result = await get_trust_list_from_validator(meta, chain_manager, 1000)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_validator_permits_returns_none(self):
        meta = _make_metagraph(stakes=[100.0, 200.0], permits=[False, False])
        chain_manager = MagicMock()

        result = await get_trust_list_from_validator(meta, chain_manager, 1000)

        assert result is None
        chain_manager.get_bucket.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_bucket_returns_none(self):
        meta = _make_metagraph(stakes=[0.0, 500.0], permits=[False, True])
        chain_manager = MagicMock()
        chain_manager.get_bucket.return_value = None

        result = await get_trust_list_from_validator(meta, chain_manager, 1000)

        assert result is None
