"""Tests for ValidationService._publish_trust_list (validator side).

Covers:
- Correct JSON schema and per-window key
- Failure exclusion reflected in eligible_hotkeys
- Upload failure is non-fatal
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grail.shared.constants import TRUST_LIST_KEY_PREFIX, TRUST_LIST_VERSION


def _make_service():
    """Build a minimal ValidationService-like object for testing _publish_trust_list."""
    from grail.validation.service import ValidationService

    # Minimal construction — only fields used by _publish_trust_list
    svc = object.__new__(ValidationService)
    svc._wallet = MagicMock()
    svc._wallet.hotkey.ss58_address = "5FakeValidatorHotkey"
    svc._credentials = MagicMock()
    svc._failure_counts = {}
    return svc


class TestPublishTrustList:
    @pytest.mark.asyncio
    async def test_publishes_correct_json(self):
        svc = _make_service()
        svc._failure_counts = {"bad_miner": 2}

        active = ["good_miner_1", "good_miner_2", "bad_miner"]
        window = 9000

        with patch(
            "grail.validation.service.upload_file_chunked",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_upload:
            await svc._publish_trust_list(window, active)

        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # Verify key
        key = call_args.args[0] if call_args.args else call_args[0][0]
        assert key == f"{TRUST_LIST_KEY_PREFIX}{window}.json"

        # Verify data payload
        data_bytes = call_args.args[1] if len(call_args.args) > 1 else call_args[0][1]
        payload = json.loads(data_bytes.decode())

        assert payload["version"] == TRUST_LIST_VERSION
        assert payload["window"] == window
        assert payload["validator_hotkey"] == "5FakeValidatorHotkey"
        assert set(payload["eligible_hotkeys"]) == {"good_miner_1", "good_miner_2"}
        assert payload["active_count"] == 3
        assert payload["excluded_failure_count"] == 1

        # Verify write credentials used
        assert call_args.kwargs.get("use_write") is True
        assert call_args.kwargs.get("credentials") is svc._credentials

    @pytest.mark.asyncio
    async def test_no_failures_all_eligible(self):
        svc = _make_service()
        svc._failure_counts = {}

        active = ["miner_a", "miner_b"]
        window = 1200

        with patch(
            "grail.validation.service.upload_file_chunked",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_upload:
            await svc._publish_trust_list(window, active)

        data_bytes = mock_upload.call_args.args[1]
        payload = json.loads(data_bytes.decode())

        assert set(payload["eligible_hotkeys"]) == {"miner_a", "miner_b"}
        assert payload["excluded_failure_count"] == 0

    @pytest.mark.asyncio
    async def test_upload_failure_logs_warning(self):
        svc = _make_service()

        with patch(
            "grail.validation.service.upload_file_chunked",
            new_callable=AsyncMock,
            return_value=False,
        ):
            # Should not raise
            await svc._publish_trust_list(500, ["miner_a"])
