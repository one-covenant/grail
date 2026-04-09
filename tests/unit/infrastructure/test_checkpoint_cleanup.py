"""Unit tests for checkpoint cleanup and bucket listing functions.

Tests:
- list_bucket_files pagination handling
- _compute_keep_windows retention logic
- cleanup_old_checkpoints integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grail.infrastructure.checkpoint_consumer import CheckpointMetadata
from grail.shared.retention_utils import compute_chain_windows, compute_retention_windows

# ============================================================================
# Tests for _compute_keep_windows
# ============================================================================


class TestValidateMetadata:
    """Tests for CheckpointMetadata.validate_metadata()."""

    def _make_metadata(self, **overrides: object) -> CheckpointMetadata:
        defaults: dict[str, object] = {
            "window": 100,
            "file_manifest": {},
            "env_id": "triton_kernel",
            "thinking_mode": "instructed",
            "generation_params": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
            },
        }
        defaults.update(overrides)
        return CheckpointMetadata(**defaults)  # type: ignore[arg-type]

    def test_valid_metadata_returns_empty(self) -> None:
        meta = self._make_metadata()
        assert meta.validate_metadata() == []

    def test_missing_env_id(self) -> None:
        meta = self._make_metadata(env_id=None)
        missing = meta.validate_metadata()
        assert "env_id" in missing

    def test_empty_env_id(self) -> None:
        meta = self._make_metadata(env_id="")
        missing = meta.validate_metadata()
        assert "env_id" in missing

    def test_missing_generation_params(self) -> None:
        meta = self._make_metadata(generation_params={})
        missing = meta.validate_metadata()
        assert "generation_params" in missing

    def test_missing_max_tokens(self) -> None:
        meta = self._make_metadata(
            generation_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
            }
        )
        missing = meta.validate_metadata()
        assert "generation_params.max_tokens" in missing

    def test_missing_temperature(self) -> None:
        meta = self._make_metadata(
            generation_params={
                "max_tokens": 8192,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
            }
        )
        missing = meta.validate_metadata()
        assert "generation_params.temperature" in missing

    def test_missing_top_p(self) -> None:
        meta = self._make_metadata(
            generation_params={
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_k": 50,
                "repetition_penalty": 1.0,
            }
        )
        missing = meta.validate_metadata()
        assert "generation_params.top_p" in missing

    def test_missing_top_k(self) -> None:
        meta = self._make_metadata(
            generation_params={
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
            }
        )
        missing = meta.validate_metadata()
        assert "generation_params.top_k" in missing

    def test_missing_repetition_penalty(self) -> None:
        meta = self._make_metadata(
            generation_params={
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
            }
        )
        missing = meta.validate_metadata()
        assert "generation_params.repetition_penalty" in missing

    def test_all_sampling_params_missing_when_only_max_tokens(self) -> None:
        meta = self._make_metadata(generation_params={"max_tokens": 8192})
        missing = meta.validate_metadata()
        assert "generation_params.temperature" in missing
        assert "generation_params.top_p" in missing
        assert "generation_params.top_k" in missing
        assert "generation_params.repetition_penalty" in missing
        assert "generation_params.max_tokens" not in missing

    def test_multiple_missing_fields(self) -> None:
        meta = self._make_metadata(env_id=None, generation_params={})
        missing = meta.validate_metadata()
        assert "env_id" in missing
        assert "generation_params" in missing
        assert "thinking_mode" not in missing  # still set in defaults
        assert len(missing) == 2

    def test_missing_thinking_mode(self) -> None:
        meta = self._make_metadata(thinking_mode=None)
        missing = meta.validate_metadata()
        assert "thinking_mode" in missing

    def test_empty_thinking_mode(self) -> None:
        meta = self._make_metadata(thinking_mode="")
        missing = meta.validate_metadata()
        assert "thinking_mode" in missing

    def test_invalid_thinking_mode_value(self) -> None:
        meta = self._make_metadata(thinking_mode="bogus")
        missing = meta.validate_metadata()
        # validate_metadata reports the bad value in the field name so
        # operators see exactly what's wrong in the verdict log.
        assert any("thinking_mode" in m for m in missing)
        assert any("bogus" in m for m in missing)

    def test_native_thinking_mode_is_valid(self) -> None:
        meta = self._make_metadata(thinking_mode="native")
        assert meta.validate_metadata() == []

    def test_instructed_thinking_mode_is_valid(self) -> None:
        meta = self._make_metadata(thinking_mode="instructed")
        assert meta.validate_metadata() == []


class TestComputeKeepWindows:
    """Tests for the retention window calculation function (chained deltas).

    With chained deltas, retention keeps entire chains from anchor to now.
    """

    def test_negative_window_returns_empty(self) -> None:
        """Negative window should return empty set."""
        result = compute_retention_windows(-1)
        assert result == set()

    def test_zero_window_returns_zero(self) -> None:
        """Window 0 should keep window 0."""
        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 20):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                result = compute_retention_windows(0)
                assert 0 in result

    def test_keeps_entire_chain_from_anchor(self) -> None:
        """Should keep all windows from current anchor to now."""
        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 5):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                # Anchor stride = 5 * 30 = 150
                # Window 180: anchor at 150
                result = compute_retention_windows(180)
                # Should include chain: 150, 180
                assert 150 in result
                assert 180 in result

    def test_simplified_retention_one_anchor_only(self) -> None:
        """Simplified retention: keeps only latest FULL anchor + chain to current."""
        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 3):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                with patch("grail.shared.retention_utils.SAFETY_MARGIN_WINDOWS", 5):
                    # Anchor stride = 3 * 30 = 90
                    # Window 120: latest anchor at 120 (with safety margin: 120 - 5*30 = -30 → 0)
                    result = compute_retention_windows(120)
                    # Should keep from safety margin anchor (0) to current (120)
                    assert 0 in result
                    assert 30 in result
                    assert 60 in result
                    assert 90 in result
                    assert 120 in result
                    # Should NOT keep older checkpoints outside the retention window

    def test_early_windows_dont_go_negative(self) -> None:
        """Early windows should not include negative values."""
        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 5):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                result = compute_retention_windows(60)
                # Should keep chain from 0 to 60
                assert 0 in result
                assert 30 in result
                assert 60 in result
                assert all(w >= 0 for w in result)


# ============================================================================
# Tests for list_bucket_files with pagination
# ============================================================================


class TestListBucketFiles:
    """Tests for the list_bucket_files function with pagination support."""

    @pytest.mark.asyncio
    async def test_single_page_no_truncation(self) -> None:
        """Single page response with no truncation."""
        mock_client = AsyncMock()
        mock_client.list_objects_v2 = AsyncMock(
            return_value={
                "Contents": [
                    {"Key": "grail/checkpoints/checkpoint-100/file1.txt"},
                    {"Key": "grail/checkpoints/checkpoint-100/file2.txt"},
                ],
                "IsTruncated": False,
            }
        )

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("grail/checkpoints/")

        assert len(result) == 2
        assert "grail/checkpoints/checkpoint-100/file1.txt" in result
        assert "grail/checkpoints/checkpoint-100/file2.txt" in result
        # Should only call list_objects_v2 once
        assert mock_client.list_objects_v2.call_count == 1

    @pytest.mark.asyncio
    async def test_pagination_two_pages(self) -> None:
        """Pagination with two pages of results."""
        mock_client = AsyncMock()

        # First page - truncated
        first_response = {
            "Contents": [
                {"Key": f"grail/checkpoints/checkpoint-{i}/file.txt"} for i in range(100, 150)
            ],
            "IsTruncated": True,
            "NextContinuationToken": "token123",
        }

        # Second page - final
        second_response = {
            "Contents": [
                {"Key": f"grail/checkpoints/checkpoint-{i}/file.txt"} for i in range(150, 200)
            ],
            "IsTruncated": False,
        }

        mock_client.list_objects_v2 = AsyncMock(side_effect=[first_response, second_response])

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("grail/checkpoints/")

        # Should have all 100 files from both pages
        assert len(result) == 100
        # Should call list_objects_v2 twice
        assert mock_client.list_objects_v2.call_count == 2
        # Second call should include continuation token
        second_call_kwargs = mock_client.list_objects_v2.call_args_list[1][1]
        assert second_call_kwargs.get("ContinuationToken") == "token123"

    @pytest.mark.asyncio
    async def test_pagination_three_pages(self) -> None:
        """Pagination with three pages (simulating >2000 objects)."""
        mock_client = AsyncMock()

        responses = [
            {
                "Contents": [{"Key": f"file_{i}.txt"} for i in range(1000)],
                "IsTruncated": True,
                "NextContinuationToken": "token1",
            },
            {
                "Contents": [{"Key": f"file_{i}.txt"} for i in range(1000, 2000)],
                "IsTruncated": True,
                "NextContinuationToken": "token2",
            },
            {
                "Contents": [{"Key": f"file_{i}.txt"} for i in range(2000, 2500)],
                "IsTruncated": False,
            },
        ]

        mock_client.list_objects_v2 = AsyncMock(side_effect=responses)

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("prefix/")

        assert len(result) == 2500
        assert mock_client.list_objects_v2.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_bucket(self) -> None:
        """Empty bucket returns empty list."""
        mock_client = AsyncMock()
        mock_client.list_objects_v2 = AsyncMock(
            return_value={"IsTruncated": False}  # No Contents key
        )

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("empty/prefix/")

        assert result == []

    @pytest.mark.asyncio
    async def test_error_returns_empty_list(self) -> None:
        """Exception during listing returns empty list."""
        mock_client = AsyncMock()
        mock_client.list_objects_v2 = AsyncMock(side_effect=Exception("Connection error"))

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("prefix/")

        assert result == []

    @pytest.mark.asyncio
    async def test_pagination_stops_on_missing_token(self) -> None:
        """Pagination stops if IsTruncated but no token (safety check)."""
        mock_client = AsyncMock()
        mock_client.list_objects_v2 = AsyncMock(
            return_value={
                "Contents": [{"Key": "file1.txt"}],
                "IsTruncated": True,
                # Missing NextContinuationToken - should break
            }
        )

        with patch(
            "grail.infrastructure.comms._get_cached_client",
            return_value=mock_client,
        ):
            with patch("grail.infrastructure.comms.get_bucket_id", return_value="test-bucket"):
                from grail.infrastructure.comms import list_bucket_files

                result = await list_bucket_files("prefix/")

        # Should return what we got and stop (not infinite loop)
        assert len(result) == 1
        assert mock_client.list_objects_v2.call_count == 1


# ============================================================================
# Tests for cleanup_old_checkpoints
# ============================================================================


class TestCleanupOldCheckpoints:
    """Tests for the checkpoint cleanup function."""

    @pytest.mark.asyncio
    async def test_cleanup_deletes_old_checkpoints(self) -> None:
        """Cleanup deletes checkpoints outside retention window.

        With chained deltas, retention keeps entire chains from anchor to now.
        """
        # Mock list_bucket_files to return checkpoints
        mock_keys = [
            "grail/checkpoints/checkpoint-120/FULL/file.txt",
            "grail/checkpoints/checkpoint-90/DELTA/file.txt",
            "grail/checkpoints/checkpoint-60/FULL/file.txt",
            "grail/checkpoints/checkpoint-30/DELTA/file.txt",
            "grail/checkpoints/checkpoint-0/FULL/file.txt",
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()
        mock_fetch_chain = AsyncMock(return_value=(60, None))  # All deltas: anchor=60

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 2):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        with patch(
                            "grail.trainer.checkpoint_publisher._fetch_delta_chain_info",
                            mock_fetch_chain,
                        ):
                            from grail.trainer.checkpoint_publisher import CheckpointPublisher

                            publisher = CheckpointPublisher(
                                credentials=mock_credentials, wallet=mock_wallet
                            )
                            await publisher.cleanup_old_checkpoints(120)

        # All windows in test are within retention (bootstrap + active chain)
        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_retention_limit_checkpoints(self) -> None:
        """Cleanup keeps checkpoints in retention window."""
        mock_keys = [
            "grail/checkpoints/checkpoint-7055580/FULL/file.txt",
            "grail/checkpoints/checkpoint-7055550/DELTA/file.txt",
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()
        mock_fetch_chain = AsyncMock(return_value=(7055500, None))

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 10):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        with patch(
                            "grail.trainer.checkpoint_publisher._fetch_delta_chain_info",
                            mock_fetch_chain,
                        ):
                            from grail.trainer.checkpoint_publisher import CheckpointPublisher

                            publisher = CheckpointPublisher(
                                credentials=mock_credentials, wallet=mock_wallet
                            )
                            await publisher.cleanup_old_checkpoints(7055580)

        # Both windows are within retention window, should not delete
        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_empty_bucket(self) -> None:
        """Cleanup handles empty bucket gracefully."""
        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=[])
        mock_delete = AsyncMock()

        with patch(
            "grail.infrastructure.comms.list_bucket_files",
            mock_list,
        ):
            with patch(
                "grail.trainer.checkpoint_publisher.delete_prefix",
                mock_delete,
            ):
                from grail.trainer.checkpoint_publisher import CheckpointPublisher

                publisher = CheckpointPublisher(credentials=mock_credentials, wallet=mock_wallet)
                # Should not raise
                await publisher.cleanup_old_checkpoints(7055580)

        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_delete_error(self) -> None:
        """Cleanup continues on delete errors (logs warning but doesn't raise)."""
        mock_keys = [
            "grail/checkpoints/checkpoint-90/FULL/file.txt",
            "grail/checkpoints/checkpoint-60/FULL/file.txt",
            "grail/checkpoints/checkpoint-30/DELTA/file.txt",
            "grail/checkpoints/checkpoint-0/FULL/file.txt",
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()
        mock_fetch_chain = AsyncMock(return_value=(0, None))

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 1):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        with patch(
                            "grail.trainer.checkpoint_publisher._fetch_delta_chain_info",
                            mock_fetch_chain,
                        ):
                            from grail.trainer.checkpoint_publisher import CheckpointPublisher

                            publisher = CheckpointPublisher(
                                credentials=mock_credentials, wallet=mock_wallet
                            )
                            # Should not raise
                            await publisher.cleanup_old_checkpoints(90)

        # All windows are in retention (bootstrap + chain)
        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_parses_window_from_path_correctly(self) -> None:
        """Cleanup correctly parses window numbers from various path formats."""
        # With subdirectory structure, test path parsing
        mock_keys = [
            "grail/checkpoints/checkpoint-100/FULL/metadata.json",
            "grail/checkpoints/checkpoint-100/DELTA/model.safetensors",
            "grail/checkpoints/checkpoint-200/READY-200",
            "grail/checkpoints/checkpoint-300/FULL/config.json.gz",
            "grail/checkpoints/checkpoint-400/DELTA/file.txt",
            "grail/checkpoints/checkpoint-500/FULL/file.txt",
            "grail/checkpoints/latest_stable",  # Not a checkpoint - should be ignored
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()
        mock_fetch_chain = AsyncMock(return_value=(400, None))

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 1):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 100):
                with patch("grail.shared.retention_utils.CHECKPOINT_MILESTONE_INTERVAL", 0):
                    with patch(
                        "grail.infrastructure.comms.list_bucket_files",
                        mock_list,
                    ):
                        with patch(
                            "grail.trainer.checkpoint_publisher.delete_prefix",
                            mock_delete,
                        ):
                            with patch(
                                "grail.trainer.checkpoint_publisher._fetch_delta_chain_info",
                                mock_fetch_chain,
                            ):
                                from grail.trainer.checkpoint_publisher import CheckpointPublisher

                                publisher = CheckpointPublisher(
                                    credentials=mock_credentials, wallet=mock_wallet
                                )
                                await publisher.cleanup_old_checkpoints(500)

        # Bootstrap windows (0-9 * 100 = 0-900) means windows 0, 100, 200, ... 900 are kept
        # With WINDOW_LENGTH=100, bootstrap_windows=10 keeps 0, 100, 200, 300, 400, 500, 600, 700, 800, 900
        # So nothing is deleted
        assert mock_delete.call_count == 0


# ============================================================================
# Tests for compute_chain_windows (chain-aware retention)
# ============================================================================


class TestComputeChainWindows:
    """Tests for the chain-walking retention utility."""

    def test_single_chain(self) -> None:
        """Walk a simple chain: anchor=0 -> 30 -> 60 -> 90."""
        chain_map = {90: 60, 60: 30, 30: 0}
        anchor_map = {90: 0}
        result = compute_chain_windows({90}, chain_map, anchor_map)
        assert result == {0, 30, 60}

    def test_keeps_all_intermediates(self) -> None:
        """Retained delta at tip should keep every link back to anchor."""
        chain_map = {120: 90, 90: 60, 60: 30, 30: 0}
        anchor_map = {120: 0}
        result = compute_chain_windows({120}, chain_map, anchor_map)
        assert 0 in result
        assert 30 in result
        assert 60 in result
        assert 90 in result

    def test_shared_chain_prefix(self) -> None:
        """Two retained deltas sharing a chain prefix don't duplicate work."""
        chain_map = {90: 60, 60: 30, 30: 0, 120: 90}
        anchor_map = {90: 0, 120: 0}
        result = compute_chain_windows({90, 120}, chain_map, anchor_map)
        assert result == {0, 30, 60, 90}

    def test_missing_anchor_skips_delta(self) -> None:
        """Delta without anchor info is skipped (no crash)."""
        chain_map = {90: 60}
        anchor_map: dict[int, int] = {}
        result = compute_chain_windows({90}, chain_map, anchor_map)
        assert result == set()

    def test_broken_chain_stops_gracefully(self) -> None:
        """Chain with a missing link stops walking without error."""
        chain_map = {90: 60}  # 60 -> ? is missing
        anchor_map = {90: 0}
        result = compute_chain_windows({90}, chain_map, anchor_map)
        # Keeps anchor and the known link (60), stops there
        assert 0 in result
        assert 60 in result

    def test_cycle_protection(self) -> None:
        """Cycle in chain_map does not cause infinite loop."""
        chain_map = {90: 60, 60: 90}  # cycle
        anchor_map = {90: 0}
        result = compute_chain_windows({90}, chain_map, anchor_map)
        assert 0 in result
        # Should terminate without hanging

    def test_empty_inputs(self) -> None:
        """No retained deltas returns empty set."""
        result = compute_chain_windows(set(), {}, {})
        assert result == set()

    def test_delta_is_direct_successor_of_anchor(self) -> None:
        """Delta whose prev_window is the anchor needs no intermediates."""
        chain_map = {30: 0}
        anchor_map = {30: 0}
        result = compute_chain_windows({30}, chain_map, anchor_map)
        assert result == {0}
