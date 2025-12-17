"""Unit tests for checkpoint cleanup and bucket listing functions.

Tests:
- list_bucket_files pagination handling
- _compute_keep_windows retention logic
- cleanup_old_checkpoints integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from grail.trainer.checkpoint_publisher import _compute_keep_windows

# ============================================================================
# Tests for _compute_keep_windows
# ============================================================================


class TestComputeKeepWindows:
    """Tests for the retention window calculation function (chained deltas).

    With chained deltas, retention keeps entire chains from anchor to now.
    """

    def test_negative_window_returns_empty(self) -> None:
        """Negative window should return empty set."""
        result = _compute_keep_windows(-1)
        assert result == set()

    def test_zero_window_returns_zero(self) -> None:
        """Window 0 should keep window 0."""
        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 20):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                result = _compute_keep_windows(0)
                assert 0 in result

    def test_keeps_entire_chain_from_anchor(self) -> None:
        """Should keep all windows from current anchor to now."""
        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 5):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                # Anchor stride = 5 * 30 = 150
                # Window 180: anchor at 150
                result = _compute_keep_windows(180)
                # Should include chain: 150, 180
                assert 150 in result
                assert 180 in result

    def test_keeps_previous_anchor_for_transition(self) -> None:
        """Should keep previous anchor and its chain for consumers catching up."""
        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 3):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                # Anchor stride = 3 * 30 = 90
                # Window 120: current anchor at 90, prev anchor at 0
                result = _compute_keep_windows(120)
                # Current chain
                assert 90 in result
                assert 120 in result
                # Previous anchor
                assert 0 in result

    def test_early_windows_dont_go_negative(self) -> None:
        """Early windows should not include negative values."""
        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 5):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                result = _compute_keep_windows(60)
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
        # With DELTA_BASE_INTERVAL=2 and WINDOW_LENGTH=30, anchor stride = 60
        # Window 120: anchor at 120, prev anchor at 60
        # Should keep: 60, 90, 120 (chain from 60 to 120)
        mock_keys = [
            "grail/checkpoints/checkpoint-120/file.txt",  # Keep (current)
            "grail/checkpoints/checkpoint-90/file.txt",  # Keep (chain)
            "grail/checkpoints/checkpoint-60/file.txt",  # Keep (prev anchor)
            "grail/checkpoints/checkpoint-30/file.txt",  # Keep (chain from prev anchor)
            "grail/checkpoints/checkpoint-0/file.txt",  # Delete (outside retention)
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()

        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 2):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                # list_bucket_files is imported locally inside cleanup_old_checkpoints
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    # delete_prefix is imported at module level
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        from grail.trainer.checkpoint_publisher import CheckpointPublisher

                        publisher = CheckpointPublisher(
                            credentials=mock_credentials, wallet=mock_wallet
                        )
                        await publisher.cleanup_old_checkpoints(120)

        # Window 0 is outside retention (prev anchor at 60)
        # Actually with anchor_stride=60, current_anchor=120, prev_anchor=60
        # Keep: 60, 90, 120 + chain from 60 (60, 90)
        # Delete: 0
        # But wait - 30 is in the chain from prev_anchor (60) down, so it should be kept
        # Actually the chain goes 60, 90, 120 and from prev_anchor 0, 30, 60
        # So 0 and 30 would also be kept. Let me reconsider...
        # anchor_stride = 2 * 30 = 60
        # current 120: anchor = 120 (since 120 % 60 == 0)
        # Wait, 120 // 60 = 2, so anchor = 2 * 60 = 120
        # Chain from 120 to 120: just 120
        # prev_anchor = 120 - 60 = 60
        # Chain from 60 to 120: 60, 90, 120
        # So keep = {60, 90, 120}
        # Delete: 0, 30
        assert mock_delete.call_count == 2
        deleted_prefixes = [call[0][0] for call in mock_delete.call_args_list]
        assert "grail/checkpoints/checkpoint-0" in deleted_prefixes
        assert "grail/checkpoints/checkpoint-30" in deleted_prefixes

    @pytest.mark.asyncio
    async def test_cleanup_keeps_retention_limit_checkpoints(self) -> None:
        """Cleanup keeps exactly retention_limit checkpoints."""
        mock_keys = [
            "grail/checkpoints/checkpoint-7055580/file.txt",
            "grail/checkpoints/checkpoint-7055550/file.txt",
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()

        with patch("grail.trainer.checkpoint_publisher.CHECKPOINT_RETENTION_LIMIT", 2):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        from grail.trainer.checkpoint_publisher import CheckpointPublisher

                        publisher = CheckpointPublisher(
                            credentials=mock_credentials, wallet=mock_wallet
                        )
                        await publisher.cleanup_old_checkpoints(7055580)

        # Should not delete anything - already at limit
        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_empty_bucket(self) -> None:
        """Cleanup handles empty bucket gracefully."""
        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=[])
        mock_delete = AsyncMock()

        with patch("grail.trainer.checkpoint_publisher.CHECKPOINT_RETENTION_LIMIT", 2):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        from grail.trainer.checkpoint_publisher import CheckpointPublisher

                        publisher = CheckpointPublisher(
                            credentials=mock_credentials, wallet=mock_wallet
                        )
                        # Should not raise
                        await publisher.cleanup_old_checkpoints(7055580)

        assert mock_delete.call_count == 0

    @pytest.mark.asyncio
    async def test_cleanup_handles_delete_error(self) -> None:
        """Cleanup continues on delete errors (logs warning but doesn't raise)."""
        # With DELTA_BASE_INTERVAL=1 and WINDOW_LENGTH=30, anchor stride = 30
        # Window 90: anchor at 90, prev at 60
        # Keep: 60, 90
        # Delete: 0, 30
        mock_keys = [
            "grail/checkpoints/checkpoint-90/file.txt",  # Keep (current anchor)
            "grail/checkpoints/checkpoint-60/file.txt",  # Keep (prev anchor)
            "grail/checkpoints/checkpoint-30/file.txt",  # Delete (outside)
            "grail/checkpoints/checkpoint-0/file.txt",  # Delete (outside)
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock(side_effect=[Exception("Delete failed"), None])

        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 1):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 30):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        from grail.trainer.checkpoint_publisher import CheckpointPublisher

                        publisher = CheckpointPublisher(
                            credentials=mock_credentials, wallet=mock_wallet
                        )
                        # Should not raise despite delete error
                        await publisher.cleanup_old_checkpoints(90)

        # Should attempt both deletes (0 and 30)
        assert mock_delete.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_parses_window_from_path_correctly(self) -> None:
        """Cleanup correctly parses window numbers from various path formats."""
        # With DELTA_BASE_INTERVAL=1 and WINDOW_LENGTH=100, anchor stride = 100
        # Window 500: anchor at 500, prev anchor at 400
        # Keep: 400, 500 (and chain between them, which is just those two)
        # Delete: 100, 200, 300
        mock_keys = [
            "grail/checkpoints/checkpoint-100/metadata.json",
            "grail/checkpoints/checkpoint-100/model.safetensors",
            "grail/checkpoints/checkpoint-200/READY-200",
            "grail/checkpoints/checkpoint-300/config.json.gz",
            "grail/checkpoints/checkpoint-400/file.txt",
            "grail/checkpoints/checkpoint-500/file.txt",
            "grail/checkpoints/latest_stable",  # Not a checkpoint - should be ignored
        ]

        mock_credentials = MagicMock()
        mock_wallet = MagicMock()

        mock_list = AsyncMock(return_value=mock_keys)
        mock_delete = AsyncMock()

        with patch("grail.trainer.checkpoint_publisher.DELTA_BASE_INTERVAL", 1):
            with patch("grail.trainer.checkpoint_publisher.WINDOW_LENGTH", 100):
                with patch(
                    "grail.infrastructure.comms.list_bucket_files",
                    mock_list,
                ):
                    with patch(
                        "grail.trainer.checkpoint_publisher.delete_prefix",
                        mock_delete,
                    ):
                        from grail.trainer.checkpoint_publisher import CheckpointPublisher

                        publisher = CheckpointPublisher(
                            credentials=mock_credentials, wallet=mock_wallet
                        )
                        await publisher.cleanup_old_checkpoints(500)

        # Should delete 100, 200, 300; keep 400, 500
        assert mock_delete.call_count == 3
        deleted_prefixes = [call[0][0] for call in mock_delete.call_args_list]
        assert "grail/checkpoints/checkpoint-100" in deleted_prefixes
        assert "grail/checkpoints/checkpoint-200" in deleted_prefixes
        assert "grail/checkpoints/checkpoint-300" in deleted_prefixes
