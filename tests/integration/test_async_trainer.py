"""Integration tests for async trainer components.

Tests the async training infrastructure including snapshot management,
upload worker, and training process coordination.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from grail.trainer.snapshot_manager import SnapshotManager


class TestSnapshotManager:
    """Test SnapshotManager atomic operations and IPC mechanisms."""

    def test_snapshot_manager_initialization(self) -> None:
        """Test snapshot manager creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            assert manager.snapshot_dir.exists()
            assert manager.staging_dir.exists()
            assert manager.locks_dir.exists()

    def test_atomic_snapshot_save(self) -> None:
        """Test atomic snapshot save creates SNAPSHOT_READY marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            # Save snapshot
            metadata = {"epoch": 1, "timestamp": time.time()}
            manager.save_snapshot_atomic(mock_model, mock_tokenizer, metadata)

            # Verify SNAPSHOT_READY marker exists
            assert manager.check_snapshot_ready()

            # Verify snapshot directory exists
            snapshot_path = manager.snapshot_dir / "latest"
            assert snapshot_path.exists()

    def test_snapshot_copy_to_staging(self) -> None:
        """Test copying snapshot to staging clears READY marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            # Save snapshot
            metadata = {"epoch": 1, "timestamp": time.time()}
            manager.save_snapshot_atomic(mock_model, mock_tokenizer, metadata)

            assert manager.check_snapshot_ready()

            # Copy to staging
            staging_path = manager.copy_snapshot_to_staging()

            # READY marker should be cleared
            assert not manager.check_snapshot_ready()

            # Staging path should exist
            assert staging_path.exists()

            # Cleanup
            manager.cleanup_staging()
            assert not staging_path.exists()

    def test_pause_training_flag(self) -> None:
        """Test PAUSE_TRAINING flag operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Initially no pause flag
            assert not manager.check_pause_flag()

            # Set pause flag
            manager.set_pause_flag()
            assert manager.check_pause_flag()

            # Clear pause flag
            manager.clear_pause_flag()
            assert not manager.check_pause_flag()

    def test_training_heartbeat(self) -> None:
        """Test training heartbeat updates and age calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Initially heartbeat age is infinite
            assert manager.get_training_heartbeat_age() == float("inf")

            # Set heartbeat
            manager.set_training_heartbeat()

            # Age should be near zero
            age = manager.get_training_heartbeat_age()
            assert 0 <= age < 1

            # Wait and check age increases
            time.sleep(0.1)
            age_after = manager.get_training_heartbeat_age()
            assert age_after > age

    def test_get_latest_snapshot_path(self) -> None:
        """Test getting latest snapshot path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Initially no snapshot
            assert manager.get_latest_snapshot_path() is None

            # Create snapshot
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            manager.save_snapshot_atomic(
                mock_model,
                mock_tokenizer,
                {"epoch": 1, "timestamp": time.time()},
            )

            # Now snapshot path should exist
            snapshot_path = manager.get_latest_snapshot_path()
            assert snapshot_path is not None
            assert snapshot_path.exists()


class TestAsyncTrainerIntegration:
    """Integration tests for async trainer orchestration.

    Note: These are basic tests. Full integration testing requires
    running actual training, upload, and evaluation processes which
    is beyond the scope of unit tests.
    """

    def test_snapshot_manager_IPC_flow(self) -> None:
        """Test complete IPC flow: save -> check ready -> copy -> cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Mock model and tokenizer
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()

            # Simulate training process: save snapshot
            manager.save_snapshot_atomic(
                mock_model,
                mock_tokenizer,
                {"epoch": 1, "timestamp": time.time()},
            )

            # Simulate upload worker: check ready
            assert manager.check_snapshot_ready()

            # Simulate upload worker: copy to staging
            staging_path = manager.copy_snapshot_to_staging()
            assert staging_path.exists()

            # READY marker should be cleared
            assert not manager.check_snapshot_ready()

            # Simulate upload worker: cleanup after upload
            manager.cleanup_staging()
            assert not staging_path.exists()

    def test_evaluation_coordination_flow(self) -> None:
        """Test evaluation coordination flow with pause/resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Simulate main process: set pause flag
            manager.set_pause_flag()

            # Simulate training process: check pause flag
            assert manager.check_pause_flag()

            # Training would save snapshot and wait here

            # Simulate main process: run evaluation, then clear pause
            manager.clear_pause_flag()

            # Training process can resume
            assert not manager.check_pause_flag()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
