"""Snapshot management for async trainer coordination.

Provides filesystem-based IPC for coordinating between main process,
training process, and upload worker process.

Design:
- Atomic snapshot writes using temp dir + rename
- Marker files for process coordination
- Heartbeat monitoring for liveness detection
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages model snapshots and inter-process coordination via filesystem.

    Directory structure:
        snapshots/
            latest/              # Current snapshot (atomically updated)
            SNAPSHOT_READY       # Flag: new snapshot available for upload
        staging/
            pending/             # Upload worker copies snapshot here
        locks/
            TRAINING_HEARTBEAT   # Timestamp file for liveness monitoring
    """

    def __init__(self, cache_root: Path) -> None:
        """Initialize snapshot manager.

        Args:
            cache_root: Root directory for snapshot storage
        """
        self.cache_root = Path(cache_root)
        self.snapshot_dir = self.cache_root / "snapshots"
        self.staging_dir = self.cache_root / "staging"
        self.locks_dir = self.cache_root / "locks"

        # Create directory structure
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        # Marker file paths
        self._snapshot_ready_marker = self.snapshot_dir / "SNAPSHOT_READY"
        self._training_heartbeat_file = self.locks_dir / "TRAINING_HEARTBEAT"

    def save_snapshot_atomic(
        self,
        model: Any,
        tokenizer: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Atomically save model snapshot and set READY marker.

        Uses temp directory + atomic rename to ensure snapshot is never
        partially written. Sets SNAPSHOT_READY marker after successful write.

        Args:
            model: Model to save (supports save_pretrained)
            tokenizer: Tokenizer to save
            metadata: Additional metadata (epoch, timestamp, metrics)
        """
        # Generate unique temp directory name
        temp_name = f"latest.tmp.{uuid.uuid4().hex[:8]}"
        temp_dir = self.snapshot_dir / temp_name
        target_dir = self.snapshot_dir / "latest"

        try:
            # Create temp directory
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save model and tokenizer
            logger.debug("Saving model to temp snapshot: %s", temp_dir)
            model.save_pretrained(
                str(temp_dir),
                safe_serialization=True,
            )
            tokenizer.save_pretrained(str(temp_dir))

            # Save metadata
            metadata_path = temp_dir / "snapshot_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Fsync directory to ensure data is on disk
            try:
                fd = os.open(temp_dir, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except (OSError, AttributeError):
                # fsync on directory not supported on all platforms
                pass

            # Atomic rename: remove old target if exists, then rename
            if target_dir.exists():
                # Remove old snapshot
                shutil.rmtree(target_dir)

            # Rename temp to target (atomic on POSIX)
            temp_dir.rename(target_dir)

            # Set SNAPSHOT_READY marker
            self._snapshot_ready_marker.touch()

            logger.info("Snapshot saved atomically to %s", target_dir)

        except Exception as exc:
            logger.error("Failed to save snapshot: %s", exc)
            # Cleanup temp directory on failure
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as cleanup_exc:
                    logger.debug("Cleanup failed: %s", cleanup_exc)
            raise

    def check_snapshot_ready(self) -> bool:
        """Check if new snapshot is available for upload.

        Returns:
            True if SNAPSHOT_READY marker exists
        """
        return self._snapshot_ready_marker.exists()

    def copy_snapshot_to_staging(self) -> Path:
        """Copy snapshot to staging directory and clear READY marker.

        Upload worker calls this to get a stable copy of the snapshot
        while training continues.

        Returns:
            Path to staging directory containing snapshot copy

        Raises:
            FileNotFoundError: If no snapshot available
        """
        source_dir = self.snapshot_dir / "latest"
        staging_pending = self.staging_dir / "pending"

        if not source_dir.exists():
            raise FileNotFoundError(f"No snapshot available at {source_dir}")

        # Remove old staging if exists
        if staging_pending.exists():
            shutil.rmtree(staging_pending)

        # Copy snapshot to staging
        logger.info("Copying snapshot to staging: %s -> %s", source_dir, staging_pending)
        shutil.copytree(source_dir, staging_pending)

        # Clear SNAPSHOT_READY marker
        if self._snapshot_ready_marker.exists():
            self._snapshot_ready_marker.unlink()

        return staging_pending

    def cleanup_staging(self) -> None:
        """Remove staging directory after successful upload."""
        staging_pending = self.staging_dir / "pending"
        if staging_pending.exists():
            shutil.rmtree(staging_pending)
            logger.debug("Cleaned up staging directory")

    def set_training_heartbeat(self) -> None:
        """Update training heartbeat timestamp for liveness monitoring."""
        self._training_heartbeat_file.write_text(str(time.time()))

    def get_training_heartbeat_age(self) -> float:
        """Get age of training heartbeat in seconds.

        Returns:
            Age in seconds, or infinity if heartbeat file doesn't exist
        """
        if not self._training_heartbeat_file.exists():
            return float("inf")

        try:
            heartbeat_time = float(self._training_heartbeat_file.read_text())
            return time.time() - heartbeat_time
        except (ValueError, OSError):
            return float("inf")

    def get_latest_snapshot_path(self) -> Path | None:
        """Get path to latest snapshot if it exists.

        Returns:
            Path to latest snapshot directory, or None if not available
        """
        snapshot_path = self.snapshot_dir / "latest"
        return snapshot_path if snapshot_path.exists() else None

    def get_latest_snapshot_metadata(self) -> dict[str, Any] | None:
        """Get metadata from latest snapshot.

        Returns:
            Metadata dict or None if unavailable
        """
        snapshot_path = self.get_latest_snapshot_path()
        if not snapshot_path:
            return None

        metadata_path = snapshot_path / "snapshot_metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read snapshot metadata: %s", exc)
            return None
