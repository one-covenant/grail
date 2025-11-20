"""Upload worker process for async checkpoint uploading.

Runs as separate process that:
1. Polls for new snapshots (SNAPSHOT_READY marker)
2. Copies snapshot to staging
3. Uploads to R2 asynchronously
4. Determines window number after upload completes
5. Sets READY marker for checkpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Any

import bittensor as bt

from grail.infrastructure.network import create_subtensor
from grail.shared.constants import (
    UPLOAD_RETRY_BACKOFF_BASE,
    UPLOAD_RETRY_MAX_ATTEMPTS,
    WINDOW_LENGTH,
)
from grail.trainer.checkpoint_publisher import CheckpointPublisher
from grail.trainer.snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)


async def upload_worker_loop(
    snapshot_manager: SnapshotManager,
    checkpoint_publisher: CheckpointPublisher,
    stop_event: multiprocessing.Event,
    poll_interval: int = 30,
) -> None:
    """Main upload worker loop.

    Polls for new snapshots and uploads them to R2 asynchronously.
    Window number is determined AFTER upload completes based on current block.
    Creates its own subtensor connection in child process.

    Args:
        snapshot_manager: Snapshot manager for IPC
        checkpoint_publisher: Publisher for R2 uploads
        stop_event: Event to signal shutdown
        poll_interval: Seconds between snapshot checks
    """
    logger.info("Upload worker starting (poll_interval=%ds)", poll_interval)

    # Create resilient subtensor connection in child process
    subtensor = await create_subtensor(resilient=True)
    logger.info("Created resilient subtensor connection in upload worker")

    while not stop_event.is_set():
        try:
            # Check if snapshot ready
            if not snapshot_manager.check_snapshot_ready():
                await asyncio.sleep(poll_interval)
                continue

            logger.info("New snapshot detected, preparing upload")

            # Copy snapshot to staging
            try:
                staging_path = snapshot_manager.copy_snapshot_to_staging()
            except FileNotFoundError as exc:
                logger.warning("Snapshot not found during copy: %s", exc)
                continue

            # Record upload start for timing
            upload_start_time = time.time()
            upload_start_block = await subtensor.get_current_block()
            checkpoint_window = (upload_start_block // WINDOW_LENGTH) * WINDOW_LENGTH

            logger.info(
                "Starting upload at block %s for checkpoint-%s",
                upload_start_block,
                checkpoint_window,
            )

            # Upload to R2 with retry logic
            success = await _upload_with_retry(
                staging_path,
                checkpoint_publisher,
                checkpoint_window,
                max_attempts=UPLOAD_RETRY_MAX_ATTEMPTS,
            )

            if not success:
                logger.error("Upload failed after retries, discarding snapshot")
                snapshot_manager.cleanup_staging()
                continue

            upload_duration = time.time() - upload_start_time

            # Calculate ready_window based on FINISH time
            current_block = await subtensor.get_current_block()
            ready_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
            blocks_elapsed = current_block - upload_start_block

            logger.info(
                "Upload completed in %.1fs (%d blocks elapsed): checkpoint-%s ready at window %s",
                upload_duration,
                blocks_elapsed,
                checkpoint_window,
                ready_window,
            )

            # Set READY-{ready_window} marker
            try:
                finalized = await checkpoint_publisher.finalize_checkpoint_ready(
                    checkpoint_window,
                    ready_window,
                )
                if finalized:
                    logger.info(
                        "✅ Set READY-%s marker for checkpoint-%s",
                        ready_window,
                        checkpoint_window,
                    )
            except Exception as exc:
                logger.error("Failed to finalize checkpoint READY marker: %s", exc)

            # Cleanup staging directory
            snapshot_manager.cleanup_staging()
            logger.info("Upload cycle complete for checkpoint-%s", checkpoint_window)

        except asyncio.CancelledError:
            logger.info("Upload worker received CancelledError, exiting")
            break
        except Exception as exc:
            logger.exception("Upload worker error: %s", exc)
            # Continue on error after delay
            await asyncio.sleep(poll_interval)

    logger.info("Upload worker exiting")


async def _upload_with_retry(
    staging_path: Path,
    checkpoint_publisher: CheckpointPublisher,
    target_window: int,
    max_attempts: int = 3,
) -> bool:
    """Upload checkpoint with exponential backoff retry.

    Args:
        staging_path: Path to staging directory containing checkpoint
        checkpoint_publisher: Publisher for R2 uploads
        target_window: Window number to upload to
        max_attempts: Maximum upload attempts

    Returns:
        True if upload succeeded, False otherwise
    """
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Upload attempt %d/%d for window %s", attempt, max_attempts, target_window)

            # Read metadata from snapshot
            metadata_path = staging_path / "snapshot_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

            # Upload from staging path to target window
            success = await checkpoint_publisher.upload_from_staging(
                staging_path,
                metadata,
                target_window,
            )

            if success:
                logger.info("Upload succeeded on attempt %d", attempt)
                return True
            else:
                logger.warning("Upload returned false on attempt %d", attempt)

        except Exception as exc:
            logger.error("Upload attempt %d failed: %s", attempt, exc)

        # Exponential backoff if not last attempt
        if attempt < max_attempts:
            backoff = UPLOAD_RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.info("Retrying upload in %ds", backoff)
            await asyncio.sleep(backoff)

    return False


def run_upload_worker(
    snapshot_manager: SnapshotManager,
    credentials: Any,
    wallet_args: dict[str, str],
    stop_event: multiprocessing.Event,
    poll_interval: int = 30,
    verbosity: int = 1,
) -> None:
    """Entry point for upload worker process.

    Sets up asyncio event loop and runs upload worker loop.
    Creates its own subtensor connection in child process.

    Args:
        snapshot_manager: Snapshot manager for IPC
        credentials: R2 credentials
        wallet_args: Wallet configuration (name, hotkey, path)
        stop_event: Event to signal shutdown
        poll_interval: Seconds between snapshot checks
        verbosity: CLI verbosity level (0=silent, 1=INFO, >=2=DEBUG)
    """
    # Configure enhanced logging for upload worker
    from grail.logging_utils import configure_process_logging

    # Map verbosity to log level (same as parent CLI)
    log_level = logging.DEBUG if verbosity >= 2 else logging.INFO
    configure_process_logging("upload", level=log_level, include_function=False)

    logger.info("Upload worker process starting (PID=%d)", multiprocessing.current_process().pid)

    try:
        # Reconstruct wallet and publisher
        logger.info("Reconstructing services in upload worker...")
        wallet = bt.wallet(**wallet_args)
        checkpoint_publisher = CheckpointPublisher(credentials=credentials, wallet=wallet)

        # Run upload loop
        asyncio.run(
            upload_worker_loop(
                snapshot_manager,
                checkpoint_publisher,
                stop_event,
                poll_interval,
            )
        )
    except KeyboardInterrupt:
        logger.info("Upload worker interrupted")
    except Exception as exc:
        logger.exception("Upload worker crashed: %s", exc)
    finally:
        logger.info("Upload worker process exiting")
