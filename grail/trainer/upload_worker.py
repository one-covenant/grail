"""Upload worker process for async checkpoint uploading.

Runs as separate process that:
1. Receives snapshot messages from training process via IPC queue
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
import queue
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
from grail.trainer.ipc import IPCChannels
from grail.trainer.snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)


async def _wait_for_snapshot(
    ipc: IPCChannels,
    poll_interval: int,
) -> dict[str, Any] | None:
    """Wait for snapshot notification via IPC queue.

    Args:
        ipc: IPC channels for communication
        poll_interval: Seconds to wait before timeout

    Returns:
        Snapshot message dict if available, None if timeout
    """
    loop = asyncio.get_event_loop()

    try:
        # Use run_in_executor to avoid blocking the event loop
        msg = await loop.run_in_executor(
            None,
            lambda: ipc.snapshot_queue.get(timeout=poll_interval),
        )
        if msg and msg.get("type") == "snapshot_ready":
            logger.debug("Received snapshot message from queue: window=%s", msg.get("window"))
            return msg
        return None
    except queue.Empty:
        # Timeout - return None to re-check stop_event
        return None


async def upload_worker_loop(
    snapshot_manager: SnapshotManager,
    checkpoint_publisher: CheckpointPublisher,
    ipc: IPCChannels,
    poll_interval: int = 30,
) -> None:
    """Main upload worker loop.

    Receives snapshot messages via IPC queue from training process.
    Window number is determined AFTER upload completes based on current block.
    Creates its own subtensor connection in child process.

    Args:
        snapshot_manager: Snapshot manager for staging/cleanup
        checkpoint_publisher: Publisher for R2 uploads
        ipc: IPC channels for coordination
        poll_interval: Seconds between snapshot checks
    """
    logger.info("Upload worker starting (poll_interval=%ds)", poll_interval)

    # Log subtensor configuration (verify env vars are propagated)
    import os

    logger.info(
        "Subtensor config: BT_CALL_TIMEOUT=%s BT_CALL_RETRIES=%s BT_CALL_BACKOFF=%s",
        os.getenv("BT_CALL_TIMEOUT", "NOT_SET"),
        os.getenv("BT_CALL_RETRIES", "NOT_SET"),
        os.getenv("BT_CALL_BACKOFF", "NOT_SET"),
    )

    # Create resilient subtensor connection in child process
    subtensor = await create_subtensor(resilient=True)
    logger.info("Created resilient subtensor connection in upload worker")

    # Track last uploaded window to prevent duplicate uploads within same window
    last_uploaded_window = -1

    while not ipc.stop.is_set():
        try:
            # Wait for snapshot via IPC queue
            snapshot_msg = await _wait_for_snapshot(ipc, poll_interval)

            if snapshot_msg is None:
                # Timeout or stop event - continue loop to check stop_event
                continue

            logger.info("New snapshot detected, preparing upload")

            # Determine target window BEFORE copying snapshot
            upload_start_block = await subtensor.get_current_block()
            checkpoint_window = (upload_start_block // WINDOW_LENGTH) * WINDOW_LENGTH

            # Skip if we already uploaded to this window
            if checkpoint_window == last_uploaded_window:
                logger.info(
                    "Already uploaded to window %s, skipping duplicate upload",
                    checkpoint_window,
                )
                continue

            # Copy snapshot to staging
            try:
                staging_path = snapshot_manager.copy_snapshot_to_staging()
            except FileNotFoundError as exc:
                logger.warning("Snapshot not found during copy: %s", exc)
                continue

            # Record upload start for timing
            upload_start_time = time.time()

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
            # ResilientSubtensor will auto-double timeout due to idle period during upload
            logger.info(
                "Getting current block after %.1fs upload (idle detection will extend timeout)",
                upload_duration,
            )
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

            # Update last uploaded window to prevent duplicates
            last_uploaded_window = checkpoint_window

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
    ipc: IPCChannels,
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
        ipc: IPC channels for coordination
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
                ipc,
                poll_interval,
            )
        )
    except KeyboardInterrupt:
        logger.info("Upload worker interrupted")
    except Exception as exc:
        logger.exception("Upload worker crashed: %s", exc)
    finally:
        logger.info("Upload worker process exiting")
