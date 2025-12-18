"""Upload worker process for async checkpoint uploading.

Runs as separate process that:
1. Receives snapshot messages from training process via IPC queue
2. Copies snapshot to staging
3. Uploads to R2 asynchronously (FULL or DELTA based on DELTA_BASE_INTERVAL)
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
import torch

from grail.infrastructure.network import create_subtensor
from grail.shared.constants import (
    DELTA_BASE_INTERVAL,
    DELTA_CHECKPOINT_ENABLED,
    UPLOAD_RETRY_BACKOFF_BASE,
    UPLOAD_RETRY_MAX_ATTEMPTS,
    WINDOW_LENGTH,
)
from grail.shared.safetensors_utils import load_model_state_dict
from grail.trainer.checkpoint_publisher import (
    CheckpointPublisher,
    UploadError,
    UploadResult,
)
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
    monitor_config: dict[str, Any] | None = None,
    poll_interval: int = 30,
) -> None:
    """Main upload worker loop.

    Receives snapshot messages via IPC queue from training process.
    Window number is determined AFTER upload completes based on current block.
    Creates its own subtensor connection in child process.

    Uploads FULL checkpoints at base intervals (every DELTA_BASE_INTERVAL windows),
    and sparse DELTA checkpoints for intermediate windows (~99% bandwidth reduction).

    Args:
        snapshot_manager: Snapshot manager for staging/cleanup
        checkpoint_publisher: Publisher for R2 uploads
        ipc: IPC channels for coordination
        poll_interval: Seconds between snapshot checks
    """
    from grail.monitoring import initialize_subprocess_monitoring

    # Initialize monitoring using the shared helper (consistent with training_process)
    # Note: We don't pass get_block_context here since subtensor isn't created yet.
    # Block context will be set in the main loop after subtensor is available.
    monitor = await initialize_subprocess_monitoring(
        monitor_config,
        process_name="upload_worker",
        test_connection=False,  # Skip test since subtensor not ready yet
    )

    delta_base_interval_windows = max(1, int(DELTA_BASE_INTERVAL))
    base_stride_blocks = delta_base_interval_windows * int(WINDOW_LENGTH)

    logger.info(
        "Upload worker starting (poll_interval=%ds, delta_enabled=%s, delta_base_interval_windows=%d, base_stride_blocks=%d)",
        poll_interval,
        DELTA_CHECKPOINT_ENABLED,
        delta_base_interval_windows,
        base_stride_blocks,
    )

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

    # Track base checkpoint for delta uploads
    base_window: int | None = None
    base_state: dict[str, torch.Tensor] | None = None

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
            if monitor:
                monitor.set_block_context(upload_start_block, checkpoint_window)

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

            # Decide FULL vs DELTA upload
            # Upload FULL if: delta disabled, no base yet, or at base interval boundary
            is_base_window = checkpoint_window % base_stride_blocks == 0
            should_upload_full = (
                not DELTA_CHECKPOINT_ENABLED
                or base_window is None
                or base_state is None
                or is_base_window
            )

            # Attempt upload with retry logic
            upload_result: UploadResult | None = None
            did_fallback_to_full = False

            try:
                if should_upload_full:
                    logger.info(
                        "Starting FULL upload at block %s for checkpoint-%s (is_base=%s, has_base=%s)",
                        upload_start_block,
                        checkpoint_window,
                        is_base_window,
                        base_window is not None,
                    )

                    upload_result = await _upload_with_retry(
                        staging_path,
                        checkpoint_publisher,
                        checkpoint_window,
                        max_attempts=UPLOAD_RETRY_MAX_ATTEMPTS,
                        is_delta=False,
                    )
                else:
                    logger.info(
                        "Starting DELTA upload at block %s for checkpoint-%s (base=%s)",
                        upload_start_block,
                        checkpoint_window,
                        base_window,
                    )

                    try:
                        upload_result = await _upload_with_retry(
                            staging_path,
                            checkpoint_publisher,
                            checkpoint_window,
                            max_attempts=UPLOAD_RETRY_MAX_ATTEMPTS,
                            is_delta=True,
                            base_window=base_window,
                            base_state=base_state,
                        )
                    except UploadError:
                        # Delta upload failed, fallback to FULL upload
                        logger.warning(
                            "DELTA upload failed, falling back to FULL upload for checkpoint-%s",
                            checkpoint_window,
                        )
                        upload_result = await _upload_with_retry(
                            staging_path,
                            checkpoint_publisher,
                            checkpoint_window,
                            max_attempts=UPLOAD_RETRY_MAX_ATTEMPTS,
                            is_delta=False,
                        )
                        did_fallback_to_full = True

            except UploadError as exc:
                logger.error("Upload failed after retries: %s", exc)
                snapshot_manager.cleanup_staging()
                continue

            # Cache base state for future deltas (after FULL upload or fallback)
            if should_upload_full or did_fallback_to_full:
                try:
                    loaded_state = load_model_state_dict(staging_path)
                    if loaded_state is not None:
                        base_window = checkpoint_window
                        base_state = loaded_state

                        # Log cached state dtype for debugging precision issues
                        sample_dtype = None
                        for name in list(base_state.keys())[:1]:
                            sample_dtype = base_state[name].dtype

                        logger.info(
                            "[upload_worker] Cached checkpoint-%s as delta BASE: "
                            "tensors=%d, params=%d, sample_dtype=%s "
                            "(this will be used as base for computing next delta)",
                            base_window,
                            len(base_state),
                            sum(t.numel() for t in base_state.values()),
                            sample_dtype,
                        )
                    else:
                        logger.warning(
                            "[upload_worker] Could not cache delta base for checkpoint-%s: "
                            "no model weights found",
                            checkpoint_window,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[upload_worker] Failed to cache delta base from checkpoint-%s: %s",
                        checkpoint_window,
                        exc,
                    )

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

            # Determine actual checkpoint type (may have fallen back to FULL)
            checkpoint_type = "FULL" if (should_upload_full or did_fallback_to_full) else "DELTA"
            logger.info(
                "%s upload completed in %.1fs (%d blocks elapsed): checkpoint-%s ready at window %s",
                checkpoint_type,
                upload_duration,
                blocks_elapsed,
                checkpoint_window,
                ready_window,
            )

            if monitor and upload_result is not None:
                monitor.set_block_context(current_block, checkpoint_window)
                try:
                    # Overall duration (wall clock time for entire upload cycle)
                    await monitor.log_gauge(
                        "profiling/upload_worker/upload_duration_s", upload_duration
                    )
                    await monitor.log_gauge(
                        "profiling/upload_worker/checkpoint_window", float(checkpoint_window)
                    )
                    await monitor.log_counter(
                        f"profiling/upload_worker/uploads/{checkpoint_type.lower()}"
                    )

                    # Log all metrics from UploadResult using to_dict()
                    for key, value in upload_result.to_dict().items():
                        if value is not None:
                            await monitor.log_gauge(
                                f"profiling/upload_worker/{key}",
                                float(value),
                            )

                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to log upload worker metrics: %s", exc, exc_info=True)

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
    is_delta: bool = False,
    base_window: int | None = None,
    base_state: dict[str, torch.Tensor] | None = None,
) -> UploadResult:
    """Upload checkpoint with exponential backoff retry.

    Args:
        staging_path: Path to staging directory containing checkpoint
        checkpoint_publisher: Publisher for R2 uploads
        target_window: Window number to upload to
        max_attempts: Maximum upload attempts
        is_delta: If True, upload as DELTA checkpoint
        base_window: For delta: window of the base checkpoint
        base_state: For delta: state dict of the base checkpoint

    Returns:
        UploadResult with timing and size metrics.

    Raises:
        UploadError: If upload fails after all retry attempts.
    """
    upload_type = "DELTA" if is_delta else "FULL"
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                "%s upload attempt %d/%d for window %s",
                upload_type,
                attempt,
                max_attempts,
                target_window,
            )

            # Read metadata from snapshot
            metadata_path = staging_path / "snapshot_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)

            if is_delta:
                if base_window is None or base_state is None:
                    raise UploadError("Cannot upload delta without base_window and base_state")

                # Upload sparse delta
                result = await checkpoint_publisher.upload_delta(
                    staging_path,
                    metadata,
                    target_window,
                    base_window,
                    base_state,
                )
            else:
                # Upload full checkpoint
                result = await checkpoint_publisher.upload_from_staging(
                    staging_path,
                    metadata,
                    target_window,
                )

            logger.info("%s upload succeeded on attempt %d", upload_type, attempt)
            return result

        except UploadError as exc:
            logger.error("%s upload attempt %d failed: %s", upload_type, attempt, exc)
            last_error = exc
        except Exception as exc:
            logger.error("%s upload attempt %d failed unexpectedly: %s", upload_type, attempt, exc)
            last_error = exc

        # Exponential backoff if not last attempt
        if attempt < max_attempts:
            backoff = UPLOAD_RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            logger.info("Retrying %s upload in %ds", upload_type, backoff)
            await asyncio.sleep(backoff)

    # All attempts exhausted
    raise UploadError(f"{upload_type} upload failed after {max_attempts} attempts") from last_error


def run_upload_worker(
    snapshot_manager: SnapshotManager,
    credentials: Any,
    wallet_args: dict[str, str],
    monitor_config: dict[str, Any] | None,
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
                monitor_config,
                poll_interval,
            )
        )
    except KeyboardInterrupt:
        logger.info("Upload worker interrupted")
    except Exception as exc:
        logger.exception("Upload worker crashed: %s", exc)
    finally:
        logger.info("Upload worker process exiting")
