"""Checkpoint publishing utilities for the trainer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import bittensor as bt

from grail.infrastructure.checkpoints import (
    CHECKPOINT_PREFIX,
    CheckpointManager,
    CheckpointMetadata,
)
from grail.infrastructure.comms import get_file_size, upload_file_chunked
from grail.shared.constants import (
    TRAINER_BATCH_SIZE,
    TRAINER_ENTROPY_COEF,
    TRAINER_EPOCHS,
    TRAINER_GRAD_CLIP,
    TRAINER_KL_COEF,
    TRAINER_LR,
    TRAINER_MAX_LENGTH,
    TRAINER_WARMUP_STEPS,
    UPLOAD_TIMEOUT,
    WINDOW_LENGTH,
)

logger = logging.getLogger(__name__)


async def finalize_checkpoint_ready(
    current_block: int,
    current_window: int,
    credentials: Any,
) -> list[int]:
    """Add READY marker to the current window's checkpoint if it's been fully uploaded.

    The checkpoint becomes READY as soon as it's uploaded (has metadata), but only
    if the current block is before the window starts (N + WINDOW_LENGTH). This ensures
    miners/validators have time to download before needing the checkpoint.

    Args:
        current_block: Current blockchain block number
        current_window: Current window number to check
        credentials: R2 credentials for uploading READY marker

    Returns:
        List of window numbers that were finalized (had READY added)
    """
    from grail.infrastructure.comms import list_bucket_files, upload_file_chunked

    # List all checkpoint directories
    keys = await list_bucket_files(
        CHECKPOINT_PREFIX,
        credentials=credentials,
        use_write=True,
    )

    finalized_windows: list[int] = []
    deadline_block = current_window + WINDOW_LENGTH

    # Skip if past deadline - checkpoint wasn't ready on time
    if current_block >= deadline_block:
        logger.warning(
            "Checkpoint %s missed READY deadline (block %s >= %s)",
            current_window,
            current_block,
            deadline_block,
        )
        return finalized_windows

    # Check if READY already exists
    ready_key = f"{CHECKPOINT_PREFIX}checkpoint-{current_window}/READY"
    if ready_key in keys:
        return finalized_windows  # Already has READY marker

    # Check if checkpoint has metadata (indicates complete upload)
    metadata_key = f"{CHECKPOINT_PREFIX}checkpoint-{current_window}/metadata.json.gz"
    metadata_key_uncompressed = f"{CHECKPOINT_PREFIX}checkpoint-{current_window}/metadata.json"
    has_metadata = metadata_key in keys or metadata_key_uncompressed in keys

    if not has_metadata:
        logger.debug(
            "Checkpoint window %s missing metadata, skipping READY marker",
            current_window,
        )
        return finalized_windows

    # Add READY marker at deterministic block
    try:
        await upload_file_chunked(
            ready_key,
            b"",
            credentials=credentials,
            use_write=True,
        )
        logger.info(
            "✅ Added READY marker for checkpoint window %s at block %s",
            current_window,
            current_block,
        )
        finalized_windows.append(current_window)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to add READY marker for checkpoint window %s: %s",
            current_window,
            exc,
        )

    return finalized_windows


async def publish_checkpoint(
    model: Any,
    tokenizer: Any,
    target_window: int,
    trained_on_window: int,
    wallet: bt.wallet,
    credentials: Any,
    checkpoint_manager: CheckpointManager,
    seed: int,
) -> bool:
    """Publish a HF-style checkpoint to R2 and update metadata pointers."""

    temp_dir = Path(tempfile.mkdtemp(prefix=f"checkpoint-{target_window}-"))
    try:
        logger.info("Saving checkpoint to %s", temp_dir)
        model.save_pretrained(temp_dir, safe_serialization=True)
        tokenizer.save_pretrained(temp_dir)

        file_manifest: dict[str, str] = {}
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(temp_dir))
                file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

        training_config = {
            "lr": TRAINER_LR,
            "epochs": TRAINER_EPOCHS,
            "batch_size": TRAINER_BATCH_SIZE,
            "max_length": TRAINER_MAX_LENGTH,
            "grad_clip": TRAINER_GRAD_CLIP,
            "warmup_steps": TRAINER_WARMUP_STEPS,
            "kl_coef": TRAINER_KL_COEF,
            "entropy_coef": TRAINER_ENTROPY_COEF,
            "seed": seed,
        }
        config_hash = hashlib.sha256(
            json.dumps(training_config, sort_keys=True).encode()
        ).hexdigest()

        metadata = CheckpointMetadata(
            window=target_window,
            parent_window=trained_on_window,
            file_manifest=file_manifest,
            training_config=training_config,
            git_commit=os.getenv("GIT_COMMIT", "unknown"),
            created_at=time.time(),
        )

        metadata_dict = {**metadata.__dict__, "config_hash": config_hash}
        metadata_path = temp_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

        canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
        signature = wallet.hotkey.sign(data=canonical_metadata).hex()
        (temp_dir / "manifest.sig").write_text(signature)

        remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"
        semaphore = asyncio.Semaphore(4)

        async def upload_file(path: Path) -> bool:
            async with semaphore:
                rel_path = path.relative_to(temp_dir)
                remote_key = f"{remote_prefix}/{rel_path}"
                try:
                    # Get upload timeout from environment (default 180 sec per chunk - 3 minutes)
                    upload_timeout = UPLOAD_TIMEOUT
                    file_size_mb = path.stat().st_size / (1024 * 1024)
                    logger.debug(
                        "Uploading %s (%d MB) with timeout=%ds",
                        rel_path,
                        file_size_mb,
                        upload_timeout,
                    )
                    return await upload_file_chunked(
                        remote_key,
                        path.read_bytes(),
                        credentials=credentials,
                        use_write=True,
                        upload_timeout=upload_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Upload TIMEOUT for %s (exceeds %s seconds)",
                        rel_path,
                        UPLOAD_TIMEOUT,
                    )
                    return False
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to upload %s: %s", rel_path, exc)
                    return False

        # Calculate total bytes to upload and time the upload phase
        upload_tasks = [upload_file(path) for path in temp_dir.rglob("*") if path.is_file()]
        total_bytes = sum(path.stat().st_size for path in temp_dir.rglob("*") if path.is_file())
        total_mb = total_bytes / (1024 * 1024)

        upload_start = time.time()
        results = await asyncio.gather(*upload_tasks)
        upload_duration = time.time() - upload_start

        if not all(results):
            logger.error("Some checkpoint files failed to upload for window %s", target_window)
            return False

        # Calculate and log upload speed
        upload_speed_mbps = total_mb / upload_duration if upload_duration > 0 else 0
        logger.info(
            "⬆️ Upload summary: %.1f MB in %.1fs (%.1f MB/s)",
            total_mb,
            upload_duration,
            upload_speed_mbps,
        )

        # Verify all uploaded files have the correct size
        logger.info("Verifying uploaded checkpoint files...")
        files_to_verify = [
            (path, path.stat().st_size) for path in temp_dir.rglob("*") if path.is_file()
        ]
        for local_path, expected_size in files_to_verify:
            rel_path = str(local_path.relative_to(temp_dir))
            remote_key = f"{remote_prefix}/{rel_path}"

            # Only small JSON files (<10MB) are compressed; check for .gz version accordingly
            is_small_json = remote_key.endswith(".json") and expected_size < 10 * 1024 * 1024
            if is_small_json:
                remote_key = remote_key + ".gz"

            remote_size = await get_file_size(remote_key, credentials=credentials, use_write=True)
            if remote_size is None:
                logger.error("Failed to verify uploaded file (not found): %s", rel_path)
                return False
            # For compressed JSON, we can't verify exact size due to compression
            if is_small_json:
                if remote_size <= 0:
                    logger.error(
                        "Invalid size for compressed file %s: remote=%d bytes",
                        rel_path,
                        remote_size,
                    )
                    return False
                logger.debug("✅ Verified compressed JSON file %s: %d bytes", rel_path, remote_size)
            else:
                if remote_size != expected_size:
                    logger.error(
                        "Size mismatch for %s: local=%d bytes, remote=%d bytes",
                        rel_path,
                        expected_size,
                        remote_size,
                    )
                    return False
        logger.info("✅ All checkpoint files verified successfully")

        # NOTE: READY marker is NOT added here to ensure determinism
        # It will be added by finalize_checkpoint_ready() before window starts (block < target_window + WINDOW_LENGTH)
        logger.info(
            "✅ Published checkpoint files for window %s (READY marker deadline: block %s)",
            target_window,
            target_window + WINDOW_LENGTH,
        )

        try:
            await checkpoint_manager.cleanup_remote(target_window)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)

        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to publish checkpoint: %s", exc)
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
