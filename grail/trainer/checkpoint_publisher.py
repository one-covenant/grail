"""Checkpoint publishing utilities for the trainer (Producer Role).

This module provides WRITE operations for publishing and managing checkpoints.
It should ONLY be used by the trainer neuron.

The CheckpointPublisher class handles:
- Publishing model weights and metadata to R2
- Finalizing checkpoints with READY markers
- Cleaning up old checkpoints per retention policy

Consumers (miners/validators) should use grail.infrastructure.checkpoint_consumer.CheckpointManager
for read-only checkpoint operations.

Design:
- CheckpointPublisher: All write operations (publish, finalize, cleanup)
- CheckpointManager: All read operations (download, validate, cache)
- Shared schema: CheckpointMetadata, constants defined in checkpoint_consumer.py
"""

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

from grail.infrastructure.checkpoint_consumer import (
    CHECKPOINT_PREFIX,
    CheckpointMetadata,
)
from grail.infrastructure.comms import delete_prefix, get_file_size, upload_file_chunked
from grail.infrastructure.delta_checkpoint import (
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.shared.constants import (
    CHECKPOINT_RETENTION_LIMIT,
    DELTA_BASE_INTERVAL,
    DELTA_THRESHOLD,
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


def _compute_keep_windows(current_window: int) -> set[int]:
    """Calculate which checkpoint windows should be retained.

    Retention policy:
    - Keep only latest N windows (CHECKPOINT_RETENTION_LIMIT)
    - Also keep base checkpoints that delta checkpoints depend on
    - No bootstrap checkpoint retention (removes 0-9)
    - No milestone retention (keeps total R2 objects under 1000)

    This ensures list_objects_v2 never needs pagination, avoiding
    stale client issues with aiobotocore cached connections.

    Args:
        current_window: Current window number

    Returns:
        Set of window numbers to retain
    """
    keep: set[int] = set()
    if current_window < 0:
        return keep

    # Keep only latest N windows (typically 1-3)
    for idx in range(CHECKPOINT_RETENTION_LIMIT):
        window = current_window - idx * WINDOW_LENGTH
        if window >= 0:
            keep.add(window)
            # Also keep the base window this delta depends on
            # Base windows are at DELTA_BASE_INTERVAL boundaries
            base_window = (window // DELTA_BASE_INTERVAL) * DELTA_BASE_INTERVAL
            if base_window >= 0:
                keep.add(base_window)

    return keep


class CheckpointPublisher:
    """Handles checkpoint publishing operations (Producer role - Trainer only).

    This class encapsulates all write operations for checkpoint management:
    - Publishing model weights and metadata to R2
    - Finalizing checkpoints with READY markers
    - Cleaning up old checkpoints per retention policy

    Design:
    - Stateless operations (no instance state)
    - All methods are async for I/O operations
    - Clean separation from CheckpointManager (consumer role)
    """

    def __init__(self, *, credentials: Any, wallet: bt.wallet) -> None:
        """Initialize checkpoint publisher.

        Args:
            credentials: R2 write credentials for uploading checkpoints
            wallet: Wallet for signing checkpoint metadata
        """
        self.credentials = credentials
        self.wallet = wallet

    async def cleanup_old_checkpoints(
        self,
        current_window: int,
    ) -> None:
        """Delete remote checkpoints outside retention policy.

        This is a write operation that should only be called by the trainer after
        publishing new checkpoints. Miners and validators should never call this.

        Args:
            current_window: Current window number
        """
        from grail.infrastructure.comms import list_bucket_files

        # Calculate windows to keep
        keep_windows = _compute_keep_windows(current_window)

        # Get list of remote windows
        keys = await list_bucket_files(
            CHECKPOINT_PREFIX,
            credentials=self.credentials,
            use_write=True,
        )

        remote_windows: set[int] = set()
        for key in keys:
            try:
                parts = key.split("/")
                if len(parts) >= 3:
                    checkpoint_segment = parts[2]
                    if checkpoint_segment.startswith("checkpoint-"):
                        window = int(checkpoint_segment.split("-", 1)[1])
                        remote_windows.add(window)
            except (IndexError, ValueError):
                continue

        # Delete windows outside retention policy
        for window in remote_windows:
            if window not in keep_windows:
                prefix = f"{CHECKPOINT_PREFIX}checkpoint-{window}"
                logger.info("Deleting remote checkpoint prefix %s", prefix)
                try:
                    await delete_prefix(prefix, credentials=self.credentials, use_write=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to delete remote checkpoint %s: %s", prefix, exc)

    async def finalize_checkpoint_ready(
        self,
        checkpoint_window: int,
        ready_window: int,
    ) -> bool:
        """Add READY-{ready_window} marker to indicate when checkpoint became available.

        The marker filename encodes the window when the upload completed, enabling
        miners/validators to discover checkpoints via filename parsing alone.

        Args:
            checkpoint_window: The checkpoint directory (based on upload start)
            ready_window: The window when upload completed (based on finish block)

        Returns:
            True if marker was added successfully, False otherwise
        """
        ready_key = f"{CHECKPOINT_PREFIX}checkpoint-{checkpoint_window}/READY-{ready_window}"

        try:
            await upload_file_chunked(
                ready_key,
                b"",
                credentials=self.credentials,
                use_write=True,
            )
            logger.info(
                "✅ Added READY-%s marker for checkpoint-%s",
                ready_window,
                checkpoint_window,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to add READY-%s marker for checkpoint-%s: %s",
                ready_window,
                checkpoint_window,
                exc,
            )
            return False

    async def publish_checkpoint(
        self,
        model: Any,
        tokenizer: Any,
        target_window: int,
        trained_on_window: int,
        seed: int,
    ) -> bool:
        """Publish a HF-style checkpoint to R2 and update metadata pointers.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            target_window: Window number for this checkpoint
            trained_on_window: Parent window this model was trained on
            seed: Random seed used for training

        Returns:
            True if publish succeeded, False otherwise
        """
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

            # Extract model name for checkpoint metadata
            model_name: str = getattr(model, "name_or_path", "no_name")

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=trained_on_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=time.time(),
                model_name=model_name,
            )

            metadata_dict = {**metadata.__dict__, "config_hash": config_hash}
            metadata_path = temp_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

            canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
            signature = self.wallet.hotkey.sign(data=canonical_metadata).hex()
            (temp_dir / "manifest.sig").write_text(signature)

            remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"
            semaphore = asyncio.Semaphore(4)

            async def upload_file(path: Path) -> bool:
                async with semaphore:
                    rel_path = path.relative_to(temp_dir)
                    remote_key = f"{remote_prefix}/{rel_path}"
                    try:
                        # Get upload timeout from environment (default 180 sec per chunk)
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
                            credentials=self.credentials,
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

                remote_size = await get_file_size(
                    remote_key, credentials=self.credentials, use_write=True
                )
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
                    logger.debug(
                        "✅ Verified compressed JSON file %s: %d bytes", rel_path, remote_size
                    )
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
            # It will be added by finalize_checkpoint_ready() before window starts
            logger.info(
                "✅ Published checkpoint files for window %s (READY marker deadline: block %s)",
                target_window,
                target_window + WINDOW_LENGTH,
            )

            try:
                await self.cleanup_old_checkpoints(target_window)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)

            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to publish checkpoint: %s", exc)
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def upload_from_staging(
        self,
        staging_path: Path,
        snapshot_metadata: dict[str, Any],
        target_window: int,
    ) -> bool:
        """Upload pre-serialized checkpoint from staging directory.

        Used by upload worker to upload snapshots that have already been
        written to disk by the training process.

        Args:
            staging_path: Path to staging directory containing checkpoint files
            snapshot_metadata: Metadata from snapshot (epoch, timestamp, metrics)
            target_window: The window number to publish this checkpoint to

        Returns:
            True if upload succeeded, False otherwise
        """
        try:
            # Build file manifest from existing files
            file_manifest: dict[str, str] = {}
            for file_path in staging_path.rglob("*"):
                if file_path.is_file() and file_path.name != "snapshot_metadata.json":
                    rel_path = str(file_path.relative_to(staging_path))
                    file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

            # Read training config from snapshot metadata or use defaults
            training_config = snapshot_metadata.get(
                "training_config",
                {
                    "lr": TRAINER_LR,
                    "epochs": TRAINER_EPOCHS,
                    "batch_size": TRAINER_BATCH_SIZE,
                    "max_length": TRAINER_MAX_LENGTH,
                    "grad_clip": TRAINER_GRAD_CLIP,
                    "warmup_steps": TRAINER_WARMUP_STEPS,
                    "kl_coef": TRAINER_KL_COEF,
                    "entropy_coef": TRAINER_ENTROPY_COEF,
                },
            )

            # Create metadata
            # Use parent_window from snapshot metadata if available (authoritative source)
            # Fallback to calculation if not present (e.g. old snapshot format)
            parent_window = snapshot_metadata.get("parent_window")
            if parent_window is None:
                parent_window = max(0, target_window - WINDOW_LENGTH)

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=parent_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=snapshot_metadata.get("timestamp", time.time()),
                model_name="async_trainer_snapshot",
            )

            config_hash = hashlib.sha256(
                json.dumps(training_config, sort_keys=True).encode()
            ).hexdigest()

            metadata_dict = {**metadata.__dict__, "config_hash": config_hash}

            # Save metadata to staging directory
            metadata_path = staging_path / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

            # Sign metadata
            canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
            signature = self.wallet.hotkey.sign(data=canonical_metadata).hex()
            (staging_path / "manifest.sig").write_text(signature)

            # Upload to target window location
            remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"

            logger.info("Uploading checkpoint from staging to %s", remote_prefix)

            semaphore = asyncio.Semaphore(4)

            async def upload_file(path: Path) -> bool:
                async with semaphore:
                    rel_path = path.relative_to(staging_path)
                    remote_key = f"{remote_prefix}/{rel_path}"
                    try:
                        upload_timeout = UPLOAD_TIMEOUT
                        file_size_mb = path.stat().st_size / (1024 * 1024)
                        logger.debug(
                            "Uploading %s (%.1f MB) with timeout=%ds",
                            rel_path,
                            file_size_mb,
                            upload_timeout,
                        )
                        return await upload_file_chunked(
                            remote_key,
                            path.read_bytes(),
                            credentials=self.credentials,
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

            # Upload all files (metadata.json last would be better, but async gather is concurrent)
            # We rely on the READY marker for visibility, so file upload order is less critical
            # provided READY is checked.
            upload_tasks = [upload_file(path) for path in staging_path.rglob("*") if path.is_file()]
            total_bytes = sum(
                path.stat().st_size for path in staging_path.rglob("*") if path.is_file()
            )
            total_mb = total_bytes / (1024 * 1024)

            upload_start = time.time()
            results = await asyncio.gather(*upload_tasks)
            upload_duration = time.time() - upload_start

            if not all(results):
                logger.error("Some checkpoint files failed to upload from staging")
                return False

            throughput_mbps = (total_mb / upload_duration) if upload_duration > 0 else 0
            logger.info(
                "Uploaded %.1f MB in %.1fs (%.1f MB/s) to %s",
                total_mb,
                upload_duration,
                throughput_mbps,
                remote_prefix,
            )

            # Cleanup old checkpoints after successful upload (same as publish_checkpoint)
            try:
                await self.cleanup_old_checkpoints(target_window)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)

            return True

        except Exception as exc:
            logger.exception("Failed to upload checkpoint from staging: %s", exc)
            return False

    async def upload_delta(
        self,
        staging_path: Path,
        snapshot_metadata: dict[str, Any],
        target_window: int,
        base_window: int,
        base_state: dict[str, "torch.Tensor"],
    ) -> bool:
        """Upload sparse delta checkpoint.

        Computes sparse delta from base_state to current state, uploads only
        non-zero differences in COO format for ~99% bandwidth reduction.

        Args:
            staging_path: Path to staging directory containing current checkpoint
            snapshot_metadata: Metadata from snapshot (epoch, timestamp, metrics)
            target_window: The window number to publish this delta to
            base_window: The window of the FULL checkpoint this delta is relative to
            base_state: State dict of the base checkpoint

        Returns:
            True if upload succeeded, False otherwise
        """
        import torch
        from safetensors.torch import load_file, save_file

        temp_dir = Path(tempfile.mkdtemp(prefix=f"delta-{target_window}-"))
        try:
            # Load current weights from staging
            model_path = staging_path / "model.safetensors"
            if not model_path.exists():
                logger.error("No model.safetensors found in staging path: %s", staging_path)
                return False

            current_state = load_file(model_path)

            # Compute sparse delta
            sparse_tensors, shapes, stats = compute_sparse_delta(
                current_state,
                base_state,
                threshold=DELTA_THRESHOLD,
            )

            # Compute hash of current state (for consumer verification)
            weights_hash = compute_weights_hash(current_state)

            # Save sparse delta to temp directory
            if sparse_tensors:
                save_file(sparse_tensors, temp_dir / "delta_sparse.safetensors")
            else:
                # No changes - create empty safetensors file
                save_file({}, temp_dir / "delta_sparse.safetensors")

            # Save delta metadata
            delta_meta = {
                "format": "sparse_coo",
                "threshold": DELTA_THRESHOLD,
                "base_window": base_window,
                "shapes": shapes,
                **stats,
            }
            (temp_dir / "delta_metadata.json").write_text(
                json.dumps(delta_meta, ensure_ascii=False, indent=2)
            )

            # Build file manifest for delta files
            file_manifest: dict[str, str] = {}
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(temp_dir))
                    file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

            # Copy non-weight files from staging (tokenizer, config, etc.)
            for file_path in staging_path.rglob("*"):
                if file_path.is_file() and file_path.name not in (
                    "model.safetensors",
                    "snapshot_metadata.json",
                    "metadata.json",
                    "manifest.sig",
                ):
                    rel_path = str(file_path.relative_to(staging_path))
                    dest_path = temp_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file_path, dest_path)
                    file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

            # Read training config from snapshot metadata or use defaults
            training_config = snapshot_metadata.get(
                "training_config",
                {
                    "lr": TRAINER_LR,
                    "epochs": TRAINER_EPOCHS,
                    "batch_size": TRAINER_BATCH_SIZE,
                    "max_length": TRAINER_MAX_LENGTH,
                    "grad_clip": TRAINER_GRAD_CLIP,
                    "warmup_steps": TRAINER_WARMUP_STEPS,
                    "kl_coef": TRAINER_KL_COEF,
                    "entropy_coef": TRAINER_ENTROPY_COEF,
                },
            )

            # Create metadata with delta-specific fields
            parent_window = snapshot_metadata.get("parent_window")
            if parent_window is None:
                parent_window = max(0, target_window - WINDOW_LENGTH)

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=parent_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=snapshot_metadata.get("timestamp", time.time()),
                model_name="async_trainer_snapshot",
                checkpoint_type="DELTA",
                base_window=base_window,
                weights_hash=weights_hash,
            )

            config_hash = hashlib.sha256(
                json.dumps(training_config, sort_keys=True).encode()
            ).hexdigest()

            metadata_dict = {**metadata.__dict__, "config_hash": config_hash}

            # Save metadata to temp directory
            metadata_path = temp_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

            # Sign metadata
            canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
            signature = self.wallet.hotkey.sign(data=canonical_metadata).hex()
            (temp_dir / "manifest.sig").write_text(signature)

            # Write DELTA marker file
            (temp_dir / "DELTA").write_text("")

            # Upload to target window location
            remote_prefix = f"{CHECKPOINT_PREFIX}checkpoint-{target_window}"

            logger.info(
                "Uploading delta checkpoint to %s (base=%d, sparsity=%.2f%%, %d non-zero params)",
                remote_prefix,
                base_window,
                stats["sparsity_ratio"] * 100,
                stats["nonzero_params"],
            )

            semaphore = asyncio.Semaphore(4)

            async def upload_file(path: Path) -> bool:
                async with semaphore:
                    rel_path = path.relative_to(temp_dir)
                    remote_key = f"{remote_prefix}/{rel_path}"
                    try:
                        upload_timeout = UPLOAD_TIMEOUT
                        file_size_mb = path.stat().st_size / (1024 * 1024)
                        logger.debug(
                            "Uploading delta %s (%.2f MB) with timeout=%ds",
                            rel_path,
                            file_size_mb,
                            upload_timeout,
                        )
                        return await upload_file_chunked(
                            remote_key,
                            path.read_bytes(),
                            credentials=self.credentials,
                            use_write=True,
                            upload_timeout=upload_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "Delta upload TIMEOUT for %s (exceeds %s seconds)",
                            rel_path,
                            UPLOAD_TIMEOUT,
                        )
                        return False
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to upload delta %s: %s", rel_path, exc)
                        return False

            # Upload all delta files
            upload_tasks = [upload_file(path) for path in temp_dir.rglob("*") if path.is_file()]
            total_bytes = sum(
                path.stat().st_size for path in temp_dir.rglob("*") if path.is_file()
            )
            total_mb = total_bytes / (1024 * 1024)

            upload_start = time.time()
            results = await asyncio.gather(*upload_tasks)
            upload_duration = time.time() - upload_start

            if not all(results):
                logger.error("Some delta checkpoint files failed to upload")
                return False

            throughput_mbps = (total_mb / upload_duration) if upload_duration > 0 else 0
            logger.info(
                "Uploaded delta %.2f MB in %.1fs (%.1f MB/s) to %s",
                total_mb,
                upload_duration,
                throughput_mbps,
                remote_prefix,
            )

            # Cleanup old checkpoints after successful upload
            try:
                await self.cleanup_old_checkpoints(target_window)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)

            return True

        except Exception as exc:
            logger.exception("Failed to upload delta checkpoint: %s", exc)
            return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
