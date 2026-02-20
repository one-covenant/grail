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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bittensor as bt
import torch

from grail.infrastructure.checkpoint_consumer import CheckpointMetadata
from grail.infrastructure.comms import delete_prefix, get_file_size, upload_file_chunked
from grail.infrastructure.delta_checkpoint import (
    compute_sparse_delta,
    compute_weights_hash,
)
from grail.infrastructure.sparse_codec import (
    encode_sparse_delta_v2,
    encode_sparse_delta_v3,
    encode_sparse_delta_v3_1,
)
from grail.shared.checkpoint_paths import (
    checkpoint_delta_prefix,
    checkpoint_full_prefix,
    checkpoint_ready_marker_key,
    checkpoint_window_prefix,
)
from grail.shared.constants import (
    BASE_CHECKPOINT_RETENTION_LIMIT,
    CHECKPOINT_PREFIX,
    CHECKPOINT_TYPE_DELTA,
    CHECKPOINT_TYPE_FULL,
    CLEANUP_INTERVAL_UPLOADS,
    CURRENT_ENV_ID,
    DELTA_CHECKPOINT_RETENTION_LIMIT,
    DELTA_CODEC_FORMAT,
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
from grail.shared.safetensors_utils import load_model_state_dict

logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────────────────────────
# Delta Format Selection
# ──────────────────────────────────────────────────────────────────────────────


def _select_delta_format() -> tuple[
    str, Callable[[dict[str, torch.Tensor], dict[str, list[int]]], bytes]
]:
    """Select delta codec format and encoder from environment.

    Supported values for GRAIL_DELTA_FORMAT:
    - "sparse_codec_v2" or "v2" (default)
    - "sparse_codec_v3" or "v3"
    - "sparse_codec_v3.1" or "v3.1"
    """
    raw = DELTA_CODEC_FORMAT.strip().lower()
    if raw in {"v2", "sparse_codec_v2"}:
        return "sparse_codec_v2", encode_sparse_delta_v2
    if raw in {"v3", "sparse_codec_v3"}:
        return "sparse_codec_v3", encode_sparse_delta_v3
    if raw in {"v3.1", "sparse_codec_v3.1"}:
        return "sparse_codec_v3.1", encode_sparse_delta_v3_1

    logger.warning("Unknown GRAIL_DELTA_FORMAT=%s, defaulting to v2", raw)
    return "sparse_codec_v2", encode_sparse_delta_v2


# COO baseline sizing (flattening is part of compression)
def _estimate_coo_raw_bytes(sparse_tensors: dict[str, torch.Tensor]) -> int:
    index_bytes = torch.tensor([], dtype=torch.int32).element_size()
    total = 0
    for key, values in sparse_tensors.items():
        if not key.endswith(".values"):
            continue
        nnz = values.numel()
        total += nnz * (2 * index_bytes + values.element_size())
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Environment and Generation Config Helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_default_env_config() -> tuple[str, dict[str, Any]]:
    """Get default environment configuration from constants.

    Returns:
        Tuple of (env_id, env_params)

    Raises:
        ValueError: If GRAIL_ENV_ID is empty or invalid
    """
    # Read from environment variables or use defaults
    env_id = os.getenv("GRAIL_ENV_ID", CURRENT_ENV_ID)
    if not env_id:
        raise ValueError("GRAIL_ENV_ID cannot be empty")

    # Validate split parameter
    split = os.getenv("GRAIL_ENV_SPLIT", "train")
    valid_splits = {"train", "test", "validation", "val"}
    if split not in valid_splits:
        logger.warning(
            f"Invalid GRAIL_ENV_SPLIT='{split}' (expected one of {valid_splits}), using 'train'"
        )
        split = "train"

    # GPU eval for kernel environments
    gpu_eval = os.getenv("GRAIL_GPU_EVAL", "false").lower() in ("1", "true", "yes")

    env_params = {
        "split": split,
        "gpu_eval": gpu_eval,
    }
    return env_id, env_params


def _parse_int_param(
    key: str, default: str, min_val: int | None = None, max_val: int | None = None
) -> int:
    """Parse integer environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Optional minimum value
        max_val: Optional maximum value

    Returns:
        Validated integer value
    """
    try:
        value = int(os.getenv(key, default))
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} below minimum {min_val}, using default {default}")
            return int(default)
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} above maximum {max_val}, using default {default}")
            return int(default)
        return value
    except ValueError as e:
        logger.warning(f"Invalid {key} value: {e}, using default {default}")
        return int(default)


def _parse_float_param(
    key: str, default: str, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Parse float environment variable with validation.

    Args:
        key: Environment variable name
        default: Default value if not set or invalid
        min_val: Optional minimum value
        max_val: Optional maximum value

    Returns:
        Validated float value
    """
    try:
        value = float(os.getenv(key, default))
        if min_val is not None and value < min_val:
            logger.warning(f"{key}={value} below minimum {min_val}, using default {default}")
            return float(default)
        if max_val is not None and value > max_val:
            logger.warning(f"{key}={value} above maximum {max_val}, using default {default}")
            return float(default)
        return value
    except ValueError as e:
        logger.warning(f"Invalid {key} value: {e}, using default {default}")
        return float(default)


def get_default_generation_params() -> dict[str, Any]:
    """Get default generation parameters from environment variables.

    All parameters are validated for type and range. Invalid values fall back
    to defaults with a warning.

    Returns:
        Dictionary of validated generation parameters
    """
    return {
        "max_tokens": _parse_int_param("GRAIL_GEN_MAX_TOKENS", "8192", min_val=1, max_val=16384),
        "temperature": _parse_float_param(
            "GRAIL_GEN_TEMPERATURE", "0.7", min_val=0.01, max_val=2.0
        ),
        "top_p": _parse_float_param("GRAIL_GEN_TOP_P", "0.9", min_val=0.0, max_val=1.0),
        "top_k": _parse_int_param("GRAIL_GEN_TOP_K", "50", min_val=0, max_val=1000),
        "repetition_penalty": _parse_float_param(
            "GRAIL_GEN_REPETITION_PENALTY", "1.0", min_val=1.0, max_val=2.0
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Upload Result Types
# ──────────────────────────────────────────────────────────────────────────────


class UploadError(Exception):
    """Raised when a checkpoint upload operation fails."""

    pass


@dataclass(frozen=True, slots=True)
class UploadTiming:
    """Granular timing breakdown for upload operations.

    All times are in seconds. Fields are optional since FULL and DELTA
    uploads have different timing characteristics.
    """

    # Delta-specific timings
    load_state_s: float = 0.0
    compute_delta_s: float = 0.0
    compression_s: float = 0.0

    # FULL-specific timings
    prep_metadata_s: float = 0.0

    # Common timings
    network_upload_s: float = 0.0
    cleanup_s: float = 0.0

    @property
    def total_s(self) -> float:
        """Total time across all phases."""
        return (
            self.load_state_s
            + self.compute_delta_s
            + self.compression_s
            + self.prep_metadata_s
            + self.network_upload_s
            + self.cleanup_s
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for logging/metrics."""
        return {
            "timing_load_state_s": self.load_state_s,
            "timing_compute_delta_s": self.compute_delta_s,
            "timing_compression_s": self.compression_s,
            "timing_prep_metadata_s": self.prep_metadata_s,
            "timing_network_upload_s": self.network_upload_s,
            "timing_cleanup_s": self.cleanup_s,
            "timing_total_s": self.total_s,
        }


@dataclass(frozen=True, slots=True)
class UploadResult:
    """Result of a successful checkpoint upload operation.

    Returned by upload_from_staging() and upload_delta() on success.
    For failures, UploadError is raised instead.
    """

    timing: UploadTiming
    total_bytes: int
    total_mb: float
    throughput_mbps: float

    # Delta-specific fields (None for FULL uploads)
    sparsity_ratio: float | None = None
    nonzero_params: int | None = None
    total_params: int | None = None
    delta_threshold: float | None = None
    delta_raw_bytes: int | None = None
    delta_compressed_bytes: int | None = None
    compression_ratio: float | None = None

    @property
    def is_delta(self) -> bool:
        """True if this result is from a delta upload."""
        return self.sparsity_ratio is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for logging/metrics."""
        result: dict[str, Any] = {
            "upload_total_bytes": self.total_bytes,
            "upload_total_mb": self.total_mb,
            "upload_throughput_mbps": self.throughput_mbps,
            **self.timing.to_dict(),
        }
        # Add delta-specific fields if present
        if self.is_delta:
            result.update(
                {
                    "delta_sparsity_ratio": self.sparsity_ratio,
                    "delta_nonzero_params": self.nonzero_params,
                    "delta_total_params": self.total_params,
                    "delta_threshold": self.delta_threshold,
                    "delta_raw_bytes": self.delta_raw_bytes,
                    "delta_compressed_bytes": self.delta_compressed_bytes,
                    "delta_compression_ratio": self.compression_ratio,
                }
            )
        return result


def _parse_checkpoint_inventory(keys: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Parse checkpoint keys into checkpoint inventory lists.

    Args:
        keys: List of S3 keys from checkpoint prefix

    Returns:
        Tuple of (all_windows, full_windows, delta_windows), each sorted descending (newest first)
    """
    delta_windows_set: set[int] = set()
    all_windows: set[int] = set()

    for key in keys:
        try:
            parts = key.split("/")
            if len(parts) >= 3 and parts[2].startswith("checkpoint-"):
                window = int(parts[2].split("-", 1)[1])
                all_windows.add(window)
                if len(parts) >= 4 and parts[3] == "DELTA":
                    delta_windows_set.add(window)
        except (IndexError, ValueError):
            continue

    all_windows_sorted = sorted(all_windows, reverse=True)
    full_windows_sorted = sorted(all_windows - delta_windows_set, reverse=True)
    delta_windows_sorted = sorted(delta_windows_set, reverse=True)
    return all_windows_sorted, full_windows_sorted, delta_windows_sorted


async def _fetch_delta_anchor_window(
    delta_window: int,
    credentials: Any,
    anchor_cache: dict[int, int] | None = None,
) -> int | None:
    """Fetch the anchor_window a delta checkpoint depends on for recovery.

    For chained deltas, each delta has:
    - prev_window: immediate predecessor (for computing the delta)
    - anchor_window: nearest FULL checkpoint (for recovery/reconstruction)

    This function retrieves the anchor_window, which is the FULL checkpoint
    that must be retained for this delta to be usable.

    Args:
        delta_window: The delta checkpoint window
        credentials: R2 credentials
        anchor_cache: Optional cache of {delta_window: anchor_window} mappings.
            If provided and delta_window is in cache, returns cached value
            without network call. If not in cache, fetches and caches the result.

    Returns:
        The anchor_window from metadata, or None if unavailable
    """
    # Check cache first to avoid redundant downloads
    if anchor_cache is not None and delta_window in anchor_cache:
        return anchor_cache[delta_window]

    from grail.infrastructure.comms import download_file_chunked

    # Delta checkpoints store their metadata under the DELTA sub-prefix so they can
    # coexist with a FULL checkpoint at the same window.
    from grail.shared.checkpoint_paths import checkpoint_delta_metadata_key

    metadata_key = checkpoint_delta_metadata_key(delta_window)
    try:
        metadata_bytes = await download_file_chunked(
            metadata_key,
            credentials=credentials,
            use_write=True,
        )
        if metadata_bytes:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
            anchor = metadata.get("anchor_window")
            # Cache the result if cache is provided
            if anchor_cache is not None and anchor is not None:
                anchor_cache[delta_window] = anchor
            return anchor
    except Exception as exc:
        logger.warning("Failed to read metadata for delta %d: %s", delta_window, exc)
    return None


async def _compute_keep_windows(
    inventory: tuple[list[int], list[int], list[int]],
    credentials: Any,
    anchor_cache: dict[int, int] | None = None,
) -> set[int]:
    """Calculate which checkpoint windows to retain with DELTA dependency tracking.

    For chained deltas, we need to keep entire chains from anchor (FULL) to tip.
    This function combines two strategies:

    1. Chain-based retention: Keep current anchor + chain, previous anchor + chain
       (from shared retention_utils)
    2. Dependency-based retention: Ensure any retained DELTA has its anchor FULL

    Args:
        inventory: Parsed checkpoint inventory (all_windows, full_windows, delta_windows)
        credentials: R2 credentials for querying checkpoint metadata
        anchor_cache: Optional cache of {delta_window: anchor_window} mappings.
            Reused across cleanup calls to avoid redundant metadata downloads.

    Returns:
        Set of window numbers to retain
    """
    from grail.shared.retention_utils import compute_retention_windows

    all_windows, full_windows, delta_windows = inventory

    if not full_windows and not delta_windows:
        return set()

    logger.debug(
        "Checkpoint retention: %d FULL, %d DELTA found",
        len(full_windows),
        len(delta_windows),
    )

    # Get current window (newest in inventory)
    current_window = max(all_windows) if all_windows else 0

    # Start with chain-based retention (keeps entire chains for reconstruction)
    keep = compute_retention_windows(current_window)

    # Also keep latest N FULL checkpoints (anchors) as additional safety margin
    keep.update(full_windows[:BASE_CHECKPOINT_RETENTION_LIMIT])

    # Also keep latest M DELTA checkpoints
    deltas_to_keep = delta_windows[:DELTA_CHECKPOINT_RETENTION_LIMIT]
    keep.update(deltas_to_keep)

    # Ensure any retained DELTA has its anchor FULL kept
    # Use cache to avoid redundant metadata downloads
    anchor_windows = await asyncio.gather(
        *(
            _fetch_delta_anchor_window(delta_window, credentials, anchor_cache)
            for delta_window in deltas_to_keep
        )
    )
    anchors_kept: set[int] = set()
    fallback_anchors_kept: set[int] = set()
    for delta_window, anchor_window in zip(deltas_to_keep, anchor_windows, strict=False):
        if anchor_window is not None:
            keep.add(int(anchor_window))
            anchors_kept.add(int(anchor_window))
            continue

        # Fallback: keep nearest FULL checkpoint before this delta
        for full_window in full_windows:
            if full_window < delta_window:
                keep.add(full_window)
                fallback_anchors_kept.add(full_window)
                break

    # Log anchor retention summary once (instead of per-delta)
    if anchors_kept or fallback_anchors_kept:
        logger.debug(
            "Anchor retention: %d anchors kept for %d deltas (fallback: %d)",
            len(anchors_kept),
            len(deltas_to_keep),
            len(fallback_anchors_kept),
        )

    logger.info(
        "Retention: keeping %d checkpoints (%d FULL, %d DELTA)",
        len(keep),
        sum(1 for w in keep if w in full_windows),
        sum(1 for w in keep if w in delta_windows),
    )

    return keep


class CheckpointPublisher:
    """Handles checkpoint publishing operations (Producer role - Trainer only).

    This class encapsulates all write operations for checkpoint management:
    - Publishing model weights and metadata to R2
    - Finalizing checkpoints with READY markers
    - Cleaning up old checkpoints per retention policy

    Design:
    - Maintains minimal state for optimization (anchor cache, upload counter)
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

        # Optimization: Cache {delta_window: anchor_window} mappings to avoid
        # redundant metadata downloads during cleanup. Anchor windows never change
        # after a delta is uploaded.
        self._anchor_cache: dict[int, int] = {}

        # Optimization: Track upload count to run cleanup less frequently.
        # Cleanup runs every CLEANUP_INTERVAL_UPLOADS uploads instead of every upload.
        self._upload_count: int = 0

    async def cleanup_old_checkpoints(
        self,
        current_window: int,
        force: bool = False,
    ) -> None:
        """Delete remote checkpoints outside retention policy.

        This is a write operation that should only be called by the trainer after
        publishing new checkpoints. Miners and validators should never call this.

        Uses separate retention limits for BASE (FULL) and DELTA checkpoints:
        - BASE_CHECKPOINT_RETENTION_LIMIT: How many FULL checkpoints to keep
        - DELTA_CHECKPOINT_RETENTION_LIMIT: How many DELTA checkpoints to keep
        - ALWAYS keeps any BASE checkpoint that a retained DELTA depends on

        This prevents orphaned deltas whose base was deleted.

        Optimization: Runs only every CLEANUP_INTERVAL_UPLOADS uploads to reduce
        overhead. Uses cached anchor_window mappings to avoid redundant metadata
        downloads.

        Args:
            current_window: Current window number
            force: If True, run cleanup regardless of upload count interval
        """
        # Increment upload counter
        self._upload_count += 1

        # Check if we should run cleanup this time
        # Run on first upload (to handle restarts) and every N uploads thereafter
        # Guard against CLEANUP_INTERVAL_UPLOADS <= 0
        interval = max(1, CLEANUP_INTERVAL_UPLOADS)
        is_first_upload = self._upload_count == 1
        is_interval_upload = self._upload_count % interval == 0
        should_run = force or is_first_upload or is_interval_upload
        if not should_run:
            logger.debug(
                "Cleanup skipped: upload %d (runs on 1st and every %d uploads)",
                self._upload_count,
                interval,
            )
            return

        logger.info(
            "Running checkpoint cleanup (upload %d, interval=%d)",
            self._upload_count,
            interval,
        )

        from grail.infrastructure.comms import list_bucket_files

        # List remote checkpoint keys once (fail-safe: empty means "skip cleanup")
        keys = await list_bucket_files(
            CHECKPOINT_PREFIX,
            credentials=self.credentials,
            use_write=True,
        )
        if not keys:
            logger.warning(
                "Checkpoint cleanup skipped: failed to list remote checkpoints (prefix=%s)",
                CHECKPOINT_PREFIX,
            )
            return

        inventory = _parse_checkpoint_inventory(keys)
        remote_windows, _, delta_windows = inventory

        # Prune cache: remove entries for deltas that no longer exist in R2
        # This prevents unbounded cache growth
        delta_set = set(delta_windows)
        stale_keys = [k for k in self._anchor_cache if k not in delta_set]
        for k in stale_keys:
            del self._anchor_cache[k]
        if stale_keys:
            logger.debug("Pruned %d stale entries from anchor cache", len(stale_keys))

        # Calculate windows to keep with proper dependency tracking
        # Pass anchor cache to avoid redundant metadata downloads
        # Count cache hits: how many deltas_to_keep are already in cache
        deltas_to_keep = delta_windows[:DELTA_CHECKPOINT_RETENTION_LIMIT]
        cache_hits = sum(1 for d in deltas_to_keep if d in self._anchor_cache)
        cache_size_before = len(self._anchor_cache)

        keep_windows = await _compute_keep_windows(inventory, self.credentials, self._anchor_cache)

        # Cache misses = new entries added (may undercount if some fetches failed)
        cache_misses = len(self._anchor_cache) - cache_size_before
        logger.debug(
            "Anchor cache: %d hits, %d misses, %d queried (total cached: %d)",
            cache_hits,
            cache_misses,
            len(deltas_to_keep),
            len(self._anchor_cache),
        )

        if not keep_windows:
            logger.warning(
                "Checkpoint cleanup skipped: keep set is empty (current_window=%d). "
                "This is a safety guard against accidental mass deletion.",
                current_window,
            )
            return

        # Delete windows outside retention policy (deletes entire window including DELTA/FULL)
        deleted_count = 0
        for window in remote_windows:
            if window not in keep_windows:
                prefix = checkpoint_window_prefix(window)
                logger.info("Deleting remote checkpoint prefix %s", prefix)
                try:
                    await delete_prefix(prefix, credentials=self.credentials, use_write=True)
                    deleted_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to delete remote checkpoint %s: %s", prefix, exc)

        if deleted_count > 0:
            logger.info(
                "Checkpoint cleanup complete: deleted %d, kept %d (current_window=%d)",
                deleted_count,
                len(keep_windows),
                current_window,
            )

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
        ready_key = checkpoint_ready_marker_key(checkpoint_window, ready_window)

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

            # Get environment and generation configuration
            env_id, env_params = get_default_env_config()
            generation_params = get_default_generation_params()

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=trained_on_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=time.time(),
                model_name=model_name,
                checkpoint_type=CHECKPOINT_TYPE_FULL,
                env_id=env_id,
                env_params=env_params,
                generation_params=generation_params,
            )

            metadata_dict = {**metadata.__dict__, "config_hash": config_hash}
            metadata_path = temp_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

            canonical_metadata = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
            signature = self.wallet.hotkey.sign(data=canonical_metadata).hex()
            (temp_dir / "manifest.sig").write_text(signature)

            # Write FULL marker file
            (temp_dir / "FULL").touch()

            # Upload to FULL subdir
            remote_prefix = checkpoint_full_prefix(target_window)
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

                remote_size = await get_file_size(
                    remote_key, credentials=self.credentials, use_write=True
                )
                if remote_size is None:
                    logger.error("Failed to verify uploaded file (not found): %s", rel_path)
                    return False
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
    ) -> UploadResult:
        """Upload pre-serialized checkpoint from staging directory.

        Used by upload worker to upload snapshots that have already been
        written to disk by the training process.

        Args:
            staging_path: Path to staging directory containing checkpoint files
            snapshot_metadata: Metadata from snapshot (epoch, timestamp, metrics)
            target_window: The window number to publish this checkpoint to

        Returns:
            UploadResult with timing and size metrics.

        Raises:
            UploadError: If the upload fails.
        """
        # Timing breakdown for granular metrics
        prep_metadata_s = 0.0
        network_upload_s = 0.0
        cleanup_s = 0.0

        try:
            # Build file manifest from existing files
            prep_start = time.time()
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

            # Get environment and generation configuration
            env_id, env_params = get_default_env_config()
            generation_params = get_default_generation_params()

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=parent_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=snapshot_metadata.get("timestamp", time.time()),
                model_name="async_trainer_snapshot",
                checkpoint_type=CHECKPOINT_TYPE_FULL,
                env_id=env_id,
                env_params=env_params,
                generation_params=generation_params,
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

            # Write FULL marker file
            (staging_path / "FULL").touch()
            prep_metadata_s = time.time() - prep_start

            # Upload to FULL subdir
            remote_prefix = checkpoint_full_prefix(target_window)

            logger.info("Uploading FULL checkpoint from staging to %s", remote_prefix)

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

            network_start = time.time()
            results = await asyncio.gather(*upload_tasks)
            network_upload_s = time.time() - network_start

            if not all(results):
                raise UploadError("Some checkpoint files failed to upload from staging")

            throughput_mbps = (total_mb / network_upload_s) if network_upload_s > 0 else 0.0
            logger.info(
                "Uploaded %.1f MB in %.1fs (%.1f MB/s) to %s",
                total_mb,
                network_upload_s,
                throughput_mbps,
                remote_prefix,
            )

            # Cleanup old checkpoints after successful upload (same as publish_checkpoint)
            cleanup_start = time.time()
            try:
                await self.cleanup_old_checkpoints(target_window)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)
            cleanup_s = time.time() - cleanup_start

            # Build result with timing breakdown
            timing = UploadTiming(
                prep_metadata_s=prep_metadata_s,
                network_upload_s=network_upload_s,
                cleanup_s=cleanup_s,
            )

            logger.info(
                "⏱️ FULL upload timing breakdown: prep=%.1fs, network=%.1fs, cleanup=%.1fs (total=%.1fs)",
                timing.prep_metadata_s,
                timing.network_upload_s,
                timing.cleanup_s,
                timing.total_s,
            )

            return UploadResult(
                timing=timing,
                total_bytes=int(total_bytes),
                total_mb=float(total_mb),
                throughput_mbps=float(throughput_mbps),
            )

        except UploadError:
            # Re-raise UploadError as-is
            raise
        except Exception as exc:
            logger.exception("Failed to upload checkpoint from staging: %s", exc)
            raise UploadError(f"FULL upload failed: {exc}") from exc

    async def upload_delta(
        self,
        staging_path: Path,
        snapshot_metadata: dict[str, Any],
        target_window: int,
        prev_window: int,
        prev_state: dict[str, torch.Tensor],
        anchor_window: int,
    ) -> UploadResult:
        """Upload sparse delta checkpoint (chained).

        Computes sparse delta from prev_state (immediate predecessor) to current
        state, uploads only non-zero differences in COO format.

        For chained deltas:
        - prev_window/prev_state: The checkpoint this delta is computed against
        - anchor_window: The nearest FULL checkpoint for recovery metadata

        Args:
            staging_path: Path to staging directory containing current checkpoint
            snapshot_metadata: Metadata from snapshot (epoch, timestamp, metrics)
            target_window: The window number to publish this delta to
            prev_window: The window of the previous checkpoint (chained predecessor)
            prev_state: State dict of the previous checkpoint
            anchor_window: The nearest FULL checkpoint for recovery

        Returns:
            UploadResult with timing, size, and delta-specific metrics.

        Raises:
            UploadError: If the upload fails.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=f"delta-{target_window}-"))

        # Timing breakdown for granular metrics
        load_state_s = 0.0
        compute_delta_s = 0.0
        compression_s = 0.0
        network_upload_s = 0.0
        cleanup_s = 0.0

        try:
            # Load current weights from staging
            load_start = time.time()
            try:
                current_state = load_model_state_dict(staging_path)
            except Exception as exc:
                raise UploadError(f"Failed to load weights from {staging_path}: {exc}") from exc
            load_state_s = time.time() - load_start

            if current_state is None:
                raise UploadError(f"No model weights found in staging path: {staging_path}")

            # Compute sparse delta (chained: relative to prev checkpoint)
            compute_start = time.time()
            sparse_tensors, shapes, stats = compute_sparse_delta(
                current_state,
                prev_state,
                threshold=DELTA_THRESHOLD,
            )

            # Hash current_state directly - since we store actual values (not deltas),
            # the consumer's reconstruction will be identical to current_state.
            weights_hash = compute_weights_hash(current_state)
            logger.info(
                "[upload_delta] Hash computed: window=%s, hash=%s...",
                target_window,
                weights_hash[:16],
            )
            compute_delta_s = time.time() - compute_start

            # Encode sparse delta with selected codec (delta encoding + zstd)
            # This achieves 8-18x compression vs 1.3x with naive safetensors+zstd
            compress_start = time.time()
            delta_format, encoder = _select_delta_format()

            # Estimate raw size for logging (COO baseline; flattening is compression)
            delta_raw_size = _estimate_coo_raw_bytes(sparse_tensors)

            # Encode with selected sparse codec (includes zstd compression)
            compressed_bytes = encoder(sparse_tensors, shapes)
            delta_compressed_size = len(compressed_bytes)
            compression_s = time.time() - compress_start

            # Write compressed file
            delta_compressed_path = temp_dir / "delta_sparse.bin.zst"
            delta_compressed_path.write_bytes(compressed_bytes)

            compression_ratio = (
                delta_raw_size / delta_compressed_size if delta_compressed_size > 0 else 1.0
            )
            logger.info(
                "🗜️ Delta compression (%s, COO baseline): %.2f MB → %.2f MB (%.1fx ratio, %.1f%% reduction)",
                delta_format,
                delta_raw_size / (1024 * 1024),
                delta_compressed_size / (1024 * 1024),
                compression_ratio,
                (1 - delta_compressed_size / delta_raw_size) * 100 if delta_raw_size > 0 else 0,
            )

            # Save delta metadata (includes prev_window for chain verification)
            delta_meta = {
                "format": delta_format,
                "threshold": DELTA_THRESHOLD,
                "prev_window": prev_window,
                "anchor_window": anchor_window,
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

            # Delta checkpoints intentionally do NOT upload non-weight artifacts
            # (tokenizer/config/etc.) to keep delta payload minimal. Consumers
            # reconstruct a full checkpoint directory by copying these artifacts
            # from the base FULL checkpoint.

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

            # Get environment and generation configuration
            env_id, env_params = get_default_env_config()
            generation_params = get_default_generation_params()

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=parent_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=snapshot_metadata.get("timestamp", time.time()),
                model_name="async_trainer_snapshot",
                checkpoint_type=CHECKPOINT_TYPE_DELTA,
                prev_window=prev_window,
                anchor_window=anchor_window,
                weights_hash=weights_hash,
                env_id=env_id,
                env_params=env_params,
                generation_params=generation_params,
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

            # Upload to DELTA subdir
            remote_prefix = checkpoint_delta_prefix(target_window)

            logger.info(
                "Uploading delta checkpoint to %s (prev=%d, anchor=%d, sparsity=%.2f%%, %d non-zero params)",
                remote_prefix,
                prev_window,
                anchor_window,
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
            total_bytes = sum(path.stat().st_size for path in temp_dir.rglob("*") if path.is_file())
            total_mb = total_bytes / (1024 * 1024)

            network_start = time.time()
            results = await asyncio.gather(*upload_tasks)
            network_upload_s = time.time() - network_start

            if not all(results):
                raise UploadError("Some delta checkpoint files failed to upload")

            throughput_mbps = (total_mb / network_upload_s) if network_upload_s > 0 else 0.0
            logger.info(
                "Uploaded delta %.2f MB in %.1fs (%.1f MB/s) to %s",
                total_mb,
                network_upload_s,
                throughput_mbps,
                remote_prefix,
            )

            # Cleanup old checkpoints after successful upload
            cleanup_start = time.time()
            try:
                await self.cleanup_old_checkpoints(target_window)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to perform remote checkpoint cleanup: %s", exc)
            cleanup_s = time.time() - cleanup_start

            # Build result with timing breakdown
            timing = UploadTiming(
                load_state_s=load_state_s,
                compute_delta_s=compute_delta_s,
                compression_s=compression_s,
                network_upload_s=network_upload_s,
                cleanup_s=cleanup_s,
            )

            logger.info(
                "⏱️ Delta upload timing breakdown: load=%.1fs, compute=%.1fs, compress=%.1fs, "
                "network=%.1fs, cleanup=%.1fs (total=%.1fs)",
                timing.load_state_s,
                timing.compute_delta_s,
                timing.compression_s,
                timing.network_upload_s,
                timing.cleanup_s,
                timing.total_s,
            )

            return UploadResult(
                timing=timing,
                total_bytes=int(total_bytes),
                total_mb=float(total_mb),
                throughput_mbps=float(throughput_mbps),
                # Delta-specific fields
                sparsity_ratio=float(stats.get("sparsity_ratio", 0.0)),
                nonzero_params=int(stats.get("nonzero_params", 0)),
                total_params=int(stats.get("total_params", 0)),
                delta_threshold=float(DELTA_THRESHOLD),
                delta_raw_bytes=int(delta_raw_size),
                delta_compressed_bytes=int(delta_compressed_size),
                compression_ratio=float(compression_ratio),
            )

        except UploadError:
            # Re-raise UploadError as-is
            raise
        except Exception as exc:
            logger.exception("Failed to upload delta checkpoint: %s", exc)
            raise UploadError(f"Delta upload failed: {exc}") from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def upload_full_background(
        self,
        staging_path: Path,
        target_window: int,
    ) -> bool:
        """Upload FULL checkpoint in background (non-blocking, fire-and-forget).

        This is called at anchor windows while DELTA upload proceeds synchronously.
        FULL checkpoints serve as bootstrap anchors for new miners joining the network.

        Unlike upload_from_staging, this method:
        - Does not raise exceptions (logs errors instead)
        - Does not block the caller
        - Skips cleanup (DELTA upload handles that)

        Args:
            staging_path: Path to staged snapshot (already uploaded as DELTA)
            target_window: Window number to upload FULL checkpoint to
        """
        try:
            # Build file manifest from existing files
            file_manifest: dict[str, str] = {}
            for file_path in staging_path.rglob("*"):
                if file_path.is_file() and file_path.name != "snapshot_metadata.json":
                    rel_path = str(file_path.relative_to(staging_path))
                    file_manifest[rel_path] = hashlib.sha256(file_path.read_bytes()).hexdigest()

            # Read snapshot metadata
            snapshot_metadata_path = staging_path / "snapshot_metadata.json"
            if snapshot_metadata_path.exists():
                snapshot_metadata = json.loads(snapshot_metadata_path.read_text())
            else:
                snapshot_metadata = {}

            # Read training config
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

            # Create metadata for FULL checkpoint
            parent_window = snapshot_metadata.get("parent_window")
            if parent_window is None:
                parent_window = max(0, target_window - WINDOW_LENGTH)

            # Get environment and generation configuration
            env_id, env_params = get_default_env_config()
            generation_params = get_default_generation_params()

            metadata = CheckpointMetadata(
                window=target_window,
                parent_window=parent_window,
                file_manifest=file_manifest,
                training_config=training_config,
                git_commit=os.getenv("GIT_COMMIT", "unknown"),
                created_at=snapshot_metadata.get("timestamp", time.time()),
                model_name="async_trainer_snapshot",
                checkpoint_type=CHECKPOINT_TYPE_FULL,
                env_id=env_id,
                env_params=env_params,
                generation_params=generation_params,
            )

            config_hash = hashlib.sha256(
                json.dumps(training_config, sort_keys=True).encode()
            ).hexdigest()

            metadata_dict = {**metadata.__dict__, "config_hash": config_hash}

            # Create temp directory for background upload metadata
            temp_dir = Path(tempfile.mkdtemp(prefix=f"full-bg-{target_window}-"))

            try:
                # Copy model files to temp directory
                for file_path in staging_path.rglob("*"):
                    if file_path.is_file() and file_path.name != "snapshot_metadata.json":
                        rel_path = file_path.relative_to(staging_path)
                        dest_path = temp_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_path)

                # Write metadata
                metadata_path = temp_dir / "metadata.json"
                metadata_path.write_text(json.dumps(metadata_dict, ensure_ascii=False, indent=2))

                # Sign metadata
                canonical_metadata = json.dumps(
                    metadata_dict, sort_keys=True, separators=(",", ":")
                )
                signature = self.wallet.hotkey.sign(data=canonical_metadata).hex()
                (temp_dir / "manifest.sig").write_text(signature)

                # Write FULL marker
                (temp_dir / "FULL").touch()

                # Upload to FULL subdir
                remote_prefix = checkpoint_full_prefix(target_window)

                semaphore = asyncio.Semaphore(4)

                async def upload_file(path: Path) -> bool:
                    async with semaphore:
                        rel_path = path.relative_to(temp_dir)
                        remote_key = f"{remote_prefix}/{rel_path}"
                        try:
                            return await upload_file_chunked(
                                remote_key,
                                path.read_bytes(),
                                credentials=self.credentials,
                                use_write=True,
                                upload_timeout=UPLOAD_TIMEOUT,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "Background FULL upload file failed: %s: %s", rel_path, exc
                            )
                            return False

                upload_tasks = [upload_file(path) for path in temp_dir.rglob("*") if path.is_file()]
                results = await asyncio.gather(*upload_tasks)

                if all(results):
                    total_mb = sum(p.stat().st_size for p in temp_dir.rglob("*") if p.is_file()) / (
                        1024 * 1024
                    )
                    logger.info(
                        "✅ Background FULL upload completed: checkpoint-%s (%.1f MB)",
                        target_window,
                        total_mb,
                    )
                    return True
                else:
                    failed_count = sum(1 for r in results if not r)
                    logger.warning(
                        "Background FULL upload partially failed: checkpoint-%s (%d/%d files failed)",
                        target_window,
                        failed_count,
                        len(results),
                    )
                    return False

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as exc:
            # Background task - log error but don't propagate
            logger.warning(
                "Background FULL upload failed for checkpoint-%s: %s",
                target_window,
                exc,
            )
            return False
