"""Checkpoint management utilities (Consumer Role).

Central entry point for discovering, downloading, caching, and cleaning up
model checkpoints stored in R2. This module provides READ-ONLY operations for
miners and validators to consume checkpoints published by the trainer.

Checkpoints are published by the trainer under ``grail/checkpoints/checkpoint-{window}/``
with a manifest describing all artifacts and their SHA256 hashes.

Design goals:
 - Integrity validation for every download using manifest hashes.
 - Atomic download process to avoid partial/corrupt states.
 - Local cache with retention policy (last N + milestone windows).
 - Read-only operations: download, validate, cache management.

Checkpoint Retrieval Strategy:
 - With the trainer always publishing checkpoints for every window, this module
   now simply downloads and validates the requested checkpoint.
 - If a checkpoint is not ready (READY marker not found), returns None and the
   caller should wait/retry rather than falling back to an older checkpoint.
 - This ensures consistent model versions across miners and validators.

The module intentionally stays independent from model-loading details. It only
manages files on disk and in R2; callers handle loading into Torch/Transformers.

Note: Remote checkpoint publishing and deletion are handled by
grail.trainer.checkpoint_publisher module (producer role). This module should never
perform write operations to R2.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from grail.infrastructure.sparse_codec import (
    decode_sparse_delta_v2,
    decode_sparse_delta_v3,
    decode_sparse_delta_v3_1,
)
from grail.shared.safetensors_utils import load_model_state_dict

from ..shared.checkpoint_paths import (
    checkpoint_delta_metadata_key,
    checkpoint_delta_prefix,
    checkpoint_full_metadata_key,
    checkpoint_full_prefix,
    checkpoint_window_prefix,
    parse_window_from_prefix,
)
from ..shared.constants import (
    BASE_CHECKPOINT_RETENTION_LIMIT,
    CHECKPOINT_PREFIX,
    CHECKPOINT_TYPE_DELTA,
    CHECKPOINT_TYPE_FULL,
    GRAIL_CHECKPOINT_MOD10,
)
from . import comms
from .delta_checkpoint import apply_sparse_delta, compute_weights_hash

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                             Metadata Schema                                 #
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointMetadata:
    """Metadata describing a checkpoint directory."""

    window: int
    file_manifest: dict[str, str]
    training_config: dict[str, Any] = field(default_factory=dict)
    git_commit: str = "unknown"
    created_at: float = 0.0
    model_name: str = "no_name"
    parent_window: int | None = None

    # Delta checkpoint fields (chained deltas)
    checkpoint_type: str = CHECKPOINT_TYPE_FULL  # "FULL" or "DELTA"
    prev_window: int | None = None  # For DELTA: immediate predecessor checkpoint (chained)
    anchor_window: int | None = None  # For DELTA: nearest FULL checkpoint for recovery
    weights_hash: str | None = None  # SHA256 of final weights for verification

    # Environment and generation configuration
    env_id: str | None = None  # Environment identifier (e.g., "mbpp", "gsm8k")
    env_params: dict[str, Any] = field(default_factory=dict)  # Environment-specific parameters
    generation_params: dict[str, Any] = field(
        default_factory=dict
    )  # Generation parameters (max_tokens, temperature, etc.)

    def validate_metadata(self) -> list[str]:
        """Validate that required fields for mining/validation are present.

        Returns a list of missing field names. Empty list = valid.
        """
        missing = []
        if not self.env_id:
            missing.append("env_id")
        if not self.generation_params:
            missing.append("generation_params")
        elif "max_tokens" not in self.generation_params:
            missing.append("generation_params.max_tokens")
        return missing

    def remote_prefix(self) -> str:
        """Get the remote R2 prefix for this checkpoint.

        DELTA checkpoints live under a dedicated sub-prefix so they can coexist
        with a FULL anchor checkpoint at the same window.
        """
        if self.is_delta():
            return checkpoint_delta_prefix(self.window)
        return checkpoint_full_prefix(self.window)

    def is_delta(self) -> bool:
        """Check if this is a delta checkpoint."""
        return self.checkpoint_type == CHECKPOINT_TYPE_DELTA


class CheckpointDownloadError(RuntimeError):
    """Raised when checkpoint download or validation fails."""


@dataclass
class CheckpointLoadResult:
    """Result of a checkpoint load operation."""

    success: bool
    window: int | None = None
    path: Path | None = None
    method: str = "none"  # "fast" (in-place), "full" (disk load), or "none"

    @property
    def is_fast_path(self) -> bool:
        return self.method == "fast"


# --------------------------------------------------------------------------- #
#                          Checkpoint Manager                                  #
# --------------------------------------------------------------------------- #


class CheckpointManager:
    """Manage checkpoint discovery, downloads, and cache cleanup (Consumer Role).

    This class provides READ-ONLY operations for discovering, downloading, and
    validating checkpoints. It is used by miners and validators to consume
    checkpoints published by the trainer.

    Write operations (publishing, remote deletion) are handled by the
    grail.trainer.checkpoint_publisher module.
    """

    def __init__(
        self,
        *,
        cache_root: Path,
        credentials: Any | None,
        keep_limit: int = BASE_CHECKPOINT_RETENTION_LIMIT,
    ) -> None:
        self.cache_root = cache_root.expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.credentials = credentials
        self.keep_limit = max(1, keep_limit)
        # Cache FULL and DELTA metadata separately per window.
        self._metadata_cache: dict[tuple[int, str], CheckpointMetadata] = {}
        self._download_locks: dict[int, asyncio.Lock] = {}
        self._fallback_attempted: set[int] = set()
        self._last_cache_task: asyncio.Task[None] | None = None
        self._last_cache_window: int | None = None

    # ----------------------------- High-level API --------------------------- #

    async def get_checkpoint(self, window: int) -> Path | None:
        """Ensure checkpoint for *window* is available locally and return path.

        With the trainer always publishing checkpoints for every window, this method
        now simply downloads and validates the requested checkpoint without fallback.
        If the checkpoint is not ready, returns None and the caller should wait/retry.

        Notes (testing only):
        If GRAIL_CHECKPOINT_MOD10 == True, the incoming window is
        deterministically remapped to [0..9] via modulo 10 to allow
        testing against a small fixed set of checkpoints.

        Args:
            window: Target window for checkpoint

        Returns:
            Path to local checkpoint directory, or None if not available/ready
        """

        # Testing hook: remap any input window to [0..9] when enabled
        if GRAIL_CHECKPOINT_MOD10:
            original_window = window
            # If window values are multiples of 10, map deterministically to [0..9]
            # by first collapsing the decade, then mod 10.
            window = (int(window) // 10) % 10
            logger.debug("[TEST MOD10] remapped window %s -> %s", original_window, window)

        if window < 0:
            return None

        local_dir = self.cache_root / f"checkpoint-{window}"
        lock = self._download_locks.setdefault(window, asyncio.Lock())

        async with lock:
            # For slow-path disk downloads, prefer FULL metadata when available.
            # FULL checkpoints (often uploaded in background at anchor windows) are the
            # most reliable recovery points and avoid long delta-chain reconstruction.
            metadata = await self._fetch_full_metadata(window)
            if metadata is None:
                metadata = await self._fetch_delta_metadata(window)
            if metadata is None:
                logger.debug(
                    "No metadata.json for window %s — attempting best-effort download",
                    window,
                )

            if local_dir.exists():
                try:
                    manifest = await self._load_manifest(local_dir)
                    if manifest and await self._verify_integrity(local_dir, manifest):
                        return local_dir
                    logger.warning("Cached checkpoint for window %s failed verification", window)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("Failed to verify cached checkpoint: %s", exc)
                shutil.rmtree(local_dir, ignore_errors=True)

            # Check READY-{window} marker to ensure checkpoint is fully uploaded
            ready_window = await self._get_checkpoint_ready_window(window)
            if ready_window is None:
                logger.warning(
                    "Checkpoint for window %s not ready (READY-{window} marker not found); will retry later",
                    window,
                )
                return None

            logger.debug(
                "Checkpoint-%s became ready at window %s",
                window,
                ready_window,
            )

            tmp_dir = _unique_partial_dir(self.cache_root, f"checkpoint-{window}")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info("Starting checkpoint download for window %s", window)

                # Handle DELTA checkpoint: walk chain to anchor, apply deltas sequentially
                if metadata is not None and metadata.is_delta():
                    logger.info(
                        "Checkpoint %s is DELTA (prev=%s, anchor=%s), will reconstruct chain",
                        window,
                        metadata.prev_window,
                        metadata.anchor_window,
                    )
                    reconstructed_dir = await self._handle_delta_checkpoint(metadata, tmp_dir)
                    if reconstructed_dir is None:
                        raise CheckpointDownloadError(
                            f"Failed to reconstruct delta checkpoint for window {window}"
                        )
                    # tmp_dir now contains the reconstructed full checkpoint
                elif metadata is not None and metadata.file_manifest:
                    # Preferred path: use manifest for exact file set and integrity checks
                    logger.debug(
                        "Downloading %s files for checkpoint window %s using manifest",
                        len(metadata.file_manifest),
                        window,
                    )
                    await self._download_files(metadata, tmp_dir)
                    logger.debug("Verifying integrity for checkpoint window %s", window)
                    if not await self._verify_integrity(tmp_dir, metadata.file_manifest):
                        raise CheckpointDownloadError(f"Integrity check failed for window {window}")

                    # Persist manifest locally for later offline verification
                    # TODO: make this meta_data handling more neat and apply DRY later
                    manifest_path = tmp_dir / "metadata.json"
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "window": metadata.window,
                                "file_manifest": metadata.file_manifest,
                                "training_config": metadata.training_config,
                                "git_commit": metadata.git_commit,
                                "created_at": metadata.created_at,
                                "model_name": metadata.model_name,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                else:
                    # Fallback path: list and download everything under the prefix
                    logger.debug(
                        "Downloading checkpoint window %s without manifest (best-effort)",
                        window,
                    )
                    await self._download_all_in_prefix(window, tmp_dir)

                logger.info("Checkpoint download completed for window %s, finalizing...", window)
                if local_dir.exists():
                    shutil.rmtree(local_dir, ignore_errors=True)
                shutil.move(str(tmp_dir), str(local_dir))
                logger.info("✅ Checkpoint for window %s ready at %s", window, local_dir)
                return local_dir
            except Exception as exc:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.error(
                    "Checkpoint download/integrity failed for window %s: %s",
                    window,
                    exc,
                )
                return None

    async def apply_delta_in_place(
        self,
        model: Any,  # torch.nn.Module
        target_window: int,
        current_window: int,
    ) -> bool:
        """Apply delta checkpoint directly to in-memory model (fast path).

        This avoids disk I/O by downloading the delta and applying it directly
        to the model's state_dict in GPU/CPU memory. Much faster than the full
        get_checkpoint + load_model path for continuously running miners/validators.

        Requirements:
        - Model must already be loaded with weights from current_window
        - target_window must be a DELTA checkpoint with prev_window == current_window

        Args:
            model: The loaded PyTorch model (torch.nn.Module) with weights from current_window
            target_window: The window to update to
            current_window: The window the model currently has loaded

        Returns:
            True if delta was applied successfully, False if fallback to full load needed
        """
        # Validate inputs
        if target_window <= current_window:
            logger.debug(
                "target_window %s <= current_window %s, no update needed",
                target_window,
                current_window,
            )
            return False

        # Fetch metadata for target window
        metadata = await self._fetch_metadata(target_window)
        if metadata is None:
            logger.debug("No metadata for window %s, fallback to full load", target_window)
            return False

        # Check if it's a delta from our current checkpoint
        if not metadata.is_delta():
            logger.debug("Window %s is FULL checkpoint, fallback to full load", target_window)
            return False

        if metadata.prev_window != current_window:
            logger.info(
                "Delta prev_window %s != current_window %s, fallback to full load",
                metadata.prev_window,
                current_window,
            )
            return False

        # Check READY marker
        ready_window = await self._get_checkpoint_ready_window(target_window)
        if ready_window is None:
            logger.debug("Checkpoint %s not ready, fallback to full load", target_window)
            return False

        try:
            logger.info(
                "⚡ Fast path: applying delta in-place from window %s → %s",
                current_window,
                target_window,
            )

            # Download and parse delta
            delta_data = await self._download_and_load_delta(metadata)
            if delta_data is None:
                logger.warning("Failed to download delta for window %s", target_window)
                return False

            sparse_tensors, shapes, delta_info = delta_data

            # Get model's current state dict (on device)
            current_state = model.state_dict()

            # Apply delta - dtype is inferred from current_state
            logger.debug(
                "Applying delta: %.2f%% sparse, %d params changed",
                delta_info.get("sparsity_ratio", 0) * 100,
                len(sparse_tensors),
            )

            reconstructed = apply_sparse_delta(
                current_state,
                sparse_tensors,
                shapes,
                target_dtype=None,  # Infer from current_state
            )

            # Verify hash if available
            if metadata.weights_hash:
                actual_hash = compute_weights_hash(reconstructed)
                if actual_hash != metadata.weights_hash:
                    logger.error(
                        "Hash mismatch after in-place delta: expected %s..., got %s...",
                        metadata.weights_hash[:16],
                        actual_hash[:16],
                    )
                    return False
                logger.debug("✅ Hash verified for window %s", target_window)

            # Load reconstructed weights back into model
            model.load_state_dict(reconstructed, strict=True)

            # Update the checkpoint window attribute for validation
            # (This attribute is normally set by get_model() during slow path)
            model.grail_checkpoint_window = target_window

            logger.info(
                "✅ Fast path complete: model updated to window %s in-place",
                target_window,
            )

            # Cache the updated checkpoint to disk for future cold starts.
            # Store the task so callers (e.g. pipeline weight sync) can await it.
            self._last_cache_task = asyncio.create_task(
                self._cache_model_state_async(reconstructed, metadata, target_window)
            )
            self._last_cache_window = target_window

            return True

        except Exception as exc:
            logger.warning(
                "Fast path failed for window %s, fallback to full load: %s",
                target_window,
                exc,
            )
            return False

    async def await_cache_complete(self) -> Path | None:
        """Wait for the last cache task to finish and return the cached path.

        Used by the pipeline weight sync to get the disk path for reloading
        the generation server after a fast-path delta update.

        Returns:
            Path to the cached checkpoint directory, or None if no cache
            was written or the path does not exist.
        """
        if self._last_cache_task is not None:
            try:
                await self._last_cache_task
            except Exception as exc:
                logger.debug("Cache task failed: %s", exc)
            self._last_cache_task = None

        if self._last_cache_window is not None:
            path = self.cache_root / f"checkpoint-{self._last_cache_window}"
            return path if path.exists() else None
        return None

    async def load_or_update_model(
        self,
        target_window: int,
        model: Any | None,
        current_window: int | None,
    ) -> tuple[CheckpointLoadResult, Path | None]:
        """Unified checkpoint loading: tries fast path first, falls back to full load.

        This is the recommended entry point for miners and validators. It handles:
        1. Fast path: In-place delta update (if model loaded and delta available)
        2. Slow path: Full checkpoint download (cold start or fast path unavailable)

        Args:
            target_window: Window to load checkpoint for
            model: Currently loaded model (None for cold start)
            current_window: Window the model currently has (None for cold start)

        Returns:
            Tuple of (CheckpointLoadResult, checkpoint_path or None)
            - result.success: True if checkpoint is ready
            - result.window: The actual checkpoint window loaded
            - result.method: "fast" or "full" or "none"
            - checkpoint_path: Path to checkpoint (None if fast path used)
        """
        # Always target the latest checkpoint that was READY before *target_window*.
        #
        # Rationale:
        # - Validators enforce that miner rollouts were generated with the same checkpoint
        #   the validator is using for that window.
        # - On cold start, selecting only a FULL checkpoint can lag behind newer DELTA
        #   checkpoints (which are still valid and reconstructable), causing
        #   CHECKPOINT_MISMATCH and false rejections.
        #
        # We still preserve cold-start efficiency via get_checkpoint(): if the target
        # checkpoint is DELTA, it reconstructs from the nearest cached/FULL base.
        checkpoint_window = await self.get_latest_ready_checkpoint(target_window)
        if checkpoint_window is None:
            return CheckpointLoadResult(success=False), None

        # Already at this checkpoint
        if checkpoint_window == current_window and model is not None:
            return CheckpointLoadResult(success=True, window=checkpoint_window, method="none"), None

        # Try fast path if model is loaded
        if model is not None and current_window is not None and current_window >= 0:
            try:
                if await self.apply_delta_in_place(model, checkpoint_window, current_window):
                    return CheckpointLoadResult(
                        success=True, window=checkpoint_window, method="fast"
                    ), None
            except Exception as exc:
                logger.debug("Fast path failed: %s", exc)

        # Slow path: full checkpoint download
        checkpoint_path = await self.get_checkpoint(checkpoint_window)
        if checkpoint_path is None:
            return CheckpointLoadResult(success=False), None

        return CheckpointLoadResult(
            success=True,
            window=checkpoint_window,
            path=checkpoint_path,
            method="full",
        ), checkpoint_path

    async def _cache_model_state_async(
        self,
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata,
        window: int,
    ) -> None:
        """Cache reconstructed model state to disk asynchronously.

        This runs in the background so in-place updates don't block on disk I/O.
        Useful for cold starts - the cached checkpoint can be loaded directly.
        """
        try:
            local_dir = self.cache_root / f"checkpoint-{window}"
            if local_dir.exists():
                return  # Already cached

            # Get a reference checkpoint path for non-weight files
            # Try to find the previous checkpoint
            prev_window = metadata.prev_window
            if prev_window is None:
                return

            prev_path = self.cache_root / f"checkpoint-{prev_window}"
            if not prev_path.exists():
                return

            # Write in background thread to avoid blocking
            await asyncio.to_thread(
                self._write_cached_checkpoint,
                state_dict,
                prev_path,
                local_dir,
                metadata,
            )
            logger.debug("Cached in-place updated checkpoint to %s", local_dir)
        except Exception as exc:
            # Non-critical - just log and continue
            logger.debug("Failed to cache in-place checkpoint: %s", exc)

    def _write_cached_checkpoint(
        self,
        state_dict: dict[str, Any],
        prev_path: Path,
        output_dir: Path,
        metadata: CheckpointMetadata,
    ) -> None:
        """Write cached checkpoint synchronously (called from thread)."""
        from safetensors.torch import save_file

        tmp_dir = _unique_partial_dir(output_dir.parent, output_dir.name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model weights
            weights_path = tmp_dir / "model.safetensors"
            save_file(state_dict, str(weights_path))

            # Copy non-weight files from previous checkpoint
            for src_file in prev_path.iterdir():
                if src_file.name in ("model.safetensors", "metadata.json"):
                    continue
                dst_file = tmp_dir / src_file.name
                if src_file.is_file():
                    shutil.copy2(src_file, dst_file)

            # Write metadata marking this as FULL (reconstructed)
            meta_dict = {
                "window": metadata.window,
                "file_manifest": {},  # Will be populated on next verify
                "checkpoint_type": "FULL",  # Reconstructed
                "weights_hash": metadata.weights_hash,
            }
            (tmp_dir / "metadata.json").write_text(json.dumps(meta_dict, indent=2))

            # Atomic move
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.move(str(tmp_dir), str(output_dir))

        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    async def cleanup_local(self, current_window: int) -> None:
        """Remove cached checkpoints outside the retention policy."""

        keep_windows = self._compute_keep_windows(current_window)

        # Apply MOD10 remapping if enabled (must match get_checkpoint)
        if GRAIL_CHECKPOINT_MOD10:
            # Remap each window to [0..9] to match cached names
            remapped_keep = {(int(w) // 10) % 10 for w in keep_windows}
            keep_windows = remapped_keep
            logger.debug(
                "[TEST MOD10] remapped keep_windows for cleanup: %s",
                keep_windows,
            )

        for candidate in self.cache_root.glob("checkpoint-*"):
            try:
                suffix = candidate.name.split("-")[1]
                window = int(suffix)
            except (IndexError, ValueError):
                continue

            if window not in keep_windows:
                logger.debug("Removing local checkpoint %s (window %s)", candidate, window)
                try:
                    shutil.rmtree(candidate)
                except Exception as e:
                    logger.error(
                        "Failed to delete checkpoint %s: %s",
                        candidate,
                        e,
                        exc_info=True,
                    )

    async def list_remote_windows(self) -> list[int]:
        """Return all checkpoint window numbers available in R2."""

        keys = await comms.list_bucket_files(
            CHECKPOINT_PREFIX,
            credentials=self.credentials,
            use_write=False,
        )
        windows: set[int] = set()
        for key in keys:
            # Extract window number using shared utility
            window = parse_window_from_prefix(key)
            if window is not None:
                windows.add(window)
        return sorted(windows)

    async def _get_checkpoint_ready_window(self, checkpoint_window: int) -> int | None:
        """Get the ready_window for a checkpoint by parsing READY-{window} marker.

        Args:
            checkpoint_window: The checkpoint directory window

        Returns:
            The ready_window (when upload finished), or None if not ready
        """
        try:
            # List files in checkpoint directory
            prefix = checkpoint_window_prefix(checkpoint_window) + "/"
            keys = await comms.list_bucket_files(
                prefix,
                credentials=self.credentials,
                use_write=False,
            )

            # Find READY-{window} marker
            for key in keys:
                if "/READY-" in key:
                    # Extract ready_window from "checkpoint-1000/READY-1100"
                    filename = key.split("/")[-1]
                    if filename.startswith("READY-"):
                        ready_window = int(filename.split("-")[1])
                        return ready_window

            return None
        except Exception as exc:
            logger.debug("Failed to get ready_window for checkpoint %s: %s", checkpoint_window, exc)
            return None

    async def get_recent_checkpoints(self, n: int) -> list[Path]:
        """Get the N most recent checkpoints available locally or remotely.

        Args:
            n: Number of recent checkpoints to retrieve

        Returns:
            List of paths to local checkpoint directories (most recent first)
        """
        windows = await self.list_remote_windows()
        if not windows:
            return []

        # Take the N most recent windows
        recent_windows = sorted(windows, reverse=True)[:n]

        # Download each checkpoint if not already local
        checkpoint_paths: list[Path] = []
        for window in recent_windows:
            checkpoint_path = await self.get_checkpoint(window)
            if checkpoint_path:
                checkpoint_paths.append(checkpoint_path)

        return checkpoint_paths

    async def get_checkpoints_for_windows(self, windows: list[int]) -> dict[int, Path]:
        """Get checkpoints for specific windows.

        Args:
            windows: List of window numbers to fetch

        Returns:
            Dict mapping window number to local checkpoint path (only successful)
        """
        results: dict[int, Path] = {}

        for window in windows:
            checkpoint_path = await self.get_checkpoint(window)
            if checkpoint_path:
                results[window] = checkpoint_path

        return results

    # --------------------------- Public API --------------------------- #

    async def get_checkpoint_metadata(self, window: int) -> CheckpointMetadata | None:
        """Public API: Fetch checkpoint metadata for a given window.

        Prefers DELTA metadata when available (for fast-path compatibility),
        falls back to FULL metadata if DELTA not found.

        Args:
            window: Window number to fetch metadata for

        Returns:
            CheckpointMetadata if found, None otherwise

        Example:
            >>> metadata = await checkpoint_manager.get_checkpoint_metadata(15000)
            >>> if metadata:
            ...     env_id = metadata.env_id
            ...     generation_params = metadata.generation_params
        """
        if window is None:
            return None
        return await self._fetch_metadata(window)

    # --------------------------- Internal helpers --------------------------- #

    async def _fetch_metadata(self, window: int) -> CheckpointMetadata | None:
        """Fetch checkpoint metadata, preferring DELTA when available.

        This enables continuously running miners/validators to use the delta fast-path
        even at anchor windows (where both FULL and DELTA may exist).
        """
        metadata = await self._fetch_delta_metadata(window)
        if metadata is not None:
            return metadata
        return await self._fetch_full_metadata(window)

    async def _fetch_delta_metadata(self, window: int) -> CheckpointMetadata | None:
        """Fetch DELTA checkpoint metadata from DELTA subdir."""
        cache_key = (window, CHECKPOINT_TYPE_DELTA)
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        remote_key = checkpoint_delta_metadata_key(window)
        payload = await comms.get_file(remote_key, credentials=self.credentials, use_write=False)
        if not payload:
            return None

        metadata = CheckpointMetadata(
            window=payload.get("window", window),
            file_manifest=payload.get("file_manifest", {}),
            training_config=payload.get("training_config", {}),
            git_commit=payload.get("git_commit", "unknown"),
            created_at=float(payload.get("created_at", 0.0)),
            model_name=payload.get("model_name", "no_name"),
            checkpoint_type=payload.get("checkpoint_type", CHECKPOINT_TYPE_DELTA),
            prev_window=payload.get("prev_window"),
            anchor_window=payload.get("anchor_window"),
            weights_hash=payload.get("weights_hash"),
            env_id=payload.get("env_id"),
            env_params=payload.get("env_params", {}),
            generation_params=payload.get("generation_params", {}),
        )
        self._metadata_cache[cache_key] = metadata
        return metadata

    async def _fetch_full_metadata(self, window: int) -> CheckpointMetadata | None:
        """Fetch FULL checkpoint metadata from FULL subdir."""
        cache_key = (window, CHECKPOINT_TYPE_FULL)
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        remote_key = checkpoint_full_metadata_key(window)
        logger.debug("Fetching FULL metadata from %s", remote_key)
        payload = await comms.get_file(remote_key, credentials=self.credentials, use_write=False)
        if not payload:
            logger.debug("FULL metadata not found at %s", remote_key)
            return None
        logger.debug("Found FULL metadata for window %s", window)

        metadata = CheckpointMetadata(
            window=payload.get("window", window),
            file_manifest=payload.get("file_manifest", {}),
            training_config=payload.get("training_config", {}),
            git_commit=payload.get("git_commit", "unknown"),
            created_at=float(payload.get("created_at", 0.0)),
            model_name=payload.get("model_name", "no_name"),
            checkpoint_type=payload.get("checkpoint_type", CHECKPOINT_TYPE_FULL),
            prev_window=payload.get("prev_window"),
            anchor_window=payload.get("anchor_window"),
            weights_hash=payload.get("weights_hash"),
            env_id=payload.get("env_id"),
            env_params=payload.get("env_params", {}),
            generation_params=payload.get("generation_params", {}),
        )
        self._metadata_cache[cache_key] = metadata
        return metadata

    async def _load_manifest(self, checkpoint_dir: Path) -> dict[str, str] | None:
        manifest_path = checkpoint_dir / "metadata.json"
        if not manifest_path.exists():
            return None
        try:
            data = json.loads(manifest_path.read_text())
            manifest = data.get("file_manifest")
            if isinstance(manifest, dict):
                return manifest
        except Exception:
            logger.debug("Failed to read manifest from %s", manifest_path, exc_info=True)
        return None

    async def _download_files(self, metadata: CheckpointMetadata, tmp_dir: Path) -> None:
        semaphore = asyncio.Semaphore(8)

        async def _download(filename: str) -> None:
            async with semaphore:
                remote_key = f"{metadata.remote_prefix()}/{filename}"
                data = await comms.download_file_chunked(
                    remote_key,
                    credentials=self.credentials,
                    use_write=False,
                )
                if data is None:
                    raise CheckpointDownloadError(f"Missing file {filename}")

                target_path = tmp_dir / filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_bytes(data)

        await asyncio.gather(*(_download(name) for name in metadata.file_manifest.keys()))

    async def _download_all_in_prefix(self, window: int, tmp_dir: Path) -> None:
        """Best-effort download of all objects under the checkpoint prefix.

        Used when metadata.json is missing or incomplete. Skips integrity
        verification and simply mirrors the prefix into tmp_dir.

        Tries FULL subdir first (for bootstrap), then DELTA subdir.
        """
        # Try FULL first (for new joiners), then DELTA
        prefix_dir = checkpoint_full_prefix(window) + "/"
        keys = await comms.list_bucket_files(
            prefix_dir, credentials=self.credentials, use_write=False
        )
        if not keys:
            raise CheckpointDownloadError(f"No files found at prefix {prefix_dir}")

        asyncio.Semaphore(6)

        async def _dl(key: str) -> None:
            if not key or not key.startswith(prefix_dir) or key.endswith("/"):
                return
            rel = key[len(prefix_dir) :]

            # Strip .gz extension from filename since download_file_chunked auto-decompresses
            # This ensures config.json.gz is saved as config.json (decompressed)
            if rel.endswith(".gz"):
                rel = rel[:-3]
                logger.debug("Stripping .gz from filename: %s -> %s", key, rel)

            data = await comms.download_file_chunked(
                key, credentials=self.credentials, use_write=False
            )
            if data is None:
                raise CheckpointDownloadError(f"Missing file {key}")
            target_path = tmp_dir / rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(data)

        await asyncio.gather(*(_dl(k) for k in keys))

    async def _verify_integrity(self, checkpoint_dir: Path, manifest: dict[str, str]) -> bool:
        for filename, expected_hash in manifest.items():
            file_path = checkpoint_dir / filename
            if not file_path.exists():
                logger.warning("Missing checkpoint file %s", file_path)
                return False

            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if digest != expected_hash:
                logger.warning(
                    "Checksum mismatch for %s (expected=%s actual=%s)",
                    file_path,
                    expected_hash,
                    digest,
                )
                return False
        return True

    async def _handle_delta_checkpoint(
        self,
        metadata: CheckpointMetadata,
        tmp_dir: Path,
    ) -> Path | None:
        """Handle DELTA checkpoint: apply single delta or reconstruct from chain.

        Optimized for the common case where the miner is running continuously:
        - Fast path: If prev_window is cached locally, apply single delta directly
        - Slow path: If prev_window missing, walk chain to nearest cached checkpoint

        Chain reconstruction is only needed for:
        - Cold start (miner just started)
        - Missed windows (miner was offline)
        - Cache invalidation

        Args:
            metadata: Metadata for the target delta checkpoint
            tmp_dir: Temporary directory for the reconstructed checkpoint

        Returns:
            Path to reconstructed checkpoint, or None if reconstruction failed
        """
        if metadata.prev_window is None:
            logger.error("DELTA checkpoint %s missing prev_window", metadata.window)
            return None

        # Fast path: Check if prev_window is cached locally
        prev_local_path = self.cache_root / f"checkpoint-{metadata.prev_window}"
        if prev_local_path.exists():
            try:
                prev_manifest = await self._load_manifest(prev_local_path)
                if prev_manifest and await self._verify_integrity(prev_local_path, prev_manifest):
                    logger.info(
                        "Fast path: applying single delta %s → %s (prev cached)",
                        metadata.prev_window,
                        metadata.window,
                    )
                    # Apply single delta directly to cached prev checkpoint
                    return await self._apply_single_delta(prev_local_path, metadata, tmp_dir)
            except Exception as exc:
                logger.debug("Fast path check failed, falling back to chain: %s", exc)

        # Slow path: Build chain to nearest cached/FULL checkpoint
        logger.info(
            "Slow path: building delta chain for window %s (prev %s not cached)",
            metadata.window,
            metadata.prev_window,
        )

        chain = await self._build_delta_chain(metadata)
        if chain is None:
            return None

        base_path, delta_chain = chain

        # Log recovery mode when catching up multiple missed windows
        if len(delta_chain) > 1:
            first_delta = delta_chain[0]
            last_delta = delta_chain[-1]
            logger.info(
                "🔄 Recovery mode: catching up %d missed windows (%s -> %s)",
                len(delta_chain),
                first_delta.prev_window,
                last_delta.window,
            )
        else:
            logger.info(
                "Built delta chain: base=%s (cached=%s), chain_length=%d, target=%s",
                delta_chain[0].prev_window if delta_chain else "N/A",
                base_path is not None,
                len(delta_chain),
                metadata.window,
            )

        if base_path is None:
            logger.error("Cannot find base checkpoint for chain reconstruction")
            return None

        # Apply chain of deltas
        return await self._apply_delta_chain(base_path, delta_chain, tmp_dir)

    async def _apply_single_delta(
        self,
        prev_path: Path,
        delta_metadata: CheckpointMetadata,
        output_dir: Path,
    ) -> Path | None:
        """Apply a single delta to a cached checkpoint (fast path).

        Args:
            prev_path: Path to the previous checkpoint (cached locally)
            delta_metadata: Metadata for the delta to apply
            output_dir: Directory to write reconstructed checkpoint

        Returns:
            Path to reconstructed checkpoint, or None on failure
        """
        try:
            # Download and load delta
            delta_data = await self._download_and_load_delta(delta_metadata)
            if delta_data is None:
                logger.error("Failed to download delta for window %s", delta_metadata.window)
                return None

            sparse_tensors, shapes, delta_info = delta_data

            # Load previous weights
            prev_state = load_model_state_dict(prev_path)
            if prev_state is None:
                logger.error("Failed to load weights from %s", prev_path)
                return None

            logger.debug(
                "Applying single delta: prev=%s, target=%s (%.2f%% sparse)",
                delta_metadata.prev_window,
                delta_metadata.window,
                delta_info.get("sparsity_ratio", 0) * 100,
            )

            # Apply delta - dtype is inferred from prev_state
            reconstructed = apply_sparse_delta(
                prev_state,
                sparse_tensors,
                shapes,
                target_dtype=None,  # Infer from prev_state
            )

            # Verify hash
            if delta_metadata.weights_hash:
                actual_hash = compute_weights_hash(reconstructed)
                if actual_hash != delta_metadata.weights_hash:
                    logger.error(
                        "Weights hash mismatch for window %s: expected %s..., got %s...",
                        delta_metadata.window,
                        delta_metadata.weights_hash[:16],
                        actual_hash[:16],
                    )
                    return None
                logger.debug("✅ Hash verified for window %s", delta_metadata.window)

            # Write reconstructed checkpoint
            return await self._write_reconstructed_checkpoint(
                reconstructed,
                prev_path,  # Copy non-weight files from prev
                output_dir,
                delta_metadata,
            )

        except Exception as exc:
            logger.exception("Failed to apply single delta: %s", exc)
            return None

    async def _build_delta_chain(
        self,
        target_metadata: CheckpointMetadata,
    ) -> tuple[Path, list[CheckpointMetadata]] | None:
        """Walk backwards from target to nearest cached/FULL checkpoint.

        Stops early if it finds a cached checkpoint locally, avoiding unnecessary
        chain reconstruction for continuously running miners.

        Args:
            target_metadata: Metadata for the target delta checkpoint

        Returns:
            Tuple of (base_path, list of delta metadata in order oldest-first),
            or None if chain cannot be built
        """
        chain: list[CheckpointMetadata] = []
        current = target_metadata
        anchors_checked: list[int] = []  # Track anchor windows we tried to use

        while current.checkpoint_type == "DELTA":
            chain.append(current)

            if current.prev_window is None:
                logger.error("DELTA checkpoint %s missing prev_window", current.window)
                return None

            # Check if prev_window is cached locally (early termination)
            prev_local_path = self.cache_root / f"checkpoint-{current.prev_window}"
            if prev_local_path.exists():
                try:
                    prev_manifest = await self._load_manifest(prev_local_path)
                    if prev_manifest and await self._verify_integrity(
                        prev_local_path, prev_manifest
                    ):
                        logger.debug(
                            "Chain early termination: found cached checkpoint %s",
                            current.prev_window,
                        )
                        chain.reverse()  # Order: oldest delta first
                        return (prev_local_path, chain)
                except Exception:
                    pass  # Continue walking if cache check fails

            # Prefer stopping at a FULL anchor when available (anchor windows may also
            # have a DELTA published for sequential, in-place updates).
            prev_meta: CheckpointMetadata | None = None
            try:
                from grail.shared.retention_utils import is_anchor_window

                if is_anchor_window(current.prev_window):
                    anchors_checked.append(current.prev_window)
                    prev_meta = await self._fetch_full_metadata(current.prev_window)
                    if prev_meta is None:
                        logger.warning(
                            "No FULL metadata at anchor window %s, will use DELTA instead",
                            current.prev_window,
                        )
                    else:
                        logger.info(
                            "Found FULL at anchor window %s, stopping chain walk",
                            current.prev_window,
                        )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch FULL metadata for anchor %s: %s",
                    current.prev_window,
                    exc,
                )
                prev_meta = None

            if prev_meta is None:
                # Normal case: prefer DELTA for intermediate windows, fallback to FULL.
                prev_meta = await self._fetch_metadata(current.prev_window)
            if prev_meta is None:
                logger.error("Cannot fetch metadata for chain link %s", current.prev_window)
                return None

            current = prev_meta

        # current is now a FULL checkpoint - get it
        logger.info(
            "Delta chain terminated at FULL checkpoint %s (chain length=%d, anchors checked without FULL: %s)",
            current.window,
            len(chain),
            anchors_checked if anchors_checked else "none",
        )
        base_path = await self.get_checkpoint(current.window)
        if base_path is None:
            logger.error(
                "Cannot get FULL checkpoint %s (READY marker missing or download failed)",
                current.window,
            )
            return None

        chain.reverse()  # Order: oldest delta first
        return (base_path, chain)

    async def _apply_delta_chain(
        self,
        anchor_path: Path,
        delta_chain: list[CheckpointMetadata],
        output_dir: Path,
    ) -> Path | None:
        """Apply chain of deltas sequentially, casting to bf16 after each.

        Args:
            anchor_path: Path to anchor (FULL) checkpoint directory
            delta_chain: List of delta metadata in order (oldest first)
            output_dir: Directory to write final reconstructed checkpoint

        Returns:
            Path to reconstructed checkpoint directory, or None on failure
        """
        try:
            # Load anchor weights
            current_state = load_model_state_dict(anchor_path)
            if current_state is None:
                logger.error("Anchor checkpoint missing model weights: %s", anchor_path)
                return None

            # Apply each delta in order
            for i, delta_meta in enumerate(delta_chain):
                logger.debug(
                    "Applying delta %d/%d: window %s",
                    i + 1,
                    len(delta_chain),
                    delta_meta.window,
                )

                # Download and load delta files
                delta_data = await self._download_and_load_delta(delta_meta)
                if delta_data is None:
                    logger.error("Failed to download/load delta for window %s", delta_meta.window)
                    return None

                sparse_tensors, shapes, delta_info = delta_data

                # Apply sparse delta - dtype is inferred from current_state
                current_state = apply_sparse_delta(
                    current_state,
                    sparse_tensors,
                    shapes,
                    target_dtype=None,  # Infer from current_state
                )

                logger.debug(
                    "Applied delta window %s (%.2f%% sparse, %d non-zero params)",
                    delta_meta.window,
                    delta_info.get("sparsity_ratio", 0) * 100,
                    delta_info.get("nonzero_params", 0),
                )

            # Verify hash at final step
            final_meta = delta_chain[-1]
            if final_meta.weights_hash:
                actual_hash = compute_weights_hash(current_state)
                if actual_hash != final_meta.weights_hash:
                    logger.error(
                        "Reconstructed weights hash mismatch: expected %s..., got %s...",
                        final_meta.weights_hash[:16],
                        actual_hash[:16],
                    )
                    raise CheckpointDownloadError(
                        f"Hash verification failed for reconstructed checkpoint {final_meta.window}"
                    )
                logger.debug("✅ Weights hash verified for reconstructed checkpoint")

            # Write final reconstructed checkpoint
            return await self._write_reconstructed_checkpoint(
                current_state,
                anchor_path,
                output_dir,
                final_meta,
            )

        except CheckpointDownloadError:
            raise
        except Exception as exc:
            logger.exception("Failed to apply delta chain: %s", exc)
            return None

    async def _download_and_load_delta(
        self,
        delta_meta: CheckpointMetadata,
    ) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, Any]] | None:
        """Download delta files and load sparse tensors.

        Args:
            delta_meta: Metadata for the delta checkpoint

        Returns:
            Tuple of (sparse_tensors, shapes, delta_info) or None on failure
        """
        # Download delta files to temp directory
        delta_tmp = _unique_partial_dir(self.cache_root, f"delta-{delta_meta.window}")
        delta_tmp.mkdir(parents=True, exist_ok=True)

        try:
            await self._download_files(delta_meta, delta_tmp)

            # Verify delta file integrity
            if not await self._verify_integrity(delta_tmp, delta_meta.file_manifest):
                logger.error("Delta integrity check failed for window %s", delta_meta.window)
                return None

            # Load delta metadata first to determine format
            delta_meta_path = delta_tmp / "delta_metadata.json"
            if not delta_meta_path.exists():
                logger.error("Delta checkpoint missing delta_metadata.json: %s", delta_tmp)
                return None
            delta_info = json.loads(delta_meta_path.read_text())
            shapes = delta_info.get("shapes", {})
            delta_format = delta_info.get("format")

            # Load sparse delta based on format
            if delta_format in {"sparse_codec_v2", "sparse_codec_v3", "sparse_codec_v3.1"}:
                # Optimized formats (8-18x compression)
                delta_compressed_path = delta_tmp / "delta_sparse.bin.zst"
                if not delta_compressed_path.exists():
                    logger.error(
                        "Delta checkpoint missing delta_sparse.bin.zst: %s",
                        delta_tmp,
                    )
                    return None
                compressed_data = delta_compressed_path.read_bytes()
                if delta_format == "sparse_codec_v2":
                    sparse_tensors, shapes = decode_sparse_delta_v2(compressed_data)
                elif delta_format == "sparse_codec_v3":
                    sparse_tensors, shapes = decode_sparse_delta_v3(compressed_data)
                else:  # sparse_codec_v3.1
                    sparse_tensors, shapes = decode_sparse_delta_v3_1(compressed_data)
                logger.debug(
                    "🗜️ Decoded optimized sparse delta (%s): %.2f MB compressed",
                    delta_format,
                    len(compressed_data) / (1024 * 1024),
                )
            else:
                logger.error(
                    "Unsupported delta format '%s' for window %s",
                    delta_format,
                    delta_meta.window,
                )
                return None

            return (sparse_tensors, shapes, delta_info)

        finally:
            # Cleanup delta temp directory
            shutil.rmtree(delta_tmp, ignore_errors=True)

    async def _write_reconstructed_checkpoint(
        self,
        state_dict: dict[str, Any],
        anchor_path: Path,
        output_dir: Path,
        final_meta: CheckpointMetadata,
    ) -> Path:
        """Write reconstructed checkpoint to output directory.

        Args:
            state_dict: Reconstructed model state dict
            anchor_path: Path to anchor checkpoint (for non-weight files)
            output_dir: Directory to write checkpoint
            final_meta: Metadata for the final checkpoint

        Returns:
            Path to output directory
        """
        from safetensors.torch import save_file

        # Save reconstructed model
        output_dir.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, output_dir / "model.safetensors")

        # Copy non-weight files from anchor (tokenizer, config, etc.)
        for f in anchor_path.iterdir():
            if f.is_file() and f.name not in (
                "model.safetensors",
                "model.safetensors.index.json",
                "model.safetensors.index.json.gz",
                "metadata.json",
                "FULL",
                "DELTA",
                "manifest.sig",
            ):
                # Skip sharded weight files from the anchor checkpoint.
                if f.name.startswith("model-") and f.name.endswith(".safetensors"):
                    continue
                # Skip READY markers
                if f.name.startswith("READY"):
                    continue
                shutil.copy(f, output_dir / f.name)

        # Write updated metadata (mark as reconstructed FULL)
        output_metadata = {
            "window": final_meta.window,
            "file_manifest": {},  # Will be regenerated
            "training_config": final_meta.training_config,
            "git_commit": final_meta.git_commit,
            "created_at": final_meta.created_at,
            "model_name": final_meta.model_name,
            "checkpoint_type": "FULL",  # Reconstructed as FULL
            "reconstructed_from_delta": True,
            "original_anchor_window": final_meta.anchor_window,
        }

        # Build file manifest for reconstructed checkpoint
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(output_dir))
                output_metadata["file_manifest"][rel_path] = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()

        (output_dir / "metadata.json").write_text(
            json.dumps(output_metadata, ensure_ascii=False, indent=2)
        )

        logger.info(
            "✅ Reconstructed checkpoint-%s from anchor-%s",
            final_meta.window,
            final_meta.anchor_window,
        )
        return output_dir

    def _compute_keep_windows(self, current_window: int) -> set[int]:
        """Calculate which checkpoint windows should be retained.

        Delegates to the shared retention utility for consistent behavior
        between publisher (remote cleanup) and consumer (local cache cleanup).

        For chained deltas, we must keep the entire chain from the current
        anchor (FULL) to now, plus the previous anchor for miners catching up.

        Retention policy:
        - Keep all windows from current anchor to now (active chain)
        - Keep previous anchor for recovery
        - Keep milestone checkpoints

        Args:
            current_window: Current window number

        Returns:
            Set of window numbers to retain
        """
        from grail.shared.retention_utils import compute_retention_windows

        return compute_retention_windows(current_window)

    async def get_latest_ready_checkpoint(self, before_window: int) -> int | None:
        """Find the latest checkpoint that became READY before the given window.

        Parses READY-{ready_window} markers to determine when each checkpoint
        became available, ensuring miners/validators use the same model version.

        Args:
            before_window: Upper bound (exclusive) for ready_window

        Returns:
            Checkpoint window number, or None if none found
        """
        try:
            # List all checkpoint directories
            bucket_id = comms.get_bucket_id(self.credentials, use_write=False)
            cred_type = type(self.credentials).__name__
            logger.debug(
                "get_latest_ready_checkpoint: listing bucket=%s cred_type=%s before_window=%s",
                bucket_id,
                cred_type,
                before_window,
            )
            keys = await comms.list_bucket_files(
                CHECKPOINT_PREFIX,
                credentials=self.credentials,
                use_write=False,
            )
            logger.debug(
                "get_latest_ready_checkpoint: found %d keys, READY markers: %s",
                len(keys),
                [k for k in keys if "/READY-" in k],
            )

            # Parse all READY-{ready_window} markers
            candidates: list[tuple[int, int]] = []  # (ready_window, checkpoint_window)
            for key in keys:
                if "/READY-" in key:
                    try:
                        # Parse: "grail/checkpoints/checkpoint-1000/READY-1100"
                        parts = key.split("/")
                        if len(parts) >= 4:
                            checkpoint_segment = parts[2]  # "checkpoint-1000"
                            ready_filename = parts[3]  # "READY-1100"

                            if checkpoint_segment.startswith(
                                "checkpoint-"
                            ) and ready_filename.startswith("READY-"):
                                checkpoint_window = int(checkpoint_segment.split("-")[1])
                                ready_window = int(ready_filename.split("-")[1])

                                # Only consider checkpoints that became ready before our window
                                if ready_window < before_window:
                                    candidates.append((ready_window, checkpoint_window))
                    except (IndexError, ValueError):
                        continue

            if not candidates:
                logger.warning(
                    "No READY checkpoints found before window %s",
                    before_window,
                )
                return None

            # Sort by ready_window descending (most recently ready first)
            candidates.sort(reverse=True)
            ready_window, checkpoint_window = candidates[0]

            logger.info(
                "Found latest checkpoint: checkpoint-%s (ready at window %s, requested < %s)",
                checkpoint_window,
                ready_window,
                before_window,
            )
            return checkpoint_window

        except Exception as exc:
            logger.error("Failed to discover latest ready checkpoint: %s", exc)
            return None

    async def get_latest_full_checkpoint(self, before_window: int) -> int | None:
        """Find the latest FULL checkpoint that became READY before the given window.

        This method specifically looks for FULL checkpoints (not DELTA), which is
        critical for cold start recovery. FULL checkpoints have complete model weights
        and can be used as a base for applying deltas.

        Optimized to minimize expensive S3 API calls by:
        1. Parsing all READY markers first (cheap, local operation)
        2. Sorting by checkpoint_window descending
        3. Checking from newest to oldest, stopping at first FULL
        4. Making only the minimum necessary metadata fetches

        Args:
            before_window: Upper bound (exclusive) for ready_window

        Returns:
            FULL checkpoint window number, or None if none found
        """
        try:
            # List all checkpoint directories (one S3 API call)
            keys = await comms.list_bucket_files(
                CHECKPOINT_PREFIX,
                credentials=self.credentials,
                use_write=False,
            )

            # Parse all READY-{ready_window} markers (cheap, no API calls)
            ready_checkpoints: list[tuple[int, int]] = []  # (checkpoint_window, ready_window)
            for key in keys:
                if "/READY-" in key:
                    try:
                        # Parse: "grail/checkpoints/checkpoint-1000/READY-1100"
                        parts = key.split("/")
                        if len(parts) >= 4:
                            checkpoint_segment = parts[2]  # "checkpoint-1000"
                            ready_filename = parts[3]  # "READY-1100"

                            if checkpoint_segment.startswith(
                                "checkpoint-"
                            ) and ready_filename.startswith("READY-"):
                                checkpoint_window = int(checkpoint_segment.split("-")[1])
                                ready_window = int(ready_filename.split("-")[1])

                                # Only consider checkpoints that became ready before our window
                                if ready_window < before_window:
                                    ready_checkpoints.append((checkpoint_window, ready_window))
                    except (IndexError, ValueError):
                        continue

            if not ready_checkpoints:
                logger.warning(
                    "No READY checkpoints found before window %s",
                    before_window,
                )
                return None

            # Sort by checkpoint_window descending (newest first)
            # This allows us to check from newest to oldest and stop early
            ready_checkpoints.sort(reverse=True)

            # Check from newest to oldest, stop at first FULL checkpoint
            # This minimizes expensive S3 metadata fetches
            for checkpoint_window, ready_window in ready_checkpoints:
                # Check if this checkpoint has FULL metadata (S3 API call)
                full_meta = await self._fetch_full_metadata(checkpoint_window)
                if full_meta is not None:
                    # Found the latest FULL checkpoint, return immediately
                    logger.info(
                        "Found latest FULL checkpoint: checkpoint-%s (ready at window %s, requested < %s)",
                        checkpoint_window,
                        ready_window,
                        before_window,
                    )
                    return checkpoint_window

            # No FULL checkpoints found among ready checkpoints
            logger.warning(
                "No FULL checkpoints found before window %s (checked %d ready checkpoints)",
                before_window,
                len(ready_checkpoints),
            )
            return None

        except Exception as exc:
            logger.error("Failed to discover latest FULL checkpoint: %s", exc)
            return None


# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #


def default_checkpoint_cache_root() -> Path:
    """Return default cache directory for checkpoints.

    Reads GRAIL_CACHE_DIR environment variable:
    - If set and non-empty: use it (with ~ expansion for home directory paths)
    - If empty or unset: default to ~/.cache/grail

    Recommended for RAM-disk (faster I/O):
        export GRAIL_CACHE_DIR=/dev/shm/grail
    """
    raw_cache_dir = os.getenv("GRAIL_CACHE_DIR", "").strip()

    if raw_cache_dir:
        # Expand ~ to home directory (Python-level, not shell-level)
        base_dir = Path(raw_cache_dir).expanduser()
    else:
        # Default to ~/.cache/grail
        base_dir = Path.home() / ".cache" / "grail"

    return base_dir / "checkpoints"


def iter_checkpoints(cache_root: Path) -> Iterable[Path]:
    """Yield checkpoint directories under *cache_root* sorted by window."""

    checkpoints = []
    for entry in cache_root.glob("checkpoint-*"):
        try:
            window = int(entry.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((window, entry))
    for _, path in sorted(checkpoints):
        yield path


def _unique_partial_dir(cache_root: Path, stem: str) -> Path:
    """Return a process-unique `.partial` directory path under *cache_root*.

    This prevents cross-process races when miner and validator share a cache root.
    """
    suffix = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return cache_root / f"{stem}.{suffix}.partial"
