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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zstandard as zstd

from grail.shared.safetensors_utils import load_model_state_dict

from ..shared.constants import (
    BASE_CHECKPOINT_RETENTION_LIMIT,
    CHECKPOINT_MILESTONE_INTERVAL,
    DELTA_BASE_INTERVAL,
    GRAIL_CHECKPOINT_MOD10,
    WINDOW_LENGTH,
)
from . import comms
from .delta_checkpoint import apply_sparse_delta, compute_weights_hash

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "grail/checkpoints/"


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

    # Delta checkpoint fields
    checkpoint_type: str = "FULL"  # "FULL" or "DELTA"
    base_window: int | None = None  # For DELTA: the FULL checkpoint this is relative to
    weights_hash: str | None = None  # SHA256 of final weights for verification

    def remote_prefix(self) -> str:
        return f"{CHECKPOINT_PREFIX}checkpoint-{self.window}"

    def is_delta(self) -> bool:
        """Check if this is a delta checkpoint."""
        return self.checkpoint_type == "DELTA"


class CheckpointDownloadError(RuntimeError):
    """Raised when checkpoint download or validation fails."""


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
        self._metadata_cache: dict[int, CheckpointMetadata] = {}
        self._download_locks: dict[int, asyncio.Lock] = {}
        self._fallback_attempted: set[int] = set()

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
            metadata = await self._fetch_metadata(window)
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

            tmp_dir = self.cache_root / f"checkpoint-{window}.partial"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info("Starting checkpoint download for window %s", window)

                # Handle DELTA checkpoint: download base + delta, then reconstruct
                if metadata is not None and metadata.is_delta():
                    logger.info(
                        "Checkpoint %s is DELTA (base=%s), will reconstruct",
                        window,
                        metadata.base_window,
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
            # Keys look like: grail/checkpoints/checkpoint-<window>/...
            # Extract window from the checkpoint-<window> segment
            try:
                # Split into parts: ['grail', 'checkpoints', 'checkpoint-<window>', ...]
                parts = key.split("/")
                if len(parts) >= 3:
                    checkpoint_segment = parts[2]  # 'checkpoint-<window>'
                    if checkpoint_segment.startswith("checkpoint-"):
                        window = int(checkpoint_segment.split("-", 1)[1])
                        windows.add(window)
            except (IndexError, ValueError):
                continue
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
            prefix = f"{CHECKPOINT_PREFIX}checkpoint-{checkpoint_window}/"
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

    # --------------------------- Internal helpers --------------------------- #

    async def _fetch_metadata(self, window: int) -> CheckpointMetadata | None:
        if window in self._metadata_cache:
            return self._metadata_cache[window]

        remote_key = f"{CHECKPOINT_PREFIX}checkpoint-{window}/metadata.json"
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
            checkpoint_type=payload.get("checkpoint_type", "FULL"),
            base_window=payload.get("base_window"),
            weights_hash=payload.get("weights_hash"),
        )
        self._metadata_cache[window] = metadata
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
        """
        prefix_dir = f"{CHECKPOINT_PREFIX}checkpoint-{window}/"
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
        """Handle DELTA checkpoint: download base + delta, reconstruct full checkpoint.

        Args:
            metadata: Metadata for the delta checkpoint
            tmp_dir: Temporary directory for the reconstructed checkpoint

        Returns:
            Path to reconstructed checkpoint, or None if reconstruction failed
        """
        logger.info(
            "[_handle_delta_checkpoint] Processing DELTA checkpoint: "
            "window=%s, base_window=%s, weights_hash=%s...",
            metadata.window,
            metadata.base_window,
            metadata.weights_hash[:16] if metadata.weights_hash else "None",
        )

        if metadata.base_window is None:
            logger.error(
                "[_handle_delta_checkpoint] DELTA checkpoint has no base_window specified: window=%s",
                metadata.window,
            )
            return None

        # 1. Ensure base checkpoint is available (recursively handles chained deltas)
        logger.info(
            "[_handle_delta_checkpoint] Fetching base checkpoint %s for delta reconstruction",
            metadata.base_window,
        )
        base_path = await self.get_checkpoint(metadata.base_window)
        if base_path is None:
            logger.error(
                "[_handle_delta_checkpoint] Cannot reconstruct delta: "
                "base checkpoint %s unavailable | target_window=%s",
                metadata.base_window,
                metadata.window,
            )
            return None

        # 2. Download delta files to a separate temp directory
        delta_tmp = self.cache_root / f"delta-{metadata.window}.partial"
        if delta_tmp.exists():
            shutil.rmtree(delta_tmp, ignore_errors=True)
        delta_tmp.mkdir(parents=True, exist_ok=True)

        try:
            # Download all delta files
            await self._download_files(metadata, delta_tmp)

            # Verify delta file integrity
            if not await self._verify_integrity(delta_tmp, metadata.file_manifest):
                raise CheckpointDownloadError(
                    f"Delta integrity check failed for window {metadata.window}"
                )

            # 3. Reconstruct full checkpoint
            return await self._reconstruct_from_delta(
                base_path,
                delta_tmp,
                tmp_dir,
                metadata,
            )
        finally:
            # Cleanup delta temp directory
            shutil.rmtree(delta_tmp, ignore_errors=True)

    async def _reconstruct_from_delta(
        self,
        base_path: Path,
        delta_path: Path,
        output_dir: Path,
        metadata: CheckpointMetadata,
    ) -> Path | None:
        """Reconstruct full checkpoint from base + sparse delta.

        Args:
            base_path: Path to base checkpoint directory
            delta_path: Path to delta files directory
            output_dir: Directory to write reconstructed checkpoint
            metadata: Metadata for the delta checkpoint

        Returns:
            Path to reconstructed checkpoint directory, or None on failure
        """
        from safetensors.torch import load_file, save_file

        try:
            # Load base weights
            try:
                base_state = load_model_state_dict(base_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load base checkpoint weights from %s: %s",
                    base_path,
                    exc,
                    exc_info=True,
                )
                return None

            if base_state is None:
                logger.error("Base checkpoint missing model weights: %s", base_path)
                return None

            # Load sparse delta (handle both compressed and uncompressed formats)
            delta_sparse_path = delta_path / "delta_sparse.safetensors"
            delta_compressed_path = delta_path / "delta_sparse.safetensors.zst"

            if delta_compressed_path.exists():
                # Decompress zstd-compressed delta
                logger.debug("Decompressing delta_sparse.safetensors.zst")
                decompressor = zstd.ZstdDecompressor()
                compressed_data = delta_compressed_path.read_bytes()
                decompressed_data = decompressor.decompress(compressed_data)
                delta_sparse_path.write_bytes(decompressed_data)
                logger.debug(
                    "🗜️ Decompressed delta: %.2f MB → %.2f MB",
                    len(compressed_data) / (1024 * 1024),
                    len(decompressed_data) / (1024 * 1024),
                )

            if not delta_sparse_path.exists():
                logger.error("Delta checkpoint missing delta_sparse.safetensors: %s", delta_path)
                return None
            sparse_tensors = load_file(delta_sparse_path)

            # Load delta metadata for shapes
            delta_meta_path = delta_path / "delta_metadata.json"
            if not delta_meta_path.exists():
                logger.error("Delta checkpoint missing delta_metadata.json: %s", delta_path)
                return None
            delta_meta = json.loads(delta_meta_path.read_text())
            shapes = delta_meta.get("shapes", {})

            logger.info(
                "Reconstructing checkpoint from base-%s (%.2f%% sparse, %d changed params)",
                metadata.base_window,
                delta_meta.get("sparsity_ratio", 0) * 100,
                delta_meta.get("nonzero_params", 0),
            )

            # Apply sparse update (direct value replacement, no arithmetic)
            # Use base_state's dtype to preserve consistency
            sample_dtype = next(iter(base_state.values())).dtype
            reconstructed = apply_sparse_delta(
                base_state,
                sparse_tensors,
                shapes,
                target_dtype=sample_dtype,
            )

            # Verify hash if available
            if metadata.weights_hash:
                actual_hash = compute_weights_hash(reconstructed)
                if actual_hash != metadata.weights_hash:
                    logger.error(
                        "[_reconstruct_from_delta] HASH VERIFICATION FAILED: "
                        "window=%s | expected=%s | actual=%s",
                        metadata.window,
                        metadata.weights_hash,
                        actual_hash,
                    )
                    raise CheckpointDownloadError(
                        f"Hash verification failed for checkpoint {metadata.window}"
                    )
                logger.info(
                    "✅ Hash verified for checkpoint-%s",
                    metadata.window,
                )

            # Save reconstructed model
            output_dir.mkdir(parents=True, exist_ok=True)
            save_file(reconstructed, output_dir / "model.safetensors")

            # Copy non-weight files from base (tokenizer, config, etc.)
            for f in base_path.iterdir():
                if f.is_file() and f.name not in (
                    "model.safetensors",
                    "model.safetensors.index.json",
                    "model.safetensors.index.json.gz",
                    "metadata.json",
                    "FULL",
                    "DELTA",
                    "manifest.sig",
                ):
                    # Skip sharded weight files from the base checkpoint.
                    if f.name.startswith("model-") and f.name.endswith(".safetensors"):
                        continue
                    # Skip READY markers
                    if f.name.startswith("READY"):
                        continue
                    shutil.copy(f, output_dir / f.name)

            # Write updated metadata (mark as reconstructed FULL)
            output_metadata = {
                "window": metadata.window,
                "file_manifest": {},  # Will be regenerated
                "training_config": metadata.training_config,
                "git_commit": metadata.git_commit,
                "created_at": metadata.created_at,
                "model_name": metadata.model_name,
                "checkpoint_type": "FULL",  # Reconstructed as FULL
                "reconstructed_from_delta": True,
                "original_base_window": metadata.base_window,
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
                "✅ Reconstructed checkpoint-%s from base-%s",
                metadata.window,
                metadata.base_window,
            )
            return output_dir

        except Exception as exc:
            logger.exception("Failed to reconstruct checkpoint from delta: %s", exc)
            return None

    def _compute_keep_windows(self, current_window: int) -> set[int]:
        keep: set[int] = set()
        if current_window < 0:
            return keep

        # Always keep windows 0-9
        keep.update(range(10))

        # Keep latest windows up to limit
        delta_base_interval_windows = max(1, int(DELTA_BASE_INTERVAL))
        base_stride_blocks = delta_base_interval_windows * int(WINDOW_LENGTH)

        for idx in range(self.keep_limit):
            window = current_window - idx * WINDOW_LENGTH
            if window >= 0:
                keep.add(window)
                # Also keep the base window this delta depends on
                # Base windows are at DELTA_BASE_INTERVAL (in windows) boundaries.
                base_window = (window // base_stride_blocks) * base_stride_blocks
                if base_window >= 0:
                    keep.add(base_window)

        # Keep milestones (every CHECKPOINT_MILESTONE_INTERVAL windows)
        interval_blocks = CHECKPOINT_MILESTONE_INTERVAL * WINDOW_LENGTH
        if interval_blocks > 0:
            milestone = (current_window // interval_blocks) * interval_blocks
            while milestone >= 0:
                keep.add(milestone)
                milestone -= interval_blocks

        return keep

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
            keys = await comms.list_bucket_files(
                CHECKPOINT_PREFIX,
                credentials=self.credentials,
                use_write=False,
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


# --------------------------------------------------------------------------- #
#                        DEPRECATED: Fallback Logic                           #
# --------------------------------------------------------------------------- #
# The following function was used when the trainer might skip publishing
# checkpoints for some windows. Now that the trainer ALWAYS publishes
# checkpoints (even if training is skipped), this fallback logic is no longer
# needed. Preserved here for reference only.


async def _find_latest_ready_checkpoint_window_DEPRECATED(
    checkpoint_manager: CheckpointManager,
) -> int | None:
    """Find the highest window number with READY marker (fully uploaded).

    DEPRECATED: No longer needed since trainer always publishes checkpoints.

    Used when requested checkpoint is not ready yet. Falls back to latest
    available ready checkpoint to keep mining/validation operational.

    Returns:
        Window number of latest ready checkpoint, or None if none are ready.
    """
    windows = await checkpoint_manager.list_remote_windows()
    if not windows:
        return None

    # Check windows in descending order to find latest ready one
    for window in sorted(windows, reverse=True):
        if await checkpoint_manager._is_checkpoint_ready(window):
            logger.info(f"Found latest ready checkpoint at window {window}")
            return window

    logger.warning("No ready checkpoints found")
    return None
