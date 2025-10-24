"""Checkpoint management utilities.

Central entry point for discovering, downloading, caching, and cleaning up
model checkpoints stored in R2. Checkpoints are published by the trainer under
``model_checkpoints/checkpoint-{window}/`` with a manifest describing all
artifacts and their SHA256 hashes. Miners and validators rely on this module to
retrieve the appropriate checkpoint for a given window while keeping local disk
usage bounded.

Design goals:
 - Integrity validation for every download using manifest hashes.
 - Atomic download process to avoid partial/corrupt states.
 - Local cache with retention policy (last N + milestone windows).
 - Remote cleanup helpers for the trainer to enforce retention upstream.

Checkpoint Retrieval Strategy:
 - With the trainer always publishing checkpoints for every window, this module
   now simply downloads and validates the requested checkpoint.
 - If a checkpoint is not ready (READY marker not found), returns None and the
   caller should wait/retry rather than falling back to an older checkpoint.
 - This ensures consistent model versions across miners and validators.

The module intentionally stays independent from model-loading details. It only
manages files on disk and in R2; callers handle loading into Torch/Transformers.
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

from ..shared.constants import (
    CHECKPOINT_MILESTONE_INTERVAL,
    CHECKPOINT_RETENTION_LIMIT,
    GRAIL_CHECKPOINT_MOD10,
    WINDOW_LENGTH,
)
from . import comms

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "grail/checkpoints/"


# --------------------------------------------------------------------------- #
#                             Metadata Schema                                 #
# --------------------------------------------------------------------------- #


@dataclass
class CheckpointMetadata:
    """Metadata describing a checkpoint directory."""

    window: int
    parent_window: int | None
    file_manifest: dict[str, str]
    training_config: dict[str, Any] = field(default_factory=dict)
    git_commit: str = "unknown"
    created_at: float = 0.0
    model_name: str = "no_name"

    def remote_prefix(self) -> str:
        return f"{CHECKPOINT_PREFIX}checkpoint-{self.window}"


class CheckpointDownloadError(RuntimeError):
    """Raised when checkpoint download or validation fails."""


# --------------------------------------------------------------------------- #
#                          Checkpoint Manager                                  #
# --------------------------------------------------------------------------- #


class CheckpointManager:
    """Manage checkpoint discovery, downloads, and cache cleanup."""

    def __init__(
        self,
        *,
        cache_root: Path,
        credentials: Any | None,
        keep_limit: int = CHECKPOINT_RETENTION_LIMIT,
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

            # Check READY marker to ensure checkpoint is fully uploaded
            is_ready = await self._is_checkpoint_ready(window)
            if not is_ready:
                logger.warning(
                    "Checkpoint for window %s not ready (READY marker not found); will retry later",
                    window,
                )
                return None

            tmp_dir = self.cache_root / f"checkpoint-{window}.partial"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                logger.info("Starting checkpoint download for window %s", window)
                if metadata is not None and metadata.file_manifest:
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
                                "parent_window": metadata.parent_window,
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
                tmp_dir.rename(local_dir)
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
                logger.debug("Removing local checkpoint %s", candidate)
                shutil.rmtree(candidate, ignore_errors=True)

    async def cleanup_remote(self, current_window: int) -> None:
        """Delete remote checkpoints outside retention policy (trainer role)."""

        keep_windows = self._compute_keep_windows(current_window)
        remote_windows = await self.list_remote_windows()

        for window in remote_windows:
            if window not in keep_windows:
                prefix = f"{CHECKPOINT_PREFIX}checkpoint-{window}"
                logger.info("Deleting remote checkpoint prefix %s", prefix)
                await comms.delete_prefix(prefix, credentials=self.credentials, use_write=True)

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

    async def _is_checkpoint_ready(self, window: int) -> bool:
        """Check if a checkpoint has been marked as READY (fully uploaded).

        The trainer creates a READY file as the final step of publishing a checkpoint.
        This prevents miners from downloading partially uploaded checkpoints.

        Returns:
            True if READY marker exists, False otherwise.
        """
        ready_key = f"{CHECKPOINT_PREFIX}checkpoint-{window}/READY"
        return await comms.file_exists(ready_key, credentials=self.credentials, use_write=False)

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
            parent_window=payload.get("parent_window"),
            file_manifest=payload.get("file_manifest", {}),
            training_config=payload.get("training_config", {}),
            git_commit=payload.get("git_commit", "unknown"),
            created_at=float(payload.get("created_at", 0.0)),
            model_name=payload.get("model_name", "no_name"),
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

    def _compute_keep_windows(self, current_window: int) -> set[int]:
        keep: set[int] = set()
        if current_window < 0:
            return keep

        # Always keep windows 0-9
        keep.update(range(10))

        # Keep latest windows up to limit
        for idx in range(self.keep_limit):
            window = current_window - idx * WINDOW_LENGTH
            if window >= 0:
                keep.add(window)

        # Keep milestones (every CHECKPOINT_MILESTONE_INTERVAL windows)
        interval_blocks = CHECKPOINT_MILESTONE_INTERVAL * WINDOW_LENGTH
        if interval_blocks > 0:
            milestone = (current_window // interval_blocks) * interval_blocks
            while milestone >= 0:
                keep.add(milestone)
                milestone -= interval_blocks

        return keep

    async def _find_latest_checkpoint_window(self) -> int | None:
        """Find the highest window number available in R2."""
        keys = await comms.list_bucket_files(
            CHECKPOINT_PREFIX,
            credentials=self.credentials,
            use_write=False,
        )

        latest_window = None
        for key in keys:
            parts = key.split("/")
            # Expect keys like 'grail/checkpoints/checkpoint-<window>/...'
            if len(parts) < 3:
                continue
            segment = parts[2]
            if not segment.startswith("checkpoint-"):
                continue
            try:
                # Extract window number from "checkpoint-<window>"
                window = int(segment.split("-", 1)[1])
                if latest_window is None or window > latest_window:
                    latest_window = window
            except (IndexError, ValueError):
                continue
        return latest_window


# --------------------------------------------------------------------------- #
#                             Helper Functions                                #
# --------------------------------------------------------------------------- #


def default_checkpoint_cache_root() -> Path:
    """Return default cache directory for checkpoints."""

    base_dir = Path(os.getenv("GRAIL_CACHE_DIR", Path.home() / ".cache/grail"))
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
