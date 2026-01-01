"""Checkpoint path utilities.

Single source of truth for building checkpoint paths in R2 storage.
All checkpoint path construction should use these utilities to ensure
consistency between publisher and consumer.

Directory Structure:
    grail/checkpoints/
    └── checkpoint-{window}/
        ├── DELTA/              # Sparse delta checkpoint (for caught-up consumers)
        │   ├── metadata.json
        │   ├── manifest.sig
        │   ├── delta_metadata.json
        │   └── delta_sparse.safetensors.zst
        └── FULL/               # Full checkpoint (for new joiners/bootstrap)
            ├── metadata.json
            ├── manifest.sig
            ├── model.safetensors
            ├── config.json
            └── tokenizer.json
"""

from __future__ import annotations

from grail.shared.constants import (
    CHECKPOINT_PREFIX,
    CHECKPOINT_SUBDIR_DELTA,
    CHECKPOINT_SUBDIR_FULL,
)


def checkpoint_window_prefix(window: int) -> str:
    """Get the base prefix for a checkpoint window.

    Args:
        window: Window number

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}"
    """
    return f"{CHECKPOINT_PREFIX}checkpoint-{window}"


def checkpoint_delta_prefix(window: int) -> str:
    """Get the prefix for a DELTA checkpoint.

    Args:
        window: Window number

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}/DELTA"
    """
    return f"{checkpoint_window_prefix(window)}/{CHECKPOINT_SUBDIR_DELTA}"


def checkpoint_full_prefix(window: int) -> str:
    """Get the prefix for a FULL checkpoint.

    Args:
        window: Window number

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}/FULL"
    """
    return f"{checkpoint_window_prefix(window)}/{CHECKPOINT_SUBDIR_FULL}"


def checkpoint_delta_metadata_key(window: int) -> str:
    """Get the key for DELTA checkpoint metadata.

    Args:
        window: Window number

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}/DELTA/metadata.json"
    """
    return f"{checkpoint_delta_prefix(window)}/metadata.json"


def checkpoint_full_metadata_key(window: int) -> str:
    """Get the key for FULL checkpoint metadata.

    Args:
        window: Window number

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}/FULL/metadata.json"
    """
    return f"{checkpoint_full_prefix(window)}/metadata.json"


def checkpoint_ready_marker_key(window: int, ready_window: int) -> str:
    """Get the key for READY marker.

    The READY marker is placed at the window level (not in DELTA/FULL subdirs)
    to indicate the checkpoint is ready for consumption.

    Args:
        window: Checkpoint window number
        ready_window: Window when the checkpoint became ready

    Returns:
        Path like "grail/checkpoints/checkpoint-{window}/READY-{ready_window}"
    """
    return f"{checkpoint_window_prefix(window)}/READY-{ready_window}"


def parse_window_from_prefix(prefix: str) -> int | None:
    """Extract window number from a checkpoint prefix.

    Args:
        prefix: Path like "grail/checkpoints/checkpoint-12345/..."

    Returns:
        Window number or None if not parseable
    """
    try:
        # Split by / and find checkpoint-{window} segment
        for segment in prefix.split("/"):
            if segment.startswith("checkpoint-"):
                return int(segment.split("-", 1)[1])
    except (ValueError, IndexError):
        pass
    return None
