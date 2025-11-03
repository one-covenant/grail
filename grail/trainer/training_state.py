"""Utilities to persist and restore full training state.

This module saves and loads optimizer/scheduler state and RNG states
so training can resume seamlessly after evaluation detours or restarts.

Design goals:
- Minimal coupling: pure functions with explicit arguments
- Deterministic reconstruction: load after re-creating optimizer/scheduler
- Safe defaults: tolerate missing files and log-friendly errors
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


SCHEMA_VERSION: int = 1
STATE_FILENAME: str = "training_state.pt"
META_FILENAME: str = "bundle.json"


@dataclass(frozen=True)
class TrainingStatePaths:
    """Resolved paths for training state bundle files.

    Attributes:
        root: Root directory of the bundle
        state_file: Torch-serialized state dict file
        meta_file: JSON metadata file containing schema and minimal info
    """

    root: Path
    state_file: Path
    meta_file: Path


def _resolve_paths(root_dir: str | os.PathLike[str]) -> TrainingStatePaths:
    root = Path(root_dir)
    return TrainingStatePaths(
        root=root,
        state_file=root / STATE_FILENAME,
        meta_file=root / META_FILENAME,
    )


def save_training_state(
    root_dir: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
) -> None:
    """Save optimizer, scheduler, and RNG states into ``root_dir``.

    The caller is responsible for ensuring the directory exists.
    """
    paths = _resolve_paths(root_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    # Collect RNG states (robust to missing CUDA)
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": rng_state,
    }

    # Write atomically: save to temp then replace
    tmp_state = paths.state_file.with_suffix(paths.state_file.suffix + ".tmp")
    torch.save(payload, tmp_state)
    tmp_state.replace(paths.state_file)

    # Minimal JSON metadata for diagnostics
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "files": [STATE_FILENAME],
    }
    tmp_meta = paths.meta_file.with_suffix(paths.meta_file.suffix + ".tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    tmp_meta.replace(paths.meta_file)


def load_training_state(
    root_dir: str | os.PathLike[str],
) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None]:
    """Load optimizer, scheduler, and RNG states from ``root_dir``.

    Returns:
        (optimizer_state, scheduler_state, rng_state) where any element
        may be None when missing.
    
    Note:
        Uses weights_only=False because training state includes numpy RNG state
        which requires unpickling numpy types. This is safe since we control
        the checkpoint creation and it's only used for internal training state.
    """
    paths = _resolve_paths(root_dir)
    if not paths.state_file.exists():
        return None, None, None

    # PyTorch 2.6+ defaults to weights_only=True for security, but training state
    # includes numpy RNG state that requires unpickling numpy._core.multiarray types.
    # Since we create and consume these checkpoints ourselves, weights_only=False is safe.
    data: Dict[str, Any] = torch.load(
        paths.state_file,
        map_location="cpu",
        weights_only=False,
    )
    return (
        data.get("optimizer"),
        data.get("scheduler"),
        data.get("rng"),
    )


def apply_training_state(
    *,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    optimizer_state: Dict[str, Any] | None,
    scheduler_state: Dict[str, Any] | None,
    rng_state: Dict[str, Any] | None,
) -> None:
    """Apply loaded states to live objects and RNGs.

    This function is safe to call with None values (no-ops).
    """
    # Restore optimizer/scheduler first (if provided)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            # Tolerate minor scheduler drift; caller should log if needed
            pass

    # Restore RNGs for determinism if available
    if rng_state is not None:
        try:
            random.setstate(rng_state.get("python"))
        except Exception:
            pass
        try:
            np.random.set_state(rng_state.get("numpy"))
        except Exception:
            pass
        try:
            torch.set_rng_state(rng_state.get("torch_cpu"))
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_state.get("torch_cuda_all", []))
            except Exception:
                pass


