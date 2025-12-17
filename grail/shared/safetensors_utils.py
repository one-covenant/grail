"""Utilities for working with safetensors checkpoints.

GRAIL checkpoints are saved via HuggingFace `save_pretrained(..., safe_serialization=True)`.
For large models, this typically produces a sharded safetensors checkpoint:

- model.safetensors.index.json
- model-00001-of-0000N.safetensors
- ...

Some parts of the codebase (delta checkpoints, upload worker caching, etc.) need to
load a full state dict regardless of whether the checkpoint is sharded or not.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def load_model_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor] | None:
    """Load a model state dict from a checkpoint directory.

    Supports:
    - Unsharded: `model.safetensors`
    - Sharded: `model.safetensors.index.json` (+ shard files)

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        State dict (param_name -> tensor), or None if no model weights are found.

    Raises:
        FileNotFoundError: If an index references shard files that are missing.
        ValueError: If the index JSON is malformed.
    """
    checkpoint_dir = Path(checkpoint_dir)

    model_path = checkpoint_dir / "model.safetensors"
    if model_path.exists():
        return load_file(model_path)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    index_gz_path = checkpoint_dir / "model.safetensors.index.json.gz"

    index: dict[str, Any] | None = None
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    elif index_gz_path.exists():
        with gzip.open(index_gz_path, "rt", encoding="utf-8") as f:
            index = json.load(f)
    else:
        return None

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Invalid safetensors index: missing/invalid 'weight_map'")

    shard_names = sorted({str(v) for v in weight_map.values()})
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard referenced by index: {shard_path}")
        state.update(load_file(shard_path))

    return state
