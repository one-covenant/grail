#!/usr/bin/env python3
"""Delta checkpoint callback for storing sparse parameter updates.

Captures parameter deltas after each optimizer step using the existing
ParameterSnapshot infrastructure. Stores only non-zero changes in sparse
COO format for extreme storage efficiency.

Storage format per checkpoint:
    {
        "step": int,
        "timestamp": float,
        "layers": {
            "layer.name": {
                "indices": torch.LongTensor,    # COO sparse indices
                "values": torch.BFloat16Tensor,  # Non-zero delta values
                "shape": tuple,                  # Original tensor shape
                "nnz": int,                      # Non-zero count
            }
        }
    }

Typical storage:
    - Dense checkpoint: ~3GB (1.5B params)
    - Sparse delta (99% sparse): ~30MB
    - 100 steps: ~3GB vs 300GB (100x savings)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

# Import existing snapshot infrastructure
import sys
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from grail.trainer.analysis.primitives import ParameterSnapshot, ParameterDelta


class DeltaCheckpointCallback(TrainerCallback):
    """Save sparse parameter deltas after each optimizer step.

    Captures W_new - W_old using ParameterSnapshot infrastructure,
    converts to sparse COO format, saves only non-zero deltas.

    Timing (verified in transformers.Trainer:2740-2752):
        1. optimizer.step() is called (weights updated)
        2. on_optimizer_step() callback runs ← we capture here
        3. zero_grad() is called (gradients cleared)

    Args:
        output_dir: Directory to save delta checkpoints
        enabled: Enable/disable delta checkpointing (default: True)
        snapshot_dtype: Dtype for storing delta values ("bfloat16" or "float32")
        save_metadata: Save metadata.json with checkpoint list (default: True)
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        snapshot_dtype: str = "bfloat16",
        save_metadata: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.save_metadata = save_metadata
        self.old_snapshot: ParameterSnapshot | None = None
        self.step_count = 0

        # Configure storage dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        self._snapshot_dtype = dtype_map.get(snapshot_dtype, torch.bfloat16)

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DeltaCheckpoint] Initialized: output_dir={output_dir}, dtype={snapshot_dtype}")
        else:
            print("[DeltaCheckpoint] Disabled (enabled=False)")

    def on_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called after optimizer.step() but before zero_grad().

        This is the perfect time to capture post-update weights.
        """
        if not self.enabled or model is None:
            return

        # Only process on main process in distributed training
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not is_world_process_zero:
            return

        # Capture current snapshot (float32 internally for precision)
        current_snapshot = ParameterSnapshot(
            model,
            device="cpu",
            dtype=torch.float32,  # Compute deltas in float32 for precision
        )

        # First step: just capture snapshot, no delta yet
        if self.old_snapshot is None:
            self.old_snapshot = current_snapshot
            print(f"[DeltaCheckpoint] Step {self.step_count}: Initial snapshot captured")
            return

        # Compute delta (W_new - W_old)
        try:
            delta = self.old_snapshot.compute_delta(current_snapshot)
        except ValueError as e:
            print(f"[DeltaCheckpoint] ⚠️  Failed to compute delta: {e}")
            self.old_snapshot = current_snapshot
            self.step_count += 1
            return

        # Convert to sparse and save
        self._save_sparse_delta(delta, step=self.step_count)

        # Update for next iteration
        self.old_snapshot = current_snapshot
        self.step_count += 1

    def _to_sparse_coo(self, delta_tensor: torch.Tensor) -> dict[str, Any]:
        """Convert dense delta to sparse COO format with threshold=0.0.

        CRITICAL: threshold=0.0 means we only skip EXACT zeros.
        Uses `!= 0.0` for exact floating-point comparison.

        Args:
            delta_tensor: Dense parameter delta tensor

        Returns:
            Dictionary with sparse COO representation
        """
        # Find non-zero elements (EXACT comparison with 0.0)
        mask = (delta_tensor != 0.0)

        # Extract non-zero indices and values
        indices = mask.nonzero(as_tuple=False).t()  # Shape: [ndim, nnz]
        values = delta_tensor[mask]

        # VERIFICATION: Ensure we captured exactly the non-zero elements
        expected_nnz = (delta_tensor != 0.0).sum().item()
        actual_nnz = values.numel()
        if actual_nnz != expected_nnz:
            raise RuntimeError(
                f"Sparsity computation bug detected: "
                f"expected {expected_nnz} non-zeros, got {actual_nnz}"
            )

        # Convert values to target dtype for storage
        values_stored = values.to(dtype=self._snapshot_dtype)

        return {
            "indices": indices.to(torch.int32),  # Compact indices (int32 vs int64)
            "values": values_stored,
            "shape": tuple(delta_tensor.shape),
            "nnz": values.numel(),
        }

    def _save_sparse_delta(self, delta: ParameterDelta, step: int) -> None:
        """Save sparse delta checkpoint to disk.

        Args:
            delta: ParameterDelta computed from old -> new snapshot
            step: Optimizer step number
        """
        sparse_layers = {}
        total_params = 0
        total_nonzero = 0

        for name, delta_tensor in delta.deltas.items():
            total_params += delta_tensor.numel()

            # Convert to sparse COO
            sparse_data = self._to_sparse_coo(delta_tensor)

            # Only save layers that have non-zero changes
            if sparse_data["nnz"] > 0:
                sparse_layers[name] = sparse_data
                total_nonzero += sparse_data["nnz"]

        # Create checkpoint
        checkpoint = {
            "step": step,
            "timestamp": time.time(),
            "layers": sparse_layers,
            "metadata": {
                "total_params": total_params,
                "total_nonzero": total_nonzero,
                "sparsity": 1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0,
                "num_changed_layers": len(sparse_layers),
                "dtype": str(self._snapshot_dtype),
            },
        }

        # Save to disk
        path = self.output_dir / f"delta_{step:06d}.pt"
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)

        # Log statistics
        sparsity = checkpoint["metadata"]["sparsity"]
        print(
            f"[DeltaCheckpoint] Step {step}: "
            f"{sparsity:.2%} sparse, "
            f"{total_nonzero:,}/{total_params:,} non-zero, "
            f"{len(sparse_layers)}/{len(delta.deltas)} layers changed"
        )

        # Update metadata file
        if self.save_metadata:
            self._update_metadata(checkpoint["metadata"], step, path)

    def _update_metadata(self, checkpoint_meta: dict, step: int, path: Path) -> None:
        """Maintain metadata.json tracking all checkpoints."""
        import json

        metadata_path = self.output_dir / "metadata.json"

        # Load existing metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "checkpoints": [],
                "total_params": checkpoint_meta["total_params"],
                "dtype": checkpoint_meta["dtype"],
            }

        # Add new checkpoint info
        metadata["checkpoints"].append({
            "step": step,
            "path": str(path.name),
            "sparsity": checkpoint_meta["sparsity"],
            "nonzero": checkpoint_meta["total_nonzero"],
            "changed_layers": checkpoint_meta["num_changed_layers"],
        })

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_sparse_delta(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a sparse delta checkpoint and reconstruct dense tensors.

    Args:
        checkpoint_path: Path to delta checkpoint file

    Returns:
        Dictionary mapping layer name to dense delta tensor

    Example:
        >>> delta = load_sparse_delta("checkpoints/delta_000042.pt")
        >>> print(delta.keys())
        >>> print(delta["model.layers.0.self_attn.q_proj.weight"].shape)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    dense_deltas = {}
    for name, sparse_data in checkpoint["layers"].items():
        # Reconstruct dense tensor from sparse COO
        indices = sparse_data["indices"]
        values = sparse_data["values"]
        shape = sparse_data["shape"]

        # Create dense tensor
        dense = torch.zeros(shape, dtype=values.dtype)

        # Fill in non-zero values
        if indices.numel() > 0:
            # Convert indices to tuple for advanced indexing
            idx_tuple = tuple(indices[i] for i in range(indices.size(0)))
            dense[idx_tuple] = values

        dense_deltas[name] = dense

    return dense_deltas


def reconstruct_weights_at_step(
    base_weights: dict[str, torch.Tensor],
    delta_dir: str | Path,
    target_step: int,
) -> dict[str, torch.Tensor]:
    """Reconstruct model weights at a specific step from base + deltas.

    Args:
        base_weights: Initial model weights (step 0)
        delta_dir: Directory containing delta checkpoints
        target_step: Target step to reconstruct

    Returns:
        Reconstructed weights: W_target = W_0 + Σ(delta_0...delta_target-1)

    Example:
        >>> base = {name: param.clone() for name, param in model.named_parameters()}
        >>> weights_100 = reconstruct_weights_at_step(base, "checkpoints/deltas", 100)
        >>> # Load into model:
        >>> model.load_state_dict(weights_100, strict=False)
    """
    delta_dir = Path(delta_dir)

    # Start with base weights
    weights = {name: tensor.clone() for name, tensor in base_weights.items()}

    # Apply deltas sequentially
    for step in range(target_step):
        delta_path = delta_dir / f"delta_{step:06d}.pt"
        if not delta_path.exists():
            raise FileNotFoundError(f"Missing delta checkpoint: {delta_path}")

        deltas = load_sparse_delta(delta_path)

        # Add deltas to current weights
        for name, delta in deltas.items():
            if name in weights:
                weights[name] = weights[name] + delta.to(weights[name].dtype)

    return weights
