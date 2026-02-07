#!/usr/bin/env python3
"""Delta checkpoint callback for storing sparse parameter updates.

Stores changed parameter VALUES (not deltas) after each optimizer step.
Compares parameters in BF16 to find exact changes, stores new BF16 values
at changed positions in sparse COO format.

Key design:
    - Compare in BF16: Matches what vLLM sees after weight sync
    - Store values, not deltas: W_new at changed positions
    - Single snapshot: Only keep one BF16 snapshot (~2 bytes/param)
    - Exact comparison: w_t != w_{t-1} finds bitwise changes in BF16

Storage format per checkpoint:
    {
        "step": int,
        "timestamp": float,
        "format": "values",  # Indicates we store values, not deltas
        "layers": {
            "layer.name": {
                "indices": torch.Int32Tensor,    # COO sparse indices
                "values": torch.BFloat16Tensor,  # NEW values at changed positions
                "shape": tuple,                  # Original tensor shape
                "nnz": int,                      # Non-zero count
            }
        }
    }

Typical storage:
    - Dense checkpoint: ~3GB (1.5B params)
    - Sparse values (99% unchanged): ~30MB
    - 100 steps: ~3GB vs 300GB (100x savings)

Reconstruction:
    To get weights at step T:
    1. Start with base weights W_0
    2. For each checkpoint, directly replace values at stored indices
    (No accumulation needed - each checkpoint has final values)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class DeltaCheckpointCallback(TrainerCallback):
    """Save sparse parameter changes after each optimizer step.

    Compares parameters in BF16 to find changed positions, stores the
    actual new BF16 values (not deltas) at those positions.

    Why BF16 comparison:
        - Training may use FP32 master weights
        - vLLM receives BF16 weights after sync
        - We track what actually changes from vLLM's perspective
        - Storing actual BF16 values enables direct reconstruction

    Timing (verified in transformers.Trainer:2740-2752):
        1. optimizer.step() is called (weights updated)
        2. on_optimizer_step() callback runs ← we capture here
        3. zero_grad() is called (gradients cleared)

    Args:
        output_dir: Directory to save delta checkpoints
        enabled: Enable/disable delta checkpointing (default: True)
        save_metadata: Save metadata.json with checkpoint list (default: True)
        profiler: Optional profiler for timing measurements
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        snapshot_dtype: str = "bfloat16",  # Kept for API compat, always uses BF16
        save_metadata: bool = True,
        profiler: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.save_metadata = save_metadata
        self.step_count = 0
        self._profiler = profiler

        # BF16 snapshot of previous weights (single copy, ~2 bytes/param)
        self._old_bf16: dict[str, torch.Tensor] = {}

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[DeltaCheckpoint] Initialized: output_dir={output_dir}, format=bf16_values")
        else:
            logger.info("[DeltaCheckpoint] Disabled (enabled=False)")

    def on_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called after optimizer.step() but before zero_grad().

        Compares current weights (as BF16) against previous BF16 snapshot,
        finds changed positions, stores new BF16 values at those positions.
        """
        if not self.enabled or model is None:
            return

        # Only process on main process in distributed training
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not is_world_process_zero:
            return

        profiler = self._profiler

        # First step: just capture BF16 snapshot, no comparison yet
        if not self._old_bf16:
            if profiler:
                with profiler.track("delta_checkpoint_init"):
                    self._capture_bf16_snapshot(model)
            else:
                self._capture_bf16_snapshot(model)
            logger.info(f"[DeltaCheckpoint] Step {self.step_count}: Initial BF16 snapshot captured")
            return

        # Compare and save changed values
        if profiler:
            with profiler.track("delta_checkpoint_compare_save"):
                self._compare_and_save(model, step=self.step_count)
        else:
            self._compare_and_save(model, step=self.step_count)

        self.step_count += 1

    def _capture_bf16_snapshot(self, model: Any) -> None:
        """Capture all model parameters as BF16 on CPU."""
        self._old_bf16.clear()
        for name, param in model.named_parameters():
            self._old_bf16[name] = param.data.detach().to(
                device="cpu", dtype=torch.bfloat16
            ).clone()

    def _compare_and_save(self, model: Any, step: int) -> None:
        """Compare current weights to old BF16 snapshot, save changed values."""
        sparse_layers = {}
        total_params = 0
        total_changed = 0

        for name, param in model.named_parameters():
            # Convert current param to BF16 on CPU
            new_bf16 = param.data.detach().to(device="cpu", dtype=torch.bfloat16)
            total_params += new_bf16.numel()

            # Get old BF16 value
            old_bf16 = self._old_bf16.get(name)
            if old_bf16 is None:
                # New parameter (shouldn't happen in normal training)
                self._old_bf16[name] = new_bf16.clone()
                continue

            # Find changed positions: exact BF16 comparison
            # This catches any bit-level change in the BF16 representation
            mask = (new_bf16 != old_bf16)
            nnz = mask.sum().item()

            if nnz > 0:
                # Extract indices and NEW values (not deltas)
                indices = mask.nonzero(as_tuple=False).t()  # Shape: [ndim, nnz]
                values = new_bf16[mask]  # Actual new BF16 values

                sparse_layers[name] = {
                    "indices": indices.to(torch.int32),  # Compact indices
                    "values": values,  # Already BF16
                    "shape": tuple(new_bf16.shape),
                    "nnz": nnz,
                }
                total_changed += nnz

            # Update old snapshot for next comparison
            self._old_bf16[name] = new_bf16.clone()

        # Create and save checkpoint
        checkpoint = {
            "step": step,
            "timestamp": time.time(),
            "format": "values",  # Indicates we store values, not deltas
            "layers": sparse_layers,
            "metadata": {
                "total_params": total_params,
                "total_changed": total_changed,
                "change_ratio": total_changed / total_params if total_params > 0 else 0.0,
                "num_changed_layers": len(sparse_layers),
                "dtype": "bfloat16",
            },
        }

        # Save to disk
        path = self.output_dir / f"delta_{step:06d}.pt"
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)

        # Log statistics
        change_ratio = checkpoint["metadata"]["change_ratio"]
        unchanged_ratio = 1.0 - change_ratio
        logger.info(
            f"[DeltaCheckpoint] Step {step}: "
            f"{unchanged_ratio:.2%} unchanged, "
            f"{total_changed:,}/{total_params:,} changed, "
            f"{len(sparse_layers)}/{len(self._old_bf16)} layers"
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
                "format": "values",  # Indicates we store values, not deltas
                "checkpoints": [],
                "total_params": checkpoint_meta["total_params"],
                "dtype": checkpoint_meta["dtype"],
            }

        # Add new checkpoint info
        metadata["checkpoints"].append({
            "step": step,
            "path": str(path.name),
            "change_ratio": checkpoint_meta["change_ratio"],
            "changed": checkpoint_meta["total_changed"],
            "changed_layers": checkpoint_meta["num_changed_layers"],
        })

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_checkpoint_values(checkpoint_path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a checkpoint and return sparse layer data with format info.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (layers dict, format string: "values" or "deltas")

    Example:
        >>> layers, fmt = load_checkpoint_values("checkpoints/delta_000042.pt")
        >>> print(f"Format: {fmt}, Layers: {len(layers)}")
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fmt = checkpoint.get("format", "deltas")  # Old format didn't have this field
    return checkpoint["layers"], fmt


def apply_sparse_values(
    weights: dict[str, torch.Tensor],
    sparse_layers: dict[str, Any],
) -> None:
    """Apply sparse values to weights dict (in-place).

    For "values" format: directly replaces values at specified indices.
    The sparse_layers contains actual new values, not deltas.

    Args:
        weights: Dictionary of weight tensors to modify in-place
        sparse_layers: Sparse layer data from checkpoint
    """
    for name, sparse_data in sparse_layers.items():
        if name not in weights:
            continue

        indices = sparse_data["indices"]
        values = sparse_data["values"]

        if indices.numel() > 0:
            # Convert indices to tuple for advanced indexing
            idx_tuple = tuple(indices[i].long() for i in range(indices.size(0)))
            # Directly set values (not add - these are the new values)
            weights[name][idx_tuple] = values.to(weights[name].dtype)


def apply_sparse_deltas(
    weights: dict[str, torch.Tensor],
    sparse_layers: dict[str, Any],
) -> None:
    """Apply sparse deltas to weights dict (in-place).

    For "deltas" format: adds delta values at specified indices.

    Args:
        weights: Dictionary of weight tensors to modify in-place
        sparse_layers: Sparse layer data from checkpoint
    """
    for name, sparse_data in sparse_layers.items():
        if name not in weights:
            continue

        indices = sparse_data["indices"]
        values = sparse_data["values"]
        shape = sparse_data["shape"]

        if indices.numel() > 0:
            # Reconstruct dense delta
            dense_delta = torch.zeros(shape, dtype=values.dtype)
            idx_tuple = tuple(indices[i].long() for i in range(indices.size(0)))
            dense_delta[idx_tuple] = values

            # Add to weights
            weights[name] = weights[name] + dense_delta.to(weights[name].dtype)


def reconstruct_weights_at_step(
    base_weights: dict[str, torch.Tensor],
    delta_dir: str | Path,
    target_step: int,
) -> dict[str, torch.Tensor]:
    """Reconstruct model weights at a specific step from base + changes.

    Handles both formats:
    - "values" format: Directly applies new values at each step
    - "deltas" format (legacy): Accumulates W_0 + Σ(deltas)

    Args:
        base_weights: Initial model weights (step 0)
        delta_dir: Directory containing delta checkpoints
        target_step: Target step to reconstruct

    Returns:
        Reconstructed weights at target_step

    Example:
        >>> base = {name: param.clone() for name, param in model.named_parameters()}
        >>> weights_100 = reconstruct_weights_at_step(base, "checkpoints/deltas", 100)
        >>> model.load_state_dict(weights_100, strict=False)
    """
    delta_dir = Path(delta_dir)

    # Start with base weights
    weights = {name: tensor.clone() for name, tensor in base_weights.items()}

    # Apply changes sequentially
    for step in range(target_step):
        delta_path = delta_dir / f"delta_{step:06d}.pt"
        if not delta_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {delta_path}")

        sparse_layers, fmt = load_checkpoint_values(delta_path)

        if fmt == "values":
            # New format: directly replace values
            apply_sparse_values(weights, sparse_layers)
        else:
            # Legacy format: add deltas
            apply_sparse_deltas(weights, sparse_layers)

    return weights


# Backward compatibility alias
def load_sparse_delta(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load a sparse checkpoint and reconstruct dense tensors.

    DEPRECATED: Use load_checkpoint_values() for new code.

    Returns dense tensors - for "values" format these are the new values,
    for "deltas" format these are the delta values.
    """
    sparse_layers, _ = load_checkpoint_values(checkpoint_path)

    dense_tensors = {}
    for name, sparse_data in sparse_layers.items():
        indices = sparse_data["indices"]
        values = sparse_data["values"]
        shape = sparse_data["shape"]

        # Create dense tensor
        dense = torch.zeros(shape, dtype=values.dtype)

        if indices.numel() > 0:
            idx_tuple = tuple(indices[i].long() for i in range(indices.size(0)))
            dense[idx_tuple] = values

        dense_tensors[name] = dense

    return dense_tensors
