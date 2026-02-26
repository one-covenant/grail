#!/usr/bin/env python3
"""Delta checkpoint callback for storing sparse parameter updates.

Stores changed parameter VALUES (not deltas) after each optimizer step.
Compares parameters in a configurable dtype (BF16 or FP32) to find exact
changes, stores new values at changed positions in sparse COO format.

Key design:
    - Configurable comparison dtype via snapshot_dtype:
        - "bfloat16": Matches what vLLM sees after weight sync (~2 bytes/param)
        - "float32": Catches smaller changes invisible in BF16 (~4 bytes/param)
    - Store values, not deltas: W_new at changed positions
    - Single snapshot: Only keep one snapshot per param
    - Exact comparison: w_t != w_{t-1} finds bitwise changes in snapshot dtype

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

Async Mode:
    When async_save=True, delta computation and saving run in a background
    thread (CPU-only), overlapping with the next training step. This can
    save ~150-200s per step while using minimal extra memory (~6GB).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
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
        2. on_optimizer_step() callback runs <- we capture here
        3. zero_grad() is called (gradients cleared)

    Args:
        output_dir: Directory to save delta checkpoints
        enabled: Enable/disable delta checkpointing (default: True)
        save_metadata: Save metadata.json with checkpoint list (default: True)
        profiler: Optional profiler for timing measurements
        async_save: Run delta computation and saving in background thread.
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        snapshot_dtype: str = "bfloat16",
        save_metadata: bool = True,
        profiler: Any | None = None,
        async_save: bool = True,
    ) -> None:
        """Initialize delta checkpoint callback.

        Args:
            output_dir: Directory to save delta checkpoints.
            enabled: Enable/disable delta checkpointing.
            snapshot_dtype: Dtype for comparison and storage ("bfloat16" or "float32").
                           BF16 matches vLLM's view; FP32 catches finer-grained changes.
            save_metadata: Save metadata.json with checkpoint list.
            profiler: Optional profiler for timing instrumentation.
            async_save: Run delta computation and saving in background thread.
                       Saves ~150-200s per step by overlapping with training.
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.save_metadata = save_metadata
        self.step_count = 0
        self._profiler = profiler

        # Resolve snapshot dtype
        _dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        if snapshot_dtype not in _dtype_map:
            raise ValueError(f"snapshot_dtype must be 'bfloat16' or 'float32', got '{snapshot_dtype}'")
        self._snapshot_dtype: torch.dtype = _dtype_map[snapshot_dtype]
        self._snapshot_dtype_name: str = snapshot_dtype

        # Snapshot of previous weights (single copy)
        self._old_snapshot: dict[str, torch.Tensor] = {}

        # Async save configuration (CPU-only background thread)
        self._async_save = async_save and enabled
        self._executor: ThreadPoolExecutor | None = None
        self._pending_future: Future | None = None
        self._save_lock = Lock()  # Ensures sequential metadata updates
        self._shutdown = False

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            if self._async_save:
                # Single worker ensures saves happen in order
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="delta_checkpoint",
                )
                logger.info(
                    f"[DeltaCheckpoint] Initialized with async save: "
                    f"output_dir={output_dir}, dtype={snapshot_dtype}"
                )
            else:
                logger.info(
                    f"[DeltaCheckpoint] Initialized (sync mode): "
                    f"output_dir={output_dir}, dtype={snapshot_dtype}"
                )
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

        Compares current weights against previous snapshot (in configured dtype),
        finds changed positions, stores new values at those positions.
        In async mode, snapshot capture is synchronous but delta computation
        and saving run in a background thread.
        """
        if not self.enabled or model is None:
            return

        # Only process on main process in distributed training
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not is_world_process_zero:
            return

        profiler = self._profiler

        # First step: just capture snapshot, no comparison yet
        if not self._old_snapshot:
            if profiler:
                with profiler.track("delta_checkpoint_init"):
                    self._capture_snapshot(model)
            else:
                self._capture_snapshot(model)
            logger.info(
                f"[DeltaCheckpoint] Step {self.step_count}: Initial snapshot captured "
                f"(dtype={self._snapshot_dtype_name})"
            )
            return

        # Compare and save changed values
        if self._async_save and self._executor is not None:
            self._dispatch_async_save(model, step=self.step_count)
        else:
            if profiler:
                with profiler.track("delta_checkpoint_compare_save"):
                    self._compare_and_save(model, step=self.step_count)
            else:
                self._compare_and_save(model, step=self.step_count)

        self.step_count += 1

    def _dispatch_async_save(self, model: Any, step: int) -> None:
        """Capture snapshot and dispatch comparison to background thread.

        Snapshot capture is synchronous (GPU->CPU copy), but the comparison
        and save run in a background thread using only CPU tensors.
        """
        # Capture current snapshot (synchronous - must happen before next step)
        current_snapshot: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            current_snapshot[name] = param.data.detach().to(
                device="cpu", dtype=self._snapshot_dtype
            ).clone()

        # Hand off old + new snapshots to background thread
        old_snapshot = self._old_snapshot
        self._old_snapshot = current_snapshot  # Update for next step immediately

        assert self._executor is not None
        self._pending_future = self._executor.submit(
            self._background_save_task,
            old_snapshot,
            current_snapshot,
            step,
        )

    def _background_save_task(
        self,
        old_snapshot: dict[str, torch.Tensor],
        current_snapshot: dict[str, torch.Tensor],
        step: int,
    ) -> None:
        """Background task: compare snapshots and save delta (CPU only).

        This method runs in a background thread and operates exclusively
        on CPU tensors. It never touches GPU memory.
        """
        try:
            sparse_layers = {}
            total_params = 0
            total_changed = 0

            for name, new_vals in current_snapshot.items():
                total_params += new_vals.numel()
                old_vals = old_snapshot.get(name)
                if old_vals is None:
                    continue

                mask = new_vals != old_vals
                nnz = mask.sum().item()

                if nnz > 0:
                    indices = mask.nonzero(as_tuple=False).t()
                    values = new_vals[mask]
                    sparse_layers[name] = {
                        "indices": indices.to(torch.int32),
                        "values": values,
                        "shape": tuple(new_vals.shape),
                        "nnz": nnz,
                    }
                    total_changed += nnz

            checkpoint = {
                "step": step,
                "timestamp": time.time(),
                "format": "values",
                "layers": sparse_layers,
                "metadata": {
                    "total_params": total_params,
                    "total_changed": total_changed,
                    "change_ratio": total_changed / total_params if total_params > 0 else 0.0,
                    "num_changed_layers": len(sparse_layers),
                    "dtype": self._snapshot_dtype_name,
                },
            }

            path = self.output_dir / f"delta_{step:06d}.pt"
            with self._save_lock:
                torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
                if self.save_metadata:
                    self._update_metadata(checkpoint["metadata"], step, path)

            change_ratio = checkpoint["metadata"]["change_ratio"]
            unchanged_ratio = 1.0 - change_ratio
            logger.info(
                f"[DeltaCheckpoint] Step {step}: "
                f"{unchanged_ratio:.2%} unchanged, "
                f"{total_changed:,}/{total_params:,} changed, "
                f"{len(sparse_layers)}/{len(current_snapshot)} layers"
            )

        except Exception as e:
            logger.error(f"[DeltaCheckpoint] Background save failed step {step}: {e}")

    def _wait_for_pending(self) -> None:
        """Wait for any pending background save to complete."""
        if self._pending_future is not None:
            try:
                self._pending_future.result(timeout=600)  # 10 min timeout
            except TimeoutError:
                logger.error("[DeltaCheckpoint] Background save timed out after 10 minutes")
            except Exception as e:
                logger.error(f"[DeltaCheckpoint] Background save raised exception: {e}")
            finally:
                self._pending_future = None

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Clean up resources when training ends."""
        if not self.enabled:
            return

        # Wait for final background save to complete
        if self._async_save:
            logger.info("[DeltaCheckpoint] Waiting for final background save...")
            self._wait_for_pending()

        # Shutdown thread pool
        self._shutdown_executor()

        logger.info(f"[DeltaCheckpoint] Training complete. Saved {self.step_count} deltas.")

    def _shutdown_executor(self) -> None:
        """Gracefully shutdown the thread pool executor."""
        if self._executor is not None and not self._shutdown:
            self._shutdown = True
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
            logger.debug("[DeltaCheckpoint] Background executor shutdown complete")

    def __del__(self) -> None:
        """Ensure executor is cleaned up on garbage collection."""
        try:
            self._shutdown_executor()
        except Exception:
            pass  # Ignore errors during garbage collection

    def _capture_snapshot(self, model: Any) -> None:
        """Capture all model parameters in snapshot dtype on CPU."""
        self._old_snapshot.clear()
        for name, param in model.named_parameters():
            self._old_snapshot[name] = param.data.detach().to(
                device="cpu", dtype=self._snapshot_dtype
            ).clone()

    def _compare_and_save(self, model: Any, step: int) -> None:
        """Compare current weights to old snapshot, save changed values."""
        sparse_layers = {}
        total_params = 0
        total_changed = 0

        for name, param in model.named_parameters():
            new_vals = param.data.detach().to(device="cpu", dtype=self._snapshot_dtype)
            total_params += new_vals.numel()

            old_vals = self._old_snapshot.get(name)
            if old_vals is None:
                # New parameter (shouldn't happen in normal training)
                self._old_snapshot[name] = new_vals.clone()
                continue

            # Find changed positions: exact comparison in snapshot dtype
            mask = (new_vals != old_vals)
            nnz = mask.sum().item()

            if nnz > 0:
                indices = mask.nonzero(as_tuple=False).t()  # Shape: [ndim, nnz]
                values = new_vals[mask]

                sparse_layers[name] = {
                    "indices": indices.to(torch.int32),
                    "values": values,
                    "shape": tuple(new_vals.shape),
                    "nnz": nnz,
                }
                total_changed += nnz

            # Update old snapshot for next comparison
            self._old_snapshot[name] = new_vals.clone()

        # Create and save checkpoint
        checkpoint = {
            "step": step,
            "timestamp": time.time(),
            "format": "values",
            "layers": sparse_layers,
            "metadata": {
                "total_params": total_params,
                "total_changed": total_changed,
                "change_ratio": total_changed / total_params if total_params > 0 else 0.0,
                "num_changed_layers": len(sparse_layers),
                "dtype": self._snapshot_dtype_name,
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
            f"{len(sparse_layers)}/{len(self._old_snapshot)} layers"
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
    - "deltas" format (legacy): Accumulates W_0 + sum(deltas)

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
