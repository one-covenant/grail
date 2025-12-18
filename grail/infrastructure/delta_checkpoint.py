"""Delta checkpoint utilities for sparse weight updates.

Provides functions for computing and applying sparse delta checkpoints,
enabling ~99% bandwidth reduction by only transmitting changed weights.

Format: Sparse COO (Coordinate) encoding in safetensors
- For each parameter: {name}.indices (int32) and {name}.values (original dtype)
- Shapes stored separately in delta_metadata.json

Key insight: We store the ACTUAL VALUES at changed positions (not deltas).
This eliminates floating-point arithmetic during reconstruction, ensuring
bit-exact results without precision loss from add operations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Default threshold for sparsity (0 = keep all non-zero changes)
DELTA_THRESHOLD = 0.0


def compute_sparse_delta(
    current_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
    threshold: float = DELTA_THRESHOLD,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], dict[str, Any]]:
    """Identify changed weights and store their actual values in COO format.

    Uses delta (current - base) only to identify WHICH weights changed.
    Stores the ACTUAL current values at those positions (not the deltas).
    This eliminates floating-point precision issues during reconstruction.

    Args:
        current_state: Current model state dict
        base_state: Base model state dict to compare against
        threshold: Minimum absolute change to include (0 = all changes)

    Returns:
        sparse_tensors: Dict with {name}.indices (int32) and {name}.values (original dtype)
        shapes: Dict mapping parameter names to original shapes
        stats: Dict with total_params, nonzero_params, sparsity_ratio
    """
    sparse_tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    total_params = 0
    changed_params = 0

    for name in current_state:
        if name not in base_state:
            logger.warning("Parameter %s not in base state, skipping", name)
            continue

        current_tensor = current_state[name].cpu()
        base_tensor = base_state[name].cpu()

        # Compute diff in float32 to find changed positions
        diff = current_tensor.float() - base_tensor.float()
        flat_diff = diff.flatten()
        total_params += flat_diff.numel()

        # Find indices where values changed
        if threshold > 0:
            changed_mask = flat_diff.abs() > threshold
        else:
            changed_mask = flat_diff != 0

        indices = changed_mask.nonzero(as_tuple=True)[0].to(torch.int32)
        changed_params += len(indices)

        if len(indices) > 0:
            # Store ACTUAL values at changed positions (not deltas)
            flat_current = current_tensor.flatten()
            values = flat_current[changed_mask]

            sparse_tensors[f"{name}.indices"] = indices
            sparse_tensors[f"{name}.values"] = values
            shapes[name] = list(current_tensor.shape)

    sparsity_ratio = 1.0 - (changed_params / total_params) if total_params > 0 else 1.0

    stats = {
        "total_params": total_params,
        "nonzero_params": changed_params,
        "sparsity_ratio": sparsity_ratio,
        "threshold": threshold,
    }

    logger.info(
        "Computed sparse update: %d/%d params changed (%.2f%% sparse), threshold=%.0e",
        changed_params,
        total_params,
        sparsity_ratio * 100,
        threshold,
    )

    return sparse_tensors, shapes, stats


def apply_sparse_delta(
    base_state: dict[str, torch.Tensor],
    sparse_tensors: dict[str, torch.Tensor],
    shapes: dict[str, list[int]],
    target_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Reconstruct full state by replacing values at changed positions.

    For unchanged weights: copies from base_state.
    For changed weights: directly assigns the stored values (no arithmetic).

    Args:
        base_state: Base model state dict
        sparse_tensors: Sparse update with {name}.indices and {name}.values
        shapes: Original shapes for each parameter
        target_dtype: Output dtype (typically bfloat16)

    Returns:
        Reconstructed state dict with full weights
    """
    result: dict[str, torch.Tensor] = {}

    for name, base_tensor in base_state.items():
        indices_key = f"{name}.indices"
        values_key = f"{name}.values"

        if indices_key in sparse_tensors:
            # Replace values at changed positions (direct assignment, no arithmetic)
            reconstructed = base_tensor.to(target_dtype).cpu().flatten()
            indices = sparse_tensors[indices_key].long()
            values = sparse_tensors[values_key].to(target_dtype)
            reconstructed[indices] = values
        else:
            # No changes - just copy base tensor
            reconstructed = base_tensor.to(target_dtype).cpu().flatten()

        original_shape = shapes.get(name, list(base_tensor.shape))
        result[name] = reconstructed.reshape(original_shape)

    return result


def compute_weights_hash(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute deterministic hash of all weights for verification.

    Uses sorted keys and raw bytes for reproducibility.
    The hash covers parameter names and their byte representations.

    Args:
        state_dict: Model state dict to hash

    Returns:
        SHA256 hex digest of all weights
    """
    hasher = hashlib.sha256()

    # Log input state info for debugging
    sample_dtypes: dict[str, int] = {}
    total_bytes = 0

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        # Convert to contiguous CPU bytes in a deterministic way.
        #
        # Note: torch.bfloat16 tensors cannot be converted to numpy directly.
        # We instead reinterpret the underlying storage as uint8 bytes.
        tensor_cpu = tensor.detach().cpu().contiguous()
        tensor_bytes = tensor_cpu.view(torch.uint8).numpy().tobytes()

        # Track dtype distribution for debugging
        dtype_str = str(tensor_cpu.dtype)
        sample_dtypes[dtype_str] = sample_dtypes.get(dtype_str, 0) + 1
        total_bytes += len(tensor_bytes)

        # Hash both name and tensor bytes
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor_cpu.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
        hasher.update(tensor_bytes)

    result_hash = hasher.hexdigest()

    logger.debug(
        "[compute_weights_hash] Computed hash: %s... | params=%d | bytes=%d | dtypes=%s",
        result_hash[:16],
        len(state_dict),
        total_bytes,
        sample_dtypes,
    )

    return result_hash


def verify_weights_hash(
    state_dict: dict[str, torch.Tensor],
    expected_hash: str,
) -> bool:
    """Verify that state dict matches expected hash.

    Args:
        state_dict: Model state dict to verify
        expected_hash: Expected SHA256 hex digest

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_weights_hash(state_dict)
    matches = actual_hash == expected_hash

    if not matches:
        # Collect diagnostic info about the state
        dtypes = {}
        for name, tensor in list(state_dict.items())[:5]:  # Sample first 5
            dtypes[name] = str(tensor.dtype)

        logger.error(
            "[verify_weights_hash] HASH MISMATCH: expected=%s, got=%s | "
            "params=%d | sample_dtypes=%s | "
            "This usually indicates floating-point precision differences during reconstruction",
            expected_hash,
            actual_hash,
            len(state_dict),
            dtypes,
        )
    else:
        logger.debug(
            "[verify_weights_hash] Hash verified: %s... | params=%d",
            actual_hash[:16],
            len(state_dict),
        )

    return matches


def estimate_sparse_size(
    nonzero_params: int,
    index_dtype: torch.dtype = torch.int32,
    value_dtype: torch.dtype = torch.float32,
) -> int:
    """Estimate size of sparse delta in bytes.

    Args:
        nonzero_params: Number of non-zero parameters
        index_dtype: Dtype for indices (typically int32)
        value_dtype: Dtype for values (typically float32)

    Returns:
        Estimated size in bytes
    """
    index_size = nonzero_params * torch.tensor([], dtype=index_dtype).element_size()
    value_size = nonzero_params * torch.tensor([], dtype=value_dtype).element_size()
    return index_size + value_size
