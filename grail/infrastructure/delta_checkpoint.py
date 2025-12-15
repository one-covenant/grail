"""Delta checkpoint utilities for sparse weight updates.

Provides functions for computing and applying sparse delta checkpoints,
enabling ~99% bandwidth reduction by only transmitting non-zero weight changes.

Format: Sparse COO (Coordinate) encoding in safetensors
- For each parameter: {name}.indices (int32) and {name}.values (float32)
- Shapes stored separately in delta_metadata.json

Precision guarantee:
- Delta computation and application done in float32
- Final weights cast to target dtype (typically bfloat16)
- Hash verification ensures bit-exact reconstruction
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Default threshold for sparsity (0 = keep all non-zero deltas)
DELTA_THRESHOLD = 0.0


def compute_sparse_delta(
    current_state: dict[str, torch.Tensor],
    base_state: dict[str, torch.Tensor],
    threshold: float = DELTA_THRESHOLD,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]], dict[str, Any]]:
    """Compute sparse delta using COO format.

    Only stores indices and values where |delta| > threshold.
    All computation done in float32 for precision.

    Args:
        current_state: Current model state dict
        base_state: Base model state dict to compute delta from
        threshold: Minimum absolute delta to include (0 = all non-zero)

    Returns:
        sparse_tensors: Dict with {name}.indices (int32) and {name}.values (float32)
        shapes: Dict mapping parameter names to original shapes
        stats: Dict with total_params, nonzero_params, sparsity_ratio
    """
    sparse_tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    total_params = 0
    nonzero_params = 0

    for name in current_state:
        if name not in base_state:
            logger.warning("Parameter %s not in base state, skipping", name)
            continue

        # Compute delta in float32 for precision
        current_fp32 = current_state[name].float().cpu()
        base_fp32 = base_state[name].float().cpu()
        delta = current_fp32 - base_fp32

        flat_delta = delta.flatten()
        total_params += flat_delta.numel()

        # Find non-zero indices (above threshold)
        if threshold > 0:
            nonzero_mask = flat_delta.abs() > threshold
        else:
            # threshold == 0: include all non-zero values
            nonzero_mask = flat_delta != 0

        indices = nonzero_mask.nonzero(as_tuple=True)[0].to(torch.int32)
        values = flat_delta[nonzero_mask].float()

        nonzero_params += len(indices)

        if len(indices) > 0:
            sparse_tensors[f"{name}.indices"] = indices
            sparse_tensors[f"{name}.values"] = values
            shapes[name] = list(delta.shape)

    sparsity_ratio = 1.0 - (nonzero_params / total_params) if total_params > 0 else 1.0

    stats = {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_ratio": sparsity_ratio,
        "threshold": threshold,
    }

    logger.info(
        "Computed sparse delta: %d/%d params (%.2f%% sparse), threshold=%.0e",
        nonzero_params,
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
    """Reconstruct full state from base + sparse delta.

    All computation done in float32 for precision, then cast to target_dtype.

    Args:
        base_state: Base model state dict
        sparse_tensors: Sparse delta with {name}.indices and {name}.values
        shapes: Original shapes for each parameter
        target_dtype: Output dtype (typically bfloat16)

    Returns:
        Reconstructed state dict with full weights
    """
    result: dict[str, torch.Tensor] = {}

    for name, base_tensor in base_state.items():
        # Start with base in float32
        reconstructed = base_tensor.float().cpu().flatten()

        # Apply sparse delta if exists
        indices_key = f"{name}.indices"
        values_key = f"{name}.values"

        if indices_key in sparse_tensors:
            indices = sparse_tensors[indices_key].long()
            values = sparse_tensors[values_key].float()
            reconstructed[indices] += values

        # Reshape and cast to target dtype
        original_shape = shapes.get(name, list(base_tensor.shape))
        result[name] = reconstructed.reshape(original_shape).to(target_dtype)

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

    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        # Convert to contiguous CPU bytes in a deterministic way
        tensor_cpu = tensor.detach().cpu().contiguous()
        tensor_bytes = tensor_cpu.numpy().tobytes()

        # Hash both name and tensor bytes
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor_bytes)

    return hasher.hexdigest()


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
        logger.error(
            "Weights hash mismatch: expected %s..., got %s...",
            expected_hash[:16],
            actual_hash[:16],
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
