"""Primitive data structures for model analysis.

This module provides immutable, type-safe primitives for capturing and analyzing
parameter changes during training. These are the foundational building blocks used
by higher-level metric computers.

Key Design Principles:
- Immutable data structures (snapshots cannot be modified after creation)
- CPU-offloaded storage (minimize GPU memory usage)
- Precision preservation (float32 for delta computation)
- Memory-safe operations (explicit device/dtype management)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn


class ParameterSnapshot:
    """Immutable snapshot of model parameters at a point in time.

    Captures all model parameters to CPU in specified dtype for memory-efficient
    storage and precise delta computation.

    Design:
        - Immutable: Once created, cannot be modified
        - CPU-offloaded: Parameters stored on CPU to save GPU memory
        - Timestamped: Records creation time for tracking
        - Hashable: Can be used as dict key or in sets

    Example:
        >>> snapshot = ParameterSnapshot(model, dtype=torch.float32)
        >>> print(f"Captured {len(snapshot)} parameters at {snapshot.timestamp}")
        >>> delta = snapshot.compute_delta(current_snapshot)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Capture model parameters to CPU.

        Args:
            model: Model to snapshot
            device: Target device for storage (default: "cpu")
            dtype: Target dtype for storage (default: torch.float32)
        """
        self._data: dict[str, torch.Tensor] = {}
        self._timestamp: float = time.time()
        self._device: str = device
        self._dtype: torch.dtype = dtype

        # Capture all parameters with explicit device/dtype conversion
        for name, param in model.named_parameters():
            self._data[name] = param.detach().to(device=device, dtype=dtype).clone()

    @property
    def data(self) -> dict[str, torch.Tensor]:
        """Read-only access to snapshot data."""
        return self._data

    @property
    def timestamp(self) -> float:
        """Unix timestamp when snapshot was captured."""
        return self._timestamp

    @property
    def device(self) -> str:
        """Device where parameters are stored."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of stored parameters."""
        return self._dtype

    def __len__(self) -> int:
        """Number of parameters in snapshot."""
        return len(self._data)

    def __contains__(self, name: str) -> bool:
        """Check if parameter name exists in snapshot."""
        return name in self._data

    def compute_delta(self, current: ParameterSnapshot) -> ParameterDelta:
        """Compute parameter change between this (old) snapshot and current snapshot.

        Args:
            current: Current snapshot (newer)

        Returns:
            ParameterDelta representing W_current - W_old

        Raises:
            ValueError: If snapshots have different parameter names
        """
        if set(self._data.keys()) != set(current._data.keys()):
            missing_in_current = set(self._data.keys()) - set(current._data.keys())
            missing_in_old = set(current._data.keys()) - set(self._data.keys())
            raise ValueError(
                f"Snapshot parameter mismatch. "
                f"Missing in current: {missing_in_current}, "
                f"Missing in old: {missing_in_old}"
            )

        return ParameterDelta(old_snapshot=self, new_snapshot=current)


class ParameterDelta:
    """Represents change between two parameter snapshots.

    Computes and stores W_new - W_old in float32 for numerical precision.
    Provides operations for analyzing and transforming deltas.

    Design:
        - Lazy computation: Deltas computed on first access
        - Float32 precision: Even if model uses bfloat16/float16
        - Sparse operations: Support masking and sparsification
        - Statistics: Built-in norm and distribution computations

    Example:
        >>> delta = old_snapshot.compute_delta(new_snapshot)
        >>> stats = delta.statistics()
        >>> print(f"L2 norm: {stats['norm_l2']:.4f}")
        >>> sparse_delta = delta.apply_sparse_mask(threshold=1e-6)
        >>> print(f"Kept {sparse_delta.sparsity_level():.2%} of parameters")
    """

    def __init__(
        self,
        old_snapshot: ParameterSnapshot,
        new_snapshot: ParameterSnapshot,
    ) -> None:
        """Compute delta between two snapshots.

        Args:
            old_snapshot: Earlier snapshot (W_old)
            new_snapshot: Later snapshot (W_new)
        """
        self._old = old_snapshot
        self._new = new_snapshot

        # Compute deltas in float32 for precision
        self._deltas: dict[str, torch.Tensor] = {}
        for name in old_snapshot.data.keys():
            old_param = old_snapshot.data[name].float()
            new_param = new_snapshot.data[name].float()
            self._deltas[name] = new_param - old_param

        # Cache for lazy statistics
        self._cached_stats: dict[str, float] | None = None

    @property
    def deltas(self) -> dict[str, torch.Tensor]:
        """Read-only access to delta tensors."""
        return self._deltas

    @property
    def old_snapshot(self) -> ParameterSnapshot:
        """Original snapshot (W_old)."""
        return self._old

    @property
    def new_snapshot(self) -> ParameterSnapshot:
        """Current snapshot (W_new)."""
        return self._new

    def apply_sparse_mask(
        self,
        threshold: float,
        mask_type: str = "magnitude",
    ) -> ParameterDelta:
        """Create new delta with small changes zeroed out.

        Args:
            threshold: Absolute threshold for keeping parameters
            mask_type: Type of mask ("magnitude" or "random")

        Returns:
            New ParameterDelta with sparse deltas

        Example:
            >>> sparse = delta.apply_sparse_mask(threshold=1e-6)
            >>> # Only parameters with |delta| > 1e-6 are kept
        """
        if mask_type == "magnitude":
            sparse_deltas = {
                name: delta * (delta.abs() > threshold) for name, delta in self._deltas.items()
            }
        elif mask_type == "random":
            # Random mask with same sparsity level as magnitude-based
            kept_ratio = self._compute_kept_ratio_at_threshold(threshold)
            sparse_deltas = {}
            for name, delta in self._deltas.items():
                rand_tensor = torch.rand_like(delta)
                mask = rand_tensor < kept_ratio
                sparse_deltas[name] = delta * mask
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}. Use 'magnitude' or 'random'.")

        return ParameterDelta._from_deltas(sparse_deltas, self._old, self._new)

    def statistics(self) -> dict[str, float]:
        """Compute statistics over all deltas.

        Returns:
            Dictionary with keys: norm_l2, norm_l1, norm_max, mean, std, min, max

        Example:
            >>> stats = delta.statistics()
            >>> print(f"L2 norm: {stats['norm_l2']:.4f}")
            >>> print(f"Mean change: {stats['mean']:.6f}")
        """
        if self._cached_stats is not None:
            return self._cached_stats

        # Concatenate all deltas for global statistics
        all_deltas = torch.cat([d.flatten() for d in self._deltas.values()])

        self._cached_stats = {
            "norm_l2": torch.norm(all_deltas, p=2).item(),
            "norm_l1": torch.norm(all_deltas, p=1).item(),
            "norm_max": all_deltas.abs().max().item(),
            "mean": all_deltas.mean().item(),
            "std": all_deltas.std().item(),
            "min": all_deltas.min().item(),
            "max": all_deltas.max().item(),
        }

        return self._cached_stats

    def per_layer_statistics(self) -> dict[str, dict[str, float]]:
        """Compute statistics per layer.

        Returns:
            Dictionary mapping layer name to statistics dict

        Example:
            >>> per_layer = delta.per_layer_statistics()
            >>> for name, stats in per_layer.items():
            ...     print(f"{name}: L2={stats['norm_l2']:.4f}")
        """
        result = {}

        for name, delta in self._deltas.items():
            flat = delta.flatten()
            result[name] = {
                "norm_l2": torch.norm(flat, p=2).item(),
                "norm_l1": torch.norm(flat, p=1).item(),
                "norm_max": flat.abs().max().item(),
                "mean": flat.mean().item(),
                "std": flat.std().item(),
            }

        return result

    def sparsity_at_threshold(self, threshold: float) -> dict[str, float]:
        """Compute sparsity statistics at given threshold.

        Args:
            threshold: Absolute threshold value

        Returns:
            Dictionary with:
                - kept_ratio: Fraction of parameters kept (non-zero after masking)
                - dropped_ratio: Fraction of parameters dropped
                - total_params: Total parameter count
                - kept_params: Parameters with |delta| > threshold

        Example:
            >>> sparsity = delta.sparsity_at_threshold(1e-6)
            >>> print(f"Kept {sparsity['kept_ratio']:.2%} of parameters")
        """
        total_params = sum(d.numel() for d in self._deltas.values())
        kept_params = sum((d.abs() > threshold).sum().item() for d in self._deltas.values())

        kept_ratio = kept_params / total_params if total_params > 0 else 0.0

        return {
            "kept_ratio": kept_ratio,
            "dropped_ratio": 1.0 - kept_ratio,
            "total_params": total_params,
            "kept_params": kept_params,
        }

    def _compute_kept_ratio_at_threshold(self, threshold: float) -> float:
        """Helper to compute kept ratio for random masking."""
        sparsity = self.sparsity_at_threshold(threshold)
        return sparsity["kept_ratio"]

    @classmethod
    def _from_deltas(
        cls,
        deltas: dict[str, torch.Tensor],
        old_snapshot: ParameterSnapshot,
        new_snapshot: ParameterSnapshot,
    ) -> ParameterDelta:
        """Internal constructor for creating ParameterDelta from pre-computed deltas.

        Used by sparse masking operations to avoid recomputing deltas.
        """
        instance = cls.__new__(cls)
        instance._deltas = deltas
        instance._old = old_snapshot
        instance._new = new_snapshot
        instance._cached_stats = None
        return instance

    def __len__(self) -> int:
        """Number of parameter tensors in delta."""
        return len(self._deltas)
