"""Gradient sparsity metrics computation.

Analyzes the sparsity pattern of gradients (what fraction are below thresholds).
Useful for understanding gradient flow and potential optimization opportunities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot

# Special prefix for histogram data (detected by logging callbacks)
HISTOGRAM_KEY_PREFIX = "gradient/_histogram/"


class GradientSparsityMetrics(MetricComputer):
    """Compute gradient sparsity statistics.

    Analyzes gradient magnitude distribution by computing:
    - Global gradient norms (L1, L2, max, mean, std)
    - Multi-threshold sparsity analysis (what % of gradients are below threshold)
    - Per-layer breakdown (optional)

    This metric requires access to the model with gradients still available
    (typically from on_optimizer_step callback).

    Example:
        >>> computer = GradientSparsityMetrics(
        ...     thresholds=[1e-8, 1e-6, 1e-4],
        ...     track_per_layer=False,
        ... )
        >>> metrics = computer.compute(context=AnalysisContext(model=model))
        >>> print(metrics["gradient/sparsity_at_1e-06"])
    """

    def __init__(
        self,
        thresholds: list[float] | None = None,
        track_per_layer: bool = False,
        histogram_samples: int = 100_000,
    ) -> None:
        """Initialize gradient sparsity metric computer.

        Args:
            thresholds: Sparsity thresholds to analyze (default: [0.0, 1e-8, 1e-6, 1e-4])
            track_per_layer: Compute per-layer statistics (more verbose)
            histogram_samples: Max samples for gradient histogram (0 to disable)
        """
        self.thresholds = thresholds if thresholds is not None else [0.0, 1e-8, 1e-6, 1e-4]
        self.track_per_layer = track_per_layer
        self.histogram_samples = histogram_samples

    def requires_model(self) -> bool:
        """Requires model to access gradients."""
        return True

    def compute(
        self,
        delta: ParameterDelta | None = None,
        old_snapshot: ParameterSnapshot | None = None,
        current_snapshot: ParameterSnapshot | None = None,
        context: AnalysisContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute gradient sparsity metrics.

        Args:
            delta: Unused (for compatibility)
            old_snapshot: Unused (for compatibility)
            current_snapshot: Unused (for compatibility)
            context: Must contain model with gradients
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metrics with "gradient/" prefix.
            Histogram data is stored under HISTOGRAM_KEY_PREFIX keys as numpy arrays.
        """
        if context is None or context.model is None:
            return {}

        model = context.model
        metrics: dict[str, Any] = {}

        # IMPORTANT: compute on CPU to avoid GPU OOM. Moving gradients to CPU
        # before float32 conversion prevents large transient GPU allocations.
        total_elements = 0
        sum_val = 0.0
        sum_sq = 0.0  # Used for both L2 norm and variance
        l1_sum = 0.0
        min_val: float | None = None
        max_val: float | None = None
        abs_max = 0.0
        small_counts: dict[float, int] = dict.fromkeys(self.thresholds, 0)

        # Reservoir sampling for histogram (memory-efficient)
        histogram_reservoir: list[float] = []
        reservoir_idx = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.to_dense() if param.grad.is_sparse else param.grad

            # Move to CPU first, then convert dtype (avoids GPU float32 allocation)
            flat = grad.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if flat.numel() == 0:
                continue

            abs_flat = flat.abs()
            n = flat.numel()
            total_elements += n
            sum_val += flat.sum().item()
            sum_sq += flat.pow(2).sum().item()
            l1_sum += abs_flat.sum().item()

            cur_min, cur_max = flat.min().item(), flat.max().item()
            min_val = cur_min if min_val is None else min(min_val, cur_min)
            max_val = cur_max if max_val is None else max(max_val, cur_max)
            abs_max = max(abs_max, abs_flat.max().item())

            for threshold in self.thresholds:
                small_counts[threshold] += int((abs_flat <= threshold).sum().item())

            # Reservoir sampling for histogram (vectorized for performance)
            if self.histogram_samples > 0:
                flat_np = flat.numpy()
                n_new = len(flat_np)
                if reservoir_idx < self.histogram_samples:
                    # Still filling reservoir - take what we can
                    take = min(n_new, self.histogram_samples - reservoir_idx)
                    histogram_reservoir.extend(flat_np[:take].tolist())
                    # Handle remaining elements with reservoir sampling
                    remaining = flat_np[take:]
                    if len(remaining) > 0:
                        start_idx = reservoir_idx + take
                        indices = np.arange(start_idx, start_idx + len(remaining))
                        probs = self.histogram_samples / (indices + 1)
                        mask = np.random.random(len(remaining)) < probs
                        replace_indices = np.random.randint(
                            0, self.histogram_samples, size=mask.sum()
                        )
                        for idx, val in zip(replace_indices, remaining[mask], strict=False):
                            histogram_reservoir[idx] = float(val)
                else:
                    # Reservoir full - vectorized replacement
                    indices = np.arange(reservoir_idx, reservoir_idx + n_new)
                    probs = self.histogram_samples / (indices + 1)
                    mask = np.random.random(n_new) < probs
                    replace_indices = np.random.randint(0, self.histogram_samples, size=mask.sum())
                    for idx, val in zip(replace_indices, flat_np[mask], strict=False):
                        histogram_reservoir[idx] = float(val)
                reservoir_idx += n_new

            if self.track_per_layer:
                metrics[f"gradient/layer/{name}/norm_l2"] = flat.pow(2).sum().sqrt().item()
                metrics[f"gradient/layer/{name}/mean_abs"] = abs_flat.mean().item()
                if self.thresholds and n > 0:
                    layer_small = int((abs_flat <= self.thresholds[0]).sum().item())
                    metrics[f"gradient/layer/{name}/sparsity"] = layer_small / n

        if total_elements == 0:
            return {}

        mean = sum_val / total_elements
        variance = (sum_sq / total_elements) - (mean**2)
        std = max(0.0, variance) ** 0.5  # Clamp to avoid sqrt of negative due to FP error

        metrics["gradient/norm_l2"] = sum_sq**0.5
        metrics["gradient/norm_l1"] = l1_sum
        metrics["gradient/norm_max"] = abs_max
        metrics["gradient/mean"] = mean
        metrics["gradient/std"] = std
        metrics["gradient/min"] = min_val if min_val is not None else 0.0
        metrics["gradient/max"] = max_val if max_val is not None else 0.0

        for threshold in self.thresholds:
            small_count = small_counts[threshold]
            dense_count = total_elements - small_count
            sparsity = small_count / total_elements
            thresh_str = f"{threshold:.0e}"
            metrics[f"gradient/sparsity_at_{thresh_str}"] = sparsity
            metrics[f"gradient/dense_ratio_at_{thresh_str}"] = 1.0 - sparsity
            metrics[f"gradient/sparse_count_at_{thresh_str}"] = float(small_count)
            metrics[f"gradient/dense_count_at_{thresh_str}"] = float(dense_count)

        # Add histogram samples (will be converted to wandb.Histogram by callback)
        if histogram_reservoir:
            metrics[f"{HISTOGRAM_KEY_PREFIX}values"] = np.array(
                histogram_reservoir, dtype=np.float32
            )
            # Also add log-scale histogram of absolute values for magnitude analysis
            abs_samples = np.abs(histogram_reservoir).astype(np.float32)
            # Clamp small values to avoid log(0)
            abs_samples = np.clip(abs_samples, 1e-38, None)
            metrics[f"{HISTOGRAM_KEY_PREFIX}log_magnitude"] = np.log10(abs_samples)

        return metrics

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "GradientSparsityMetrics"
