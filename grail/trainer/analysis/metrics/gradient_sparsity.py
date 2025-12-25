"""Gradient sparsity metrics computation.

Analyzes the sparsity pattern of gradients (what fraction are below thresholds).
Useful for understanding gradient flow and potential optimization opportunities.
"""

from __future__ import annotations

from typing import Any

import torch

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot


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
    ) -> None:
        """Initialize gradient sparsity metric computer.

        Args:
            thresholds: Sparsity thresholds to analyze (default: [0.0, 1e-8, 1e-6, 1e-4])
            track_per_layer: Compute per-layer statistics (more verbose)
        """
        self.thresholds = thresholds if thresholds is not None else [0.0, 1e-8, 1e-6, 1e-4]
        self.track_per_layer = track_per_layer

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
    ) -> dict[str, float]:
        """Compute gradient sparsity metrics.

        Args:
            delta: Unused (for compatibility)
            old_snapshot: Unused (for compatibility)
            current_snapshot: Unused (for compatibility)
            context: Must contain model with gradients
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metrics with "gradient/" prefix
        """
        if context is None or context.model is None:
            return {}

        model = context.model
        metrics: dict[str, float] = {}

        # Collect all gradients
        all_grads = []
        layer_grads: dict[str, list[torch.Tensor]] = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                all_grads.append(grad.flatten())

                if self.track_per_layer:
                    layer_grads[name] = [grad]

        if not all_grads:
            return {}

        # Concatenate all gradients
        all_grads_tensor = torch.cat(all_grads)

        # Global statistics
        metrics["gradient/norm_l2"] = torch.norm(all_grads_tensor, p=2).item()
        metrics["gradient/norm_l1"] = torch.norm(all_grads_tensor, p=1).item()
        metrics["gradient/norm_max"] = all_grads_tensor.abs().max().item()
        metrics["gradient/mean"] = all_grads_tensor.mean().item()
        metrics["gradient/std"] = all_grads_tensor.std().item()
        metrics["gradient/min"] = all_grads_tensor.min().item()
        metrics["gradient/max"] = all_grads_tensor.max().item()

        # Multi-threshold sparsity analysis
        total_elements = all_grads_tensor.numel()
        for threshold in self.thresholds:
            small_count = (all_grads_tensor.abs() <= threshold).sum().item()
            sparsity = small_count / total_elements
            thresh_str = f"{threshold:.0e}"

            metrics[f"gradient/sparsity_at_{thresh_str}"] = sparsity
            metrics[f"gradient/dense_ratio_at_{thresh_str}"] = 1.0 - sparsity

        # Per-layer analysis (optional)
        if self.track_per_layer:
            for name, grad_list in layer_grads.items():
                if not grad_list:
                    continue

                layer_grad = grad_list[0].flatten()
                layer_total = layer_grad.numel()

                metrics[f"gradient/layer/{name}/norm_l2"] = torch.norm(layer_grad, p=2).item()
                metrics[f"gradient/layer/{name}/mean_abs"] = layer_grad.abs().mean().item()

                # Sparsity at primary threshold
                if self.thresholds:
                    threshold = self.thresholds[0]
                    small_count = (layer_grad.abs() <= threshold).sum().item()
                    sparsity = small_count / layer_total if layer_total > 0 else 0.0
                    metrics[f"gradient/layer/{name}/sparsity"] = sparsity

        return metrics

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "GradientSparsityMetrics"
