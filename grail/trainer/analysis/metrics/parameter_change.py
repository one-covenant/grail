"""Parameter change metrics computation.

Analyzes magnitude and sparsity of parameter updates during training.
Ported from grail.trainer.param_tracker with cleaner architecture.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot

# Component patterns for Llama/Qwen-style architectures
COMPONENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "q_proj": re.compile(r"\.q_proj\."),
    "k_proj": re.compile(r"\.k_proj\."),
    "v_proj": re.compile(r"\.v_proj\."),
    "o_proj": re.compile(r"\.o_proj\."),
    "gate_proj": re.compile(r"\.gate_proj\."),
    "up_proj": re.compile(r"\.up_proj\."),
    "down_proj": re.compile(r"\.down_proj\."),
    "embed_tokens": re.compile(r"embed_tokens"),
    "lm_head": re.compile(r"lm_head"),
    "layernorm": re.compile(r"(layernorm|layer_norm|norm)", re.IGNORECASE),
}

# Layer index extraction pattern
LAYER_PATTERN = re.compile(r"layers\.(\d+)")


def _extract_layer_index(name: str) -> int | None:
    """Extract layer index from parameter name.

    Args:
        name: Parameter name (e.g., "model.layers.15.self_attn.q_proj.weight")

    Returns:
        Layer index as int, or None if not found
    """
    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def _extract_component(name: str) -> str | None:
    """Extract component type from parameter name.

    Args:
        name: Parameter name

    Returns:
        Component name (e.g., "q_proj", "gate_proj"), or None if not matched
    """
    for component_name, pattern in COMPONENT_PATTERNS.items():
        if pattern.search(name):
            return component_name
    return None


class ParameterChangeMetrics(MetricComputer):
    """Compute parameter change statistics.

    Analyzes parameter updates after optimizer steps, computing:
    - Global sparsity and magnitude statistics
    - Multi-threshold sparsity analysis
    - Per-layer breakdowns (optional)
    - Per-component breakdowns (optional)
    - Sign flip tracking (optional)

    This is a stateless metric computer - all state is in the ParameterDelta.

    Example:
        >>> computer = ParameterChangeMetrics(
        ...     thresholds=[1e-8, 1e-6, 1e-4],
        ...     track_per_layer=True,
        ... )
        >>> metrics = computer.compute(delta=parameter_delta)
        >>> print(metrics["param_change/sparsity_at_1e-06"])
    """

    def __init__(
        self,
        thresholds: list[float] | None = None,
        track_per_layer: bool = False,
        track_components: bool = False,
        track_sign_flips: bool = False,
    ) -> None:
        """Initialize parameter change metric computer.

        Args:
            thresholds: Sparsity thresholds to analyze (default: [1e-8, 1e-6, 1e-4])
            track_per_layer: Compute per-layer statistics
            track_components: Compute per-component statistics (attention, MLP, etc.)
            track_sign_flips: Track parameters that changed sign
        """
        self.thresholds = thresholds if thresholds is not None else [1e-8, 1e-6, 1e-4]
        self.track_per_layer = track_per_layer
        self.track_components = track_components
        self.track_sign_flips = track_sign_flips

    def compute(
        self,
        delta: ParameterDelta | None = None,
        old_snapshot: ParameterSnapshot | None = None,
        current_snapshot: ParameterSnapshot | None = None,
        context: AnalysisContext | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute parameter change metrics.

        Args:
            delta: Parameter delta to analyze
            old_snapshot: Old snapshot (for sign flip tracking)
            current_snapshot: Current snapshot (for sign flip tracking)
            context: Analysis context (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metrics with "param_change/" prefix
        """
        if delta is None:
            return {}

        metrics: dict[str, float] = {}

        # Global statistics from delta.statistics()
        stats = delta.statistics()
        metrics["param_change/norm_l2"] = stats["norm_l2"]
        metrics["param_change/norm_l1"] = stats["norm_l1"]
        metrics["param_change/norm_max"] = stats["norm_max"]
        metrics["param_change/mean"] = stats["mean"]
        metrics["param_change/std"] = stats["std"]

        # Multi-threshold sparsity analysis
        for threshold in self.thresholds:
            sparsity_info = delta.sparsity_at_threshold(threshold)
            thresh_str = f"{threshold:.0e}"

            # Sparsity = fraction unchanged (dropped)
            metrics[f"param_change/sparsity_at_{thresh_str}"] = sparsity_info["dropped_ratio"]
            # Changed ratio = fraction kept (changed)
            metrics[f"param_change/changed_ratio_at_{thresh_str}"] = sparsity_info["kept_ratio"]

        # Per-layer analysis
        if self.track_per_layer:
            per_layer = self._compute_per_layer_stats(delta)
            metrics.update(per_layer)

        # Per-component analysis
        if self.track_components:
            per_component = self._compute_per_component_stats(delta)
            metrics.update(per_component)

        # Sign flip tracking
        if self.track_sign_flips and old_snapshot is not None and current_snapshot is not None:
            sign_flip = self._compute_sign_flips(old_snapshot, current_snapshot)
            metrics.update(sign_flip)

        return metrics

    def _compute_per_layer_stats(self, delta: ParameterDelta) -> dict[str, float]:
        """Compute statistics grouped by layer index.

        Args:
            delta: Parameter delta

        Returns:
            Dictionary with per-layer metrics
        """
        metrics: dict[str, float] = {}
        layer_stats: dict[int, list[torch.Tensor]] = defaultdict(list)

        # Group deltas by layer index
        for name, delta_tensor in delta.deltas.items():
            layer_idx = _extract_layer_index(name)
            if layer_idx is not None:
                layer_stats[layer_idx].append(delta_tensor.flatten())

        # Compute stats per layer
        for layer_idx, tensors in layer_stats.items():
            if not tensors:
                continue

            all_deltas = torch.cat(tensors)
            total = all_deltas.numel()

            # Mean absolute delta
            metrics[f"param_change/layer_{layer_idx}/mean_abs_delta"] = (
                all_deltas.abs().mean().item()
            )

            # Sparsity at primary threshold (if available)
            if self.thresholds:
                threshold = self.thresholds[0] if len(self.thresholds) > 0 else 1e-6
                unchanged = (all_deltas.abs() <= threshold).sum().item()
                sparsity = unchanged / total if total > 0 else 0.0
                metrics[f"param_change/layer_{layer_idx}/sparsity"] = sparsity

        return metrics

    def _compute_per_component_stats(self, delta: ParameterDelta) -> dict[str, float]:
        """Compute statistics grouped by component type.

        Args:
            delta: Parameter delta

        Returns:
            Dictionary with per-component metrics
        """
        metrics: dict[str, float] = {}
        component_stats: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Group deltas by component
        for name, delta_tensor in delta.deltas.items():
            component = _extract_component(name)
            if component is not None:
                component_stats[component].append(delta_tensor.flatten())

        # Compute stats per component
        for component, tensors in component_stats.items():
            if not tensors:
                continue

            all_deltas = torch.cat(tensors)
            total = all_deltas.numel()

            # Mean absolute delta
            metrics[f"param_change/component/{component}/mean_abs_delta"] = (
                all_deltas.abs().mean().item()
            )

            # Sparsity at primary threshold
            if self.thresholds:
                threshold = self.thresholds[0] if len(self.thresholds) > 0 else 1e-6
                unchanged = (all_deltas.abs() <= threshold).sum().item()
                sparsity = unchanged / total if total > 0 else 0.0
                metrics[f"param_change/component/{component}/sparsity"] = sparsity

        return metrics

    def _compute_sign_flips(
        self,
        old_snapshot: ParameterSnapshot,
        current_snapshot: ParameterSnapshot,
    ) -> dict[str, float]:
        """Count parameters that changed sign.

        Args:
            old_snapshot: Old parameter values
            current_snapshot: Current parameter values

        Returns:
            Dictionary with sign flip metrics
        """
        total_params = 0
        sign_flips = 0

        for name in old_snapshot.data.keys():
            if name not in current_snapshot.data:
                continue

            old_param = old_snapshot.data[name]
            new_param = current_snapshot.data[name]

            # Sign flip: old and new have different signs (excluding zeros)
            old_sign = torch.sign(old_param)
            new_sign = torch.sign(new_param)

            # Count sign changes (ignoring zeros)
            non_zero_mask = (old_sign != 0) & (new_sign != 0)
            sign_changed = (old_sign != new_sign) & non_zero_mask

            sign_flips += sign_changed.sum().item()
            total_params += non_zero_mask.sum().item()

        sign_flip_ratio = sign_flips / total_params if total_params > 0 else 0.0

        return {
            "param_change/sign_flip_count": float(sign_flips),
            "param_change/sign_flip_ratio": sign_flip_ratio,
        }

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "ParameterChangeMetrics"
