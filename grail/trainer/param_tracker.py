"""Parameter change tracking for training analysis.

Measures sparsity and magnitude of parameter updates after optimizer steps.
Designed for memory-efficient operation via CPU offloading and streaming computation.
"""

from __future__ import annotations

import gc
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Multi-threshold analysis for distribution insights
# Include 0 to see truly unchanged params, plus small thresholds for low LR training
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4)

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


@dataclass
class LayerStats:
    """Accumulated statistics for a single layer or component."""

    total_params: int = 0
    changed_count: int = 0
    abs_delta_sum: float = 0.0
    max_abs_delta: float = 0.0


@dataclass
class ParamChangeMetrics:
    """Aggregated parameter change statistics after an optimizer step."""

    # Global statistics
    total_params: int = 0
    changed_count: int = 0
    sparsity_ratio: float = 0.0  # unchanged / total (1.0 = no changes)
    mean_abs_delta: float = 0.0
    max_abs_delta: float = 0.0
    mean_relative_delta: float = 0.0

    # Sign flip tracking
    sign_flip_count: int = 0
    sign_flip_ratio: float = 0.0

    # Multi-threshold sparsity
    sparsity_by_threshold: dict[float, float] = field(default_factory=dict)

    # Per-layer breakdown (layer index -> stats)
    per_layer_sparsity: dict[int, float] = field(default_factory=dict)
    per_layer_mean_delta: dict[int, float] = field(default_factory=dict)

    # Per-component breakdown (component name -> stats)
    per_component_sparsity: dict[str, float] = field(default_factory=dict)
    per_component_mean_delta: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self, prefix: str = "param_change") -> dict[str, float]:
        """Convert metrics to flat dictionary for logging.

        Args:
            prefix: Prefix for all metric keys

        Returns:
            Flat dictionary suitable for W&B logging
        """
        result: dict[str, float] = {
            f"{prefix}/total_params": float(self.total_params),
            f"{prefix}/changed_count": float(self.changed_count),
            f"{prefix}/sparsity_ratio": self.sparsity_ratio,
            f"{prefix}/mean_abs_delta": self.mean_abs_delta,
            f"{prefix}/max_abs_delta": self.max_abs_delta,
            f"{prefix}/mean_relative_delta": self.mean_relative_delta,
            f"{prefix}/sign_flip_count": float(self.sign_flip_count),
            f"{prefix}/sign_flip_ratio": self.sign_flip_ratio,
        }

        # Multi-threshold sparsity
        for thresh, sparsity in self.sparsity_by_threshold.items():
            # Format threshold as exponent for readability (1e-7 -> "1e-07")
            thresh_str = f"{thresh:.0e}"
            result[f"{prefix}/sparsity_at_{thresh_str}"] = sparsity
            # Complement to sparsity_at_* (which is unchanged/total)
            result[f"{prefix}/changed_ratio_at_{thresh_str}"] = 1.0 - sparsity

        # Per-layer metrics
        for layer_idx, sparsity in self.per_layer_sparsity.items():
            result[f"{prefix}/layer_{layer_idx}/sparsity"] = sparsity
        for layer_idx, mean_delta in self.per_layer_mean_delta.items():
            result[f"{prefix}/layer_{layer_idx}/mean_delta"] = mean_delta

        # Per-component metrics
        for comp_name, sparsity in self.per_component_sparsity.items():
            result[f"{prefix}/component/{comp_name}/sparsity"] = sparsity
        for comp_name, mean_delta in self.per_component_mean_delta.items():
            result[f"{prefix}/component/{comp_name}/mean_delta"] = mean_delta

        return result


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


class ParamChangeTracker:
    """Tracks parameter changes across optimizer steps.

    Measures the cumulative change over k steps (step t vs step t-k),
    not consecutive steps.

    Memory-safe design:
    - Snapshots stored on CPU to preserve GPU VRAM
    - Streaming computation avoids large intermediate tensors
    - Snapshot persists across k steps, then updated after measurement

    Usage:
        tracker = ParamChangeTracker(measure_interval=4)  # k=4

        optimizer.step()
        step_count += 1

        # Every k steps: measure diff from k steps ago, then update snapshot
        if step_count % tracker.measure_interval == 0:
            if tracker.has_snapshot():
                metrics = tracker.compute_metrics(model)  # diff: step t vs step t-k
            tracker.clear_snapshot()
            tracker.capture_snapshot(model)  # snapshot for next measurement
    """

    def __init__(
        self,
        measure_interval: int = 4,
        threshold: float = 1e-7,
        track_per_layer: bool = True,
        track_components: bool = True,
        track_sign_flips: bool = True,
        relative_eps: float = 1e-10,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    ) -> None:
        """Initialize the tracker.

        Args:
            measure_interval: Measure every N optimizer steps (0 disables)
            threshold: Primary threshold for "changed" classification
            track_per_layer: Enable per-layer breakdown
            track_components: Enable per-component breakdown
            track_sign_flips: Track sign changes
            relative_eps: Epsilon for relative delta computation
            thresholds: Tuple of thresholds for multi-threshold analysis
        """
        self.measure_interval = measure_interval
        self.threshold = threshold
        self.track_per_layer = track_per_layer
        self.track_components = track_components
        self.track_sign_flips = track_sign_flips
        self.relative_eps = relative_eps
        # Ensure the primary threshold is included in the multi-threshold list
        combined_thresholds = list(thresholds)
        if threshold not in combined_thresholds:
            combined_thresholds.append(threshold)
        self.thresholds = tuple(sorted(set(combined_thresholds)))

        # CPU snapshot storage (None when not capturing)
        self._snapshot: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_config(cls, config: Any) -> ParamChangeTracker:
        """Create tracker from TrainingConfig.

        Args:
            config: TrainingConfig instance

        Returns:
            Configured ParamChangeTracker
        """
        return cls(
            measure_interval=getattr(config, "param_change_measure_interval", 4),
            threshold=getattr(config, "param_change_threshold", 1e-7),
            track_per_layer=getattr(config, "param_change_track_per_layer", True),
            track_components=getattr(config, "param_change_track_components", True),
            track_sign_flips=getattr(config, "param_change_track_sign_flips", True),
            relative_eps=getattr(config, "param_change_relative_eps", 1e-10),
        )

    def is_enabled(self) -> bool:
        """Check if tracking is enabled.

        Returns:
            True if measure_interval > 0
        """
        return self.measure_interval > 0

    def should_measure(self, optimizer_step_count: int) -> bool:
        """Check if we should measure at this step.

        Args:
            optimizer_step_count: Current optimizer step count (1-indexed after step)

        Returns:
            True if this is a measurement step (step is divisible by interval)
        """
        if not self.is_enabled():
            return False
        return optimizer_step_count % self.measure_interval == 0

    def has_snapshot(self) -> bool:
        """Check if a snapshot is currently held.

        Returns:
            True if snapshot exists
        """
        return self._snapshot is not None

    def get_snapshot(self) -> dict[str, torch.Tensor]:
        """Get the current parameter snapshot.

        Returns:
            Dictionary mapping parameter names to CPU tensors

        Raises:
            RuntimeError: If no snapshot exists
        """
        if self._snapshot is None:
            raise RuntimeError("No snapshot captured. Call capture_snapshot() first.")
        return self._snapshot

    def capture_snapshot(self, model: nn.Module) -> None:
        """Capture parameter snapshot to CPU memory.

        Args:
            model: Model to snapshot
        """
        if self._snapshot is not None:
            logger.warning("Overwriting existing snapshot without computing metrics")
            self.clear_snapshot()

        self._snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # CRITICAL: Clone FIRST on original device, THEN move to CPU
                # This ensures we get a true copy before any device transfer
                # Using .clone().detach() instead of .detach().clone() to be safe
                self._snapshot[name] = param.data.clone().detach().cpu()

        logger.debug(
            "Captured parameter snapshot: %d parameters",
            len(self._snapshot),
        )

    def clear_snapshot(self) -> None:
        """Clear the snapshot and free memory."""
        if self._snapshot is not None:
            # Explicit deletion for memory cleanup
            for tensor in self._snapshot.values():
                del tensor
            self._snapshot.clear()
            self._snapshot = None
            gc.collect()

    def compute_metrics(self, model: nn.Module) -> ParamChangeMetrics:
        """Compute parameter change metrics via streaming comparison.

        Memory-safe: processes one parameter at a time, never materializes
        full model-sized tensors beyond the snapshot.

        Args:
            model: Model after optimizer step

        Returns:
            Computed metrics

        Raises:
            RuntimeError: If no snapshot was captured
        """
        if self._snapshot is None:
            raise RuntimeError("No snapshot captured. Call capture_snapshot() first.")

        # Running accumulators
        total_params = 0
        changed_counts: dict[float, int] = dict.fromkeys(self.thresholds, 0)
        abs_delta_sum = 0.0
        max_abs_delta = 0.0
        relative_delta_sum = 0.0
        sign_flip_count = 0

        # Per-layer and per-component accumulators
        layer_stats: dict[int, LayerStats] = defaultdict(LayerStats)
        component_stats: dict[str, LayerStats] = defaultdict(LayerStats)

        # Diagnostic: track first param for debugging
        first_param_logged = False

        for name, param in model.named_parameters():
            if name not in self._snapshot:
                continue

            # CRITICAL: Convert to float32 BEFORE subtraction for precision
            # bfloat16 cannot represent small deltas (1e-10, 1e-12)
            current = param.data.cpu().float()
            before = self._snapshot[name].float()

            # Compute absolute delta in float32
            delta = (current - before).abs()
            n = delta.numel()
            total_params += n

            # Diagnostic logging for first parameter
            if not first_param_logged:
                delta_max_first = float(delta.max().item())
                delta_mean_first = float(delta.mean().item())
                is_same_object = current.data_ptr() == before.data_ptr()
                logger.debug(
                    "Param change diagnostic [%s]: delta_max=%.2e, delta_mean=%.2e, "
                    "same_ptr=%s, current_device=%s, before_device=%s, dtype=%s",
                    name[:50],
                    delta_max_first,
                    delta_mean_first,
                    is_same_object,
                    current.device,
                    before.device,
                    current.dtype,
                )
                first_param_logged = True

            # Multi-threshold counting
            for thresh in self.thresholds:
                changed_counts[thresh] += int((delta > thresh).sum().item())

            # Aggregate statistics
            delta_sum = float(delta.sum().item())
            delta_max = float(delta.max().item())
            abs_delta_sum += delta_sum
            max_abs_delta = max(max_abs_delta, delta_max)

            # Relative delta: |delta| / (|before| + eps)
            relative = delta / (before.abs() + self.relative_eps)
            relative_delta_sum += float(relative.sum().item())

            # Sign flips: count where sign changed (product < 0)
            if self.track_sign_flips:
                # Only count where both values are non-zero
                sign_changed = ((current * before) < 0).sum().item()
                sign_flip_count += int(sign_changed)

            # Per-layer aggregation
            if self.track_per_layer:
                layer_idx = _extract_layer_index(name)
                if layer_idx is not None:
                    stats = layer_stats[layer_idx]
                    stats.total_params += n
                    stats.changed_count += int((delta > self.threshold).sum().item())
                    stats.abs_delta_sum += delta_sum
                    stats.max_abs_delta = max(stats.max_abs_delta, delta_max)

            # Per-component aggregation
            if self.track_components:
                component = _extract_component(name)
                if component is not None:
                    stats = component_stats[component]
                    stats.total_params += n
                    stats.changed_count += int((delta > self.threshold).sum().item())
                    stats.abs_delta_sum += delta_sum
                    stats.max_abs_delta = max(stats.max_abs_delta, delta_max)

            # Immediately free temporary tensors
            del current, delta, relative

        # Diagnostic: log observed max delta to help debug threshold issues
        if max_abs_delta > 0:
            logger.debug(
                "Param change global stats: max_delta=%.2e, mean_delta=%.2e, "
                "total_params=%d, threshold=%.0e",
                max_abs_delta,
                abs_delta_sum / total_params if total_params > 0 else 0.0,
                total_params,
                self.threshold,
            )
        else:
            logger.warning(
                "Param change: max_delta=0! This suggests snapshot and current "
                "params are identical. Check if optimizer.step() is being called."
            )

        # Compute final metrics
        changed_at_primary = changed_counts.get(self.threshold, 0)
        unchanged = total_params - changed_at_primary

        metrics = ParamChangeMetrics(
            total_params=total_params,
            changed_count=changed_at_primary,
            sparsity_ratio=unchanged / total_params if total_params > 0 else 1.0,
            mean_abs_delta=abs_delta_sum / total_params if total_params > 0 else 0.0,
            max_abs_delta=max_abs_delta,
            mean_relative_delta=(relative_delta_sum / total_params if total_params > 0 else 0.0),
            sign_flip_count=sign_flip_count,
            sign_flip_ratio=(sign_flip_count / total_params if total_params > 0 else 0.0),
        )

        # Multi-threshold sparsity
        for thresh, changed in changed_counts.items():
            unchanged_at_thresh = total_params - changed
            metrics.sparsity_by_threshold[thresh] = (
                unchanged_at_thresh / total_params if total_params > 0 else 1.0
            )

        # Per-layer breakdown
        if self.track_per_layer:
            for layer_idx, stats in sorted(layer_stats.items()):
                if stats.total_params > 0:
                    unchanged_layer = stats.total_params - stats.changed_count
                    metrics.per_layer_sparsity[layer_idx] = unchanged_layer / stats.total_params
                    metrics.per_layer_mean_delta[layer_idx] = (
                        stats.abs_delta_sum / stats.total_params
                    )

        # Per-component breakdown
        if self.track_components:
            for comp_name, stats in sorted(component_stats.items()):
                if stats.total_params > 0:
                    unchanged_comp = stats.total_params - stats.changed_count
                    metrics.per_component_sparsity[comp_name] = unchanged_comp / stats.total_params
                    metrics.per_component_mean_delta[comp_name] = (
                        stats.abs_delta_sum / stats.total_params
                    )

        return metrics


async def log_param_change_metrics(
    metrics: ParamChangeMetrics,
    monitor: Any,
    step: int,
) -> None:
    """Log parameter change metrics to monitoring system.

    Args:
        metrics: Computed metrics
        monitor: Monitoring system instance (e.g., WandB)
        step: Optimizer step count for x-axis
    """
    if monitor is None:
        return

    log_dict = metrics.to_log_dict(prefix="param_change")

    for key, value in log_dict.items():
        try:
            await monitor.log_gauge(
                key,
                value,
                tags={"optimizer_step": str(step)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log param change metric %s: %s", key, exc)

    # Log summary to console
    logger.info(
        "Parameter changes at step %d: sparsity=%.4f, mean_delta=%.2e, "
        "max_delta=%.2e, sign_flips=%d (%.4f%%)",
        step,
        metrics.sparsity_ratio,
        metrics.mean_abs_delta,
        metrics.max_abs_delta,
        metrics.sign_flip_count,
        metrics.sign_flip_ratio * 100,
    )
