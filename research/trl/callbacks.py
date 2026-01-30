#!/usr/bin/env python3
"""Shared training callbacks for TRL scripts (GRPO, SFT, etc.).

This module provides reusable callbacks and utilities:
- Profiling infrastructure (TimingStats, Profiler)
- SparsityCallback for parameter change and gradient tracking
- Re-exported DeltaCheckpointCallback

Usage:
    from callbacks import (
        Profiler, get_profiler, TimingStats,
        SparsityCallback,
        DeltaCheckpointCallback,
    )
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# PROFILING UTILITIES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class TimingStats:
    """Accumulates timing statistics for a named operation."""

    name: str
    total_seconds: float = 0.0
    call_count: int = 0
    min_seconds: float = float("inf")
    max_seconds: float = 0.0

    def record(self, elapsed: float) -> None:
        """Record a timing measurement."""
        self.total_seconds += elapsed
        self.call_count += 1
        self.min_seconds = min(self.min_seconds, elapsed)
        self.max_seconds = max(self.max_seconds, elapsed)

    @property
    def mean_seconds(self) -> float:
        """Average time per call."""
        return self.total_seconds / self.call_count if self.call_count > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.call_count == 0:
            return f"{self.name}: never called"
        if self.call_count == 1:
            return f"{self.name}: {self.total_seconds:.2f}s"
        return (
            f"{self.name}: {self.total_seconds:.2f}s total, "
            f"{self.call_count} calls, "
            f"{self.mean_seconds:.2f}s avg, "
            f"{self.min_seconds:.2f}s min, "
            f"{self.max_seconds:.2f}s max"
        )


class Profiler:
    """Simple profiler for tracking time spent in major operations.

    Usage:
        profiler = Profiler()

        with profiler.track("model_loading"):
            model = load_model()

        with profiler.track("training_step"):
            trainer.train()

        profiler.print_summary()
        profiler.log_to_wandb()
    """

    def __init__(self) -> None:
        self._stats: dict[str, TimingStats] = {}
        self._active_timers: dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        """Context manager to track time for a named operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self._stats:
                self._stats[name] = TimingStats(name=name)
            self._stats[name].record(elapsed)

    def start(self, name: str) -> None:
        """Start a timer (for non-context-manager usage)."""
        self._active_timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop a timer and record the elapsed time."""
        if name not in self._active_timers:
            return 0.0
        elapsed = time.perf_counter() - self._active_timers.pop(name)
        if name not in self._stats:
            self._stats[name] = TimingStats(name=name)
        self._stats[name].record(elapsed)
        return elapsed

    def get_stats(self, name: str) -> TimingStats | None:
        """Get stats for a specific operation."""
        return self._stats.get(name)

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get all timing stats."""
        return self._stats.copy()

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for logging (prefixed with 'profiler/')."""
        result = {}
        for name, stats in self._stats.items():
            prefix = f"profiler/{name}"
            result[f"{prefix}/total_seconds"] = stats.total_seconds
            result[f"{prefix}/call_count"] = float(stats.call_count)
            result[f"{prefix}/mean_seconds"] = stats.mean_seconds
            if stats.call_count > 1:
                result[f"{prefix}/min_seconds"] = stats.min_seconds
                result[f"{prefix}/max_seconds"] = stats.max_seconds
        return result

    def print_summary(self) -> None:
        """Print timing summary to console."""
        if not self._stats:
            logger.info("No profiling data collected")
            return

        logger.info("\n" + "=" * 70)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 70)

        # Sort by total time descending
        sorted_stats = sorted(
            self._stats.values(), key=lambda s: s.total_seconds, reverse=True
        )
        for stats in sorted_stats:
            logger.info(f"  {stats.summary()}")

        total_tracked = sum(s.total_seconds for s in self._stats.values())
        logger.info("-" * 70)
        logger.info(f"  Total tracked time: {total_tracked:.2f}s")
        logger.info("=" * 70 + "\n")

    def log_to_wandb(self) -> None:
        """Log timing stats to WandB."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(self.to_dict())
        except Exception:
            pass


# Global profiler instance
_profiler: Profiler | None = None


def get_profiler() -> Profiler:
    """Get the global profiler instance (creates one if needed)."""
    global _profiler
    if _profiler is None:
        _profiler = Profiler()
    return _profiler


# ════════════════════════════════════════════════════════════════════════════
# SPARSITY ANALYSIS CALLBACK
# ════════════════════════════════════════════════════════════════════════════


class SparsityCallback(TrainerCallback):
    """Minimal callback for parameter change ratio and gradient tracking.

    Uses on_optimizer_step which is called after optimizer.step() but BEFORE
    zero_grad(), so gradients are still available.

    Logs to WandB:
    - sparsity/param_change/* - Parameter change statistics
    - sparsity/param_change/gradient_norm - Total gradient norm
    - sparsity/gradient/* - Per-layer gradient histograms
    """

    # Prefix for histogram data from GradientSparsityMetrics
    HISTOGRAM_KEY_PREFIX = "gradient/_histogram/"

    def __init__(self, analyzer: Any):
        """Initialize the callback.

        Args:
            analyzer: ModelAnalysisManager instance from grail.trainer.analysis
        """
        self.analyzer = analyzer
        self._wandb_configured = False

    def on_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Called after optimizer.step() but before zero_grad() - gradients available."""
        model = kwargs.get("model")
        if model is None:
            return

        # Avoid duplicate WandB logs in distributed runs.
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not is_world_process_zero:
            return

        optimizer = kwargs.get("optimizer")
        inputs = kwargs.get("inputs")

        # Run analysis manager (computes metrics only at configured interval).
        profiler = get_profiler()
        try:
            with profiler.track("sparsity_analysis"):
                analysis_metrics = self.analyzer.on_optimizer_step(
                    model=model,
                    inputs=inputs,
                    optimizer=optimizer,
                )
        except Exception:
            return

        # Log only at analysis measurement steps to match AnalysisConfig.interval.
        is_measurement_step = (self.analyzer.step_count % self.analyzer.config.interval) == 0
        if not (analysis_metrics or is_measurement_step):
            return

        # Separate scalar metrics from histogram data
        scalar_metrics: dict[str, float] = {}
        histogram_data: dict[str, Any] = {}

        for key, value in analysis_metrics.items():
            if key.startswith(self.HISTOGRAM_KEY_PREFIX):
                # Strip prefix for cleaner wandb key
                hist_name = key[len(self.HISTOGRAM_KEY_PREFIX) :]
                histogram_data[hist_name] = value
            elif isinstance(value, (int, float)):
                scalar_metrics[key] = float(value)

        # Capture gradient statistics (available here, before zero_grad) but only at measurement steps.
        if is_measurement_step:
            grad_norm = self._compute_gradient_norm(model)
            if grad_norm is not None:
                scalar_metrics["param_change/gradient_norm"] = grad_norm

        if not scalar_metrics and not histogram_data:
            return

        # Log to WandB with custom x-axis (optimizer_step).
        try:
            import wandb

            if wandb.run:
                # Configure custom x-axis on first call
                if not self._wandb_configured:
                    wandb.define_metric("optimizer_step")
                    wandb.define_metric("sparsity/*", step_metric="optimizer_step")
                    self._wandb_configured = True

                # Log with our own step counter
                optimizer_step = self.analyzer.step_count
                wandb_data: dict[str, Any] = {"optimizer_step": optimizer_step}

                # Add scalar metrics with sparsity/ prefix
                for k, v in scalar_metrics.items():
                    wandb_data[f"sparsity/{k}"] = v

                # Add histograms with sparsity/gradient/ prefix
                for hist_name, hist_values in histogram_data.items():
                    wandb_data[f"sparsity/gradient/{hist_name}"] = wandb.Histogram(hist_values)

                wandb.log(wandb_data)
        except Exception:
            pass

    def _compute_gradient_norm(self, model: Any) -> float | None:
        """Compute total gradient norm across all parameters."""
        total_norm_sq = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.norm(2).item() ** 2
                count += 1
        return (total_norm_sq**0.5) if count > 0 else None


# ════════════════════════════════════════════════════════════════════════════
# RE-EXPORT DeltaCheckpointCallback
# ════════════════════════════════════════════════════════════════════════════
from delta_checkpoint_callback import DeltaCheckpointCallback  # noqa: E402, F401

__all__ = [
    "TimingStats",
    "Profiler",
    "get_profiler",
    "SparsityCallback",
    "DeltaCheckpointCallback",
]
