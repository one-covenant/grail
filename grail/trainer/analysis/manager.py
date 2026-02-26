"""Model analysis manager - main orchestration layer.

Provides the high-level API for model analysis during training.
Manages snapshot lifecycle and coordinates metric computation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from grail.trainer.analysis.config import AnalysisConfig
from grail.trainer.analysis.metrics.adam_sign_descent import AdamSignDescentMetrics
from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.metrics.parameter_change import ParameterChangeMetrics
from grail.trainer.analysis.metrics.sparse_quality import SparseQualityMetrics
from grail.trainer.analysis.primitives import ParameterSnapshot

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


class ModelAnalysisManager:
    """Main entry point for model analysis during training.

    Manages snapshot lifecycle and coordinates metric computation across
    multiple metric computers. Call `on_optimizer_step` after each optimizer
    step; metrics are computed at regular intervals.

    Design:
    - Single responsibility: Orchestration only (no metric computation)
    - Builder pattern: Use add_metric() to configure
    - Factory method: Use create() for common configurations
    - Fail-safe: Errors in individual metrics don't break training

    Example:
        >>> config = AnalysisConfig(interval=100)
        >>> manager = ModelAnalysisManager.create(config)
        >>>
        >>> # In training loop
        >>> for batch in dataloader:
        ...     loss.backward()
        ...     optimizer.step()
        ...     metrics = manager.on_optimizer_step(model, inputs=batch)
        ...     if metrics:
        ...         wandb.log(metrics)
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the analysis manager.

        Args:
            config: Analysis configuration
        """
        self.config = config
        self.step_count = 0
        self.old_snapshot: ParameterSnapshot | None = None
        self.metric_computers: list[MetricComputer] = []

        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self._snapshot_dtype = dtype_map.get(config.snapshot_dtype, torch.float32)

    def add_metric(self, computer: MetricComputer) -> ModelAnalysisManager:
        """Add a metric computer (builder pattern).

        Args:
            computer: Metric computer to add

        Returns:
            Self for chaining

        Example:
            >>> manager = (ModelAnalysisManager(config)
            ...     .add_metric(ParameterChangeMetrics())
            ...     .add_metric(CustomMetrics())
            ... )
        """
        self.metric_computers.append(computer)
        return self

    def on_optimizer_step(
        self,
        model: nn.Module,
        inputs: dict | None = None,
        optimizer: Any | None = None,
        **metadata: Any,
    ) -> dict[str, float]:
        """Call this after optimizer.step() to trigger analysis.

        Metrics are computed every N steps (configured by config.interval).
        Between measurement points, this returns an empty dict.

        Args:
            model: Model being trained
            inputs: Optional batch inputs (required for sparse quality analysis)
            optimizer: Optional optimizer (for future momentum analysis)
            **metadata: Additional metadata to pass to metric computers

        Returns:
            Dictionary of metrics (empty if not a measurement step)

        Example:
            >>> optimizer.step()
            >>> metrics = manager.on_optimizer_step(model, inputs=batch)
            >>> if metrics:
            ...     logger.info(f"Step {manager.step_count}: {metrics}")
        """
        self.step_count += 1

        # Only measure at specified intervals
        if self.step_count % self.config.interval != 0:
            return {}

        # Capture current snapshot
        current_snapshot = ParameterSnapshot(
            model,
            device=self.config.snapshot_device,
            dtype=self._snapshot_dtype,
        )

        # First measurement: just capture snapshot, no metrics
        if self.old_snapshot is None:
            self.old_snapshot = current_snapshot
            logger.debug(
                "Analysis: First snapshot captured at step %d. "
                "Metrics will be available at step %d.",
                self.step_count,
                self.step_count + self.config.interval,
            )
            return {}

        # Compute delta
        try:
            delta = self.old_snapshot.compute_delta(current_snapshot)
        except ValueError as e:
            logger.warning("Failed to compute delta: %s. Skipping analysis.", e)
            self.old_snapshot = current_snapshot
            return {}

        # Build context
        context = AnalysisContext(
            model=model,
            inputs=inputs,
            attention_mask=inputs.get("attention_mask") if inputs else None,
            optimizer=optimizer,
            step=self.step_count,
            metadata=metadata,
        )

        # Run all metric computers
        metrics: dict[str, float] = {}

        for computer in self.metric_computers:
            try:
                result = computer.compute(
                    delta=delta,
                    old_snapshot=self.old_snapshot,
                    current_snapshot=current_snapshot,
                    context=context,
                )
                metrics.update(result)
            except Exception as e:
                logger.warning(
                    "Metric computation failed for %s: %s. Skipping this metric.",
                    computer.name,
                    e,
                    exc_info=True,
                )

        # Update snapshot for next measurement
        self.old_snapshot = current_snapshot

        logger.debug(
            "Analysis: Computed %d metrics at step %d (interval: %d)",
            len(metrics),
            self.step_count,
            self.config.interval,
        )

        return metrics

    def reset(self) -> None:
        """Reset internal state (clear snapshots and counters).

        Useful when starting a new training phase or after checkpoint load.
        """
        self.step_count = 0
        self.old_snapshot = None
        logger.info("Analysis manager reset")

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dictionary with step count and snapshot timestamp
        """
        return {
            "step_count": self.step_count,
            "snapshot_timestamp": self.old_snapshot.timestamp if self.old_snapshot else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint.

        Note: Snapshots are NOT saved (too large). After loading, the first
        measurement will just capture a new snapshot.

        Args:
            state: State dictionary from state_dict()
        """
        self.step_count = state.get("step_count", 0)
        self.old_snapshot = None  # Intentionally not restored
        logger.info("Analysis manager state loaded (step_count=%d)", self.step_count)

    @classmethod
    def create(cls, config: AnalysisConfig) -> ModelAnalysisManager:
        """Factory method: Create manager with standard metric computers.

        Automatically adds metric computers based on config flags:
        - ParameterChangeMetrics if param_change_enabled
        - SparseQualityMetrics if sparse_quality_enabled

        Args:
            config: Analysis configuration

        Returns:
            Configured ModelAnalysisManager

        Example:
            >>> config = AnalysisConfig(
            ...     param_change_enabled=True,
            ...     sparse_quality_enabled=True,
            ... )
            >>> manager = ModelAnalysisManager.create(config)
        """
        manager = cls(config)

        # Add parameter change metrics
        if config.param_change_enabled:
            param_change = ParameterChangeMetrics(
                thresholds=config.param_change_thresholds,
                track_per_layer=config.param_change_per_layer,
                track_components=config.param_change_track_components,
            )
            manager.add_metric(param_change)
            logger.info("Added ParameterChangeMetrics to analysis manager")

        # Add sparse quality metrics
        if config.sparse_quality_enabled:
            sparse_quality = SparseQualityMetrics(
                thresholds=config.sparse_quality_thresholds,
                include_random_baseline=config.sparse_quality_include_random,
            )
            manager.add_metric(sparse_quality)
            logger.info("Added SparseQualityMetrics to analysis manager")

        # Future: Add gradient metrics
        if config.gradient_enabled:
            logger.warning("Gradient analysis not yet implemented")

        # Future: Add momentum metrics
        if config.momentum_enabled:
            logger.warning("Momentum analysis not yet implemented")

        # Add Adam sign descent metrics
        if config.adam_sign_enabled:
            adam_sign = AdamSignDescentMetrics(
                track_per_component=config.adam_sign_track_per_component,
                track_per_layer=config.adam_sign_track_per_layer,
                histogram_samples=config.adam_sign_histogram_samples,
                near_lr_tolerance=config.adam_sign_near_lr_tolerance,
            )
            manager.add_metric(adam_sign)
            logger.info("Added AdamSignDescentMetrics to analysis manager")

        return manager

    def __len__(self) -> int:
        """Number of metric computers registered."""
        return len(self.metric_computers)

    def __repr__(self) -> str:
        """String representation."""
        computers = ", ".join(c.name for c in self.metric_computers)
        return (
            f"ModelAnalysisManager("
            f"step={self.step_count}, "
            f"interval={self.config.interval}, "
            f"computers=[{computers}]"
            f")"
        )
