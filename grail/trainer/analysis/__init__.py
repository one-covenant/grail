"""Model analysis framework for training diagnostics.

This package provides a reusable, extensible framework for analyzing model
behavior during training. It measures parameter changes, sparse update quality,
and other training dynamics.

Key Components:
    - Primitives: ParameterSnapshot, ParameterDelta (immutable, type-safe)
    - Metrics: Pluggable metric computers (stateless, composable)
    - Manager: Orchestration and lifecycle management
    - Config: Centralized configuration

Quick Start:
    >>> from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig
    >>>
    >>> # Setup
    >>> config = AnalysisConfig(interval=100)
    >>> analyzer = ModelAnalysisManager.create(config)
    >>>
    >>> # Training loop
    >>> for batch in dataloader:
    ...     loss.backward()
    ...     optimizer.step()
    ...     metrics = analyzer.on_optimizer_step(model, inputs=batch)
    ...     if metrics:
    ...         wandb.log(metrics)

Custom Metrics:
    >>> from grail.trainer.analysis import MetricComputer
    >>>
    >>> class MyMetric(MetricComputer):
    ...     def compute(self, delta, **kwargs):
    ...         return {"my_metric/value": delta.statistics()["norm_l2"]}
    >>>
    >>> analyzer = ModelAnalysisManager(config).add_metric(MyMetric())
"""

# Public API exports
from grail.trainer.analysis.config import AnalysisConfig
from grail.trainer.analysis.manager import ModelAnalysisManager
from grail.trainer.analysis.metrics import (
    AdamSignDescentMetrics,
    AnalysisContext,
    GradientSparsityMetrics,
    MetricComputer,
    ParameterChangeMetrics,
    SparseQualityMetrics,
)
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot

__all__ = [
    # Main API
    "ModelAnalysisManager",
    "AnalysisConfig",
    # Primitives (for advanced usage)
    "ParameterSnapshot",
    "ParameterDelta",
    # Metrics (for extension)
    "MetricComputer",
    "AnalysisContext",
    "ParameterChangeMetrics",
    "SparseQualityMetrics",
    "GradientSparsityMetrics",
    "AdamSignDescentMetrics",
]

__version__ = "1.0.0"
