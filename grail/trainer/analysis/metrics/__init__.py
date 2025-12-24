"""Metric computers for model analysis.

This package contains stateless metric computers that analyze parameter changes,
sparse update quality, and other training dynamics.
"""

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.metrics.parameter_change import ParameterChangeMetrics
from grail.trainer.analysis.metrics.sparse_quality import SparseQualityMetrics

__all__ = [
    # Base classes
    "MetricComputer",
    "AnalysisContext",
    # Concrete metrics
    "ParameterChangeMetrics",
    "SparseQualityMetrics",
]
