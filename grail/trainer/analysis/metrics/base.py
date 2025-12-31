"""Base classes and interfaces for metric computation.

This module defines the core abstractions for implementing metric computers.
All metric computers are stateless and implement the MetricComputer interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot


@dataclass
class AnalysisContext:
    """Context information passed to metric computers.

    Contains optional inputs and metadata that some metric computers may need.
    This design allows new context fields to be added without breaking existing
    metric computers (they can ignore fields they don't need).

    Attributes:
        model: Optional model reference (for forward passes)
        inputs: Optional batch inputs (for sparse quality analysis)
        attention_mask: Optional attention mask from inputs
        optimizer: Optional optimizer reference (for momentum analysis)
        step: Current optimizer step number
        metadata: Additional custom metadata
    """

    model: nn.Module | None = None
    inputs: dict[str, Any] | None = None
    attention_mask: Any | None = None
    optimizer: Any | None = None
    step: int = 0
    metadata: dict[str, Any] | None = None


class MetricComputer(ABC):
    """Abstract base class for stateless metric computation.

    All metric computers must implement the `compute` method which takes
    analysis primitives and context, and returns a dictionary of metrics.

    Design Principles:
        - Stateless: No internal state; all inputs via method parameters
        - Pure functions: Same inputs always produce same outputs
        - Fail-safe: Return empty dict on errors rather than raising
        - Prefixed keys: All metric keys should be prefixed for namespace isolation

    Example:
        >>> class MyMetric(MetricComputer):
        ...     def compute(self, delta, **kwargs):
        ...         return {"my_metric/value": delta.statistics()["norm_l2"]}
        ...
        >>> computer = MyMetric()
        >>> metrics = computer.compute(delta=parameter_delta)
    """

    @abstractmethod
    def compute(
        self,
        delta: ParameterDelta | None = None,
        old_snapshot: ParameterSnapshot | None = None,
        current_snapshot: ParameterSnapshot | None = None,
        context: AnalysisContext | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from provided inputs.

        Args:
            delta: Parameter delta (W_current - W_old), if available
            old_snapshot: Snapshot from previous measurement point
            current_snapshot: Snapshot from current measurement point
            context: Additional context (model, inputs, etc.)
            **kwargs: Additional arguments for extensibility

        Returns:
            Dictionary mapping metric names to float values.
            Keys should be prefixed with metric type (e.g., "param_change/norm_l2")

        Note:
            Not all arguments will be needed by all metric computers.
            Implementations should gracefully handle missing arguments.
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for this metric computer.

        Returns:
            Name derived from class name by default
        """
        return self.__class__.__name__

    def requires_model(self) -> bool:
        """Whether this metric computer requires model in context.

        Returns:
            True if model is required, False otherwise
        """
        return False

    def requires_inputs(self) -> bool:
        """Whether this metric computer requires batch inputs in context.

        Returns:
            True if inputs are required, False otherwise
        """
        return False

    def requires_optimizer(self) -> bool:
        """Whether this metric computer requires optimizer in context.

        Returns:
            True if optimizer is required, False otherwise
        """
        return False

    def validate_context(self, context: AnalysisContext | None) -> bool:
        """Validate that context has required fields.

        Args:
            context: Analysis context to validate

        Returns:
            True if context is valid for this metric computer
        """
        if context is None:
            return not (
                self.requires_model() or self.requires_inputs() or self.requires_optimizer()
            )

        if self.requires_model() and context.model is None:
            return False

        if self.requires_inputs() and context.inputs is None:
            return False

        if self.requires_optimizer() and context.optimizer is None:
            return False

        return True
