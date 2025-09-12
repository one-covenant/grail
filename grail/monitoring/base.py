"""
Abstract base classes for monitoring backends.

This module defines the core interfaces that all monitoring backends must implement,
ensuring consistent behavior and easy switching between different telemetry platforms.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Types of metrics that can be logged."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricData:
    """Container for metric data with metadata."""

    name: str
    value: Any
    metric_type: MetricType
    tags: dict[str, str] | None = None
    timestamp: float | None = None
    block_number: int | None = None
    window_number: int | None = None

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class MonitoringBackend(ABC):
    """Abstract base class for monitoring backends.

    This interface defines the contract that all monitoring backends must implement.
    It supports both synchronous and asynchronous operations, with proper error handling
    and graceful degradation for production environments.
    """

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the monitoring backend with configuration (synchronous).

        This method should:
        - Validate configuration parameters
        - Import required libraries
        - Set up internal state
        - NOT make network connections or start background tasks

        Args:
            config: Configuration dictionary containing backend-specific settings

        Raises:
            Exception: If initialization fails (should be handled gracefully by manager)
        """
        pass

    @abstractmethod
    async def log_metric(self, metric: MetricData) -> None:
        """Log a single metric.

        Args:
            metric: The metric data to log

        Note:
            Implementation should handle errors gracefully and not raise exceptions
            that would disrupt the main application flow.
        """
        pass

    @abstractmethod
    async def log_metrics(self, metrics: list[MetricData]) -> None:
        """Log multiple metrics efficiently in a batch.

        Args:
            metrics: List of metrics to log

        Note:
            This method should be more efficient than calling log_metric multiple times.
            Implementation should handle partial failures gracefully.
        """
        pass

    @abstractmethod
    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Args:
            name: Name of the timer metric
            tags: Optional tags to attach to the metric

        Yields:
            None

        Example:
            with backend.timer("operation_duration"):
                # ... timed operation ...
        """
        pass

    @abstractmethod
    async def log_artifact(self, name: str, data: Any, artifact_type: str) -> None:
        """Log artifacts like model checkpoints, plots, etc.

        Args:
            name: Name/identifier for the artifact
            data: The artifact data (file path, plot object, etc.)
            artifact_type: Type of artifact ("model", "plot", "file", etc.)
        """
        pass

    @abstractmethod
    async def start_run(self, run_name: str, config: dict[str, Any]) -> str:
        """Start a new monitoring run/session.

        Args:
            run_name: Name for this monitoring run
            config: Configuration and metadata for the run

        Returns:
            Run ID or identifier for this session
        """
        pass

    @abstractmethod
    async def finish_run(self, run_id: str) -> None:
        """Finish a monitoring run/session.

        Args:
            run_id: The run identifier returned by start_run
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and operational.

        Returns:
            True if the backend is healthy, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources.

        This method should ensure all pending metrics are flushed
        and resources are properly cleaned up.
        """
        pass
