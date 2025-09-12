"""
Null monitoring backend implementation.

This module provides a no-op implementation of the MonitoringBackend interface
for scenarios where monitoring is disabled or unavailable. It ensures the
application continues to function normally without any monitoring overhead.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ..base import MetricData, MonitoringBackend


class NullBackend(MonitoringBackend):
    """No-op monitoring backend for when monitoring is disabled.

    This backend implements all MonitoringBackend methods but performs no actual
    operations. It's used as a fallback when monitoring systems are unavailable
    or explicitly disabled, ensuring the application continues to work normally.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the null backend (no-op).

        Args:
            config: Configuration dictionary (ignored)
        """
        pass

    async def log_metric(self, metric: MetricData) -> None:
        """Log a single metric (no-op).

        Args:
            metric: The metric data (ignored)
        """
        pass

    async def log_metrics(self, metrics: list[MetricData]) -> None:
        """Log multiple metrics (no-op).

        Args:
            metrics: List of metrics (ignored)
        """
        pass

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Context manager for timing operations (no-op).

        Args:
            name: Name of the timer metric (ignored)
            tags: Optional tags (ignored)

        Yields:
            None
        """
        yield

    async def log_artifact(self, name: str, data: Any, artifact_type: str) -> None:
        """Log artifacts (no-op).

        Args:
            name: Name/identifier for the artifact (ignored)
            data: The artifact data (ignored)
            artifact_type: Type of artifact (ignored)
        """
        pass

    async def start_run(self, run_name: str, config: dict[str, Any]) -> str:
        """Start a new monitoring run (no-op).

        Args:
            run_name: Name for this run (ignored)
            config: Configuration and metadata (ignored)

        Returns:
            A placeholder run ID
        """
        return "null_run"

    async def finish_run(self, run_id: str) -> None:
        """Finish a monitoring run (no-op).

        Args:
            run_id: The run identifier (ignored)
        """
        pass

    async def health_check(self) -> bool:
        """Check backend health (always returns True).

        Returns:
            Always True since null backend is always "healthy"
        """
        return True

    async def shutdown(self) -> None:
        """Shutdown the backend (no-op)."""
        pass
