"""
Monitoring manager and high-level interface.

This module provides the MonitoringManager class which serves as the main
interface for the monitoring system. It handles buffering, batching, and
provides a simple API for logging metrics and artifacts.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any

from .backends.null_backend import NullBackend
from .backends.wandb_backend import WandBBackend
from .base import MetricData, MetricType, MonitoringBackend

logger = logging.getLogger(__name__)


class MonitoringManager:
    """High-level interface for monitoring operations.

    The MonitoringManager provides a simplified API for logging metrics and artifacts.
    It handles buffering, batching, error recovery, and provides both sync and async
    interfaces for different use cases.
    """

    def __init__(self, backend: MonitoringBackend | None = None):
        """Initialize the monitoring manager.

        Args:
            backend: The monitoring backend to use. If None, uses NullBackend.
        """
        self.backend = backend or NullBackend()
        self._metric_buffer: list[MetricData] = []
        self._buffer_size = 100
        self._flush_interval = 30.0  # seconds
        self._flush_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._initialized = False
        self._config: dict[str, Any] = {}
        self._current_block: int | None = None
        self._current_window: int | None = None
        self._start_time: float | None = None

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the monitoring system synchronously.

        Args:
            config: Configuration dictionary for the backend
        """
        try:
            # Update configuration
            self._buffer_size = config.get("buffer_size", 100)
            self._flush_interval = config.get("flush_interval", 30.0)

            # Initialize backend synchronously
            self.backend.initialize(config)
            self._config = config.copy()
            self._initialized = True

            logger.info("Monitoring manager initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize monitoring manager: {e}")
            # Fall back to null backend
            self.backend = NullBackend()
            self.backend.initialize({})
            self._initialized = True

    def _ensure_async_components(self) -> None:
        """Ensure async components are initialized (lazy initialization)."""
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._periodic_flush())

    def _start_flush_task(self) -> None:
        """Start the periodic metric flushing task."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._periodic_flush())

    async def _periodic_flush(self) -> None:
        """Periodically flush buffered metrics."""
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self._flush_interval)
                # If we get here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Timeout is expected - time to flush
                await self.flush_metrics()
            except Exception as e:
                logger.warning(f"Error in periodic flush: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

    def set_block_context(self, block_number: int, window_number: int | None = None) -> None:
        """Set the current block and window context for metrics.

        Args:
            block_number: Current block number
            window_number: Current window number (optional)
        """
        self._current_block = block_number
        self._current_window = window_number

    async def log_counter(
        self,
        name: str,
        value: int | float = 1,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a counter metric.

        Args:
            name: Name of the counter
            value: Value to increment by (default 1)
            tags: Optional tags to attach
        """
        if not self._initialized:
            return

        # Ensure async components are ready
        self._ensure_async_components()

        metric = MetricData(
            name,
            value,
            MetricType.COUNTER,
            tags,
            block_number=self._current_block,
            window_number=self._current_window,
        )
        self._metric_buffer.append(metric)

        if len(self._metric_buffer) >= self._buffer_size:
            await self.flush_metrics()

    async def log_gauge(
        self,
        name: str,
        value: int | float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a gauge metric.

        Args:
            name: Name of the gauge
            value: Current value
            tags: Optional tags to attach
        """
        if not self._initialized:
            return

        # Gauges are logged immediately since they represent current state
        metric = MetricData(
            name,
            value,
            MetricType.GAUGE,
            tags,
            block_number=self._current_block,
            window_number=self._current_window,
        )
        try:
            await self.backend.log_metric(metric)
        except Exception as e:
            logger.warning(f"Failed to log gauge {name}: {e}")

    async def log_histogram(
        self, name: str, value: Any, tags: dict[str, str] | None = None
    ) -> None:
        """Log a histogram metric.

        Args:
            name: Name of the histogram
            value: Value to record (number or list/array for distributions)
            tags: Optional tags to attach
        """
        if not self._initialized:
            return

        metric = MetricData(
            name,
            value,
            MetricType.HISTOGRAM,
            tags,
            block_number=self._current_block,
            window_number=self._current_window,
        )
        self._metric_buffer.append(metric)

        if len(self._metric_buffer) >= self._buffer_size:
            await self.flush_metrics()

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Any:
        """Time an operation.

        Args:
            name: Name of the timer
            tags: Optional tags to attach

        Example:
            with manager.timer("operation_duration"):
                # ... timed operation ...
        """
        if not self._initialized:
            yield
            return

        with self.backend.timer(name, tags):
            yield

    async def log_artifact(self, name: str, data: Any, artifact_type: str) -> None:
        """Log an artifact.

        Args:
            name: Name/identifier for the artifact
            data: The artifact data
            artifact_type: Type of artifact ("model", "plot", "file", etc.)
        """
        if not self._initialized:
            return

        try:
            await self.backend.log_artifact(name, data, artifact_type)
        except Exception as e:
            logger.warning(f"Failed to log artifact {name}: {e}")

    async def flush_metrics(self) -> None:
        """Flush all buffered metrics to the backend."""
        if not self._metric_buffer or not self._initialized:
            return

        try:
            # Get current buffer and reset
            metrics_to_flush = self._metric_buffer.copy()
            self._metric_buffer.clear()

            # Send to backend
            await self.backend.log_metrics(metrics_to_flush)

        except Exception as e:
            logger.warning(f"Failed to flush metrics: {e}")
            # On failure, we lose the metrics but don't crash the application

    async def health_check(self) -> bool:
        """Check monitoring system health.

        Returns:
            True if the monitoring system is healthy, False otherwise
        """
        if not self._initialized:
            return False

        try:
            return await self.backend.health_check()
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def start_run(self, run_name: str, config: dict[str, Any] | None = None) -> str:
        """Start a new monitoring run.

        Args:
            run_name: Name for this run
            config: Optional additional configuration

        Returns:
            Run ID for this session
        """
        if not self._initialized:
            return "not_initialized"

        try:
            return await self.backend.start_run(run_name, config or {})
        except Exception as e:
            logger.warning(f"Failed to start run {run_name}: {e}")
            return "failed_start"

    async def finish_run(self, run_id: str) -> None:
        """Finish a monitoring run.

        Args:
            run_id: The run identifier
        """
        if not self._initialized:
            return

        # Flush any remaining metrics before finishing
        await self.flush_metrics()

        try:
            await self.backend.finish_run(run_id)
        except Exception as e:
            logger.warning(f"Failed to finish run {run_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the monitoring system and cleanup resources."""
        logger.info("Shutting down monitoring manager...")

        # Signal shutdown to background tasks
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Cancel flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining metrics
        await self.flush_metrics()

        # Shutdown backend
        if self._initialized:
            try:
                await self.backend.shutdown()
            except Exception as e:
                logger.warning(f"Error during backend shutdown: {e}")

        self._initialized = False
        logger.info("Monitoring manager shutdown completed")


# Global monitoring instance
_monitoring_manager: MonitoringManager | None = None


def get_monitoring_manager() -> MonitoringManager:
    """Get the global monitoring manager instance.

    Returns:
        The global MonitoringManager instance
    """
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager


def initialize_monitoring(backend_type: str = "wandb", **config: Any) -> None:
    """Initialize monitoring with the specified backend.

    Args:
        backend_type: Type of backend to use ("wandb", "null")
        **config: Configuration parameters for the backend

    Raises:
        ValueError: If backend_type is not supported
    """
    global _monitoring_manager

    # Create backend instance
    if backend_type == "wandb":
        backend: MonitoringBackend = WandBBackend()
    elif backend_type == "null":
        backend = NullBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

    # Create manager with backend
    _monitoring_manager = MonitoringManager(backend)

    # Initialize synchronously
    _monitoring_manager.initialize(config)

    logger.info(f"Monitoring initialized with {backend_type} backend")


async def shutdown_monitoring() -> None:
    """Shutdown the global monitoring system."""
    global _monitoring_manager
    if _monitoring_manager:
        await _monitoring_manager.shutdown()
        _monitoring_manager = None
