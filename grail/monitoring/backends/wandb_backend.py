"""
Weights & Biases monitoring backend implementation.

This module provides a concrete implementation of the MonitoringBackend interface
for Weights & Biases (wandb), with proper async support and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..base import MetricData, MetricType, MonitoringBackend

logger = logging.getLogger(__name__)


class WandBBackend(MonitoringBackend):
    """Weights & Biases monitoring backend implementation.

    This backend provides integration with Weights & Biases for experiment tracking,
    metrics logging, and artifact storage. It handles async operations properly
    and provides graceful error handling for production environments.
    """

    def __init__(self) -> None:
        """Initialize the WandB backend."""
        self.run: Any = None
        self.config: dict[str, Any] = {}
        self._initialized = False
        self._wandb_module: Any = None
        self._wandb_run_started = False
        self._start_time: float | None = None
        # Cache of persistent tables by name to allow incremental row appends
        self._tables: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize wandb backend synchronously (no network calls).

        Args:
            config: Configuration dictionary with wandb settings

        Expected config keys:
            - project: wandb project name
            - entity: wandb entity/team name (optional)
            - run_name: name for this run (optional)
            - mode: "online", "offline", or "disabled"
            - tags: list of tags for the run
            - notes: description/notes for the run
            - hyperparameters: dict of hyperparameters to log
            - resume: "allow", "must", "never", or "auto"
        """
        try:
            # Import wandb module (sync operation)
            self._wandb_module = self._import_wandb()

            if self._wandb_module is None:
                logger.warning("wandb not available, monitoring will be disabled")
                self._initialized = False
                return

            # Store configuration for later use
            self.config = config.copy()

            # Validate required configuration
            if not config.get("project"):
                logger.warning("wandb project not specified, using default")
                self.config["project"] = "grail"

            self._initialized = True
            logger.info("WandB backend initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self._initialized = False

    def _import_wandb(self) -> Any:
        """Import wandb module safely."""
        try:
            import wandb

            return wandb
        except ImportError:
            return None

    async def _ensure_wandb_run(self) -> None:
        """Ensure wandb run is initialized (lazy initialization)."""
        if self._wandb_run_started or not self._initialized or self._wandb_module is None:
            return

        try:
            # Run wandb.init() in thread pool to avoid blocking
            run = await asyncio.get_event_loop().run_in_executor(None, self._sync_wandb_init)
            if run is not None:
                self._wandb_run_started = True
                logger.debug("WandB run started successfully")
            else:
                logger.warning("WandB run initialization returned None")
        except Exception as e:
            logger.warning(f"Failed to start WandB run: {e}")

    def _sync_wandb_init(self) -> Any:
        """Synchronous wandb.init() call.

        Returns:
            The wandb run object or None if initialization failed
        """
        if self._wandb_module is None:
            return None

        # Handle mode setting
        mode = self.config.get("mode", "online")
        if mode == "disabled":
            logger.info("WandB monitoring disabled by configuration")
            return None

        init_kwargs = {
            "project": self.config.get("project", "grail"),
            "name": self.config.get("run_name"),
            "config": self.config.get("hyperparameters", {}),
            "mode": mode,
            "tags": self.config.get("tags", []),
            "notes": self.config.get("notes", ""),
            "resume": self.config.get("resume", "allow"),
            # Use resume instead of deprecated reinit
            "force": True,  # Force new run even if one exists
        }

        # Only set entity if provided
        if self.config.get("entity"):
            init_kwargs["entity"] = self.config["entity"]

        run = self._wandb_module.init(**init_kwargs)
        self.run = run

        # Define block_number as the primary x-axis for all metrics
        # This is more meaningful for blockchain systems
        if run is not None:
            self._wandb_module.define_metric("block_number")
            self._wandb_module.define_metric("*", step_metric="block_number")
            logger.debug("Configured wandb to use block_number as x-axis")

        return run

    async def log_metric(self, metric: MetricData) -> None:
        """Log a single metric to wandb.

        Args:
            metric: The metric data to log
        """
        if not self._initialized or self._wandb_module is None:
            return

        try:
            # Ensure wandb run is started
            await self._ensure_wandb_run()

            if not self._wandb_run_started:
                return

            # Prepare data for wandb
            data = self._prepare_metric_data(metric)

            # Include temporal context in the data
            self._add_temporal_context(data, metric)

            # Log to wandb in thread pool without step parameter
            # wandb will use timestamp as x-axis as configured
            await asyncio.get_event_loop().run_in_executor(None, self._wandb_module.log, data)

        except Exception as e:
            logger.warning(f"Failed to log metric {metric.name}: {e}")

    async def log_metrics(self, metrics: list[MetricData]) -> None:
        """Log multiple metrics efficiently to wandb.

        Args:
            metrics: List of metrics to log
        """
        if not self._initialized or not metrics or self._wandb_module is None:
            return

        try:
            # Ensure wandb run is started
            await self._ensure_wandb_run()

            if not self._wandb_run_started:
                return

            # Combine all metrics into a single data dict
            data: dict[str, Any] = {}
            latest_timestamp = None

            for metric in metrics:
                metric_data = self._prepare_metric_data(metric)
                data.update(metric_data)
                # Track the latest timestamp
                if latest_timestamp is None or (
                    metric.timestamp and metric.timestamp > latest_timestamp
                ):
                    latest_timestamp = metric.timestamp

            # Include temporal context using the latest metric
            if metrics:
                self._add_temporal_context(data, metrics[-1])

            # Log all metrics in one call without step parameter
            # wandb will use timestamp as x-axis as configured
            await asyncio.get_event_loop().run_in_executor(None, self._wandb_module.log, data)

        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")

    def _prepare_metric_data(self, metric: MetricData) -> dict[str, Any]:
        """Prepare metric data for wandb logging.

        Args:
            metric: The metric to prepare

        Returns:
            Dictionary suitable for wandb.log()
        """
        # Create metric name with tags
        name = metric.name
        if metric.tags:
            # Flatten tags into metric name for wandb
            tag_parts = [f"{k}_{v}" for k, v in metric.tags.items()]
            name = f"{metric.name}_{'_'.join(tag_parts)}"

        # Ensure value is properly typed for wandb
        value = metric.value
        if metric.metric_type == MetricType.HISTOGRAM:
            # Wrap arrays/lists as a wandb.Histogram media object
            value = self._to_wandb_histogram(value)
        else:
            # Convert to Python native types to avoid wandb serialization issues
            if hasattr(value, "item"):  # Handle numpy/torch scalars
                value = value.item()
            elif isinstance(value, (int, float)):
                # Ensure it's a Python native type, not numpy
                value = float(value) if isinstance(value, float) else int(value)

        return {name: value}

    def _to_wandb_histogram(self, value: Any) -> Any:
        """Convert arbitrary value into a wandb.Histogram when possible.

        Falls back to the original value if conversion fails or wandb is unavailable.
        """
        if self._wandb_module is None:
            return value
        try:
            # Normalize to a 1D numpy array
            array_like: np.ndarray | None = None
            # Handle torch tensors without importing torch explicitly
            if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
                try:
                    array_like = value.detach().cpu().numpy().ravel()
                except Exception:
                    array_like = None
            if array_like is None:
                try:
                    array_like = np.asarray(value).ravel()
                except Exception:
                    array_like = None
            if array_like is None:
                return value
            # Create wandb Histogram from sequence of values
            return self._wandb_module.Histogram(array_like)
        except Exception as exc:
            logger.debug(f"Failed to convert value to wandb.Histogram: {exc}")
            return value

    def _to_optional_float(self, value: Any) -> float | None:
        """Best-effort conversion to float with None passthrough.

        Args:
            value: Arbitrary input that might represent a numeric value

        Returns:
            A float if conversion succeeds, otherwise None.
        """
        if value is None:
            return None
        try:
            return float(value)  # Accepts int, float, and numeric strings
        except Exception:
            return None

    def _add_temporal_context(self, data: dict[str, Any], metric: MetricData) -> None:
        """Add temporal context to the data for better visualization.

        Args:
            data: The data dictionary to update
            metric: The metric containing temporal information
        """
        # Primary x-axis: block number
        if metric.block_number is not None:
            data["block_number"] = metric.block_number

        # Secondary metrics for different views
        if metric.timestamp:
            data["timestamp"] = metric.timestamp

            # Calculate elapsed time if we have a start time
            if hasattr(self, "_start_time") and self._start_time:
                elapsed_seconds = metric.timestamp - self._start_time
                data["elapsed_minutes"] = round(elapsed_seconds / 60, 2)
                data["elapsed_hours"] = round(elapsed_seconds / 3600, 3)
            elif not hasattr(self, "_start_time"):
                self._start_time = metric.timestamp

        # Window number for GRAIL-specific context
        if metric.window_number is not None:
            data["window_number"] = metric.window_number

    @contextmanager
    def timer(self, name: str, tags: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Context manager for timing operations.

        Args:
            name: Name of the timer metric
            tags: Optional tags to attach to the metric

        Yields:
            None
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            metric = MetricData(
                name=f"{name}_duration",
                value=duration,
                metric_type=MetricType.TIMER,
                tags=tags,
            )
            # Schedule async logging without blocking
            asyncio.create_task(self.log_metric(metric))

    async def log_artifact(self, name: str, data: Any, artifact_type: str) -> None:
        """Log artifacts to wandb.

        Args:
            name: Name/identifier for the artifact
            data: The artifact data
            artifact_type: Type of artifact ("model", "plot", "file", etc.)
        """
        if not self._initialized or self._wandb_module is None:
            return

        try:
            if artifact_type == "model":
                # For model files, use wandb.save
                await asyncio.get_event_loop().run_in_executor(None, self._wandb_module.save, data)
            elif artifact_type == "plot":
                # For plots, log as wandb object
                await asyncio.get_event_loop().run_in_executor(
                    None, self._wandb_module.log, {name: data}
                )
            elif artifact_type == "file":
                # For general files, use wandb.save
                await asyncio.get_event_loop().run_in_executor(None, self._wandb_module.save, data)
            elif artifact_type == "text":
                # Represent text samples as a wandb.Table for consistent rendering
                # across dashboards and to avoid media-vs-string key conflicts.
                await self._ensure_wandb_run()
                if not self._wandb_run_started:
                    return

                # Prefer a stable schema; missing fields will be None.
                base_columns = [
                    "window",
                    "wallet",
                    "group",
                    "nonce",
                    "reward",
                    "advantage",
                    "success",
                    "text",
                ]

                # Use a persistent table per name so rows accumulate
                if isinstance(data, dict):
                    row = [
                        data.get("window"),
                        data.get("wallet"),
                        data.get("group"),
                        data.get("nonce"),
                        self._to_optional_float(data.get("reward")),
                        self._to_optional_float(data.get("advantage")),
                        (bool(data.get("success")) if data.get("success") is not None else None),
                        str(data.get("text") or ""),
                    ]
                    table = self._tables.get(name)
                    if table is None:
                        table = self._wandb_module.Table(columns=base_columns, log_mode="MUTABLE")
                        self._tables[name] = table
                    table.add_data(*row)
                else:
                    table = self._tables.get(name)
                    if table is None:
                        table = self._wandb_module.Table(columns=["text"], log_mode="MUTABLE")
                        self._tables[name] = table
                    table.add_data(str(data))

                await asyncio.get_event_loop().run_in_executor(
                    None, self._wandb_module.log, {name: table}
                )
                return
            else:
                # Default: try to log as data
                await asyncio.get_event_loop().run_in_executor(
                    None, self._wandb_module.log, {name: data}
                )

        except Exception as e:
            logger.warning(f"Failed to log artifact {name}: {e}")

    async def start_run(self, run_name: str, config: dict[str, Any]) -> str:
        """Start a new wandb run.

        Args:
            run_name: Name for this run
            config: Configuration and metadata for the run

        Returns:
            Run ID for this session
        """
        # Update config with new run name
        self.config.update({**config, "run_name": run_name})

        # Ensure wandb run is started
        await self._ensure_wandb_run()

        if self.run and hasattr(self.run, "id"):
            return str(self.run.id)
        return "wandb_run_unknown"

    async def finish_run(self, run_id: str) -> None:
        """Finish the current wandb run.

        Args:
            run_id: The run identifier (not used by wandb)
        """
        if self._initialized and self.run and self._wandb_module:
            try:
                await asyncio.get_event_loop().run_in_executor(None, self._wandb_module.finish)
                self.run = None
                self._initialized = False
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")

    async def health_check(self) -> bool:
        """Check if wandb is healthy and operational.

        Returns:
            True if wandb is healthy, False otherwise
        """
        return self._initialized and self._wandb_module is not None and self.run is not None

    async def shutdown(self) -> None:
        """Shutdown the wandb backend and cleanup resources."""
        if self._initialized:
            await self.finish_run("shutdown")
            logger.info("WandB backend shutdown completed")
