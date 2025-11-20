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
        # Track which metric names we have already defined a step for (avoid spam)
        self._defined_step_for: set[str] = set()

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

    def _build_wandb_settings(self) -> Any | None:
        """Build wandb.Settings with shared mode, init_timeout, and custom overrides.
        
        WandB shared mode (>= 0.19.9) enables multiple processes to write to ONE run:
        - Primary process: mode="shared", x_primary=True
        - Worker processes: mode="shared", x_primary=False, x_label="worker_name"
        
        This replaces the old resume-based approach which caused 180s timeouts.
        
        Returns:
            wandb.Settings instance if any overrides specified, else None
        """
        if self._wandb_module is None:
            return None

        # CRITICAL: For shared mode, ONLY pass minimal settings (x_primary, x_label, init_timeout)
        # Adding any other settings (disable_git, heartbeat_seconds, etc.) may cause API timeouts
        if self.config.get("wandb_shared_mode"):
            settings_kwargs: dict[str, Any] = {}
            
            # Primary process or worker process?
            x_primary = self.config.get("wandb_x_primary", False)
            settings_kwargs["x_primary"] = x_primary
            
            # Label to distinguish processes in logs
            x_label = self.config.get("wandb_x_label", "unknown")
            settings_kwargs["x_label"] = x_label
            
            # Init timeout (default 120s, increase for slow networks)
            init_timeout = self.config.get("init_timeout", 120.0)
            if isinstance(init_timeout, (int, float)) and init_timeout > 0:
                settings_kwargs["init_timeout"] = float(init_timeout)
            
            logger.info(
                "Configuring MINIMAL WandB Settings for shared mode: x_primary=%s x_label=%s init_timeout=%ss",
                x_primary,
                x_label,
                settings_kwargs.get("init_timeout"),
            )
            
            try:
                return self._wandb_module.Settings(**settings_kwargs)
            except Exception as exc:
                logger.warning("Failed to build WandB settings %s: %s", settings_kwargs, exc)
                return None
        
        # Non-shared mode: standard settings
        settings_kwargs: dict[str, Any] = {}
        
        # Init timeout (for slow network connections)
        init_timeout = self.config.get("init_timeout")
        if isinstance(init_timeout, (int, float)) and init_timeout > 0:
            settings_kwargs["init_timeout"] = float(init_timeout)

        # Allow additional custom settings from config (non-shared mode only!)
        custom_settings = self.config.get("wandb_settings")
        if isinstance(custom_settings, dict):
            settings_kwargs.update(custom_settings)

        if not settings_kwargs:
            return None

        try:
            return self._wandb_module.Settings(**settings_kwargs)
        except Exception as exc:
            logger.warning("Failed to build WandB settings %s: %s", settings_kwargs, exc)
            return None

    async def _ensure_wandb_run(self) -> None:
        """Ensure wandb run is initialized (lazy initialization)."""
        if self._wandb_run_started or not self._initialized or self._wandb_module is None:
            return

        try:
            # Run wandb.init() in thread pool to avoid blocking event loop
            run = await asyncio.to_thread(self._sync_wandb_init)
            if run is not None:
                self._wandb_run_started = True
                logger.debug("WandB run started successfully")
            else:
                logger.warning("WandB run initialization returned None")
        except Exception as e:
            logger.warning(f"Failed to start WandB run: {e}")

    def _sync_wandb_init(self) -> Any:
        """Synchronous wandb.init() call.
        
        CRITICAL: In shared mode, workers MUST NOT pass name/config/tags/notes!
        Only primary process sets metadata. Workers only pass: id, project, entity.
        Passing metadata to workers causes 120s+ API timeout.

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

        init_kwargs: dict[str, Any] = {
            "project": self.config.get("project", "grail"),
        }

        # Build settings first (includes shared mode + timeout config)
        wandb_settings = self._build_wandb_settings()
        
        # Handle multi-process coordination
        if self.config.get("wandb_shared_mode"):
            # Shared mode (wandb >= 0.19.9): multiple processes write to ONE run
            # Per WandB docs, mode="shared" is a direct init() parameter
            # Settings contains x_primary=True/False, x_label="..." for process identification
            init_kwargs["mode"] = "shared"
            
            # Primary process: sets run metadata
            # Worker processes: only pass id, project, entity to connect to existing run
            is_primary = self.config.get("wandb_x_primary", False)
            
            if is_primary:
                # Primary creates the run with full metadata
                init_kwargs.update({
                    "name": self.config.get("run_name"),
                    "config": self.config.get("hyperparameters", {}),
                    "tags": self.config.get("tags", []),
                    "notes": self.config.get("notes", ""),
                })
                logger.info(
                    "Using WandB shared mode as PRIMARY (x_label=%s)",
                    self.config.get("wandb_x_label"),
                )
            else:
                # CRITICAL: Worker connects to existing run
                # Only pass 'id', 'project', 'entity' - DO NOT pass name/config/tags/notes!
                # Passing metadata params causes WandB API timeout (120s+) while it tries
                # to validate/merge them with the existing run. Test scripts verify this:
                # - With metadata: 120s timeout
                # - Without metadata: <5s connection
                if self.config.get("run_id"):
                    init_kwargs["id"] = self.config["run_id"]
                    logger.info(
                        "Using WandB shared mode as WORKER with run ID: %s (x_label=%s) - metadata inherited from primary",
                        self.config["run_id"],
                        self.config.get("wandb_x_label"),
                    )
                else:
                    logger.warning(
                        "Worker process missing run_id - cannot connect to existing shared run"
                    )
        else:
            # Legacy mode: use resume-based multi-process (slower, deprecated)
            # In legacy mode, always pass full metadata for resume
            init_kwargs.update({
                "name": self.config.get("run_name"),
                "config": self.config.get("hyperparameters", {}),
                "tags": self.config.get("tags", []),
                "notes": self.config.get("notes", ""),
                "mode": mode,
                "resume": self.config.get("resume", "allow"),
            })
            
            if self.config.get("run_id"):
                init_kwargs["id"] = self.config["run_id"]
                init_kwargs["resume"] = "allow"
                logger.info("Resuming W&B run with ID: %s", self.config["run_id"])

        # Only set entity if provided
        if self.config.get("entity"):
            init_kwargs["entity"] = self.config["entity"]

        if wandb_settings is not None:
            init_kwargs["settings"] = wandb_settings

        # For shared mode workers, log minimal params being used
        if self.config.get("wandb_shared_mode") and not self.config.get("wandb_x_primary", False):
            logger.info(
                "🔗 WORKER connecting with MINIMAL params (matches test_wandb_shared.py): "
                "keys=%s (expecting: project, mode, id, entity, settings [+ dir])",
                list(init_kwargs.keys()),
            )
            if "name" in init_kwargs or "config" in init_kwargs or "tags" in init_kwargs:
                logger.error(
                    "❌ BUG: Worker has metadata params (name/config/tags) - this causes 120s timeout!"
                )
            
            # CRITICAL FIX: Use separate directory for worker process to avoid file conflicts
            # This prevents "Stale file handle" errors when parent and child both write to wandb/
            import os
            subprocess_wandb_dir = os.path.join(os.getcwd(), "wandb_training")
            try:
                os.makedirs(subprocess_wandb_dir, exist_ok=True)
                init_kwargs["dir"] = subprocess_wandb_dir
                logger.info(
                    "🔧 Using separate WandB directory for worker: %s (prevents file conflicts)",
                    subprocess_wandb_dir,
                )
            except Exception as dir_exc:
                logger.warning("Failed to create separate WandB dir %s: %s", subprocess_wandb_dir, dir_exc)
            
            # Debug: Log WandB-related environment variables that might affect connection
            wandb_env_vars = {k: v for k, v in os.environ.items() if 'WANDB' in k.upper()}
            if wandb_env_vars:
                logger.debug("WandB env vars in subprocess: %s", list(wandb_env_vars.keys()))

        # Debug: Log exact parameters being passed to wandb.init()
        # Log all wandb.init parameters for traceability (excluding any secrets)
        formatted_params = {
            k: ("***" if "key" in k or "secret" in k else v)
            for k, v in init_kwargs.items()
        }
        logger.debug(
            "Calling wandb.init() with parameters: %s",
            formatted_params,
            extra={"wandb_init_keys": list(init_kwargs.keys())},
        )

        run = self._wandb_module.init(**init_kwargs)
        self.run = run

        # Define custom x-axis metrics for different metric families
        # This allows different panels to use different x-axes
        if run is not None:
            # Define all possible x-axis metrics
            self._wandb_module.define_metric("block_number")
            self._wandb_module.define_metric("epoch")
            self._wandb_module.define_metric("batch_step")
            self._wandb_module.define_metric("global_step")

            # Define which metric families use which x-axis
            # IMPORTANT: Define specific patterns LAST so they override the default
            # Training metrics use epoch/batch_step for batch-level granularity
            self._wandb_module.define_metric("training/epoch/*", step_metric="epoch")
            self._wandb_module.define_metric("training/batch/*", step_metric="batch_step")
            # Training block metrics go under their own namespace
            self._wandb_module.define_metric("training/block/*", step_metric="block_number")
            # Prefilter metrics should step by block_number as well
            self._wandb_module.define_metric("training/prefilter/*", step_metric="block_number")

            # Mining and validation use block_number (blockchain progression)
            self._wandb_module.define_metric("mining/*", step_metric="block_number")
            self._wandb_module.define_metric("validation/*", step_metric="block_number")
            self._wandb_module.define_metric("weights/*", step_metric="block_number")
            self._wandb_module.define_metric("miner_sampling/*", step_metric="block_number")

            # Evaluation metrics use block_number for alignment with training progression
            self._wandb_module.define_metric("eval/*", step_metric="block_number")

            # Profiling metrics for mining/validation: use block_number for consistency
            self._wandb_module.define_metric("profiling/*", step_metric="block_number")

            # Per-UID metrics (e.g., "55/total_rollouts_avg") use block_number
            # These are logged during validation for each miner UID
            for uid in range(256):  # Cover all possible UIDs
                self._wandb_module.define_metric(f"{uid}/*", step_metric="block_number")

            logger.debug(
                "Configured wandb with multi-axis metrics (epoch, batch_step, window_number, block_number)"
            )

        return run

    def _maybe_define_step_for_name(self, name: str) -> None:
        """Define the appropriate step metric for a specific metric name.

        This is idempotent and safe to call repeatedly.
        """
        if self._wandb_module is None or self.run is None:
            return
        if name in self._defined_step_for:
            return

        step_metric: str | None = None
        if name.startswith("training/epoch/"):
            step_metric = "epoch"
        elif name.startswith("training/batch/"):
            step_metric = "batch_step"
        elif name.startswith("training/block/"):
            step_metric = "block_number"
        elif name.startswith("training/prefilter/"):
            step_metric = "block_number"
        elif name.startswith("eval/"):
            step_metric = "block_number"
        elif name.startswith("mining/"):
            step_metric = "block_number"
        elif name.startswith("validation/"):
            step_metric = "block_number"
        elif name.startswith("weights/"):
            step_metric = "block_number"
        elif name.startswith("miner_sampling/"):
            step_metric = "block_number"
        elif name.startswith("profiling/"):
            step_metric = "block_number"
        else:
            # Per-UID metrics like "55/total_rollouts_avg"
            try:
                uid_prefix = name.split("/", 1)[0]
                if uid_prefix.isdigit():
                    step_metric = "block_number"
            except Exception:
                step_metric = None

        if step_metric is not None:
            try:
                self._wandb_module.define_metric(name, step_metric=step_metric)
                self._defined_step_for.add(name)
            except Exception:
                # Best effort; if define_metric fails we still log the metric
                pass

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

            # Proactively define step per metric name to force correct x-axis
            self._maybe_define_step_for_name(metric.name)

            # Prepare data for wandb
            data = self._prepare_metric_data(metric)

            # Include temporal context in the data
            self._add_temporal_context(data, metric)

            # Log to wandb in thread pool without step parameter
            # wandb will use timestamp as x-axis as configured
            await asyncio.to_thread(self._wandb_module.log, data)

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

            # Proactively define step for each metric in the batch
            for metric in metrics:
                self._maybe_define_step_for_name(metric.name)

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
            # This adds window_number, block_number, and timestamp as reference metrics
            if metrics:
                self._add_temporal_context(data, metrics[-1])

            # Log all metrics in one call without step parameter
            # WandB uses the configured step_metric (window_number for most metrics)
            await asyncio.to_thread(self._wandb_module.log, data)

        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")

    def _prepare_metric_data(self, metric: MetricData) -> dict[str, Any]:
        """Prepare metric data for wandb logging.

        Args:
            metric: The metric to prepare

        Returns:
            Dictionary suitable for wandb.log()
        """
        # Reserved tag keys that should be extracted as x-axis fields, not appended to name
        # These are used for custom step metrics (epoch, batch_step, window_number, etc.)
        RESERVED_TAGS = {"epoch", "batch_step", "window_number", "global_step", "block_number"}

        # Start with clean metric name (no tags appended)
        name = metric.name
        result = {}

        # Process tags if present
        if metric.tags:
            reserved_fields: dict[str, Any] = {}
            non_reserved_tags: dict[str, str] = {}

            for key, raw in metric.tags.items():
                if key in RESERVED_TAGS:
                    # Reserved tags become x-axis fields in the logged dict.
                    # Coerce from string to int where possible, else float, else leave as string.
                    # Note: tags are typed as str in MetricData, so we avoid unreachable
                    # branches for numeric types to satisfy static analysis.
                    coerced: Any = raw
                    if isinstance(raw, str):
                        if raw.isdigit():
                            coerced = int(raw)
                        else:
                            try:
                                coerced = float(raw)
                            except Exception:
                                coerced = raw
                    reserved_fields[key] = coerced
                else:
                    # Non-reserved tags: preserve old behavior (append to name)
                    # This maintains backward compatibility for any custom tags
                    non_reserved_tags[key] = str(raw)

            # Only append non-reserved tags to metric name (backward compatible)
            if non_reserved_tags:
                tag_parts = [f"{k}_{v}" for k, v in non_reserved_tags.items()]
                name = f"{metric.name}_{'_'.join(tag_parts)}"

            # Add reserved fields to result dict (will be logged alongside metric)
            result.update(reserved_fields)

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
                try:
                    value = float(value) if not float(value).is_integer() else int(value)
                except Exception:
                    value = float(value)

        # Add the metric itself
        result[name] = value
        return result

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
        # Add window_number if present (for training/window metrics)
        if metric.window_number is not None and "window_number" not in data:
            data["window_number"] = metric.window_number

        # Add block_number for all metrics (primary x-axis for mining/validation/profiling)
        if metric.block_number is not None:
            data["block_number"] = metric.block_number

        # Secondary metrics for different views (these don't affect x-axis)
        if metric.timestamp:
            data["timestamp"] = metric.timestamp

            # Calculate elapsed time if we have a start time
            if hasattr(self, "_start_time") and self._start_time:
                elapsed_seconds = metric.timestamp - self._start_time
                data["elapsed_minutes"] = round(elapsed_seconds / 60, 2)
                data["elapsed_hours"] = round(elapsed_seconds / 3600, 3)
            elif not hasattr(self, "_start_time"):
                self._start_time = metric.timestamp

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
            config: Configuration and metadata for the run (from MonitoringConfig.for_training/mining/validation)

        Returns:
            Run ID for this session
        """
        # Debug: Log what we received
        logger.debug(
            "Backend start_run called: run_name=%s, config_keys=%s, wandb_shared_mode=%s",
            run_name,
            list(config.keys()) if config else None,
            config.get("wandb_shared_mode") if config else None,
        )
        
        # Update config with new values from training/mining/validation config
        # Note: config already includes run_name from MonitoringConfig.for_training()
        self.config.update(config)
        
        # Ensure run_name is set (in case it's not in config)
        if "run_name" not in self.config:
            self.config["run_name"] = run_name
        
        # Debug: Verify shared mode config is present after update
        logger.debug(
            "Backend after config update: wandb_shared_mode=%s x_primary=%s x_label=%s",
            self.config.get("wandb_shared_mode"),
            self.config.get("wandb_x_primary"),
            self.config.get("wandb_x_label"),
        )

        # Ensure wandb run is started
        await self._ensure_wandb_run()

        if not self._wandb_run_started:
            raise RuntimeError(
                f"WandB run initialization failed (timeout or error). "
                f"Check WANDB_INIT_TIMEOUT (current: {self.config.get('init_timeout', 120)}) "
                f"and network connectivity."
            )

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
