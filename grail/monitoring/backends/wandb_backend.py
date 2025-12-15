"""
Weights & Biases monitoring backend implementation.

This module provides a concrete implementation of the MonitoringBackend interface
for Weights & Biases (wandb), with proper async support and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..base import MetricData, MetricType, MonitoringBackend

logger = logging.getLogger(__name__)

# Reserved tag keys that should be extracted as x-axis fields, not appended to name
RESERVED_TAGS = {
    "epoch",
    "batch_step",
    "window_number",
    "global_step",
    "block_number",
    "optimizer_step",
}

# Step metric mappings: prefix -> step_metric
STEP_METRIC_PREFIXES = {
    "training/epoch/": "epoch",
    "training/batch/": "batch_step",
    "training/block/": "block_number",
    "training/prefilter/": "block_number",
    "param_change/": "optimizer_step",
    "param_change/sparse/": "optimizer_step",
    "eval/": "block_number",
    "mining/": "block_number",
    "validation/": "block_number",
    "weights/": "block_number",
    "miner_sampling/": "block_number",
    "profiling/": "block_number",
}

# Metric families that use block_number as step metric
BLOCK_NUMBER_METRICS = [
    "training/block/*",
    "training/prefilter/*",
    "mining/*",
    "validation/*",
    "weights/*",
    "miner_sampling/*",
    "eval/*",
    "profiling/*",
]


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

    def _is_available(self) -> bool:
        """Check if wandb module is available and initialized."""
        return self._initialized and self._wandb_module is not None

    def _get_config(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    def _is_shared_mode(self) -> bool:
        """Check if shared mode is enabled."""
        return bool(self._get_config("wandb_shared_mode"))

    def _is_primary(self) -> bool:
        """Check if this is the primary process in shared mode."""
        return bool(self._get_config("wandb_x_primary", False))

    def _run_async(self, func: Any, *args: Any) -> Any:
        """Run synchronous function in thread pool."""
        return asyncio.to_thread(func, *args)

    def _run_executor(self, func: Any, *args: Any) -> Any:
        """Run synchronous function in executor."""
        return asyncio.get_event_loop().run_in_executor(None, func, *args)

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

    def _extract_init_timeout(self, default: float = 120.0) -> float | None:
        """Extract and validate init_timeout from config."""
        init_timeout = self._get_config("init_timeout", default)
        if isinstance(init_timeout, (int, float)) and init_timeout > 0:
            return float(init_timeout)
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

        settings_kwargs: dict[str, Any] = {}

        if self._is_shared_mode():
            # CRITICAL: For shared mode, ONLY pass minimal settings (x_primary, x_label, init_timeout)
            # Adding any other settings (disable_git, heartbeat_seconds, etc.) may cause API timeouts
            settings_kwargs["x_primary"] = self._is_primary()
            settings_kwargs["x_label"] = self._get_config("wandb_x_label", "unknown")
            init_timeout = self._extract_init_timeout()
            if init_timeout is not None:
                settings_kwargs["init_timeout"] = init_timeout

            logger.info(
                "Configuring MINIMAL WandB Settings for shared mode: x_primary=%s x_label=%s init_timeout=%ss",
                settings_kwargs["x_primary"],
                settings_kwargs["x_label"],
                settings_kwargs.get("init_timeout"),
            )
        else:
            # Non-shared mode: standard settings
            init_timeout = self._extract_init_timeout()
            if init_timeout is not None:
                settings_kwargs["init_timeout"] = init_timeout

            # Allow additional custom settings from config (non-shared mode only!)
            custom_settings = self._get_config("wandb_settings")
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

    def _get_metadata_dict(self) -> dict[str, Any]:
        """Get metadata dictionary for wandb init."""
        return {
            "name": self._get_config("run_name"),
            "config": self._get_config("hyperparameters", {}),
            "tags": self._get_config("tags", []),
            "notes": self._get_config("notes", ""),
        }

    def _setup_worker_directory(self, init_kwargs: dict[str, Any]) -> None:
        """Setup separate directory for worker process to avoid file conflicts."""
        label = str(self._get_config("wandb_x_label", "worker"))
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
        subprocess_wandb_dir = os.path.join(os.getcwd(), f"wandb_{safe_label}")
        try:
            os.makedirs(subprocess_wandb_dir, exist_ok=True)
            init_kwargs["dir"] = subprocess_wandb_dir
            logger.info(
                "🔧 Using separate WandB directory for worker: %s (prevents file conflicts)",
                subprocess_wandb_dir,
            )
        except Exception as dir_exc:
            logger.warning(
                "Failed to create separate WandB dir %s: %s", subprocess_wandb_dir, dir_exc
            )

        # Debug: Log WandB-related environment variables that might affect connection
        wandb_env_vars = {k: v for k, v in os.environ.items() if "WANDB" in k.upper()}
        if wandb_env_vars:
            logger.debug("WandB env vars in subprocess: %s", list(wandb_env_vars.keys()))

    def _validate_worker_params(self, init_kwargs: dict[str, Any]) -> None:
        """Validate worker process parameters (must not include metadata)."""
        logger.info(
            "🔗 WORKER connecting with MINIMAL params (matches test_wandb_shared.py): "
            "keys=%s (expecting: project, mode, id, entity, settings [+ dir])",
            list(init_kwargs.keys()),
        )
        if "name" in init_kwargs or "config" in init_kwargs or "tags" in init_kwargs:
            logger.error(
                "❌ BUG: Worker has metadata params (name/config/tags) - this causes 120s timeout!"
            )

    def _configure_shared_mode_primary(self, init_kwargs: dict[str, Any]) -> None:
        """Configure init_kwargs for shared mode primary process."""
        init_kwargs.update(self._get_metadata_dict())
        logger.info(
            "Using WandB shared mode as PRIMARY (x_label=%s)",
            self._get_config("wandb_x_label"),
        )

    def _configure_shared_mode_worker(self, init_kwargs: dict[str, Any]) -> None:
        """Configure init_kwargs for shared mode worker process."""
        run_id = self._get_config("run_id")
        if run_id:
            init_kwargs["id"] = run_id
            logger.info(
                "Using WandB shared mode as WORKER with run ID: %s (x_label=%s) - metadata inherited from primary",
                run_id,
                self._get_config("wandb_x_label"),
            )
        else:
            logger.warning("Worker process missing run_id - cannot connect to existing shared run")

        self._validate_worker_params(init_kwargs)
        self._setup_worker_directory(init_kwargs)

    def _configure_legacy_mode(self, init_kwargs: dict[str, Any], mode: str) -> None:
        """Configure init_kwargs for legacy (non-shared) mode."""
        init_kwargs.update(self._get_metadata_dict())
        init_kwargs["mode"] = mode
        init_kwargs["resume"] = self._get_config("resume", "allow")

        run_id = self._get_config("run_id")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
            logger.info("Resuming W&B run with ID: %s", run_id)

    def _define_base_metrics(self) -> None:
        """Define base x-axis metrics."""
        if self._wandb_module is None:
            return

        self._wandb_module.define_metric("block_number")
        self._wandb_module.define_metric("epoch")
        self._wandb_module.define_metric("batch_step")
        self._wandb_module.define_metric("global_step")
        self._wandb_module.define_metric("optimizer_step")

    def _define_metric_families(self) -> None:
        """Define step metrics for different metric families."""
        if self._wandb_module is None:
            return

        # Training metrics use epoch/batch_step for batch-level granularity
        self._wandb_module.define_metric("training/epoch/*", step_metric="epoch")
        self._wandb_module.define_metric("training/batch/*", step_metric="batch_step")

        # Parameter change metrics use optimizer_step for x-axis (separate tab)
        self._wandb_module.define_metric("param_change/*", step_metric="optimizer_step")

        # All other metric families use block_number
        for pattern in BLOCK_NUMBER_METRICS:
            self._wandb_module.define_metric(pattern, step_metric="block_number")

        # Per-UID metrics (e.g., "55/total_rollouts_avg") use block_number
        for uid in range(256):  # Cover all possible UIDs
            self._wandb_module.define_metric(f"{uid}/*", step_metric="block_number")

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

        mode = self._get_config("mode", "online")
        if mode == "disabled":
            logger.info("WandB monitoring disabled by configuration")
            return None

        init_kwargs: dict[str, Any] = {"project": self._get_config("project", "grail")}
        wandb_settings = self._build_wandb_settings()

        if self._is_shared_mode():
            init_kwargs["mode"] = "shared"
            if self._is_primary():
                self._configure_shared_mode_primary(init_kwargs)
            else:
                self._configure_shared_mode_worker(init_kwargs)
        else:
            self._configure_legacy_mode(init_kwargs, mode)

        if self._get_config("entity"):
            init_kwargs["entity"] = self._get_config("entity")

        if wandb_settings is not None:
            init_kwargs["settings"] = wandb_settings

        # Debug: Log exact parameters being passed to wandb.init()
        formatted_params = {
            k: ("***" if "key" in k or "secret" in k else v) for k, v in init_kwargs.items()
        }
        logger.debug(
            "Calling wandb.init() with parameters: %s",
            formatted_params,
            extra={"wandb_init_keys": list(init_kwargs.keys())},
        )

        run = self._wandb_module.init(**init_kwargs)
        self.run = run

        if run is not None:
            self._define_base_metrics()
            self._define_metric_families()
            logger.debug(
                "Configured wandb with multi-axis metrics (epoch, batch_step, window_number, block_number)"
            )

        return run

    def _get_step_metric_for_name(self, name: str) -> str | None:
        """Determine the step metric for a given metric name."""
        # Check prefix-based mappings first
        for prefix, step_metric in STEP_METRIC_PREFIXES.items():
            if name.startswith(prefix):
                return step_metric

        # Per-UID metrics like "55/total_rollouts_avg"
        try:
            uid_prefix = name.split("/", 1)[0]
            if uid_prefix.isdigit():
                return "block_number"
        except Exception:
            pass

        return None

    def _maybe_define_step_for_name(self, name: str) -> None:
        """Define the appropriate step metric for a specific metric name.

        This is idempotent and safe to call repeatedly.
        """
        if self._wandb_module is None or self.run is None:
            return
        if name in self._defined_step_for:
            return

        step_metric = self._get_step_metric_for_name(name)
        if step_metric is not None:
            try:
                self._wandb_module.define_metric(name, step_metric=step_metric)
                self._defined_step_for.add(name)
            except Exception:
                # Best effort; if define_metric fails we still log the metric
                pass

    async def _ensure_run_and_define_steps(self, metrics: list[MetricData]) -> bool:
        """Ensure wandb run is started and define step metrics.

        Returns:
            True if ready to log, False otherwise
        """
        if not self._is_available():
            return False

        await self._ensure_wandb_run()
        if not self._wandb_run_started:
            return False

        for metric in metrics:
            self._maybe_define_step_for_name(metric.name)

        return True

    async def log_metric(self, metric: MetricData) -> None:
        """Log a single metric to wandb.

        Args:
            metric: The metric data to log
        """
        if not await self._ensure_run_and_define_steps([metric]):
            return

        try:
            data = self._prepare_metric_data(metric)
            self._add_temporal_context(data, metric)
            await self._run_async(self._wandb_module.log, data)
        except Exception as e:
            logger.warning(f"Failed to log metric {metric.name}: {e}")

    async def log_metrics(self, metrics: list[MetricData]) -> None:
        """Log multiple metrics efficiently to wandb.

        Args:
            metrics: List of metrics to log
        """
        if not metrics or not await self._ensure_run_and_define_steps(metrics):
            return

        try:
            data: dict[str, Any] = {}
            for metric in metrics:
                data.update(self._prepare_metric_data(metric))

            if metrics:
                self._add_temporal_context(data, metrics[-1])

            await self._run_async(self._wandb_module.log, data)
        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")

    def _coerce_tag_value(self, raw: Any) -> Any:
        """Coerce tag value from string to appropriate type."""
        if isinstance(raw, str):
            if raw.isdigit():
                return int(raw)
            try:
                return float(raw)
            except Exception:
                return raw
        return raw

    def _process_tags(self, metric: MetricData) -> tuple[str, dict[str, Any]]:
        """Process metric tags, returning (metric_name, reserved_fields)."""
        name = metric.name
        reserved_fields: dict[str, Any] = {}
        non_reserved_tags: dict[str, str] = {}

        if not metric.tags:
            return name, reserved_fields

        for key, raw in metric.tags.items():
            if key in RESERVED_TAGS:
                reserved_fields[key] = self._coerce_tag_value(raw)
            else:
                non_reserved_tags[key] = str(raw)

        if non_reserved_tags:
            tag_parts = [f"{k}_{v}" for k, v in non_reserved_tags.items()]
            name = f"{metric.name}_{'_'.join(tag_parts)}"

        return name, reserved_fields

    def _normalize_value(self, value: Any, metric_type: MetricType) -> Any:
        """Normalize metric value for wandb logging."""
        if metric_type == MetricType.HISTOGRAM:
            return self._to_wandb_histogram(value)

        if hasattr(value, "item"):  # Handle numpy/torch scalars
            return value.item()

        if isinstance(value, (int, float)):
            try:
                return float(value) if not float(value).is_integer() else int(value)
            except Exception:
                return float(value)

        return value

    def _prepare_metric_data(self, metric: MetricData) -> dict[str, Any]:
        """Prepare metric data for wandb logging.

        Args:
            metric: The metric to prepare

        Returns:
            Dictionary suitable for wandb.log()
        """
        name, reserved_fields = self._process_tags(metric)
        value = self._normalize_value(metric.value, metric.metric_type)

        result = {**reserved_fields, name: value}
        return result

    def _convert_torch_tensor(self, value: Any) -> np.ndarray | None:
        """Convert torch tensor to numpy array if possible."""
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            try:
                return value.detach().cpu().numpy().ravel()
            except Exception:
                return None
        return None

    def _to_wandb_histogram(self, value: Any) -> Any:
        """Convert arbitrary value into a wandb.Histogram when possible.

        Falls back to the original value if conversion fails or wandb is unavailable.
        """
        if self._wandb_module is None:
            return value

        try:
            array_like = self._convert_torch_tensor(value)
            if array_like is None:
                try:
                    array_like = np.asarray(value).ravel()
                except Exception:
                    return value

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

    def _get_or_create_table(self, name: str, columns: list[str]) -> Any:
        """Get existing table or create new one."""
        table = self._tables.get(name)
        if table is None:
            table = self._wandb_module.Table(columns=columns, log_mode="MUTABLE")
            self._tables[name] = table
        return table

    def _log_text_artifact(self, name: str, data: Any) -> None:
        """Log text artifact as wandb table."""
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
            table = self._get_or_create_table(name, base_columns)
            table.add_data(*row)
        else:
            table = self._get_or_create_table(name, ["text"])
            table.add_data(str(data))

        self._wandb_module.log({name: table})

    async def log_artifact(self, name: str, data: Any, artifact_type: str) -> None:
        """Log artifacts to wandb.

        Args:
            name: Name/identifier for the artifact
            data: The artifact data
            artifact_type: Type of artifact ("model", "plot", "file", etc.)
        """
        if not self._is_available():
            return

        try:
            if artifact_type == "text":
                await self._ensure_wandb_run()
                if not self._wandb_run_started:
                    return
                await self._run_executor(self._log_text_artifact, name, data)
            elif artifact_type in ("model", "file"):
                await self._run_executor(self._wandb_module.save, data)
            elif artifact_type == "plot":
                await self._run_executor(self._wandb_module.log, {name: data})
            else:
                # Default: try to log as data
                await self._run_executor(self._wandb_module.log, {name: data})
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
        logger.debug(
            "Backend start_run called: run_name=%s, config_keys=%s, wandb_shared_mode=%s",
            run_name,
            list(config.keys()) if config else None,
            config.get("wandb_shared_mode") if config else None,
        )

        self.config.update(config)
        if "run_name" not in self.config:
            self.config["run_name"] = run_name

        logger.debug(
            "Backend after config update: wandb_shared_mode=%s x_primary=%s x_label=%s",
            self._get_config("wandb_shared_mode"),
            self._get_config("wandb_x_primary"),
            self._get_config("wandb_x_label"),
        )

        await self._ensure_wandb_run()

        if not self._wandb_run_started:
            raise RuntimeError(
                f"WandB run initialization failed (timeout or error). "
                f"Check WANDB_INIT_TIMEOUT (current: {self._get_config('init_timeout', 120)}) "
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
        if self._is_available() and self.run:
            try:
                await self._run_executor(self._wandb_module.finish)
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
