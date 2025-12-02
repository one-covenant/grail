"""Validation metrics aggregation and reporting.

Provides comprehensive metrics collection, aggregation, and reporting
for validation pipeline performance and validator effectiveness.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidatorMetrics:
    """Metrics for a single validator."""

    check_name: str
    total_runs: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    failure_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate (0.0 to 1.0)."""
        if self.total_runs == 0:
            return 0.0
        return self.passed / self.total_runs

    @property
    def fail_rate(self) -> float:
        """Calculate fail rate (0.0 to 1.0)."""
        if self.total_runs == 0:
            return 0.0
        return self.failed / self.total_runs

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)."""
        if self.total_runs == 0:
            return 0.0
        return self.errors / self.total_runs

    @property
    def avg_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.total_runs == 0:
            return 0.0
        return self.total_time_ms / self.total_runs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_name": self.check_name,
            "total_runs": self.total_runs,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "fail_rate": self.fail_rate,
            "error_rate": self.error_rate,
            "avg_time_ms": self.avg_time_ms,
            "total_time_ms": self.total_time_ms,
            "failure_reasons": dict(self.failure_reasons),
        }


@dataclass
class WindowMetrics:
    """Aggregated metrics for a validation window."""

    window_start: int
    total_rollouts: int = 0
    valid_rollouts: int = 0
    invalid_rollouts: int = 0
    validator_metrics: dict[str, ValidatorMetrics] = field(
        default_factory=lambda: defaultdict(lambda: ValidatorMetrics(""))
    )
    miner_stats: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"valid": 0, "invalid": 0})
    )

    @property
    def validation_rate(self) -> float:
        """Calculate overall validation pass rate."""
        if self.total_rollouts == 0:
            return 0.0
        return self.valid_rollouts / self.total_rollouts

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "window_start": self.window_start,
            "total_rollouts": self.total_rollouts,
            "valid_rollouts": self.valid_rollouts,
            "invalid_rollouts": self.invalid_rollouts,
            "validation_rate": self.validation_rate,
            "validator_metrics": {
                name: metrics.to_dict()
                for name, metrics in self.validator_metrics.items()
            },
            "miner_stats": dict(self.miner_stats),
        }


class ValidationMetricsAggregator:
    """Aggregates validation metrics across rollouts and windows.

    Usage:
        aggregator = ValidationMetricsAggregator()

        # Record validation result
        aggregator.record_validation(
            window_start=100,
            miner_address="0x123...",
            checks={"schema_valid": True, "proof_valid": False},
            timings={"schema_valid": 1.2, "proof_valid": 45.3},
            metadata={"schema_errors": ["missing_field"]},
            is_valid=False
        )

        # Get metrics
        window_metrics = aggregator.get_window_metrics(100)
        all_metrics = aggregator.get_all_metrics()
    """

    def __init__(self):
        self._window_metrics: dict[int, WindowMetrics] = {}
        self._global_metrics: dict[str, ValidatorMetrics] = defaultdict(
            lambda: ValidatorMetrics("")
        )

    def record_validation(
        self,
        window_start: int,
        miner_address: str,
        checks: dict[str, bool],
        timings: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        is_valid: bool = False,
    ) -> None:
        """Record validation result for a single rollout.

        Args:
            window_start: Window start block number
            miner_address: Miner's address/hotkey
            checks: Dict of check_name -> passed/failed
            timings: Dict of check_name -> execution_time_ms (optional)
            metadata: Additional metadata from validation (optional)
            is_valid: Overall validation result
        """
        # Initialize window metrics if needed
        if window_start not in self._window_metrics:
            self._window_metrics[window_start] = WindowMetrics(window_start=window_start)

        window = self._window_metrics[window_start]
        window.total_rollouts += 1

        if is_valid:
            window.valid_rollouts += 1
            window.miner_stats[miner_address]["valid"] += 1
        else:
            window.invalid_rollouts += 1
            window.miner_stats[miner_address]["invalid"] += 1

        # Record per-validator metrics
        timings = timings or {}
        metadata = metadata or {}

        for check_name, passed in checks.items():
            # Window-level metrics
            if check_name not in window.validator_metrics:
                window.validator_metrics[check_name] = ValidatorMetrics(check_name)

            vm = window.validator_metrics[check_name]
            vm.total_runs += 1

            if passed:
                vm.passed += 1
            else:
                vm.failed += 1

                # Record failure reason if available
                failure_key = f"{check_name}_failure"
                if failure_key in metadata:
                    reason = metadata[failure_key]
                    vm.failure_reasons[reason] += 1

            # Record timing if available
            if check_name in timings:
                vm.total_time_ms += timings[check_name]

            # Global metrics (across all windows)
            if check_name not in self._global_metrics:
                self._global_metrics[check_name] = ValidatorMetrics(check_name)

            gm = self._global_metrics[check_name]
            gm.total_runs += 1
            if passed:
                gm.passed += 1
            else:
                gm.failed += 1
            if check_name in timings:
                gm.total_time_ms += timings[check_name]

    def get_window_metrics(self, window_start: int) -> WindowMetrics | None:
        """Get metrics for a specific window."""
        return self._window_metrics.get(window_start)

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all aggregated metrics."""
        return {
            "global_metrics": {
                name: metrics.to_dict()
                for name, metrics in self._global_metrics.items()
            },
            "window_metrics": {
                window: metrics.to_dict()
                for window, metrics in self._window_metrics.items()
            },
            "total_windows": len(self._window_metrics),
            "total_rollouts": sum(w.total_rollouts for w in self._window_metrics.values()),
        }

    def get_validator_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary statistics for each validator."""
        summary = {}
        for check_name, metrics in self._global_metrics.items():
            summary[check_name] = {
                "pass_rate": metrics.pass_rate,
                "fail_rate": metrics.fail_rate,
                "avg_time_ms": metrics.avg_time_ms,
                "total_runs": metrics.total_runs,
                "top_failures": sorted(
                    metrics.failure_reasons.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }
        return summary

    def get_miner_summary(
        self, window_start: int | None = None
    ) -> dict[str, dict[str, Any]]:
        """Get per-miner validation statistics.

        Args:
            window_start: Specific window to analyze, or None for all windows

        Returns:
            Dict mapping miner_address to stats
        """
        if window_start is not None:
            window = self._window_metrics.get(window_start)
            if not window:
                return {}
            return {
                miner: {
                    "valid": stats["valid"],
                    "invalid": stats["invalid"],
                    "total": stats["valid"] + stats["invalid"],
                    "success_rate": (
                        stats["valid"] / (stats["valid"] + stats["invalid"])
                        if (stats["valid"] + stats["invalid"]) > 0
                        else 0.0
                    ),
                }
                for miner, stats in window.miner_stats.items()
            }

        # Aggregate across all windows
        all_miner_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"valid": 0, "invalid": 0}
        )
        for window in self._window_metrics.values():
            for miner, stats in window.miner_stats.items():
                all_miner_stats[miner]["valid"] += stats["valid"]
                all_miner_stats[miner]["invalid"] += stats["invalid"]

        return {
            miner: {
                "valid": stats["valid"],
                "invalid": stats["invalid"],
                "total": stats["valid"] + stats["invalid"],
                "success_rate": (
                    stats["valid"] / (stats["valid"] + stats["invalid"])
                    if (stats["valid"] + stats["invalid"]) > 0
                    else 0.0
                ),
            }
            for miner, stats in all_miner_stats.items()
        }

    def print_summary(self, window_start: int | None = None) -> None:
        """Print human-readable summary of metrics.

        Args:
            window_start: Specific window to summarize, or None for global summary
        """
        if window_start is not None:
            window = self._window_metrics.get(window_start)
            if not window:
                logger.info(f"No metrics for window {window_start}")
                return

            logger.info(f"\n{'='*60}")
            logger.info(f"Validation Metrics - Window {window_start}")
            logger.info(f"{'='*60}")
            logger.info(
                f"Total Rollouts: {window.total_rollouts} "
                f"(Valid: {window.valid_rollouts}, Invalid: {window.invalid_rollouts})"
            )
            logger.info(f"Validation Rate: {window.validation_rate:.2%}")
            logger.info(f"\nPer-Validator Metrics:")

            for check_name, metrics in sorted(window.validator_metrics.items()):
                logger.info(
                    f"  {check_name:30s} | "
                    f"Pass: {metrics.pass_rate:6.2%} | "
                    f"Fail: {metrics.fail_rate:6.2%} | "
                    f"Avg: {metrics.avg_time_ms:6.2f}ms"
                )

                if metrics.failure_reasons:
                    top_reasons = sorted(
                        metrics.failure_reasons.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    for reason, count in top_reasons:
                        logger.info(f"    └─ {reason}: {count}")

        else:
            # Global summary
            logger.info(f"\n{'='*60}")
            logger.info("Global Validation Metrics")
            logger.info(f"{'='*60}")
            logger.info(f"Total Windows: {len(self._window_metrics)}")
            total_rollouts = sum(w.total_rollouts for w in self._window_metrics.values())
            logger.info(f"Total Rollouts: {total_rollouts}")

            logger.info(f"\nValidator Performance:")
            for check_name, metrics in sorted(self._global_metrics.items()):
                logger.info(
                    f"  {check_name:30s} | "
                    f"Pass: {metrics.pass_rate:6.2%} | "
                    f"Runs: {metrics.total_runs:6d} | "
                    f"Avg: {metrics.avg_time_ms:6.2f}ms"
                )

    def reset_window(self, window_start: int) -> None:
        """Clear metrics for a specific window."""
        if window_start in self._window_metrics:
            del self._window_metrics[window_start]

    def reset_all(self) -> None:
        """Clear all metrics."""
        self._window_metrics.clear()
        self._global_metrics.clear()


class ValidationTimer:
    """Context manager for timing validator execution.

    Usage:
        with ValidationTimer() as timer:
            validator.validate(ctx)
        elapsed_ms = timer.elapsed_ms
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> ValidationTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        return False

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0
