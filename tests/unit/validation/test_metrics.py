"""Tests for validation metrics aggregation and reporting."""

from __future__ import annotations

import pytest

from grail.validation.metrics import (
    ValidationMetricsAggregator,
    ValidationTimer,
    ValidatorMetrics,
    WindowMetrics,
)


class TestValidatorMetrics:
    """Test suite for ValidatorMetrics dataclass."""

    def test_pass_rate_calculation(self) -> None:
        """Pass rate is calculated correctly."""
        metrics = ValidatorMetrics(check_name="test")
        metrics.total_runs = 100
        metrics.passed = 80
        metrics.failed = 20

        assert metrics.pass_rate == 0.8
        assert metrics.fail_rate == 0.2

    def test_zero_runs_returns_zero_rate(self) -> None:
        """Zero runs returns 0.0 rate."""
        metrics = ValidatorMetrics(check_name="test")
        assert metrics.pass_rate == 0.0
        assert metrics.fail_rate == 0.0
        assert metrics.error_rate == 0.0

    def test_avg_time_calculation(self) -> None:
        """Average time is calculated correctly."""
        metrics = ValidatorMetrics(check_name="test")
        metrics.total_runs = 10
        metrics.total_time_ms = 1000.0

        assert metrics.avg_time_ms == 100.0

    def test_to_dict_serialization(self) -> None:
        """Metrics can be serialized to dict."""
        metrics = ValidatorMetrics(check_name="test")
        metrics.total_runs = 100
        metrics.passed = 80
        metrics.failed = 15
        metrics.errors = 5
        metrics.total_time_ms = 5000.0
        metrics.failure_reasons["reason1"] = 10
        metrics.failure_reasons["reason2"] = 5

        result = metrics.to_dict()

        assert result["check_name"] == "test"
        assert result["total_runs"] == 100
        assert result["passed"] == 80
        assert result["pass_rate"] == 0.8
        assert result["avg_time_ms"] == 50.0
        assert "reason1" in result["failure_reasons"]


class TestWindowMetrics:
    """Test suite for WindowMetrics dataclass."""

    def test_validation_rate_calculation(self) -> None:
        """Validation rate is calculated correctly."""
        metrics = WindowMetrics(window_start=100)
        metrics.total_rollouts = 100
        metrics.valid_rollouts = 75
        metrics.invalid_rollouts = 25

        assert metrics.validation_rate == 0.75

    def test_zero_rollouts_returns_zero_rate(self) -> None:
        """Zero rollouts returns 0.0 rate."""
        metrics = WindowMetrics(window_start=100)
        assert metrics.validation_rate == 0.0

    def test_to_dict_serialization(self) -> None:
        """Window metrics can be serialized to dict."""
        metrics = WindowMetrics(window_start=100)
        metrics.total_rollouts = 50
        metrics.valid_rollouts = 40
        metrics.invalid_rollouts = 10

        result = metrics.to_dict()

        assert result["window_start"] == 100
        assert result["total_rollouts"] == 50
        assert result["validation_rate"] == 0.8


class TestValidationMetricsAggregator:
    """Test suite for ValidationMetricsAggregator."""

    @pytest.fixture
    def aggregator(self) -> ValidationMetricsAggregator:
        """Create fresh aggregator for each test."""
        return ValidationMetricsAggregator()

    def test_record_single_validation(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Records single validation result correctly."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True, "proof_valid": True},
            timings={"schema_valid": 1.0, "proof_valid": 50.0},
            is_valid=True,
        )

        window = aggregator.get_window_metrics(100)
        assert window is not None
        assert window.total_rollouts == 1
        assert window.valid_rollouts == 1
        assert window.invalid_rollouts == 0

        # Check validator metrics
        assert "schema_valid" in window.validator_metrics
        schema_metrics = window.validator_metrics["schema_valid"]
        assert schema_metrics.total_runs == 1
        assert schema_metrics.passed == 1
        assert schema_metrics.total_time_ms == 1.0

    def test_record_multiple_validations(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Records multiple validation results correctly."""
        for i in range(10):
            aggregator.record_validation(
                window_start=100,
                miner_address=f"miner{i}",
                checks={"schema_valid": True, "proof_valid": i % 2 == 0},
                is_valid=i % 2 == 0,
            )

        window = aggregator.get_window_metrics(100)
        assert window is not None
        assert window.total_rollouts == 10
        assert window.valid_rollouts == 5
        assert window.invalid_rollouts == 5

        # Proof validator should have 50% pass rate
        proof_metrics = window.validator_metrics["proof_valid"]
        assert proof_metrics.pass_rate == 0.5

    def test_multiple_windows(self, aggregator: ValidationMetricsAggregator) -> None:
        """Handles multiple windows independently."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=150,
            miner_address="miner1",
            checks={"schema_valid": False},
            is_valid=False,
        )

        window_100 = aggregator.get_window_metrics(100)
        window_150 = aggregator.get_window_metrics(150)

        assert window_100 is not None
        assert window_150 is not None
        assert window_100.valid_rollouts == 1
        assert window_150.invalid_rollouts == 1

    def test_failure_reasons_tracked(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Tracks failure reasons from metadata."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": False},
            metadata={"schema_valid_failure": "missing_field"},
            is_valid=False,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner2",
            checks={"schema_valid": False},
            metadata={"schema_valid_failure": "missing_field"},
            is_valid=False,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner3",
            checks={"schema_valid": False},
            metadata={"schema_valid_failure": "type_error"},
            is_valid=False,
        )

        window = aggregator.get_window_metrics(100)
        assert window is not None
        schema_metrics = window.validator_metrics["schema_valid"]

        assert schema_metrics.failure_reasons["missing_field"] == 2
        assert schema_metrics.failure_reasons["type_error"] == 1

    def test_miner_stats_tracking(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Tracks per-miner statistics."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": False},
            is_valid=False,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner2",
            checks={"schema_valid": True},
            is_valid=True,
        )

        window = aggregator.get_window_metrics(100)
        assert window is not None
        assert window.miner_stats["miner1"]["valid"] == 1
        assert window.miner_stats["miner1"]["invalid"] == 1
        assert window.miner_stats["miner2"]["valid"] == 1

    def test_get_validator_summary(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Gets validator summary statistics."""
        for i in range(100):
            aggregator.record_validation(
                window_start=100,
                miner_address=f"miner{i}",
                checks={"schema_valid": i < 80, "proof_valid": i < 60},
                timings={"schema_valid": 1.0, "proof_valid": 50.0},
                is_valid=i < 60,
            )

        summary = aggregator.get_validator_summary()

        assert "schema_valid" in summary
        assert "proof_valid" in summary
        assert summary["schema_valid"]["pass_rate"] == 0.8
        assert summary["proof_valid"]["pass_rate"] == 0.6
        assert summary["schema_valid"]["avg_time_ms"] == 1.0
        assert summary["proof_valid"]["avg_time_ms"] == 50.0

    def test_get_miner_summary_single_window(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Gets miner summary for single window."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": False},
            is_valid=False,
        )

        summary = aggregator.get_miner_summary(window_start=100)

        assert "miner1" in summary
        assert summary["miner1"]["valid"] == 2
        assert summary["miner1"]["invalid"] == 1
        assert summary["miner1"]["total"] == 3
        assert summary["miner1"]["success_rate"] == 2 / 3

    def test_get_miner_summary_all_windows(
        self, aggregator: ValidationMetricsAggregator
    ) -> None:
        """Gets miner summary across all windows."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=150,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=200,
            miner_address="miner1",
            checks={"schema_valid": False},
            is_valid=False,
        )

        summary = aggregator.get_miner_summary()

        assert "miner1" in summary
        assert summary["miner1"]["valid"] == 2
        assert summary["miner1"]["invalid"] == 1
        assert summary["miner1"]["success_rate"] == 2 / 3

    def test_get_all_metrics(self, aggregator: ValidationMetricsAggregator) -> None:
        """Gets all aggregated metrics."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=150,
            miner_address="miner2",
            checks={"schema_valid": False},
            is_valid=False,
        )

        all_metrics = aggregator.get_all_metrics()

        assert "global_metrics" in all_metrics
        assert "window_metrics" in all_metrics
        assert "total_windows" in all_metrics
        assert "total_rollouts" in all_metrics
        assert all_metrics["total_windows"] == 2
        assert all_metrics["total_rollouts"] == 2

    def test_reset_window(self, aggregator: ValidationMetricsAggregator) -> None:
        """Resets metrics for specific window."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )

        aggregator.reset_window(100)
        window = aggregator.get_window_metrics(100)
        assert window is None

    def test_reset_all(self, aggregator: ValidationMetricsAggregator) -> None:
        """Resets all metrics."""
        aggregator.record_validation(
            window_start=100,
            miner_address="miner1",
            checks={"schema_valid": True},
            is_valid=True,
        )
        aggregator.record_validation(
            window_start=150,
            miner_address="miner2",
            checks={"schema_valid": False},
            is_valid=False,
        )

        aggregator.reset_all()

        all_metrics = aggregator.get_all_metrics()
        assert all_metrics["total_windows"] == 0
        assert all_metrics["total_rollouts"] == 0


class TestValidationTimer:
    """Test suite for ValidationTimer context manager."""

    def test_timer_measures_elapsed_time(self) -> None:
        """Timer measures elapsed time correctly."""
        import time

        with ValidationTimer() as timer:
            time.sleep(0.01)  # Sleep 10ms

        # Should be approximately 10ms (allow some variance)
        assert 5.0 <= timer.elapsed_ms <= 50.0

    def test_timer_can_be_used_multiple_times(self) -> None:
        """Timer can be reused."""
        timer = ValidationTimer()

        with timer:
            pass
        first_elapsed = timer.elapsed_ms

        with timer:
            pass
        second_elapsed = timer.elapsed_ms

        # Both should have valid measurements
        assert first_elapsed >= 0
        assert second_elapsed >= 0
