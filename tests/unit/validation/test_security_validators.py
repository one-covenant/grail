"""Tests for security-focused validators.

Tests timestamp validation, hidden state anomaly detection,
rate limiting, and model fingerprint validation.
"""

from __future__ import annotations

import time

import pytest
import torch

from grail.validation.context import ValidationContext
from grail.validation.validators.security import (
    HiddenStateAnomalyValidator,
    ModelFingerprintValidator,
    RateLimitValidator,
    TimestampValidator,
)


class TestTimestampValidator:
    """Test suite for TimestampValidator."""

    @pytest.fixture
    def validator(self) -> TimestampValidator:
        """Create validator instance."""
        return TimestampValidator()

    @pytest.fixture
    def mock_ctx(self) -> ValidationContext:
        """Create mock validation context."""
        ctx = ValidationContext(
            commit={"timestamp": time.time()},
            model=None,  # type: ignore
            tokenizer=None,  # type: ignore
            device="cpu",
            prover_address="test_miner",
            challenge_randomness="test_rand",
            window_hash="test_hash",
            group_index=0,
        )
        ctx.checks = {}
        ctx.metadata = {}
        return ctx

    def test_recent_timestamp_passes(
        self, validator: TimestampValidator, mock_ctx: ValidationContext
    ) -> None:
        """Recent timestamp passes validation."""
        mock_ctx.commit["timestamp"] = time.time()
        result = validator.validate(mock_ctx)
        assert result is True
        assert mock_ctx.checks["timestamp_valid"] is True

    def test_future_timestamp_fails(
        self, validator: TimestampValidator, mock_ctx: ValidationContext
    ) -> None:
        """Future timestamp fails validation."""
        mock_ctx.commit["timestamp"] = time.time() + 120  # 2 minutes in future
        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["timestamp_valid"] is False
        assert "future_timestamp" in mock_ctx.metadata.get("timestamp_failure", "")

    def test_stale_timestamp_fails(
        self, validator: TimestampValidator, mock_ctx: ValidationContext
    ) -> None:
        """Very old timestamp fails validation."""
        # Timestamp from 2 hours ago
        mock_ctx.commit["timestamp"] = time.time() - 7200
        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["timestamp_valid"] is False
        assert "stale_timestamp" in mock_ctx.metadata.get("timestamp_failure", "")

    def test_missing_timestamp_passes(
        self, validator: TimestampValidator, mock_ctx: ValidationContext
    ) -> None:
        """Missing timestamp is allowed (optional field)."""
        mock_ctx.commit.pop("timestamp", None)
        result = validator.validate(mock_ctx)
        assert result is True

    def test_invalid_timestamp_format_fails(
        self, validator: TimestampValidator, mock_ctx: ValidationContext
    ) -> None:
        """Invalid timestamp format fails validation."""
        mock_ctx.commit["timestamp"] = "not_a_number"
        result = validator.validate(mock_ctx)
        assert result is False
        assert "invalid_format" in mock_ctx.metadata.get("timestamp_failure", "")


class TestHiddenStateAnomalyValidator:
    """Test suite for HiddenStateAnomalyValidator."""

    @pytest.fixture
    def validator(self) -> HiddenStateAnomalyValidator:
        """Create validator instance."""
        return HiddenStateAnomalyValidator()

    @pytest.fixture
    def mock_ctx(self) -> ValidationContext:
        """Create mock validation context."""
        ctx = ValidationContext(
            commit={"tokens": list(range(100))},
            model=None,  # type: ignore
            tokenizer=None,  # type: ignore
            device="cpu",
            prover_address="test_miner",
            challenge_randomness="test_rand",
            window_hash="test_hash",
            group_index=0,
        )
        ctx.checks = {}
        ctx.metadata = {}
        return ctx

    def test_normal_logits_pass(
        self, validator: HiddenStateAnomalyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Normal logit statistics pass validation."""
        # Create normal logits
        mock_ctx.cached_logits = torch.randn(100, 32000) * 10  # Normal range
        result = validator.validate(mock_ctx)
        assert result is True
        assert mock_ctx.checks["hidden_state_anomaly_valid"] is True
        assert "logit_stats" in mock_ctx.metadata

    def test_high_norm_detected(
        self, validator: HiddenStateAnomalyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Detects unusually high logit norms."""
        # Create logits with very high norms
        mock_ctx.cached_logits = torch.randn(100, 32000) * 100  # Very high
        result = validator.validate(mock_ctx)
        assert result is False
        assert "hidden_state_anomalies" in mock_ctx.metadata
        anomalies = mock_ctx.metadata["hidden_state_anomalies"]
        assert any("high_max_norm" in a for a in anomalies)

    def test_low_norm_detected(
        self, validator: HiddenStateAnomalyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Detects unusually low logit norms."""
        # Create logits with very low norms (near-zero values)
        # Use mostly zeros with tiny values to get norm < 0.001
        mock_ctx.cached_logits = torch.randn(100, 32000) * 0.000001  # Extremely low
        result = validator.validate(mock_ctx)
        assert result is False
        assert "hidden_state_anomalies" in mock_ctx.metadata
        anomalies = mock_ctx.metadata["hidden_state_anomalies"]
        assert any("low_min_norm" in a for a in anomalies)

    def test_extreme_mean_detected(
        self, validator: HiddenStateAnomalyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Detects extreme mean activation."""
        # Create logits with extreme mean
        mock_ctx.cached_logits = torch.ones(100, 32000) * 100  # Extreme positive
        result = validator.validate(mock_ctx)
        assert result is False
        assert "hidden_state_anomalies" in mock_ctx.metadata

    def test_no_cached_logits_passes(
        self, validator: HiddenStateAnomalyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Passes when no cached logits available."""
        mock_ctx.cached_logits = None
        result = validator.validate(mock_ctx)
        assert result is True


class TestRateLimitValidator:
    """Test suite for RateLimitValidator."""

    @pytest.fixture
    def validator(self) -> RateLimitValidator:
        """Create validator instance."""
        # Clear class-level state before each test
        RateLimitValidator._submission_counts.clear()
        RateLimitValidator._last_cleanup = 0.0
        return RateLimitValidator()

    @pytest.fixture
    def mock_ctx(self) -> ValidationContext:
        """Create mock validation context."""
        ctx = ValidationContext(
            commit={},
            model=None,  # type: ignore
            tokenizer=None,  # type: ignore
            device="cpu",
            prover_address="test_miner_123",
            challenge_randomness="test_rand",
            window_hash="test_hash",
            group_index=0,
        )
        ctx.checks = {}
        ctx.metadata = {}
        ctx.window_start = 100
        return ctx

    def test_first_submission_passes(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """First submission from miner passes."""
        result = validator.validate(mock_ctx)
        assert result is True
        assert mock_ctx.checks["rate_limit_valid"] is True

    def test_normal_rate_passes(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """Normal submission rate passes."""
        # Submit 100 times (well below limit)
        for _ in range(100):
            result = validator.validate(mock_ctx)
            assert result is True

    def test_excessive_rate_fails(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """Excessive submission rate fails."""
        # Submit more than MAX_SUBMISSIONS_PER_WINDOW
        for i in range(validator.MAX_SUBMISSIONS_PER_WINDOW + 10):
            result = validator.validate(mock_ctx)
            if i < validator.MAX_SUBMISSIONS_PER_WINDOW:
                assert result is True
            else:
                assert result is False
                assert mock_ctx.checks["rate_limit_valid"] is False
                assert mock_ctx.metadata.get("rate_limit_exceeded") is True

    def test_different_miners_independent(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """Different miners have independent rate limits."""
        # Miner 1 submits many times
        mock_ctx.prover_address = "miner_1"
        for _ in range(500):
            validator.validate(mock_ctx)

        # Miner 2 should still pass
        mock_ctx.prover_address = "miner_2"
        result = validator.validate(mock_ctx)
        assert result is True

    def test_different_windows_independent(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """Different windows have independent rate limits."""
        # Submit many times in window 100
        mock_ctx.window_start = 100
        for _ in range(500):
            validator.validate(mock_ctx)

        # Window 150 should start fresh
        mock_ctx.window_start = 150
        result = validator.validate(mock_ctx)
        assert result is True

    def test_cleanup_old_data(
        self, validator: RateLimitValidator, mock_ctx: ValidationContext
    ) -> None:
        """Old submission data is cleaned up."""
        # Add old submission
        old_time = time.time() - 90000  # 25 hours ago
        validator._submission_counts["old_miner"].append((old_time, 0))

        # Trigger cleanup
        validator._last_cleanup = 0.0
        validator.validate(mock_ctx)

        # Old miner should be removed
        assert "old_miner" not in validator._submission_counts


class TestModelFingerprintValidator:
    """Test suite for ModelFingerprintValidator."""

    @pytest.fixture
    def validator(self) -> ModelFingerprintValidator:
        """Create validator instance."""
        return ModelFingerprintValidator()

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""

        class MockModel:
            name_or_path = "test-model"

        return MockModel()

    @pytest.fixture
    def mock_ctx(self, mock_model) -> ValidationContext:
        """Create mock validation context."""
        ctx = ValidationContext(
            commit={
                "model": {"name": "test-model"},
                "tokens": list(range(100)),
                "rollout": {"prompt_length": 50, "completion_length": 50},
            },
            model=mock_model,  # type: ignore
            tokenizer=None,  # type: ignore
            device="cpu",
            prover_address="test_miner",
            challenge_randomness="test_rand",
            window_hash="test_hash",
            group_index=0,
        )
        ctx.checks = {}
        ctx.metadata = {}
        return ctx

    def test_matching_model_passes(
        self, validator: ModelFingerprintValidator, mock_ctx: ValidationContext
    ) -> None:
        """Matching model name passes validation."""
        result = validator.validate(mock_ctx)
        assert result is True
        assert mock_ctx.checks["model_fingerprint_valid"] is True

    def test_mismatched_model_fails(
        self, validator: ModelFingerprintValidator, mock_ctx: ValidationContext
    ) -> None:
        """Mismatched model name fails validation."""
        mock_ctx.commit["model"]["name"] = "different-model"
        result = validator.validate(mock_ctx)
        assert result is False

    def test_low_token_uniqueness_flagged(
        self, validator: ModelFingerprintValidator, mock_ctx: ValidationContext
    ) -> None:
        """Low token uniqueness is flagged."""
        # Create repetitive completion
        mock_ctx.commit["tokens"] = [1] * 50 + [2] * 50  # Only 2 unique tokens
        mock_ctx.cached_logits = torch.randn(100, 32000)

        result = validator.validate(mock_ctx)
        assert result is True  # Not a hard failure
        assert mock_ctx.metadata.get("low_token_uniqueness") is True

    def test_high_token_uniqueness_passes(
        self, validator: ModelFingerprintValidator, mock_ctx: ValidationContext
    ) -> None:
        """High token uniqueness passes without flags."""
        # Create diverse completion
        mock_ctx.commit["tokens"] = list(range(100))  # All unique
        mock_ctx.cached_logits = torch.randn(100, 32000)

        result = validator.validate(mock_ctx)
        assert result is True
        assert "low_token_uniqueness" not in mock_ctx.metadata
