"""Tests for ConsistencyValidator.

Validates cross-validator consistency checks catch edge cases
where individual validators pass but results are inconsistent.
"""

from __future__ import annotations

import pytest
import torch

from grail.validation.context import ValidationContext
from grail.validation.validators.consistency import ConsistencyValidator


class TestConsistencyValidator:
    """Test suite for ConsistencyValidator."""

    @pytest.fixture
    def validator(self) -> ConsistencyValidator:
        """Create validator instance."""
        return ConsistencyValidator()

    @pytest.fixture
    def base_commit(self) -> dict:
        """Create valid base commit for testing."""
        return {
            "tokens": list(range(100)),  # 100 tokens
            "commitments": [{"sketch_hash": f"hash_{i}"} for i in range(100)],
            "rollout": {
                "prompt_length": 50,
                "completion_length": 50,
                "token_logprobs": [0.0] * 50 + [-1.5] * 50,  # Zeros for prompt
                "total_reward": 1.0,
            },
            "model": {"name": "test-model"},
        }

    @pytest.fixture
    def mock_ctx(self, base_commit: dict) -> ValidationContext:
        """Create mock validation context."""
        ctx = ValidationContext(
            commit=base_commit,
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

    def test_valid_commit_passes(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Valid commit with consistent data passes all checks."""
        result = validator.validate(mock_ctx)
        assert result is True
        assert mock_ctx.checks["consistency_valid"] is True

    def test_token_count_mismatch_fails(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Fails when token count doesn't match prompt + completion."""
        # Claim 60 tokens but only have 100
        mock_ctx.commit["rollout"]["prompt_length"] = 60
        mock_ctx.commit["rollout"]["completion_length"] = 60
        # tokens list still has 100 items

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "token_count_mismatch" in mock_ctx.metadata.get("consistency_failure", "")

    def test_commitment_length_mismatch_fails(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Fails when commitment array length doesn't match tokens."""
        # Remove some commitments
        mock_ctx.commit["commitments"] = mock_ctx.commit["commitments"][:80]

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "commitment_length_mismatch" in mock_ctx.metadata.get(
            "consistency_failure", ""
        )

    def test_logprob_length_mismatch_fails(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Fails when logprobs array length doesn't match tokens."""
        # Wrong logprobs length
        mock_ctx.commit["rollout"]["token_logprobs"] = [-1.5] * 80

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "logprobs_length_mismatch" in mock_ctx.metadata.get(
            "consistency_failure", ""
        )

    def test_logprob_wrong_type_fails(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Fails when logprobs is not a list."""
        mock_ctx.commit["rollout"]["token_logprobs"] = "not_a_list"

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "logprobs_wrong_type" in mock_ctx.metadata.get("consistency_failure", "")

    def test_completion_too_short_fails(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Fails when completion is shorter than CHALLENGE_K."""
        # Set completion to 10 tokens (< CHALLENGE_K=16)
        mock_ctx.commit["rollout"]["completion_length"] = 10
        mock_ctx.commit["rollout"]["prompt_length"] = 90
        mock_ctx.commit["rollout"]["token_logprobs"] = [0.0] * 90 + [-1.5] * 10

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "completion_too_short" in mock_ctx.metadata.get("consistency_failure", "")

    def test_reward_consistency_check(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Checks reward consistency when env evaluation is present."""
        # Add environment evaluation result
        mock_ctx.metadata["env_eval_result"] = {"reward": 1.0}
        mock_ctx.commit["rollout"]["total_reward"] = 1.0

        result = validator.validate(mock_ctx)
        assert result is True

        # Test sign mismatch (warning but not hard failure)
        mock_ctx.metadata["env_eval_result"] = {"reward": 1.0}
        mock_ctx.commit["rollout"]["total_reward"] = -1.0

        result = validator.validate(mock_ctx)
        # Should still pass (RewardValidator handles this)
        assert result is True
        assert "reward_sign_mismatch" in mock_ctx.metadata.get("consistency_warning", "")

    def test_non_zero_prompt_logprobs_warning(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Warns when prompt logprobs are non-zero."""
        # Set some prompt logprobs to non-zero
        mock_ctx.commit["rollout"]["token_logprobs"] = [-0.5] * 50 + [-1.5] * 50

        result = validator.validate(mock_ctx)
        assert result is True  # Not a hard failure
        assert "prompt_logprobs_non_zero" in mock_ctx.metadata

    def test_high_logprob_mismatch_warning(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Warns when logprob mismatch rate is high."""
        # Simulate high mismatch rate from LogprobValidator
        mock_ctx.metadata["logprob_mismatches"] = 10
        mock_ctx.metadata["logprob_total"] = 16

        result = validator.validate(mock_ctx)
        assert result is True  # Not a hard failure
        assert "high_logprob_mismatch_rate" in mock_ctx.metadata.get(
            "consistency_warning", ""
        )

    def test_extreme_low_probability_warning(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Warns when extremely low token probabilities detected."""
        # Simulate distribution metrics with extreme low probability
        mock_ctx.metadata["distribution_metrics"] = {
            "min": 1e-15,
            "mean": 0.5,
            "median": 0.5,
        }

        result = validator.validate(mock_ctx)
        assert result is True  # Not a hard failure
        assert "extreme_low_probability" in mock_ctx.metadata.get(
            "consistency_warning", ""
        )

    def test_empty_commitments_allowed(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Empty commitments are allowed (optional field)."""
        mock_ctx.commit["commitments"] = []

        result = validator.validate(mock_ctx)
        assert result is True

    def test_missing_env_eval_allowed(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Missing env evaluation result is allowed."""
        # Don't add env_eval_result to metadata
        result = validator.validate(mock_ctx)
        assert result is True

    def test_exception_handling(
        self, validator: ConsistencyValidator, mock_ctx: ValidationContext
    ) -> None:
        """Handles exceptions gracefully."""
        # Corrupt data that will cause exception
        mock_ctx.commit["rollout"] = None

        result = validator.validate(mock_ctx)
        assert result is False
        assert mock_ctx.checks["consistency_valid"] is False
        assert "consistency_error" in mock_ctx.metadata
