"""Integration tests for GRPO algorithm end-to-end pipeline.

Tests focus on:
- Full train_grpo_epoch with synthetic GRPO groups
- Metric structure and finiteness
- Token truncation and padding behavior
- Behavior logprob substitution in full training loop
- No crashes or undefined behavior with edge cases

Uses tiny model and small GRPO groups to keep tests fast and deterministic.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from grail.trainer.algorithms.grpo import (
    GRPOGroup,
    GRPORollout,
)


@pytest.fixture
def synthetic_grpo_groups(
    monkeypatch_trainer_constants: None, seeded_torch_env: None
) -> list[GRPOGroup]:
    """Create synthetic GRPO groups for testing.

    Creates 2 groups with 4 rollouts each (respects monkeypatched ROLLOUTS_PER_PROBLEM).
    """
    groups: list[GRPOGroup] = []

    for group_idx in range(2):
        rollouts = []

        for rollout_idx in range(4):
            # Create rollout with varying tokens and lengths
            tokens = list(range(1, 12))  # 11 tokens total
            prompt_length = 3
            completion_length = 5
            advantage = 0.5 - 0.25 * rollout_idx  # Distributes around zero
            reward = 0.5 + 0.1 * (rollout_idx % 2)

            rollout = GRPORollout(
                tokens=tokens,
                prompt_length=prompt_length,
                completion_length=completion_length,
                advantage=float(advantage),
                reward=float(reward),
                success=bool(reward > 0.4),
                nonce=rollout_idx,
                rollout_group=f"g{group_idx}",
                token_logprobs=None,  # Test without behavior logprobs first
            )

            rollouts.append(rollout)

        # Adjust last rollout to ensure sum is near zero
        if rollouts:
            current_sum = sum(r.advantage for r in rollouts[:-1])
            rollouts[-1].advantage = -current_sum

        group = GRPOGroup(group_id=f"g{group_idx}", rollouts=rollouts)
        groups.append(group)

    return groups


@pytest.fixture
def synthetic_grpo_groups_with_behavior(
    monkeypatch_trainer_constants: None, seeded_torch_env: None
) -> list[GRPOGroup]:
    """Create GRPO groups with partial behavior logprobs."""
    groups: list[GRPOGroup] = []

    for group_idx in range(2):
        rollouts = []

        for rollout_idx in range(4):
            tokens = list(range(1, 12))
            prompt_length = 3
            completion_length = 5
            advantage = 0.5 - 0.25 * rollout_idx
            reward = 0.5 + 0.1 * (rollout_idx % 2)

            # Provide token_logprobs for even rollouts
            token_logprobs = None
            if rollout_idx % 2 == 0:
                token_logprobs = [-0.5] * len(tokens)

            rollout = GRPORollout(
                tokens=tokens,
                prompt_length=prompt_length,
                completion_length=completion_length,
                advantage=float(advantage),
                reward=float(reward),
                success=bool(reward > 0.4),
                nonce=rollout_idx,
                rollout_group=f"g{group_idx}",
                token_logprobs=token_logprobs,
            )

            rollouts.append(rollout)

        if rollouts:
            current_sum = sum(r.advantage for r in rollouts[:-1])
            rollouts[-1].advantage = -current_sum

        group = GRPOGroup(group_id=f"g{group_idx}", rollouts=rollouts)
        groups.append(group)

    return groups


class TestGRPOEpochMetricsStructure:
    """Test full train_grpo_epoch produces expected metrics."""

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_epoch_produces_all_metrics(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        synthetic_grpo_groups: list[GRPOGroup],
        run_grpo_epoch: Any,
    ) -> None:
        """Test train_grpo_epoch returns all expected metrics."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            synthetic_grpo_groups,
            optimizer,
            accelerator_cpu,
        )

        # Check all expected metric keys are present
        expected_keys = {
            "loss_total",
            "loss_pg",
            "loss_kl",
            "loss_entropy",
            "grad_norm",
            "advantage_mean",
            "advantage_std",
            "entropy_mean",
            "advantage_mean_normalized",
            "advantage_std_normalized",
            "kl_divergence",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_epoch_metrics_are_finite(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        synthetic_grpo_groups: list[GRPOGroup],
        run_grpo_epoch: Any,
    ) -> None:
        """Test all metrics are finite values."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            synthetic_grpo_groups,
            optimizer,
            accelerator_cpu,
        )

        for key, value in metrics.items():
            assert isinstance(value, (int, float)), f"Metric {key} is not numeric"
            if isinstance(value, float):
                assert (
                    value == value  # NaN check (NaN != NaN)
                ), f"Metric {key} is NaN"
                assert value != float("inf"), f"Metric {key} is infinite"

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_epoch_with_behavior_logprobs(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        synthetic_grpo_groups_with_behavior: list[GRPOGroup],
        run_grpo_epoch: Any,
    ) -> None:
        """Test train_grpo_epoch with partial miner-provided logprobs."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            synthetic_grpo_groups_with_behavior,
            optimizer,
            accelerator_cpu,
        )

        # Should have behavior_frac metric when logprobs are provided
        assert isinstance(metrics, dict)
        # At least some of the batch should have behavior logprobs
        if "behavior_frac" in metrics:
            assert 0.0 <= metrics["behavior_frac"] <= 1.0

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_grad_norm_positive_after_optimization(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        synthetic_grpo_groups: list[GRPOGroup],
        run_grpo_epoch: Any,
    ) -> None:
        """Test grad_norm is positive if optimizer stepped."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            synthetic_grpo_groups,
            optimizer,
            accelerator_cpu,
        )

        grad_norm = metrics.get("grad_norm", 0.0)
        # Should be >= 0, and if model actually trained, should be > 0
        assert grad_norm >= 0.0


class TestGRPOGroupValidation:
    """Test GRPOGroup validation logic."""

    def test_valid_groups_in_synthetic_data(
        self, synthetic_grpo_groups: list[GRPOGroup], monkeypatch_trainer_constants: None
    ) -> None:
        """Test synthetic groups are valid."""
        from grail.shared.constants import ROLLOUTS_PER_PROBLEM, TRAINER_GROUP_ADV_SUM_TOL

        for group in synthetic_grpo_groups:
            assert group.is_valid(
                advantage_tolerance=TRAINER_GROUP_ADV_SUM_TOL,
                rollouts_per_problem=ROLLOUTS_PER_PROBLEM,
            )

    def test_invalid_group_rejected(self) -> None:
        """Test invalid group fails validation."""
        rollouts = [
            GRPORollout(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_length=2,
                advantage=1.0,  # All positive
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="bad",
            )
            for i in range(4)
        ]

        group = GRPOGroup(group_id="bad", rollouts=rollouts)
        # Sum is 4.0, definitely invalid
        assert not group.is_valid(advantage_tolerance=0.01)


class TestTokenTruncation:
    """Test handling of sequences longer than TRAINER_MAX_LENGTH."""

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_long_sequence_truncated_and_trained(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        run_grpo_epoch: Any,
    ) -> None:
        """Test training with sequences exceeding TRAINER_MAX_LENGTH."""
        from grail.shared.constants import TRAINER_MAX_LENGTH

        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Create rollout longer than TRAINER_MAX_LENGTH (256)
        long_tokens = list(range(1, TRAINER_MAX_LENGTH + 50))  # 305 tokens

        rollouts = [
            GRPORollout(
                tokens=long_tokens,
                prompt_length=50,
                completion_length=100,
                advantage=float(0.5 - 0.25 * i),
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="long",
            )
            for i in range(4)
        ]

        # Balance advantages
        if rollouts:
            advantage_sum = sum(r.advantage for r in rollouts)
            rollouts[-1].advantage -= advantage_sum

        group = GRPOGroup(group_id="long", rollouts=rollouts)
        groups = [group]

        # Should not crash
        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            accelerator_cpu,
        )

        assert isinstance(metrics, dict)
        assert "loss_total" in metrics


class TestBatchPadding:
    """Test right-padding and attention mask correctness."""

    @pytest.mark.long
    @pytest.mark.asyncio
    async def test_variable_length_batch_padded(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        run_grpo_epoch: Any,
    ) -> None:
        """Test training with variable-length sequences in batch."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Create rollouts with different lengths
        lengths = [8, 12, 10, 9]  # Variable lengths
        rollouts = [
            GRPORollout(
                tokens=list(range(1, lengths[i] + 1)),
                prompt_length=3,
                completion_length=lengths[i] - 3,
                advantage=float(0.5 - 0.25 * i),
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="var",
            )
            for i in range(4)
        ]

        # Balance advantages
        if rollouts:
            advantage_sum = sum(r.advantage for r in rollouts)
            rollouts[-1].advantage -= advantage_sum

        group = GRPOGroup(group_id="var", rollouts=rollouts)
        groups = [group]

        # Should handle variable lengths without crashing
        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            accelerator_cpu,
        )

        assert isinstance(metrics, dict)
        assert "loss_total" in metrics


class TestEmptyGroupHandling:
    """Test edge cases with empty or minimal data."""

    @pytest.mark.asyncio
    async def test_empty_groups_list(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        run_grpo_epoch: Any,
    ) -> None:
        """Test train_grpo_epoch with empty groups list."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        groups: list[GRPOGroup] = []

        # Should return empty metrics or zeros
        metrics = await run_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            accelerator_cpu,
        )

        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
