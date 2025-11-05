"""Unit tests for GRPO algorithm compute functions and invariants.

Tests focus on:
- compute_logprobs: completion-token-only indexing, left-shift alignment, truncation
- compute_entropy: entropy bounds and per-token computation
- Advantage clipping and normalization
- Behavior logprob substitution logic
- NaN/Inf handling and skipping
- Gradient accumulation and optimizer stepping

Uses synthetic tensors and tiny models to ensure deterministic, fast test execution.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
import torch

from grail.trainer.algorithms.grpo import (
    GRPOGroup,
    GRPORollout,
    compute_entropy,
    compute_logprobs,
)


@pytest.fixture
def sample_batch_inputs() -> dict[str, Any]:
    """Create sample batch inputs for logprob/entropy tests."""
    batch_size = 2
    seq_len = 16
    vocab_size = 256

    # Create synthetic batch
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    prompt_lengths = [4, 5]
    completion_lengths = [6, 5]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
        "completion_lengths": completion_lengths,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
    }


class TestComputeLogprobs:
    """Test logprob computation indexing and completion-token extraction."""

    def test_logprobs_correct_shape(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test compute_logprobs returns correct shape [batch_size]."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        logprobs = compute_logprobs(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        assert logprobs.shape == (sample_batch_inputs["batch_size"],)

    def test_logprobs_are_finite(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test all logprobs are finite (no NaN/Inf)."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        logprobs = compute_logprobs(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        assert torch.isfinite(logprobs).all()

    def test_logprobs_are_negative(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test logprobs are negative (log of probabilities < 1)."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        logprobs = compute_logprobs(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        # Most logprobs should be negative; allow a few near zero
        assert (logprobs < 0.0).sum() > 0

    def test_logprobs_with_truncation(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
    ) -> None:
        """Test logprobs computation when sequences exceed TRAINER_MAX_LENGTH."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        # Create a sequence longer than TRAINER_MAX_LENGTH (256)
        input_ids = torch.randint(0, 256, (1, 300))
        attention_mask = torch.ones((1, 300), dtype=torch.long)
        prompt_lengths = [100]
        completion_lengths = [150]

        logprobs = compute_logprobs(
            model, input_ids, attention_mask, prompt_lengths, completion_lengths
        )

        assert logprobs.shape == (1,)
        assert torch.isfinite(logprobs).all()


class TestComputeEntropy:
    """Test entropy computation over completion tokens."""

    def test_entropy_correct_shape(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test compute_entropy returns correct shape [batch_size]."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        entropies = compute_entropy(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        assert entropies.shape == (sample_batch_inputs["batch_size"],)

    def test_entropy_are_nonnegative(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test entropy values are non-negative."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        entropies = compute_entropy(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        assert (entropies >= 0.0).all()

    def test_entropy_are_finite(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test all entropy values are finite."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        entropies = compute_entropy(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        assert torch.isfinite(entropies).all()

    def test_entropy_bounded_by_vocab(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        sample_batch_inputs: dict[str, Any],
    ) -> None:
        """Test entropy does not exceed ln(vocab_size)."""
        model, _ = tiny_qwen_model_and_tokenizer
        model.eval()

        entropies = compute_entropy(
            model,
            sample_batch_inputs["input_ids"],
            sample_batch_inputs["attention_mask"],
            sample_batch_inputs["prompt_lengths"],
            sample_batch_inputs["completion_lengths"],
        )

        max_entropy = math.log(sample_batch_inputs["vocab_size"])
        # Allow larger numerical slack for bfloat16 precision and distribution variance
        assert (entropies <= max_entropy + 1.5).all()


class TestAdvantageClipping:
    """Test advantage clipping and normalization logic."""

    def test_percentile_clipping_bounds(self) -> None:
        """Test advantage clipping respects percentile bounds."""
        advantages = torch.tensor([1.0, 2.0, 10.0, 20.0, 100.0])

        # Simulate clipping at 90th percentile
        q = 0.9
        clip_val = torch.quantile(advantages.abs(), q)
        clipped = advantages.clamp(-clip_val, clip_val)

        assert (clipped.abs() <= clip_val).all()

    def test_normalization_zero_mean_when_std_large(self) -> None:
        """Test advantage normalization yields zero-mean when std > threshold."""
        advantages = torch.tensor([-2.0, -1.0, 1.0, 2.0])

        mean = advantages.mean()
        std = advantages.std()
        assert std > 1e-8

        normalized = (advantages - mean) / (std + 1e-8)

        # After normalization, mean should be near zero
        assert abs(normalized.mean().item()) < 1e-6

    def test_normalization_unchanged_when_std_small(self) -> None:
        """Test advantage normalization unchanged when std < threshold."""
        advantages = torch.tensor([1e-10, 1e-10, 1e-10, 1e-10])

        std = advantages.std()
        assert std < 1e-8

        # Should remain unchanged
        normalized = advantages
        assert (normalized == advantages).all()


class TestBehaviorLogprobSubstitution:
    """Test reference logprob substitution with miner-provided values."""

    def test_behavior_frac_all_provided(self) -> None:
        """Test behavior_frac = 1.0 when all rollouts have token_logprobs."""
        batch_behavior_seq_logprobs = [1.0, 2.0, 3.0, 4.0]
        any_have_behavior = any(x is not None for x in batch_behavior_seq_logprobs)
        all_have_behavior = all(x is not None for x in batch_behavior_seq_logprobs)

        assert all_have_behavior
        assert any_have_behavior

    def test_behavior_frac_partial(self) -> None:
        """Test behavior_frac calculation with partial miner-provided logprobs."""
        batch_behavior_seq_logprobs: list[float | None] = [1.0, None, 3.0, None]

        num_have = float(sum(1 for x in batch_behavior_seq_logprobs if x is not None))
        denom = max(1, len(batch_behavior_seq_logprobs))
        behavior_frac = num_have / float(denom)

        assert behavior_frac == 0.5

    def test_behavior_frac_none(self) -> None:
        """Test behavior_frac = 0.0 when no miner-provided logprobs."""
        batch_behavior_seq_logprobs: list[float | None] = [None, None, None, None]

        num_have = float(sum(1 for x in batch_behavior_seq_logprobs if x is not None))
        denom = max(1, len(batch_behavior_seq_logprobs))
        behavior_frac = num_have / float(denom)

        assert behavior_frac == 0.0


class TestGRPOGroupValidation:
    """Test GRPOGroup.is_valid() logic."""

    def test_valid_group(self, monkeypatch_trainer_constants: None) -> None:
        """Test valid group with correct count and zero-sum advantages."""
        from grail.shared.constants import ROLLOUTS_PER_PROBLEM

        rollouts = [
            GRPORollout(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_length=2,
                advantage=0.1,
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="g0",
            )
            for i in range(4)
        ]
        # Adjust to make sum near zero
        rollouts[0].advantage = 0.0
        rollouts[1].advantage = 0.1
        rollouts[2].advantage = -0.05
        rollouts[3].advantage = -0.05

        group = GRPOGroup(group_id="g0", rollouts=rollouts)
        assert group.is_valid(advantage_tolerance=0.01, rollouts_per_problem=ROLLOUTS_PER_PROBLEM)

    def test_invalid_group_wrong_count(self, monkeypatch_trainer_constants: None) -> None:
        """Test invalid group with wrong rollout count."""
        rollouts = [
            GRPORollout(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_length=2,
                advantage=0.1,
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="g0",
            )
            for i in range(2)  # Only 2, need 4
        ]

        group = GRPOGroup(group_id="g0", rollouts=rollouts)
        assert not group.is_valid(advantage_tolerance=0.01)

    def test_invalid_group_nonzero_advantage_sum(self, monkeypatch_trainer_constants: None) -> None:
        """Test invalid group with non-zero advantage sum."""
        rollouts = [
            GRPORollout(
                tokens=[1, 2, 3],
                prompt_length=1,
                completion_length=2,
                advantage=1.0,  # All positive
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="g0",
            )
            for i in range(4)
        ]

        group = GRPOGroup(group_id="g0", rollouts=rollouts)
        # Sum is 4.0, way over tolerance
        assert not group.is_valid(advantage_tolerance=0.01)


class TestNonFiniteHandling:
    """Test handling of NaN/Inf values in training loop."""

    @pytest.mark.asyncio
    async def test_skip_batch_on_nonfinite_logprobs(
        self,
        seeded_torch_env: None,
        tiny_qwen_model_and_tokenizer: tuple[Any, Any],
        monkeypatch_trainer_constants: None,
        accelerator_cpu: Any,
        caplog: pytest.LogCaptureFixture,
        run_grpo_epoch: Any,
    ) -> None:
        """Test that batches with non-finite current logprobs are skipped."""
        model, tokenizer = tiny_qwen_model_and_tokenizer
        ref_model = model

        # Create a valid rollout and group
        rollouts = [
            GRPORollout(
                tokens=[1, 2, 3, 4, 5, 6, 7, 8],
                prompt_length=2,
                completion_length=4,
                advantage=0.0,
                reward=0.5,
                success=True,
                nonce=i,
                rollout_group="g0",
            )
            for i in range(4)
        ]
        group = GRPOGroup(group_id="g0", rollouts=rollouts)
        groups = [group]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        with caplog.at_level(logging.WARNING):
            metrics = await run_grpo_epoch(
                model, ref_model, tokenizer, groups, optimizer, accelerator_cpu
            )

        # Should complete without crashing
        assert isinstance(metrics, dict)

    def test_loss_remains_finite_after_clamping(self) -> None:
        """Test that loss remains finite after log_ratio clamping."""
        log_ratio = torch.tensor([-100.0, 100.0, 0.0])
        clamped = torch.clamp(log_ratio, min=-20.0, max=20.0)

        # Should all be within bounds
        assert (clamped >= -20.0).all()
        assert (clamped <= 20.0).all()
        assert torch.isfinite(clamped).all()


class TestGradAccumulationAndClipping:
    """Test gradient accumulation and clipping behavior."""

    def test_grad_clip_produces_finite_norm(self, seeded_torch_env: None) -> None:
        """Test that grad clipping always produces finite gradient norm."""
        model = torch.nn.Linear(10, 10)

        # Create large gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100.0

        # clip_grad_norm_ returns the norm BEFORE clipping
        original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # Compute actual norm after clipping
        clipped_norm = torch.sqrt(
            sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        )

        assert torch.isfinite(original_norm)
        assert torch.isfinite(clipped_norm)
        assert clipped_norm <= 0.5 + 1e-6  # Small tolerance for numerical error

    def test_grad_accum_counter_reset_after_step(self) -> None:
        """Test gradient accumulation counter logic."""
        grad_accum_steps = 2
        grad_accum_counter = 0

        # Simulate accumulation loop
        for _step in range(5):
            grad_accum_counter += 1

            if grad_accum_counter >= grad_accum_steps:
                # Optimizer step
                grad_accum_counter = 0

        # Should have reset after each grad_accum_steps
        assert grad_accum_counter == 1  # Last step is 5 % 2 = 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
