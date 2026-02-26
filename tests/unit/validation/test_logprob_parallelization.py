"""Unit tests for batched log-softmax precomputation and validator usage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from grail.protocol.crypto import indices_from_root_in_range
from grail.shared.constants import CHALLENGE_K
from grail.validation.context import ValidationContext
from grail.validation.miner_validator import MinerValidator
from grail.validation.validators.environment import LogprobValidator

WINDOW_RAND = "a1" * 32


def _make_rollout(
    *,
    vocab_size: int = 97,
    prompt_len: int = 8,
    completion_len: int = CHALLENGE_K + 4,
) -> tuple[list[int], torch.Tensor, int, int]:
    """Build deterministic rollout tokens and logits for tests."""
    seq_len = prompt_len + completion_len
    tokens = [((i * 13) + 7) % vocab_size for i in range(seq_len)]
    generator = torch.Generator().manual_seed(1234)
    logits = torch.randn(seq_len, vocab_size, generator=generator, dtype=torch.float32)
    return tokens, logits, prompt_len, completion_len


def _make_context(
    *,
    tokens: list[int],
    logits: torch.Tensor,
    prompt_len: int,
    completion_len: int,
    claimed_logprobs: list[float],
    precomputed_logprobs: dict[int, float] | None,
) -> ValidationContext:
    """Create a lightweight validation context for LogprobValidator tests."""
    model_stub = SimpleNamespace(name_or_path="test-model", device=torch.device("cpu"))
    tokenizer_stub = SimpleNamespace()
    commit = {
        "tokens": tokens,
        "rollout": {
            "prompt_length": prompt_len,
            "completion_length": completion_len,
            "token_logprobs": claimed_logprobs,
        },
    }
    return ValidationContext(
        commit=commit,
        prover_address="test-hotkey",
        challenge_randomness=WINDOW_RAND,
        model=model_stub,  # type: ignore[arg-type]
        tokenizer=tokenizer_stub,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        cached_logits=logits,
        precomputed_logprobs=precomputed_logprobs,
    )


def _compute_challenged_logprobs(
    *,
    tokens: list[int],
    logits: torch.Tensor,
    prompt_len: int,
    completion_len: int,
) -> dict[int, float]:
    """Compute per-position reference logprobs using scalar fallback logic."""
    challenged = indices_from_root_in_range(
        tokens,
        WINDOW_RAND,
        prompt_len,
        prompt_len + completion_len,
        CHALLENGE_K,
    )
    return {
        abs_idx: float(torch.log_softmax(logits[abs_idx - 1], dim=-1)[tokens[abs_idx]].item())
        for abs_idx in challenged
    }


def test_batched_precompute_matches_per_position_log_softmax() -> None:
    """Batched challenged logprobs should match original scalar computation."""
    miner_validator = object.__new__(MinerValidator)
    tokens, logits, prompt_len, completion_len = _make_rollout()
    inferences = [
        {
            "commit": {
                "tokens": tokens,
                "rollout": {
                    "prompt_length": prompt_len,
                    "completion_length": completion_len,
                },
            }
        }
    ]
    batched_cache = {0: (torch.empty(0), logits)}

    precomputed = miner_validator._precompute_challenged_logprobs(
        inferences=inferences,
        indices_to_check=[0],
        batched_cache=batched_cache,
        window_rand=WINDOW_RAND,
    )

    assert 0 in precomputed
    expected = _compute_challenged_logprobs(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )
    assert set(precomputed[0].keys()) == set(expected.keys())
    for abs_idx, expected_lp in expected.items():
        assert precomputed[0][abs_idx] == pytest.approx(expected_lp, abs=1e-7, rel=0.0)


def test_batched_precompute_skips_missing_cache_and_short_rollouts() -> None:
    """Precompute should only return rollouts with cache and enough completion tokens."""
    miner_validator = object.__new__(MinerValidator)

    tokens_ok, logits_ok, prompt_ok, completion_ok = _make_rollout()
    tokens_short, logits_short, prompt_short, _ = _make_rollout(completion_len=CHALLENGE_K - 1)

    inferences = [
        {
            "commit": {
                "tokens": tokens_ok,
                "rollout": {"prompt_length": prompt_ok, "completion_length": completion_ok},
            }
        },
        {
            "commit": {
                "tokens": tokens_ok,
                "rollout": {"prompt_length": prompt_ok, "completion_length": completion_ok},
            }
        },
        {
            "commit": {
                "tokens": tokens_short,
                "rollout": {"prompt_length": prompt_short, "completion_length": CHALLENGE_K - 1},
            }
        },
    ]
    batched_cache = {
        0: (torch.empty(0), logits_ok),
        2: (torch.empty(0), logits_short),
    }

    precomputed = miner_validator._precompute_challenged_logprobs(
        inferences=inferences,
        indices_to_check=[0, 1, 2],
        batched_cache=batched_cache,
        window_rand=WINDOW_RAND,
    )

    assert set(precomputed.keys()) == {0}
    assert len(precomputed[0]) == CHALLENGE_K


def test_logprob_validator_uses_precomputed_branch_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validator should pass using precomputed values even if fallback is unavailable."""
    tokens, logits, prompt_len, completion_len = _make_rollout()
    challenged_logprobs = _compute_challenged_logprobs(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )

    claimed = [0.0 for _ in tokens]
    for abs_idx, lp in challenged_logprobs.items():
        claimed[abs_idx] = lp

    ctx = _make_context(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
        claimed_logprobs=claimed,
        precomputed_logprobs=challenged_logprobs,
    )

    def _fail_log_softmax(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise AssertionError("fallback log_softmax should not be called")

    monkeypatch.setattr(torch, "log_softmax", _fail_log_softmax)

    validator = LogprobValidator()
    assert validator.validate(ctx) is True


def test_logprob_validator_precomputed_matches_fallback_path() -> None:
    """Precomputed and fallback paths should produce identical validation result."""
    tokens, logits, prompt_len, completion_len = _make_rollout()
    challenged_logprobs = _compute_challenged_logprobs(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
    )

    claimed = [0.0 for _ in tokens]
    for abs_idx, lp in challenged_logprobs.items():
        claimed[abs_idx] = lp

    fallback_ctx = _make_context(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
        claimed_logprobs=claimed,
        precomputed_logprobs=None,
    )
    precomputed_ctx = _make_context(
        tokens=tokens,
        logits=logits,
        prompt_len=prompt_len,
        completion_len=completion_len,
        claimed_logprobs=claimed,
        precomputed_logprobs=challenged_logprobs,
    )

    validator = LogprobValidator()
    fallback_ok = validator.validate(fallback_ctx)
    precomputed_ok = validator.validate(precomputed_ctx)

    assert fallback_ok is True
    assert precomputed_ok is True
    assert fallback_ctx.metadata["logprob_total"] == precomputed_ctx.metadata["logprob_total"]
    assert (
        fallback_ctx.metadata["logprob_mismatches"]
        == precomputed_ctx.metadata["logprob_mismatches"]
    )
