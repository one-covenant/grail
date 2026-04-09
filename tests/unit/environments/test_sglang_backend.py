"""Unit tests for SGLangServerBackend's context-length safeguard.

The ``_cap_max_tokens`` helper decides whether a (prompt, max_new_tokens)
pair fits the configured model context length. The contract is binary:
either fit (return ``max_new_tokens`` unchanged) or skip (return ``0``).

We deliberately do NOT return a partially clamped value: the validator's
``TerminationValidator`` requires ``completion_length == min(metadata.max_tokens,
protocol_cap)`` or EOS, so a clamped intermediate completion would
hard-fail ``termination_valid``. Returning ``0`` lets the per-group
``CHALLENGE_K`` gate drop the whole group (skip-iteration crash domain).
"""

from __future__ import annotations

import pytest

from grail.environments.backends import SGLangServerBackend


def _backend(max_model_len: int | None = 12288) -> SGLangServerBackend:
    return SGLangServerBackend(
        base_url="http://localhost:30000",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer=None,
        max_model_len=max_model_len,
    )


def test_cap_max_tokens_passthrough_when_fits() -> None:
    """Prompt + max_new fits comfortably: return max_new unchanged."""
    backend = _backend(max_model_len=12288)
    assert backend._cap_max_tokens(prompt_len=1000, max_new_tokens=2048) == 2048  # noqa: SLF001
    assert backend._cap_max_tokens(prompt_len=10239, max_new_tokens=2048) == 2048  # noqa: SLF001


def test_cap_max_tokens_returns_zero_at_exact_boundary_overflow() -> None:
    """Exactly one token over: return 0 so the caller skips the prompt.

    Regression for the validator-rejection bug: previously this returned
    a partially clamped value (e.g. 2047), the SGLang server stopped at
    that intermediate length, and the validator hard-rejected the rollout
    because completion_length != min(metadata.max_tokens, protocol_cap).
    """
    backend = _backend(max_model_len=12288)
    # 12288 - 10241 = 2047 room < requested 2048 → skip
    assert backend._cap_max_tokens(prompt_len=10241, max_new_tokens=2048) == 0  # noqa: SLF001


def test_cap_max_tokens_returns_zero_when_prompt_dominates() -> None:
    """Prompt eats most of the context: return 0, do not generate a stub."""
    backend = _backend(max_model_len=12288)
    assert backend._cap_max_tokens(prompt_len=11000, max_new_tokens=2048) == 0  # noqa: SLF001


def test_cap_max_tokens_returns_zero_when_prompt_already_too_long() -> None:
    """Prompt alone exceeds max_model_len: zero room means skip."""
    backend = _backend(max_model_len=12288)
    assert backend._cap_max_tokens(prompt_len=12500, max_new_tokens=2048) == 0  # noqa: SLF001
    assert backend._cap_max_tokens(prompt_len=12288, max_new_tokens=1) == 0  # noqa: SLF001


def test_cap_max_tokens_unset_max_model_len_is_passthrough() -> None:
    """Without max_model_len configured, the helper trusts the caller.

    This is the legacy behaviour for tests that construct a backend
    without going through the mining-path wiring; the live mining path
    always provides max_model_len from PipelineConfig.
    """
    backend = _backend(max_model_len=None)
    assert backend._cap_max_tokens(prompt_len=99999, max_new_tokens=2048) == 2048  # noqa: SLF001


@pytest.mark.parametrize(
    "prompt_len,max_new_tokens,max_model_len,expected",
    [
        # Comfortable fit
        (500, 2048, 12288, 2048),
        # Exactly fits (no headroom)
        (10240, 2048, 12288, 2048),
        # One token over
        (10241, 2048, 12288, 0),
        # Larger model context
        (10000, 4096, 16384, 4096),
        (12289, 4096, 16384, 0),
    ],
)
def test_cap_max_tokens_parametrized(
    prompt_len: int, max_new_tokens: int, max_model_len: int, expected: int
) -> None:
    """Decision boundary across realistic (prompt, max_new) combinations."""
    backend = _backend(max_model_len=max_model_len)
    assert backend._cap_max_tokens(prompt_len, max_new_tokens) == expected  # noqa: SLF001
