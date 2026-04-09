"""Integration tests for the trainer→miner→validator generation_params flow.

These tests cover the bug fixed in Stage 1: the pipeline path was driving
the SGLang backend with ``GenerationParams.max_new_tokens = MAX_NEW_TOKENS_PROTOCOL_CAP``
(8192) regardless of the trainer-published checkpoint metadata, while the
validator's ``TerminationValidator`` caps at ``min(metadata.max_tokens,
MAX_NEW_TOKENS_PROTOCOL_CAP)``. The result was hard-rejection on every long completion.

The tests here exercise the full conversion chain end-to-end:

    trainer payload dict
        → CheckpointMetadata.generation_params
        → GenerationParams.from_checkpoint_metadata
        → environments.backends.GenerationParams
        → fed to TerminationValidator.validate via the same pathway

and verify that the value the miner drives the backend with is identical to
the value the validator's ``TerminationValidator`` will accept (no drift).

Wiring tests for ``SGLangServerBackend`` live in
``tests/unit/mining/test_weight_sync.py`` to keep the backend constructor
contract in one place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from grail.environments.backends import GenerationParams
from grail.infrastructure.checkpoint_consumer import CheckpointMetadata
from grail.protocol.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
from grail.protocol.errors import ProtocolViolationError
from grail.validation.context import ValidationContext
from grail.validation.validators.termination import TerminationValidator

# --------------------------------------------------------------------------- #
#         GenerationParams.from_checkpoint_metadata trust-boundary tests      #
# --------------------------------------------------------------------------- #


def _full_payload(**overrides: object) -> dict[str, object]:
    """Build a complete trainer-style payload with optional field overrides.

    The trust boundary requires every sampling field to be present;
    callers vary individual fields for failure tests by overriding here.
    Defaults match grail/trainer/checkpoint_publisher.py exactly.
    """
    payload: dict[str, object] = {
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
    }
    payload.update(overrides)
    return payload


def test_from_checkpoint_metadata_happy_path() -> None:
    """All trainer fields propagate to GenerationParams unchanged."""
    gp = GenerationParams.from_checkpoint_metadata(
        _full_payload(temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.05)
    )
    assert isinstance(gp, GenerationParams)
    assert gp.max_new_tokens == 2048
    assert gp.temperature == pytest.approx(0.7)
    assert gp.top_p == pytest.approx(0.9)
    assert gp.top_k == 20
    assert gp.repetition_penalty == pytest.approx(1.05)
    assert gp.do_sample is True


def test_from_checkpoint_metadata_caps_at_protocol_max() -> None:
    """A trainer publishing more than MAX_NEW_TOKENS_PROTOCOL_CAP is capped, never raised."""
    gp = GenerationParams.from_checkpoint_metadata(
        _full_payload(max_tokens=MAX_NEW_TOKENS_PROTOCOL_CAP * 4)
    )
    assert gp.max_new_tokens == MAX_NEW_TOKENS_PROTOCOL_CAP


@pytest.mark.parametrize(
    "missing_field",
    ["max_tokens", "temperature", "top_p", "top_k", "repetition_penalty"],
)
def test_from_checkpoint_metadata_missing_required_field_raises(missing_field: str) -> None:
    """Every sampling field is required at the trust boundary.

    Silent defaults would let a publisher bug ship a half-specified
    policy that the validator interprets differently. Each missing field
    must produce a ProtocolViolationError naming the field.
    """
    payload = _full_payload()
    payload.pop(missing_field)
    with pytest.raises(ProtocolViolationError, match=missing_field):
        GenerationParams.from_checkpoint_metadata(payload)


def test_from_checkpoint_metadata_zero_max_tokens_raises() -> None:
    """Non-positive max_tokens fails loud (catch publisher bugs early)."""
    with pytest.raises(ProtocolViolationError, match="must be > 0"):
        GenerationParams.from_checkpoint_metadata(_full_payload(max_tokens=0))


def test_from_checkpoint_metadata_top_k_zero_disables() -> None:
    """top_k=0 (sampling-API "disabled") becomes None, not 0."""
    gp = GenerationParams.from_checkpoint_metadata(_full_payload(top_k=0))
    assert gp.top_k is None


def test_from_checkpoint_metadata_non_dict_raises() -> None:
    """Non-dict generation_params payload is a protocol violation."""
    with pytest.raises(ProtocolViolationError, match="max_tokens"):
        GenerationParams.from_checkpoint_metadata([])  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_value", [-1, -2048, "abc", "2048.5"])
def test_from_checkpoint_metadata_rejects_negative_or_malformed(bad_value: object) -> None:
    """Negative ints and non-int strings hit the trust boundary loudly.

    Trainer publisher bugs (typo, env-var parsing slip) must not corrupt
    the miner's sampling policy silently.
    """
    with pytest.raises(ProtocolViolationError):
        GenerationParams.from_checkpoint_metadata(_full_payload(max_tokens=bad_value))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        # Type errors
        ("temperature", "abc"),
        ("top_p", "xyz"),
        ("top_k", "abc"),
        ("top_k", "1.5"),
        ("repetition_penalty", "abc"),
        # Range violations: temperature must be in [0.01, 2.0]
        ("temperature", 0.0),
        ("temperature", -0.5),
        ("temperature", 3.0),
        # Range violations: top_p must be in [0.0, 1.0]
        ("top_p", -0.1),
        ("top_p", 1.5),
        # Range violations: top_k must be in [0, 1000]
        ("top_k", -1),
        ("top_k", 1001),
        # Range violations: repetition_penalty must be in [1.0, 2.0]
        # (matches trainer publisher's range exactly to prevent contract drift)
        ("repetition_penalty", 0.0),
        ("repetition_penalty", -1.0),
        ("repetition_penalty", 0.5),
        ("repetition_penalty", 2.5),
    ],
)
def test_from_checkpoint_metadata_rejects_malformed_secondary_fields(
    field: str, bad_value: object
) -> None:
    """Non-max_tokens fields also fail loud with ProtocolViolationError.

    Without this hardening, bad values would raise raw ValueError/TypeError
    and bypass the miner's ProtocolViolationError skip-window branch, falling
    into the generic 10s retry loop instead. The accepted ranges match
    grail/trainer/checkpoint_publisher.py exactly so trainer/miner stay
    in lock-step.
    """
    payload = _full_payload(**{field: bad_value})
    with pytest.raises(ProtocolViolationError, match=field):
        GenerationParams.from_checkpoint_metadata(payload)


# --------------------------------------------------------------------------- #
#       Trainer-payload → CheckpointMetadata → GenerationParams round-trip    #
# --------------------------------------------------------------------------- #


def test_explicit_payload_round_trip() -> None:
    """An explicit non-default payload survives the metadata → builder chain.

    Uses an explicit dict (NOT ``get_default_generation_params``) so the
    test is hermetic against env-var defaults like ``GRAIL_GEN_MAX_TOKENS``,
    and so the regression for the 8192/2048 bug fails deterministically if
    the miner ever reverts to hardcoded ``MAX_NEW_TOKENS_PROTOCOL_CAP``.
    """
    payload = _full_payload(
        max_tokens=2048, temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.05
    )
    metadata = CheckpointMetadata(window=1000, file_manifest={}, generation_params=payload)
    gp = GenerationParams.from_checkpoint_metadata(metadata.generation_params)

    assert gp.max_new_tokens == 2048  # would FAIL if miner hardcodes MAX_NEW_TOKENS_PROTOCOL_CAP
    assert gp.temperature == pytest.approx(0.7)
    assert gp.top_p == pytest.approx(0.9)
    assert gp.top_k == 20
    assert gp.repetition_penalty == pytest.approx(1.05)


# --------------------------------------------------------------------------- #
#                   Miner ↔ TerminationValidator parity                       #
# --------------------------------------------------------------------------- #


def _make_termination_ctx(*, completion_length: int, max_tokens: int) -> ValidationContext:
    """Build the minimal ValidationContext that TerminationValidator needs.

    The max-length termination path returns True/False before touching
    tokenizer/model, so MagicMocks are sufficient for those resources.
    Tokens list is non-empty (the validator only checks ``not tokens``
    at the entrance, then compares ``completion_length`` for the cap path).
    """
    return ValidationContext(
        commit={
            "tokens": [1, 2, 3],
            "rollout": {"completion_length": completion_length},
        },
        prover_address="hotkey",
        challenge_randomness="00",
        model=MagicMock(),
        tokenizer=MagicMock(),
        device=torch.device("cpu"),
        generation_params={"max_tokens": max_tokens},
    )


@pytest.mark.parametrize(
    "trainer_max_tokens", [256, 2048, MAX_NEW_TOKENS_PROTOCOL_CAP, MAX_NEW_TOKENS_PROTOCOL_CAP * 2]
)
def test_miner_max_new_tokens_accepted_by_termination_validator(
    trainer_max_tokens: int,
) -> None:
    """Miner-driven max_new_tokens must be accepted by TerminationValidator.

    Drives the REAL ``TerminationValidator.validate`` (not a re-implementation
    of its clamp) with a completion of length equal to the miner's chosen
    cap. Catches BOTH miner drift AND validator drift, since either side
    diverging from the ``min(metadata.max_tokens, MAX_NEW_TOKENS_PROTOCOL_CAP)`` rule
    fails the assertion.
    """
    metadata = CheckpointMetadata(
        window=1000,
        file_manifest={},
        generation_params=_full_payload(max_tokens=trainer_max_tokens),
    )
    gp = GenerationParams.from_checkpoint_metadata(metadata.generation_params)

    ctx = _make_termination_ctx(
        completion_length=gp.max_new_tokens,
        max_tokens=trainer_max_tokens,
    )
    assert TerminationValidator().validate(ctx) is True
    assert ctx.checks["termination_valid"] is True


@pytest.mark.parametrize("trainer_max_tokens", [256, 2048])
def test_termination_validator_rejects_completion_above_cap(
    trainer_max_tokens: int,
) -> None:
    """One-token-over the cap must hard-fail termination_valid.

    This is the exact failure mode the Stage 1 bug produced (validator log
    line: ``Exceeds max tokens: 8192 > 2048``).
    """
    ctx = _make_termination_ctx(
        completion_length=trainer_max_tokens + 1,
        max_tokens=trainer_max_tokens,
    )
    assert TerminationValidator().validate(ctx) is False
    assert ctx.checks["termination_valid"] is False
