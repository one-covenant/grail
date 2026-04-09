"""Unit tests for WindowEnvConfig + resolve_window_env_config.

The trainer is the single source of truth for env_id, env_params,
generation_params, and thinking_mode. The validator resolves all four ONCE
per window from the matching CheckpointMetadata, raising
MissingCheckpointMetadataError on any failure so the WHOLE window is skipped
(never silently scoring miners against an unverifiable baseline).

These tests pin that contract.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from grail.infrastructure.checkpoint_consumer import CheckpointMetadata
from grail.validation.miner_validator import (
    MinerValidator,
    MissingCheckpointMetadataError,
    WindowEnvConfig,
)
from grail.validation.pipeline import create_env_validation_pipeline

# --------------------------------------------------------------------------- #
#                              fixtures                                       #
# --------------------------------------------------------------------------- #


def _full_metadata(**overrides: Any) -> CheckpointMetadata:
    """Return a CheckpointMetadata with all required fields populated."""
    payload: dict[str, Any] = {
        "window": 1234,
        "file_manifest": {},
        "env_id": "mbpp",
        "env_params": {"split": "train"},
        "generation_params": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
        },
        "thinking_mode": "instructed",
    }
    payload.update(overrides)
    return CheckpointMetadata(**payload)


def _make_validator(
    metadata: CheckpointMetadata | None, raises: Exception | None = None
) -> MinerValidator:
    """Build a MinerValidator with a stubbed checkpoint manager."""
    pipeline = create_env_validation_pipeline()
    cm = MagicMock()
    if raises is not None:
        cm.get_checkpoint_metadata = AsyncMock(side_effect=raises)
    else:
        cm.get_checkpoint_metadata = AsyncMock(return_value=metadata)
    return MinerValidator(pipeline, checkpoint_manager=cm)


def _make_model(checkpoint_window: int | None = 1234) -> MagicMock:
    """Stub of the validator-loaded model with a grail_checkpoint_window attr."""
    model = MagicMock()
    if checkpoint_window is None:
        # Use spec to make sure the attribute lookup truly returns None.
        del model.grail_checkpoint_window
        return model
    model.grail_checkpoint_window = checkpoint_window
    return model


# --------------------------------------------------------------------------- #
#                       happy path: full round-trip                           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_returns_window_env_config_with_all_fields() -> None:
    """A complete CheckpointMetadata round-trips into WindowEnvConfig."""
    metadata = _full_metadata()
    validator = _make_validator(metadata)
    model = _make_model()

    config = await validator.resolve_window_env_config(model)

    assert isinstance(config, WindowEnvConfig)
    assert config.env_id == "mbpp"
    assert config.env_params == {"split": "train"}
    assert config.generation_params["max_tokens"] == 2048
    assert config.thinking_mode == "instructed"


@pytest.mark.asyncio
async def test_resolve_overrides_grail_thinking_mode_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_window_env_config sets os.environ['GRAIL_THINKING_MODE'].

    This is the trust boundary that protects the validator from local-host
    drift: even if the docker container booted with the wrong default, the
    checkpoint-derived value wins for all subsequent prompt rendering.
    """
    monkeypatch.setenv("GRAIL_THINKING_MODE", "native")  # wrong/stale value
    metadata = _full_metadata(thinking_mode="instructed")
    validator = _make_validator(metadata)
    model = _make_model()

    await validator.resolve_window_env_config(model)

    assert os.environ["GRAIL_THINKING_MODE"] == "instructed"


@pytest.mark.asyncio
async def test_resolve_does_not_touch_env_var_when_already_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No spurious env-var write when the host already matches the checkpoint."""
    monkeypatch.setenv("GRAIL_THINKING_MODE", "instructed")
    metadata = _full_metadata(thinking_mode="instructed")
    validator = _make_validator(metadata)
    model = _make_model()

    await validator.resolve_window_env_config(model)

    assert os.environ["GRAIL_THINKING_MODE"] == "instructed"


# --------------------------------------------------------------------------- #
#                  failure modes: every branch must raise                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resolve_raises_when_checkpoint_manager_is_none() -> None:
    pipeline = create_env_validation_pipeline()
    validator = MinerValidator(pipeline, checkpoint_manager=None)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="checkpoint_manager"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_model_has_no_checkpoint_window() -> None:
    validator = _make_validator(_full_metadata())
    model = MagicMock(spec=[])  # no grail_checkpoint_window attr

    with pytest.raises(MissingCheckpointMetadataError, match="grail_checkpoint_window"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_metadata_fetch_throws() -> None:
    validator = _make_validator(None, raises=RuntimeError("R2 down"))
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="Failed to fetch"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_metadata_is_none() -> None:
    validator = _make_validator(None)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="No checkpoint metadata"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_thinking_mode_missing() -> None:
    metadata = _full_metadata(thinking_mode=None)
    validator = _make_validator(metadata)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="thinking_mode"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_thinking_mode_is_invalid_value() -> None:
    metadata = _full_metadata(thinking_mode="bogus")
    validator = _make_validator(metadata)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="thinking_mode"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_env_id_missing() -> None:
    metadata = _full_metadata(env_id=None)
    validator = _make_validator(metadata)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="env_id"):
        await validator.resolve_window_env_config(model)


@pytest.mark.asyncio
async def test_resolve_raises_when_generation_params_missing() -> None:
    metadata = _full_metadata(generation_params={})
    validator = _make_validator(metadata)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError, match="generation_params"):
        await validator.resolve_window_env_config(model)


# --------------------------------------------------------------------------- #
#                       WindowEnvConfig is frozen                             #
# --------------------------------------------------------------------------- #


def test_window_env_config_is_frozen() -> None:
    """Resolved values cannot drift mid-window."""
    cfg = WindowEnvConfig(
        env_id="mbpp",
        env_params={"split": "train"},
        generation_params={"max_tokens": 2048},
        thinking_mode="instructed",
    )
    with pytest.raises(AttributeError):
        cfg.thinking_mode = "native"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_resolve_does_not_mutate_env_var_when_metadata_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing resolve must NOT touch GRAIL_THINKING_MODE.

    The env-var override only fires after validation succeeds. A test that
    pins this contract prevents a future refactor from accidentally writing
    a half-validated value into process state.
    """
    monkeypatch.setenv("GRAIL_THINKING_MODE", "native")
    metadata = _full_metadata(thinking_mode=None)  # invalid
    validator = _make_validator(metadata)
    model = _make_model()

    with pytest.raises(MissingCheckpointMetadataError):
        await validator.resolve_window_env_config(model)

    # Env var unchanged.
    assert os.environ["GRAIL_THINKING_MODE"] == "native"


# --------------------------------------------------------------------------- #
#                  Cache poisoning regression (Claude H4)                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_invalid_metadata_is_not_cached() -> None:
    """A legacy/invalid R2 payload must NOT poison the metadata cache.

    Otherwise a one-time bad publish would force a process restart to
    recover. The fix in checkpoint_consumer.py builds the metadata, runs
    validate_metadata(), and refuses to cache anything that fails.
    """
    from unittest.mock import AsyncMock

    from grail.infrastructure.checkpoint_consumer import (
        CHECKPOINT_TYPE_FULL,
        CheckpointManager,
    )

    # Build a manager with mocked R2 fetch
    manager = CheckpointManager.__new__(CheckpointManager)
    manager._metadata_cache = {}  # type: ignore[attr-defined]
    manager.credentials = MagicMock()  # type: ignore[attr-defined]

    # First fetch: legacy payload missing thinking_mode → returned but NOT cached
    legacy_payload = {
        "window": 7926000,
        "file_manifest": {},
        "env_id": "mbpp",
        "env_params": {"split": "train"},
        "generation_params": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
        },
        # NO thinking_mode
    }
    fixed_payload = {**legacy_payload, "thinking_mode": "instructed"}

    # Patch comms.get_file to return legacy first, then fixed
    fetch_results = [legacy_payload, fixed_payload]

    async def fake_get_file(*args: object, **kwargs: object) -> dict[str, object]:
        return fetch_results.pop(0)

    import grail.infrastructure.checkpoint_consumer as cc_module

    original = cc_module.comms.get_file
    cc_module.comms.get_file = AsyncMock(side_effect=fake_get_file)
    try:
        # First fetch: legacy payload returned, NOT cached (validate fails)
        m1 = await manager._fetch_full_metadata(7926000)
        assert m1 is not None
        assert m1.thinking_mode is None  # legacy
        assert (7926000, CHECKPOINT_TYPE_FULL) not in manager._metadata_cache  # type: ignore[attr-defined]

        # Second fetch: trainer republished with the fix → cached
        m2 = await manager._fetch_full_metadata(7926000)
        assert m2 is not None
        assert m2.thinking_mode == "instructed"
        assert (7926000, CHECKPOINT_TYPE_FULL) in manager._metadata_cache  # type: ignore[attr-defined]
    finally:
        cc_module.comms.get_file = original
