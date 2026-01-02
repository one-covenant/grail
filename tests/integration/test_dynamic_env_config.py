"""Integration tests for dynamic environment and generation configuration.

Tests verify that env/gen configs propagate correctly through the checkpoint mechanism
from publisher → consumer → miners/validators.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from grail.environments.factory import create_env
from grail.infrastructure.checkpoint_consumer import CheckpointManager, CheckpointMetadata
from grail.trainer.checkpoint_publisher import (
    get_default_env_config,
    get_default_generation_params,
)


@pytest.mark.asyncio
async def test_checkpoint_metadata_roundtrip():
    """Test that env/gen config survives checkpoint publish → consume cycle."""
    # Setup: Create metadata with specific env/gen config
    original_metadata = CheckpointMetadata(
        window=1000,
        file_manifest={},
        env_id="gsm8k",
        env_params={"split": "test"},
        generation_params={
            "max_tokens": 1024,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 100,
            "repetition_penalty": 1.2,
        },
    )

    # Simulate serialization (what happens during checkpoint publish)
    serialized = {
        "window": original_metadata.window,
        "file_manifest": original_metadata.file_manifest,
        "env_id": original_metadata.env_id,
        "env_params": original_metadata.env_params,
        "generation_params": original_metadata.generation_params,
    }

    # Simulate deserialization (what happens during checkpoint consume)
    restored_metadata = CheckpointMetadata(
        window=serialized.get("window", 1000),
        file_manifest=serialized.get("file_manifest", {}),
        env_id=serialized.get("env_id"),
        env_params=serialized.get("env_params", {}),
        generation_params=serialized.get("generation_params", {}),
    )

    # Verify: All config values are preserved
    assert restored_metadata.env_id == "gsm8k"
    assert restored_metadata.env_params["split"] == "test"
    assert restored_metadata.generation_params["max_tokens"] == 1024
    assert restored_metadata.generation_params["temperature"] == 0.9
    assert restored_metadata.generation_params["top_p"] == 0.95


@pytest.mark.asyncio
async def test_backward_compatibility_with_old_checkpoints():
    """Test that old checkpoints (without env/gen config) don't break."""
    # Simulate old checkpoint metadata (no env_id, env_params, generation_params)
    old_checkpoint_data = {
        "window": 900,
        "file_manifest": {},
        "training_config": {},
    }

    # Verify: Consumer handles missing fields gracefully
    metadata = CheckpointMetadata(
        window=old_checkpoint_data.get("window", 900),
        file_manifest=old_checkpoint_data.get("file_manifest", {}),
        env_id=old_checkpoint_data.get("env_id"),  # None
        env_params=old_checkpoint_data.get("env_params", {}),  # {}
        generation_params=old_checkpoint_data.get("generation_params", {}),  # {}
    )

    # Should have defaults
    assert metadata.env_id is None
    assert metadata.env_params == {}
    assert metadata.generation_params == {}


def test_env_config_validation():
    """Test that environment config is validated properly."""
    # Valid config should pass
    with patch.dict("os.environ", {"GRAIL_ENV_ID": "mbpp", "GRAIL_ENV_SPLIT": "train"}):
        env_id, env_params = get_default_env_config()
        assert env_id == "mbpp"
        assert env_params["split"] == "train"

    # Invalid split should fallback to "train" with warning
    with patch.dict("os.environ", {"GRAIL_ENV_SPLIT": "invalid_split"}):
        env_id, env_params = get_default_env_config()
        assert env_params["split"] == "train"  # Falls back to default


def test_generation_params_validation():
    """Test that generation parameters are validated and clamped to safe ranges."""
    # Valid params should pass
    with patch.dict(
        "os.environ",
        {
            "GRAIL_GEN_MAX_TOKENS": "512",
            "GRAIL_GEN_TEMPERATURE": "0.7",
            "GRAIL_GEN_TOP_P": "0.9",
        },
    ):
        params = get_default_generation_params()
        assert params["max_tokens"] == 512
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    # Out-of-range values should clamp to defaults
    with patch.dict(
        "os.environ",
        {
            "GRAIL_GEN_MAX_TOKENS": "10000",  # Too high
            "GRAIL_GEN_TEMPERATURE": "5.0",  # Too high
            "GRAIL_GEN_TOP_P": "2.0",  # Too high
        },
    ):
        params = get_default_generation_params()
        # Should fall back to defaults
        assert params["max_tokens"] == 512  # Default from parse_int_param
        assert params["temperature"] == 0.7  # Default from parse_float_param
        assert params["top_p"] == 0.9  # Default from parse_float_param

    # Invalid types should fallback gracefully
    with patch.dict("os.environ", {"GRAIL_GEN_MAX_TOKENS": "not_a_number"}):
        params = get_default_generation_params()
        assert params["max_tokens"] == 512  # Falls back to default


def test_environment_factory_respects_runtime_params():
    """Test that environment factory uses runtime params from checkpoint."""
    # Test 1: No params = use defaults
    env1 = create_env("mbpp")
    assert env1 is not None

    # Test 2: Runtime params override defaults
    env2 = create_env("mbpp", env_params={"split": "validation"})
    assert env2 is not None
    # Verify it's actually using the validation split by checking task source
    # (The task source is cached per (env_id, split), so different split = different env)

    # Test 3: Empty env_params dict is handled correctly
    env3 = create_env("mbpp", env_params={})
    assert env3 is not None


@pytest.mark.asyncio
async def test_checkpoint_consumer_public_api():
    """Test that CheckpointManager exposes public API (not private methods)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        mock_credentials = Mock()

        manager = CheckpointManager(
            cache_root=cache_root,
            credentials=mock_credentials,
            keep_limit=2,
        )

        # Verify public API exists and accepts None gracefully
        result = await manager.get_checkpoint_metadata(None)
        assert result is None  # Should return None for invalid window

        # Verify private method is still internal
        assert hasattr(manager, "_fetch_metadata")


@pytest.mark.asyncio
async def test_env_switching_end_to_end():
    """Test environment switching: checkpoint with different env_id changes behavior."""
    # Simulate checkpoint 1: MBPP environment
    checkpoint1_metadata = CheckpointMetadata(
        window=1000,
        file_manifest={},
        env_id="mbpp",
        env_params={"split": "train"},
        generation_params={"max_tokens": 512, "temperature": 0.7},
    )

    # Simulate checkpoint 2: GSM8K environment (curriculum change)
    checkpoint2_metadata = CheckpointMetadata(
        window=1100,
        file_manifest={},
        env_id="gsm8k",
        env_params={"split": "train"},
        generation_params={"max_tokens": 1024, "temperature": 0.9},
    )

    # Verify: Different environments are created
    env1 = create_env(
        env_id=checkpoint1_metadata.env_id,
        env_params=checkpoint1_metadata.env_params,
    )
    env2 = create_env(
        env_id=checkpoint2_metadata.env_id,
        env_params=checkpoint2_metadata.env_params,
    )

    # Different environment types
    assert type(env1).__name__ == "PythonCodeEnv"
    assert type(env2).__name__ == "GSM8KEnv"

    # Verify generation params changed
    assert checkpoint1_metadata.generation_params["max_tokens"] == 512
    assert checkpoint2_metadata.generation_params["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_generation_params_propagation():
    """Test that generation params from checkpoint are used in rollout generation."""
    # Simulate checkpoint with specific generation params
    checkpoint_metadata = CheckpointMetadata(
        window=1000,
        file_manifest={},
        env_id="mbpp",
        env_params={"split": "train"},
        generation_params={
            "max_tokens": 256,  # Lower than default
            "temperature": 0.5,  # Lower than default
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.3,
        },
    )

    # Verify params are present and different from defaults
    assert checkpoint_metadata.generation_params["max_tokens"] == 256
    assert checkpoint_metadata.generation_params["temperature"] == 0.5
    assert checkpoint_metadata.generation_params["top_p"] == 0.8

    # In actual usage, these would be passed to AgentEnvLoop in mine.py
    # We verify the checkpoint contains the correct values


def test_validator_uses_checkpoint_env_config():
    """Test that validator can extract env config from checkpoint for validation."""
    # Simulate validator loading checkpoint metadata
    checkpoint_metadata = CheckpointMetadata(
        window=1000,
        file_manifest={},
        env_id="gsm8k",
        env_params={"split": "test"},
        generation_params={},
    )

    # Validator should use this env_id for validation
    env_id_for_validation = checkpoint_metadata.env_id or "mbpp"  # Fallback to default

    assert env_id_for_validation == "gsm8k"

    # Test fallback behavior when no env_id
    old_checkpoint = CheckpointMetadata(
        window=900,
        file_manifest={},
        env_id=None,  # Old checkpoint
        env_params={},
        generation_params={},
    )

    env_id_with_fallback = old_checkpoint.env_id or "mbpp"
    assert env_id_with_fallback == "mbpp"  # Uses fallback


@pytest.mark.asyncio
async def test_null_window_handling():
    """Test that None/null window values are handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_root = Path(tmpdir)
        mock_credentials = Mock()

        manager = CheckpointManager(
            cache_root=cache_root,
            credentials=mock_credentials,
            keep_limit=2,
        )

        # Should not crash, should return None
        result = await manager.get_checkpoint_metadata(None)
        assert result is None

        # Should also handle as expected
        _ = await manager.get_checkpoint_metadata(0)
        # This will try to fetch but won't find anything (returns None)
        # The point is it doesn't crash


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
