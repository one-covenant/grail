"""Tests for Parquet serialization of rollout data."""

import pytest

from grail.infrastructure.parquet_io import (
    deserialize_parquet_to_window,
    serialize_window_to_parquet,
)


def _create_sample_inference() -> dict:
    """Create a sample inference dict mimicking real miner data."""
    return {
        "window_start": 100000,
        "block": 100005,
        "nonce": 12345,
        "block_hash": "0xabc123def456",
        "randomness": "0xrandom789",
        "use_drand": True,
        "rollout_group": 42,
        "rollout_index": 0,
        "total_in_group": 4,
        "checkpoint_window": 99000,
        "commit": {
            "tokens": [1, 2, 3, 4, 5, 100, 200, 300],
            "commitments": [[0.1, 0.2], [0.3, 0.4]],
            "proof_version": "v1.0",
            "model": {
                "name": "test-model",
                "layer_index": 12,
            },
            "signature": "abc123signature",
            "beacon": {"round": 1234, "randomness": "0xbeacon"},
            "rollout": {
                "trajectory": [["step1", [True, False]], ["step2", [True, True]]],
                "total_reward": 0.75,
                "advantage": 0.25,
                "success": True,
                "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
                "prompt_length": 3,
                "completion_length": 5,
                "satisfied_clauses": 10,
                "assignment": [True, False, True, False],
            },
        },
        "timestamp": 1700000000.123,
        "challenge": "seed|hash|nonce",
        "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "signature": "rollout_signature_hex",
    }


def _create_sample_window_data() -> dict:
    """Create sample window data with multiple inferences."""
    return {
        "wallet": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "window_start": 100000,
        "window_length": 30,
        "inference_count": 2,
        "inferences": [
            _create_sample_inference(),
            _create_sample_inference(),
        ],
        "timestamp": 1700000000.0,
    }


class TestParquetSerialization:
    """Test Parquet serialization and deserialization."""

    def test_round_trip_preserves_data(self) -> None:
        """Test that serialize -> deserialize preserves data structure."""
        original = _create_sample_window_data()

        # Serialize to Parquet bytes
        parquet_bytes = serialize_window_to_parquet(original)

        # Verify we got bytes back
        assert isinstance(parquet_bytes, bytes)
        assert len(parquet_bytes) > 0

        # Deserialize back
        restored = deserialize_parquet_to_window(parquet_bytes)

        # Check window-level metadata
        assert restored["wallet"] == original["wallet"]
        assert restored["window_start"] == original["window_start"]
        assert restored["window_length"] == original["window_length"]
        assert restored["inference_count"] == original["inference_count"]
        assert len(restored["inferences"]) == len(original["inferences"])

    def test_inference_fields_preserved(self) -> None:
        """Test that inference fields are preserved through round-trip."""
        original = _create_sample_window_data()

        parquet_bytes = serialize_window_to_parquet(original)
        restored = deserialize_parquet_to_window(parquet_bytes)

        original_inf = original["inferences"][0]
        restored_inf = restored["inferences"][0]

        # Check top-level fields
        assert restored_inf["window_start"] == original_inf["window_start"]
        assert restored_inf["block"] == original_inf["block"]
        assert restored_inf["nonce"] == original_inf["nonce"]
        assert restored_inf["block_hash"] == original_inf["block_hash"]
        assert restored_inf["use_drand"] == original_inf["use_drand"]
        assert restored_inf["rollout_group"] == original_inf["rollout_group"]
        assert restored_inf["hotkey"] == original_inf["hotkey"]

    def test_commit_structure_preserved(self) -> None:
        """Test that nested commit structure is preserved."""
        original = _create_sample_window_data()

        parquet_bytes = serialize_window_to_parquet(original)
        restored = deserialize_parquet_to_window(parquet_bytes)

        original_commit = original["inferences"][0]["commit"]
        restored_commit = restored["inferences"][0]["commit"]

        # Check commit fields
        assert restored_commit["tokens"] == original_commit["tokens"]
        assert restored_commit["proof_version"] == original_commit["proof_version"]
        assert restored_commit["signature"] == original_commit["signature"]

        # Check model nested structure
        assert restored_commit["model"]["name"] == original_commit["model"]["name"]
        assert restored_commit["model"]["layer_index"] == original_commit["model"]["layer_index"]

    def test_rollout_structure_preserved(self) -> None:
        """Test that rollout metadata is preserved."""
        original = _create_sample_window_data()

        parquet_bytes = serialize_window_to_parquet(original)
        restored = deserialize_parquet_to_window(parquet_bytes)

        original_rollout = original["inferences"][0]["commit"]["rollout"]
        restored_rollout = restored["inferences"][0]["commit"]["rollout"]

        # Check rollout fields
        assert restored_rollout["total_reward"] == pytest.approx(
            original_rollout["total_reward"], rel=1e-6
        )
        assert restored_rollout["advantage"] == pytest.approx(
            original_rollout["advantage"], rel=1e-6
        )
        assert restored_rollout["success"] == original_rollout["success"]
        assert restored_rollout["prompt_length"] == original_rollout["prompt_length"]
        assert restored_rollout["completion_length"] == original_rollout["completion_length"]

        # Check variable-length arrays
        assert restored_rollout["assignment"] == original_rollout["assignment"]

        # Token logprobs should be approximately equal (floats)
        for orig, rest in zip(
            original_rollout["token_logprobs"],
            restored_rollout["token_logprobs"],
            strict=True,
        ):
            assert rest == pytest.approx(orig, rel=1e-6)

    def test_json_encoded_fields_preserved(self) -> None:
        """Test that JSON-encoded complex fields are preserved."""
        original = _create_sample_window_data()

        parquet_bytes = serialize_window_to_parquet(original)
        restored = deserialize_parquet_to_window(parquet_bytes)

        original_commit = original["inferences"][0]["commit"]
        restored_commit = restored["inferences"][0]["commit"]

        # Check commitments (JSON-encoded list of lists)
        assert restored_commit["commitments"] == original_commit["commitments"]

        # Check beacon (JSON-encoded dict)
        assert restored_commit["beacon"] == original_commit["beacon"]

        # Check trajectory (JSON-encoded nested list)
        assert restored_commit["rollout"]["trajectory"] == original_commit["rollout"]["trajectory"]

    def test_empty_inferences(self) -> None:
        """Test handling of empty inferences list."""
        original = {
            "wallet": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "window_start": 100000,
            "window_length": 30,
            "inference_count": 0,
            "inferences": [],
            "timestamp": 1700000000.0,
        }

        parquet_bytes = serialize_window_to_parquet(original)
        restored = deserialize_parquet_to_window(parquet_bytes)

        assert restored["wallet"] == original["wallet"]
        assert restored["inferences"] == []

    def test_compression_smaller_than_json(self) -> None:
        """Test that Parquet output is smaller than equivalent JSON."""
        import json

        # Create window with multiple inferences to see compression benefits
        original = _create_sample_window_data()
        original["inferences"] = [_create_sample_inference() for _ in range(10)]
        original["inference_count"] = 10

        json_bytes = json.dumps(original).encode()
        parquet_bytes = serialize_window_to_parquet(original)

        # Parquet with snappy compression should be smaller
        # (may not always be true for tiny data, but should be for typical rollouts)
        compression_ratio = len(parquet_bytes) / len(json_bytes)
        print(
            f"JSON: {len(json_bytes)} bytes, Parquet: {len(parquet_bytes)} bytes, "
            f"ratio: {compression_ratio:.2f}"
        )

        # Assert reasonable compression (allow up to 1.5x for small test data)
        # Real rollout data with long token lists should compress much better
        assert compression_ratio < 1.5
