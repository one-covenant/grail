"""Unit tests for grail.infrastructure.delta_checkpoint module."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from grail.infrastructure.delta_checkpoint import (
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
    estimate_sparse_size,
    verify_weights_hash,
)


class TestComputeSparseDelta:
    """Tests for compute_sparse_delta function."""

    def test_basic_delta_computation(self) -> None:
        """Test basic delta computation between two state dicts."""
        base_state = {
            "layer1.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "layer2.weight": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        }
        current_state = {
            "layer1.weight": torch.tensor([1.1, 2.0, 3.2, 4.0]),
            "layer2.weight": torch.tensor([0.0, 0.5, 0.0, 0.0]),
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(current_state, base_state)

        # layer1: 2 changed (indices 0 and 2)
        # layer2: 1 changed (index 1)
        assert "layer1.weight.indices" in sparse_tensors
        assert "layer1.weight.values" in sparse_tensors
        assert "layer2.weight.indices" in sparse_tensors
        assert "layer2.weight.values" in sparse_tensors

        # Check shapes recorded
        assert shapes["layer1.weight"] == [4]
        assert shapes["layer2.weight"] == [4]

        # Check stats
        assert stats["total_params"] == 8
        assert stats["nonzero_params"] == 3
        assert stats["sparsity_ratio"] == pytest.approx(0.625, rel=1e-3)

    def test_no_changes_produces_empty_sparse(self) -> None:
        """Test that identical states produce empty sparse tensors."""
        state = {
            "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(state, state)

        assert len(sparse_tensors) == 0
        assert len(shapes) == 0
        assert stats["nonzero_params"] == 0
        assert stats["sparsity_ratio"] == 1.0

    def test_all_changed_produces_full_sparse(self) -> None:
        """Test that completely different states produce full sparse tensors."""
        base_state = {
            "layer.weight": torch.zeros(4),
        }
        current_state = {
            "layer.weight": torch.ones(4),
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(current_state, base_state)

        assert sparse_tensors["layer.weight.indices"].shape[0] == 4
        assert sparse_tensors["layer.weight.values"].shape[0] == 4
        assert stats["sparsity_ratio"] == 0.0

    def test_threshold_filters_small_deltas(self) -> None:
        """Test that threshold filters out small changes."""
        base_state = {
            "layer.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        current_state = {
            "layer.weight": torch.tensor([1.0001, 2.0, 3.5, 4.0]),  # Only 3.5 is significant
        }

        # With threshold 0.01, only the 0.5 delta should be captured
        sparse_tensors, shapes, stats = compute_sparse_delta(
            current_state, base_state, threshold=0.01
        )

        assert sparse_tensors["layer.weight.indices"].shape[0] == 1
        assert sparse_tensors["layer.weight.values"].item() == pytest.approx(0.5)

    def test_multidimensional_tensors(self) -> None:
        """Test delta computation on 2D tensors."""
        base_state = {
            "weight": torch.zeros(3, 4),
        }
        current_state = {
            "weight": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 3.0]]
            ),
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(current_state, base_state)

        assert shapes["weight"] == [3, 4]
        assert stats["nonzero_params"] == 3

        # Check indices are flat
        indices = sparse_tensors["weight.indices"]
        assert indices.tolist() == [0, 6, 11]  # Flat indices

    def test_float32_precision(self) -> None:
        """Test that deltas are computed in float32 for precision."""
        # Use bfloat16 tensors
        base_state = {
            "weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        }
        current_state = {
            "weight": torch.tensor([1.0001, 2.0001], dtype=torch.bfloat16),
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(current_state, base_state)

        # Values should be float32
        if sparse_tensors:
            for key in sparse_tensors:
                if "values" in key:
                    assert sparse_tensors[key].dtype == torch.float32


class TestApplySparseDelta:
    """Tests for apply_sparse_delta function."""

    def test_basic_delta_application(self) -> None:
        """Test applying sparse delta to base state."""
        base_state = {
            "layer.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        }
        sparse_tensors = {
            "layer.weight.indices": torch.tensor([0, 2], dtype=torch.int32),
            "layer.weight.values": torch.tensor([0.5, -0.5], dtype=torch.float32),
        }
        shapes = {"layer.weight": [4]}

        result = apply_sparse_delta(base_state, sparse_tensors, shapes, torch.float32)

        expected = torch.tensor([1.5, 2.0, 2.5, 4.0])
        assert torch.allclose(result["layer.weight"], expected)

    def test_empty_sparse_delta(self) -> None:
        """Test applying empty sparse delta (no changes)."""
        base_state = {
            "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        }

        result = apply_sparse_delta(base_state, {}, {}, torch.float32)

        assert torch.allclose(result["layer.weight"], base_state["layer.weight"])

    def test_multidimensional_reconstruction(self) -> None:
        """Test reconstruction of 2D tensors from flat sparse delta."""
        base_state = {
            "weight": torch.zeros(2, 3),
        }
        sparse_tensors = {
            "weight.indices": torch.tensor([0, 5], dtype=torch.int32),  # (0,0) and (1,2)
            "weight.values": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }
        shapes = {"weight": [2, 3]}

        result = apply_sparse_delta(base_state, sparse_tensors, shapes, torch.float32)

        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        assert torch.allclose(result["weight"], expected)

    def test_dtype_conversion(self) -> None:
        """Test that output is converted to target dtype."""
        base_state = {
            "weight": torch.tensor([1.0, 2.0]),
        }
        sparse_tensors = {
            "weight.indices": torch.tensor([0], dtype=torch.int32),
            "weight.values": torch.tensor([0.5], dtype=torch.float32),
        }
        shapes = {"weight": [2]}

        result = apply_sparse_delta(
            base_state, sparse_tensors, shapes, target_dtype=torch.bfloat16
        )

        assert result["weight"].dtype == torch.bfloat16


class TestRoundTrip:
    """Test round-trip: compute delta, apply delta, verify identical."""

    def test_round_trip_exact_reconstruction(self) -> None:
        """Test that compute + apply produces exact original state."""
        base_state = {
            "layer1.weight": torch.randn(32, 64),
            "layer2.bias": torch.randn(64),
        }
        current_state = {
            "layer1.weight": base_state["layer1.weight"] + torch.randn(32, 64) * 0.01,
            "layer2.bias": base_state["layer2.bias"] + torch.randn(64) * 0.1,
        }

        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            current_state, base_state, threshold=0.0
        )

        # Apply delta
        reconstructed = apply_sparse_delta(
            base_state, sparse_tensors, shapes, target_dtype=torch.float32
        )

        # Verify reconstruction
        for name in current_state:
            assert torch.allclose(
                reconstructed[name], current_state[name].float(), atol=1e-6
            ), f"Mismatch in {name}"

    def test_round_trip_with_bf16(self) -> None:
        """Test round-trip with bfloat16 tensors."""
        base_state = {
            "weight": torch.randn(16, 16, dtype=torch.bfloat16),
        }
        current_state = {
            "weight": (base_state["weight"].float() + torch.randn(16, 16) * 0.1).to(
                torch.bfloat16
            ),
        }

        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            current_state, base_state, threshold=0.0
        )

        # Apply delta back to bfloat16
        reconstructed = apply_sparse_delta(
            base_state, sparse_tensors, shapes, target_dtype=torch.bfloat16
        )

        # Should be very close (some precision loss due to bf16)
        assert torch.allclose(
            reconstructed["weight"].float(),
            current_state["weight"].float(),
            atol=1e-2,
        )


class TestComputeWeightsHash:
    """Tests for compute_weights_hash function."""

    def test_deterministic_hash(self) -> None:
        """Test that hash is deterministic for same input."""
        state = {
            "layer1": torch.tensor([1.0, 2.0, 3.0]),
            "layer2": torch.tensor([4.0, 5.0]),
        }

        hash1 = compute_weights_hash(state)
        hash2 = compute_weights_hash(state)

        assert hash1 == hash2

    def test_different_states_different_hash(self) -> None:
        """Test that different states produce different hashes."""
        state1 = {"layer": torch.tensor([1.0, 2.0])}
        state2 = {"layer": torch.tensor([1.0, 2.1])}

        hash1 = compute_weights_hash(state1)
        hash2 = compute_weights_hash(state2)

        assert hash1 != hash2

    def test_order_independent_keys(self) -> None:
        """Test that hash is independent of dict key order (sorted internally)."""
        state1 = {
            "b": torch.tensor([1.0]),
            "a": torch.tensor([2.0]),
        }
        state2 = {
            "a": torch.tensor([2.0]),
            "b": torch.tensor([1.0]),
        }

        hash1 = compute_weights_hash(state1)
        hash2 = compute_weights_hash(state2)

        assert hash1 == hash2

    def test_hash_format(self) -> None:
        """Test that hash is a valid hex string."""
        state = {"layer": torch.tensor([1.0])}
        hash_value = compute_weights_hash(state)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in hash_value)


class TestVerifyWeightsHash:
    """Tests for verify_weights_hash function."""

    def test_valid_hash_verification(self) -> None:
        """Test that correct hash verifies successfully."""
        state = {"layer": torch.tensor([1.0, 2.0, 3.0])}
        expected_hash = compute_weights_hash(state)

        assert verify_weights_hash(state, expected_hash) is True

    def test_invalid_hash_verification(self) -> None:
        """Test that incorrect hash fails verification."""
        state = {"layer": torch.tensor([1.0, 2.0, 3.0])}
        wrong_hash = "0" * 64

        assert verify_weights_hash(state, wrong_hash) is False


class TestEstimateSparseSize:
    """Tests for estimate_sparse_size function."""

    def test_size_estimation(self) -> None:
        """Test sparse size estimation."""
        # 1000 non-zero params: 4 bytes per index + 4 bytes per value = 8000 bytes
        size = estimate_sparse_size(1000)
        assert size == 8000

    def test_size_with_different_dtypes(self) -> None:
        """Test size estimation with different dtypes."""
        size = estimate_sparse_size(
            1000, index_dtype=torch.int64, value_dtype=torch.float64
        )
        # 8 bytes per index + 8 bytes per value = 16000 bytes
        assert size == 16000


class TestSafetensorsIntegration:
    """Test integration with safetensors file format."""

    def test_save_and_load_sparse_delta(self) -> None:
        """Test that sparse delta can be saved and loaded with safetensors."""
        base_state = {
            "layer.weight": torch.randn(10, 10),
        }
        current_state = {
            "layer.weight": base_state["layer.weight"] + torch.randn(10, 10) * 0.01,
        }

        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            current_state, base_state, threshold=0.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            delta_path = Path(tmpdir) / "delta_sparse.safetensors"

            # Save sparse tensors
            save_file(sparse_tensors, delta_path)

            # Load and apply
            loaded_sparse = load_file(delta_path)
            reconstructed = apply_sparse_delta(
                base_state, loaded_sparse, shapes, torch.float32
            )

            # Verify reconstruction
            assert torch.allclose(
                reconstructed["layer.weight"],
                current_state["layer.weight"],
                atol=1e-6,
            )
