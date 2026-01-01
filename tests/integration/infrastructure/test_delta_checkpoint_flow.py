"""Integration tests for delta checkpoint upload and download flow.

These tests verify the end-to-end flow of:
1. Publishing FULL checkpoints
2. Publishing DELTA checkpoints
3. Downloading and reconstructing DELTA checkpoints
4. Hash verification during reconstruction
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    CheckpointMetadata,
)
from grail.infrastructure.delta_checkpoint import (
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)


@pytest.fixture
def temp_cache() -> Path:
    """Create a temporary cache directory."""
    tmpdir = tempfile.mkdtemp(prefix="checkpoint_cache_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_model_state() -> dict[str, torch.Tensor]:
    """Create a sample model state dict."""
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 64),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(16, 64),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
        "model.layers.0.mlp.up_proj.weight": torch.randn(128, 64),
        "model.layers.0.mlp.down_proj.weight": torch.randn(64, 128),
        "model.embed_tokens.weight": torch.randn(1000, 64),
        "lm_head.weight": torch.randn(1000, 64),
    }


@pytest.fixture
def updated_model_state(sample_model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create an updated model state with sparse changes."""
    updated = {}
    for name, tensor in sample_model_state.items():
        # Add small perturbations to ~1% of weights
        mask = torch.rand_like(tensor) < 0.01
        delta = torch.randn_like(tensor) * 0.01
        updated[name] = tensor + delta * mask.float()
    return updated


class TestDeltaCheckpointComputation:
    """Test sparse delta computation and application."""

    def test_compute_and_apply_delta_roundtrip(
        self,
        sample_model_state: dict[str, torch.Tensor],
        updated_model_state: dict[str, torch.Tensor],
    ) -> None:
        """Test that compute + apply produces exact original weights."""
        # Compute sparse delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            updated_model_state,
            sample_model_state,
            threshold=0.0,
        )

        # Verify sparsity is reasonable
        assert stats["sparsity_ratio"] > 0.9, "Expected high sparsity (>90%)"

        # Apply delta to base
        reconstructed = apply_sparse_delta(
            sample_model_state,
            sparse_tensors,
            shapes,
            target_dtype=torch.float32,
        )

        # Verify reconstruction
        for name in updated_model_state:
            expected = updated_model_state[name].float()
            actual = reconstructed[name]
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch in {name}"

    def test_hash_verification(
        self,
        sample_model_state: dict[str, torch.Tensor],
        updated_model_state: dict[str, torch.Tensor],
    ) -> None:
        """Test that hash verification works correctly."""
        # Compute hash of updated state
        expected_hash = compute_weights_hash(updated_model_state)

        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            updated_model_state,
            sample_model_state,
            threshold=0.0,
        )

        # Apply delta
        reconstructed = apply_sparse_delta(
            sample_model_state,
            sparse_tensors,
            shapes,
            target_dtype=torch.float32,
        )

        # Verify hash
        actual_hash = compute_weights_hash(reconstructed)
        assert actual_hash == expected_hash


class TestDeltaCheckpointFilesystem:
    """Test saving and loading delta checkpoints from filesystem."""

    def test_save_and_load_sparse_delta(
        self,
        temp_cache: Path,
        sample_model_state: dict[str, torch.Tensor],
        updated_model_state: dict[str, torch.Tensor],
    ) -> None:
        """Test saving sparse delta to safetensors and loading back."""
        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            updated_model_state,
            sample_model_state,
            threshold=0.0,
        )

        # Save sparse delta
        delta_path = temp_cache / "delta_sparse.safetensors"
        save_file(sparse_tensors, delta_path)

        # Save metadata
        delta_meta = {
            "format": "sparse_coo",
            "threshold": 0.0,
            "shapes": shapes,
            **stats,
        }
        meta_path = temp_cache / "delta_metadata.json"
        meta_path.write_text(json.dumps(delta_meta))

        # Load and apply
        loaded_sparse = load_file(delta_path)
        loaded_meta = json.loads(meta_path.read_text())

        reconstructed = apply_sparse_delta(
            sample_model_state,
            loaded_sparse,
            loaded_meta["shapes"],
            target_dtype=torch.float32,
        )

        # Verify
        for name in updated_model_state:
            expected = updated_model_state[name].float()
            actual = reconstructed[name]
            assert torch.allclose(actual, expected, atol=1e-5), f"Mismatch in {name}"

    def test_simulate_full_checkpoint_flow(
        self,
        temp_cache: Path,
        sample_model_state: dict[str, torch.Tensor],
    ) -> None:
        """Simulate creating a full checkpoint directory structure."""
        checkpoint_dir = temp_cache / "checkpoint-100"
        checkpoint_dir.mkdir(parents=True)

        # Save model
        model_path = checkpoint_dir / "model.safetensors"
        save_file(sample_model_state, model_path)

        # Create metadata
        metadata = {
            "window": 100,
            "checkpoint_type": "FULL",
            "file_manifest": {},
            "training_config": {},
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))

        # Write FULL marker
        (checkpoint_dir / "FULL").write_text("")

        # Verify structure
        assert model_path.exists()
        assert (checkpoint_dir / "metadata.json").exists()
        assert (checkpoint_dir / "FULL").exists()

    def test_simulate_delta_checkpoint_flow(
        self,
        temp_cache: Path,
        sample_model_state: dict[str, torch.Tensor],
        updated_model_state: dict[str, torch.Tensor],
    ) -> None:
        """Simulate creating a delta checkpoint directory structure."""
        # First, create base checkpoint
        base_dir = temp_cache / "checkpoint-100"
        base_dir.mkdir(parents=True)
        save_file(sample_model_state, base_dir / "model.safetensors")
        base_metadata = {
            "window": 100,
            "checkpoint_type": "FULL",
            "file_manifest": {},
        }
        (base_dir / "metadata.json").write_text(json.dumps(base_metadata))
        (base_dir / "FULL").write_text("")

        # Compute delta
        sparse_tensors, shapes, stats = compute_sparse_delta(
            updated_model_state,
            sample_model_state,
            threshold=0.0,
        )
        weights_hash = compute_weights_hash(updated_model_state)

        # Create delta checkpoint
        delta_dir = temp_cache / "checkpoint-110"
        delta_dir.mkdir(parents=True)

        # Save sparse delta
        save_file(sparse_tensors, delta_dir / "delta_sparse.safetensors")

        # Save delta metadata
        delta_meta = {
            "format": "sparse_coo",
            "threshold": 0.0,
            "base_window": 100,
            "shapes": shapes,
            **stats,
        }
        (delta_dir / "delta_metadata.json").write_text(json.dumps(delta_meta))

        # Save checkpoint metadata
        checkpoint_meta = {
            "window": 110,
            "checkpoint_type": "DELTA",
            "base_window": 100,
            "weights_hash": weights_hash,
            "file_manifest": {},
        }
        (delta_dir / "metadata.json").write_text(json.dumps(checkpoint_meta))
        (delta_dir / "DELTA").write_text("")

        # Verify structure
        assert (delta_dir / "delta_sparse.safetensors").exists()
        assert (delta_dir / "delta_metadata.json").exists()
        assert (delta_dir / "DELTA").exists()

        # Simulate reconstruction
        base_state = load_file(base_dir / "model.safetensors")
        loaded_sparse = load_file(delta_dir / "delta_sparse.safetensors")
        loaded_delta_meta = json.loads((delta_dir / "delta_metadata.json").read_text())

        reconstructed = apply_sparse_delta(
            base_state,
            loaded_sparse,
            loaded_delta_meta["shapes"],
            target_dtype=torch.float32,
        )

        # Verify hash
        actual_hash = compute_weights_hash(reconstructed)
        assert actual_hash == weights_hash


class TestCheckpointMetadata:
    """Test CheckpointMetadata with delta fields."""

    def test_metadata_is_delta(self) -> None:
        """Test is_delta() method."""
        full_metadata = CheckpointMetadata(
            window=100,
            file_manifest={},
            checkpoint_type="FULL",
        )
        assert not full_metadata.is_delta()

        delta_metadata = CheckpointMetadata(
            window=110,
            file_manifest={},
            checkpoint_type="DELTA",
            anchor_window=100,
            weights_hash="abc123",
        )
        assert delta_metadata.is_delta()

    def test_metadata_serialization(self) -> None:
        """Test that metadata with delta fields can be serialized."""
        metadata = CheckpointMetadata(
            window=110,
            file_manifest={"delta_sparse.safetensors": "hash123"},
            checkpoint_type="DELTA",
            anchor_window=100,
            weights_hash="abcdef1234567890",
        )

        # Serialize
        metadata_dict = {**metadata.__dict__}
        json_str = json.dumps(metadata_dict)

        # Deserialize
        loaded = json.loads(json_str)
        assert loaded["checkpoint_type"] == "DELTA"
        assert loaded["anchor_window"] == 100
        assert loaded["weights_hash"] == "abcdef1234567890"


class TestRetentionPolicy:
    """Test that retention policy protects base checkpoints."""

    def test_compute_keep_windows_includes_base(self, temp_cache: Path) -> None:
        """Test that _compute_keep_windows includes base checkpoints."""
        from grail.shared.constants import DELTA_BASE_INTERVAL, WINDOW_LENGTH

        manager = CheckpointManager(
            cache_root=temp_cache,
            credentials=None,
            keep_limit=5,
        )

        # Current window is 250 windows (in blocks), with base interval of 100 windows
        # Should keep recent windows AND their base windows
        current_window = 250 * WINDOW_LENGTH

        keep = manager._compute_keep_windows(current_window)

        # Should include base windows at 200 and 100 boundaries
        # Assuming DELTA_BASE_INTERVAL = 100
        if DELTA_BASE_INTERVAL == 100:
            assert 200 * WINDOW_LENGTH in keep, "Base window 200 should be kept"
            assert 100 * WINDOW_LENGTH in keep, "Base window 100 should be kept"


class TestEdgeCases:
    """Test edge cases for delta checkpoint handling."""

    def test_zero_percent_sparsity(self, sample_model_state: dict[str, torch.Tensor]) -> None:
        """Test handling when all weights change (0% sparsity)."""
        # Completely different state
        different_state = {
            name: torch.randn_like(tensor) for name, tensor in sample_model_state.items()
        }

        sparse_tensors, shapes, stats = compute_sparse_delta(
            different_state,
            sample_model_state,
            threshold=0.0,
        )

        assert stats["sparsity_ratio"] == 0.0
        assert stats["nonzero_params"] == stats["total_params"]

    def test_hundred_percent_sparsity(self, sample_model_state: dict[str, torch.Tensor]) -> None:
        """Test handling when no weights change (100% sparsity)."""
        sparse_tensors, shapes, stats = compute_sparse_delta(
            sample_model_state,
            sample_model_state,
            threshold=0.0,
        )

        assert stats["sparsity_ratio"] == 1.0
        assert stats["nonzero_params"] == 0
        assert len(sparse_tensors) == 0

    def test_single_param_changed(self, sample_model_state: dict[str, torch.Tensor]) -> None:
        """Test handling when only a single parameter changes."""
        updated = {name: tensor.clone() for name, tensor in sample_model_state.items()}
        # Change single element
        first_key = list(updated.keys())[0]
        updated[first_key][0, 0] += 1.0

        sparse_tensors, shapes, stats = compute_sparse_delta(
            updated,
            sample_model_state,
            threshold=0.0,
        )

        assert stats["nonzero_params"] == 1
        assert f"{first_key}.indices" in sparse_tensors
        assert sparse_tensors[f"{first_key}.indices"].shape[0] == 1
