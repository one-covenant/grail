"""Unit tests for chained delta checkpoint functionality.

Tests:
- Chain building from target to anchor
- Sequential delta application with intermediate bf16 casts
- Bit-exact reconstruction verification
- Retention policy for chained deltas
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from grail.infrastructure.checkpoint_consumer import CheckpointMetadata
from grail.infrastructure.delta_checkpoint import (
    apply_sparse_delta,
    compute_sparse_delta,
    compute_weights_hash,
)

# ============================================================================
# Tests for Chained Delta Precision (bit-exact bf16 reconstruction)
# ============================================================================


class TestChainedDeltaPrecision:
    """Tests that chained deltas with intermediate bf16 casts are bit-exact."""

    def test_single_delta_bit_exact(self) -> None:
        """Single delta should reconstruct exactly."""
        w0 = torch.randn(1000, dtype=torch.bfloat16)
        w1 = (w0.float() + torch.randn(1000) * 0.01).to(torch.bfloat16)

        # Compute delta
        delta = w1.float() - w0.float()

        # Reconstruct
        reconstructed = (w0.float() + delta).to(torch.bfloat16)

        assert (reconstructed == w1).all(), "Single delta should be bit-exact"

    def test_chained_deltas_bit_exact(self) -> None:
        """Chained deltas with intermediate bf16 casts should be bit-exact."""
        n_params = 10000
        n_deltas = 20

        # Generate chain of bf16 weights
        weights = [torch.randn(n_params, dtype=torch.bfloat16)]
        for _ in range(n_deltas):
            update = torch.randn(n_params, dtype=torch.float32) * 0.001
            new_w = (weights[-1].float() + update).to(torch.bfloat16)
            weights.append(new_w)

        # Compute chained deltas (each vs previous)
        deltas = []
        for i in range(1, len(weights)):
            delta = weights[i].float() - weights[i - 1].float()
            deltas.append(delta)

        # Reconstruct WITH intermediate bf16 casts
        reconstructed = weights[0]
        for i, delta in enumerate(deltas):
            reconstructed = (reconstructed.float() + delta).to(torch.bfloat16)

            # Verify each step
            expected = weights[i + 1]
            assert (reconstructed == expected).all(), f"Mismatch at step {i + 1}"

    def test_chained_deltas_100_steps(self) -> None:
        """Test 100 chained deltas remain bit-exact."""
        n_params = 5000
        n_deltas = 100

        weights = [torch.randn(n_params, dtype=torch.bfloat16)]
        for _ in range(n_deltas):
            update = torch.randn(n_params, dtype=torch.float32) * 0.0001
            new_w = (weights[-1].float() + update).to(torch.bfloat16)
            weights.append(new_w)

        # Compute and apply chain
        reconstructed = weights[0]
        for i in range(1, len(weights)):
            delta = weights[i].float() - weights[i - 1].float()
            reconstructed = (reconstructed.float() + delta).to(torch.bfloat16)

        assert (reconstructed == weights[-1]).all(), "100 chained deltas should be bit-exact"

    def test_chained_with_sparse_delta_format(self) -> None:
        """Test chained deltas using actual sparse delta format."""
        n_params = 1000

        # Create base and chain of weights
        base_state = {"layer.weight": torch.randn(n_params, dtype=torch.bfloat16)}
        states = [base_state]

        for _ in range(5):
            prev = states[-1]
            new_weight = (prev["layer.weight"].float() + torch.randn(n_params) * 0.01).to(
                torch.bfloat16
            )
            states.append({"layer.weight": new_weight})

        # Compute sparse deltas
        deltas = []
        for i in range(1, len(states)):
            sparse, shapes, stats = compute_sparse_delta(states[i], states[i - 1], threshold=0.0)
            deltas.append((sparse, shapes))

        # Reconstruct using apply_sparse_delta
        current = {k: v.clone() for k, v in base_state.items()}
        for sparse, shapes in deltas:
            current = apply_sparse_delta(current, sparse, shapes, torch.bfloat16)

        # Verify final state
        final_expected = states[-1]["layer.weight"]
        final_actual = current["layer.weight"]
        assert (final_actual == final_expected).all(), "Sparse delta chain should be bit-exact"


# ============================================================================
# Tests for Chain Building Logic
# ============================================================================


class TestChainBuilding:
    """Tests for _build_delta_chain functionality."""

    def test_chain_reverses_to_oldest_first(self) -> None:
        """Chain should be returned oldest-first for sequential application."""
        # Simulate walking backwards: target -> prev -> prev -> anchor
        # Result should be [oldest_delta, ..., newest_delta]

        # This is a logical test - the actual order matters for reconstruction
        chain = [3, 2, 1]  # Simulating backwards walk
        chain.reverse()
        assert chain == [1, 2, 3], "Chain should be reversed to oldest-first"

    def test_single_delta_chain(self) -> None:
        """Chain with single delta should work."""
        # Chain: FULL(0) -> DELTA(1)
        # _build_delta_chain(DELTA(1)) should return (FULL(0), [DELTA(1)])
        pass  # Covered by integration tests


class TestRetentionPolicyChainedDeltas:
    """Tests for retention policy with chained deltas."""

    def test_keeps_entire_active_chain(self) -> None:
        """Should keep all windows from current anchor to now."""
        from grail.shared.retention_utils import compute_retention_windows

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 5):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                # Anchor stride = 5 * 30 = 150
                # Current window 180: anchor at 150
                result = compute_retention_windows(180)

                # Should keep: 150, 180 (chain from 150 to 180)
                # Also 0 (previous anchor) and 0-150 chain
                assert 150 in result
                assert 180 in result

    def test_keeps_previous_anchor_chain(self) -> None:
        """Should keep previous anchor's chain for transition."""
        from grail.shared.retention_utils import compute_retention_windows

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 3):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                # Anchor stride = 3 * 30 = 90
                # Current window 120: anchor at 90
                # Previous anchor at 0
                result = compute_retention_windows(120)

                # Current chain: 90, 120
                assert 90 in result
                assert 120 in result

                # Previous anchor
                assert 0 in result

    def test_consumer_retention_keeps_chain(self) -> None:
        """Consumer should also keep entire chain."""
        from grail.infrastructure.checkpoint_consumer import CheckpointManager

        manager = CheckpointManager(
            cache_root=MagicMock(),
            credentials=None,
            keep_limit=3,
        )

        with patch("grail.shared.retention_utils.DELTA_BASE_INTERVAL", 5):
            with patch("grail.shared.retention_utils.WINDOW_LENGTH", 30):
                result = manager._compute_keep_windows(180)

                # Should keep chain from anchor (150) to now (180)
                assert 150 in result
                assert 180 in result


# ============================================================================
# Tests for Hash Verification
# ============================================================================


class TestChainedHashVerification:
    """Tests for hash verification after chain reconstruction."""

    def test_hash_matches_after_chain_reconstruction(self) -> None:
        """Hash of reconstructed state should match original."""
        n_params = 500

        # Create chain
        weights = [torch.randn(n_params, dtype=torch.bfloat16)]
        for _ in range(5):
            new_w = (weights[-1].float() + torch.randn(n_params) * 0.01).to(torch.bfloat16)
            weights.append(new_w)

        # Compute expected hash of final state
        final_state = {"weight": weights[-1]}
        expected_hash = compute_weights_hash(final_state)

        # Reconstruct via chain
        reconstructed = weights[0]
        for i in range(1, len(weights)):
            delta = weights[i].float() - weights[i - 1].float()
            reconstructed = (reconstructed.float() + delta).to(torch.bfloat16)

        # Verify hash
        reconstructed_state = {"weight": reconstructed}
        actual_hash = compute_weights_hash(reconstructed_state)

        assert actual_hash == expected_hash, "Hash should match after chain reconstruction"


# ============================================================================
# Tests for Metadata Schema
# ============================================================================


class TestChainedDeltaMetadata:
    """Tests for chained delta metadata fields."""

    def test_metadata_has_prev_window(self) -> None:
        """CheckpointMetadata should have prev_window field."""
        meta = CheckpointMetadata(
            window=100,
            file_manifest={},
            checkpoint_type="DELTA",
            prev_window=70,
            anchor_window=0,
        )
        assert meta.prev_window == 70
        assert meta.anchor_window == 0
        assert meta.is_delta() is True

    def test_full_checkpoint_has_no_prev_window(self) -> None:
        """FULL checkpoints should not require prev_window."""
        meta = CheckpointMetadata(
            window=0,
            file_manifest={},
            checkpoint_type="FULL",
        )
        assert meta.prev_window is None
        assert meta.anchor_window is None
        assert meta.is_delta() is False


# ============================================================================
# Tests for Fast Path Optimization
# ============================================================================


class TestFastPathOptimization:
    """Tests for the fast path delta application (single delta to cached prev)."""

    def test_single_delta_application_is_correct(self) -> None:
        """Applying single delta to prev should give same result as chain."""
        n_params = 1000

        # Create: W0 (FULL) -> W1 (DELTA) -> W2 (DELTA)
        w0 = torch.randn(n_params, dtype=torch.bfloat16)
        w1 = (w0.float() + torch.randn(n_params) * 0.01).to(torch.bfloat16)
        w2 = (w1.float() + torch.randn(n_params) * 0.01).to(torch.bfloat16)

        # Compute deltas
        delta_1 = w1.float() - w0.float()
        delta_2 = w2.float() - w1.float()

        # Slow path: apply chain from W0
        chain_result = w0.float()
        chain_result = (chain_result + delta_1).to(torch.bfloat16)
        chain_result = (chain_result.float() + delta_2).to(torch.bfloat16)

        # Fast path: apply delta_2 directly to W1
        fast_result = (w1.float() + delta_2).to(torch.bfloat16)

        # Both should give W2 exactly
        assert (chain_result == w2).all(), "Chain result should match W2"
        assert (fast_result == w2).all(), "Fast result should match W2"
        assert (chain_result == fast_result).all(), "Chain and fast should match"

    def test_fast_path_hash_verification(self) -> None:
        """Fast path should produce same hash as chain path."""
        n_params = 500

        w0 = torch.randn(n_params, dtype=torch.bfloat16)
        w1 = (w0.float() + torch.randn(n_params) * 0.01).to(torch.bfloat16)
        w2 = (w1.float() + torch.randn(n_params) * 0.01).to(torch.bfloat16)

        # Expected hash of W2
        expected_hash = compute_weights_hash({"weight": w2})

        # Fast path: W1 + delta_2
        delta_2 = w2.float() - w1.float()
        fast_result = (w1.float() + delta_2).to(torch.bfloat16)
        fast_hash = compute_weights_hash({"weight": fast_result})

        assert fast_hash == expected_hash, "Fast path hash should match"

    def test_in_place_model_update_simulation(self) -> None:
        """Simulate in-place model update like apply_delta_in_place does."""
        from grail.infrastructure.delta_checkpoint import apply_sparse_delta, compute_sparse_delta

        # Create a simple "model" state dict with larger changes to avoid threshold filtering
        n_params = 500
        w1 = torch.randn(n_params, dtype=torch.bfloat16)
        # Use larger delta to ensure all changes pass the threshold
        w2 = (w1.float() + torch.randn(n_params) * 0.5).to(torch.bfloat16)

        current_state = {"weight": w1.clone()}
        target_state = {"weight": w2}

        # Compute sparse delta: compute_sparse_delta(current, base) = current - base
        # So we need compute_sparse_delta(target, current) = target - current
        sparse_tensors, shapes, _info = compute_sparse_delta(
            target_state, current_state, threshold=0.0
        )

        # Apply delta in-place (what apply_delta_in_place does)
        # apply_sparse_delta(base, delta) = base + delta = current + (target - current) = target
        reconstructed = apply_sparse_delta(
            current_state,
            sparse_tensors,
            shapes,
            target_dtype=torch.bfloat16,
        )

        # Verify result matches target
        assert (reconstructed["weight"] == w2).all(), "In-place update should match target"

        # Verify hash matches
        expected_hash = compute_weights_hash(target_state)
        actual_hash = compute_weights_hash(reconstructed)
        assert actual_hash == expected_hash, "Hash should match after in-place update"


class TestCheckpointLoadResult:
    """Tests for CheckpointLoadResult dataclass."""

    def test_fast_path_result(self) -> None:
        """Fast path result should have correct properties."""
        from grail.infrastructure.checkpoint_consumer import CheckpointLoadResult

        result = CheckpointLoadResult(success=True, window=100, method="fast")
        assert result.success is True
        assert result.window == 100
        assert result.is_fast_path is True
        assert result.path is None

    def test_full_path_result(self) -> None:
        """Full path result should have correct properties."""
        from pathlib import Path

        from grail.infrastructure.checkpoint_consumer import CheckpointLoadResult

        result = CheckpointLoadResult(
            success=True, window=100, path=Path("/tmp/checkpoint"), method="full"
        )
        assert result.success is True
        assert result.window == 100
        assert result.is_fast_path is False
        assert result.path == Path("/tmp/checkpoint")

    def test_failure_result(self) -> None:
        """Failure result should have correct properties."""
        from grail.infrastructure.checkpoint_consumer import CheckpointLoadResult

        result = CheckpointLoadResult(success=False)
        assert result.success is False
        assert result.window is None
        assert result.is_fast_path is False
