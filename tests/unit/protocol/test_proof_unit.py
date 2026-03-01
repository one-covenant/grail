"""Unit tests for GRAIL proof verifier components.

Tests individual functions and properties of the GRAIL proof verification system.
"""

import pytest
import torch

from grail.protocol.grail_verifier import (
    GRAILVerifier,
    adaptive_sketch_tolerance,
    log_magnitude_bucket,
    log_magnitude_bucket_vectorized,
)
from grail.shared.constants import (
    PROOF_NUM_BUCKETS,
    PROOF_SKETCH_TOLERANCE_BASE,
)


class TestLogMagnitudeBucket:
    """Test suite for logarithmic magnitude bucketing."""

    def test_near_zero_values_map_to_zero(self) -> None:
        """Near-zero values should map to bucket 0."""
        assert log_magnitude_bucket(0.0) == 0
        assert log_magnitude_bucket(1e-7) == 0
        assert log_magnitude_bucket(-1e-7) == 0

    def test_sign_preservation(self) -> None:
        """Bucket function preserves sign."""
        bucket_pos = log_magnitude_bucket(2.0)
        bucket_neg = log_magnitude_bucket(-2.0)
        assert bucket_pos > 0
        assert bucket_neg < 0
        assert bucket_pos == -bucket_neg

    def test_logarithmic_coarseness(self) -> None:
        """Small differences in small values map to same bucket (robustness)."""
        b1 = log_magnitude_bucket(0.1)
        b2 = log_magnitude_bucket(0.2)
        assert abs(b1 - b2) <= 1  # May differ by at most 1 bucket

    def test_larger_values_get_higher_buckets(self) -> None:
        """Larger magnitude values get higher bucket indices."""
        b_small = log_magnitude_bucket(0.5)
        b_large = log_magnitude_bucket(10.0)
        assert b_large > b_small

    def test_bucket_bounds(self) -> None:
        """Bucket indices are bounded by num_buckets."""
        b_huge = log_magnitude_bucket(1000.0)
        assert abs(b_huge) <= PROOF_NUM_BUCKETS


class TestAdaptiveSketchTolerance:
    """Test suite for adaptive sketch tolerance computation."""

    def test_early_position_tighter(self) -> None:
        """Early positions should have tighter or equal tolerance."""
        tol_early = adaptive_sketch_tolerance(0, 100)
        assert tol_early == PROOF_SKETCH_TOLERANCE_BASE

    def test_late_position_looser(self) -> None:
        """Late positions should have looser tolerance."""
        tol_late = adaptive_sketch_tolerance(99, 100)
        assert tol_late > PROOF_SKETCH_TOLERANCE_BASE

    def test_monotonic_increase(self) -> None:
        """Tolerance should increase monotonically with position."""
        tol_early = adaptive_sketch_tolerance(0, 100)
        tol_mid = adaptive_sketch_tolerance(50, 100)
        tol_late = adaptive_sketch_tolerance(99, 100)
        assert tol_early <= tol_mid <= tol_late


class TestGRAILVerifier:
    """Test suite for GRAILVerifier class."""

    @pytest.fixture
    def verifier(self) -> GRAILVerifier:
        """GRAILVerifier fixture."""
        return GRAILVerifier(hidden_dim=4096, topk=256)

    @pytest.fixture
    def randomness(self) -> str:
        """Test randomness fixture."""
        return "feedbeefcafebabe1234567890abcdef"

    def test_r_vec_generation_shape(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Generated r_vec should have correct shape."""
        r_vec = verifier.generate_r_vec(randomness)
        assert r_vec.shape == (256,)

    def test_r_vec_generation_bounds(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Generated r_vec coefficients should be bounded."""
        r_vec = verifier.generate_r_vec(randomness)
        assert r_vec.min() >= -127
        assert r_vec.max() <= 127

    def test_r_vec_generation_deterministic(self, verifier: GRAILVerifier, randomness: str) -> None:
        """r_vec generation should be deterministic."""
        r_vec1 = verifier.generate_r_vec(randomness)
        r_vec2 = verifier.generate_r_vec(randomness)
        assert torch.equal(r_vec1, r_vec2)

    def test_r_vec_different_randomness(self, verifier: GRAILVerifier) -> None:
        """Different randomness should produce different r_vec."""
        r_vec1 = verifier.generate_r_vec("feedbeef")
        r_vec2 = verifier.generate_r_vec("deadc0de")
        assert not torch.equal(r_vec1, r_vec2)

    def test_commitment_structure(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Commitment should have correct structure."""
        r_vec = verifier.generate_r_vec(randomness)
        hidden = torch.randn(4096)
        commitment = verifier.create_commitment(hidden, r_vec, position=0)

        assert "sketch" in commitment
        assert "indices" in commitment
        assert "position" in commitment

    def test_commitment_sizes(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Commitment components should have correct sizes."""
        r_vec = verifier.generate_r_vec(randomness)
        hidden = torch.randn(4096)
        commitment = verifier.create_commitment(hidden, r_vec, position=0)

        assert len(commitment["indices"]) == 256

    def test_self_verification(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Commitment should verify against itself perfectly."""
        r_vec = verifier.generate_r_vec(randomness)
        hidden = torch.randn(4096)
        commitment = verifier.create_commitment(hidden, r_vec, position=0)

        is_valid, diagnostics = verifier.verify_commitment(
            hidden, commitment, r_vec, sequence_length=100
        )

        assert is_valid
        assert diagnostics["sketch_diff"] == 0

    def test_robustness_to_small_drift(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Small drift should not break verification when top activations are well-separated.

        Realistic scenario: Important activations in trained models are clearly separated,
        so tiny numerical drift from framework/GPU differences shouldn't change bucketing.
        """
        r_vec = verifier.generate_r_vec(randomness)

        # Build realistic hidden state: well-separated top activations with background noise
        hidden = torch.randn(4096) * 0.1
        top_indices = [10, 20, 30, 40, 50]
        hidden[top_indices] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

        commitment = verifier.create_commitment(hidden, r_vec, position=0)

        # Add small drift (simulating cross-framework variance)
        hidden_drifted = hidden + torch.randn(4096) * 1e-5

        is_valid, diagnostics = verifier.verify_commitment(
            hidden_drifted, commitment, r_vec, sequence_length=100
        )

        assert is_valid, (
            f"Well-separated activations should be robust to small drift: {diagnostics}"
        )
        assert diagnostics["sketch_diff"] < PROOF_SKETCH_TOLERANCE_BASE

    def test_rejects_different_hidden_state(self, verifier: GRAILVerifier, randomness: str) -> None:
        """Completely different hidden state should be rejected."""
        torch.manual_seed(7)
        r_vec = verifier.generate_r_vec(randomness)
        hidden1 = torch.randn(4096)
        hidden2 = torch.randn(4096)  # Completely different

        commitment = verifier.create_commitment(hidden1, r_vec, position=0)

        is_valid, diagnostics = verifier.verify_commitment(
            hidden2, commitment, r_vec, sequence_length=100
        )

        assert not is_valid, "Different hidden state should be rejected"


class TestVectorizedBucketing:
    """Verify log_magnitude_bucket_vectorized matches scalar version exactly."""

    def test_basic_values(self) -> None:
        test_values = [0.0, 1e-7, -1e-7, 1.0, -1.0, 0.1, -0.1, 5.0, -5.0, 100.0, -100.0]
        vectorized = log_magnitude_bucket_vectorized(torch.tensor(test_values)).tolist()
        scalar = [log_magnitude_bucket(v) for v in test_values]
        assert vectorized == scalar

    def test_edge_cases(self) -> None:
        test_values = [
            float("nan"),
            float("inf"),
            float("-inf"),
            1e-6,
            -1e-6,
            1e-7,
            5e-7,
        ]
        vectorized = log_magnitude_bucket_vectorized(torch.tensor(test_values)).tolist()
        scalar = [log_magnitude_bucket(v) for v in test_values]
        assert vectorized == scalar

    def test_random_1000(self) -> None:
        torch.manual_seed(42)
        values = torch.randn(1000) * 3.0
        values[0] = float("nan")
        values[1] = float("inf")
        values[2] = float("-inf")
        values[3] = 0.0
        values[4] = 1e-8
        vectorized = log_magnitude_bucket_vectorized(values).tolist()
        scalar = [log_magnitude_bucket(v.item()) for v in values]
        assert vectorized == scalar

    def test_2d_matches_per_row(self) -> None:
        torch.manual_seed(123)
        values_2d = torch.randn(100, 32) * 5.0
        vectorized = log_magnitude_bucket_vectorized(values_2d)
        for row in range(100):
            for col in range(32):
                s = log_magnitude_bucket(values_2d[row, col].item())
                assert vectorized[row, col].item() == s, (
                    f"Mismatch at [{row},{col}]: vec={vectorized[row, col].item()}, "
                    f"scalar={s}, input={values_2d[row, col].item()}"
                )


class TestCreateCommitmentsBatch:
    """Verify create_commitments_batch matches per-position scalar loop."""

    @pytest.fixture
    def verifier(self) -> GRAILVerifier:
        return GRAILVerifier(hidden_dim=3584, topk=32)

    @pytest.fixture
    def randomness(self) -> str:
        return "feedbeefcafebabe1234567890abcdef"

    def test_batch_matches_scalar(self, verifier: GRAILVerifier, randomness: str) -> None:
        r_vec = verifier.generate_r_vec(randomness)
        torch.manual_seed(99)
        h_layer = torch.randn(500, 3584)

        scalar_commitments = [
            verifier.create_commitment(h_layer[pos], r_vec, pos) for pos in range(500)
        ]
        batch_commitments = verifier.create_commitments_batch(h_layer, r_vec)

        assert len(batch_commitments) == len(scalar_commitments)
        for pos in range(500):
            assert batch_commitments[pos]["position"] == pos
            assert batch_commitments[pos]["sketch"] == scalar_commitments[pos]["sketch"], (
                f"Sketch mismatch at pos {pos}: "
                f"batch={batch_commitments[pos]['sketch']}, scalar={scalar_commitments[pos]['sketch']}"
            )
            assert batch_commitments[pos]["indices"] == scalar_commitments[pos]["indices"]

    def test_edge_case_hidden_states(self, verifier: GRAILVerifier, randomness: str) -> None:
        r_vec = verifier.generate_r_vec(randomness)
        h_layer = torch.randn(10, 3584)
        h_layer[0, :] = 0.0
        h_layer[1, :] = float("inf")
        h_layer[2, :] = float("-inf")
        h_layer[3, :100] = float("nan")

        scalar_commitments = [
            verifier.create_commitment(h_layer[pos], r_vec, pos) for pos in range(10)
        ]
        batch_commitments = verifier.create_commitments_batch(h_layer, r_vec)

        for pos in range(10):
            assert batch_commitments[pos] == scalar_commitments[pos], f"Mismatch at pos {pos}"

    def test_batch_self_verification(self, verifier: GRAILVerifier, randomness: str) -> None:
        r_vec = verifier.generate_r_vec(randomness)
        torch.manual_seed(42)
        h_layer = torch.randn(100, 3584)

        batch_commitments = verifier.create_commitments_batch(h_layer, r_vec)

        for pos in [0, 10, 25, 50, 75, 99]:
            is_valid, diagnostics = verifier.verify_commitment(
                h_layer[pos], batch_commitments[pos], r_vec, sequence_length=100
            )
            assert is_valid, f"Self-verification failed at pos {pos}: {diagnostics}"
            assert diagnostics["sketch_diff"] == 0
