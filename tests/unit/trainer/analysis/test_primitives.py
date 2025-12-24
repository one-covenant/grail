"""Tests for analysis primitives (ParameterSnapshot, ParameterDelta)."""

import torch
import torch.nn as nn

from grail.trainer.analysis.primitives import ParameterSnapshot


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def test_parameter_snapshot_creation():
    """Test creating a parameter snapshot."""
    model = SimpleModel()

    snapshot = ParameterSnapshot(model)

    assert len(snapshot) == 4  # 2 weights + 2 biases
    assert "linear1.weight" in snapshot
    assert "linear1.bias" in snapshot
    assert "linear2.weight" in snapshot
    assert "linear2.bias" in snapshot

    # Check device and dtype
    assert snapshot.device == "cpu"
    assert snapshot.dtype == torch.float32


def test_parameter_snapshot_immutable():
    """Test that snapshot data is read-only."""
    model = SimpleModel()
    snapshot = ParameterSnapshot(model)

    # Should not be able to modify snapshot data directly
    # (This is enforced by returning a dict view, not a settable attribute)
    original_weight = snapshot.data["linear1.weight"].clone()

    # Modifying model should not affect snapshot
    model.linear1.weight.data.fill_(42.0)

    assert not torch.allclose(snapshot.data["linear1.weight"], model.linear1.weight.data)
    assert torch.allclose(snapshot.data["linear1.weight"], original_weight)


def test_parameter_delta_computation():
    """Test computing delta between two snapshots."""
    model = SimpleModel()

    # Take initial snapshot
    snapshot1 = ParameterSnapshot(model)

    # Modify model
    with torch.no_grad():
        model.linear1.weight.data += 0.5
        model.linear1.bias.data -= 0.1

    # Take new snapshot
    snapshot2 = ParameterSnapshot(model)

    # Compute delta
    delta = snapshot1.compute_delta(snapshot2)

    assert len(delta) == 4

    # Check that deltas are correct
    assert torch.allclose(
        delta.deltas["linear1.weight"], torch.full_like(delta.deltas["linear1.weight"], 0.5)
    )
    assert torch.allclose(
        delta.deltas["linear1.bias"], torch.full_like(delta.deltas["linear1.bias"], -0.1)
    )
    assert torch.allclose(
        delta.deltas["linear2.weight"], torch.zeros_like(delta.deltas["linear2.weight"])
    )


def test_parameter_delta_statistics():
    """Test delta statistics computation."""
    model = SimpleModel()

    snapshot1 = ParameterSnapshot(model)

    # Make known changes
    with torch.no_grad():
        model.linear1.weight.data += 1.0  # 10x20 = 200 params, each +1.0
        model.linear1.bias.data += 2.0  # 20 params, each +2.0

    snapshot2 = ParameterSnapshot(model)
    delta = snapshot1.compute_delta(snapshot2)

    stats = delta.statistics()

    assert "norm_l2" in stats
    assert "norm_l1" in stats
    assert "norm_max" in stats
    assert "mean" in stats
    assert "std" in stats

    # L1 norm should be: 200*1.0 + 20*2.0 = 240
    expected_l1 = 200 * 1.0 + 20 * 2.0 + 100 * 0.0 + 5 * 0.0
    assert abs(stats["norm_l1"] - expected_l1) < 1e-5

    # Max should be 2.0
    assert abs(stats["norm_max"] - 2.0) < 1e-5


def test_parameter_delta_sparsity():
    """Test sparsity computation at different thresholds."""
    model = SimpleModel()

    snapshot1 = ParameterSnapshot(model)

    # Create varied changes
    with torch.no_grad():
        model.linear1.weight.data += 1e-5  # Above 1e-6 threshold
        model.linear1.bias.data += 1e-10  # Below 1e-6 threshold

    snapshot2 = ParameterSnapshot(model)
    delta = snapshot1.compute_delta(snapshot2)

    sparsity_1e6 = delta.sparsity_at_threshold(1e-6)

    # linear1.weight (200 params) should be kept (above threshold)
    # linear1.bias (20 params) should be dropped (below threshold)
    # linear2.* (105 params) should be dropped (zero)
    total_params = 200 + 20 + 100 + 5  # 325
    kept_params = 200

    assert sparsity_1e6["total_params"] == total_params
    assert sparsity_1e6["kept_params"] == kept_params
    assert abs(sparsity_1e6["kept_ratio"] - (kept_params / total_params)) < 1e-5


def test_parameter_delta_sparse_mask():
    """Test applying sparse mask to delta."""
    model = SimpleModel()

    snapshot1 = ParameterSnapshot(model)

    with torch.no_grad():
        model.linear1.weight.data += 1e-5

    snapshot2 = ParameterSnapshot(model)
    delta = snapshot1.compute_delta(snapshot2)

    # Apply sparse mask at 1e-6
    sparse_delta = delta.apply_sparse_mask(threshold=1e-6)

    # Check that small changes were zeroed
    assert torch.allclose(
        sparse_delta.deltas["linear1.weight"],
        torch.full_like(sparse_delta.deltas["linear1.weight"], 1e-5),
    )
    assert torch.allclose(
        sparse_delta.deltas["linear1.bias"],
        torch.zeros_like(sparse_delta.deltas["linear1.bias"]),
    )


def test_parameter_delta_per_layer_stats():
    """Test per-layer statistics computation."""
    model = SimpleModel()

    snapshot1 = ParameterSnapshot(model)

    with torch.no_grad():
        model.linear1.weight.data += 1.0
        model.linear2.weight.data += 2.0

    snapshot2 = ParameterSnapshot(model)
    delta = snapshot1.compute_delta(snapshot2)

    per_layer = delta.per_layer_statistics()

    assert len(per_layer) == 4
    assert "linear1.weight" in per_layer
    assert "linear2.weight" in per_layer

    # Check that means are correct
    assert abs(per_layer["linear1.weight"]["mean"] - 1.0) < 1e-5
    assert abs(per_layer["linear2.weight"]["mean"] - 2.0) < 1e-5
