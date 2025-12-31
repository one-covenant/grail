"""Tests for ModelAnalysisManager."""

import torch
import torch.nn as nn

from grail.trainer.analysis import (
    AnalysisConfig,
    ModelAnalysisManager,
    ParameterChangeMetrics,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_manager_creation():
    """Test creating a manager with factory method."""
    config = AnalysisConfig(interval=10)
    manager = ModelAnalysisManager.create(config)

    assert len(manager) == 2  # ParameterChange + SparseQuality (both enabled by default)
    assert manager.step_count == 0


def test_manager_builder_pattern():
    """Test building manager with custom metrics."""
    config = AnalysisConfig(
        interval=10,
        param_change_enabled=False,
        sparse_quality_enabled=False,
    )

    manager = ModelAnalysisManager(config).add_metric(ParameterChangeMetrics(thresholds=[1e-6]))

    assert len(manager) == 1


def test_manager_interval():
    """Test that metrics are only computed at intervals."""
    config = AnalysisConfig(interval=5)
    manager = ModelAnalysisManager.create(config)
    model = SimpleModel()

    # Steps 1-4: No metrics
    for i in range(1, 5):
        metrics = manager.on_optimizer_step(model)
        assert metrics == {}
        assert manager.step_count == i

    # Step 5: First snapshot (no metrics yet)
    metrics = manager.on_optimizer_step(model)
    assert metrics == {}
    assert manager.step_count == 5

    # Modify model
    with torch.no_grad():
        model.linear.weight.data += 0.1

    # Steps 6-9: No metrics
    for _i in range(6, 10):
        metrics = manager.on_optimizer_step(model)
        assert metrics == {}

    # Step 10: Metrics computed
    metrics = manager.on_optimizer_step(model)
    assert len(metrics) > 0  # Should have metrics now
    assert "param_change/norm_l2" in metrics


def test_manager_reset():
    """Test resetting manager state."""
    config = AnalysisConfig(interval=5)
    manager = ModelAnalysisManager.create(config)
    model = SimpleModel()

    # Advance to step 10
    for _ in range(10):
        manager.on_optimizer_step(model)

    assert manager.step_count == 10

    # Reset
    manager.reset()

    assert manager.step_count == 0
    assert manager.old_snapshot is None


def test_manager_state_dict():
    """Test saving and loading state."""
    config = AnalysisConfig(interval=5)
    manager = ModelAnalysisManager.create(config)
    model = SimpleModel()

    # Advance to step 10
    for _ in range(10):
        manager.on_optimizer_step(model)

    # Save state
    state = manager.state_dict()
    assert state["step_count"] == 10

    # Create new manager and load state
    new_manager = ModelAnalysisManager.create(config)
    new_manager.load_state_dict(state)

    assert new_manager.step_count == 10


def test_manager_minimal_config():
    """Test minimal configuration."""
    config = AnalysisConfig.minimal()
    manager = ModelAnalysisManager.create(config)

    assert config.interval == 500
    assert config.param_change_enabled is True
    assert config.sparse_quality_enabled is False
    assert len(manager) == 1  # Only param change


def test_manager_comprehensive_config():
    """Test comprehensive configuration."""
    config = AnalysisConfig.comprehensive()

    assert config.interval == 50
    assert config.param_change_enabled is True
    assert config.sparse_quality_enabled is True
    assert config.param_change_per_layer is True
