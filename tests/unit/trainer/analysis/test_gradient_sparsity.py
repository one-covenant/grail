"""Unit tests for GradientSparsityMetrics."""

from __future__ import annotations

import torch
import torch.nn as nn

from grail.trainer.analysis.metrics import AnalysisContext, GradientSparsityMetrics


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


def test_gradient_sparsity_basic() -> None:
    """Test basic gradient sparsity computation."""
    model = SimpleModel()

    # Create some gradients
    x = torch.randn(4, 10)
    y = model(x).sum()
    y.backward()

    # Compute metrics
    computer = GradientSparsityMetrics(thresholds=[0.0, 1e-6, 1e-4])
    context = AnalysisContext(model=model)
    metrics = computer.compute(context=context)

    # Verify metrics exist
    assert "gradient/norm_l2" in metrics
    assert "gradient/norm_l1" in metrics
    assert "gradient/norm_max" in metrics
    assert "gradient/mean" in metrics
    assert "gradient/std" in metrics

    # Verify sparsity metrics (including zero threshold)
    assert "gradient/sparsity_at_0e+00" in metrics
    assert "gradient/sparsity_at_1e-06" in metrics
    assert "gradient/sparsity_at_1e-04" in metrics
    assert "gradient/dense_ratio_at_0e+00" in metrics

    # Verify sparsity is between 0 and 1
    assert 0.0 <= metrics["gradient/sparsity_at_0e+00"] <= 1.0
    assert 0.0 <= metrics["gradient/sparsity_at_1e-06"] <= 1.0
    assert 0.0 <= metrics["gradient/sparsity_at_1e-04"] <= 1.0

    # Verify sparsity and dense ratio sum to 1
    assert (
        abs(metrics["gradient/sparsity_at_0e+00"] + metrics["gradient/dense_ratio_at_0e+00"] - 1.0)
        < 1e-5
    )


def test_gradient_sparsity_no_gradients() -> None:
    """Test that metric returns empty dict when no gradients."""
    model = SimpleModel()

    computer = GradientSparsityMetrics()
    context = AnalysisContext(model=model)
    metrics = computer.compute(context=context)

    assert metrics == {}


def test_gradient_sparsity_per_layer() -> None:
    """Test per-layer gradient sparsity tracking."""
    model = SimpleModel()

    # Create some gradients
    x = torch.randn(4, 10)
    y = model(x).sum()
    y.backward()

    # Compute metrics with per-layer tracking
    computer = GradientSparsityMetrics(track_per_layer=True)
    context = AnalysisContext(model=model)
    metrics = computer.compute(context=context)

    # Verify per-layer metrics exist
    per_layer_metrics = [k for k in metrics if "gradient/layer/" in k]
    assert len(per_layer_metrics) > 0

    # Check that we have metrics for each layer
    assert any("linear1" in k for k in per_layer_metrics)
    assert any("linear2" in k for k in per_layer_metrics)


def test_gradient_sparsity_no_model() -> None:
    """Test that metric returns empty dict when context has no model."""
    computer = GradientSparsityMetrics()
    context = AnalysisContext(model=None)
    metrics = computer.compute(context=context)

    assert metrics == {}


def test_gradient_sparsity_requires_model() -> None:
    """Test that metric declares model requirement."""
    computer = GradientSparsityMetrics()
    assert computer.requires_model() is True
    assert computer.requires_inputs() is False
    assert computer.requires_optimizer() is False


def test_gradient_sparsity_name() -> None:
    """Test metric name."""
    computer = GradientSparsityMetrics()
    assert computer.name == "GradientSparsityMetrics"
