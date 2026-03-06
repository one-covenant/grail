"""Unit tests for AdamSignDescentMetrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from grail.trainer.analysis.metrics import AnalysisContext
from grail.trainer.analysis.metrics.adam_sign_descent import (
    AdamSignDescentMetrics,
    _extract_component,
    _extract_layer_idx,
    _reservoir_sample_batch,
)

# ════════════════════════════════════════════════════════════════════════════
# TEST MODELS
# ════════════════════════════════════════════════════════════════════════════


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TransformerLikeModel(nn.Module):
    """Model with transformer-like naming for component extraction tests."""

    def __init__(self) -> None:
        super().__init__()
        # Simulate transformer layer naming
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, 32)
        self.model.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(32, 32, bias=False)
            layer.self_attn.k_proj = nn.Linear(32, 32, bias=False)
            layer.self_attn.v_proj = nn.Linear(32, 32, bias=False)
            layer.self_attn.o_proj = nn.Linear(32, 32, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(32, 64, bias=False)
            layer.mlp.up_proj = nn.Linear(32, 64, bias=False)
            layer.mlp.down_proj = nn.Linear(64, 32, bias=False)
            self.model.layers.append(layer)
        self.lm_head = nn.Linear(32, 100, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # Not used in tests


# ════════════════════════════════════════════════════════════════════════════
# OPTIMIZER HELPER
# ════════════════════════════════════════════════════════════════════════════


def _create_adam_and_step(model: nn.Module, n_steps: int = 5) -> torch.optim.AdamW:
    """Create an AdamW optimizer and run n_steps to populate state.

    Returns the optimizer with populated exp_avg and exp_avg_sq for all params.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for _ in range(n_steps):
        x = torch.randn(4, 10) if isinstance(model, SimpleModel) else torch.randint(0, 100, (4,))
        if isinstance(model, SimpleModel):
            loss = model(x).sum()
        else:
            # For TransformerLikeModel, just do a simple forward on embed_tokens
            loss = model.model.embed_tokens(x).sum()
            # Also need gradients for other params — add a dummy sum
            for p in model.parameters():
                if p.requires_grad:
                    loss = loss + p.sum() * 0.0  # ensures grad flows

        loss.backward()
        optimizer.step()
        # Don't zero_grad on last step — we need gradients available for compute()
        if _ < n_steps - 1:
            optimizer.zero_grad()

    return optimizer


# ════════════════════════════════════════════════════════════════════════════
# BASIC FUNCTIONALITY TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_basic() -> None:
    """Test basic Adam sign descent metric computation."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=3)

    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Core metrics should exist
    assert "adam_sign/update_ratio_mean" in metrics
    assert "adam_sign/update_ratio_std" in metrics
    assert "adam_sign/frac_within_10pct_of_lr" in metrics
    assert "adam_sign/frac_within_50pct_of_lr" in metrics
    assert "adam_sign/frac_within_2x_of_lr" in metrics
    assert "adam_sign/grad_sign_agreement" in metrics
    assert "adam_sign/sign_agreement_with_momentum" in metrics
    assert "adam_sign/effective_lr_mean" in metrics
    assert "adam_sign/lr" in metrics
    assert "adam_sign/bias_correction_m" in metrics
    assert "adam_sign/bias_correction_v" in metrics
    assert "adam_sign/optimizer_step_count" in metrics
    assert "adam_sign/gradient_snr_mean" in metrics

    # Median from reservoir
    assert "adam_sign/update_ratio_median" in metrics
    assert "adam_sign/gradient_snr_median" in metrics

    # Novel metrics
    assert "adam_sign/update_magnitude_entropy" in metrics
    assert "adam_sign/effective_bits_per_update" in metrics

    # Histograms
    assert "adam_sign/_histogram/update_ratio" in metrics
    assert "adam_sign/_histogram/norm_ratio" in metrics
    assert "adam_sign/_histogram/gradient_snr" in metrics
    assert "adam_sign/_histogram/log_update_ratio" in metrics


def test_adam_sign_metric_values_valid() -> None:
    """Test that metric values are in valid ranges."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=5)

    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Fractions should be between 0 and 1
    assert 0.0 <= metrics["adam_sign/frac_within_10pct_of_lr"] <= 1.0
    assert 0.0 <= metrics["adam_sign/frac_within_50pct_of_lr"] <= 1.0
    assert 0.0 <= metrics["adam_sign/frac_within_2x_of_lr"] <= 1.0

    # Band ordering: 10% ⊂ 50% ⊂ 2x
    assert (
        metrics["adam_sign/frac_within_10pct_of_lr"] <= metrics["adam_sign/frac_within_50pct_of_lr"]
    )
    assert metrics["adam_sign/frac_within_50pct_of_lr"] <= metrics["adam_sign/frac_within_2x_of_lr"]

    # Sign agreement between 0 and 1
    assert 0.0 <= metrics["adam_sign/grad_sign_agreement"] <= 1.0
    assert 0.0 <= metrics["adam_sign/sign_agreement_with_momentum"] <= 1.0

    # Mean absolute ratio should be positive
    assert metrics["adam_sign/update_ratio_mean"] > 0.0

    # Std should be non-negative
    assert metrics["adam_sign/update_ratio_std"] >= 0.0

    # Learning rate should match optimizer
    assert metrics["adam_sign/lr"] == pytest.approx(1e-3)

    # Optimizer step count
    assert metrics["adam_sign/optimizer_step_count"] == 5.0

    # Bias corrections should be > 1 (since beta^t < 1)
    assert metrics["adam_sign/bias_correction_m"] > 1.0
    assert metrics["adam_sign/bias_correction_v"] > 1.0

    # SNR should be non-negative
    assert metrics["adam_sign/gradient_snr_mean"] >= 0.0

    # Entropy should be non-negative
    assert metrics["adam_sign/update_magnitude_entropy"] >= 0.0


def test_adam_sign_histogram_types() -> None:
    """Test that histogram values are numpy arrays of float32."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=3)

    computer = AdamSignDescentMetrics(histogram_samples=1000)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    for key in [
        "adam_sign/_histogram/update_ratio",
        "adam_sign/_histogram/norm_ratio",
        "adam_sign/_histogram/gradient_snr",
        "adam_sign/_histogram/log_update_ratio",
    ]:
        assert key in metrics
        assert isinstance(metrics[key], np.ndarray)
        assert metrics[key].dtype == np.float32
        assert len(metrics[key]) > 0


def test_adam_sign_histogram_disabled() -> None:
    """Test that histograms are not produced when disabled."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=3)

    computer = AdamSignDescentMetrics(histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Scalar metrics should still exist
    assert "adam_sign/update_ratio_mean" in metrics

    # Histograms should NOT exist
    assert "adam_sign/_histogram/update_ratio" not in metrics
    assert "adam_sign/_histogram/norm_ratio" not in metrics

    # Median should NOT exist (requires reservoir)
    assert "adam_sign/update_ratio_median" not in metrics


# ════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_no_model() -> None:
    """Test that metric returns empty dict when context has no model."""
    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=None)
    metrics = computer.compute(context=context)
    assert metrics == {}


def test_adam_sign_no_optimizer() -> None:
    """Test that metric returns empty dict when context has no optimizer."""
    model = SimpleModel()
    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=model, optimizer=None)
    metrics = computer.compute(context=context)
    assert metrics == {}


def test_adam_sign_no_context() -> None:
    """Test that metric returns empty dict when context is None."""
    computer = AdamSignDescentMetrics()
    metrics = computer.compute(context=None)
    assert metrics == {}


def test_adam_sign_no_gradients() -> None:
    """Test that metric returns empty dict when no gradients exist."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Don't run any steps, so no optimizer state
    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)
    assert metrics == {}


def test_adam_sign_empty_param_groups() -> None:
    """Test with optimizer that has no param groups."""
    model = SimpleModel()
    optimizer = MagicMock()
    optimizer.param_groups = []
    optimizer.state = {}

    computer = AdamSignDescentMetrics()
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)
    assert metrics == {}


# ════════════════════════════════════════════════════════════════════════════
# REQUIREMENT DECLARATION TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_requires_model() -> None:
    """Test that metric declares model requirement."""
    computer = AdamSignDescentMetrics()
    assert computer.requires_model() is True


def test_adam_sign_requires_optimizer() -> None:
    """Test that metric declares optimizer requirement."""
    computer = AdamSignDescentMetrics()
    assert computer.requires_optimizer() is True


def test_adam_sign_does_not_require_inputs() -> None:
    """Test that metric does not require inputs."""
    computer = AdamSignDescentMetrics()
    assert computer.requires_inputs() is False


def test_adam_sign_name() -> None:
    """Test metric name."""
    computer = AdamSignDescentMetrics()
    assert computer.name == "AdamSignDescentMetrics"


# ════════════════════════════════════════════════════════════════════════════
# PER-COMPONENT TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_per_component_enabled() -> None:
    """Test per-component metrics with transformer-like model."""
    model = TransformerLikeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run a few steps to populate optimizer state with non-zero gradients.
    # Use p.sum() (not * 0.0) so all params get real gradients.
    for _step_i in range(3):
        optimizer.zero_grad()
        x = torch.randint(0, 100, (4,))
        loss = model.model.embed_tokens(x).sum()
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + p.sum()
        loss.backward()
        optimizer.step()

    # Need gradients available for compute() — run one more forward/backward without zero_grad
    x = torch.randint(0, 100, (4,))
    loss = model.model.embed_tokens(x).sum()
    for p in model.parameters():
        if p.requires_grad:
            loss = loss + p.sum()
    loss.backward()

    computer = AdamSignDescentMetrics(track_per_component=True)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Should have per-component metrics for recognized components
    component_keys = [k for k in metrics if k.startswith("adam_sign/component/")]
    assert len(component_keys) > 0

    # Check specific components exist
    expected_components = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]
    for comp in expected_components:
        assert f"adam_sign/component/{comp}/frac_within_10pct_of_lr" in metrics, (
            f"Missing frac_within_10pct_of_lr for {comp}"
        )
        assert f"adam_sign/component/{comp}/norm_ratio_mean_abs" in metrics, (
            f"Missing norm_ratio_mean_abs for {comp}"
        )
        assert f"adam_sign/component/{comp}/gradient_snr_mean" in metrics

    # Values should be valid (non-zero since all params have real gradients)
    for comp in expected_components:
        assert 0.0 <= metrics[f"adam_sign/component/{comp}/frac_within_10pct_of_lr"] <= 1.0
        assert metrics[f"adam_sign/component/{comp}/norm_ratio_mean_abs"] > 0.0
        assert metrics[f"adam_sign/component/{comp}/gradient_snr_mean"] >= 0.0


def test_adam_sign_per_component_disabled() -> None:
    """Test that per-component metrics are not produced when disabled."""
    model = TransformerLikeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        x = torch.randint(0, 100, (4,))
        loss = model.model.embed_tokens(x).sum()
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + p.sum() * 0.0
        loss.backward()
        optimizer.step()
        if _ < 2:
            optimizer.zero_grad()

    computer = AdamSignDescentMetrics(track_per_component=False)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    component_keys = [k for k in metrics if k.startswith("adam_sign/component/")]
    assert len(component_keys) == 0


# ════════════════════════════════════════════════════════════════════════════
# PER-LAYER TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_per_layer_enabled() -> None:
    """Test per-layer metrics with transformer-like model."""
    model = TransformerLikeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        x = torch.randint(0, 100, (4,))
        loss = model.model.embed_tokens(x).sum()
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + p.sum() * 0.0
        loss.backward()
        optimizer.step()
        if _ < 2:
            optimizer.zero_grad()

    computer = AdamSignDescentMetrics(track_per_layer=True, track_per_component=False)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Should have per-layer metrics for layers 0 and 1
    assert "adam_sign/layer_0/norm_ratio_mean_abs" in metrics
    assert "adam_sign/layer_0/frac_within_10pct_of_lr" in metrics
    assert "adam_sign/layer_1/norm_ratio_mean_abs" in metrics
    assert "adam_sign/layer_1/frac_within_10pct_of_lr" in metrics

    # Fractions should be valid
    assert 0.0 <= metrics["adam_sign/layer_0/frac_within_10pct_of_lr"] <= 1.0
    assert 0.0 <= metrics["adam_sign/layer_1/frac_within_10pct_of_lr"] <= 1.0


def test_adam_sign_per_layer_disabled() -> None:
    """Test that per-layer metrics are not produced when disabled."""
    model = TransformerLikeModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(3):
        x = torch.randint(0, 100, (4,))
        loss = model.model.embed_tokens(x).sum()
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + p.sum() * 0.0
        loss.backward()
        optimizer.step()
        if _ < 2:
            optimizer.zero_grad()

    computer = AdamSignDescentMetrics(track_per_layer=False, track_per_component=False)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    layer_keys = [k for k in metrics if "adam_sign/layer_" in k]
    assert len(layer_keys) == 0


# ════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL CORRECTNESS TESTS
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_ratio_known_values() -> None:
    """Test Adam ratio computation against manually computed values.

    When all gradients are the same constant, after many steps:
    - m_t converges to that constant
    - v_t converges to constant^2
    - ratio = m_hat / (sqrt(v_hat) + eps) ≈ sign(constant)
    """
    model = nn.Linear(10, 1, bias=False)  # 10 params
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    # Use constant gradients for predictable behavior
    for _i in range(100):
        optimizer.zero_grad()
        # Set a constant gradient
        model.weight.grad = torch.ones_like(model.weight) * 0.5
        optimizer.step()

    # Don't zero_grad — keep the last gradient
    model.weight.grad = torch.ones_like(model.weight) * 0.5

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # After 100 steps of constant gradient:
    # m_t ≈ g (converged to gradient)
    # v_t ≈ g^2 (converged to gradient squared)
    # ratio = m_hat / (sqrt(v_hat) + eps) ≈ g / (|g| + eps) ≈ sign(g) ≈ 1.0
    assert metrics["adam_sign/update_ratio_mean"] == pytest.approx(1.0, abs=0.05)

    # All ratios should be near 1.0
    assert metrics["adam_sign/frac_within_10pct_of_lr"] == pytest.approx(1.0, abs=0.01)

    # Sign agreement should be 1.0 (all gradients positive, momentum positive)
    assert metrics["adam_sign/grad_sign_agreement"] == pytest.approx(1.0, abs=0.01)


def test_adam_sign_negative_gradients() -> None:
    """Test that negative gradients produce ratio ≈ -1 (abs ≈ 1)."""
    model = nn.Linear(10, 1, bias=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    for _ in range(100):
        optimizer.zero_grad()
        model.weight.grad = torch.ones_like(model.weight) * (-0.3)
        optimizer.step()

    model.weight.grad = torch.ones_like(model.weight) * (-0.3)

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # ratio should be ≈ -1 in signed form, so |ratio| ≈ 1
    assert metrics["adam_sign/update_ratio_mean"] == pytest.approx(1.0, abs=0.05)
    assert metrics["adam_sign/frac_within_10pct_of_lr"] == pytest.approx(1.0, abs=0.01)


def test_adam_sign_bias_correction_values() -> None:
    """Test that bias correction values match expected formulas."""
    model = SimpleModel()
    beta1, beta2 = 0.9, 0.999
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(beta1, beta2))

    n_steps = 7
    for i in range(n_steps):
        optimizer.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        if i < n_steps - 1:
            optimizer.zero_grad()

    # Re-create gradient for last step
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    expected_bc_m = 1.0 / (1.0 - beta1**n_steps)
    expected_bc_v = 1.0 / (1.0 - beta2**n_steps)

    assert metrics["adam_sign/bias_correction_m"] == pytest.approx(expected_bc_m, rel=1e-5)
    assert metrics["adam_sign/bias_correction_v"] == pytest.approx(expected_bc_v, rel=1e-5)
    assert metrics["adam_sign/optimizer_step_count"] == float(n_steps)


def test_adam_sign_effective_lr() -> None:
    """Test that effective_lr_mean = lr * mean(|ratio|)."""
    model = SimpleModel()
    lr = 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(5):
        optimizer.zero_grad()
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        if i < 4:
            optimizer.zero_grad()

    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # effective_lr_mean should equal lr * update_ratio_mean
    expected = lr * metrics["adam_sign/update_ratio_mean"]
    assert metrics["adam_sign/effective_lr_mean"] == pytest.approx(expected, rel=1e-5)


def test_adam_sign_snr_positive() -> None:
    """Test gradient SNR is positive when gradients are non-zero."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=5)

    computer = AdamSignDescentMetrics(histogram_samples=0, track_per_component=False)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    assert metrics["adam_sign/gradient_snr_mean"] > 0.0


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestExtractComponent:
    """Tests for _extract_component helper."""

    def test_q_proj(self) -> None:
        assert _extract_component("model.layers.0.self_attn.q_proj.weight") == "q_proj"

    def test_k_proj(self) -> None:
        assert _extract_component("model.layers.0.self_attn.k_proj.weight") == "k_proj"

    def test_v_proj(self) -> None:
        assert _extract_component("model.layers.5.self_attn.v_proj.weight") == "v_proj"

    def test_o_proj(self) -> None:
        assert _extract_component("model.layers.3.self_attn.o_proj.weight") == "o_proj"

    def test_gate_proj(self) -> None:
        assert _extract_component("model.layers.0.mlp.gate_proj.weight") == "gate_proj"

    def test_up_proj(self) -> None:
        assert _extract_component("model.layers.0.mlp.up_proj.weight") == "up_proj"

    def test_down_proj(self) -> None:
        assert _extract_component("model.layers.0.mlp.down_proj.weight") == "down_proj"

    def test_embed_tokens(self) -> None:
        assert _extract_component("model.embed_tokens.weight") == "embed_tokens"

    def test_lm_head(self) -> None:
        assert _extract_component("lm_head.weight") == "lm_head"

    def test_layernorm(self) -> None:
        assert _extract_component("model.layers.0.input_layernorm.weight") == "layernorm"
        assert _extract_component("model.layers.0.post_attention_layernorm.weight") == "layernorm"
        assert _extract_component("model.norm.weight") == "layernorm"

    def test_unknown(self) -> None:
        assert _extract_component("model.some_unknown_module.weight") is None


class TestExtractLayerIdx:
    """Tests for _extract_layer_idx helper."""

    def test_layer_0(self) -> None:
        assert _extract_layer_idx("model.layers.0.self_attn.q_proj.weight") == 0

    def test_layer_5(self) -> None:
        assert _extract_layer_idx("model.layers.5.mlp.gate_proj.weight") == 5

    def test_layer_23(self) -> None:
        assert _extract_layer_idx("model.layers.23.self_attn.k_proj.weight") == 23

    def test_no_layer(self) -> None:
        assert _extract_layer_idx("model.embed_tokens.weight") is None
        assert _extract_layer_idx("lm_head.weight") is None


# ════════════════════════════════════════════════════════════════════════════
# RESERVOIR SAMPLING TESTS
# ════════════════════════════════════════════════════════════════════════════


class TestReservoirSampling:
    """Tests for _reservoir_sample_batch."""

    def test_filling_phase(self) -> None:
        """Test that reservoir fills up to max_samples."""
        reservoir: list[float] = []
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        _reservoir_sample_batch(values, reservoir, current_idx=0, max_samples=10)
        assert len(reservoir) == 5
        assert reservoir == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_fills_exactly_to_max(self) -> None:
        """Test reservoir fills exactly to max_samples."""
        reservoir: list[float] = []
        values = np.arange(20, dtype=np.float32)
        _reservoir_sample_batch(values, reservoir, current_idx=0, max_samples=20)
        assert len(reservoir) == 20

    def test_overflow_maintains_size(self) -> None:
        """Test that reservoir doesn't grow beyond max_samples."""
        np.random.seed(42)
        reservoir: list[float] = list(range(10))
        values = np.arange(100, 200, dtype=np.float32)
        _reservoir_sample_batch(values, reservoir, current_idx=10, max_samples=10)
        assert len(reservoir) == 10

    def test_empty_input(self) -> None:
        """Test with empty input array."""
        reservoir: list[float] = [1.0, 2.0]
        values = np.array([], dtype=np.float32)
        _reservoir_sample_batch(values, reservoir, current_idx=2, max_samples=10)
        assert reservoir == [1.0, 2.0]

    def test_multiple_batches_fill(self) -> None:
        """Test filling reservoir across multiple batches."""
        reservoir: list[float] = []
        max_samples = 10

        # First batch: 4 items
        _reservoir_sample_batch(np.array([1, 2, 3, 4], dtype=np.float32), reservoir, 0, max_samples)
        assert len(reservoir) == 4

        # Second batch: 4 more items
        _reservoir_sample_batch(np.array([5, 6, 7, 8], dtype=np.float32), reservoir, 4, max_samples)
        assert len(reservoir) == 8

        # Third batch: crosses max_samples boundary
        _reservoir_sample_batch(
            np.array([9, 10, 11, 12], dtype=np.float32), reservoir, 8, max_samples
        )
        assert len(reservoir) == 10

    def test_statistical_uniformity(self) -> None:
        """Test that reservoir sampling produces approximately uniform samples.

        After seeing N >> max_samples items, each item should have equal
        probability of being in the reservoir.
        """
        np.random.seed(123)
        max_samples = 1000
        n_total = 100_000
        n_trials = 50

        # Track how often each "region" appears in reservoir
        region_counts = np.zeros(10)  # 10 regions of 10k items each

        for _ in range(n_trials):
            reservoir: list[float] = []
            batch_size = 5000
            for start in range(0, n_total, batch_size):
                values = np.arange(start, min(start + batch_size, n_total), dtype=np.float32)
                _reservoir_sample_batch(values, reservoir, start, max_samples)

            # Count items from each region
            arr = np.array(reservoir)
            for r in range(10):
                lo, hi = r * 10000, (r + 1) * 10000
                region_counts[r] += np.sum((arr >= lo) & (arr < hi))

        # Each region should get ~100 samples per trial (1000/10)
        # Over 50 trials, expected = 5000 per region
        expected = max_samples / 10 * n_trials
        for r in range(10):
            # Allow 20% deviation
            assert abs(region_counts[r] - expected) / expected < 0.2, (
                f"Region {r}: got {region_counts[r]}, expected ~{expected}"
            )


# ════════════════════════════════════════════════════════════════════════════
# BIMODAL COEFFICIENT TEST
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_bimodal_coefficient_exists() -> None:
    """Test that bimodal coefficient is computed when conditions are met."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=5)

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Bimodal coefficient should exist (variance > 0 for non-trivial gradients)
    assert "adam_sign/bimodal_coefficient" in metrics
    assert metrics["adam_sign/bimodal_coefficient"] > 0.0


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH CONFIG AND MANAGER
# ════════════════════════════════════════════════════════════════════════════


def test_config_adam_sign_fields() -> None:
    """Test that AnalysisConfig has adam_sign fields."""
    from grail.trainer.analysis.config import AnalysisConfig

    config = AnalysisConfig()
    assert config.adam_sign_enabled is False
    assert config.adam_sign_track_per_component is True
    assert config.adam_sign_track_per_layer is False
    assert config.adam_sign_histogram_samples == 1_000_000
    assert config.adam_sign_near_lr_tolerance == 0.1


def test_config_comprehensive_enables_adam_sign() -> None:
    """Test that comprehensive config enables adam_sign."""
    from grail.trainer.analysis.config import AnalysisConfig

    config = AnalysisConfig.comprehensive()
    assert config.adam_sign_enabled is True
    assert config.adam_sign_track_per_component is True


def test_config_to_dict_includes_adam_sign() -> None:
    """Test that to_dict includes adam_sign fields."""
    from grail.trainer.analysis.config import AnalysisConfig

    config = AnalysisConfig(adam_sign_enabled=True)
    d = config.to_dict()
    assert "adam_sign_enabled" in d
    assert "adam_sign_track_per_component" in d
    assert "adam_sign_track_per_layer" in d
    assert "adam_sign_histogram_samples" in d
    assert "adam_sign_near_lr_tolerance" in d
    assert d["adam_sign_enabled"] is True


def test_manager_create_with_adam_sign() -> None:
    """Test that ModelAnalysisManager.create adds AdamSignDescentMetrics when enabled."""
    from grail.trainer.analysis import AnalysisConfig, ModelAnalysisManager

    config = AnalysisConfig(
        adam_sign_enabled=True,
        param_change_enabled=False,
        sparse_quality_enabled=False,
    )
    manager = ModelAnalysisManager.create(config)

    assert len(manager) == 1
    assert manager.metric_computers[0].name == "AdamSignDescentMetrics"


def test_manager_create_without_adam_sign() -> None:
    """Test that ModelAnalysisManager.create does not add AdamSignDescentMetrics when disabled."""
    from grail.trainer.analysis import AnalysisConfig, ModelAnalysisManager

    config = AnalysisConfig(
        adam_sign_enabled=False,
        param_change_enabled=False,
        sparse_quality_enabled=False,
    )
    manager = ModelAnalysisManager.create(config)
    assert len(manager) == 0


def test_manager_end_to_end_with_adam_sign() -> None:
    """Test full end-to-end flow: manager with adam_sign computes metrics."""
    from grail.trainer.analysis import AnalysisConfig, ModelAnalysisManager

    config = AnalysisConfig(
        interval=1,
        adam_sign_enabled=True,
        param_change_enabled=False,
        sparse_quality_enabled=False,
    )
    manager = ModelAnalysisManager.create(config)

    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Step 1: first snapshot (no metrics yet)
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    metrics = manager.on_optimizer_step(model, optimizer=optimizer)
    assert metrics == {}  # First measurement just captures snapshot

    # Step 2: second measurement — should have adam_sign metrics
    optimizer.zero_grad()
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    metrics = manager.on_optimizer_step(model, optimizer=optimizer)

    assert "adam_sign/update_ratio_mean" in metrics
    assert "adam_sign/frac_within_10pct_of_lr" in metrics
    assert "adam_sign/gradient_snr_mean" in metrics


# ════════════════════════════════════════════════════════════════════════════
# CALLBACK HISTOGRAM DETECTION TEST
# ════════════════════════════════════════════════════════════════════════════


def test_histogram_key_detection() -> None:
    """Test that the generalized histogram key detection works for adam_sign.

    Simulates what SparsityCallback does to separate histogram data from scalars.
    """
    # Simulate metrics dict from AdamSignDescentMetrics
    metrics = {
        "adam_sign/update_ratio_mean": 1.0,
        "adam_sign/frac_within_10pct_of_lr": 0.8,
        "adam_sign/_histogram/update_ratio": np.array([1.0, 0.9]),
        "adam_sign/_histogram/norm_ratio": np.array([-1.0, 1.0]),
        "gradient/_histogram/values": np.array([0.01, 0.02]),
        "gradient/norm_l2": 0.5,
    }

    scalar_metrics = {}
    histogram_data = {}

    for key, value in metrics.items():
        if "/_histogram/" in key:
            hist_name = key.replace("/_histogram/", "/")
            histogram_data[hist_name] = value
        elif isinstance(value, (int, float)):
            scalar_metrics[key] = float(value)

    # Scalars
    assert "adam_sign/update_ratio_mean" in scalar_metrics
    assert "adam_sign/frac_within_10pct_of_lr" in scalar_metrics
    assert "gradient/norm_l2" in scalar_metrics

    # Histograms — keys should have /_histogram/ stripped
    assert "adam_sign/update_ratio" in histogram_data
    assert "adam_sign/norm_ratio" in histogram_data
    assert "gradient/values" in histogram_data

    # No histograms in scalars
    assert "adam_sign/_histogram/update_ratio" not in scalar_metrics
    assert "gradient/_histogram/values" not in scalar_metrics


# ════════════════════════════════════════════════════════════════════════════
# TENSOR STEP COUNT HANDLING
# ════════════════════════════════════════════════════════════════════════════


def test_adam_sign_tensor_step_count() -> None:
    """Test that tensor-valued step counts in optimizer state are handled."""
    model = SimpleModel()
    optimizer = _create_adam_and_step(model, n_steps=3)

    # Verify PyTorch stores step as tensor (modern PyTorch behavior)
    for param in model.parameters():
        state = optimizer.state.get(param)
        if state and "step" in state:
            # PyTorch >= 2.0 uses tensor step counts
            assert isinstance(state["step"], (int, float, torch.Tensor))

    computer = AdamSignDescentMetrics(track_per_component=False, histogram_samples=0)
    context = AnalysisContext(model=model, optimizer=optimizer)
    metrics = computer.compute(context=context)

    # Should successfully compute metrics regardless of step type
    assert "adam_sign/update_ratio_mean" in metrics
    assert metrics["adam_sign/optimizer_step_count"] == 3.0
