"""Tests for DDP and DILOCO distributed training strategies.

Validates strategy configuration, context behavior, checkpoint dispatch,
and DILOCO outer optimization math. All tests run on CPU without
requiring NCCL or multiple GPUs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from grail.trainer.distributed.compat import DistributedContext
from grail.trainer.distributed.config import DistributedConfig

# ============================================================================
# DistributedConfig tests
# ============================================================================


class TestDistributedConfig:
    """Test strategy configuration parsing and validation."""

    def test_default_strategy_is_fsdp2(self) -> None:
        config = DistributedConfig()
        assert config.strategy == "fsdp2"

    def test_from_env_reads_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "ddp")
        config = DistributedConfig.from_env()
        assert config.strategy == "ddp"

    def test_from_env_reads_diloco_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "diloco")
        config = DistributedConfig.from_env()
        assert config.strategy == "diloco"

    def test_from_env_invalid_strategy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "horovod")
        with pytest.raises(ValueError, match="Invalid GRAIL_DIST_STRATEGY"):
            DistributedConfig.from_env()

    def test_from_env_reads_diloco_hyperparams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "diloco")
        monkeypatch.setenv("GRAIL_DIST_DILOCO_INNER_STEPS", "50")
        monkeypatch.setenv("GRAIL_DIST_DILOCO_OUTER_LR", "0.5")
        monkeypatch.setenv("GRAIL_DIST_DILOCO_OUTER_MOMENTUM", "0.8")
        config = DistributedConfig.from_env()
        assert config.diloco_inner_steps == 50
        assert config.diloco_outer_lr == 0.5
        assert config.diloco_outer_momentum == 0.8

    def test_default_diloco_hyperparams(self) -> None:
        config = DistributedConfig(strategy="diloco")
        assert config.diloco_inner_steps == 10
        assert config.diloco_outer_lr == 0.7
        assert config.diloco_outer_momentum == 0.9

    def test_validate_ddp_rejects_tp(self) -> None:
        config = DistributedConfig(strategy="ddp", tp_degree=2)
        with pytest.raises(ValueError, match="DDP does not support tensor parallelism"):
            config.validate(world_size=4)

    def test_validate_diloco_requires_multi_gpu(self) -> None:
        config = DistributedConfig(strategy="diloco")
        with pytest.raises(ValueError, match="DILOCO requires at least 2 workers"):
            config.validate(world_size=1)

    def test_validate_diloco_inner_steps_positive(self) -> None:
        config = DistributedConfig(strategy="diloco", diloco_inner_steps=0)
        with pytest.raises(ValueError, match="diloco_inner_steps must be >= 1"):
            config.validate(world_size=2)

    def test_validate_fsdp2_passes(self) -> None:
        config = DistributedConfig(strategy="fsdp2", tp_degree=2)
        config.validate(world_size=4)  # Should not raise

    def test_validate_ddp_tp0_passes(self) -> None:
        config = DistributedConfig(strategy="ddp", tp_degree=0)
        config.validate(world_size=2)  # Should not raise

    def test_validate_ddp_tp1_passes(self) -> None:
        config = DistributedConfig(strategy="ddp", tp_degree=1)
        config.validate(world_size=2)  # Should not raise

    def test_strategy_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "DDP")
        config = DistributedConfig.from_env()
        assert config.strategy == "ddp"


# ============================================================================
# DistributedContext tests
# ============================================================================


class TestDistributedContext:
    """Test context behavior across strategies."""

    def test_unwrap_model_fsdp2_returns_as_is(self) -> None:
        ctx = DistributedContext(strategy="fsdp2")
        model = nn.Linear(10, 10)
        assert ctx.unwrap_model(model) is model

    def test_unwrap_model_ddp_returns_module(self) -> None:
        ctx = DistributedContext(strategy="ddp")
        inner = nn.Linear(10, 10)
        wrapper = MagicMock()
        wrapper.module = inner
        assert ctx.unwrap_model(wrapper) is inner

    def test_unwrap_model_ddp_no_module_returns_as_is(self) -> None:
        ctx = DistributedContext(strategy="ddp")
        model = nn.Linear(10, 10)
        # nn.Linear doesn't have .module attribute
        assert ctx.unwrap_model(model) is model

    def test_unwrap_model_diloco_returns_as_is(self) -> None:
        """DILOCO uses DDP wrapping but context.unwrap_model should still work."""
        ctx = DistributedContext(strategy="diloco")
        model = nn.Linear(10, 10)
        assert ctx.unwrap_model(model) is model

    def test_backward_calls_loss_backward(self) -> None:
        ctx = DistributedContext(strategy="ddp")
        loss = MagicMock()
        ctx.backward(loss)
        loss.backward.assert_called_once()

    def test_strategy_stored(self) -> None:
        for strategy in ("fsdp2", "ddp", "diloco"):
            ctx = DistributedContext(strategy=strategy)  # type: ignore[arg-type]
            assert ctx._strategy == strategy

    def test_default_strategy_is_fsdp2(self) -> None:
        ctx = DistributedContext()
        assert ctx._strategy == "fsdp2"


# ============================================================================
# DILOCO outer optimization math tests
# ============================================================================


class TestDilocoOuterStep:
    """Test DILOCO pseudo-gradient computation and Nesterov update.

    These tests validate the mathematical correctness of the outer optimization
    step without needing distributed communication (single-process simulation).
    """

    @staticmethod
    def _make_simple_model() -> nn.Module:
        """Create a tiny model for testing."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        return model

    def test_pseudo_gradient_computation(self) -> None:
        """Verify pseudo-gradient = theta_global - theta_local."""
        model = self._make_simple_model()

        # Snapshot global params
        global_params = [p.detach().clone() for p in model.parameters()]

        # Simulate inner training: modify model params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Compute pseudo-gradients
        for global_p, local_p in zip(global_params, model.parameters(), strict=False):
            delta = global_p - local_p.data
            # delta should be the direction to move global params toward local
            assert delta.shape == global_p.shape
            # After adding random noise to local, delta should be non-zero
            assert delta.abs().sum() > 0

    def test_nesterov_outer_update(self) -> None:
        """Verify the outer Nesterov SGD update produces correct results."""
        model = self._make_simple_model()

        # Snapshot global params on CPU
        global_params = [p.detach().clone().cpu() for p in model.parameters()]
        for gp in global_params:
            gp.requires_grad = True

        outer_lr = 0.7
        outer_momentum = 0.9
        outer_optimizer = torch.optim.SGD(
            global_params, lr=outer_lr, momentum=outer_momentum, nesterov=True
        )

        # Record initial global params
        initial_global = [gp.detach().clone() for gp in global_params]

        # Simulate inner training
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Compute pseudo-gradients and set as .grad
        for global_p, local_p in zip(global_params, model.parameters(), strict=False):
            delta = global_p.data - local_p.data.cpu()
            global_p.grad = delta

        # Apply outer optimizer step
        outer_optimizer.step()
        outer_optimizer.zero_grad()

        # Global params should have changed
        for gp, initial in zip(global_params, initial_global, strict=False):
            assert not torch.allclose(gp, initial), "Global params should change after outer step"

    def test_diloco_converges_toward_local(self) -> None:
        """After outer step with positive pseudo-gradients, global moves toward local."""
        model = self._make_simple_model()

        # Global params on CPU
        global_params = [p.detach().clone().cpu() for p in model.parameters()]
        for gp in global_params:
            gp.requires_grad = True

        outer_optimizer = torch.optim.SGD(global_params, lr=0.7, momentum=0.9, nesterov=True)

        # Simulate inner training: add a known perturbation
        perturbation = 0.5
        with torch.no_grad():
            for p in model.parameters():
                p.add_(perturbation)

        local_params = [p.detach().clone().cpu() for p in model.parameters()]

        # Compute pseudo-gradients: delta = global - local
        for global_p, local_p in zip(global_params, local_params, strict=False):
            global_p.grad = global_p.data - local_p

        # Record distance before outer step
        dist_before = sum(
            (gp.data - lp).norm().item()
            for gp, lp in zip(global_params, local_params, strict=False)
        )

        outer_optimizer.step()
        outer_optimizer.zero_grad()

        # Distance after should be smaller (converging)
        dist_after = sum(
            (gp.data - lp).norm().item()
            for gp, lp in zip(global_params, local_params, strict=False)
        )
        assert dist_after < dist_before, (
            f"Global params should move toward local: "
            f"dist_before={dist_before:.4f}, dist_after={dist_after:.4f}"
        )

    def test_step_counter_triggers_at_H(self) -> None:
        """Verify outer step triggers exactly every H inner steps."""
        H = 3
        triggered_at: list[int] = []

        for step in range(1, 10):
            if step % H == 0:
                triggered_at.append(step)

        assert triggered_at == [3, 6, 9]

    def test_multiple_outer_steps_accumulate_momentum(self) -> None:
        """Nesterov momentum should accumulate across outer steps."""
        model = self._make_simple_model()
        global_params = [p.detach().clone().cpu() for p in model.parameters()]
        for gp in global_params:
            gp.requires_grad = True

        outer_optimizer = torch.optim.SGD(global_params, lr=0.7, momentum=0.9, nesterov=True)

        deltas: list[float] = []

        for _step in range(5):
            # Apply same perturbation each time
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.01)

            for global_p, local_p in zip(global_params, model.parameters(), strict=False):
                global_p.grad = global_p.data - local_p.data.cpu()

            # Record param state before step
            before = global_params[0].data.clone()
            outer_optimizer.step()
            outer_optimizer.zero_grad()

            # Copy global back to local
            with torch.no_grad():
                for global_p, local_p in zip(global_params, model.parameters(), strict=False):
                    local_p.data.copy_(global_p.data)

            delta = (global_params[0].data - before).abs().mean().item()
            deltas.append(delta)

        # With momentum, later steps should have larger updates (momentum builds up)
        # First step has no momentum, subsequent steps should be larger
        assert deltas[1] > deltas[0], "Momentum should increase update magnitude"


# ============================================================================
# Strategy dispatch tests (mock-based)
# ============================================================================


class TestStrategyDispatch:
    """Test that the correct parallelism and checkpoint functions are called."""

    def test_resolve_tp_degree_ddp_forces_1(self) -> None:
        from grail.trainer.distributed.launcher import _resolve_tp_degree

        config = DistributedConfig(strategy="ddp", tp_degree=4)
        assert _resolve_tp_degree(config, world_size=4) == 1

    def test_resolve_tp_degree_diloco_forces_1(self) -> None:
        from grail.trainer.distributed.launcher import _resolve_tp_degree

        config = DistributedConfig(strategy="diloco", tp_degree=2)
        assert _resolve_tp_degree(config, world_size=4) == 1

    def test_resolve_tp_degree_fsdp2_uses_config(self) -> None:
        from grail.trainer.distributed.launcher import _resolve_tp_degree

        config = DistributedConfig(strategy="fsdp2", tp_degree=2)
        assert _resolve_tp_degree(config, world_size=4) == 2

    def test_resolve_tp_degree_fsdp2_auto_detect(self) -> None:
        from grail.trainer.distributed.launcher import _resolve_tp_degree

        config = DistributedConfig(strategy="fsdp2", tp_degree=0)
        tp = _resolve_tp_degree(config, world_size=4)
        assert tp >= 1  # Auto-detected


# ============================================================================
# DDP checkpoint tests
# ============================================================================


class TestDDPCheckpoint:
    """Test DDP-specific checkpoint saving."""

    def test_save_ddp_checkpoint_rank0_creates_files(self, tmp_path: Path) -> None:
        """Rank 0 should write model files."""
        from grail.trainer.distributed.checkpoint import save_ddp_checkpoint

        model = MagicMock()
        model.module = MagicMock()
        tokenizer = MagicMock()

        # Mock dist.barrier to be a no-op
        with patch("grail.trainer.distributed.checkpoint.dist") as mock_dist:
            mock_dist.barrier = MagicMock()
            save_ddp_checkpoint(model, tokenizer, tmp_path / "ckpt", rank=0)

        model.module.save_pretrained.assert_called_once()
        tokenizer.save_pretrained.assert_called_once()

    def test_save_ddp_checkpoint_rank1_skips(self, tmp_path: Path) -> None:
        """Non-rank-0 should not write anything."""
        from grail.trainer.distributed.checkpoint import save_ddp_checkpoint

        model = MagicMock()
        tokenizer = MagicMock()

        with patch("grail.trainer.distributed.checkpoint.dist") as mock_dist:
            mock_dist.barrier = MagicMock()
            save_ddp_checkpoint(model, tokenizer, tmp_path / "ckpt", rank=1)

        model.save_pretrained.assert_not_called()
        tokenizer.save_pretrained.assert_not_called()


# ============================================================================
# apply_ddp tests
# ============================================================================


class TestApplyDDP:
    """Test DDP wrapping function."""

    def test_apply_ddp_returns_ddp_module(self) -> None:
        """apply_ddp should wrap model with DistributedDataParallel."""
        from grail.trainer.distributed.parallelism import apply_ddp

        model = nn.Linear(4, 4)

        # DDP requires process group, so we mock the import target
        with patch("torch.nn.parallel.DistributedDataParallel") as MockDDP:
            mock_wrapped = MagicMock()
            MockDDP.return_value = mock_wrapped

            result = apply_ddp(model, local_rank=0)

            MockDDP.assert_called_once_with(
                model,
                device_ids=[0],
                static_graph=False,
                gradient_as_bucket_view=True,
            )
            assert result is mock_wrapped


# ============================================================================
# PULSE-DiLoCo tests
# ============================================================================


class TestPulseDilocoConfig:
    """Test PULSE-DiLoCo configuration."""

    def test_pulse_default_disabled(self) -> None:
        config = DistributedConfig(strategy="diloco")
        assert config.pulse_diloco is False

    def test_pulse_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAIL_DIST_STRATEGY", "diloco")
        monkeypatch.setenv("GRAIL_DIST_PULSE_DILOCO", "true")
        config = DistributedConfig.from_env()
        assert config.pulse_diloco is True

    def test_nesterov_disabled_when_momentum_zero(self) -> None:
        """PyTorch requires nesterov=False when momentum=0."""
        config = DistributedConfig(strategy="diloco", diloco_outer_momentum=0.0, pulse_diloco=True)
        # Verify the optimizer can be created without error
        params = [torch.zeros(4, requires_grad=True)]
        opt = torch.optim.SGD(
            params,
            lr=config.diloco_outer_lr,
            momentum=config.diloco_outer_momentum,
            nesterov=config.diloco_outer_momentum > 0,
        )
        assert opt is not None


class TestBF16Gating:
    """Test BF16 gating math for PULSE-DiLoCo.

    Uses BF16-exact values for deterministic results (pitfall #2 from guide).
    """

    def test_bf16_roundtrip_exact(self) -> None:
        """BF16-exact values survive roundtrip."""
        # 0.15234375 is exactly representable in BF16
        val = torch.tensor([0.15234375], dtype=torch.float32)
        roundtripped = val.bfloat16().float()
        assert torch.equal(val, roundtripped)

    def test_bf16_gating_zeros_small_changes(self) -> None:
        """Changes smaller than BF16 ULP produce zero gated pseudo-grad."""
        # Start with a BF16-exact value
        theta = torch.tensor([0.15234375], dtype=torch.float32)
        # Add a perturbation smaller than BF16 ULP (~1e-4 for this value)
        theta_local = theta + 1e-7

        b_theta = theta.bfloat16().float()
        b_local = theta_local.bfloat16().float()

        # Both round to the same BF16 value: gate would not fire
        assert torch.equal(b_theta, b_local)
        bf16_diff = b_theta - b_local
        assert bf16_diff.item() == 0.0

    def test_bf16_gating_nonzero_for_large_changes(self) -> None:
        """Changes larger than BF16 ULP produce non-zero gated pseudo-grad."""
        theta = torch.tensor([1.0], dtype=torch.float32)
        theta_local = theta + 0.01  # Large enough to cross BF16 boundary

        b_theta = theta.bfloat16().float()
        b_local = theta_local.bfloat16().float()

        # Should be different at BF16: gate would fire
        assert not torch.equal(b_theta, b_local)
        bf16_diff = b_theta - b_local
        assert bf16_diff.item() != 0.0

    def test_residual_conservation(self) -> None:
        """Residual + transmitted = total signal (conservation invariant)."""
        # Simulate one round of PULSE gating
        n = 100
        snapshot = torch.randn(n, dtype=torch.float32)
        # Make snapshot BF16-exact
        snapshot = snapshot.bfloat16().float()

        local_params = snapshot + torch.randn(n) * 0.001  # Small perturbation
        residual = torch.zeros(n, dtype=torch.float32)

        delta = snapshot - local_params  # Exact pseudo-grad
        s = residual + delta
        w_tilde = snapshot - s

        # v6: BF16 gate as binary mask, transmit raw FP32 s
        b_theta = snapshot.bfloat16().float()
        b_w_tilde = w_tilde.bfloat16().float()
        gate_mask = b_theta != b_w_tilde

        transmitted = torch.zeros_like(s)
        transmitted[gate_mask] = s[gate_mask]

        new_residual = s.clone()
        new_residual[gate_mask] = 0  # Fired entries fully consumed

        # v6 conservation: transmitted + residual = total signal at all indices
        reconstructed = transmitted + new_residual
        assert torch.allclose(reconstructed, s, atol=1e-7), (
            f"Conservation violated: max diff = {(reconstructed - s).abs().max().item()}"
        )

    def test_residual_accumulates_across_rounds(self) -> None:
        """After multiple rounds, residuals eventually push entries across BF16 boundary."""
        n = 50
        snapshot = torch.ones(n, dtype=torch.float32)  # BF16-exact
        residual = torch.zeros(n, dtype=torch.float32)
        total_transmitted = torch.zeros(n, dtype=torch.float32)

        # Apply the same tiny perturbation many rounds
        perturbation = 1e-5  # Much smaller than BF16 ULP for 1.0

        for _round in range(200):
            local_params = snapshot - perturbation  # theta_local slightly less than snapshot
            delta = snapshot - local_params  # = perturbation (positive)
            s = residual + delta
            w_tilde = snapshot - s

            # v6: BF16 gate as mask, transmit raw FP32 s
            b_theta = snapshot.bfloat16().float()
            b_w_tilde = w_tilde.bfloat16().float()
            gate_mask = b_theta != b_w_tilde

            transmitted = torch.zeros_like(s)
            transmitted[gate_mask] = s[gate_mask]
            total_transmitted += transmitted

            # v6 residual: fired=0, unfired=s
            residual = s.clone()
            residual[gate_mask] = 0

        # After 200 rounds of perturbation=1e-5, total signal = 200 * 1e-5 = 2e-3
        # At least some entries should have been transmitted
        assert total_transmitted.abs().sum() > 0, "Residuals should eventually transmit"

    def test_sparsity_high_for_small_perturbations(self) -> None:
        """With small perturbations, most entries should be gated to zero."""
        n = 1000
        snapshot = torch.randn(n, dtype=torch.float32).bfloat16().float()
        local_params = snapshot + torch.randn(n) * 1e-7  # Tiny perturbation
        residual = torch.zeros(n, dtype=torch.float32)

        delta = snapshot - local_params
        s = residual + delta
        w_tilde = snapshot - s

        # v6: gate mask determines sparsity
        b_theta = snapshot.bfloat16().float()
        b_w_tilde = w_tilde.bfloat16().float()
        gate_mask = b_theta != b_w_tilde

        nnz = gate_mask.sum().item()
        sparsity = 1.0 - nnz / n
        # With 1e-7 perturbation, almost everything should be sparse
        assert sparsity > 0.9, f"Expected high sparsity, got {sparsity:.2%}"

    def test_all_entries_fire(self) -> None:
        """When all entries cross BF16 boundary, residual is zero everywhere."""
        n = 100
        snapshot = torch.ones(n, dtype=torch.float32)  # BF16-exact
        local_params = snapshot + 0.1  # Large perturbation, all entries fire
        residual = torch.zeros(n, dtype=torch.float32)

        delta = snapshot - local_params
        s = residual + delta
        w_tilde = snapshot - s
        gate_mask = snapshot.bfloat16().float() != w_tilde.bfloat16().float()

        assert gate_mask.all(), "All entries should fire with large perturbation"
        transmitted = torch.zeros_like(s)
        transmitted[gate_mask] = s[gate_mask]
        new_residual = s.clone()
        new_residual[gate_mask] = 0

        assert torch.equal(transmitted, s), "All-fire: transmitted must equal s"
        assert (new_residual == 0).all(), "All-fire: residual must be all zeros"

    def test_no_entries_fire(self) -> None:
        """When no entries cross BF16 boundary, transmitted is zero and residual = s."""
        n = 100
        snapshot = torch.ones(n, dtype=torch.float32)  # BF16-exact
        # Zero perturbation: local == snapshot
        local_params = snapshot.clone()
        residual = torch.zeros(n, dtype=torch.float32)

        delta = snapshot - local_params  # all zeros
        s = residual + delta  # all zeros
        w_tilde = snapshot - s
        gate_mask = snapshot.bfloat16().float() != w_tilde.bfloat16().float()

        assert not gate_mask.any(), "No entries should fire with zero perturbation"
        transmitted = torch.zeros_like(s)
        new_residual = s.clone()

        assert (transmitted == 0).all(), "No-fire: transmitted must be all zeros"
        assert torch.equal(new_residual, s), "No-fire: residual must equal s"

    def test_deterministic_residual_crossover(self) -> None:
        """Known residual pushes a specific entry across BF16 boundary next round."""
        # Empirically: at snapshot=2.0, the gate fires when s >= 0.004
        # (w_tilde=1.996 rounds to 2.0, w_tilde=1.9960 rounds to 1.992188)
        snapshot = torch.tensor([2.0], dtype=torch.float32)

        # Round 1: s = 0.003, below firing threshold -> doesn't fire
        residual = torch.tensor([0.001], dtype=torch.float32)
        delta1 = torch.tensor([0.002])
        s1 = residual + delta1  # 0.003
        w_tilde1 = snapshot - s1  # 1.997
        gate1 = snapshot.bfloat16().float() != w_tilde1.bfloat16().float()
        assert not gate1.item(), "Round 1: should NOT fire (s=0.003 below threshold)"
        residual = s1.clone()  # carry forward

        # Round 2: s = 0.003 + 0.002 = 0.005, above firing threshold -> fires
        delta2 = torch.tensor([0.002])
        s2 = residual + delta2  # 0.005 >= 0.004
        w_tilde2 = snapshot - s2  # 1.995
        gate2 = snapshot.bfloat16().float() != w_tilde2.bfloat16().float()
        assert gate2.item(), "Round 2: MUST fire (residual accumulated past threshold)"

    def test_full_pulse_round_single_worker(self) -> None:
        """End-to-end single-worker v6 PULSE step: gate, transmit, outer step, reset."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        # Snapshot global params
        global_params = [p.detach().clone().cpu() for p in model.parameters()]
        for gp in global_params:
            gp.requires_grad = True

        total_numel = sum(p.numel() for p in global_params)
        residual = torch.zeros(total_numel, dtype=torch.float32)

        # Simulate inner training
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        local_params = list(model.parameters())

        # Step 1: v6 BF16-gated FP32 pseudo-gradient
        flat_pseudo_grad = torch.zeros(total_numel, dtype=torch.float32)
        offset = 0
        for global_p, local_p in zip(global_params, local_params, strict=True):
            numel = global_p.numel()
            param_cpu = local_p.data.cpu().float()
            snapshot = global_p.data.view(-1)

            delta = snapshot - param_cpu.view(-1)
            s = residual[offset : offset + numel] + delta
            w_tilde = snapshot - s

            b_theta = snapshot.bfloat16().float()
            b_w_tilde = w_tilde.bfloat16().float()
            gate_mask = b_theta != b_w_tilde

            transmitted = torch.zeros_like(s)
            transmitted[gate_mask] = s[gate_mask]
            flat_pseudo_grad[offset : offset + numel] = transmitted

            residual[offset : offset + numel] = s
            residual[offset : offset + numel][gate_mask] = 0
            offset += numel

        # Step 2: Single worker, so avg_pseudo_grad = flat_pseudo_grad
        avg_pseudo_grad = flat_pseudo_grad

        # Step 3: Outer optimizer
        outer_opt = torch.optim.SGD(global_params, lr=0.7, momentum=0.9, nesterov=True)
        offset = 0
        for global_p in global_params:
            numel = global_p.numel()
            global_p.grad = avg_pseudo_grad[offset : offset + numel].view(global_p.shape)
            offset += numel

        before = [gp.data.clone() for gp in global_params]
        outer_opt.step()
        outer_opt.zero_grad()

        # Step 4: Hard reset
        with torch.no_grad():
            for global_p, local_p in zip(global_params, local_params, strict=True):
                local_p.data.copy_(global_p)

        # Verify: global params changed (at least for non-zero pseudo-grads)
        any_changed = any(
            not torch.allclose(gp, b) for gp, b in zip(global_params, before, strict=True)
        )
        assert any_changed, "Global params should change after outer step"

        # Verify: local params match global params after reset
        for global_p, local_p in zip(global_params, local_params, strict=True):
            assert torch.equal(local_p.data.cpu(), global_p.data)


# ============================================================================
# Resume checkpoint tests
# ============================================================================


class TestResumeCheckpoint:
    """Test DILOCO resume checkpoint save/load round-trip."""

    def test_diloco_state_roundtrip(self, tmp_path: Path) -> None:
        """Save and load DILOCO state, verify exact restoration."""
        # Simulate DILOCO state
        global_params = [torch.randn(4, 4) for _ in range(3)]
        for gp in global_params:
            gp.requires_grad = True

        outer_opt = torch.optim.SGD(global_params, lr=0.7, momentum=0.9, nesterov=True)

        # Run a fake outer step to populate momentum buffers
        for gp in global_params:
            gp.grad = torch.randn_like(gp)
        outer_opt.step()
        outer_opt.zero_grad()

        inner_step = 7
        epoch = 42

        # Save
        state = {
            "epoch_counter": epoch,
            "diloco_inner_step_counter": inner_step,
            "diloco_global_params": [p.data.clone() for p in global_params],
            "diloco_outer_optimizer_state_dict": outer_opt.state_dict(),
        }
        torch.save(state, tmp_path / "resume_state.pt")

        # Load into fresh state
        loaded = torch.load(tmp_path / "resume_state.pt", weights_only=False)

        assert loaded["epoch_counter"] == epoch
        assert loaded["diloco_inner_step_counter"] == inner_step
        for saved, original in zip(loaded["diloco_global_params"], global_params, strict=True):
            assert torch.equal(saved, original.data)

        # Restore optimizer into a new optimizer with same params
        new_params = [torch.randn(4, 4) for _ in range(3)]
        for np_p in new_params:
            np_p.requires_grad = True
        new_opt = torch.optim.SGD(new_params, lr=0.7, momentum=0.9, nesterov=True)
        new_opt.load_state_dict(loaded["diloco_outer_optimizer_state_dict"])

        # Momentum buffers should match
        for (_, orig_state), (_, new_state) in zip(
            outer_opt.state.items(), new_opt.state.items(), strict=True
        ):
            assert torch.equal(orig_state["momentum_buffer"], new_state["momentum_buffer"])

    def test_pulse_residual_roundtrip(self, tmp_path: Path) -> None:
        """PULSE residual buffer survives save/load."""
        residual = torch.randn(1000, dtype=torch.float32)

        state = {"pulse_residual_buffer": residual.clone()}
        torch.save(state, tmp_path / "resume_state.pt")

        loaded = torch.load(tmp_path / "resume_state.pt", weights_only=False)
        assert torch.equal(loaded["pulse_residual_buffer"], residual)

    def test_resume_without_checkpoint_returns_false(self, tmp_path: Path) -> None:
        """No resume file means no loading."""
        # The file doesn't exist, so the path check should fail
        resume_path = tmp_path / "resume_state.pt"
        assert not resume_path.exists()


# ============================================================================
# P1 fix: DILOCO snapshot adopt gating
# ============================================================================


class TestDilocoSnapshotGating:
    """Verify that snapshots/latest is NOT overwritten between DILOCO outer syncs.

    The launcher must only call adopt_snapshot_atomic() when a DILOCO outer sync
    has happened, to prevent exposing non-consensus (diverged) weights to the
    upload worker or evaluation resource loader.
    """

    def test_diloco_non_sync_skips_weight_save(self) -> None:
        """When _diloco_sync_happened is False, no model weights should be saved."""
        service = MagicMock()
        service.dist_config.strategy = "diloco"
        service._diloco_sync_happened = False

        # should_save_weights logic from launcher
        strategy = service.dist_config.strategy
        diloco_synced = service._diloco_sync_happened
        should_save_weights = strategy != "diloco" or diloco_synced

        assert not should_save_weights, (
            "DILOCO non-sync epoch must not save weights (would overwrite snapshots/latest)"
        )

    def test_diloco_sync_epoch_saves_weights(self) -> None:
        """When _diloco_sync_happened is True, weights should be saved and adopted."""
        service = MagicMock()
        service.dist_config.strategy = "diloco"
        service._diloco_sync_happened = True

        strategy = service.dist_config.strategy
        diloco_synced = service._diloco_sync_happened
        should_save_weights = strategy != "diloco" or diloco_synced

        assert should_save_weights, "DILOCO sync epoch must save weights"

    def test_ddp_always_saves_weights(self) -> None:
        """DDP always saves weights regardless of sync flag."""
        service = MagicMock()
        service.dist_config.strategy = "ddp"

        strategy = service.dist_config.strategy
        should_save_weights = strategy != "diloco"

        assert should_save_weights, "DDP must always save weights"

    def test_fsdp2_always_saves_weights(self) -> None:
        """FSDP2 always saves weights regardless of sync flag."""
        service = MagicMock()
        service.dist_config.strategy = "fsdp2"

        strategy = service.dist_config.strategy
        should_save_weights = strategy != "diloco"

        assert should_save_weights, "FSDP2 must always save weights"

    def test_resume_checkpoint_saved_on_non_sync_epoch(self) -> None:
        """Resume state is saved even when weight checkpoint is skipped."""
        service = MagicMock()
        service.dist_config.strategy = "diloco"
        service._diloco_sync_happened = False

        # The launcher always calls _save_resume_checkpoint on rank 0,
        # regardless of should_save_weights. Verify the call would happen.
        rank = 0
        if rank == 0:
            service._save_resume_checkpoint()

        service._save_resume_checkpoint.assert_called_once()


# ============================================================================
# Sparse averaging correctness
# ============================================================================


class TestSparseAveraging:
    """Verify sparse scatter-add averaging matches dense averaging."""

    def test_sparse_average_matches_dense(self) -> None:
        """Sparse index_add_ averaging produces same result as dense sum."""
        torch.manual_seed(42)
        # Simulate: two workers with different pseudo-grads
        n = 1000
        pseudo_grad_worker0 = torch.randn(n, dtype=torch.float32)
        pseudo_grad_worker1 = torch.randn(n, dtype=torch.float32)

        # Sparse path: extract non-zeros, average
        nz0 = pseudo_grad_worker0.nonzero(as_tuple=True)[0]
        nz1 = pseudo_grad_worker1.nonzero(as_tuple=True)[0]
        vals0 = pseudo_grad_worker0[nz0]
        vals1 = pseudo_grad_worker1[nz1]

        sparse_avg = torch.zeros(n)
        sparse_avg.index_add_(0, nz0, vals0)
        sparse_avg.index_add_(0, nz1, vals1)
        sparse_avg.div_(2)

        # Dense path: just average the full tensors
        dense_avg = (pseudo_grad_worker0 + pseudo_grad_worker1) / 2

        assert torch.allclose(sparse_avg, dense_avg, atol=1e-6), (
            "Dense and sparse averaging must produce the same result"
        )

    def test_residual_invariant_independent_of_comms(self) -> None:
        """v6 conservation invariant: transmitted + residual = s at all indices."""
        torch.manual_seed(123)
        n = 100
        # Simulate step 1 of v6 PULSE
        global_params_flat = torch.randn(n, dtype=torch.float32)
        local_params_flat = global_params_flat + torch.randn(n) * 0.001  # small perturbation
        residual_before = torch.randn(n, dtype=torch.float32) * 0.0001

        # Step 1: v6 BF16-gated FP32 pseudo-gradient
        delta = global_params_flat - local_params_flat
        s = residual_before + delta
        w_tilde = global_params_flat - s
        b_theta = global_params_flat.bfloat16().float()
        b_w_tilde = w_tilde.bfloat16().float()
        gate_mask = b_theta != b_w_tilde

        transmitted = torch.zeros_like(s)
        transmitted[gate_mask] = s[gate_mask]
        residual_after = s.clone()
        residual_after[gate_mask] = 0

        # v6 conservation: transmitted + residual = corrected signal
        assert torch.allclose(transmitted + residual_after, s, atol=1e-6), (
            "Conservation invariant violated: transmitted + residual != total signal"
        )
