"""Integration tests for Triton kernel GPU evaluation.

Requires GPU. Run on basilica with:
    KERNEL_EVAL_GPU_IDS=0 pytest -m gpu_real_data tests/integration/environments/test_triton_kernel_gpu_integration.py -v
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from grail.shared.thinking import get_thinking_config

pytestmark = [pytest.mark.integration, pytest.mark.gpu_real_data]

_cfg = get_thinking_config()


# =============================================================================
# Helpers
# =============================================================================


def _get_subprocess_backend(gpu_id: int = 0, timeout: float = 60.0):
    """Create a SubprocessBackend for testing."""
    from grail.environments.gpu_kernel.eval_backends.subprocess_backend import (
        SubprocessBackend,
    )

    return SubprocessBackend(gpu_ids=[gpu_id], timeout=timeout)


# Simple correct Triton kernel for ReLU
_CORRECT_TRITON_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    if idx < n:
        x = tl.load(x_ptr + idx)
        tl.store(out_ptr + idx, tl.maximum(x, 0.0))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        relu_kernel[(n,)](x, out, n)
        return out
'''

# Incorrect kernel (adds 100 instead of relu)
_INCORRECT_TRITON_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def wrong_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    if idx < n:
        x = tl.load(x_ptr + idx)
        tl.store(out_ptr + idx, x + 100.0)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        wrong_kernel[(n,)](x, out, n)
        return out
'''

# Test code with check_correctness
_TEST_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(64)]

def get_init_inputs():
    return []

def check_correctness(model_new_cls):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_model = Model().to(device).eval()
    new_model = model_new_cls().to(device).eval()

    max_diff = 0.0
    for trial in range(3):
        torch.manual_seed(42 + trial)
        inputs = [torch.randn(64).to(device)]
        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)
        diff = torch.max(torch.abs(ref_out.float() - new_out.float())).item()
        max_diff = max(max_diff, diff)

    correct = max_diff <= 1e-2
    return {
        "correct": correct,
        "compiled": True,
        "error": None if correct else f"max_diff={max_diff:.6f}",
        "max_diff": max_diff,
    }
'''

_SYNTAX_ERROR_CODE = "def broken(:\n    return"


# =============================================================================
# TestSubprocessBackend
# =============================================================================


class TestSubprocessBackend:
    """Real GPU execution tests."""

    def test_correct_kernel_passes(self) -> None:
        """Known-correct Triton kernel -> EvalResult(correct=True)."""
        backend = _get_subprocess_backend()
        result = backend.evaluate(_TEST_CODE, _CORRECT_TRITON_CODE)
        assert result.compiled is True
        assert result.correct is True

    def test_incorrect_kernel_fails(self) -> None:
        """Wrong output -> EvalResult(correct=False)."""
        backend = _get_subprocess_backend()
        result = backend.evaluate(_TEST_CODE, _INCORRECT_TRITON_CODE)
        assert result.compiled is True
        assert result.correct is False

    def test_compile_error_detected(self) -> None:
        """Syntax error in Triton -> compiled=False."""
        backend = _get_subprocess_backend()
        result = backend.evaluate(_TEST_CODE, _SYNTAX_ERROR_CODE)
        assert result.compiled is False
        assert result.correct is False

    def test_timeout_enforcement(self) -> None:
        """Infinite loop -> timeout error within timeout period."""
        infinite_code = '''
import torch
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        while True:
            pass
        return x
'''
        backend = _get_subprocess_backend(timeout=5.0)
        start = time.monotonic()
        result = backend.evaluate(_TEST_CODE, infinite_code)
        elapsed = time.monotonic() - start

        assert result.correct is False
        assert elapsed < 15.0  # Should timeout within ~5s + overhead
        assert result.error is not None

    def test_batch_evaluation(self) -> None:
        """evaluate_batch() processes multiple kernels."""
        backend = _get_subprocess_backend()
        backend.start()
        try:
            results = backend.evaluate_batch([
                (_TEST_CODE, _CORRECT_TRITON_CODE),
                (_TEST_CODE, _INCORRECT_TRITON_CODE),
            ])
            assert len(results) == 2
            assert results[0].correct is True
            assert results[1].correct is False
        finally:
            backend.shutdown()


# =============================================================================
# TestWarmup
# =============================================================================


class TestWarmup:
    """JIT warmup effectiveness."""

    def test_warmup_completes(self) -> None:
        """warmup(sample_codes) doesn't crash."""
        backend = _get_subprocess_backend()
        backend.warmup([_TEST_CODE])

    def test_warmup_idempotent(self) -> None:
        """Calling warmup twice is safe."""
        backend = _get_subprocess_backend()
        backend.warmup([_TEST_CODE])
        backend.warmup([_TEST_CODE])


# =============================================================================
# TestEndToEndEnv
# =============================================================================


class TestEndToEndEnv:
    """Full environment flow with real GPU."""

    def _make_env(self, gpu_eval: bool = True) -> Any:
        from grail.environments.gpu_kernel.env import TritonKernelEnv
        from grail.environments.providers import TaskSpec

        class _GpuTestSource:
            def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
                return TaskSpec(
                    id="gpu_test_0",
                    payload={
                        "pytorch_code": _TEST_CODE.split("def check_correctness")[0],
                        "test_code": _TEST_CODE,
                        "problem_name": "relu_test",
                    },
                    metadata={"split": "test", "index": 0},
                )

        backend = _get_subprocess_backend()
        return TritonKernelEnv(
            task_source=_GpuTestSource(),
            gpu_eval=gpu_eval,
            eval_backend=backend,
        )

    def test_reset_step_reward_flow(self) -> None:
        """reset -> step -> reward with real backend."""
        from grail.environments.core import ChatMessage

        env = self._make_env(gpu_eval=True)
        obs = env.reset(seed=42)
        assert len(obs.messages) == 1

        completion = (
            f"{_cfg.thinking_open}\nOptimizing ReLU.\n{_cfg.thinking_close}\n"
            f"{_cfg.solution_open}\n{_CORRECT_TRITON_CODE}\n{_cfg.solution_close}"
        )
        _, reward, terminated, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )
        assert terminated is True
        assert reward > 0.5
        assert info["success"] is True

    def test_correct_kernel_high_reward(self) -> None:
        """Known-correct code -> reward near 1.0."""
        from grail.environments.core import ChatMessage

        env = self._make_env(gpu_eval=True)
        env.reset(seed=42)

        completion = (
            f"{_cfg.thinking_open}\nOptimizing ReLU.\n{_cfg.thinking_close}\n"
            f"{_cfg.solution_open}\n{_CORRECT_TRITON_CODE}\n{_cfg.solution_close}"
        )
        _, reward, _, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )
        assert reward == pytest.approx(1.0)
        assert info["exec_result"]["correct"] is True

    def test_incorrect_kernel_partial_reward(self) -> None:
        """Structural but wrong -> reward ~0.35 (structure only, no correctness)."""
        from grail.environments.core import ChatMessage

        env = self._make_env(gpu_eval=True)
        env.reset(seed=42)

        completion = (
            f"{_cfg.thinking_open}\nOptimizing ReLU.\n{_cfg.thinking_close}\n"
            f"{_cfg.solution_open}\n{_INCORRECT_TRITON_CODE}\n{_cfg.solution_close}"
        )
        _, reward, _, _, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )
        # Should get: compilation(0.05) + structure(0.10) + gpu_comp(0.15) + format(0.10) + thinking(0.10) = 0.50
        # But NOT correctness (0.50)
        assert 0.3 < reward < 0.7
        assert info["exec_result"]["correct"] is False


# =============================================================================
# TestMinerValidatorAgreement
# =============================================================================


class TestMinerValidatorAgreement:
    """Critical: reward determinism between miner and validator paths."""

    def test_same_seed_same_reward(self) -> None:
        """Miner and validator paths produce identical reward for same seed+completion."""
        from grail.environments.core import ChatMessage
        from grail.environments.gpu_kernel.env import TritonKernelEnv
        from grail.environments.providers import TaskSpec

        class _DeterministicSource:
            def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
                return TaskSpec(
                    id="det_test_0",
                    payload={
                        "pytorch_code": _TEST_CODE.split("def check_correctness")[0],
                        "test_code": _TEST_CODE,
                        "problem_name": "relu_test",
                    },
                    metadata={"split": "test", "index": 0},
                )

        backend = _get_subprocess_backend()
        completion = (
            f"{_cfg.thinking_open}\nOptimizing.\n{_cfg.thinking_close}\n"
            f"{_cfg.solution_open}\n{_CORRECT_TRITON_CODE}\n{_cfg.solution_close}"
        )

        rewards = []
        for _ in range(2):
            env = TritonKernelEnv(
                task_source=_DeterministicSource(),
                gpu_eval=True,
                eval_backend=backend,
            )
            env.reset(seed=42)
            _, reward, _, _, _ = env.step(
                ChatMessage(role="assistant", content=completion)
            )
            rewards.append(reward)

        assert rewards[0] == pytest.approx(rewards[1])
