"""GPU integration tests for persistent worker pool backend.

Requires GPU. Run on basilica with:
    KERNEL_EVAL_GPU_IDS=7 pytest -m gpu_real_data tests/integration/environments/test_persistent_backend_gpu_integration.py -v -s

Correctness tests verify that PersistentWorkerPool produces identical results
to SubprocessBackend across a suite of premade kernels.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu_real_data]


# =============================================================================
# Premade kernel fixtures
# =============================================================================

RELU_TEST_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024)]

def get_init_inputs():
    return []
"""

RELU_TRITON_CODE = """
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
"""

ADD_TEST_CODE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1.0

def get_inputs():
    return [torch.randn(2048)]

def get_init_inputs():
    return []
"""

ADD_TRITON_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_one_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    if idx < n:
        x = tl.load(x_ptr + idx)
        tl.store(out_ptr + idx, x + 1.0)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        n = x.numel()
        add_one_kernel[(n,)](x, out, n)
        return out
"""

WRONG_TRITON_CODE = """
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
"""

SYNTAX_ERROR_CODE = "def broken(:\n    return"

KERNEL_CASES = [
    pytest.param(RELU_TEST_CODE, RELU_TRITON_CODE, True, True, id="relu_correct"),
    pytest.param(ADD_TEST_CODE, ADD_TRITON_CODE, True, True, id="add_one_correct"),
    pytest.param(RELU_TEST_CODE, WRONG_TRITON_CODE, False, True, id="relu_wrong_output"),
    pytest.param(RELU_TEST_CODE, SYNTAX_ERROR_CODE, False, False, id="syntax_error"),
]


# =============================================================================
# Helpers
# =============================================================================


def _get_subprocess_backend(gpu_id: int = 0, timeout: float = 60.0):
    from grail.environments.gpu_kernel.eval_backends.subprocess_backend import SubprocessBackend

    return SubprocessBackend(gpu_ids=[gpu_id], timeout=timeout)


def _get_persistent_backend(gpu_id: int = 0, timeout: float = 60.0, max_evals: int = 100):
    from grail.environments.gpu_kernel.eval_backends.persistent_backend import PersistentWorkerPool

    return PersistentWorkerPool(
        gpu_ids=[gpu_id],
        timeout=timeout,
        max_evals_per_worker=max_evals,
    )


# =============================================================================
# TestPersistentCorrectness
# =============================================================================


class TestPersistentCorrectness:
    """PersistentWorkerPool produces correct results on real GPU."""

    @pytest.mark.parametrize("test_code,triton_code,expect_correct,expect_compiled", KERNEL_CASES)
    def test_single_eval(
        self,
        test_code: str,
        triton_code: str,
        expect_correct: bool,
        expect_compiled: bool,
    ) -> None:
        backend = _get_persistent_backend()
        backend.start()
        try:
            result = backend.evaluate(test_code, triton_code)
            assert result.correct is expect_correct
            assert result.compiled is expect_compiled
        finally:
            backend.shutdown()

    def test_batch_eval(self) -> None:
        backend = _get_persistent_backend()
        backend.start()
        try:
            items = [
                (RELU_TEST_CODE, RELU_TRITON_CODE),
                (RELU_TEST_CODE, WRONG_TRITON_CODE),
                (ADD_TEST_CODE, ADD_TRITON_CODE),
            ]
            results = backend.evaluate_batch(items)
            assert len(results) == 3
            assert results[0].correct is True
            assert results[1].correct is False
            assert results[2].correct is True
        finally:
            backend.shutdown()

    def test_warmup_completes(self) -> None:
        backend = _get_persistent_backend()
        backend.start()
        try:
            backend.warmup([RELU_TEST_CODE])
        finally:
            backend.shutdown()


# =============================================================================
# TestBackendParity
# =============================================================================


class TestBackendParity:
    """Critical: SubprocessBackend and PersistentWorkerPool produce identical results."""

    @pytest.mark.parametrize("test_code,triton_code,expect_correct,expect_compiled", KERNEL_CASES)
    def test_parity(
        self,
        test_code: str,
        triton_code: str,
        expect_correct: bool,
        expect_compiled: bool,
    ) -> None:
        sub = _get_subprocess_backend()
        pers = _get_persistent_backend()
        pers.start()
        try:
            sub_result = sub.evaluate(test_code, triton_code)
            pers_result = pers.evaluate(test_code, triton_code)

            assert sub_result.correct == pers_result.correct, (
                f"correct mismatch: subprocess={sub_result.correct} persistent={pers_result.correct}"
            )
            assert sub_result.compiled == pers_result.compiled, (
                f"compiled mismatch: subprocess={sub_result.compiled} persistent={pers_result.compiled}"
            )

            if sub_result.max_diff is not None and pers_result.max_diff is not None:
                assert sub_result.max_diff == pytest.approx(pers_result.max_diff, abs=1e-6)
        finally:
            pers.shutdown()


# =============================================================================
# TestWorkerRecycling
# =============================================================================


class TestWorkerRecycling:
    """Worker recycling works correctly on real GPU."""

    def test_recycle_after_3_evals(self) -> None:
        """Set max_evals=3, run 5 evals, verify all succeed."""
        backend = _get_persistent_backend(max_evals=3)
        backend.start()
        try:
            for i in range(5):
                result = backend.evaluate(RELU_TEST_CODE, RELU_TRITON_CODE)
                assert result.correct is True, f"eval {i} failed: {result.error}"
        finally:
            backend.shutdown()
