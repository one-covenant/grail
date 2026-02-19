"""Unit tests for eval backend protocol, factory, and global management.

No GPU required.
"""

from __future__ import annotations

import pytest

from grail.environments.gpu_kernel.eval_backends import (
    EvalResult,
    create_backend,
    get_global_backend,
    set_global_backend,
)
from tests.fixtures.fakes import FakeEvalBackend


# =============================================================================
# TestEvalResult
# =============================================================================


class TestEvalResult:
    """EvalResult dataclass."""

    def test_default_values(self) -> None:
        r = EvalResult(correct=False, compiled=False)
        assert r.correct is False
        assert r.compiled is False
        assert r.error is None
        assert r.max_diff is None

    def test_successful_result(self) -> None:
        r = EvalResult(correct=True, compiled=True, max_diff=0.001)
        assert r.correct is True
        assert r.compiled is True
        assert r.max_diff == pytest.approx(0.001)

    def test_error_result(self) -> None:
        r = EvalResult(correct=False, compiled=False, error="timeout")
        assert r.error == "timeout"


# =============================================================================
# TestBackendFactory
# =============================================================================


class TestBackendFactory:
    """create_backend() factory."""

    def test_create_subprocess_backend(self) -> None:
        from grail.environments.gpu_kernel.eval_backends.subprocess_backend import (
            SubprocessBackend,
        )

        backend = create_backend("subprocess", gpu_ids=[])
        assert isinstance(backend, SubprocessBackend)

    def test_create_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown kernel eval backend"):
            create_backend("nonexistent")

    def test_default_backend_is_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from grail.environments.gpu_kernel.eval_backends.subprocess_backend import (
            SubprocessBackend,
        )

        monkeypatch.delenv("KERNEL_EVAL_BACKEND", raising=False)
        backend = create_backend(gpu_ids=[])
        assert isinstance(backend, SubprocessBackend)


# =============================================================================
# TestGlobalBackend
# =============================================================================


class TestGlobalBackend:
    """Module-level global backend."""

    def test_set_and_get_global_backend(self) -> None:
        import grail.environments.gpu_kernel.eval_backends as eb

        original = eb._global_backend
        try:
            backend = FakeEvalBackend()
            set_global_backend(backend)
            assert get_global_backend() is backend
        finally:
            eb._global_backend = original

    def test_get_global_returns_none_initially(self) -> None:
        import grail.environments.gpu_kernel.eval_backends as eb

        original = eb._global_backend
        try:
            eb._global_backend = None
            assert get_global_backend() is None
        finally:
            eb._global_backend = original

    def test_set_global_replaces_previous(self) -> None:
        import grail.environments.gpu_kernel.eval_backends as eb

        original = eb._global_backend
        try:
            b1 = FakeEvalBackend()
            b2 = FakeEvalBackend()
            set_global_backend(b1)
            set_global_backend(b2)
            assert get_global_backend() is b2
        finally:
            eb._global_backend = original


# =============================================================================
# TestFakeBackendProtocol
# =============================================================================


class TestFakeBackendProtocol:
    """FakeEvalBackend matches KernelEvalBackend protocol."""

    def test_evaluate_returns_eval_result(self) -> None:
        backend = FakeEvalBackend()
        result = backend.evaluate("test_code", "triton_code")
        assert isinstance(result, EvalResult)

    def test_evaluate_batch_delegates(self) -> None:
        backend = FakeEvalBackend()
        results = backend.evaluate_batch([("tc1", "tr1"), ("tc2", "tr2")])
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)

    def test_warmup_sets_flag(self) -> None:
        backend = FakeEvalBackend()
        assert backend.warmed_up is False
        backend.warmup(["code1"])
        assert backend.warmed_up is True

    def test_start_shutdown_lifecycle(self) -> None:
        backend = FakeEvalBackend()
        assert backend.started is False
        backend.start()
        assert backend.started is True
        backend.shutdown()
        assert backend.started is False

    def test_call_log_records_inputs(self) -> None:
        backend = FakeEvalBackend()
        backend.evaluate("test1", "triton1")
        backend.evaluate("test2", "triton2")
        assert len(backend.call_log) == 2
        assert backend.call_log[0] == ("test1", "triton1")
        assert backend.call_log[1] == ("test2", "triton2")

    def test_results_by_code_routing(self) -> None:
        """Different results for different code inputs."""
        correct = EvalResult(correct=True, compiled=True)
        backend = FakeEvalBackend(results_by_code={"good_code": correct})

        r1 = backend.evaluate("test", "good_code")
        r2 = backend.evaluate("test", "bad_code")

        assert r1.correct is True
        assert r2.correct is False  # default result
