"""Unit tests for the PersistentWorkerPool backend.

No GPU required — tests use mocked workers and pipes.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from grail.environments.gpu_kernel.eval_backends import create_backend
from grail.environments.gpu_kernel.eval_backends.persistent_backend import (
    PersistentWorkerPool,
    _WorkerHandle,
)

# =============================================================================
# TestFactory
# =============================================================================


class TestFactory:
    """PersistentWorkerPool is wired into the backend factory."""

    def test_create_persistent_backend(self) -> None:
        backend = create_backend("persistent", gpu_ids=[])
        assert isinstance(backend, PersistentWorkerPool)

    def test_factory_error_message_includes_persistent(self) -> None:
        with pytest.raises(ValueError, match="persistent"):
            create_backend("nonexistent")


# =============================================================================
# TestLifecycle
# =============================================================================


class TestLifecycle:
    """Start / shutdown without real GPU processes."""

    def test_start_shutdown_no_gpus(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[], timeout=5.0)
        pool.start()
        assert pool._started is True
        pool.shutdown()
        assert pool._started is False

    def test_start_is_idempotent(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[], timeout=5.0)
        pool.start()
        pool.start()  # Should not error
        assert pool._started is True
        pool.shutdown()

    def test_evaluate_no_workers_returns_error(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[], timeout=5.0)
        pool.start()
        result = pool.evaluate("test", "triton")
        assert result.correct is False
        assert result.error == "no_workers"
        pool.shutdown()

    def test_evaluate_batch_empty(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[], timeout=5.0)
        pool.start()
        assert pool.evaluate_batch([]) == []
        pool.shutdown()


# =============================================================================
# TestWorkerHandle
# =============================================================================


class TestWorkerHandle:
    """_WorkerHandle bookkeeping."""

    def test_default_state(self) -> None:
        h = _WorkerHandle(gpu_id=0)
        assert h.process is None
        assert h.pipe is None
        assert h.eval_count == 0
        assert h.alive is False

    def test_lock_is_independent(self) -> None:
        h1 = _WorkerHandle(gpu_id=0)
        h2 = _WorkerHandle(gpu_id=1)
        assert h1.lock is not h2.lock


# =============================================================================
# TestEvalWithMockWorker
# =============================================================================


class TestEvalWithMockWorker:
    """Evaluate using a mock pipe to simulate worker responses."""

    def _make_pool_with_mock_worker(
        self, gpu_id: int = 0, timeout: float = 5.0
    ) -> tuple[PersistentWorkerPool, _WorkerHandle]:
        """Create a pool with a single mock worker (no real subprocess)."""
        pool = PersistentWorkerPool(gpu_ids=[gpu_id], timeout=timeout)

        mock_pipe = MagicMock()
        handle = _WorkerHandle(
            gpu_id=gpu_id,
            process=MagicMock(),
            pipe=mock_pipe,
            alive=True,
        )
        pool._workers = {gpu_id: handle}
        pool._started = True

        return pool, handle

    def test_successful_eval(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": True,
            "compiled": True,
            "error": None,
            "max_diff": 0.0001,
        }

        result = pool.evaluate("test_code", "triton_code")

        assert result.correct is True
        assert result.compiled is True
        assert result.max_diff == pytest.approx(0.0001)
        handle.pipe.send.assert_called_once_with(("test_code", "triton_code"))

    def test_failed_eval(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": False,
            "compiled": True,
            "error": "max_diff=0.500000",
            "max_diff": 0.5,
        }

        result = pool.evaluate("test_code", "bad_triton")

        assert result.correct is False
        assert result.compiled is True

    def test_compile_error(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": False,
            "compiled": False,
            "error": "SyntaxError: invalid syntax",
        }

        result = pool.evaluate("test_code", "bad syntax")

        assert result.correct is False
        assert result.compiled is False
        assert "SyntaxError" in result.error

    def test_timeout_respawns_worker(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = False  # Simulate timeout

        with patch.object(pool, "_respawn_worker") as mock_respawn:
            new_handle = _WorkerHandle(gpu_id=0, alive=True)
            mock_respawn.return_value = new_handle

            result = pool.evaluate("test_code", "triton_code")

            assert result.error == "timeout"
            mock_respawn.assert_called_once()

    def test_cuda_corruption_respawns_worker(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": False,
            "compiled": False,
            "error": "illegal memory access",
            "_cuda_corrupted": True,
        }

        with patch.object(pool, "_respawn_worker") as mock_respawn:
            new_handle = _WorkerHandle(gpu_id=0, alive=True)
            mock_respawn.return_value = new_handle

            result = pool.evaluate("test_code", "triton_code")

            assert result.correct is False
            mock_respawn.assert_called_once()

    def test_eval_count_increments(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": True,
            "compiled": True,
            "error": None,
        }

        pool.evaluate("t1", "k1")
        pool.evaluate("t2", "k2")

        assert handle.eval_count == 2

    def test_recycling_after_max_evals(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        pool._max_evals = 2
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = {
            "correct": True,
            "compiled": True,
            "error": None,
        }

        with patch.object(pool, "_respawn_worker") as mock_respawn:
            new_handle = _WorkerHandle(gpu_id=0, alive=True)
            new_handle.pipe = MagicMock()
            new_handle.pipe.poll.return_value = True
            new_handle.pipe.recv.return_value = {
                "correct": True,
                "compiled": True,
                "error": None,
            }
            mock_respawn.return_value = new_handle

            pool.evaluate("t1", "k1")  # eval_count=1
            assert mock_respawn.call_count == 0

            pool.evaluate("t2", "k2")  # eval_count=2 → recycle
            assert mock_respawn.call_count == 1

    def test_worker_init_failure(self) -> None:
        pool, handle = self._make_pool_with_mock_worker()
        handle.pipe.poll.return_value = True
        handle.pipe.recv.return_value = None  # Init failure signal

        with patch.object(pool, "_respawn_worker") as mock_respawn:
            new_handle = _WorkerHandle(gpu_id=0, alive=True)
            mock_respawn.return_value = new_handle

            result = pool.evaluate("test_code", "triton_code")

            assert result.error == "worker_init_failed"
            mock_respawn.assert_called_once()


# =============================================================================
# TestBatchEval
# =============================================================================


class TestBatchEval:
    """Batch evaluation with mock workers."""

    def test_batch_dispatches_to_workers(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[0, 1], timeout=5.0)

        mock_pipes = {}
        for gpu_id in [0, 1]:
            pipe = MagicMock()
            pipe.poll.return_value = True
            pipe.recv.return_value = {
                "correct": True,
                "compiled": True,
                "error": None,
            }
            mock_pipes[gpu_id] = pipe

            handle = _WorkerHandle(
                gpu_id=gpu_id,
                process=MagicMock(),
                pipe=pipe,
                alive=True,
            )
            pool._workers[gpu_id] = handle

        pool._started = True

        results = pool.evaluate_batch(
            [
                ("t1", "k1"),
                ("t2", "k2"),
                ("t3", "k3"),
            ]
        )

        assert len(results) == 3
        assert all(r.correct for r in results)


# =============================================================================
# TestRoundRobin
# =============================================================================


class TestRoundRobin:
    """Worker selection distributes across GPUs."""

    def test_round_robin_cycles(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[0, 1, 2], timeout=5.0)
        for gpu_id in [0, 1, 2]:
            pool._workers[gpu_id] = _WorkerHandle(gpu_id=gpu_id, alive=True)

        selected = [pool._pick_worker().gpu_id for _ in range(6)]
        assert selected == [0, 1, 2, 0, 1, 2]

    def test_round_robin_thread_safe(self) -> None:
        pool = PersistentWorkerPool(gpu_ids=[0, 1], timeout=5.0)
        for gpu_id in [0, 1]:
            pool._workers[gpu_id] = _WorkerHandle(gpu_id=gpu_id, alive=True)

        results = []
        barrier = threading.Barrier(10)

        def pick():
            barrier.wait()
            results.append(pool._pick_worker().gpu_id)

        threads = [threading.Thread(target=pick) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        # Both GPUs should have been selected
        assert set(results) == {0, 1}
