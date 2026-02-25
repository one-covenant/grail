"""Persistent-worker GPU evaluation backend.

Each GPU gets one long-lived worker process that keeps its CUDA context alive
between evaluations.  This amortises the ~5 s subprocess-spawn + CUDA-init
cost to once at pool creation, reducing per-eval latency to ~1-2 s (Triton JIT
+ forward pass).

Workers are health-checked after every eval with a tiny CUDA canary op.  If
the canary fails (CUDA sticky error), the worker exits and the pool respawns a
fresh one automatically.  Workers are also recycled after *max_evals* evals as
a defence-in-depth measure against slow memory leaks.

Communication uses ``multiprocessing.Pipe`` (one pair per worker).  The parent
side is protected by a ``threading.Lock`` so ``evaluate_batch`` can dispatch
from a thread pool safely.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker handle (parent-side bookkeeping)
# ---------------------------------------------------------------------------


@dataclass
class _WorkerHandle:
    """Parent-side handle for one persistent worker process."""

    gpu_id: int
    process: Any = None  # mp.Process or SpawnProcess (context-dependent)
    pipe: Any = None  # mp.Connection (not generic-typed in stdlib)
    lock: threading.Lock = field(default_factory=threading.Lock)
    eval_count: int = 0
    alive: bool = False


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _persistent_worker(pipe: Any, gpu_id: int) -> None:
    """Long-lived subprocess: pin GPU, init CUDA once, serve evals via *pipe*.

    Protocol:
        - Receives ``(test_code, triton_code)`` tuples.
        - Sends back ``dict`` results (same schema as ``run_kernel_eval``).
        - Receives ``None`` as a shutdown sentinel.
        - Exits on shutdown, health-check failure, or unrecoverable error.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Persistent worker started on GPU %d (device=%s)", gpu_id, device)
    except Exception:
        logger.exception("Persistent worker failed to init CUDA on GPU %d", gpu_id)
        pipe.send(None)  # Signal init failure
        return

    from .subprocess_backend import run_kernel_eval

    while True:
        try:
            msg = pipe.recv()
        except (EOFError, OSError):
            logger.info("Persistent worker GPU %d: pipe closed, exiting", gpu_id)
            break

        if msg is None:
            logger.info("Persistent worker GPU %d: shutdown sentinel received", gpu_id)
            break

        test_code, triton_code = msg

        # Run eval
        result = run_kernel_eval(test_code, triton_code, device)

        # Health check: tiny CUDA canary op
        healthy = _cuda_health_check(device)

        # Cleanup: free cached memory
        _cleanup_device(device)

        if not healthy:
            result["_cuda_corrupted"] = True
            logger.warning(
                "Persistent worker GPU %d: CUDA corrupted, will exit after sending result", gpu_id
            )

        try:
            pipe.send(result)
        except (EOFError, OSError):
            logger.info("Persistent worker GPU %d: pipe broken on send, exiting", gpu_id)
            break

        if not healthy:
            break


def _cuda_health_check(device: Any) -> bool:
    """Run a tiny CUDA op to detect sticky errors. Returns False if corrupted."""
    try:
        import torch

        x = torch.ones(1, device=device)
        return (x + 1).item() == 2.0
    except Exception:
        return False


def _cleanup_device(device: Any) -> None:
    """Best-effort GPU memory cleanup between evals."""
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# PersistentWorkerPool
# ---------------------------------------------------------------------------


class PersistentWorkerPool:
    """GPU kernel evaluation backend with persistent worker processes.

    One long-lived worker per GPU.  Workers keep their CUDA context alive
    between evals, eliminating the ~5 s spawn+init overhead of
    :class:`SubprocessBackend`.

    Args:
        gpu_ids: GPU device indices for kernel evaluation.
        timeout: Per-kernel evaluation timeout in seconds.
        max_workers: Max parallel evals.  Defaults to ``len(gpu_ids)`` or 1.
        max_evals_per_worker: Recycle a worker after this many evals to
            guard against slow memory leaks.  ``0`` disables recycling.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        timeout: float = 60.0,
        max_workers: int | None = None,
        max_evals_per_worker: int = 100,
        **_kwargs: Any,
    ) -> None:
        self._gpu_ids = gpu_ids or []
        self._timeout = timeout
        self._max_workers = max_workers or max(len(self._gpu_ids), 1)
        self._max_evals = max_evals_per_worker
        self._workers: dict[int, _WorkerHandle] = {}
        self._started = False
        self._round_robin = 0
        self._rr_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn one persistent worker per GPU."""
        if self._started:
            return
        for gpu_id in self._gpu_ids:
            self._spawn_worker(gpu_id)
        self._started = True
        logger.info(
            "PersistentWorkerPool started: %d workers on GPUs %s (max %d parallel)",
            len(self._workers),
            self._gpu_ids,
            self._max_workers,
        )

    def shutdown(self) -> None:
        """Send shutdown sentinel to all workers and join them."""
        for _gpu_id, handle in list(self._workers.items()):
            self._stop_worker(handle)
        self._workers.clear()
        self._started = False
        logger.info("PersistentWorkerPool shut down")

    # -- public API ---------------------------------------------------------

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single kernel on a persistent worker."""
        if not self._workers:
            return EvalResult(correct=False, compiled=False, error="no_workers")

        handle = self._pick_worker()
        return self._evaluate_on_worker(handle, test_code, triton_code)

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels in parallel across persistent workers."""
        if not items:
            return []

        results: list[EvalResult | None] = [None] * len(items)

        def _run(idx: int, test_code: str, triton_code: str) -> None:
            handle = self._pick_worker()
            results[idx] = self._evaluate_on_worker(handle, test_code, triton_code)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [pool.submit(_run, i, tc, tr) for i, (tc, tr) in enumerate(items)]
            for f in futures:
                f.result()

        return [
            r
            if r is not None
            else EvalResult(correct=False, compiled=False, error="internal_error")
            for r in results
        ]

    def warmup(self, sample_test_codes: list[str]) -> None:
        """Warm up Triton JIT cache by sending a warmup kernel to each worker."""
        if not self._workers:
            logger.info("Warmup: no workers, skipping")
            return

        from .subprocess_backend import WARMUP_TRITON_CODE

        n = min(len(sample_test_codes), len(self._workers))
        logger.info("Warmup: sending %d kernels to %d workers", n, len(self._workers))
        start = time.monotonic()

        for i, (gpu_id, handle) in enumerate(self._workers.items()):
            if i >= n:
                break
            try:
                result = self._evaluate_on_worker(handle, sample_test_codes[i], WARMUP_TRITON_CODE)
                logger.info(
                    "Warmup: GPU %d compiled=%s (%.1fs)",
                    gpu_id,
                    result.compiled,
                    time.monotonic() - start,
                )
            except Exception as e:
                logger.warning("Warmup: GPU %d failed: %s", gpu_id, e)

        logger.info("Warmup: completed in %.1fs", time.monotonic() - start)

    # -- internals ----------------------------------------------------------

    def _spawn_worker(self, gpu_id: int) -> _WorkerHandle:
        """Spawn a persistent worker for *gpu_id* and register it."""
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()

        proc = ctx.Process(
            target=_persistent_worker,
            args=(child_conn, gpu_id),
            daemon=True,
        )
        proc.start()
        child_conn.close()

        handle = _WorkerHandle(
            gpu_id=gpu_id,
            process=proc,
            pipe=parent_conn,
            alive=True,
        )
        self._workers[gpu_id] = handle
        logger.info("Spawned persistent worker for GPU %d (pid=%d)", gpu_id, proc.pid)
        return handle

    def _stop_worker(self, handle: _WorkerHandle) -> None:
        """Send shutdown sentinel and join."""
        if not handle.alive:
            return
        handle.alive = False
        try:
            handle.pipe.send(None)
        except (EOFError, OSError, BrokenPipeError):
            pass
        if handle.process is not None:
            handle.process.join(timeout=10)
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join()
        try:
            handle.pipe.close()
        except OSError:
            pass

    def _respawn_worker(self, handle: _WorkerHandle) -> _WorkerHandle:
        """Stop a worker and spawn a fresh replacement."""
        gpu_id = handle.gpu_id
        logger.info("Respawning worker for GPU %d (evals=%d)", gpu_id, handle.eval_count)
        self._stop_worker(handle)
        return self._spawn_worker(gpu_id)

    def _pick_worker(self) -> _WorkerHandle:
        """Round-robin worker selection (thread-safe)."""
        with self._rr_lock:
            gpu_ids = list(self._workers.keys())
            idx = self._round_robin % len(gpu_ids)
            self._round_robin += 1
        return self._workers[gpu_ids[idx]]

    def _evaluate_on_worker(
        self,
        handle: _WorkerHandle,
        test_code: str,
        triton_code: str,
    ) -> EvalResult:
        """Send eval job to a worker, receive result, handle failures."""
        with handle.lock:
            if not handle.alive:
                # Worker died between pick and lock acquisition; respawn
                handle = self._respawn_worker(handle)

            try:
                handle.pipe.send((test_code, triton_code))
            except (EOFError, OSError, BrokenPipeError):
                handle = self._respawn_worker(handle)
                try:
                    handle.pipe.send((test_code, triton_code))
                except (EOFError, OSError, BrokenPipeError):
                    return EvalResult(correct=False, compiled=False, error="worker_send_failed")

            # Wait for result with timeout
            if not handle.pipe.poll(timeout=self._timeout):
                logger.warning("Worker GPU %d timed out after %.1fs", handle.gpu_id, self._timeout)
                handle = self._respawn_worker(handle)
                return EvalResult(correct=False, compiled=False, error="timeout")

            try:
                result_dict = handle.pipe.recv()
            except (EOFError, OSError):
                handle = self._respawn_worker(handle)
                return EvalResult(correct=False, compiled=False, error="worker_recv_failed")

            # Worker init failure
            if result_dict is None:
                handle = self._respawn_worker(handle)
                return EvalResult(correct=False, compiled=False, error="worker_init_failed")

            handle.eval_count += 1

            # CUDA corruption → respawn
            if result_dict.pop("_cuda_corrupted", False):
                logger.warning("Worker GPU %d: CUDA corrupted, respawning", handle.gpu_id)
                handle = self._respawn_worker(handle)

            # Periodic recycling
            elif self._max_evals > 0 and handle.eval_count >= self._max_evals:
                logger.info(
                    "Worker GPU %d: reached %d evals, recycling", handle.gpu_id, handle.eval_count
                )
                handle = self._respawn_worker(handle)

            return EvalResult(
                correct=result_dict.get("correct", False),
                compiled=result_dict.get("compiled", False),
                error=result_dict.get("error"),
                max_diff=result_dict.get("max_diff"),
            )
