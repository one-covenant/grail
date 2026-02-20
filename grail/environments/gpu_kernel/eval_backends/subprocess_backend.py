"""Subprocess-based GPU evaluation backend.

Uses a ProcessPoolExecutor with spawn context for persistent GPU workers.
Workers run kernel evaluations directly (no nested subprocess), reusing the
CUDA context across calls for speed. The pool is monitored for health — if
consecutive failures exceed a threshold, the pool is torn down and recreated.

Fallback: when the pool is unavailable (during restart or if no GPUs configured),
evaluations run in isolated one-shot subprocesses.

Note: Triton's @jit decorator requires source code from a real .py file
(it calls inspect.getsourcelines). We write code to temp files before exec.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
import threading
import time
from concurrent.futures import (
    BrokenExecutor,
    ProcessPoolExecutor,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from typing import Any

from . import EvalResult

logger = logging.getLogger(__name__)

# Pool health thresholds
_CONSECUTIVE_FAILURE_THRESHOLD = 3  # Restart pool after N consecutive failures
_MAX_RESTARTS = 5  # Give up restarting after N attempts


class SubprocessBackend:
    """GPU kernel evaluation backend with resilient process pool.

    Architecture:
    - A ProcessPoolExecutor with spawn context keeps persistent workers.
    - Workers pin to GPU via CUDA_VISIBLE_DEVICES and keep CUDA context warm.
    - Pool health is tracked: consecutive failures trigger automatic restart.
    - If pool is broken/restarting, falls back to one-shot subprocess isolation.

    Args:
        gpu_ids: List of GPU device indices to use for evaluation.
        timeout: Per-kernel evaluation timeout in seconds.
        max_workers: Maximum concurrent workers. Defaults to len(gpu_ids) or 1.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        timeout: float = 60.0,
        max_workers: int | None = None,
        **kwargs: object,
    ) -> None:
        self._gpu_ids = gpu_ids or []
        self._timeout = timeout
        self._max_workers = max_workers or max(len(self._gpu_ids), 1)
        self._pool: ProcessPoolExecutor | None = None
        self._started = False
        # Health tracking
        self._consecutive_failures = 0
        self._restart_count = 0
        self._lock = threading.Lock()

    def _try_restart_pool(self) -> None:
        """Restart the pool if consecutive failures exceed threshold.

        Thread-safe. Only one restart can happen at a time.
        """
        with self._lock:
            if self._consecutive_failures < _CONSECUTIVE_FAILURE_THRESHOLD:
                return
            if self._restart_count >= _MAX_RESTARTS:
                logger.error(
                    "Pool exceeded max restarts (%d). "
                    "Falling back to one-shot subprocess for all evals.",
                    _MAX_RESTARTS,
                )
                self._pool = None
                return

            logger.warning(
                "Pool health check: %d consecutive failures, restarting pool (attempt %d/%d)",
                self._consecutive_failures,
                self._restart_count + 1,
                _MAX_RESTARTS,
            )
            # Tear down old pool
            if self._pool is not None:
                try:
                    self._pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._pool = None

            # Create new pool
            try:
                ctx = mp.get_context("spawn")
                self._pool = ProcessPoolExecutor(
                    max_workers=self._max_workers,
                    mp_context=ctx,
                )
                self._restart_count += 1
                self._consecutive_failures = 0
                logger.info(
                    "Pool restarted successfully (%d workers on GPUs %s)",
                    self._max_workers,
                    self._gpu_ids,
                )
            except Exception as e:
                logger.error("Failed to restart pool: %s", e)
                self._pool = None

    def _record_success(self) -> None:
        """Reset consecutive failure count on success."""
        self._consecutive_failures = 0

    def _record_failure(self, error: str) -> None:
        """Track a failure and trigger restart if needed."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
            logger.warning(
                "Pool failure %d/%d: %s",
                self._consecutive_failures,
                _CONSECUTIVE_FAILURE_THRESHOLD,
                error,
            )
            self._try_restart_pool()

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single kernel. Uses pool if healthy, else one-shot subprocess."""
        gpu_id = self._gpu_ids[0] if self._gpu_ids else None

        # Try pool first
        if self._pool is not None:
            try:
                future = self._pool.submit(_eval_worker, test_code, triton_code, gpu_id)
                result = future.result(timeout=self._timeout)
                self._record_success()
                return result
            except FuturesTimeoutError:
                self._record_failure("timeout")
                return EvalResult(correct=False, compiled=False, error="timeout")
            except BrokenExecutor as e:
                self._record_failure(f"broken_pool: {e}")
                # Fall through to one-shot subprocess
            except Exception as e:
                self._record_failure(str(e))
                return EvalResult(correct=False, compiled=False, error=str(e))

        # Fallback: one-shot isolated subprocess (always safe)
        return _run_eval_in_subprocess(test_code, triton_code, gpu_id, self._timeout)

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels, distributing across GPU workers."""
        if not items:
            return []

        results: list[EvalResult] = []
        if self._pool is not None and self._gpu_ids:
            futures = []
            for i, (test_code, triton_code) in enumerate(items):
                gpu_id = self._gpu_ids[i % len(self._gpu_ids)]
                try:
                    future = self._pool.submit(_eval_worker, test_code, triton_code, gpu_id)
                    futures.append((future, i))
                except BrokenExecutor:
                    self._record_failure("broken_pool_batch")
                    # Fall back to one-shot for remaining items
                    for test_c, triton_c in items[i:]:
                        gid = self._gpu_ids[0] if self._gpu_ids else None
                        results.append(
                            _run_eval_in_subprocess(test_c, triton_c, gid, self._timeout)
                        )
                    break

            for future, _idx in futures:
                try:
                    result = future.result(timeout=self._timeout)
                    self._record_success()
                    results.append(result)
                except FuturesTimeoutError:
                    self._record_failure("timeout_batch")
                    results.append(EvalResult(correct=False, compiled=False, error="timeout"))
                except Exception as e:
                    self._record_failure(str(e))
                    results.append(EvalResult(correct=False, compiled=False, error=str(e)))
        else:
            # Sequential fallback (no pool)
            for test_code, triton_code in items:
                results.append(self.evaluate(test_code, triton_code))

        return results

    def warmup(self, sample_test_codes: list[str]) -> None:
        """Warm up JIT cache and CUDA context on each eval GPU.

        Uses the pool if available (warms the persistent workers). Falls back
        to one-shot subprocesses (warms JIT cache on disk, not workers).
        """
        if not self._gpu_ids:
            logger.info("Warmup: no GPU IDs configured, skipping")
            return

        n = min(len(sample_test_codes), 20)
        if n == 0:
            logger.info("Warmup: no sample codes provided, skipping")
            return

        logger.info("Warmup: compiling %d kernels across GPUs %s", n, self._gpu_ids)
        start = time.monotonic()

        warmup_triton_code = _WARMUP_TRITON_CODE

        for i in range(n):
            gpu_id = self._gpu_ids[i % len(self._gpu_ids)]
            try:
                if self._pool is not None:
                    future = self._pool.submit(
                        _eval_worker, sample_test_codes[i], warmup_triton_code, gpu_id
                    )
                    result = future.result(timeout=self._timeout)
                else:
                    result = _run_eval_in_subprocess(
                        sample_test_codes[i], warmup_triton_code, gpu_id, self._timeout
                    )
                elapsed = time.monotonic() - start
                logger.info(
                    "Warmup: %d/%d on GPU %d [%.1fs] compiled=%s",
                    i + 1,
                    n,
                    gpu_id,
                    elapsed,
                    result.compiled,
                )
            except Exception as e:
                logger.warning("Warmup: %d/%d failed on GPU %d: %s", i + 1, n, gpu_id, e)

        elapsed = time.monotonic() - start
        logger.info("Warmup: completed %d kernels in %.1fs", n, elapsed)

    def start(self) -> None:
        """Start the process pool."""
        if self._started:
            return

        if self._gpu_ids:
            ctx = mp.get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=ctx,
            )
            logger.info(
                "SubprocessBackend started: %d workers on GPUs %s",
                self._max_workers,
                self._gpu_ids,
            )
        else:
            logger.info("SubprocessBackend started: no GPUs, eval will return compiled=False")

        self._started = True
        self._consecutive_failures = 0
        self._restart_count = 0

    def shutdown(self) -> None:
        """Shut down the process pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        self._started = False
        logger.info("SubprocessBackend shut down")


# ---------------------------------------------------------------------------
# Subprocess worker functions (module-level for pickling)
# ---------------------------------------------------------------------------


def _run_eval_in_subprocess(
    test_code: str,
    triton_code: str,
    gpu_id: int | None,
    timeout: float,
) -> EvalResult:
    """Run evaluation in a fresh spawned subprocess for CUDA isolation."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=_eval_worker_process,
        args=(child_conn, test_code, triton_code, gpu_id),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return EvalResult(correct=False, compiled=False, error="timeout")

    if parent_conn.poll():
        result_dict = parent_conn.recv()
        return EvalResult(
            correct=result_dict.get("correct", False),
            compiled=result_dict.get("compiled", False),
            error=result_dict.get("error"),
            max_diff=result_dict.get("max_diff"),
        )

    return EvalResult(correct=False, compiled=False, error="no_result")


def _exec_code_from_file(code: str, globals_dict: dict[str, Any]) -> None:
    """Execute Python code via a temp file so inspect.getsourcelines() works.

    Triton's @jit decorator calls inspect.getsourcelines(fn), which requires
    the decorated function to exist in a real .py file. Plain exec() creates
    code in '<string>' which makes getsourcelines() raise OSError.

    This writes the code to a temporary .py file, compiles it with the file
    path set, and then execs the compiled code. The temp file is kept alive
    long enough for Triton's JIT to read it.
    """
    fd, path = tempfile.mkstemp(suffix=".py", prefix="grail_kernel_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        compiled = compile(code, path, "exec")
        globals_dict["__file__"] = path
        exec(compiled, globals_dict)  # noqa: S102
    finally:
        # Don't delete yet — Triton JIT may read the file lazily during forward().
        # We rely on OS temp cleanup or delete after eval completes.
        pass


def _cleanup_temp_file(path: str | None) -> None:
    """Best-effort cleanup of a temp file."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _eval_worker(test_code: str, triton_code: str, gpu_id: int | None) -> EvalResult:
    """Worker function for ProcessPoolExecutor — runs eval directly in pool worker.

    This avoids spawning another subprocess (faster), accepting that a CUDA
    sticky error could corrupt this worker. The pool will replace dead workers.
    """
    return _eval_direct(test_code, triton_code, gpu_id)


def _eval_direct(test_code: str, triton_code: str, gpu_id: int | None) -> EvalResult:
    """Run kernel eval directly in the current process (no subprocess spawn)."""
    import traceback

    _log = logging.getLogger(__name__)
    triton_file: str | None = None
    test_file: str | None = None
    t_total = time.monotonic()
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch  # noqa: F811

        # Phase 1: Execute test code
        t0 = time.monotonic()
        test_globals: dict[str, Any] = {}
        _exec_code_from_file(test_code, test_globals)
        test_file = test_globals.get("__file__")
        t_test = time.monotonic() - t0

        model_class = test_globals.get("Model")
        get_inputs = test_globals.get("get_inputs")
        get_init_inputs = test_globals.get("get_init_inputs")
        check_correctness = test_globals.get("check_correctness")

        if model_class is None or get_inputs is None:
            _log.info(
                "[kernel_eval] gpu=%s test_exec=%.2fs FAIL: missing_model_or_inputs", gpu_id, t_test
            )
            return EvalResult(correct=False, compiled=False, error="missing_model_or_inputs")

        # Phase 2: Execute Triton code (JIT compilation happens here)
        t0 = time.monotonic()
        gen_globals: dict[str, Any] = {}
        _exec_code_from_file(triton_code, gen_globals)
        triton_file = gen_globals.get("__file__")
        t_triton_exec = time.monotonic() - t0

        model_new_class = gen_globals.get("ModelNew")
        if model_new_class is None:
            _log.info(
                "[kernel_eval] gpu=%s test_exec=%.2fs triton_exec=%.2fs FAIL: no_model_new_class",
                gpu_id,
                t_test,
                t_triton_exec,
            )
            return EvalResult(correct=False, compiled=True, error="no_model_new_class")

        # Phase 3: Correctness check
        if check_correctness is not None:
            t0 = time.monotonic()
            try:
                result = check_correctness(model_new_class)
                t_check = time.monotonic() - t0
                if isinstance(result, dict):
                    correct = result.get("correct", False)
                    err = result.get("error")
                    _log.info(
                        "[kernel_eval] gpu=%s test=%.2fs triton=%.2fs check=%.2fs correct=%s err=%s total=%.2fs",
                        gpu_id,
                        t_test,
                        t_triton_exec,
                        t_check,
                        correct,
                        err,
                        time.monotonic() - t_total,
                    )
                    return EvalResult(
                        correct=correct,
                        compiled=True,
                        error=err,
                        max_diff=result.get("max_diff"),
                    )
                t_check = time.monotonic() - t0
                _log.info(
                    "[kernel_eval] gpu=%s test=%.2fs triton=%.2fs check=%.2fs correct=%s total=%.2fs",
                    gpu_id,
                    t_test,
                    t_triton_exec,
                    t_check,
                    bool(result),
                    time.monotonic() - t_total,
                )
                return EvalResult(
                    correct=bool(result),
                    compiled=True,
                    error=None if result else "check_correctness_failed",
                )
            except Exception as e:
                t_check = time.monotonic() - t0
                _log.info(
                    "[kernel_eval] gpu=%s test=%.2fs triton=%.2fs check=%.2fs FAIL: %s total=%.2fs",
                    gpu_id,
                    t_test,
                    t_triton_exec,
                    t_check,
                    e,
                    time.monotonic() - t_total,
                )
                return EvalResult(
                    correct=False, compiled=True, error=f"check_correctness_error: {e}"
                )

        # Fallback: manual correctness check
        t0 = time.monotonic()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_inputs = get_init_inputs() if get_init_inputs else []
        ref_model = model_class(*init_inputs).to(device).eval()
        new_model = model_new_class(*init_inputs).to(device).eval()
        t_model_init = time.monotonic() - t0

        max_diff = 0.0
        tolerance = 1e-2

        t0 = time.monotonic()
        for trial in range(3):
            torch.manual_seed(42 + trial)
            inputs = get_inputs()
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

            with torch.no_grad():
                ref_out = ref_model(*inputs)
                new_out = new_model(*inputs)

            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            if isinstance(new_out, tuple):
                new_out = new_out[0]

            if ref_out.shape != new_out.shape:
                t_trials = time.monotonic() - t0
                _log.info(
                    "[kernel_eval] gpu=%s test=%.2fs triton=%.2fs init=%.2fs trials=%.2fs FAIL: shape_mismatch %s vs %s total=%.2fs",
                    gpu_id,
                    t_test,
                    t_triton_exec,
                    t_model_init,
                    t_trials,
                    ref_out.shape,
                    new_out.shape,
                    time.monotonic() - t_total,
                )
                return EvalResult(
                    correct=False,
                    compiled=True,
                    error=f"shape_mismatch: {ref_out.shape} vs {new_out.shape}",
                )

            diff = torch.max(torch.abs(ref_out.float() - new_out.float())).item()
            max_diff = max(max_diff, diff)

        t_trials = time.monotonic() - t0
        correct = max_diff <= tolerance
        _log.info(
            "[kernel_eval] gpu=%s test=%.2fs triton=%.2fs init=%.2fs trials=%.2fs correct=%s max_diff=%.6f total=%.2fs",
            gpu_id,
            t_test,
            t_triton_exec,
            t_model_init,
            t_trials,
            correct,
            max_diff,
            time.monotonic() - t_total,
        )
        return EvalResult(
            correct=correct,
            compiled=True,
            error=None if correct else f"max_diff={max_diff:.6f}",
            max_diff=max_diff,
        )
    except Exception:
        _log.info(
            "[kernel_eval] gpu=%s EXCEPTION after %.2fs: %s",
            gpu_id,
            time.monotonic() - t_total,
            traceback.format_exc()[-500:],
        )
        return EvalResult(
            correct=False, compiled=False, error=f"eval_error: {traceback.format_exc()}"
        )
    finally:
        _cleanup_temp_file(triton_file)
        _cleanup_temp_file(test_file)


def _eval_worker_process(
    conn: Any,
    test_code: str,
    triton_code: str,
    gpu_id: int | None,
) -> None:
    """Subprocess entry point for kernel evaluation.

    Executes test_code to get Model, get_inputs(), get_init_inputs(), and
    check_correctness(). Then executes triton_code to get ModelNew.
    Calls check_correctness(ModelNew) to verify correctness.
    """
    import traceback

    triton_file: str | None = None
    test_file: str | None = None
    try:
        # Pin to specific GPU
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch  # noqa: F811

        # Execute test code via temp file
        test_globals: dict[str, Any] = {}
        _exec_code_from_file(test_code, test_globals)
        test_file = test_globals.get("__file__")

        model_class = test_globals.get("Model")
        get_inputs = test_globals.get("get_inputs")
        get_init_inputs = test_globals.get("get_init_inputs")
        check_correctness = test_globals.get("check_correctness")

        if model_class is None or get_inputs is None:
            conn.send(
                {
                    "correct": False,
                    "compiled": False,
                    "error": "missing_model_or_inputs",
                }
            )
            return

        # Execute Triton code via temp file (required for @triton.jit)
        gen_globals: dict[str, Any] = {}
        _exec_code_from_file(triton_code, gen_globals)
        triton_file = gen_globals.get("__file__")

        model_new_class = gen_globals.get("ModelNew")
        if model_new_class is None:
            conn.send(
                {
                    "correct": False,
                    "compiled": True,
                    "error": "no_model_new_class",
                }
            )
            return

        # Use check_correctness if available (preferred path)
        if check_correctness is not None:
            try:
                result = check_correctness(model_new_class)
                if isinstance(result, dict):
                    conn.send(result)
                elif isinstance(result, bool):
                    conn.send(
                        {
                            "correct": result,
                            "compiled": True,
                            "error": None if result else "check_correctness_failed",
                        }
                    )
                else:
                    conn.send(
                        {
                            "correct": bool(result),
                            "compiled": True,
                            "error": None,
                        }
                    )
            except Exception as e:
                conn.send(
                    {
                        "correct": False,
                        "compiled": True,
                        "error": f"check_correctness_error: {e}",
                    }
                )
            return

        # Fallback: manual correctness check (same as old _gpu_eval_worker)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        init_inputs = get_init_inputs() if get_init_inputs else []
        ref_model = model_class(*init_inputs).to(device).eval()
        new_model = model_new_class(*init_inputs).to(device).eval()

        n_trials = 3
        max_diff = 0.0
        tolerance = 1e-2

        for trial in range(n_trials):
            torch.manual_seed(42 + trial)
            inputs = get_inputs()
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

            with torch.no_grad():
                ref_out = ref_model(*inputs)
                new_out = new_model(*inputs)

            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            if isinstance(new_out, tuple):
                new_out = new_out[0]

            if ref_out.shape != new_out.shape:
                conn.send(
                    {
                        "correct": False,
                        "compiled": True,
                        "error": f"shape_mismatch: {ref_out.shape} vs {new_out.shape}",
                        "max_diff": None,
                    }
                )
                return

            diff = torch.max(torch.abs(ref_out.float() - new_out.float())).item()
            max_diff = max(max_diff, diff)

        correct = max_diff <= tolerance
        conn.send(
            {
                "correct": correct,
                "compiled": True,
                "error": None if correct else f"max_diff={max_diff:.6f}",
                "max_diff": max_diff,
            }
        )

    except Exception as e:
        tb = traceback.format_exc()
        conn.send(
            {
                "correct": False,
                "compiled": False,
                "error": f"{type(e).__name__}: {e}\n{tb}",
                "max_diff": None,
            }
        )
    finally:
        _cleanup_temp_file(triton_file)
        _cleanup_temp_file(test_file)


# Minimal Triton kernel for warmup (exercises JIT compilation)
_WARMUP_TRITON_CODE = """
import torch
import triton
import triton.language as tl

@triton.jit
def _warmup_kernel(x_ptr, out_ptr, n: tl.constexpr):
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
        _warmup_kernel[(n,)](x, out, n)
        return out
"""
