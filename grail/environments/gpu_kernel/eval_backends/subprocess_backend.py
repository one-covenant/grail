"""Subprocess-isolated GPU evaluation backend.

Every kernel evaluation runs in a **fresh spawned subprocess** with its own
CUDA context.  This guarantees that a CUDA sticky error (illegal memory
access, device-side assert, …) in one kernel can never poison subsequent
evaluations — each subprocess is born clean and dies after one job.

Batch evaluation runs subprocesses in parallel via a ThreadPoolExecutor
(threads just wait on their child process; the real GPU work is in the
subprocess).  Concurrency is bounded by the number of configured GPUs.

Transient CUDA errors are automatically retried (up to 3 attempts) using
``tenacity``.  Non-CUDA errors (NameError, shape mismatch, …) are
legitimate failures and are **not** retried.

Note: Triton's ``@jit`` decorator requires source code from a real ``.py``
file (it calls ``inspect.getsourcelines``).  We write code to temp files
before ``exec()``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tenacity import (
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_none,
)

from . import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA sticky-error detection
# ---------------------------------------------------------------------------

_CUDA_STICKY_PATTERNS = (
    "illegal memory access",
    "device-side assert",
    "an illegal instruction",
    "unspecified launch failure",
    "cudaErrorIllegalAddress",
    "cudaErrorAssert",
    "cudaErrorLaunchFailure",
)


def _is_cuda_sticky_error(error: str | None) -> bool:
    """Return True if *error* indicates a CUDA sticky error."""
    if error is None:
        return False
    error_lower = error.lower()
    return any(p.lower() in error_lower for p in _CUDA_STICKY_PATTERNS)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class SubprocessBackend:
    """GPU kernel evaluation backend with per-eval subprocess isolation.

    Every ``evaluate()`` call spawns a fresh subprocess that pins to the
    target GPU, runs the kernel, and exits.  This eliminates CUDA context
    corruption at the cost of ~5-10 s per eval for CUDA/Triton init (amortised
    by Triton's on-disk JIT cache after the first compilation).

    For batch evaluation, subprocesses run in parallel via a thread pool so
    the wall-clock overhead stays roughly constant regardless of batch size.

    Args:
        gpu_ids: GPU device indices for kernel evaluation.
        timeout: Per-kernel evaluation timeout in seconds.
        max_workers: Max parallel subprocesses.  Defaults to ``len(gpu_ids)``
            (one subprocess per GPU) or 1.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        timeout: float = 60.0,
        max_workers: int | None = None,
        **_kwargs: object,
    ) -> None:
        self._gpu_ids = gpu_ids or []
        self._timeout = timeout
        self._max_workers = max_workers or max(len(self._gpu_ids), 1)
        self._started = False

    # -- public API ---------------------------------------------------------

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single kernel in an isolated subprocess.

        Automatically retries (up to 3×) when a CUDA sticky error is detected,
        since each retry runs in a brand-new process with a clean context.
        """
        gpu_id = self._gpu_ids[0] if self._gpu_ids else None
        return _eval_subprocess_with_retry(
            test_code,
            triton_code,
            gpu_id,
            self._timeout,
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels in parallel subprocesses.

        Each item is dispatched to a thread that spawns an isolated subprocess.
        GPU IDs are assigned round-robin across the batch.
        """
        if not items:
            return []

        results: list[EvalResult | None] = [None] * len(items)

        def _run(idx: int, test_code: str, triton_code: str) -> None:
            gpu_id = self._gpu_ids[idx % len(self._gpu_ids)] if self._gpu_ids else None
            results[idx] = _eval_subprocess_with_retry(
                test_code,
                triton_code,
                gpu_id,
                self._timeout,
            )

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = []
            for i, (test_code, triton_code) in enumerate(items):
                futures.append(pool.submit(_run, i, test_code, triton_code))
            # Wait for all to finish; propagate first exception if any
            for f in futures:
                f.result()

        # Every slot should be filled; guard against programming errors.
        return [
            r
            if r is not None
            else EvalResult(correct=False, compiled=False, error="internal_error")
            for r in results
        ]

    def warmup(self, sample_test_codes: list[str]) -> None:
        """Warm up Triton JIT cache on each eval GPU via isolated subprocesses."""
        if not self._gpu_ids:
            logger.info("Warmup: no GPU IDs configured, skipping")
            return

        n = min(len(sample_test_codes), 20)
        if n == 0:
            logger.info("Warmup: no sample codes provided, skipping")
            return

        logger.info("Warmup: compiling %d kernels across GPUs %s", n, self._gpu_ids)
        start = time.monotonic()

        for i in range(n):
            gpu_id = self._gpu_ids[i % len(self._gpu_ids)]
            try:
                result = _run_eval_in_subprocess(
                    sample_test_codes[i],
                    WARMUP_TRITON_CODE,
                    gpu_id,
                    self._timeout,
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
        """Mark the backend as started (no persistent resources to create)."""
        if self._started:
            return
        if self._gpu_ids:
            logger.info(
                "SubprocessBackend started: subprocess isolation on GPUs %s (max %d parallel)",
                self._gpu_ids,
                self._max_workers,
            )
        else:
            logger.info("SubprocessBackend started: no GPUs, eval will return compiled=False")
        self._started = True

    def shutdown(self) -> None:
        """Mark the backend as stopped (no persistent resources to release)."""
        self._started = False
        logger.info("SubprocessBackend shut down")


# ---------------------------------------------------------------------------
# Subprocess helpers (module-level for pickling / multiprocessing)
# ---------------------------------------------------------------------------


def _should_retry(result: EvalResult) -> bool:
    """Tenacity retry predicate: retry only on CUDA sticky errors."""
    return _is_cuda_sticky_error(result.error)


@retry(
    retry=retry_if_result(_should_retry),
    stop=stop_after_attempt(3),
    wait=wait_none(),
    reraise=True,
)
def _eval_subprocess_with_retry(
    test_code: str,
    triton_code: str,
    gpu_id: int | None,
    timeout: float,
) -> EvalResult:
    """Run evaluation in a subprocess, retrying on CUDA sticky errors.

    Each attempt spawns a brand-new process (clean CUDA context), so a
    transient CUDA error in attempt N cannot affect attempt N+1.
    """
    result = _run_eval_in_subprocess(test_code, triton_code, gpu_id, timeout)
    if _is_cuda_sticky_error(result.error):
        logger.warning(
            "CUDA sticky error on GPU %s, retrying in fresh subprocess: %s",
            gpu_id,
            (result.error or "")[:200],
        )
    return result


def _run_eval_in_subprocess(
    test_code: str,
    triton_code: str,
    gpu_id: int | None,
    timeout: float,
) -> EvalResult:
    """Spawn a fresh subprocess, run the eval, collect the result via Pipe."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=_eval_worker_process,
        args=(child_conn, test_code, triton_code, gpu_id),
    )
    proc.start()
    # Close the child end in the parent so recv() raises on child death.
    child_conn.close()

    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join()
        parent_conn.close()
        return EvalResult(correct=False, compiled=False, error="timeout")

    if parent_conn.poll():
        try:
            result_dict = parent_conn.recv()
        except (EOFError, OSError):
            parent_conn.close()
            return EvalResult(correct=False, compiled=False, error="pipe_error")
        parent_conn.close()
        return EvalResult(
            correct=result_dict.get("correct", False),
            compiled=result_dict.get("compiled", False),
            error=result_dict.get("error"),
            max_diff=result_dict.get("max_diff"),
        )

    parent_conn.close()
    return EvalResult(correct=False, compiled=False, error="no_result")


# ---------------------------------------------------------------------------
# Code-execution helpers (shared by subprocess entry point)
# ---------------------------------------------------------------------------


def _exec_code_from_file(code: str, globals_dict: dict[str, Any]) -> None:
    """Execute Python code via a temp file so ``inspect.getsourcelines()`` works.

    Triton's ``@jit`` decorator calls ``inspect.getsourcelines(fn)``, which
    requires the decorated function to exist in a real ``.py`` file.  Plain
    ``exec()`` creates code in ``'<string>'`` which makes ``getsourcelines()``
    raise ``OSError``.
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
        # Cleanup happens in the finally block of _eval_worker_process.
        pass


def _cleanup_temp_file(path: str | None) -> None:
    """Best-effort cleanup of a temp file."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def run_kernel_eval(test_code: str, triton_code: str, device: Any = None) -> dict:
    """Execute test harness + generated kernel and return result dict.

    This is the core eval logic shared by :class:`SubprocessBackend` (per-eval
    process) and :class:`~.persistent_backend.PersistentWorkerPool` (long-lived
    process).  Both call this function identically, ensuring miner/validator
    agreement regardless of backend choice.

    Args:
        test_code: Test harness containing ``Model``, ``get_inputs()``,
            ``get_init_inputs()``, and optionally ``check_correctness()``.
        triton_code: Generated code defining ``ModelNew``.
        device: ``torch.device`` to use.  Defaults to ``"cuda"`` if available.

    Returns:
        Dict with keys ``correct``, ``compiled``, ``error``, ``max_diff``.
    """
    import traceback

    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    triton_file: str | None = None
    test_file: str | None = None
    try:
        # Phase 1 — execute test harness
        test_globals: dict[str, Any] = {}
        _exec_code_from_file(test_code, test_globals)
        test_file = test_globals.get("__file__")

        model_class = test_globals.get("Model")
        get_inputs = test_globals.get("get_inputs")
        get_init_inputs = test_globals.get("get_init_inputs")
        check_correctness = test_globals.get("check_correctness")

        if model_class is None or get_inputs is None:
            return {"correct": False, "compiled": False, "error": "missing_model_or_inputs"}

        # Phase 2 — execute generated Triton code
        gen_globals: dict[str, Any] = {}
        _exec_code_from_file(triton_code, gen_globals)
        triton_file = gen_globals.get("__file__")

        model_new_class = gen_globals.get("ModelNew")
        if model_new_class is None:
            return {"correct": False, "compiled": True, "error": "no_model_new_class"}

        # Phase 3a — preferred: use check_correctness() from the test harness
        if check_correctness is not None:
            try:
                result = check_correctness(model_new_class)
                if isinstance(result, dict):
                    return result
                if isinstance(result, bool):
                    return {
                        "correct": result,
                        "compiled": True,
                        "error": None if result else "check_correctness_failed",
                    }
                return {"correct": bool(result), "compiled": True, "error": None}
            except Exception as e:
                return {
                    "correct": False,
                    "compiled": True,
                    "error": f"check_correctness_error: {e}",
                }

        # Phase 3b — fallback: manual correctness check
        from ..task_sources import (
            KERNEL_EVAL_NUM_TRIALS,
            KERNEL_EVAL_SEED,
            KERNEL_EVAL_TOLERANCE,
        )

        init_inputs = get_init_inputs() if get_init_inputs else []

        # Seed before constructing EACH model so parameterised layers
        # produce identical random weights (matches KernelBench convention).
        torch.manual_seed(KERNEL_EVAL_SEED)
        torch.cuda.manual_seed(KERNEL_EVAL_SEED)
        ref_model = model_class(*init_inputs).to(device).eval()

        torch.manual_seed(KERNEL_EVAL_SEED)
        torch.cuda.manual_seed(KERNEL_EVAL_SEED)
        new_model = model_new_class(*init_inputs).to(device).eval()

        max_diff = 0.0

        for trial in range(KERNEL_EVAL_NUM_TRIALS):
            torch.manual_seed(KERNEL_EVAL_SEED + trial)
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
                return {
                    "correct": False,
                    "compiled": True,
                    "error": f"shape_mismatch: {ref_out.shape} vs {new_out.shape}",
                    "max_diff": None,
                }

            diff = torch.max(torch.abs(ref_out.float() - new_out.float())).item()
            max_diff = max(max_diff, diff)

        correct = max_diff <= KERNEL_EVAL_TOLERANCE
        return {
            "correct": correct,
            "compiled": True,
            "error": None if correct else f"max_diff={max_diff:.6f}",
            "max_diff": max_diff,
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "correct": False,
            "compiled": False,
            "error": f"{type(e).__name__}: {e}\n{tb}",
            "max_diff": None,
        }
    finally:
        _cleanup_temp_file(triton_file)
        _cleanup_temp_file(test_file)


def _eval_worker_process(
    conn: Any,
    test_code: str,
    triton_code: str,
    gpu_id: int | None,
) -> None:
    """Entry point for the spawned subprocess.

    Pins to *gpu_id*, imports torch (initialising CUDA), then delegates to
    :func:`run_kernel_eval` and sends the result dict back via *conn*.
    """
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        import torch  # noqa: F811  (deferred so CUDA_VISIBLE_DEVICES takes effect)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = run_kernel_eval(test_code, triton_code, device)
        conn.send(result)
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        conn.send(
            {
                "correct": False,
                "compiled": False,
                "error": f"{type(e).__name__}: {e}\n{tb}",
                "max_diff": None,
            }
        )


# ---------------------------------------------------------------------------
# Warmup kernel
# ---------------------------------------------------------------------------

WARMUP_TRITON_CODE = """
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
