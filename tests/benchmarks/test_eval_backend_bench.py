"""Performance benchmarks for kernel eval backends.

Measures end-to-end eval pipeline latency (parent sends code -> gets result)
for SubprocessBackend and PersistentWorkerPool.  Wall-clock ``time.monotonic()``
is the correct metric here because we're measuring IPC + CUDA init + Triton JIT
+ forward pass from the orchestrator's perspective.

Methodology
-----------
1. **Warmup** — run ``WARMUP_EVALS`` evals (not timed) to fill Triton JIT cache.
2. **Measure** — run ``MEASURED_EVALS`` evals, record wall-clock per eval.
3. **Gate** — test fails if median exceeds a threshold derived from the initial
   baseline (basilica A100, Feb 2025) with a 2x safety margin.  Median is robust
   to occasional GPU spikes.
4. **Compare** — persistent median must be strictly less than subprocess median.
5. **Report** — JSON with raw times written to ``benchmark_results/``.

Run on basilica:
    KERNEL_EVAL_GPU_IDS=7 pytest -m benchmark tests/benchmarks/ -v -s
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.gpu_real_data]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WARMUP_EVALS = int(os.environ.get("BENCH_WARMUP_EVALS", "3"))
MEASURED_EVALS = int(os.environ.get("BENCH_MEASURED_EVALS", "10"))
RESULT_DIR = Path(os.environ.get("BENCH_RESULT_DIR", "benchmark_results"))

# ---------------------------------------------------------------------------
# Regression thresholds (basilica A100 baseline, Feb 2025)
#
# Subprocess baseline median: ~4.3s  -> threshold 8.5s  (2x)
# Persistent baseline median: ~0.09s -> threshold 0.5s  (generous for JIT variance)
# ---------------------------------------------------------------------------

SUBPROCESS_MAX_MEDIAN_S = 8.5
PERSISTENT_MAX_MEDIAN_S = 0.5

# ---------------------------------------------------------------------------
# Premade kernel
# ---------------------------------------------------------------------------

_TEST_CODE = """
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

_TRITON_CODE = """
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


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Structured output for one benchmark run.

    Only ``median_s`` is used for pass/fail gating.  The other stats
    (mean, stdev, p5, p95, min, max) are recorded in the JSON report
    for offline trend analysis but do not affect test outcomes.
    """

    backend: str
    warmup_evals: int
    measured_evals: int
    times_s: list[float] = field(default_factory=list)
    # Primary metric (used for threshold assertion)
    median_s: float = 0.0
    # Informational stats (JSON only, not asserted)
    mean_s: float = 0.0
    stdev_s: float = 0.0
    p5_s: float = 0.0
    p95_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    device_name: str = ""
    timestamp: str = ""

    def compute_stats(self) -> None:
        t = sorted(self.times_s)
        n = len(t)
        self.median_s = statistics.median(t)
        self.mean_s = statistics.mean(t)
        self.stdev_s = statistics.stdev(t) if n >= 2 else 0.0
        self.min_s = t[0]
        self.max_s = t[-1]
        self.p5_s = t[max(0, int(n * 0.05))]
        self.p95_s = t[min(n - 1, int(n * 0.95))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _run_benchmark(backend, backend_name: str) -> BenchmarkResult:
    """Warmup + timed measurement loop."""
    for _ in range(WARMUP_EVALS):
        result = backend.evaluate(_TEST_CODE, _TRITON_CODE)
        assert result.correct is True, f"warmup failed: {result.error}"

    times: list[float] = []
    for i in range(MEASURED_EVALS):
        start = time.monotonic()
        result = backend.evaluate(_TEST_CODE, _TRITON_CODE)
        elapsed = time.monotonic() - start
        assert result.correct is True, f"{backend_name} eval {i} failed: {result.error}"
        times.append(elapsed)

    br = BenchmarkResult(
        backend=backend_name,
        warmup_evals=WARMUP_EVALS,
        measured_evals=MEASURED_EVALS,
        times_s=times,
        device_name=_get_device_name(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    br.compute_stats()
    return br


def _print_result(r: BenchmarkResult) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {r.backend}  ({r.measured_evals} measured, {r.warmup_evals} warmup)")
    print(f"  device: {r.device_name}")
    print(f"{'=' * 64}")
    for i, t in enumerate(r.times_s):
        print(f"    eval {i:>2d}: {t:.3f}s")
    print(f"  {'─' * 40}")
    print(f"    median: {r.median_s:.3f}s")
    print(f"{'=' * 64}")


def _save_results(results: list[BenchmarkResult]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "eval_backend_bench.json"
    data = {
        "results": [asdict(r) for r in results],
        "config": {
            "warmup_evals": WARMUP_EVALS,
            "measured_evals": MEASURED_EVALS,
        },
    }
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nJSON report: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Backend constructors
# ---------------------------------------------------------------------------


def _make_subprocess():
    from grail.environments.gpu_kernel.eval_backends.subprocess_backend import SubprocessBackend

    return SubprocessBackend(gpu_ids=[0], timeout=60.0)


def _make_persistent():
    from grail.environments.gpu_kernel.eval_backends.persistent_backend import PersistentWorkerPool

    return PersistentWorkerPool(gpu_ids=[0], timeout=60.0)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestSubprocessBenchmark:
    def test_median_within_threshold(self) -> None:
        backend = _make_subprocess()
        backend.start()
        try:
            result = _run_benchmark(backend, "SubprocessBackend")
        finally:
            backend.shutdown()

        _print_result(result)
        _save_results([result])

        assert result.median_s < SUBPROCESS_MAX_MEDIAN_S, (
            f"SubprocessBackend regression: median {result.median_s:.3f}s "
            f"> threshold {SUBPROCESS_MAX_MEDIAN_S}s"
        )


class TestPersistentBenchmark:
    def test_median_within_threshold(self) -> None:
        backend = _make_persistent()
        backend.start()
        try:
            result = _run_benchmark(backend, "PersistentWorkerPool")
        finally:
            backend.shutdown()

        _print_result(result)
        _save_results([result])

        assert result.median_s < PERSISTENT_MAX_MEDIAN_S, (
            f"PersistentWorkerPool regression: median {result.median_s:.3f}s "
            f"> threshold {PERSISTENT_MAX_MEDIAN_S}s"
        )


class TestBackendComparison:
    def test_persistent_faster_than_subprocess(self) -> None:
        sub_backend = _make_subprocess()
        sub_backend.start()
        try:
            sub_result = _run_benchmark(sub_backend, "SubprocessBackend")
        finally:
            sub_backend.shutdown()

        pers_backend = _make_persistent()
        pers_backend.start()
        try:
            pers_result = _run_benchmark(pers_backend, "PersistentWorkerPool")
        finally:
            pers_backend.shutdown()

        _print_result(sub_result)
        _print_result(pers_result)

        speedup = (
            sub_result.median_s / pers_result.median_s if pers_result.median_s > 0 else float("inf")
        )
        print(f"\n  speedup (median): {speedup:.1f}x")

        _save_results([sub_result, pers_result])

        assert pers_result.median_s < sub_result.median_s, (
            f"Persistent ({pers_result.median_s:.3f}s) should be faster "
            f"than subprocess ({sub_result.median_s:.3f}s)"
        )
