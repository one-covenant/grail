"""GPU benchmarking with CUDA Events, proper warmup, and L2 cache clearing.

Ported from basilica-infra/kernel-bench for use in all local eval backends.
Produces reproducible timing where two independent runs on the same GPU model
produce approximately the same result (within ~3% CV on pinned clocks).

Protocol:
1. Trigger compilation (single call + sync).
2. Warmup for warmup_ms (default 25ms, matching Triton do_bench).
3. Estimate kernel time (5 warm-cache runs, no L2 flush).
4. Compute adaptive run count from 100ms time budget, capped at num_runs.
5. For each timed run: clear L2 -> setup outside events -> record -> fn -> record.
6. Single synchronize after ALL runs (batch sync).
7. Discard first run, apply IQR outlier filtering.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Pre-allocated L2 flush buffers (one per device, reused across all benchmarks)
_l2_buffers: dict[int, torch.Tensor] = {}

_REP_MS = 100.0  # Target measurement time budget (matches Triton do_bench)
_MIN_RUNS = 5  # Minimum timed runs regardless of budget


def _get_l2_buffer(device: torch.device) -> torch.Tensor:
    """Get or create a pre-allocated L2 flush buffer for the given device.

    Auto-detects L2 cache size and uses 2x that (matching Triton do_bench).
    Falls back to 80MB if detection fails.
    """
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    buf = _l2_buffers.get(dev_idx)
    if buf is None:
        props = torch.cuda.get_device_properties(device)
        l2_bytes = props.L2_cache_size
        if l2_bytes > 0:
            target_bytes = 2 * l2_bytes
        else:
            target_bytes = 80 * 1024 * 1024
        n_elements = target_bytes // 4  # int32
        buf = torch.empty(n_elements, dtype=torch.int32, device=device)
        _l2_buffers[dev_idx] = buf
        logger.info(
            "L2 flush buffer: %.1f MB (L2 cache: %.1f MB) on device %d",
            target_bytes / (1024 * 1024),
            l2_bytes / (1024 * 1024) if l2_bytes > 0 else 0,
            dev_idx,
        )
    return buf


def clear_l2_cache(device: torch.device) -> None:
    """Flush GPU L2 cache by zeroing a pre-allocated buffer.

    No sync needed: GPU stream ordering ensures the flush completes before
    any subsequently recorded CUDA events. Matches Triton do_bench approach.
    """
    _get_l2_buffer(device).zero_()


def filter_outliers(times: list[float]) -> tuple[list[float], int]:
    """Remove outliers using IQR method. Returns (filtered_times, num_removed)."""
    if len(times) < 4:
        return times, 0
    arr = np.array(times)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (arr >= lower) & (arr <= upper)
    filtered = arr[mask].tolist()
    return filtered, int((~mask).sum())


@dataclass
class BenchmarkStats:
    """Timing statistics computed from timed runs."""

    raw_times_ms: list[float]
    times_ms: list[float] = field(default_factory=list)
    median_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p20_ms: float = 0.0
    p80_ms: float = 0.0
    cv_percent: float = 0.0
    num_effective_runs: int = 0
    num_outliers_removed: int = 0

    def __post_init__(self) -> None:
        if not self.raw_times_ms:
            return
        self.times_ms, self.num_outliers_removed = filter_outliers(self.raw_times_ms)
        if not self.times_ms:
            self.times_ms = self.raw_times_ms
            self.num_outliers_removed = 0
        self.num_effective_runs = len(self.times_ms)
        arr = np.array(self.times_ms)
        self.median_ms = float(np.median(arr))
        self.mean_ms = float(np.mean(arr))
        self.std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        self.min_ms = float(np.min(arr))
        self.max_ms = float(np.max(arr))
        self.p20_ms = float(np.percentile(arr, 20))
        self.p80_ms = float(np.percentile(arr, 80))
        self.cv_percent = (self.std_ms / self.mean_ms * 100.0) if self.mean_ms > 0 else 0.0


def _call_fn(fn: Callable[..., Any], args: tuple, kwargs_fn: Callable[..., Any] | None) -> None:
    """Call fn with either positional args or kwargs from kwargs_fn."""
    if kwargs_fn is not None:
        fn(**kwargs_fn())
    else:
        fn(*args)


def _trigger_compilation(
    fn: Callable[..., Any],
    args: tuple,
    device: torch.device,
    kwargs_fn: Callable[..., Any] | None = None,
) -> None:
    """Call fn once to trigger Triton JIT compilation and CUDA lazy init."""
    _call_fn(fn, args, kwargs_fn)
    torch.cuda.synchronize(device)


def _warmup(
    fn: Callable[..., Any],
    args: tuple,
    warmup_ms: float,
    device: torch.device,
    kwargs_fn: Callable[..., Any] | None = None,
) -> int:
    """Run fn repeatedly until warmup_ms have elapsed.

    Time-based (not count-based) to handle kernels of varying duration.
    Matches Triton do_bench default of 25ms.
    """
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    count = 0
    while (time.perf_counter() - start) * 1000.0 < warmup_ms:
        _call_fn(fn, args, kwargs_fn)
        count += 1
    torch.cuda.synchronize(device)
    return count


def benchmark_fn(
    fn: Callable[..., Any],
    args: tuple = (),
    kwargs_fn: Callable[..., Any] | None = None,
    num_runs: int = 50,
    warmup_ms: float = 25.0,
    device: torch.device | None = None,
    skip_compilation: bool = False,
) -> BenchmarkStats:
    """Benchmark a GPU function using CUDA Events with adaptive run count.

    Args:
        fn: Function to benchmark.
        args: Positional args for fn (used if kwargs_fn is None).
        kwargs_fn: If provided, called before each run to get kwargs for fn.
            Called OUTSIDE the CUDA event recording window so setup cost
            (e.g. input cloning) is not included in the measured time.
        num_runs: Maximum number of timed runs.
        warmup_ms: Warmup duration in milliseconds.
        device: CUDA device to use. Defaults to current device.
        skip_compilation: If True, skip the compilation trigger call
            (use when fn was already called, e.g. during correctness check).

    Returns:
        BenchmarkStats with timing data after IQR outlier filtering.
    """
    if device is None:
        device = torch.device("cuda")

    if not skip_compilation:
        _trigger_compilation(fn, args, device, kwargs_fn)
    _warmup(fn, args, warmup_ms, device, kwargs_fn)

    # Estimate kernel time (5 warm-cache runs, no L2 flush)
    n_estimate = 5
    est_starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_estimate)]
    est_ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_estimate)]
    for i in range(n_estimate):
        if kwargs_fn is not None:
            kw = kwargs_fn()
            est_starts[i].record()
            fn(**kw)
        else:
            est_starts[i].record()
            fn(*args)
        est_ends[i].record()
    torch.cuda.synchronize(device)
    est_times = [est_starts[i].elapsed_time(est_ends[i]) for i in range(n_estimate)]
    estimate_ms = float(np.median(est_times))

    # Adaptive run count: fill 100ms budget, clamp to [_MIN_RUNS, num_runs]
    if estimate_ms > 0:
        budget_runs = max(1, int(_REP_MS / estimate_ms))
    else:
        budget_runs = num_runs
    actual_runs = max(_MIN_RUNS, min(num_runs, budget_runs))

    # Timed runs: setup (kwargs_fn) happens OUTSIDE the event recording
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(actual_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(actual_runs)]

    for i in range(actual_runs):
        clear_l2_cache(device)
        if kwargs_fn is not None:
            kw = kwargs_fn()
            start_events[i].record()
            fn(**kw)
        else:
            start_events[i].record()
            fn(*args)
        end_events[i].record()

    torch.cuda.synchronize(device)

    all_times = [start_events[i].elapsed_time(end_events[i]) for i in range(actual_runs)]

    # Discard first run (may have residual initialization noise)
    effective_times = all_times[1:]

    return BenchmarkStats(raw_times_ms=effective_times)
