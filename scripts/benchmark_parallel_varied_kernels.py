"""
Parallel Varied Kernel Benchmark - 1000 tasks with different operations and sizes

Spawns 1000 tasks in parallel with:
- Different operations: addition, subtraction, multiplication
- Different vector sizes: 256K, 512K, 1M, 2M, 4M elements
- Each task runs its kernel 10 times
"""

import random
import time
from dataclasses import dataclass

import modal

# Modal Infrastructure Setup
app = modal.App("parallel_varied_kernel_benchmark")

# CUDA image setup
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install("torch", "triton", "numpy")
)


@dataclass
class TaskResult:
    """Results from a single task"""

    task_id: int
    operation: str
    vector_size: int
    success: bool
    gpu_wait_time: float
    compilation_time: float
    trial_times: list[float]
    total_time: float
    num_trials: int
    is_correct: bool = False
    error: str | None = None


# Triton kernel templates for different operations
TRITON_ADDITION = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def run_kernel(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def verify(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    expected = x + y
    return (output - expected).abs().max().item() < 1e-5
"""

TRITON_SUBTRACTION = """
import torch
import triton
import triton.language as tl

@triton.jit
def sub_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x - y
    tl.store(output_ptr + offsets, output, mask=mask)

def run_kernel(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sub_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def verify(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    expected = x - y
    return (output - expected).abs().max().item() < 1e-5
"""

TRITON_MULTIPLICATION = """
import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output, mask=mask)

def run_kernel(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

def verify(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    expected = x * y
    return (output - expected).abs().max().item() < 1e-5
"""

# Kernel configurations
KERNELS = {
    "addition": TRITON_ADDITION,
    "subtraction": TRITON_SUBTRACTION,
    "multiplication": TRITON_MULTIPLICATION,
}

# Vector size options (in number of elements)
VECTOR_SIZES = {
    "256K": 256 * 1024,
    "512K": 512 * 1024,
    "1M": 1024 * 1024,
    "2M": 2 * 1024 * 1024,
    "4M": 4 * 1024 * 1024,
}


@app.cls(
    image=image,
    gpu="A100",
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
    ),
    timeout=600,  # Increased timeout for sequential processing
    container_idle_timeout=900,  # Keep container alive
)
@modal.concurrent(max_inputs=1)  # Only 1 task at a time - forces sequential execution
class VariedKernelExecutor:
    """Execute different kernels with different sizes on Modal GPU"""

    def _wait_for_gpu(self, timeout: int = 30) -> bool:
        """Wait for GPU to be available with progressive backoff"""
        import torch

        start = time.time()
        backoff = 0.1

        while time.time() - start < timeout:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                try:
                    _ = torch.zeros(1, device="cuda")
                    return True
                except Exception:
                    pass
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 2.0)

        return False

    @modal.method()
    def run_kernel_n_times(
        self,
        task_id: int,
        operation: str,
        kernel_code: str,
        vector_size: int,
        num_trials: int,
    ) -> TaskResult:
        """
        Run a specific kernel N times

        Args:
            task_id: Unique task identifier
            operation: Name of operation (addition, subtraction, multiplication)
            kernel_code: Triton kernel code as string
            vector_size: Size of input vectors
            num_trials: Number of times to execute the kernel

        Returns:
            TaskResult with timing information
        """
        import importlib.util
        import os
        import sys
        import tempfile
        import traceback

        import torch

        total_start = time.time()

        # Wait for GPU
        if not self._wait_for_gpu():
            return TaskResult(
                task_id=task_id,
                operation=operation,
                vector_size=vector_size,
                success=False,
                gpu_wait_time=time.time() - total_start,
                compilation_time=0.0,
                trial_times=[],
                total_time=time.time() - total_start,
                num_trials=0,
                error="GPU not available after 30s",
            )

        gpu_wait_time = time.time() - total_start

        try:
            # Write kernel to temporary file (Triton requires actual file)
            compilation_start = time.time()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(kernel_code)
                tmp_file_path = tmp_file.name

            # Import the kernel module from file
            spec = importlib.util.spec_from_file_location(f"kernel_module_{task_id}", tmp_file_path)
            kernel_module = importlib.util.module_from_spec(spec)
            sys.modules[f"kernel_module_{task_id}"] = kernel_module
            spec.loader.exec_module(kernel_module)

            # Get the kernel function and verifier
            run_kernel = kernel_module.run_kernel
            verify = kernel_module.verify

            # Create test data on GPU
            x = torch.randn(vector_size, device="cuda")
            y = torch.randn(vector_size, device="cuda")

            # First run to compile (JIT compilation)
            _ = run_kernel(x, y)
            torch.cuda.synchronize()

            compilation_time = time.time() - compilation_start

            # Run multiple timed trials
            trial_times = []
            for _trial_idx in range(num_trials):
                torch.cuda.synchronize()
                trial_start = time.time()

                output = run_kernel(x, y)

                torch.cuda.synchronize()
                trial_time = time.time() - trial_start
                trial_times.append(trial_time)

            # Verify correctness
            is_correct = verify(x, y, output)

            total_time = time.time() - total_start

            # Cleanup
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass

            return TaskResult(
                task_id=task_id,
                operation=operation,
                vector_size=vector_size,
                success=True,
                gpu_wait_time=gpu_wait_time,
                compilation_time=compilation_time,
                trial_times=trial_times,
                total_time=total_time,
                num_trials=num_trials,
                is_correct=is_correct,
            )

        except Exception as e:
            return TaskResult(
                task_id=task_id,
                operation=operation,
                vector_size=vector_size,
                success=False,
                gpu_wait_time=gpu_wait_time,
                compilation_time=0.0,
                trial_times=[],
                total_time=time.time() - total_start,
                num_trials=0,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )


@app.local_entrypoint()
def main(
    num_tasks: int = 1000,
    trials_per_task: int = 10,
    gpu: str = "A100",
    seed: int = 42,
):
    """
    Spawn N tasks in parallel with varied kernels and sizes

    Args:
        num_tasks: Number of parallel tasks to spawn (default: 1000)
        trials_per_task: Number of kernel runs per task (default: 10)
        gpu: GPU type to use (default: A100)
        seed: Random seed for reproducibility (default: 42)
    """
    print("=" * 80)
    print("Sequential Varied Kernel Benchmark (Single GPU)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  GPU Type:          {gpu}")
    print("  Execution Mode:    Sequential (1 GPU)")
    print(f"  Total Tasks:       {num_tasks}")
    print(f"  Trials per Task:   {trials_per_task}")
    print(f"  Total Executions:  {num_tasks * trials_per_task}")
    print(f"  Operations:        {', '.join(KERNELS.keys())}")
    print(f"  Vector Sizes:      {', '.join(VECTOR_SIZES.keys())}")
    print()

    # Generate task configurations with varied operations and sizes
    random.seed(seed)
    task_configs = []
    for task_id in range(num_tasks):
        operation = random.choice(list(KERNELS.keys()))
        size_name = random.choice(list(VECTOR_SIZES.keys()))
        vector_size = VECTOR_SIZES[size_name]
        kernel_code = KERNELS[operation]

        task_configs.append(
            {
                "task_id": task_id,
                "operation": operation,
                "kernel_code": kernel_code,
                "vector_size": vector_size,
                "size_name": size_name,
            }
        )

    # Show distribution
    from collections import Counter

    op_counts = Counter(t["operation"] for t in task_configs)
    size_counts = Counter(t["size_name"] for t in task_configs)

    print("Task Distribution:")
    print("  Operations:")
    for op, count in sorted(op_counts.items()):
        print(f"    {op:15s}: {count:4d} tasks ({count / num_tasks * 100:.1f}%)")
    print("  Vector Sizes:")
    for size, count in sorted(size_counts.items(), key=lambda x: VECTOR_SIZES[x[0]]):
        print(f"    {size:6s}: {count:4d} tasks ({count / num_tasks * 100:.1f}%)")
    print()

    # Override GPU type if different from default
    executor_cls = (
        VariedKernelExecutor.with_options(gpu=gpu) if gpu != "A100" else VariedKernelExecutor
    )

    # Spawn all tasks in parallel
    print(f"[1/2] Spawning {num_tasks} tasks in parallel...")
    spawn_start = time.time()

    futures = []
    for config in task_configs:
        future = executor_cls().run_kernel_n_times.spawn(
            task_id=config["task_id"],
            operation=config["operation"],
            kernel_code=config["kernel_code"],
            vector_size=config["vector_size"],
            num_trials=trials_per_task,
        )
        futures.append(future)

    spawn_time = time.time() - spawn_start
    print(f"✓ Spawned {num_tasks} tasks in {spawn_time:.3f}s")

    # Collect results
    print(f"\n[2/2] Collecting results from {num_tasks} tasks...")
    collect_start = time.time()

    results = []
    for i, future in enumerate(futures, 1):
        if i % 100 == 0 or i == num_tasks:
            print(f"  Progress: {i}/{num_tasks}", end="\r")
        results.append(future.get())

    collect_time = time.time() - collect_start
    total_wall_time = time.time() - spawn_start

    print(f"\n✓ Collected all results in {collect_time:.3f}s")

    # Analyze results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\nExecution Summary:")
    print(f"  Total tasks:           {len(results)}")
    print(
        f"  Successful:            {len(successful)} ({len(successful) / len(results) * 100:.1f}%)"
    )
    print(f"  Failed:                {len(failed)}")
    print(f"  Total kernel runs:     {sum(r.num_trials for r in successful)}")

    if successful:
        all_correct = all(r.is_correct for r in successful)
        print(f"  All outputs correct:   {'✓ YES' if all_correct else '✗ NO'}")

    print("\nTiming:")
    print(f"  Spawn time:            {spawn_time:.3f}s")
    print(f"  Collect time:          {collect_time:.3f}s")
    print(f"  Total wall time:       {total_wall_time:.3f}s")

    if successful:
        total_kernel_runs = sum(r.num_trials for r in successful)
        throughput = total_kernel_runs / total_wall_time
        print(f"  Throughput:            {throughput:.2f} kernel runs/second")

    # Per-operation breakdown
    if successful:
        import numpy as np

        print("\n" + "=" * 80)
        print("Per-Operation Performance")
        print("=" * 80)

        for operation in sorted(KERNELS.keys()):
            op_results = [r for r in successful if r.operation == operation]
            if not op_results:
                continue

            op_trial_times = []
            for r in op_results:
                op_trial_times.extend(r.trial_times)

            trial_times_ms = [t * 1000 for t in op_trial_times]

            print(f"\n{operation.upper()} ({len(op_results)} tasks, {len(op_trial_times)} runs):")
            print(f"  Mean:     {np.mean(trial_times_ms):.6f} ms")
            print(f"  Median:   {np.median(trial_times_ms):.6f} ms")
            print(f"  Std Dev:  {np.std(trial_times_ms):.6f} ms")
            print(f"  Min:      {np.min(trial_times_ms):.6f} ms")
            print(f"  Max:      {np.max(trial_times_ms):.6f} ms")

    # Per-size breakdown
    if successful:
        print("\n" + "=" * 80)
        print("Per-Size Performance")
        print("=" * 80)

        for size_name in sorted(VECTOR_SIZES.keys(), key=lambda x: VECTOR_SIZES[x]):
            vector_size = VECTOR_SIZES[size_name]
            size_results = [r for r in successful if r.vector_size == vector_size]
            if not size_results:
                continue

            size_trial_times = []
            for r in size_results:
                size_trial_times.extend(r.trial_times)

            trial_times_ms = [t * 1000 for t in size_trial_times]

            print(
                f"\n{size_name} ({vector_size:,} elements, {len(size_results)} tasks, {len(size_trial_times)} runs):"
            )
            print(f"  Mean:     {np.mean(trial_times_ms):.6f} ms")
            print(f"  Median:   {np.median(trial_times_ms):.6f} ms")
            print(f"  Std Dev:  {np.std(trial_times_ms):.6f} ms")
            print(f"  Min:      {np.min(trial_times_ms):.6f} ms")
            print(f"  Max:      {np.max(trial_times_ms):.6f} ms")

    # Overall kernel statistics
    if successful:
        all_trial_times = []
        for r in successful:
            all_trial_times.extend(r.trial_times)

        trial_times_ms = [t * 1000 for t in all_trial_times]

        print("\n" + "=" * 80)
        print(f"Overall Kernel Statistics ({len(all_trial_times)} total runs)")
        print("=" * 80)
        print(f"  Mean:     {np.mean(trial_times_ms):.6f} ms")
        print(f"  Median:   {np.median(trial_times_ms):.6f} ms")
        print(f"  Std Dev:  {np.std(trial_times_ms):.6f} ms")
        print(f"  Min:      {np.min(trial_times_ms):.6f} ms")
        print(f"  Max:      {np.max(trial_times_ms):.6f} ms")
        print(f"  P95:      {np.percentile(trial_times_ms, 95):.6f} ms")
        print(f"  P99:      {np.percentile(trial_times_ms, 99):.6f} ms")

        # Per-task averages
        gpu_wait_times = [r.gpu_wait_time for r in successful]
        compilation_times = [r.compilation_time for r in successful]
        total_task_times = [r.total_time for r in successful]

        print(f"\nPer-Task Averages ({len(successful)} tasks):")
        print(
            f"  GPU wait time:         {np.mean(gpu_wait_times):.3f}s (max: {np.max(gpu_wait_times):.3f}s)"
        )
        print(
            f"  Compilation time:      {np.mean(compilation_times):.3f}s (max: {np.max(compilation_times):.3f}s)"
        )
        print(
            f"  Total task time:       {np.mean(total_task_times):.3f}s (max: {np.max(total_task_times):.3f}s)"
        )

    # Show errors if any
    if failed:
        print("\n" + "=" * 80)
        print(f"Failed Tasks ({len(failed)}):")
        for r in failed[:5]:
            print(
                f"  Task {r.task_id} ({r.operation}, {r.vector_size:,} elements): {r.error.split(chr(10))[0] if r.error else 'Unknown error'}"
            )
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("Run with:")
    print("  modal run benchmark_parallel_varied_kernels.py")
    print("  modal run benchmark_parallel_varied_kernels.py --num-tasks 1000 --trials-per-task 10")
    print("  modal run benchmark_parallel_varied_kernels.py --num-tasks 500 --trials-per-task 20")
