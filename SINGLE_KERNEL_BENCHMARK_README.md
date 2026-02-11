# Single Kernel Benchmark

Runs a single Triton kernel 100 times on Modal GPU to measure detailed performance characteristics.

## What This Does

1. **Spawns 1 task** on Modal GPU
2. **Waits for GPU** attachment
3. **Compiles kernel** (JIT compilation)
4. **Executes kernel 100 times** with precise timing
5. **Reports detailed statistics**

## Usage

```bash
# Install Modal
pip install modal
modal setup

# Run with defaults (100 trials, A10G GPU)
modal run benchmark_single_kernel_modal.py

# Custom number of trials
modal run benchmark_single_kernel_modal.py --num-trials 200

# Different GPU
modal run benchmark_single_kernel_modal.py --num-trials 100 --gpu A100

# Larger problem size
modal run benchmark_single_kernel_modal.py --vector-size 10485760  # 10M elements
```

## What Gets Measured

### Timing Breakdown

```
Task spawn time:     Time to submit task to Modal (~0.5-2s)
Task collect time:   Time to wait for result (~30-60s)
Total wall time:     End-to-end benchmark duration

GPU wait time:       Time waiting for GPU attachment (~0-30s)
Compilation time:    JIT compilation + first run (~1-5s)
Total GPU time:      All GPU operations (wait + compile + execute)
```

### Per-Trial Statistics

For each of the 100 kernel executions:
- **Mean** execution time
- **Median** execution time
- **Standard deviation** (kernel stability)
- **Min/Max** (outliers)
- **P95/P99** (tail latency)

### Throughput

- Total time for all trials
- Kernels executed per second

## Example Output

```
================================================================================
Single Kernel Benchmark - 100 Trials
================================================================================

Configuration:
  GPU Type:        A10G
  Trials:          100
  Vector Size:     1,048,576 elements
  Kernel:          Triton Vector Addition

[1/2] Spawning task on Modal...
✓ Task spawned in 1.234s

[2/2] Waiting for task to complete...
✓ Task completed in 45.678s

================================================================================
BENCHMARK RESULTS
================================================================================

✅ Execution SUCCESSFUL
Correctness: PASSED

Timing Breakdown:
  Task spawn time:       1.234s
  Task collect time:     45.678s
  Total wall time:       46.912s

GPU Execution:
  GPU wait time:         2.345s
  Compilation time:      3.456s (JIT + first run)
  Total GPU time:        45.678s

Kernel Execution Statistics (100 trials):
  Mean:     0.234567 ms
  Median:   0.234123 ms
  Std Dev:  0.012345 ms
  Min:      0.220000 ms
  Max:      0.298765 ms
  P95:      0.256789 ms
  P99:      0.267890 ms

Throughput:
  Total execution time:  23.457s (all 100 trials)
  Kernels per second:    4.26

Execution Timeline:
  [0s - 1.234s]         Task spawn
  [1.234s - 3.579s]      GPU wait
  [3.579s - 7.035s]  Compilation
  [7.035s - 46.912s]  100 kernel executions
  [Total: 46.912s]

First 10 Trial Times (ms):
  Trial   1: 0.234567 ms
  Trial   2: 0.235678 ms
  Trial   3: 0.233456 ms
  Trial   4: 0.234789 ms
  Trial   5: 0.232345 ms
  Trial   6: 0.236789 ms
  Trial   7: 0.234012 ms
  Trial   8: 0.235890 ms
  Trial   9: 0.233678 ms
  Trial  10: 0.234567 ms
  ... (90 more trials)

================================================================================
```

## Code Structure

### Modal Setup (Same as KernelBench)

```python
app = modal.App("single_kernel_benchmark")

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install("torch", "triton", "numpy")
)

@app.cls(
    image=image,
    gpu="A10G",
    retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=1.0),
    timeout=600,
)
class SingleKernelExecutor:
    ...
```

### GPU Wait Pattern (Same as KernelBench)

```python
def _wait_for_gpu(self, timeout: int = 30) -> bool:
    """Wait for GPU with progressive backoff"""
    start = time.time()
    backoff = 0.1

    while time.time() - start < timeout:
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                return True
            except Exception:
                pass
        time.sleep(backoff)
        backoff = min(backoff * 1.5, 2.0)

    return False
```

### Execution Pattern

```python
# Compile once
exec(kernel_code, namespace)
vector_add_triton = namespace["vector_add_triton"]

# JIT compile
_ = vector_add_triton(x, y)
torch.cuda.synchronize()

# Time each trial
trial_times = []
for trial in range(num_trials):
    torch.cuda.synchronize()
    start = time.time()

    output = vector_add_triton(x, y)

    torch.cuda.synchronize()
    trial_times.append(time.time() - start)
```

## Differences from Full Benchmark

| Aspect | Single Kernel | Full Benchmark |
|--------|--------------|----------------|
| **Tasks** | 1 task | 1000 tasks |
| **Batching** | No batching | Batched (100 per batch) |
| **Execution** | 100 runs of same kernel | 1 run of 1000 different kernels |
| **Purpose** | Measure kernel performance | Measure Modal throughput |
| **Cold start** | 1 cold start | ~100 cold starts (amortized) |
| **Results** | Per-trial statistics | Aggregate statistics |

## When to Use This

✅ **Profiling a single kernel's performance**
✅ **Understanding kernel execution variance**
✅ **Measuring JIT compilation overhead**
✅ **Testing GPU warm-up effects**
✅ **Debugging kernel timing issues**

Use the full batched benchmark (`benchmark_modal_batched.py`) to measure Modal's ability to handle many different kernels in parallel.

## Kernel Code

The default kernel is a simple Triton vector addition:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
```

You can modify `TRITON_VECTOR_ADD` in the script to test different kernels.

## Performance Expectations

### A10G GPU (Default)
- **GPU wait**: 2-15s (depends on cold start)
- **Compilation**: 1-3s
- **Per-trial**: 0.2-0.5ms (1M element vector add)
- **Total time**: ~30-60s (for 100 trials)

### A100 GPU
- **GPU wait**: 2-15s
- **Compilation**: 1-3s
- **Per-trial**: 0.1-0.3ms (faster GPU)
- **Total time**: ~20-40s

## Troubleshooting

**GPU not available after 30s**
- Modal may be experiencing high load
- Try again or increase timeout in `_wait_for_gpu(timeout=60)`

**High variance in trial times**
- First few trials may be slower (cache warm-up)
- Look at median instead of mean
- Increase `num_trials` for more stable statistics

**Compilation takes too long**
- Large kernels take longer to JIT compile
- This is normal for first run
- Subsequent trials should be fast
