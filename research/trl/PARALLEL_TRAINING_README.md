# Parallel Training with Multiple Seeds

Run 4 parallel GRPO training instances with different random seeds on 8 GPUs.

## Overview

This setup runs **4 parallel training instances**, each with:
- **Dedicated VLLM server** (for fast generation)
- **Dedicated training GPU**
- **Unique random seed** (42, 1337, 2024, 9999)
- **Unique port** (8000, 8001, 8002, 8003)

### GPU Allocation

| Instance | VLLM GPU | Training GPU | Port | Seed | Run ID |
|----------|----------|--------------|------|------|--------|
| 0        | GPU 0    | GPU 1        | 8000 | 42   | instance0_seed42 |
| 1        | GPU 2    | GPU 3        | 8001 | 1337 | instance1_seed1337 |
| 2        | GPU 4    | GPU 5        | 8002 | 2024 | instance2_seed2024 |
| 3        | GPU 6    | GPU 7        | 8003 | 9999 | instance3_seed9999 |

**Total**: 8 GPUs (perfect for this server)

## Quick Start

### 1. Test Single Instance First (Recommended)

Before running all 4 instances, test with a single instance to verify everything works:

```bash
cd /home/ubuntu/grail/research/trl
./test_single_instance.sh gsm8k
```

This will:
- Start VLLM server on GPU 0, port 8000
- Start training on GPU 1
- Run a short test to verify connectivity
- Clean up on exit (Ctrl+C)

**Expected output:**
```
✓ VLLM server is ready!
Starting training on GPU 1...
🚀 Starting TRL GRPO training with GSM8K dataset
   Seed: 42 | VLLM Port: 8000
...
```

### 2. Run Parallel Training

Once the single instance test passes, run all 4 instances:

```bash
cd /home/ubuntu/grail/research/trl
python run_parallel_training.py --dataset gsm8k --eval-every 40
```

**Options:**
- `--dataset`: gsm8k, math, or mbpp (default: gsm8k)
- `--eval-every`: Evaluation frequency in steps (default: 40)
- `--model`: Model ID (default: Qwen/Qwen2.5-1.5B-Instruct)

**Example with MATH dataset:**
```bash
python run_parallel_training.py --dataset math --eval-every 50
```

### 3. Monitor Progress

The launcher will show status updates:

```
📡 Phase 1: Starting VLLM servers...
[Instance 0] Starting VLLM server: GPU: 0, Port: 8000
[Instance 1] Starting VLLM server: GPU: 2, Port: 8001
...

⏳ Phase 2: Waiting for VLLM servers to be ready...
[Instance 0] ✓ VLLM server ready! (45.2s)
[Instance 1] ✓ VLLM server ready! (47.1s)
...

🏋️  Phase 3: Starting training processes...
[Instance 0] Starting training: GPU: 1, Seed: 42, VLLM Port: 8000
...

📊 MONITORING ACTIVE PROCESSES
[12:34:56] Training processes: 4/4 running
```

**Check logs:**
```bash
# VLLM server logs
tail -f logs/parallel_training/vllm_instance0_gpu0_port8000.log

# Training logs
tail -f logs/parallel_training/training_instance0_gpu1_seed42.log

# All training logs
tail -f logs/parallel_training/training_instance*
```

### 4. Graceful Shutdown

Press **Ctrl+C** to stop all processes gracefully:

```
^C
🛑 Received signal 2, shutting down gracefully...
🧹 Shutting down all processes...
  Stopping training instance 0...
  Stopping training instance 1...
  ...
  Stopping VLLM instance 0...
  ...
✓ All processes stopped
```

## Output Organization

Each instance creates separate outputs:

### Training Outputs
```
outputs/
  trl_gsm8k_instance0_seed42/
  trl_gsm8k_instance1_seed1337/
  trl_gsm8k_instance2_seed2024/
  trl_gsm8k_instance3_seed9999/
```

### Delta Checkpoints
```
checkpoints/
  deltas_gsm8k_instance0_seed42/
  deltas_gsm8k_instance1_seed1337/
  deltas_gsm8k_instance2_seed2024/
  deltas_gsm8k_instance3_seed9999/
```

### Logs
```
logs/parallel_training/
  vllm_instance0_gpu0_port8000.log
  vllm_instance1_gpu2_port8001.log
  vllm_instance2_gpu4_port8002.log
  vllm_instance3_gpu6_port8003.log
  training_instance0_gpu1_seed42.log
  training_instance1_gpu3_seed1337.log
  training_instance2_gpu5_seed2024.log
  training_instance3_gpu7_seed9999.log
```

## Seed Reproducibility

The fixed seeds (42, 1337, 2024, 9999) are:
- ✅ **Consistent across experiments**: Reuse for other methods
- ✅ **Set everywhere**: `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- ✅ **Logged in WandB**: Run names include seed for tracking

To use the same seeds in another experiment:
```python
SEEDS = [42, 1337, 2024, 9999]  # Copy from run_parallel_training.py
```

## Troubleshooting

### GPU Memory Issues

If you see OOM errors:

1. Check GPU utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. Reduce VLLM memory usage in `run_parallel_training.py`:
   ```python
   "--gpu-memory-utilization", "0.8",  # Default: 0.9
   ```

3. Or reduce batch size in `train_trl_grpo.py` Config:
   ```python
   batch_size: int = 2  # Default: 4
   ```

### VLLM Server Not Ready

If VLLM servers timeout:

1. Check VLLM logs:
   ```bash
   cat logs/parallel_training/vllm_instance0_gpu0_port8000.log
   ```

2. Common issues:
   - Model download in progress (wait for HuggingFace cache)
   - Insufficient GPU memory (reduce `--gpu-memory-utilization`)
   - Port already in use (check with `lsof -i :8000`)

3. Increase timeout in `run_parallel_training.py`:
   ```python
   if not self.wait_for_vllm_ready(port, timeout=600, instance_id=i):  # 10 minutes
   ```

### Training Connection Errors

If training can't connect to VLLM:

1. Verify VLLM is accessible:
   ```bash
   curl http://127.0.0.1:8000/v1/models
   ```

2. Check firewall rules (should be fine for localhost)

3. Test with single instance first:
   ```bash
   ./test_single_instance.sh gsm8k
   ```

### Process Cleanup Issues

If processes don't stop cleanly:

```bash
# Find stray VLLM processes
ps aux | grep vllm

# Kill them manually
pkill -f "vllm.entrypoints"

# Or kill by port
lsof -ti :8000 | xargs kill -9
```

### Checking Results

After training completes:

```bash
# Check WandB for metrics
# Each run is tagged with instance ID and seed

# Check final checkpoints
ls -lh outputs/trl_gsm8k_*/checkpoint-*

# Check delta sparsity
cat checkpoints/deltas_gsm8k_instance0_seed42/metadata.json
```

## Advanced Usage

### Custom Seeds

Edit `run_parallel_training.py`:

```python
SEEDS = [100, 200, 300, 400]  # Your custom seeds
```

### Different GPU Allocation

Edit `GPU_PAIRS` in `run_parallel_training.py`:

```python
GPU_PAIRS = [
    (0, 1),  # Instance 0
    (2, 3),  # Instance 1
    # ... customize as needed
]
```

### Run Only 2 Instances (If You Have 4 GPUs)

```python
SEEDS = [42, 1337]  # Only 2 seeds
GPU_PAIRS = [
    (0, 1),
    (2, 3),
]
```

### Different Models per Instance

Modify `start_training()` to pass different `--model` arguments.

## Performance Tips

1. **Stagger startup**: The launcher already staggers VLLM server startup by 2s and training by 5s
2. **Monitor GPU utilization**: Ensure all GPUs are being used effectively
3. **Check bottlenecks**: VLLM generation should be fast (<1s per batch)
4. **Disable eval for speed**: Set `--eval-every 999999` to skip evaluation during training

## Files Reference

- **`run_parallel_training.py`**: Main launcher script
- **`test_single_instance.sh`**: Single instance test script
- **`train_trl_grpo.py`**: Training script (accepts `--seed`, `--vllm-port`, `--run-suffix`)
- **Logs**: `logs/parallel_training/*.log`
- **Outputs**: `outputs/trl_{dataset}_{run_id}/`
- **Deltas**: `checkpoints/deltas_{dataset}_{run_id}/`

## Expected Timeline

For 100 training steps on GSM8K (typical run):

- VLLM startup: ~45-60s per instance
- Training: ~2-5 hours (depends on dataset size and GPU speed)
- Total: ~2-5 hours for all 4 instances (running in parallel)

**Monitor with:**
```bash
# Check all training logs
tail -f logs/parallel_training/training_instance*.log
```

## Stopping and Resuming

The launcher does **not** support resume from checkpoint automatically. To resume:

1. Stop the launcher (Ctrl+C)
2. Find the latest checkpoint: `ls outputs/trl_gsm8k_instance0_seed42/checkpoint-*`
3. Manually restart individual instances with the appropriate checkpoint (not currently implemented)

For now, let runs complete or accept they start from scratch.
