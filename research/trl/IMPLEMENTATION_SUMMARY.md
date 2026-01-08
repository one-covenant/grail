# Parallel Training Implementation Summary

## ✅ Implementation Complete

Created a robust parallel training system that runs **4 simultaneous training instances** with different seeds on **8 GPUs**.

## Files Created/Modified

### Modified Files
1. **`train_trl_grpo.py`** - Added CLI arguments and seed handling
   - `--seed`: Random seed (default: 42)
   - `--vllm-port`: VLLM server port (default: 8000)
   - `--run-suffix`: Custom run identifier (default: empty)
   - Seed setting: `random`, `numpy`, `torch` all seeded
   - Dynamic output directories based on run_id
   - Dynamic VLLM URL based on port

### New Files
2. **`run_parallel_training.py`** - Main parallel launcher (473 lines)
   - Manages 4 VLLM servers + 4 training processes
   - Automatic GPU assignment (no conflicts)
   - Health checks (waits for VLLM readiness)
   - Graceful shutdown (Ctrl+C handling)
   - Process monitoring and logging

3. **`test_single_instance.sh`** - Single instance test script
   - Quick verification before full run
   - Tests VLLM + training connection
   - Automatic cleanup on exit

4. **`PARALLEL_TRAINING_README.md`** - Comprehensive documentation
5. **`QUICK_START.md`** - Quick reference guide
6. **`IMPLEMENTATION_SUMMARY.md`** - This file

## Configuration Details

### Fixed Seeds (Reproducible Across Experiments)
```python
SEEDS = [42, 1337, 2024, 9999]
```

### GPU Allocation (8 GPUs Total)
```
Instance 0: VLLM GPU 0 → Training GPU 1, Port 8000, Seed 42
Instance 1: VLLM GPU 2 → Training GPU 3, Port 8001, Seed 1337
Instance 2: VLLM GPU 4 → Training GPU 5, Port 8002, Seed 2024
Instance 3: VLLM GPU 6 → Training GPU 7, Port 8003, Seed 9999
```

### Output Organization
```
outputs/trl_{dataset}_instance{0-3}_seed{seed}/     # Model checkpoints
checkpoints/deltas_{dataset}_instance{0-3}_seed{seed}/  # Delta checkpoints
logs/parallel_training/*.log                        # Runtime logs
```

## Verification Tests Passed

✅ **CLI arguments** registered correctly
✅ **Imports** work (torch, numpy, random)
✅ **Seed setting** is reproducible
✅ **Syntax** checks pass
✅ **Configuration** validated (8 GPUs, 4 ports, 4 seeds)

## Safety Features

1. **No GPU Conflicts**
   - Each process gets explicit `CUDA_VISIBLE_DEVICES`
   - VLLM and training use different GPUs
   - All 8 GPUs utilized, no overlap

2. **Connection Safety**
   - VLLM servers start first
   - Health checks before training starts
   - Configurable timeouts (default: 300s)
   - Automatic port assignment

3. **Process Management**
   - Process groups for clean shutdown
   - Signal handlers (SIGINT, SIGTERM)
   - Graceful cleanup on exit
   - Force kill fallback if needed

4. **Error Handling**
   - VLLM startup timeout detection
   - Training failure detection
   - Log file capture for debugging
   - Clear error messages

## Usage

### Quick Test (Recommended First)
```bash
cd /home/ubuntu/grail/research/trl
./test_single_instance.sh gsm8k
```

### Full Parallel Training
```bash
python run_parallel_training.py --dataset gsm8k --eval-every 40
```

### Monitor Progress
```bash
# All training logs
tail -f logs/parallel_training/training_instance*.log

# GPU utilization
watch -n 1 nvidia-smi

# Specific instance
tail -f logs/parallel_training/training_instance0_gpu1_seed42.log
```

### Stop Training
```bash
# Graceful (recommended)
Ctrl+C

# If stuck
pkill -f "vllm.entrypoints"
pkill -f "train_trl_grpo"
```

## Expected Behavior

### Startup Sequence (Total: ~2-3 minutes)
1. **Phase 1**: Start 4 VLLM servers (staggered by 2s)
2. **Phase 2**: Wait for all servers ready (~45-60s each)
3. **Phase 3**: Start 4 training processes (staggered by 5s)
4. **Phase 4**: Monitor until completion

### Runtime Logs
```
📡 Phase 1: Starting VLLM servers...
[Instance 0] Starting VLLM server: GPU: 0, Port: 8000
[Instance 1] Starting VLLM server: GPU: 2, Port: 8001
...

⏳ Phase 2: Waiting for VLLM servers to be ready...
[Instance 0] ✓ VLLM server ready! (47.3s)
[Instance 1] ✓ VLLM server ready! (49.1s)
...

🏋️  Phase 3: Starting training processes...
[Instance 0] Starting training: GPU: 1, Seed: 42, VLLM Port: 8000
...

📊 MONITORING ACTIVE PROCESSES
[12:34:56] Training processes: 4/4 running
```

### Completion
```
[12:34:56] Training processes: 0/4 running
✅ All training processes completed successfully!

🧹 Shutting down all processes...
  Stopping VLLM instance 0...
  ...
✓ All processes stopped
```

## Testing Checklist

Before production run:

- [ ] Run `./test_single_instance.sh gsm8k` (5-10 min)
- [ ] Check test outputs created
  - [ ] `outputs/trl_gsm8k_test_instance/`
  - [ ] `checkpoints/deltas_gsm8k_test_instance/`
  - [ ] `logs/test_instance/`
- [ ] Verify GPU availability: `nvidia-smi` shows 8 GPUs
- [ ] Check disk space: `df -h` (need ~50GB for checkpoints)
- [ ] Verify WANDB_API_KEY set: `echo $WANDB_API_KEY`
- [ ] If test passes, run full parallel training

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Reduce `gpu-memory-utilization` or `batch_size` |
| "VLLM server timeout" | Check logs, increase timeout, or check model download |
| "Connection refused" | VLLM not ready - wait longer or check server logs |
| Processes won't stop | `pkill -9 -f vllm && pkill -9 -f train_trl_grpo` |
| Port already in use | `lsof -ti :8000 \| xargs kill` |
| Can't find logs | Check `logs/parallel_training/` directory |

## Performance Expectations

- **VLLM startup**: 45-60s per instance
- **Training (100 steps)**: 2-5 hours (dataset dependent)
- **Total runtime**: ~2-5 hours (all parallel)
- **Disk usage**: ~10-15GB per instance (~50GB total)
- **GPU utilization**: Should be >90% on all 8 GPUs

## Next Steps

1. **Test single instance**: `./test_single_instance.sh gsm8k`
2. **Run parallel training**: `python run_parallel_training.py --dataset gsm8k`
3. **Monitor progress**: `tail -f logs/parallel_training/training_instance*.log`
4. **Check results**: WandB dashboard + `outputs/` directory
5. **Analyze deltas**: Use delta checkpoint tools (see `DELTA_CHECKPOINT_README.md`)

## Seed Reproducibility Guarantee

Seeds are set in **multiple places** for full reproducibility:

```python
# In train_trl_grpo.py main():
random.seed(args.seed)           # Python random
np.random.seed(args.seed)        # NumPy random
torch.manual_seed(args.seed)     # PyTorch CPU
torch.cuda.manual_seed_all(args.seed)  # PyTorch all GPUs
```

**To reuse these seeds in other experiments:**
```python
SEEDS = [42, 1337, 2024, 9999]  # From run_parallel_training.py
```

This ensures comparable results across different methods.

## WandB Integration

Each instance creates a separate run:
- **Run names**: `trl_gsm8k_grpo_qwen15b_instance{N}_seed{SEED}`
- **Output dirs**: `outputs/trl_gsm8k_instance{N}_seed{SEED}/`
- **Delta dirs**: `checkpoints/deltas_gsm8k_instance{N}_seed{SEED}/`

All runs tagged with seed for easy filtering and comparison.

## Implementation Notes

### Why This Design?

1. **Separate GPUs for VLLM + Training**: VLLM is memory-intensive, separating prevents OOM
2. **Fixed Seeds**: Ensures reproducibility across experiments
3. **Unique Ports**: Prevents connection conflicts
4. **Staggered Startup**: Reduces initialization spikes
5. **Process Groups**: Enables clean shutdown of all child processes
6. **Health Checks**: Prevents training from starting before VLLM ready

### Key Design Decisions

- ✅ VLLM starts first (training depends on it)
- ✅ Health checks with retries (network can be flaky)
- ✅ Separate log files per instance (easy debugging)
- ✅ Graceful shutdown with signal handlers
- ✅ Force kill fallback (if graceful fails)
- ✅ Run IDs include seed (track provenance)

## Credits

Implementation based on:
- TRL GRPOTrainer API
- GRAIL training infrastructure
- Delta checkpoint system (separate implementation)
