# Quick Start Guide - Parallel Training

## TL;DR

```bash
cd /home/ubuntu/grail/research/trl

# Test single instance first (recommended)
./test_single_instance.sh gsm8k

# If test passes, run all 4 parallel instances
python run_parallel_training.py --dataset gsm8k --eval-every 40

# Monitor
tail -f logs/parallel_training/training_instance*.log
```

## What This Does

Runs **4 parallel training instances** with different seeds on **8 GPUs**:

| Instance | Seeds | GPUs (VLLM/Training) | Port | Output Dir |
|----------|-------|----------------------|------|------------|
| 0 | 42 | GPU 0 / GPU 1 | 8000 | `outputs/trl_gsm8k_instance0_seed42/` |
| 1 | 1337 | GPU 2 / GPU 3 | 8001 | `outputs/trl_gsm8k_instance1_seed1337/` |
| 2 | 2024 | GPU 4 / GPU 5 | 8002 | `outputs/trl_gsm8k_instance2_seed2024/` |
| 3 | 9999 | GPU 6 / GPU 7 | 8003 | `outputs/trl_gsm8k_instance3_seed9999/` |

**Seeds are fixed** - reuse them for other experiments for reproducibility.

## Stop Everything

```bash
# Graceful shutdown (recommended)
Ctrl+C

# If processes don't stop
pkill -f "vllm.entrypoints"
pkill -f "train_trl_grpo"
```

## Check Progress

```bash
# GPU utilization
nvidia-smi

# Training logs (live)
tail -f logs/parallel_training/training_instance0_gpu1_seed42.log

# VLLM logs (if issues)
tail -f logs/parallel_training/vllm_instance0_gpu0_port8000.log

# All training logs at once
tail -f logs/parallel_training/training_instance*.log
```

## Common Issues

### "CUDA out of memory"
- Reduce `gpu-memory-utilization` in `run_parallel_training.py` (line ~90): `"0.8"` instead of `"0.9"`
- Or reduce batch size in `train_trl_grpo.py` Config: `batch_size: int = 2`

### "VLLM server timeout"
- Check logs: `cat logs/parallel_training/vllm_instance0_gpu0_port8000.log`
- First run? Model download takes time (10-20 min)
- Increase timeout in `run_parallel_training.py` (line ~152): `timeout=600`

### "Connection refused" during training
- VLLM server not ready - check status: `curl http://127.0.0.1:8000/v1/models`
- Test with single instance: `./test_single_instance.sh`

### Processes won't stop
```bash
# Nuclear option
pkill -9 -f vllm
pkill -9 -f train_trl_grpo
```

## Files Created

```
research/trl/
├── outputs/                          # Training checkpoints
│   ├── trl_gsm8k_instance0_seed42/
│   ├── trl_gsm8k_instance1_seed1337/
│   ├── trl_gsm8k_instance2_seed2024/
│   └── trl_gsm8k_instance3_seed9999/
├── checkpoints/                      # Delta checkpoints
│   ├── deltas_gsm8k_instance0_seed42/
│   └── ...
└── logs/parallel_training/           # Runtime logs
    ├── vllm_instance0_gpu0_port8000.log
    ├── training_instance0_gpu1_seed42.log
    └── ...
```

## Expected Runtime

- VLLM startup: ~45-60s per instance
- Training (100 steps): ~2-5 hours depending on dataset
- **Total: ~2-5 hours** (all run in parallel)

## WandB Tracking

Each instance creates a separate WandB run:
- `trl_gsm8k_grpo_qwen15b_instance0_seed42`
- `trl_gsm8k_grpo_qwen15b_instance1_seed1337`
- `trl_gsm8k_grpo_qwen15b_instance2_seed2024`
- `trl_gsm8k_grpo_qwen15b_instance3_seed9999`

## Verify Everything Works

Before a long run:

```bash
# 1. Test single instance (5-10 min test)
./test_single_instance.sh gsm8k

# 2. Check outputs were created
ls -lh outputs/trl_gsm8k_test_instance/
ls -lh checkpoints/deltas_gsm8k_test_instance/

# 3. If test passes, run full parallel training
python run_parallel_training.py --dataset gsm8k
```

## Need Help?

Read the full documentation:
- `PARALLEL_TRAINING_README.md` - Detailed guide
- `DELTA_CHECKPOINT_README.md` - Delta checkpoint system
- `train_trl_grpo_README.md` - Training script details

Or check logs:
```bash
ls -lh logs/parallel_training/
```
