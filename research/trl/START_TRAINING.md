# START TRAINING - Quick Guide

## Run 4 Parallel Instances on MATH Dataset (Nohup Mode)

### Step 1: Start Training
```bash
cd /home/ubuntu/grail/research/trl
./run_parallel_training_nohup.sh
```

**This will:**
- Run on **MATH dataset** (default)
- Use **4 different seeds**: 42, 1337, 2024, 9999
- Use **all 8 GPUs** (2 per instance: 1 for VLLM, 1 for training)
- Run in **background** - you can log out safely

### Step 2: Verify It Started
```bash
./check_status.sh
```

**Expected output:**
```
✓ Launcher running (PID: XXXXX)
VLLM servers: 4 running
Training processes: 4 running
Port status:
  Port 8000: OPEN
  Port 8001: OPEN
  Port 8002: OPEN
  Port 8003: OPEN
```

### Step 3: Monitor (Optional)
```bash
# Watch training logs
tail -f logs/parallel_training/training_instance0_gpu1_seed42.log

# Check GPU usage
nvidia-smi
```

### Step 4: Log Out Safely
```bash
exit
```

**Training will continue in background!**

---

## Check Progress After Re-login

```bash
ssh user@server
cd /home/ubuntu/grail/research/trl

# Check status
./check_status.sh

# View recent logs
tail -50 logs/parallel_training/launcher_*.log
```

---

## Stop Training (If Needed)

```bash
./stop_parallel_training.sh
```

---

## Run on Different Dataset

**GSM8K:**
```bash
./run_parallel_training_nohup.sh gsm8k
```

**MBPP:**
```bash
./run_parallel_training_nohup.sh mbpp
```

---

## Files Locations

### Training Outputs
```
outputs/trl_math_instance{0-3}_seed{42,1337,2024,9999}/
```

### Delta Checkpoints
```
checkpoints/deltas_math_instance{0-3}_seed{42,1337,2024,9999}/
```

### Logs
```
logs/parallel_training/
├── launcher_TIMESTAMP.log           # Main launcher log
├── training_instance0_gpu1_seed42.log
├── training_instance1_gpu3_seed1337.log
├── training_instance2_gpu5_seed2024.log
├── training_instance3_gpu7_seed9999.log
└── vllm_instance*.log               # VLLM server logs
```

---

## Timeline (MATH Dataset, 100 Steps)

| Phase | Duration |
|-------|----------|
| Startup | 2-3 minutes |
| Training | 2-5 hours |
| **Total** | **~2-5 hours** |

---

## Troubleshooting

### Issue: "Launcher already running"
```bash
./stop_parallel_training.sh
./run_parallel_training_nohup.sh
```

### Issue: Check logs for errors
```bash
tail -100 logs/parallel_training/launcher_*.log
tail -100 logs/parallel_training/training_instance0*.log
```

### Issue: CUDA out of memory
Reduce batch size in `train_trl_grpo.py` Config:
```python
batch_size: int = 2  # Default: 4
```

---

## Complete Command Reference

| Action | Command |
|--------|---------|
| Start (MATH) | `./run_parallel_training_nohup.sh` |
| Start (GSM8K) | `./run_parallel_training_nohup.sh gsm8k` |
| Check status | `./check_status.sh` |
| Stop | `./stop_parallel_training.sh` |
| Monitor logs | `tail -f logs/parallel_training/training_instance*.log` |
| GPU usage | `nvidia-smi` |

---

## What You Get

4 parallel training runs with **fixed seeds for reproducibility**:

- **Instance 0**: Seed 42, GPUs 0+1, Port 8000
- **Instance 1**: Seed 1337, GPUs 2+3, Port 8001
- **Instance 2**: Seed 2024, GPUs 4+5, Port 8002
- **Instance 3**: Seed 9999, GPUs 6+7, Port 8003

Each instance:
- ✅ Runs independently
- ✅ Saves to separate directories
- ✅ Logs to WandB with unique run name
- ✅ Creates delta checkpoints
- ✅ Uses different random seed

---

For detailed documentation, see:
- `NOHUP_USAGE.md` - Complete nohup guide
- `PARALLEL_TRAINING_README.md` - Detailed training docs
- `DELTA_CHECKPOINT_README.md` - Delta checkpoint system
