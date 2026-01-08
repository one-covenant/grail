# Running Parallel Training in Nohup Mode

## Quick Start (MATH Dataset)

```bash
cd /home/ubuntu/grail/research/trl

# Start training in background (runs on MATH dataset by default)
./run_parallel_training_nohup.sh

# Check status
./check_status.sh

# Monitor logs
tail -f logs/parallel_training/launcher_*.log
tail -f logs/parallel_training/training_instance*.log

# Stop training
./stop_parallel_training.sh
```

**You can now safely log out** - training will continue in the background.

---

## Detailed Usage

### Start Training

**MATH dataset (default):**
```bash
./run_parallel_training_nohup.sh
```

**GSM8K dataset:**
```bash
./run_parallel_training_nohup.sh gsm8k
```

**MBPP dataset:**
```bash
./run_parallel_training_nohup.sh mbpp
```

**Custom evaluation frequency:**
```bash
./run_parallel_training_nohup.sh math 50  # Eval every 50 steps
```

### Monitor Progress

**Check status:**
```bash
./check_status.sh
```

**Output:**
```
==================================================
PARALLEL TRAINING STATUS
==================================================
✓ Launcher running (PID: 12345)

VLLM servers: 4 running
  PIDs: 12346 12347 12348 12349

Training processes: 4 running
  PIDs: 12350 12351 12352 12353

Port status:
  Port 8000: OPEN (PID: 12346)
  Port 8001: OPEN (PID: 12347)
  Port 8002: OPEN (PID: 12348)
  Port 8003: OPEN (PID: 12349)
...
```

**Watch logs in real-time:**
```bash
# Launcher log
tail -f logs/parallel_training/launcher_*.log

# All training logs
tail -f logs/parallel_training/training_instance*.log

# Specific instance
tail -f logs/parallel_training/training_instance0_gpu1_seed42.log

# VLLM logs (if debugging)
tail -f logs/parallel_training/vllm_instance0_gpu0_port8000.log
```

**GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

### Stop Training

**Graceful shutdown:**
```bash
./stop_parallel_training.sh
```

This will:
1. Send SIGTERM to launcher (triggers graceful shutdown)
2. Wait up to 30s for processes to stop
3. Force kill if needed
4. Clean up any stray VLLM/training processes

**Output:**
```
==================================================
STOPPING PARALLEL TRAINING
==================================================
Found launcher process (PID: 12345)
Sending SIGTERM for graceful shutdown...
Waiting for shutdown (max 30s)...
......
✓ Launcher stopped gracefully

Checking for stray VLLM/training processes...
✓ All processes stopped
```

---

## Workflow

### Initial Setup

1. **Test single instance first:**
   ```bash
   ./test_single_instance.sh math
   ```

2. **If test passes, start full training:**
   ```bash
   ./run_parallel_training_nohup.sh math
   ```

3. **Verify everything started:**
   ```bash
   ./check_status.sh
   ```

4. **Safe to log out:**
   ```bash
   exit
   ```

### Check Progress (After Re-login)

1. **SSH back in:**
   ```bash
   ssh user@server
   cd /home/ubuntu/grail/research/trl
   ```

2. **Check status:**
   ```bash
   ./check_status.sh
   ```

3. **View recent logs:**
   ```bash
   tail -100 logs/parallel_training/launcher_*.log
   ```

4. **Check GPU usage:**
   ```bash
   nvidia-smi
   ```

### Stop Training (If Needed)

1. **Graceful stop:**
   ```bash
   ./stop_parallel_training.sh
   ```

2. **Verify stopped:**
   ```bash
   ./check_status.sh
   pgrep -f vllm    # Should be empty
   pgrep -f train_trl_grpo  # Should be empty
   ```

---

## Files and Locations

### Scripts
- `./run_parallel_training_nohup.sh` - Start training in nohup mode
- `./check_status.sh` - Check training status
- `./stop_parallel_training.sh` - Stop all training processes
- `./test_single_instance.sh` - Test single instance before full run

### Logs
```
logs/parallel_training/
├── launcher_20240115_123456.log        # Launcher output (timestamped)
├── launcher.pid                        # Launcher PID file
├── vllm_instance0_gpu0_port8000.log   # VLLM server logs
├── vllm_instance1_gpu2_port8001.log
├── vllm_instance2_gpu4_port8002.log
├── vllm_instance3_gpu6_port8003.log
├── training_instance0_gpu1_seed42.log  # Training logs
├── training_instance1_gpu3_seed1337.log
├── training_instance2_gpu5_seed2024.log
└── training_instance3_gpu7_seed9999.log
```

### Outputs
```
outputs/
├── trl_math_instance0_seed42/
├── trl_math_instance1_seed1337/
├── trl_math_instance2_seed2024/
└── trl_math_instance3_seed9999/

checkpoints/
├── deltas_math_instance0_seed42/
├── deltas_math_instance1_seed1337/
├── deltas_math_instance2_seed2024/
└── deltas_math_instance3_seed9999/
```

---

## Troubleshooting

### "Launcher already running"

Another instance is running:
```bash
# Check what's running
./check_status.sh

# Stop it
./stop_parallel_training.sh

# Then start again
./run_parallel_training_nohup.sh math
```

### Processes don't start

Check launcher log:
```bash
tail -100 logs/parallel_training/launcher_*.log
```

Common issues:
- GPU memory full: Check `nvidia-smi`
- Ports in use: Check `./check_status.sh`
- Model download: First run takes 10-20 min to download model

### Training stopped unexpectedly

1. **Check if processes are still running:**
   ```bash
   ./check_status.sh
   ```

2. **Check logs for errors:**
   ```bash
   tail -100 logs/parallel_training/training_instance0_gpu1_seed42.log
   ```

3. **Common causes:**
   - OOM error: Reduce batch size or GPU memory utilization
   - Dataset error: Check dataset name and availability
   - Network error: VLLM connection issues

### Can't find logs

Logs are timestamped. Find the latest:
```bash
ls -lht logs/parallel_training/launcher_*.log | head -1
```

---

## Advanced Usage

### Run only 2 instances (4 GPUs)

Edit `run_parallel_training.py`:
```python
SEEDS = [42, 1337]  # Only 2 seeds
GPU_PAIRS = [
    (0, 1),
    (2, 3),
]
```

### Custom model

```bash
./run_parallel_training_nohup.sh math 40 "Qwen/Qwen2.5-3B-Instruct"
```

### Check completion

```bash
# Check if training finished
./check_status.sh

# If no processes running, check final logs
tail -50 logs/parallel_training/launcher_*.log

# Look for "All training processes completed successfully!"
```

### Resume from checkpoint (manual)

Nohup launcher doesn't auto-resume. To resume:

1. Stop current run: `./stop_parallel_training.sh`
2. Find latest checkpoint: `ls outputs/trl_math_instance0_seed42/checkpoint-*`
3. Manually start individual instances with checkpoint (not automated)

---

## Monitoring Best Practices

### During Initial Startup (First 5 minutes)

```bash
# Watch launcher log
tail -f logs/parallel_training/launcher_*.log

# Look for:
# - "Starting VLLM servers..." ✓
# - "✓ VLLM server ready!" (4 times) ✓
# - "Starting training processes..." ✓
# - "MONITORING ACTIVE PROCESSES" ✓
```

### During Training (Periodic checks)

```bash
# Check every hour
./check_status.sh

# Verify GPU usage
nvidia-smi  # Should show ~90% utilization on all 8 GPUs

# Check training progress
tail -20 logs/parallel_training/training_instance0_gpu1_seed42.log
```

### Before Logging Out

```bash
# Verify everything is running
./check_status.sh

# Should see:
# - Launcher running ✓
# - 4 VLLM servers ✓
# - 4 training processes ✓
# - All ports open ✓

# Safe to exit
exit
```

---

## Expected Timeline (100 steps on MATH)

| Phase | Duration | What to Expect |
|-------|----------|----------------|
| Startup | 2-3 min | VLLM servers start, health checks |
| First step | 5-10 min | Model download (first run only) |
| Subsequent steps | 1-2 min/step | Normal training |
| Total (100 steps) | 2-5 hours | All 4 instances in parallel |

---

## Verification Checklist

Before logging out:

- [ ] Run `./check_status.sh` - shows all processes running
- [ ] Check GPU usage: `nvidia-smi` - shows ~90% utilization
- [ ] Verify logs updating: `tail -f logs/parallel_training/training_instance0*.log`
- [ ] Check WandB dashboard - runs are logging metrics
- [ ] Note launcher PID: `cat logs/parallel_training/launcher.pid`

---

## Quick Reference

| Task | Command |
|------|---------|
| Start training (MATH) | `./run_parallel_training_nohup.sh` |
| Start training (GSM8K) | `./run_parallel_training_nohup.sh gsm8k` |
| Check status | `./check_status.sh` |
| Stop training | `./stop_parallel_training.sh` |
| Watch logs | `tail -f logs/parallel_training/launcher_*.log` |
| GPU status | `nvidia-smi` |
| Find launcher PID | `cat logs/parallel_training/launcher.pid` |
| Kill by PID | `kill $(cat logs/parallel_training/launcher.pid)` |

---

## Safety Features

✅ **Nohup mode** - Training continues after logout
✅ **PID file** - Prevents multiple instances
✅ **Graceful shutdown** - SIGTERM → cleanup → SIGKILL fallback
✅ **Unbuffered output** - Logs written immediately
✅ **Process groups** - Clean shutdown of all child processes
✅ **Health checks** - VLLM readiness before training starts
