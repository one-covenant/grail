# Complete Setup Guide for Lium Infrastructure

This guide walks you through the complete setup process for running distributed training experiments on Lium infrastructure.

## GPU Allocation Strategy

**Important**: This infrastructure uses an efficient 1+1 GPU allocation:
- **1 GPU for VLLM generation** (tensor-parallel-size=1)
- **1 GPU for training** (single device)
- **4 parallel experiments per 8-GPU pod**

GPU Assignment Pattern:
```
Experiment 0: VLLM on GPU 0, Training on GPU 1, Port 8000
Experiment 1: VLLM on GPU 2, Training on GPU 3, Port 8001
Experiment 2: VLLM on GPU 4, Training on GPU 5, Port 8002
Experiment 3: VLLM on GPU 6, Training on GPU 7, Port 8003
```

This is 2x more efficient than using 4 GPUs for VLLM + 1 for training.

## Complete Setup Sequence on Pods

When you deploy a pod, the following steps happen automatically:

### Step 1: Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Step 2: Get Code

**Option A: Git Clone (Recommended)**
```bash
git clone -b main https://github.com/manifold-inc/grail.git ~/grail
cd ~/grail
```

**Option B: Rsync (For local changes)**
```bash
# Run from local machine
rsync -avz --exclude='.git' --exclude='.venv' \
  /path/to/local/grail/ \
  user@pod:~/grail/
```

### Step 3: Install Dependencies in Main Directory

```bash
cd ~/grail
uv sync --all-extras
```

This creates `.venv/` and installs all main dependencies.

### Step 4: Install VLLM Server Dependencies

```bash
cd ~/grail/tools/vllm-server
uv sync
```

This creates `tools/vllm-server/.venv/` with VLLM dependencies.

### Step 5: Copy .env File

```bash
# The deploy script automatically copies your local .env file
# Or create one manually:
cat > ~/grail/.env << 'EOF'
WANDB_API_KEY=your-key-here
WANDB_PROJECT=grail-experiments
EOF
```

### Step 6: Run Experiments

Each experiment script does:
```bash
#!/bin/bash
cd ~/grail
export PATH="$HOME/.cargo/bin:$PATH"

# Activate main environment
source .venv/bin/activate

# Start VLLM server on GPU 0, port 8000
cd tools/vllm-server
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 1 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  > vllm_server_exp0.log 2>&1 &

# Wait for VLLM to start
sleep 60

# Start training on GPU 1
cd ~/grail
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python research/trl/train_trl_grpo.py \
  --dataset gsm8k \
  --eval-every 20
```

## Local Machine Setup

### Prerequisites

1. **Python 3.11+**
2. **Lium SDK and AsyncSSH**
   ```bash
   uv pip install lium-sdk asyncssh
   ```
3. **Lium API Key**
   ```bash
   export LIUM_API_KEY="your-api-key"
   ```
4. **WandB (Optional)**
   ```bash
   export WANDB_API_KEY="your-wandb-key"
   export WANDB_PROJECT="grail-experiments"
   ```

### Directory Structure

```
research/infrastructure/
├── lium_manager.py        # Infrastructure management
├── experiment_runner.py   # Experiment orchestration
├── experiment_configs.py  # Predefined configs
├── deploy.py             # Main CLI
└── .env                  # Your secrets (gitignored)
```

## Usage Examples

### Example 1: Git Clone (Recommended)

```bash
cd research/infrastructure

# Deploy 1 pod and run 4 parallel experiments
python deploy.py --config lr_sweep

# What happens:
# 1. Deploy pod with 8xA100 GPUs
# 2. Git clone repo on pod
# 3. Install uv
# 4. Run uv sync in ~/grail
# 5. Run uv sync in ~/grail/tools/vllm-server
# 6. Copy .env file (if exists)
# 7. Start 4 experiments in parallel (GPU pairs: 0+1, 2+3, 4+5, 6+7)
```

### Example 2: With Custom Branch

```bash
python deploy.py \
  --config lr_sweep \
  --git-branch feat/my-experiment
```

### Example 3: Using Rsync (Local Changes)

```bash
python deploy.py \
  --config lr_sweep \
  --use-rsync \
  --local-code-path /path/to/local/grail
```

### Example 4: With Custom .env File

```bash
python deploy.py \
  --config lr_sweep \
  --env-file /path/to/custom/.env
```

### Example 5: Sequential Execution (Debugging)

```bash
python deploy.py \
  --config lr_sweep \
  --sequential  # Run experiments one at a time
```

## Monitoring Experiments

### SSH into Pod

```bash
# Get pod info
lium ps

# SSH into pod
lium ssh <pod-name>
```

### Check Logs

```bash
# Training logs
tail -f ~/grail/train_*.log

# VLLM server logs
tail -f ~/grail/tools/vllm-server/vllm_server_*.log

# GPU usage
watch -n 1 nvidia-smi
```

### Check WandB

Navigate to your WandB project to see real-time metrics:
```
https://wandb.ai/<your-username>/<project-name>
```

## Debugging Common Issues

### Issue: "uv: command not found"

**Cause**: UV not in PATH after installation

**Fix**: The script automatically exports PATH, but if running manually:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: "No module named 'grail'"

**Cause**: Not in correct venv or dependencies not installed

**Fix**:
```bash
cd ~/grail
source .venv/bin/activate  # Activate correct venv
uv sync --all-extras       # Reinstall if needed
```

### Issue: "VLLM server not responding"

**Cause**: VLLM crashed during startup

**Fix**:
```bash
# Check VLLM logs
tail -50 ~/grail/tools/vllm-server/vllm_server_*.log

# Common issues:
# - Out of memory: reduce --gpu-memory-utilization
# - Model not found: check MODEL_ID
# - Port conflict: check if port already in use
```

### Issue: "Training GPU out of memory"

**Cause**: Batch size too large for single GPU

**Fix**: Reduce batch size or increase gradient accumulation:
```python
ExperimentConfig(
    batch_size=2,          # Reduced from 4
    grad_accum_steps=256,  # Increased from 128
    # Effective batch = 2 * 256 = 512 (same)
)
```

### Issue: ".env file not found"

**Cause**: .env not copied or doesn't exist locally

**Fix**:
```bash
# Create .env locally first
cat > /path/to/grail/.env << 'EOF'
WANDB_API_KEY=your-key
WANDB_PROJECT=grail-experiments
EOF

# Then deploy with --env-file
python deploy.py --config lr_sweep --env-file /path/to/grail/.env
```

## Cost Optimization

### 1. Use Git Clone (Faster & Cheaper)

Git clone is faster than rsync for initial setup:
- Git clone: ~30 seconds
- Rsync: ~2-5 minutes (depends on bandwidth)

### 2. Maximize Parallel Experiments

Running 4 experiments in parallel on one 8-GPU pod is cheaper than 4 separate 2-GPU pods:
- **1x 8-GPU pod**: ~$2/hour
- **4x 2-GPU pods**: ~$3-4/hour

### 3. Set TTL

Always set `ttl_hours` to auto-terminate:
```python
PodSpec(
    name="my-pod",
    ttl_hours=6,  # Auto-terminate after 6 hours
)
```

### 4. Use --deploy-only First

Deploy pods and inspect before running experiments:
```bash
# Deploy pods only
python deploy.py --config lr_sweep --deploy-only

# SSH in and test manually
lium ssh lr-sweep

# Then run experiments
python deploy.py --config lr_sweep --no-deploy
```

## Advanced Configuration

### Custom Experiment with 4 Parallel Runs

```python
# In experiment_configs.py
def my_custom_sweep():
    pods = [PodSpec(name="my-sweep", gpu_count=8)]

    experiments = [
        ExperimentConfig(
            name=f"exp_{i}",
            dataset="gsm8k",
            learning_rate=lr,
            # ... other params ...
        )
        for i, lr in enumerate([1e-6, 3e-6, 5e-6, 1e-5])
    ]

    # Assign GPUs automatically
    experiments = assign_gpus_for_parallel(experiments)
    # Results in:
    # exp_0: VLLM GPU 0, Train GPU 1, Port 8000
    # exp_1: VLLM GPU 2, Train GPU 3, Port 8001
    # exp_2: VLLM GPU 4, Train GPU 5, Port 8002
    # exp_3: VLLM GPU 6, Train GPU 7, Port 8003

    return pods, {"my-sweep": experiments}
```

## Summary: Complete Automated Flow

When you run `python deploy.py --config lr_sweep`, here's what happens:

```
Local Machine                           Remote Pod
─────────────                          ──────────────
1. Load config (lr_sweep)
2. Deploy pod via Lium API   ──────►   3. Pod boots (~2 min)
                                       4. SSH ready
                             ◄──────
5. SSH connect               ──────►   6. Remove old ~/grail (if exists)
6. Git clone command         ──────►   7. git clone repo
7. Install uv command        ──────►   8. curl install uv
8. Main uv sync command      ──────►   9. cd ~/grail && uv sync
9. VLLM uv sync command      ──────►   10. cd tools/vllm-server && uv sync
10. Copy .env file           ──────►   11. Write .env to ~/grail/.env
11. Start 4 experiments      ──────►   12. Run experiments in parallel:
                                           - Exp 0: GPU 0+1, Port 8000
                                           - Exp 1: GPU 2+3, Port 8001
                                           - Exp 2: GPU 4+5, Port 8002
                                           - Exp 3: GPU 6+7, Port 8003
12. Monitor via WandB
13. SSH in to check logs (optional)
14. Wait for completion
15. Download results (if needed)
16. Destroy pod               ──────►   17. Pod terminated
```

Total setup time: ~5-10 minutes
Then: Experiments run for configured duration (e.g., 100 steps ~30-60 min each)

## Next Steps

1. **Test locally first**: Run `research/trl/train_trl_grpo.py` on your machine
2. **Start small**: Use `total_steps=5` for quick validation
3. **Check one pod**: Deploy single pod with `--deploy-only`, SSH in, test manually
4. **Scale up**: Once validated, run full sweeps

---

For more details, see:
- **README.md**: Full documentation
- **QUICKSTART.md**: 5-minute quick start
- **VALIDATION.md**: Test results
