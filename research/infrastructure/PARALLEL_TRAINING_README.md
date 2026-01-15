# Parallel Nohup Training on Lium

Automated multi-model GRPO training experiments on Lium cloud infrastructure with automatic R2 upload.

## Overview

This system orchestrates large-scale parallel training experiments across multiple Lium pods:

1. **Deploy** pods on Lium (8xA100 GPU instances)
2. **Execute** `run_parallel_training_nohup.sh` on each pod (4 seeds in parallel)
3. **Monitor** training until completion
4. **Download** artifacts (logs, outputs, checkpoints)
5. **Upload** to Cloudflare R2 for storage

Each pod runs 4 training instances in parallel with different random seeds (42, 1337, 2024, 9999).

## Quick Start

### Test with Single Model (Qwen2.5-0.5B)

```bash
cd /home/ubuntu/grail/research/infrastructure

# Deploy and run test
python deploy_parallel.py --config test_qwen_0.5b

# Monitor progress (in another terminal)
tail -f .lium_state.json
```

### Full Multi-Model Sweep (9 Models)

```bash
# Deploy and run all 9 model configurations
python deploy_parallel.py --config multi_model

# Results will be uploaded to R2 automatically
```

## Model Configurations

The `multi_model` configuration runs 9 experiments:

| Experiment Name | Model | Size | num_iterations | R2 Folder |
|----------------|-------|------|----------------|-----------|
| qwen2.5-0.5b-iter1 | Qwen/Qwen2.5-Instruct | 0.5B | 1 | experiments/qwen2.5-0.5b-iter1/ |
| qwen2.5-1.5b-iter1 | Qwen/Qwen2.5-Instruct | 1.5B | 1 | experiments/qwen2.5-1.5b-iter1/ |
| qwen2.5-7b-iter1 | Qwen/Qwen2.5-7B-Instruct | 7B | 1 | experiments/qwen2.5-7b-iter1/ |
| qwen2.5-7b-iter8 | Qwen/Qwen2.5-7B-Instruct | 7B | 8 | experiments/qwen2.5-7b-iter8/ |
| qwen2.5-7b-iter16 | Qwen/Qwen2.5-7B-Instruct | 7B | 16 | experiments/qwen2.5-7b-iter16/ |
| llama3.2-1b-iter1 | meta-llama/Llama-3.2-1B-Instruct | 1B | 1 | experiments/llama3.2-1b-iter1/ |
| llama3.2-3b-iter1 | meta-llama/Llama-3.2-3B-Instruct | 3B | 1 | experiments/llama3.2-3b-iter1/ |
| gemma3-1b-iter1 | google/gemma-3-1b-it | 1B | 1 | experiments/gemma3-1b-iter1/ |
| gemma3-4b-iter1 | google/gemma-3-4b-it | 4B | 1 | experiments/gemma3-4b-iter1/ |

## Prerequisites

### 1. Lium SDK

```bash
pip install lium.io
export LIUM_API_KEY="your-lium-api-key"
```

### 2. Required Python Packages

The system uses **separate virtual environments**:

- **research/trl/.venv** - For training scripts (TRL with all extras)
- **tools/vllm-server/.venv** - For vLLM server
- **Root/.venv** - For infrastructure scripts

Setup is **automatic** on remote pods, but for local testing:

```bash
# Setup TRL environment (for training)
cd /home/ubuntu/grail/research/trl
uv sync --all-extras

# Setup vLLM environment (for inference)
cd /home/ubuntu/grail/tools/vllm-server
uv sync --all-extras

# Setup infrastructure environment (for deploy scripts)
cd /home/ubuntu/grail/research/infrastructure
uv sync  # Recommended: installs all dependencies from pyproject.toml

# Or manually:
# pip install lium.io asyncssh boto3 tqdm python-dotenv
```

**Note**: The training scripts automatically use `research/trl/.venv`, not the root venv.

### 3. Environment Variables

Ensure `.env` file in project root contains:

```bash
# R2 Storage
R2_BUCKET_ID=91561e574629960f78e985efa5a37e59
R2_ACCOUNT_ID=91561e574629960f78e985efa5a37e59
R2_WRITE_ACCESS_KEY_ID=5961758bc74f3554506f2ba05390a6dd
R2_WRITE_SECRET_ACCESS_KEY=0a1fbf3a324e889d44d2a235eb58de661758aeba08a0c23d8f744dfd9fc3566a

# WandB (optional)
WANDB_API_KEY=your-wandb-key
WANDB_PROJECT=grail

# HuggingFace (for private models)
HF_TOKEN=your-hf-token
```

## Usage

### Basic Commands

```bash
# List available configurations
python deploy_parallel.py --list-configs

# Deploy and run test configuration
python deploy_parallel.py --config test_qwen_0.5b

# Deploy and run full multi-model sweep
python deploy_parallel.py --config multi_model

# Deploy pods only (no training)
python deploy_parallel.py --config multi_model --deploy-only

# Run on existing pods (skip deployment)
python deploy_parallel.py --config multi_model --no-deploy

# Destroy all managed pods
python deploy_parallel.py --destroy
```

### Advanced Options

```bash
# Use different dataset
python deploy_parallel.py --config test_qwen_0.5b --dataset gsm8k

# Change evaluation interval
python deploy_parallel.py --config test_qwen_0.5b --eval-every 50

# Skip code sync (for faster iteration)
python deploy_parallel.py --config test_qwen_0.5b --no-sync --no-setup

# Use custom state file
python deploy_parallel.py --config multi_model --state-file .lium_prod.json
```

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ deploy_parallel.py (Main Orchestrator)                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load experiment config (experiment_configs.py)              │
│ 2. Deploy pods (lium_manager.py)                               │
│ 3. For each pod:                                                │
│    └─> nohup_experiment_runner.py                              │
│        ├─> SSH connect                                          │
│        ├─> Sync code (rsync)                                    │
│        ├─> Setup environment (uv sync)                          │
│        ├─> Run: run_parallel_training_nohup.sh                 │
│        │   └─> run_parallel_training.py                        │
│        │       └─> 4 parallel: train_trl_grpo.py (4 seeds)    │
│        ├─> Monitor completion (check launcher.pid)             │
│        ├─> Download artifacts (logs/, outputs/, checkpoints/)  │
│        └─> Upload to R2 (r2_uploader.py)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Per-Pod Execution

Each 8xA100 pod runs 4 training instances in parallel:

```
GPU Allocation:
- Instance 0: VLLM GPU 0, Training GPU 1, Seed 42,   Port 8000
- Instance 1: VLLM GPU 2, Training GPU 3, Seed 1337, Port 8001
- Instance 2: VLLM GPU 4, Training GPU 5, Seed 2024, Port 8002
- Instance 3: VLLM GPU 6, Training GPU 7, Seed 9999, Port 8003
```

### R2 Storage Structure

```
s3://91561e574629960f78e985efa5a37e59/
  └── experiments/
      ├── qwen2.5-0.5b-iter1/
      │   ├── logs/
      │   │   └── parallel_training/
      │   │       ├── launcher_*.log
      │   │       ├── vllm_instance*.log
      │   │       └── training_instance*.log
      │   ├── outputs/
      │   │   └── trl_math_instance*_seed*/
      │   └── checkpoints/
      │       └── deltas_math_instance*_seed*/
      ├── qwen2.5-7b-iter8/
      └── ...
```

## Monitoring

### Check Pod Status

```bash
# View managed pods
cat .lium_state.json | jq '.pods'

# SSH into specific pod
ssh -p <port> root@<host>

# Monitor training logs on pod
ssh -p <port> root@<host> "tail -f ~/grail/research/trl/logs/parallel_training/launcher_*.log"
```

### Check Training Progress

On the pod:

```bash
cd ~/grail/research/trl

# Check if launcher is running
ps -p $(cat logs/parallel_training/launcher.pid)

# Monitor all logs
tail -f logs/parallel_training/*.log

# Check GPU utilization
nvidia-smi
```

### WandB Monitoring

All training runs are logged to WandB (if configured):

- Project: `grail` (or as set in `.env`)
- Run names: `trl_math_grpo_qwen15b_instance{N}_seed{SEED}`

## Troubleshooting

### Pod Deployment Issues

**Problem**: No executors found

**Solution**:
```bash
# Remove bandwidth requirements (already removed in configs)
# Or try different GPU types
```

**Problem**: SSH connection timeout

**Solution**:
```bash
# Wait longer (pods take 2-5 minutes to boot)
# Check Lium dashboard for pod status
```

### Training Issues

**Problem**: VLLM server fails to start

**Solution**:
```bash
# SSH into pod and check logs
ssh -p <port> root@<host>
cd ~/grail/research/trl
tail -100 logs/parallel_training/vllm_instance*.log

# Common issues:
# - Out of memory: Reduce model size or vLLM memory utilization
# - Port conflict: Check ports 8000-8003 are free
```

**Problem**: Training crashes

**Solution**:
```bash
# Check training logs
tail -100 logs/parallel_training/training_instance*.log

# Common issues:
# - OOM: Reduce batch size in train_trl_grpo.py
# - Model not found: Check HF_TOKEN in .env
```

### R2 Upload Issues

**Problem**: R2 upload fails

**Solution**:
```bash
# Verify R2 credentials
python -c "
from r2_uploader import R2Uploader
import os
uploader = R2Uploader(
    account_id=os.getenv('R2_ACCOUNT_ID'),
    access_key=os.getenv('R2_WRITE_ACCESS_KEY_ID'),
    secret_key=os.getenv('R2_WRITE_SECRET_ACCESS_KEY'),
)
print('✓ R2 connection OK' if uploader.verify_bucket_access(os.getenv('R2_BUCKET_ID')) else '✗ R2 connection failed')
"

# Manual upload if needed
python r2_uploader.py ./downloads/<experiment_name> <experiment_name>
```

## Cost Estimation

Approximate costs for Lium pods (varies by executor):

| Configuration | Pods | GPU-Hours | Est. Cost* |
|--------------|------|-----------|-----------|
| test_qwen_0.5b | 1x8xA100 | ~48 | $75-100 |
| multi_model | 9x8xA100 | ~864 | $1350-1800 |

*Assuming ~$2/hr per 8xA100 pod. Actual prices vary.

**Cost Optimization**:
- Use `ttl_hours` in pod specs (auto-termination)
- Test with `test_qwen_0.5b` before full sweep
- Use `--deploy-only` to verify pod creation first

## File Reference

### Core Files

- **deploy_parallel.py**: Main orchestration script
- **nohup_experiment_runner.py**: Runs experiments on remote pods
- **r2_uploader.py**: Uploads artifacts to R2
- **experiment_configs.py**: Experiment configurations
- **lium_manager.py**: Lium pod management

### Training Scripts (research/trl/)

- **run_parallel_training_nohup.sh**: Wrapper for nohup mode
- **run_parallel_training.py**: Launches 4 parallel training instances
- **train_trl_grpo.py**: Main GRPO training script (updated with --num-iterations)

## Best Practices

### 1. Test Locally First

```bash
# Setup TRL environment first
cd /home/ubuntu/grail/research/trl
uv sync --all-extras

# Activate the local venv (research/trl/.venv, NOT root .venv)
source .venv/bin/activate

# Quick sanity check
CUDA_VISIBLE_DEVICES=0 python train_trl_grpo.py --dataset math --num-iterations 1

# Deactivate when done
deactivate
```

**Important**: Always use `research/trl/.venv`, not the root `.venv`.

### 2. Start with Test Config

```bash
# Always test with single pod before full sweep
python deploy_parallel.py --config test_qwen_0.5b
```

### 3. Monitor Early

```bash
# SSH into first pod and watch logs for first 10 minutes
# Catch issues before committing to full sweep
```

### 4. Use TTL

All pod specs include `ttl_hours` to prevent runaway costs.

### 5. Keep State File Safe

```bash
# Backup state file
cp .lium_state.json .lium_state.backup.json

# State file tracks all managed pods
```

## Support

For issues:

1. Check troubleshooting section above
2. Review Lium SDK docs: https://docs.lium.ai
3. Check WandB runs for training metrics
4. SSH into pods for detailed logs

## License

Part of the GRAIL project. See main repository LICENSE.
