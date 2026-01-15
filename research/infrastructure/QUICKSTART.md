# Quick Start Guide

Get up and running with Lium infrastructure in 5 minutes.

## Prerequisites

- Python 3.11+
- SSH key for authentication
- Lium account and API key
- WandB account (optional, for tracking)

## Setup

### 1. Install Dependencies

```bash
cd research/infrastructure
pip install -r requirements.txt
```

Or using uv (recommended):

```bash
uv pip install -r requirements.txt
```

### 2. Configure Lium API

```bash
export LIUM_API_KEY="your-api-key-here"
```

Or create a Lium config file:

```bash
lium config set api_key "your-api-key-here"
```

### 3. Configure WandB (Optional)

```bash
export WANDB_API_KEY="your-wandb-key"
export WANDB_PROJECT="grail-experiments"
```

## Your First Experiment

### Option 1: Simple Example (Recommended for First-Time Users)

Run the simple example to deploy a single pod and run a quick 5-step experiment:

```bash
python example_simple.py
```

This will:
1. Deploy 1 pod with 8xA100 GPUs
2. Sync your code
3. Setup the environment
4. Run a quick 5-step training test
5. Prompt you to destroy the pod when done

**Expected runtime**: ~15 minutes
**Expected cost**: ~$0.50

### Option 2: Predefined Configuration

Run a predefined hyperparameter sweep:

```bash
# List available configurations
python deploy.py --list-configs

# Run learning rate sweep (4 experiments on 2 pods)
python deploy.py --config lr_sweep
```

**Expected runtime**: ~6 hours
**Expected cost**: ~$150-200

## Inspect Before Deploy

Want to see what's available without deploying?

```bash
# Check available executors and bandwidth
python deploy.py --inspect-executors --gpu-type A100

# Deploy pods only (don't run experiments yet)
python deploy.py --config lr_sweep --deploy-only

# Later, run experiments on existing pods
python deploy.py --config lr_sweep --no-deploy
```

## Monitoring

### Check Pod Status

```bash
# Via Lium CLI
lium ps

# Via Python
python -c "from lium_manager import LiumInfra; infra = LiumInfra(); print(infra.list_pods())"
```

### Watch Training Logs

```bash
# SSH into a pod
lium ssh <pod-name>

# Check VLLM server logs
tail -f tools/vllm-server/vllm_server_*.log

# Check training logs
tail -f train_*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### WandB Dashboard

Navigate to your WandB project to see real-time metrics:

```
https://wandb.ai/<your-username>/<project-name>
```

## Cleanup

**Important**: Always cleanup when done to avoid unnecessary costs!

```bash
# Destroy all managed pods
python deploy.py --destroy

# Verify pods are gone
lium ps
```

## What's Next?

- **Create custom configurations**: See `experiment_configs.py` examples
- **Run longer experiments**: Increase `total_steps` in configs
- **Try different datasets**: Change `dataset` to "math" or "mbpp"
- **Scale up**: Add more pods for parallel experiments
- **Read full docs**: See `README.md` for advanced usage

## Common Issues

### "No executors found"

**Solution**: Relax bandwidth requirements or try different GPU type

```python
PodSpec(
    min_upload_mbps=100,  # Reduced from 500
    min_download_mbps=100,
)
```

### "SSH connection failed"

**Solution**: Wait longer (pods can take 2-5 minutes to boot)

```bash
# Check pod status
lium ps

# Wait for status to be "running"
```

### "Out of memory"

**Solution**: Reduce batch size or use fewer GPUs for VLLM

```python
ExperimentConfig(
    batch_size=2,  # Reduced from 4
    vllm_tensor_parallel=2,  # Reduced from 4
)
```

## Cost Tips

1. **Always set TTL**: `ttl_hours=2` in PodSpec
2. **Start small**: Test with `total_steps=5` first
3. **Use deploy-only**: Inspect costs before running experiments
4. **Cleanup promptly**: Run `--destroy` when done

## Getting Help

- Full documentation: `README.md`
- Example configs: `experiment_configs.py`
- Lium docs: https://docs.lium.ai
- File issues: https://github.com/your-org/grail/issues

---

**Happy experimenting! 🚀**
