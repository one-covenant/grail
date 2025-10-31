# Offline GRPO Trainer - Complete Usage Guide

## Overview

A production-ready offline GRPO training pipeline that enables hyperparameter tuning and model development without bittensor dependencies.

## Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU (tested on 8x A100-80GB)
- UV package manager (recommended)

## Installation

### Option 1: UV (Recommended)

```bash
cd scripts/offline_trainer
uv sync
```

### Option 2: Pip

```bash
cd scripts/offline_trainer
pip install -e .
```

## Quick Start

### 1. Start Generation Server

Choose your backend:

**vLLM (Recommended):**
```bash
# In separate terminal - runs on GPU 0
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B \
  --port 30000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5
```

**SGLang:**
```bash
# In separate terminal
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --port 30001
```

### 2. Run Training

```bash
cd scripts/offline_trainer

# With UV
uv run python run_offline_grpo.py \
  model.train_id=Qwen/Qwen2.5-1.5B \
  model.ref_id=Qwen/Qwen2.5-1.5B \
  model.device=cuda:2 \
  generation.backend=vllm_server \
  generation.base_url=http://127.0.0.1:30000 \
  eval.backend=vllm_server \
  eval.base_url=http://127.0.0.1:30000 \
  train.iterations=10

# Or directly
python3 run_offline_grpo.py [same args]
```

### 3. Monitor Results

Hydra creates structured outputs:
```
outputs/
└── 2024-10-30/
    └── 14-30-45/
        ├── .hydra/           # Config snapshot
        ├── metrics/
        │   ├── train_iter_000.json
        │   ├── train_iter_001.json
        │   ├── eval_iter_000.json
        │   └── ...
        └── checkpoints/
            ├── iter_000/
            ├── iter_001/
            └── ...
```

## Configuration

### Model Settings

```yaml
model:
  train_id: "Qwen/Qwen2.5-1.5B"    # HF model ID or local path
  ref_id: "Qwen/Qwen2.5-1.5B"      # Reference model
  device: "cuda:2"                  # Training GPU device
```

### Generation (Server-backed)

```yaml
generation:
  backend: "vllm_server"            # vllm_server|sglang_server
  base_url: "http://127.0.0.1:30000"
  batch_size: 4                     # Concurrent requests
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
  top_k: 50
  repetition_penalty: 1.1
```

### Training Hyperparameters

```yaml
train:
  iterations: 10                    # Number of training iterations
  batch_size: 8                     # Training batch size
  lr: 1.0e-5                        # Learning rate
  grad_clip: 1.0                    # Gradient clipping
  kl_coef: 0.05                     # KL coefficient
  entropy_coef: 0.0                 # Entropy bonus coefficient
  grad_accum_steps: 1               # Gradient accumulation
```

### Data Generation

```yaml
data:
  problems_per_iteration: 4         # Problems per training iteration
  rollouts_per_problem: 4           # Rollouts per problem (K in GRPO)
  train_seed_start: 1000            # Starting seed for training set
  num_train_seeds: 64               # Total training problems
```

### Evaluation

```yaml
eval:
  enabled: true                     # Enable periodic evaluation
  interval: 1                       # Evaluate every N iterations
  backend: "vllm_server"            # Backend for eval
  base_url: "http://127.0.0.1:30000"
  batch_size: 8
  replicates: 4                     # Replicates per task for pass@k
  num_ids: 16                       # Number of eval tasks
  id_seed_start: 5000               # Eval task seeds
```

## Advanced Usage

### Hyperparameter Sweeps

Hydra multirun for grid search:

```bash
uv run python run_offline_grpo.py -m \
  train.lr='[5e-6,1e-5,2e-5]' \
  train.kl_coef='[0.02,0.05,0.1]' \
  train.batch_size='[4,8]'
```

This creates separate output dirs for each combination.

### Using Different Models

```bash
# Larger model
python run_offline_grpo.py \
  model.train_id=Qwen/Qwen2.5-7B \
  model.ref_id=Qwen/Qwen2.5-7B \
  model.device=cuda:2

# Local checkpoint
python run_offline_grpo.py \
  model.train_id=/path/to/checkpoint \
  model.ref_id=Qwen/Qwen2.5-1.5B
```

### Adjusting Generation

```bash
# More conservative sampling
python run_offline_grpo.py \
  generation.temperature=0.5 \
  generation.top_p=0.9 \
  generation.max_new_tokens=512

# Larger batches (if server can handle)
python run_offline_grpo.py \
  generation.batch_size=16 \
  eval.batch_size=16
```

### Training Configuration

```bash
# Aggressive training
python run_offline_grpo.py \
  train.iterations=50 \
  train.lr=5e-5 \
  train.batch_size=16 \
  data.problems_per_iteration=8

# Conservative (stability focus)
python run_offline_grpo.py \
  train.lr=5e-6 \
  train.kl_coef=0.02 \
  train.batch_size=4
```

## Testing

### Quick Smoke Test (No Server)

```bash
python smoke_test.py
```

Expected output:
```
Train metrics: {'loss_total': 0.0039, 'kl_divergence': 0.464, ...}
Eval metrics: {'pass@1': 0.0, 'mean@1': -0.2, ...}
```

### Unit Tests

```bash
python run_tests.py
```

Expected: `7/7 tests passed`

### GPU Integration Tests (With vLLM Server)

```bash
python test_gpu_integration.py
```

Expected: `4/4 passed` including:
- vLLM server rollout generation
- Training epoch with server-backed data
- Evaluator with vLLM server
- End-to-end iteration

## Performance Tuning

### GPU Memory

**Training models on cuda:2:**
- Qwen2.5-0.5B: ~2GB
- Qwen2.5-1.5B: ~6GB
- Qwen2.5-7B: ~28GB

**vLLM server (separate GPU):**
- Adjust `--gpu-memory-utilization` (0.3-0.9)
- Use smaller max-model-len if needed

### Throughput Optimization

**Generation:**
- Increase `generation.batch_size` (4→8→16)
- Use vLLM server for best performance
- Run server on dedicated GPU

**Training:**
- Increase `train.batch_size` if memory allows
- Use `grad_accum_steps` for effective larger batches
- Monitor GPU utilization with `nvidia-smi`

**Evaluation:**
- Increase `eval.batch_size`
- Reduce `eval.replicates` for faster cycles
- Decrease `eval.num_ids` for quick checks

## Debugging

### Server Not Responding

```bash
# Check server is running
curl http://127.0.0.1:30000/v1/models

# Check vLLM server logs
# Look for "Application startup complete"
```

### Import Errors

```bash
# Verify dependencies
uv run python -c "import torch, transformers, accelerate, hydra; print('OK')"

# Check PYTHONPATH includes repo root
export PYTHONPATH=/root/grail:$PYTHONPATH
```

### Device Errors

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify device placement
# All training models should be on same GPU (e.g., cuda:2)
# Server can run on different GPU
```

### Empty Rollouts

- Verify server is returning completions
- Check `generation.max_new_tokens` is reasonable
- Look for server errors in server terminal

## Metrics Interpretation

### Training Metrics (train_iter_XXX.json)

```json
{
  "loss_total": 0.1554,      // Total loss
  "loss_pg": 0.0423,         // Policy gradient loss
  "loss_kl": 0.1131,         // KL divergence penalty
  "loss_entropy": 0.0,       // Entropy bonus (usually 0)
  "kl_divergence": 5.65,     // Actual KL value
  "reward_mean": -0.181,     // Average reward
  "grad_norm": 0.234         // Gradient norm
}
```

### Evaluation Metrics (eval_iter_XXX.json)

```json
{
  "pass@1": 0.125,           // Success rate with 1 attempt
  "mean@1": 0.342,           // Average reward
  "pass@5": 0.375,           // Success rate with 5 attempts
  "mean@5": 0.456            // Average of best-of-5
}
```

## Best Practices

1. **Start Small** - Test with 0.5B model and 5 iterations first
2. **Monitor Metrics** - Check loss curves for stability
3. **Separate GPUs** - Server on one GPU, training on another
4. **Deterministic** - Use fixed seeds for reproducibility
5. **Sweep Carefully** - Test one hyperparameter at a time first
6. **Save Often** - Use `checkpoint.save_interval=1` initially

## Troubleshooting Checklist

- [ ] Server is running and responding to `/v1/models`
- [ ] `base_url` matches server address
- [ ] Training models use same GPU device
- [ ] GPU has enough memory for models + batch
- [ ] Dependencies installed via `uv sync`
- [ ] Repo root is importable
- [ ] Seeds are deterministic (not random.randint)

## Example Workflows

### Quick Iteration Test (5 min)

```bash
python run_offline_grpo.py \
  model.train_id=Qwen/Qwen2.5-0.5B \
  model.ref_id=Qwen/Qwen2.5-0.5B \
  train.iterations=3 \
  data.problems_per_iteration=2
```

### Full Hyperparameter Search (1-2 hours)

```bash
python run_offline_grpo.py -m \
  model.train_id=Qwen/Qwen2.5-1.5B \
  train.lr='[5e-6,1e-5,2e-5]' \
  train.kl_coef='[0.02,0.05,0.1]' \
  train.iterations=20
```

### Production-like Run (overnight)

```bash
python run_offline_grpo.py \
  model.train_id=Qwen/Qwen2.5-7B \
  model.ref_id=Qwen/Qwen2.5-7B \
  train.iterations=100 \
  train.lr=1e-5 \
  data.problems_per_iteration=8 \
  eval.interval=5
```

## Support

See also:
- `README.md` - Architecture and features
- `IMPLEMENTATION_STATUS.md` - Test results and status
- `conf/offline_grpo.yaml` - Default configuration

