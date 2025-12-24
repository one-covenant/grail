# TRL GRPO Training Script

Unified TRL GRPO training script supporting both GSM8K and MATH (Hendrycks) datasets with exact parity to GRAIL environment implementations.

## Quickstart

### 1. Launch the vLLM Server (Generation GPUs)

The vLLM server handles rollout generation on separate GPUs while the trainer runs on its own GPU.

```bash
# Activate vLLM environment
source tools/vllm-server/.venv/bin/activate

# Launch vLLM server on GPUs 1-4 (4-way tensor parallel)
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  > vllm_server.log 2>&1 &

# Wait for server to be ready (check logs)
tail -f vllm_server.log
```

### 2. Start GRPO Training (Training GPU)

```bash
# Train on GSM8K (default)
CUDA_VISIBLE_DEVICES=0 nohup python research/trl/train_trl_grpo.py \
  --dataset gsm8k \
  > research/trl/train_gsm8k.log 2>&1 &

# Train on MATH (Hendrycks)
CUDA_VISIBLE_DEVICES=0 nohup python research/trl/train_trl_grpo.py \
  --dataset math \
  > research/trl/train_math.log 2>&1 &

# Custom eval frequency
CUDA_VISIBLE_DEVICES=0 python research/trl/train_trl_grpo.py \
  --dataset math \
  --eval-every 50
```

Training logs stream to the respective log files.

## Features

- **Factory Pattern**: Easy switching between datasets via `--dataset` CLI flag
- **GRAIL Parity**: Uses exact same task sources, validation logic, and reward weights
- **Multi-Strategy Validation** (MATH): Exact match → Symbolic (sympy) → Numeric
- **Stratified Splits** (MATH): 7,000 train / 500 val (stratified by subject)
- **vLLM Evaluation**: Async batched evaluation with KMetrics aggregation

## Dataset Comparison

| Aspect | GSM8K | MATH |
|--------|-------|------|
| **Train Size** | 7,473 | 7,000 |
| **Eval Size** | 1,319 (test) | 500 (stratified val) |
| **Gold Format** | `#### answer` | `\boxed{answer}` |
| **Validation** | Numeric exact | Multi-strategy (exact/sympy/numeric) |
| **Correctness Weight** | 0.6 | 0.7 |
| **Success Threshold** | ≥0.6 | ≥0.7 |

## Reward Components

### GSM8K (Total: 1.0)
| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.6 | Exact numeric match |
| Strict format | 0.15 | Numeric-only + no trailing |
| Thinking | 0.1 | Has reasoning block |
| Answer | 0.1 | Has solution tags |
| No trailing | 0.05 | No text after answer |

### MATH (Total: 1.0)
| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.7 | Multi-strategy validation |
| Answer format | 0.15 | Has answer + trailing < 50 chars |
| Thinking | 0.1 | Has reasoning block |
| No trailing | 0.05 | No text after answer |

## Hyperparameters (from .env)

| Parameter | Value | Source |
|-----------|-------|--------|
| Learning rate | 3e-6 | `GRAIL_TRAINER_LR` |
| Epochs | 1 | `GRAIL_TRAINER_EPOCHS` |
| Batch size | 4 | `GRAIL_TRAINER_BATCH_SIZE` |
| Grad accum | 128 | `GRAIL_TRAINER_GRAD_ACCUM_STEPS` |
| Max length | 2048 | `GRAIL_TRAINER_MAX_LENGTH` |
| Max completion | 1024 | `GRPO_MAX_COMPLETION_TOKENS` |
| Loss type | dapo | `GRAIL_GRPO_VARIANT` |

## Architecture

```
train_trl_grpo.py
├── DatasetAdapter (ABC)
│   ├── GSM8KAdapter      # Uses GSM8KTaskSource from GRAIL
│   └── MATHAdapter       # Uses MATHTaskSource from GRAIL
├── get_dataset_adapter() # Factory function
├── VLLMEvalCallback      # Dataset-agnostic evaluation
└── main()                # CLI entry point
```

## GPU Layout (Example: 8x A100)

```
┌─────────────────────────────────────────────────────────────┐
│  GPU 0: Training (GRPO backward pass)                       │
├─────────────────────────────────────────────────────────────┤
│  GPUs 1-4: vLLM Server (4-way tensor parallel generation)   │
├─────────────────────────────────────────────────────────────┤
│  GPUs 5-7: Available for other tasks                        │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- TRL with vLLM support (`pip install trl[vllm]`)
- GRAIL codebase (for task sources and validation logic)
- vLLM server running on port 8000
- Flash Attention 2 (optional, for faster training)

## Files

| File | Description |
|------|-------------|
| `train_trl_grpo.py` | Main training script (unified GSM8K + MATH) |
| `train_trl_grpo_README.md` | This documentation |
| `train_trl_gsm8k.py` | Legacy GSM8K-only script (deprecated) |