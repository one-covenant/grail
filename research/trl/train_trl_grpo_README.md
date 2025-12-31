# TRL GRPO Training Script

Unified TRL GRPO training script supporting both GSM8K and MATH (Hendrycks) datasets with exact parity to GRAIL environment implementations.

---

## Quick Start

### Step 1: Launch the vLLM Server (Generation GPUs)

**Activate vLLM environment:**
```bash
source tools/vllm-server/.venv/bin/activate
```

**Launch vLLM server on GPUs 1-4 (4-way tensor parallel):**
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 nohup trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  > vllm_server.log 2>&1 &
```

**Wait for server to be ready:**
```bash
tail -f vllm_server.log
```

### Step 2: Start GRPO Training (Training GPU)

Navigate to the training directory:
```bash
cd research/trl/
```

**Option A: Train on GSM8K (default dataset)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_trl_grpo.py \
  --dataset gsm8k \
  > train_gsm8k.log 2>&1 &
```

**Option B: Train on MATH (Hendrycks dataset)**
```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_trl_grpo.py \
  --dataset math \
  > train_math.log 2>&1 &
```

**Option C: Train with custom eval frequency**
```bash
CUDA_VISIBLE_DEVICES=0 python train_trl_grpo.py \
  --dataset math \
  --eval-every 50
```

> **Monitor progress:** `tail -f train_gsm8k.log` or `tail -f train_math.log`

---

## Requirements

- **TRL:** `pip install trl[vllm]`
- **GRAIL codebase** (for task sources and validation logic)
- **vLLM server** running on port 8000
- **Flash Attention 2** (optional, for faster training)
- **Hardware:** 8x A100 GPUs (GPU 0: training, GPUs 1-4: vLLM, GPUs 5-7: available)

---

## Datasets

| Aspect | GSM8K | MATH |
|--------|-------|------|
| Train Size | 7,473 | 7,000 |
| Eval Size | 1,319 (test) | 500 (stratified val) |
| Gold Format | `#### answer` | `\boxed{answer}` |
| Validation | Numeric exact | Multi-strategy (exact/sympy/numeric) |

---

## Configuration

All hyperparameters are set via `.env`:

| Parameter | Value | Env Variable |
|-----------|-------|--------------|
| Learning rate | 3e-6 | `GRAIL_TRAINER_LR` |
| Batch size | 4 | `GRAIL_TRAINER_BATCH_SIZE` |
| Gradient accum | 128 | `GRAIL_TRAINER_GRAD_ACCUM_STEPS` |
| Max length | 2048 | `GRAIL_TRAINER_MAX_LENGTH` |
| Max completion | 1024 | `GRPO_MAX_COMPLETION_TOKENS` |
| Loss type | dapo | `GRAIL_GRPO_VARIANT` |

---

## Reward System

### GSM8K Weights (Total: 1.0)

| Component | Weight |
|-----------|--------|
| Correctness (numeric match) | 0.6 |
| Answer format (numeric-only) | 0.15 |
| Reasoning block | 0.1 |
| Solution tags | 0.1 |
| No trailing text | 0.05 |

### MATH Weights (Total: 1.0)

| Component | Weight |
|-----------|--------|
| Correctness (multi-strategy) | 0.7 |
| Answer format (< 50 chars trailing) | 0.15 |
| Reasoning block | 0.1 |
| No trailing text | 0.05 |

---

## Architecture

```
train_trl_grpo.py
├── DatasetAdapter (ABC)
│   ├── GSM8KAdapter
│   └── MATHAdapter
├── get_dataset_adapter()
├── VLLMEvalCallback
└── main()
```

- **DatasetAdapter**: Abstract base for dataset implementations
- **GSM8KAdapter / MATHAdapter**: Dataset-specific task sources and validation
- **VLLMEvalCallback**: Async batched evaluation via vLLM server

---

## Files

| File | Description |
|------|-------------|
| `train_trl_grpo.py` | Main training script (unified GSM8K + MATH) |
| `train_trl_grpo_README.md` | This documentation |
| `train_trl_gsm8k.py` | Legacy GSM8K-only script (deprecated) |
