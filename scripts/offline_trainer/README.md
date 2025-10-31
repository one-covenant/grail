# Offline GRPO Trainer

A self-contained offline GRPO (Group Relative Policy Optimization) training and evaluation runner that works without bittensor dependencies. Designed for hyperparameter tuning and trainer development before deploying on-chain.

## Features

- **Server-backed generation**: Uses SGLang or vLLM OpenAI-compatible servers for inference
- **Complete GRPO loop**: Generate rollouts → Train epoch → Evaluate → Checkpoint
- **Hydra configuration**: Easy hyperparameter sweeps and experimentation
- **No bittensor dependency**: Runs completely offline
- **GPU and CPU support**: Tested on both platforms

## Structure

```
scripts/offline_trainer/
├── run_offline_grpo.py      # Main Hydra entrypoint
├── offline_rollouts.py      # Server-backed rollout generator
├── conf/
│   └── offline_grpo.yaml    # Hydra configuration
├── smoke_test.py            # Quick CPU smoke test
├── test_offline_trainer.py  # Comprehensive test suite
├── run_tests.py             # Test runner
└── pyproject.toml           # uv dependencies (no bittensor)
```

## Quick Start

### 0. Install Dependencies

```bash
cd scripts/offline_trainer
uv sync  # Installs all dependencies (no bittensor)
```

### 1. Start a Generation Server

**Important:** Server runs on separate GPU from training. Choose one backend:

**vLLM Server (Recommended):**
```bash
# In separate terminal - uses GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B \
  --port 30000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5
```

**SGLang Server:**
```bash
# In separate terminal - uses GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B \
  --port 30001
```

### 2. Run Training

```bash
cd scripts/offline_trainer

# With UV (recommended)
uv run python run_offline_grpo.py \
  model.train_id=Qwen/Qwen2.5-1.5B \
  model.ref_id=Qwen/Qwen2.5-1.5B \
  model.device=cuda:2 \
  generation.backend=vllm_server \
  generation.base_url=http://127.0.0.1:30000 \
  eval.backend=vllm_server \
  eval.base_url=http://127.0.0.1:30000 \
  train.iterations=10

# Or directly with system Python
python3 run_offline_grpo.py [same args]
```

### 3. Hyperparameter Sweep

```bash
# UV multirun
uv run python run_offline_grpo.py -m \
  train.lr='[5e-6,1e-5,2e-5]' \
  train.kl_coef='[0.02,0.05,0.1]'
  
# Creates separate runs for each combination (3×3=9 runs)
```

## Configuration

Edit `conf/offline_grpo.yaml` or override via CLI:

```yaml
model:
  train_id: "Qwen/Qwen2.5-1.5B"
  ref_id: "Qwen/Qwen2.5-1.5B"
  device: "auto"  # auto|cuda|cpu

generation:
  backend: "sglang_server"  # sglang_server|vllm_server
  base_url: "http://127.0.0.1:30001"
  batch_size: 4
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95

train:
  iterations: 5
  batch_size: 8
  lr: 1.0e-5
  kl_coef: 0.05
  entropy_coef: 0.0

data:
  problems_per_iteration: 4
  rollouts_per_problem: 4
  train_seed_start: 1000
  num_train_seeds: 64

eval:
  enabled: true
  interval: 1
  backend: "sglang_server"
  base_url: "http://127.0.0.1:30001"
  replicates: 4
  num_ids: 16
```

## Testing

**Quick smoke test (CPU, no server needed):**
```bash
# With UV
uv run python smoke_test.py

# Or directly
python3 smoke_test.py
```

**Full test suite (includes GPU tests):**
```bash
# With UV
uv run python run_tests.py

# Or directly  
python3 run_tests.py
```

**Expected output:**
```
============================================================
Results: 7/7 tests passed
============================================================
```

**GPU integration tests (with vLLM server):**
```bash
# Starts vLLM servers automatically for testing
# With UV
uv run python test_gpu_integration.py

# Or directly
python3 test_gpu_integration.py
```

**Expected output:**
```
============================================================
GPU Integration Test Results: 4/4 passed
============================================================
```

## Output

Hydra creates structured output in `outputs/YYYY-MM-DD/HH-MM-SS/`:

```
outputs/
└── 2024-01-15/
    └── 10-30-45/
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

## Architecture

### Rollout Generation

`OfflineRolloutGenerator` uses server backends to:
1. Generate SAT problem prompts deterministically from seeds
2. Request completions from SGLang/vLLM server with per-replicate seeds
3. Step environments to compute rewards
4. Package into `GRPOGroup` objects with zero-mean, variance-normalized advantages

### Training Loop

The runner:
1. Loads train/ref models and tokenizer via `grail.model.provider`
2. Generates fresh rollouts per iteration
3. Runs GRPO training epoch using `GRPOAlgorithm.train_epoch`
4. Periodically evaluates using `EvaluatorService`
5. Saves checkpoints and metrics to Hydra output dir

### Core Integration

Minimal changes to core codebase:
- `grail/environments/loop.py`: Optional bittensor import, added `VLLMServerBackend`
- `grail/trainer/evaluator.py`: Added `vllm_server` backend option
- `grail/__init__.py`: Optional comms import for offline mode
- `grail/trainer/algorithms/grpo.py`: Optional miner_data import

## Dependencies

Via `uv sync` in `scripts/offline_trainer/`:
- `torch>=2.4.1`
- `transformers==4.57.1`
- `accelerate>=0.20.0`
- `hydra-core>=1.3.2`
- `openai>=1.40.0`
- `numpy>=1.20.0`
- `rich>=14.1.0`

**No bittensor dependency** - this is completely offline.

## Troubleshooting

**Import errors**: Ensure repo root is on `PYTHONPATH` or run from repo root.

**Server connection errors**: Verify server is running and `base_url` is correct.

**GPU out of memory**: Reduce `batch_size` or use CPU (`model.device=cpu`).

**Empty rollouts**: Check that server returns valid completions and seeds are deterministic.

## Design Notes

- **Server separation**: Generation happens in separate server process, keeping training GPU memory clean
- **Determinism**: All generation uses seeds for reproducibility
- **Modular**: Core training algorithm (`GRPOAlgorithm`) unchanged; offline runner wraps it
- **Type-safe**: Full type hints throughout, PEP8 compliant

