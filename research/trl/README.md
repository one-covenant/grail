# TRL GRPO Training Experiments

This directory contains TRL-based GRPO training experiments for GRAIL, with support for GSM8K, MATH, and MBPP datasets.

## Self-Contained Environment

This research project has its own isolated Python environment managed by `uv` through `pyproject.toml`.

### Why Separate?

- **Isolation**: TRL experiments don't need blockchain/networking dependencies from main GRAIL
- **Clarity**: Explicit dependencies for reproducibility
- **Speed**: Faster installs (only ML/training dependencies)
- **Independence**: Can update TRL versions without affecting main project

### Dependency Structure

```
research/trl/pyproject.toml
├── Core ML: torch, transformers, trl, datasets
├── Training: accelerate, peft
├── Logging: wandb, tensorboard
├── Utilities: python-dotenv, tqdm, aiohttp
├── Math: numpy, sympy (for MATH dataset)
└── grail (editable via tool.uv.sources)
    └── Provides: dataset providers, code execution, metrics, chat templates
```

The main GRAIL package is installed as an **editable dependency**, so we can import:
- `grail.environments.providers` (dataset sources)
- `grail.environments.execution` (code execution for MBPP)
- `grail.trainer.metrics` (pass@k evaluation)
- `grail.shared.chat_templates` (Qwen templates)

## Setup

### Local Development

```bash
cd research/trl

# Install dependencies (creates .venv/)
uv sync

# Activate environment
source .venv/bin/activate

# Run training
python train_trl_grpo.py --dataset gsm8k
```

### Using run_algo.sh

The included `run_algo.sh` script handles everything:

```bash
cd research/trl

# Default: GSM8K with GPUs 0+1
./run_algo.sh

# Run on MATH dataset
./run_algo.sh math

# Custom GPU assignment (for multi-experiment setups)
VLLM_GPU=2 TRAIN_GPU=3 VLLM_PORT=8001 ./run_algo.sh gsm8k
```

The script:
1. Sets up TRL environment if needed (`uv sync`)
2. Sets up VLLM environment if needed
3. Starts VLLM server on specified GPU
4. Waits for VLLM to be ready
5. Runs training on training GPU
6. Cleans up VLLM server when done

### On Lium Pods

The infrastructure automatically handles this:

```bash
# Automated by deploy.py:
cd ~/grail/research/trl
uv sync  # Installs TRL deps + grail as editable

# Then experiments run from this directory:
source .venv/bin/activate
python train_trl_grpo.py --dataset gsm8k
```

## Training Script

### Supported Datasets

- **GSM8K**: Grade school math (7,473 train / 1,319 test)
- **MATH**: Hendrycks MATH benchmark (7,000 train / 500 val / 5,000 test)
- **MBPP**: Python code generation (374 train / 90 validation / 500 test)

### Usage

```bash
# Basic usage
python train_trl_grpo.py --dataset gsm8k

# With custom eval interval
python train_trl_grpo.py --dataset math --eval-every 20

# MBPP (with code execution)
python train_trl_grpo.py --dataset mbpp
```

### Configuration

Edit hyperparameters in the `Config` dataclass in `train_trl_grpo.py`:

```python
@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 3e-6
    batch_size: int = 4
    grad_accum_steps: int = 128
    total_steps: int = 100
    # ... more config ...
```

Or pass via environment variables (see `.env` file).

## Directory Structure

```
research/trl/
├── pyproject.toml              # Dependencies (uv managed)
├── train_trl_grpo.py          # Main training script
├── analysis_integration_example.py  # Analysis example
├── run_algo.sh                # Launch script (for local development)
├── .venv/                     # Virtual environment (created by uv sync)
└── outputs/                   # Training outputs (gitignored)
```

## Integration with GRAIL

The script imports GRAIL modules for:

1. **Dataset Providers**
   ```python
   from grail.environments.providers import (
       GSM8KTaskSource,
       MATHTaskSource,
       MBPPTaskSource,
   )
   ```

2. **Code Execution** (for MBPP)
   ```python
   from grail.environments.execution import (
       CodeExecutionPool,
       check_code_executes,
   )
   ```

3. **Metrics** (pass@k evaluation)
   ```python
   from grail.trainer.metrics import KMetricsAggregator
   ```

4. **Chat Templates**
   ```python
   from grail.shared.chat_templates import build_qwen_chat_template
   ```

Since GRAIL is installed as an **editable dependency**, any changes to the main GRAIL codebase are immediately reflected (no reinstall needed).

## VLLM Server

TRL uses VLLM for fast generation during training. The VLLM server has its own environment:

```bash
cd ../../tools/vllm-server
uv sync
source .venv/bin/activate
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct ...
```

See `tools/vllm-server/README.md` for details.

## Distributed Training via Lium

For running experiments on Lium infrastructure, see `../infrastructure/`:

```bash
cd ../infrastructure

# Run 4 parallel experiments on 1 pod
python deploy.py --config lr_sweep

# The infrastructure automatically:
# 1. Deploys pods
# 2. Git clones repo
# 3. Installs uv
# 4. Runs `uv sync` in research/trl
# 5. Runs `uv sync` in tools/vllm-server
# 6. Starts experiments
```

## Updating Dependencies

```bash
# Add a new dependency
uv add package-name

# Update existing dependency
uv add package-name@latest

# Sync (install/update from lock file)
uv sync

# Regenerate lock file
uv lock
```

## Troubleshooting

### Import Error: "No module named 'grail'"

**Cause**: GRAIL editable dependency not installed

**Fix**:
```bash
cd research/trl
uv sync  # Reinstall, includes grail
```

### VLLM Server Not Found

**Cause**: Wrong venv activated

**Fix**:
```bash
cd tools/vllm-server
source .venv/bin/activate  # Activate VLLM venv, not TRL venv
trl vllm-serve ...
```

### Flash Attention Errors

**Cause**: Optional dependency not installed

**Fix**:
```bash
# Install with GPU extras
uv sync --extra gpu

# Or fall back to SDPA (the script handles this automatically)
```

## Performance Notes

- **1 GPU for VLLM + 1 GPU for training** = 2 GPUs per experiment
- **4 parallel experiments** on 8-GPU pod
- Each experiment logs to WandB independently
- Logs saved to `train_*.log` and `vllm_server_*.log`

## Related Documentation

- **Main GRAIL**: `../../README.md`
- **Lium Infrastructure**: `../infrastructure/README.md`
- **VLLM Server**: `../../tools/vllm-server/README.md`
