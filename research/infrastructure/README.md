# Lium Infrastructure for Distributed Training

This directory contains infrastructure automation for running distributed training experiments on Lium cloud infrastructure. It provides declarative pod management, experiment orchestration, and hyperparameter sweep capabilities.

## Overview

The infrastructure system consists of four main components:

1. **`lium_manager.py`**: Declarative infrastructure management with bandwidth filtering
2. **`experiment_runner.py`**: SSH-based experiment orchestration and execution
3. **`experiment_configs.py`**: Predefined experiment configurations and templates
4. **`deploy.py`**: Main CLI for deployment and experiment execution

## Features

- **Declarative Infrastructure**: Define desired pod state and let the system handle creation/deletion
- **Bandwidth Filtering**: Specify minimum upload/download bandwidth requirements
- **State Persistence**: Track managed pods across sessions with JSON state file
- **Parallel Execution**: Run multiple experiments across multiple pods simultaneously
- **Code Synchronization**: Automatic code sync via rsync over SSH
- **Environment Setup**: Automated Python environment and dependency management
- **Log Collection**: Capture and save experiment logs locally
- **Auto-termination**: Schedule pods to auto-terminate after specified TTL

## Quick Start

### Prerequisites

1. **Lium SDK**: Install the Lium SDK
   ```bash
   pip install lium-sdk
   ```

2. **SSH Access**: Ensure you have SSH key authentication set up
   ```bash
   ssh-keygen -t ed25519  # If you don't have a key
   ```

3. **Lium API Key**: Set your Lium API key
   ```bash
   export LIUM_API_KEY="your-api-key"
   ```

4. **WandB** (optional): For experiment tracking
   ```bash
   export WANDB_API_KEY="your-wandb-key"
   export WANDB_PROJECT="grail-experiments"
   ```

### Basic Usage

1. **List available configurations**:
   ```bash
   cd research/infrastructure
   python deploy.py --list-configs
   ```

2. **Inspect available executors**:
   ```bash
   python deploy.py --inspect-executors --gpu-type A100
   ```

3. **Deploy and run experiments**:
   ```bash
   python deploy.py --config lr_sweep
   ```

4. **Cleanup when done**:
   ```bash
   python deploy.py --destroy
   ```

## Predefined Configurations

### 1. Learning Rate Sweep (`lr_sweep`)

Tests different learning rates (1e-6, 3e-6, 5e-6, 1e-5) on GSM8K dataset.

- **Pods**: 2x A100 (8 GPUs each)
- **Experiments**: 4 (2 per pod)
- **Runtime**: ~6 hours

```bash
python deploy.py --config lr_sweep
```

### 2. Multi-Dataset Comparison (`multi_dataset`)

Compares performance across GSM8K, MATH, and MBPP datasets.

- **Pods**: 3x A100 (8 GPUs each, one per dataset)
- **Experiments**: 3 (1 per pod)
- **Runtime**: ~8 hours

```bash
python deploy.py --config multi_dataset
```

### 3. Model Size Ablation (`model_size`)

Compares Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B models.

- **Pods**: 3x A100 (8 GPUs each)
- **Experiments**: 3 (1 per pod)
- **Runtime**: ~10-12 hours

```bash
python deploy.py --config model_size
```

### 4. Batch Size Grid Search (`batch_grid`)

Grid search over batch size and gradient accumulation (constant effective batch = 512).

- **Pods**: 4x A100 (8 GPUs each)
- **Experiments**: 4 (1 per pod)
- **Runtime**: ~8 hours

```bash
python deploy.py --config batch_grid
```

### 5. Custom Advanced (`custom_advanced`)

Example with custom environment variables and arguments.

- **Pods**: 1x A100 (8 GPUs)
- **Experiments**: 1
- **Runtime**: ~10 hours

```bash
python deploy.py --config custom_advanced
```

## Advanced Usage

### Deploy Pods Only (No Experiments)

Useful for setting up infrastructure ahead of time:

```bash
python deploy.py --config lr_sweep --deploy-only
```

### Run on Existing Pods

Skip pod deployment and use existing managed pods:

```bash
python deploy.py --config lr_sweep --no-deploy
```

### Skip Code Sync/Setup

Useful for faster iteration during debugging:

```bash
python deploy.py --config lr_sweep --no-sync --no-setup
```

### Custom State File

Use a different state file for separate clusters:

```bash
python deploy.py --config lr_sweep --state-file .lium_state_prod.json
```

## Creating Custom Configurations

### Example: Simple Custom Configuration

Create a new function in `experiment_configs.py`:

```python
def my_custom_sweep() -> tuple[list[PodSpec], dict[str, list[ExperimentConfig]]]:
    """My custom hyperparameter sweep."""

    # Define pods
    pods = [
        PodSpec(
            name="custom-pod-0",
            gpu_type="A100",
            gpu_count=8,
            min_upload_mbps=500,
            min_download_mbps=500,
            ttl_hours=6,
        ),
    ]

    # Define experiments
    experiments = [
        ExperimentConfig(
            name="my_exp_1",
            dataset="gsm8k",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=1e-6,
            batch_size=4,
            grad_accum_steps=128,
            total_steps=50,
            eval_every=10,
        ),
        ExperimentConfig(
            name="my_exp_2",
            dataset="gsm8k",
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            learning_rate=5e-6,
            batch_size=4,
            grad_accum_steps=128,
            total_steps=50,
            eval_every=10,
        ),
    ]

    pod_experiments = {
        "custom-pod-0": experiments,
    }

    return pods, pod_experiments
```

Then add it to the `get_config()` function:

```python
def get_config(name: str):
    configs = {
        # ... existing configs ...
        "my_custom": my_custom_sweep,
    }
    # ...
```

### Example: Programmatic Configuration

For more complex sweeps, use loops:

```python
def sweep_all_learning_rates():
    pods = [PodSpec(name=f"lr-pod-{i}", ...) for i in range(10)]

    learning_rates = [10 ** (-i) for i in range(4, 8)]  # 1e-4 to 1e-7

    pod_experiments = {}
    for i, lr in enumerate(learning_rates):
        pod_name = f"lr-pod-{i}"
        pod_experiments[pod_name] = [
            ExperimentConfig(
                name=f"gsm8k_lr_{lr:.0e}",
                learning_rate=lr,
                # ... other params ...
            )
        ]

    return pods, pod_experiments
```

## Architecture Details

### Pod Lifecycle

```
┌─────────────┐
│  apply()    │  ← Declarative: specify desired state
└──────┬──────┘
       │
       ├─→ Create missing pods
       ├─→ Keep existing pods
       └─→ Destroy extra pods
              ┌──────────────┐
              │  Pod Ready   │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Sync Code    │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ Setup Env    │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │Run Experiments│
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │   Cleanup    │
              └──────────────┘
```

### Experiment Execution Flow

For each pod:

1. **SSH Connection**: Establish connection via asyncssh
2. **Code Sync**: Rsync local code to remote pod
3. **Environment Setup**: Install uv, sync dependencies
4. **Experiment Loop**: For each experiment config:
   - Generate experiment script
   - Start VLLM server (background)
   - Run training (foreground)
   - Collect logs
   - Cleanup (kill VLLM, remove temp files)
5. **Disconnect**: Close SSH connection

### Parallel Execution

Experiments run in parallel across pods using `asyncio`:

```
Pod 0: [Exp 1] → [Exp 2] → [Exp 3]
Pod 1: [Exp 4] → [Exp 5] → [Exp 6]  ← All pods run concurrently
Pod 2: [Exp 7] → [Exp 8] → [Exp 9]
```

Within each pod, experiments run **sequentially** to avoid GPU conflicts.

## Configuration Reference

### PodSpec

```python
@dataclass
class PodSpec:
    name: str                          # Unique pod identifier
    gpu_type: str = "A100"             # GPU type
    gpu_count: int = 8                 # Number of GPUs
    country: Optional[str] = None      # Location filter
    volume_id: Optional[str] = None    # Persistent storage
    ttl_hours: Optional[int] = None    # Auto-terminate after N hours
    min_upload_mbps: Optional[float] = None    # Min upload bandwidth
    min_download_mbps: Optional[float] = None  # Min download bandwidth
```

### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    name: str                          # Unique experiment name
    dataset: str = "gsm8k"             # Dataset: gsm8k, math, mbpp
    model_id: str = "..."              # HuggingFace model ID

    # Training hyperparameters
    learning_rate: float = 3e-6
    batch_size: int = 4
    grad_accum_steps: int = 128
    total_steps: int = 100
    eval_every: int = 40

    # GPU configuration
    gpu_devices: str = "1,2,3,4"       # VLLM GPUs
    train_gpu_device: str = "0"        # Training GPU
    vllm_tensor_parallel: int = 4

    # Optional overrides
    custom_env: dict[str, str]         # Extra env vars
    custom_args: dict[str, Any]        # Extra CLI args
```

## Troubleshooting

### Pod Creation Fails

**Issue**: No executors found matching requirements

**Solution**:
- Relax bandwidth requirements
- Try different GPU types
- Check executor availability: `python deploy.py --inspect-executors`

### SSH Connection Fails

**Issue**: Cannot connect to pod

**Solution**:
- Verify pod is ready: check Lium dashboard
- Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
- Wait longer (pods can take 2-5 minutes to boot)

### Code Sync Fails

**Issue**: Rsync fails or times out

**Solution**:
- Check SSH connectivity first
- Ensure rsync is installed locally: `which rsync`
- Try manual sync: `rsync -avz . user@host:~/grail`

### VLLM Server Won't Start

**Issue**: Training waits forever for VLLM

**Solution**:
- Increase sleep time in experiment script (default: 100s)
- Check VLLM logs: `ssh pod "tail -f tools/vllm-server/vllm_server_*.log"`
- Verify GPU availability: `ssh pod "nvidia-smi"`

### Out of Memory

**Issue**: CUDA OOM during training/inference

**Solution**:
- Reduce `batch_size`
- Increase `grad_accum_steps` (keeps effective batch size constant)
- Reduce VLLM `gpu_memory_utilization` (default: 0.9)
- Use smaller model or fewer GPUs for VLLM

## Cost Estimation

Approximate costs for different configurations (varies by executor):

| Configuration | Pods | GPU-Hours | Est. Cost* |
|--------------|------|-----------|-----------|
| lr_sweep | 2x8xA100 | 96 | $150-200 |
| multi_dataset | 3x8xA100 | 192 | $300-400 |
| model_size | 3x8xA100 | 240 | $400-500 |
| batch_grid | 4x8xA100 | 256 | $400-550 |

*Assuming ~$2/hr per 8xA100 pod (actual prices vary by location and provider)

**Cost Optimization Tips**:
- Use `ttl_hours` to auto-terminate pods
- Use `--deploy-only` to inspect costs before running
- Run shorter experiments first (`total_steps=10`) to validate
- Share pods across related experiments

## Best Practices

### 1. Test Locally First

Before deploying to Lium, test your configuration locally:

```bash
# Quick local test (1 step)
cd research/trl
CUDA_VISIBLE_DEVICES=0 python train_trl_grpo.py --dataset gsm8k
```

### 2. Start Small

Begin with a single pod and short experiments:

```python
pods = [PodSpec(name="test", gpu_count=8, ttl_hours=1)]
experiments = [ExperimentConfig(name="quick_test", total_steps=5)]
```

### 3. Use TTL

Always set `ttl_hours` to avoid runaway costs:

```python
PodSpec(name="...", ttl_hours=6)  # Auto-terminate after 6 hours
```

### 4. Monitor Progress

Check WandB dashboard during runs:
- Navigate to your project
- Filter by run name prefix
- Compare metrics across experiments

### 5. Save State

The `.lium_state.json` file tracks your pods. Keep it safe:

```bash
# Backup state
cp .lium_state.json .lium_state.backup.json

# Version control (optional, but exclude from git if using API keys)
echo ".lium_state.json" >> .gitignore
```

## Advanced Topics

### Custom Training Scripts

To use a different training script:

1. Modify `experiment_runner.py::run_experiment()` script template
2. Update `ExperimentConfig` with new parameters
3. Create custom config in `experiment_configs.py`

### Multi-Node Training

For true distributed training across multiple pods:

1. Use PyTorch DDP with `torchrun`
2. Configure `MASTER_ADDR` and `MASTER_PORT` in `custom_env`
3. Coordinate pod IPs via Lium API

### Persistent Volumes

Attach persistent storage for checkpoints:

```python
PodSpec(
    name="trainer",
    volume_id="vol-xxxxx",  # Create via Lium dashboard
)
```

### Custom Executors

Filter by specific executor features:

```python
# In lium_manager.py::_find_executor(), add custom filters:
if e.location.get("city") == "Amsterdam":
    candidates.append(e)
```

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review Lium SDK docs: https://docs.lium.ai
3. File an issue in the GRAIL repository

## License

This infrastructure code is part of the GRAIL project. See main repository LICENSE for details.
