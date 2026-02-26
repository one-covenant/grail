# Infrastructure: Parallel Training Deployment

Deploy and run GRPO training experiments on Basilica/Lium cloud instances.

## Quick Start

```bash
cd research/infrastructure
source .venv/bin/activate

# Deploy and run
python deploy_parallel_basilica.py --config iter_ablation_1_5b

# Destroy instances when done
python deploy_parallel_basilica.py --destroy --config iter_ablation_1_5b
```

## Commands

```bash
# List available configs
python deploy_parallel_basilica.py --list-configs

# Deploy only (no training)
python deploy_parallel_basilica.py --config <name> --deploy-only

# Run on existing instances (skip deploy)
python deploy_parallel_basilica.py --config <name> --no-deploy

# Skip code sync and env setup (faster reruns)
python deploy_parallel_basilica.py --config <name> --no-deploy --no-sync --no-setup

# Custom dataset/eval interval
python deploy_parallel_basilica.py --config <name> --dataset gsm8k --eval-every 50
```

## Model Configs (MODEL_CONFIGS dict)

| Key | Description |
|-----|-------------|
| `model` | HuggingFace model ID |
| `num_iterations` | GRPO training iterations per rollout |
| `wandb_project` | W&B project name |
| `wandb_tags` | Comma-separated tags |
| `batch_size` | Batch size per device |
| `grad_accum_steps` | Gradient accumulation steps |
| `num_instances` | Parallel instances (1-4) |
| `run_prefix` | Unique prefix for run names |
| `seed` | Override seed |
| `start_instance` | GPU pair index (0=GPUs 0,1; 1=GPUs 2,3) |
| `base_port` | vLLM server port |
| `base_group_port` | NCCL group port |
| `vllm_nixl_port_base` | vLLM NIXL port |
| `vllm_master_port_base` | vLLM master port |

## Running Multiple Experiments on Same Node

Configure two experiments to run on different GPU pairs:

```python
MODEL_CONFIGS = {
    "exp1": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 8,
        "num_instances": 1,
        "run_prefix": "exp1_",
        "seed": 1337,
        "start_instance": 0,      # GPUs 0,1
        "base_port": 8000,
        "base_group_port": 51200,
    },
    "exp2": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 16,
        "num_instances": 1,
        "run_prefix": "exp2_",
        "seed": 1337,
        "start_instance": 1,      # GPUs 2,3
        "base_port": 8010,
        "base_group_port": 51300,
        "vllm_nixl_port_base": 5567,
        "vllm_master_port_base": 29510,
    },
}
```

## Environment Variables (.env)

```bash
# R2 Storage
R2_BUCKET_NAME=your-bucket
R2_ACCOUNT_ID=your-account-id
R2_WRITE_ACCESS_KEY_ID=your-key
R2_WRITE_SECRET_ACCESS_KEY=your-secret

# W&B
WANDB_API_KEY=your-key

# HuggingFace (for gated models)
HF_TOKEN=your-token

# Basilica
BASILICA_API_KEY=your-key
```

## Monitoring

```bash
# SSH into instance
ssh ubuntu@<host>

# Check GPU usage
nvidia-smi

# View training logs
tail -f ~/grail/research/trl/logs/parallel_training/launcher_*.log
tail -f ~/grail/research/trl/logs/parallel_training/training_instance*.log

# Check running processes
pgrep -af 'train_trl_grpo|vllm-serve'
```

## Output Structure

```
/ephemeral/                      # On cloud instance
  outputs/trl_{dataset}_{run_suffix}/
  checkpoints/deltas_{dataset}_{run_suffix}/
  wandb/

./downloads/<experiment_name>/    # Downloaded locally
  logs/
  outputs/
  checkpoints/
```

## Files

| File | Description |
|------|-------------|
| `deploy_parallel_basilica.py` | Main deploy script for Basilica |
| `nohup_experiment_runner.py` | Remote experiment execution |
| `experiment_configs.py` | Experiment configurations |
| `basilica_manager.py` | Basilica API wrapper |
| `r2_uploader.py` | R2 storage uploads |
