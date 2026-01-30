# Parallel Training

Run GRPO training with vLLM server on GPU pairs.

## Quick Start

```bash
cd research/trl

# Single experiment (uses GPUs 0,1)
./run_parallel_training_nohup.sh math 40 Qwen/Qwen2.5-1.5B-Instruct 8 1
```

## Command Arguments

```bash
./run_parallel_training_nohup.sh [DATASET] [EVAL_EVERY] [MODEL] [NUM_ITERATIONS] [NUM_INSTANCES] [BATCH_SIZE] [GRAD_ACCUM]
```

| Arg | Default | Description |
|-----|---------|-------------|
| DATASET | math | gsm8k, math, or mbpp |
| EVAL_EVERY | 40 | Evaluation frequency (steps) |
| MODEL | Qwen/Qwen2.5-1.5B-Instruct | Model ID |
| NUM_ITERATIONS | 1 | Training updates per rollout batch |
| NUM_INSTANCES | 1 | Parallel instances (1, 2, or 4) |
| BATCH_SIZE | (default) | Batch size per device |
| GRAD_ACCUM | (default) | Gradient accumulation steps |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAIL_RUN_PREFIX` | - | Run name prefix (e.g., `iter8_`) |
| `GRAIL_SEED` | - | Override seed (e.g., `1337`) |
| `GRAIL_START_INSTANCE` | 0 | GPU pair index (0=GPUs 0,1; 1=GPUs 2,3; etc.) |
| `GRAIL_BASE_PORT` | 8000 | vLLM server port |
| `GRAIL_BASE_GROUP_PORT` | 51200 | NCCL group coordination port |
| `GRAIL_VLLM_NIXL_PORT_BASE` | 5557 | vLLM internal NIXL port |
| `GRAIL_VLLM_MASTER_PORT_BASE` | 29500 | vLLM master port |
| `GRAIL_OUTPUT_BASE` | . | Output directory (use `/ephemeral` on cloud) |
| `WANDB_PROJECT` | grail-lium-sweep | W&B project name |
| `WANDB_TAGS` | - | Comma-separated W&B tags |

## Running Multiple Experiments on Same Server

Run two experiments on different GPU pairs with isolated ports:

```bash
# Experiment 1: GPUs 0,1
GRAIL_RUN_PREFIX=exp1_ \
GRAIL_SEED=1337 \
GRAIL_START_INSTANCE=0 \
GRAIL_BASE_PORT=8000 \
GRAIL_BASE_GROUP_PORT=51200 \
GRAIL_OUTPUT_BASE=/ephemeral \
./run_parallel_training_nohup.sh math 40 Qwen/Qwen2.5-1.5B-Instruct 8 1

# Experiment 2: GPUs 2,3
GRAIL_RUN_PREFIX=exp2_ \
GRAIL_SEED=1337 \
GRAIL_START_INSTANCE=1 \
GRAIL_BASE_PORT=8010 \
GRAIL_BASE_GROUP_PORT=51300 \
GRAIL_VLLM_NIXL_PORT_BASE=5567 \
GRAIL_VLLM_MASTER_PORT_BASE=29510 \
GRAIL_OUTPUT_BASE=/ephemeral \
./run_parallel_training_nohup.sh math 40 Qwen/Qwen2.5-1.5B-Instruct 16 1
```

## GPU Allocation

| Instance Index | vLLM GPU | Training GPU |
|----------------|----------|--------------|
| 0 | 0 | 1 |
| 1 | 2 | 3 |
| 2 | 4 | 5 |
| 3 | 6 | 7 |

## Monitoring

```bash
# Check GPU usage
nvidia-smi

# View launcher logs
tail -f logs/parallel_training/launcher_*.log

# View training logs
tail -f logs/parallel_training/training_instance*.log

# Check running processes
pgrep -af 'train_trl_grpo|vllm-serve'
```

## Stopping

```bash
# Kill all training processes
pkill -f 'vllm-serve'
pkill -f 'train_trl_grpo'
pkill -f 'run_parallel_training'

# Or kill specific experiment by prefix
pkill -f 'exp1_'
```

## Output Locations

```
logs/parallel_training/
  launcher_{prefix}_{timestamp}.log
  training_instance{i}_gpu{g}_seed{s}.log
  vllm_instance{i}_gpu{g}_port{p}.log

/ephemeral/  (or GRAIL_OUTPUT_BASE)
  outputs/trl_{dataset}_{run_suffix}/
  checkpoints/deltas_{dataset}_{run_suffix}/
  wandb/
```
