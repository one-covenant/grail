# VeRL GRPO Training

This directory contains scripts for training language models using the VeRL framework with GRPO (Group Relative Policy Optimization), providing equivalent functionality to the TRL-based training in `research/trl/`.

## Overview

VeRL (Volcano Engine Reinforcement Learning) is a flexible, efficient RL training framework for LLMs that supports:
- PPO and GRPO algorithms
- FSDP and Megatron-LM backends
- vLLM and SGLang for fast inference
- Distributed training with Ray

## Files

- `train_verl_grpo.py` - Main training script with data preparation
- `reward_functions.py` - Custom reward functions matching GRAIL environments
- `run_verl_grpo.sh` - Shell script for easy training execution
- `config_*.yaml` - Generated configuration files

## Quick Start

### 1. Install VeRL

```bash
# Using pip (recommended for FSDP-only setup)
pip install verl

# Or from source
git clone https://github.com/volcengine/verl.git
cd verl && pip install -e .
```

### 2. Prepare Data

```bash
# Generate parquet files for your dataset
python train_verl_grpo.py --dataset gsm8k --prepare-data-only

# Or use the shell script
./run_verl_grpo.sh prepare
```

### 3. Run Training

```bash
# Using the shell script (recommended)
./run_verl_grpo.sh train

# Or using the Python script
python train_verl_grpo.py --dataset gsm8k --run-training

# Or directly with VeRL
python -m verl.trainer.main_ppo \
    data.train_files=~/data/grail_gsm8k/train.parquet \
    data.val_files=~/data/grail_gsm8k/test.parquet \
    data.train_batch_size=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    +custom_reward_function.path=./reward_functions.py \
    +custom_reward_function.name=compute_score
```

## Supported Datasets

| Dataset | Description | Train Size | Eval Size |
|---------|-------------|------------|-----------|
| gsm8k   | Grade school math | 7,473 | 1,319 |
| math    | Hendrycks MATH | 7,000 | 500 |
| mbpp    | Python code generation | 374 | 90 |

## Configuration

### Environment Variables

```bash
# Dataset and model
export DATASET=gsm8k          # gsm8k, math, or mbpp
export MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Training hyperparameters
export BATCH_SIZE=2           # Per-device batch size
export GRAD_ACCUM=256         # Gradient accumulation steps
export LR=3e-6                # Learning rate
export WARMUP_STEPS=20        # LR warmup steps
export TOTAL_STEPS=400        # Total training steps
export ROLLOUTS=16            # Rollouts per problem (GRPO)

# DAPO-style asymmetric clipping
export CLIP_LOW=0.2           # Lower clip bound
export CLIP_HIGH=0.28         # Upper clip bound

# Generation parameters
export TEMPERATURE=0.7
export TOP_P=0.95
export TOP_K=50
```

### Key VeRL Config Parameters

```yaml
# GRPO-specific settings
algorithm:
  adv_estimator: grpo         # Use GRPO advantage estimator

actor_rollout_ref:
  rollout:
    n: 16                     # Multiple completions per prompt (key for GRPO)
  actor:
    clip_ratio_low: 0.2       # Asymmetric clipping (DAPO)
    clip_ratio_high: 0.28
    loss_agg_mode: token-mean # Loss aggregation
```

## Reward Functions

The reward functions match the GRAIL environment implementations:

### GSM8K (Total: 1.0)
- Correctness: 0.6 (exact numeric match)
- Strict format: 0.15 (numeric-only + no trailing)
- Thinking: 0.1 (has reasoning block)
- Answer: 0.1 (has answer block)
- No trailing: 0.05

### MATH (Total: 1.0)
- Correctness: 0.7 (multi-strategy validation)
- Answer format: 0.15 (has answer + minimal trailing)
- Thinking: 0.1
- No trailing: 0.05

### MBPP (Total: 1.0)
- Correctness: 0.7 (test pass rate)
- Syntax: 0.1 (code compiles)
- Format: 0.1 (has solution tags)
- Thinking: 0.1

## Comparison with TRL

| Feature | TRL | VeRL |
|---------|-----|------|
| Backend | HuggingFace Trainer | Ray + FSDP/Megatron |
| Inference | vLLM server mode | vLLM/SGLang integrated |
| Config | Python GRPOConfig | Hydra YAML |
| Distributed | torch.distributed | Ray |
| Checkpointing | HF Trainer | Custom + HDFS support |

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce `BATCH_SIZE` or enable gradient checkpointing
2. **Slow Generation**: Increase `actor_rollout_ref.rollout.gpu_memory_utilization`
3. **Ray Issues**: Check Ray cluster status with `ray status`
4. **MBPP Slow Rewards**: The code execution uses subprocess fallback without the execution pool. For faster MBPP training, initialize `CodeExecutionPool` before training.

### Important Notes

- **Data Format**: VeRL expects `prompt` as a list of message dicts (not pre-formatted strings). The data preparation handles this correctly.
- **VeRL Version**: This config is tested with VeRL as of January 2026. Some parameter names may differ in other versions.
- **Custom Chat Template**: The GRAIL-specific chat template with `<start_working_out>` tags is included in the system prompt.

### Memory Requirements

| Model Size | GPU Memory | Recommended Batch Size |
|------------|------------|------------------------|
| 1.5B | 24GB | 4-8 |
| 7B | 80GB | 1-2 |
| 14B | 80GB+ | 1 (with offloading) |

## References

- [VeRL Documentation](https://verl.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [DAPO Paper](https://arxiv.org/abs/2503.14476)
