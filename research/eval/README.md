# MATH Benchmark Evaluation

Evaluate language models on the [Hendrycks MATH](https://github.com/hendrycks/math) dataset using [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with vLLM backend.

## Overview

This directory contains custom evaluation tasks for reasoning models that use the GRAIL format:
- `<start_working_out>` ... `</end_working_out>` for chain-of-thought reasoning
- `<SOLUTION>` ... `</SOLUTION>` for final answers

## Prerequisites

```bash
# Activate the vLLM environment
source /root/grail/tools/vllm-server/.venv/bin/activate
```

## Quick Start

### 1. Base Model (Standard Evaluation)

Standard 4-shot evaluation without reasoning format:

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096" \
    --tasks hendrycks_math \
    --num_fewshot 4 \
    --batch_size auto \
    --gen_kwargs "temperature=0,do_sample=False" \
    --output_path ./results/base_4shot \
    --log_samples
```

### 2. Reasoning Model (Custom Template)

For models trained with the GRAIL reasoning format, use the custom task with chat template:

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=/path/to/checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks hendrycks_math_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 4 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/reasoning_4shot
```

## Evaluation Configurations

### Base Model Configurations

| Config | Command Flags | Use Case |
|--------|--------------|----------|
| 0-shot | `--num_fewshot 0` | Zero-shot baseline |
| 4-shot | `--num_fewshot 4` | Standard MATH benchmark |

### Reasoning Model Configurations

| Config | Command Flags | Use Case |
|--------|--------------|----------|
| 0-shot | `--num_fewshot 0 --apply_chat_template` | Zero-shot with reasoning template |
| 4-shot multiturn | `--num_fewshot 4 --apply_chat_template --fewshot_as_multiturn` | **Recommended** - Few-shot as conversation |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--tasks hendrycks_math` | Standard MATH evaluation (7 subjects) |
| `--tasks hendrycks_math_grail` | Custom GRAIL reasoning format |
| `--include_path` | Path to custom task definitions |
| `--apply_chat_template` | Apply model's chat template |
| `--fewshot_as_multiturn` | Format few-shot examples as multi-turn conversation |
| `--think_end_token` | Token marking end of reasoning (extracts answer after this) |
| `--max_model_len` | Context length (use 8192+ for 4-shot) |
| `--log_samples` | Save per-sample outputs for analysis |

## Example Commands

### Evaluate GRAIL Checkpoint (Recommended)

```bash
cd /root/grail && source tools/vllm-server/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks hendrycks_math_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 4 \
    --batch_size auto \
    --log_samples \
    --output_path /root/grail/eval_results/grail_checkpoint
```

### Evaluate Base Model with Reasoning Template

First, prepare the base model with custom chat template:

```bash
# Download and patch the model (one-time setup)
python -c "
from huggingface_hub import snapshot_download
import json

# Download model
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='./models/Qwen2.5-1.5B-Instruct-reasoning')

# Patch tokenizer config with reasoning template
with open('./models/Qwen2.5-1.5B-Instruct-reasoning/tokenizer_config.json', 'r') as f:
    config = json.load(f)

config['chat_template'] = \"\"\"{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ 'You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and </end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>.' + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '' }}{% endif %}\"\"\"

with open('./models/Qwen2.5-1.5B-Instruct-reasoning/tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

Then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
    --model vllm \
    --model_args "pretrained=./models/Qwen2.5-1.5B-Instruct-reasoning,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks hendrycks_math_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 4 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/base_reasoning_4shot
```

### Run in Background

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks hendrycks_math_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --num_fewshot 4 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/grail_checkpoint \
    > eval.log 2>&1 &
echo "Started. PID: $!"
```

## Benchmark Results

| Model | Config | Accuracy |
|-------|--------|----------|
| Qwen2.5-1.5B-Instruct | 0-shot standard | 1.90% |
| Qwen2.5-1.5B-Instruct | 4-shot standard | 12.66% |
| Qwen2.5-1.5B-Instruct + reasoning template | 4-shot multiturn | 28.00% |
| grail_final_checkpoint | 4-shot multiturn | **30.34%** |

## Task Structure

```
tasks/hendrycks_math_grail/
├── _default_template.yaml    # Base config with reasoning format
├── hendrycks_math_grail.yaml # Task group definition
├── hendrycks_math_grail_algebra.yaml
├── hendrycks_math_grail_counting_and_prob.yaml
├── hendrycks_math_grail_geometry.yaml
├── hendrycks_math_grail_intermediate_algebra.yaml
├── hendrycks_math_grail_num_theory.yaml
├── hendrycks_math_grail_prealgebra.yaml
├── hendrycks_math_grail_precalc.yaml
└── utils.py                  # Answer extraction and comparison
```

## Reasoning Format

The custom chat template instructs the model to:

```
You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and </end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>.
```

Example output:
```
<start_working_out>
Let me solve this step by step...
The answer is 42.
</end_working_out>
<SOLUTION>42</SOLUTION>
```

The `think_end_token=</end_working_out>` argument tells the evaluator to extract the answer from text **after** this token, effectively using only the `<SOLUTION>` content for scoring.

## AIME 2024 Benchmark

AIME (American Invitational Mathematics Examination) is an extremely challenging competition math benchmark. The dataset contains 30 problems from AIME 2024.

### Running AIME Evaluations

**Base model (0-shot or 4-shot):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096,max_gen_toks=2048" \
    --tasks aime24 \
    --num_fewshot 0 \
    --batch_size auto \
    --gen_kwargs "temperature=0,do_sample=False,max_gen_toks=2048" \
    --log_samples \
    --output_path ./results/aime24_base
```

**Reasoning model (GRAIL checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192,enforce_eager=True" \
    --tasks aime24_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/aime24_grail
```

**Note**: AIME is extremely difficult - even 70B+ models typically achieve only 3-10% on AIME. Small models (1.5B) are expected to score near 0%.

## Pass@k Evaluation

For sampling-based evaluation with pass@k metrics:

### Best Practices

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `repeats` | 10 (for pass@5), 100 (for pass@100) | Number of samples per problem |
| `temperature` | 0.6 - 0.8 | Higher = more diversity |
| `top_p` | 0.95 | Nucleus sampling |
| `do_sample` | true | Required for sampling |

### Formula

pass@k = 1 - C(n-c, k) / C(n, k)

Where:
- n = total samples generated
- c = number of correct samples
- k = number of samples to consider

### Example: Pass@5 on MATH

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192" \
    --tasks hendrycks_math_pass_at_5 \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --batch_size auto \
    --log_samples \
    --output_path ./results/math_pass_at_5
```

### Key Differences from Greedy Evaluation

| Greedy (pass@1) | Sampling (pass@k) |
|-----------------|-------------------|
| `temperature=0` | `temperature=0.7` |
| `do_sample=false` | `do_sample=true` |
| `repeats=1` | `repeats=10+` |
| Single deterministic output | Multiple diverse outputs |

### Custom Pass@k Tasks

Create a task YAML with:
```yaml
repeats: 10  # Generate 10 samples per problem
generation_kwargs:
  do_sample: true
  temperature: 0.7
  top_p: 0.95
metric_list:
  - metric: !function utils.aggregate_pass_at_5
    aggregation: mean
    higher_is_better: true
```

## AMC 2023 Benchmark

AMC (American Mathematics Competition) is a high school math competition. The AMC 2023 dataset contains 40 problems.

### Running AMC Evaluations

**Base model (0-shot or 4-shot):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096,max_gen_toks=2048" \
    --tasks amc2023 \
    --include_path /root/grail/research/eval/tasks \
    --num_fewshot 0 \
    --batch_size auto \
    --gen_kwargs "temperature=0,do_sample=False" \
    --log_samples \
    --output_path ./results/amc2023_base
```

**Reasoning model (GRAIL checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192,enforce_eager=True" \
    --tasks amc2023_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/amc2023_grail
```

### AMC 2023 Results

| Model | Config | Accuracy |
|-------|--------|----------|
| Qwen2.5-1.5B-Instruct | 0-shot | 17.5% |
| Qwen2.5-1.5B-Instruct | 4-shot | 17.5% |
| grail_final_checkpoint | reasoning template | 17.5% |

## GSM8K Benchmark

GSM8K (Grade School Math 8K) is a dataset of 8.5K high-quality linguistically diverse grade school math word problems. The test set contains 1319 problems.

### Running GSM8K Evaluations

**Base model (0-shot or 4-shot):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096" \
    --tasks gsm8k \
    --num_fewshot 4 \
    --batch_size auto \
    --gen_kwargs "temperature=0,do_sample=False" \
    --log_samples \
    --output_path ./results/gsm8k_base
```

**Reasoning model (GRAIL checkpoint):**
```bash
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model vllm \
    --model_args "pretrained=/root/grail/grail_final_checkpoint,dtype=bfloat16,think_end_token=</end_working_out>,gpu_memory_utilization=0.9,max_model_len=8192,enforce_eager=True" \
    --tasks gsm8k_grail \
    --include_path /root/grail/research/eval/tasks \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto \
    --log_samples \
    --output_path ./results/gsm8k_grail
```
