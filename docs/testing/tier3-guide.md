# Tier 3 Integration Testing Guide

## Overview

Tier 3 testing runs actual miner and validator processes with full debugging output and WandB integration. This guide covers using the `run_tier3_test.py` script for comprehensive integration testing.

## Prerequisites

- Configured `.env` file with valid credentials
- WandB API key set
- Sufficient system resources for running multiple models
- Models will be downloaded on first run

## Quick Start

### Basic Usage

```bash
# Run 2 miners with same model
python scripts/run_tier3_test.py --n-miners 2

# Run 3 miners with different models
python scripts/run_tier3_test.py --miners "Qwen/Qwen2-0.5B-Instruct,google/gemma-3-1b-it,Qwen/Qwen2-1.5B-Instruct"

# Custom validator model
python scripts/run_tier3_test.py --n-miners 2 --validator "Qwen/Qwen2-1.5B-Instruct"
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--miners` | Comma-separated list of models for miners | - |
| `--n-miners` | Number of miners with same model | - |
| `--miner-model` | Model for all miners (with --n-miners) | Qwen/Qwen2-0.5B-Instruct |
| `--validator` | Model for validator | google/gemma-3-1b-it |
| `--start-gpu` | Starting GPU index (useful to skip busy GPUs) | 0 |

## Testing Scenarios

### 1. Model Size Comparison

Test how different model sizes perform:

```bash
python scripts/run_tier3_test.py \
  --miners "Qwen/Qwen2-0.5B-Instruct,Qwen/Qwen2-1.5B-Instruct" \
  --validator "google/gemma-3-1b-it"
```

### 2. Homogeneous Network

Test with all nodes using the same model:

```bash
python scripts/run_tier3_test.py \
  --n-miners 3 \
  --miner-model "google/gemma-3-1b-it" \
  --validator "google/gemma-3-1b-it"
```

### 3. Stress Test

Run maximum miners (use with caution):

```bash
python scripts/run_tier3_test.py \
  --n-miners 4 \
  --miner-model "Qwen/Qwen2-0.5B-Instruct"
```

### 4. GPU Management

Skip busy GPUs and use specific GPU range:

```bash
# Skip GPU 0 which might be in use
python scripts/run_tier3_test.py \
  --n-miners 3 \
  --start-gpu 1

# Run 7 miners + 1 validator using all 8 GPUs
python scripts/run_tier3_test.py \
  --n-miners 7 \
  --miner-model "Qwen/Qwen2-0.5B-Instruct"
```

## Output and Monitoring

### Console Output

The script provides real-time output from all services with GPU assignments:

```
Starting miner-0 with model: Qwen/Qwen2-0.5B-Instruct on GPU 0
Starting miner-1 with model: google/gemma-3-1b-it on GPU 1
Starting validator with model: google/gemma-3-1b-it on GPU 2

GPU Assignments:
  miner-0: GPU 0
  miner-1: GPU 1
  validator: GPU 2

[miner-0] 2025-09-05 16:30:00 INFO Starting miner with model Qwen/Qwen2-0.5B-Instruct
[miner-1] 2025-09-05 16:30:02 INFO Starting miner with model google/gemma-3-1b-it
[validator] 2025-09-05 16:30:10 INFO Starting validation cycle
```

### WandB Integration

Each service creates its own WandB run:
- **Run names**: `tier3-miner-0-Qwen2-0.5B-Instruct`, `tier3-validator-gemma-3-1b-it`
- **Tags**: Automatically added for filtering (`tier3`, `miner`/`validator`, model name)
- **Project**: Uses the project from `.env` file

View runs at: https://wandb.ai/{your-entity}/{your-project}

## Environment Variables

The script uses all variables from `.env`, with these overrides per service:

- `GRAIL_MODEL_NAME`: Set per service based on command line
- `WANDB_RUN_NAME`: Unique name for each service
- `WANDB_TAGS`: Additional tags for filtering

## Debugging Tips

### 1. Verbose Output

The script automatically uses `-vv` flag for maximum verbosity. Look for:
- Model loading confirmations
- GRAIL protocol operations
- Network communications
- Error messages

### 2. Common Issues

**"Model not found"**
- Check model name spelling
- Ensure HF_TOKEN is set for private models
- First download may take time

**"Out of memory"**
- Use smaller models (0.5B instead of 1.5B+)
- Reduce number of miners
- Check GPU availability

**"WandB error"**
- Verify WANDB_API_KEY in .env
- Check internet connectivity
- Ensure WANDB_PROJECT exists

### 3. Log Analysis

While the test runs, you can:
1. Monitor console output for immediate issues
2. Check WandB for metrics and detailed logs
3. Look for patterns in miner/validator interactions

## Best Practices

1. **Start Small**: Begin with 2 miners before scaling up
2. **Model Selection**: Use smaller models for initial testing
3. **Resource Monitoring**: Watch system resources during tests
4. **Clean Shutdown**: Use Ctrl+C for graceful shutdown
5. **WandB Organization**: Use consistent naming for easy comparison

## Advanced Usage

### Custom Environment

Override specific variables:

```bash
GRAIL_WINDOW_LENGTH=5 GRAIL_MAX_NEW_TOKENS=512 \
  python scripts/run_tier3_test.py --n-miners 2
```

### Batch Testing

Create a test script:

```bash
#!/bin/bash
# test_models.sh

models=("Qwen/Qwen2-0.5B-Instruct" "google/gemma-3-1b-it")

for model in "${models[@]}"; do
  echo "Testing with $model"
  python scripts/run_tier3_test.py \
    --n-miners 2 \
    --miner-model "$model" \
    --validator "$model"
  sleep 60  # Run for 1 minute
  pkill -f "grail.*mine\|grail.*validate"
  sleep 5
done
```

## Integration with CI/CD

For automated testing:

```yaml
# .github/workflows/tier3-test.yml
- name: Run Tier 3 Test
  timeout-minutes: 10
  run: |
    python scripts/run_tier3_test.py \
      --n-miners 2 \
      --miner-model "Qwen/Qwen2-0.5B-Instruct" &
    TEST_PID=$!
    sleep 300  # Run for 5 minutes
    kill $TEST_PID
```

## Next Steps

After running Tier 3 tests:
1. Analyze WandB runs for performance metrics
2. Compare different model configurations
3. Identify bottlenecks or issues
4. Adjust parameters in `.env` as needed
5. Document findings for team review
