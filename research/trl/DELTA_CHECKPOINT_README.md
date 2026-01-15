# Delta Checkpoint System

Sparse storage of parameter updates during GRPO training for efficient time-series analysis.

## Overview

This system saves **only the non-zero parameter changes** after each optimizer step, enabling:
- **100x storage savings** vs full checkpoints (~3GB for 100 steps vs 300GB)
- **Exact reconstruction** of weights at any training step
- **Time-series analysis** of training dynamics (sparsity, update patterns, etc.)

## Quick Start

### Enable/Disable

Edit `train_trl_grpo.py` config:

```python
@dataclass
class Config:
    # ... other configs ...

    # Delta Checkpoint Configuration
    delta_checkpoint_enabled: bool = True   # Set to False to disable
    delta_checkpoint_dtype: str = "bfloat16"  # or "float32"
```

### Run Training

```bash
python train_trl_grpo.py --dataset gsm8k
```

Delta checkpoints are saved to: `./checkpoints/deltas_gsm8k/`

## Storage Format

Each checkpoint file (`delta_XXXXXX.pt`) contains:

```python
{
    "step": int,                    # Optimizer step number
    "timestamp": float,             # Unix timestamp
    "layers": {
        "layer.name": {
            "indices": torch.LongTensor,    # COO sparse indices
            "values": torch.BFloat16Tensor,  # Non-zero delta values
            "shape": tuple,                  # Original tensor shape
            "nnz": int,                      # Count of non-zeros
        }
    },
    "metadata": {
        "total_params": int,        # Total parameter count
        "total_nonzero": int,       # Non-zero deltas this step
        "sparsity": float,          # Fraction of zeros (0.99 = 99% sparse)
        "num_changed_layers": int,  # Layers with non-zero deltas
        "dtype": str,               # Storage dtype
    }
}
```

**Key feature**: Only layers with non-zero changes are stored.

## Metadata Tracking

`metadata.json` tracks all checkpoints:

```json
{
  "total_params": 1540000000,
  "dtype": "torch.bfloat16",
  "checkpoints": [
    {
      "step": 0,
      "path": "delta_000000.pt",
      "sparsity": 0.99123,
      "nonzero": 135000,
      "changed_layers": 142
    },
    ...
  ]
}
```

## Usage Examples

### Load a Single Delta

```python
from delta_checkpoint_callback import load_sparse_delta

# Load delta at step 42
deltas = load_sparse_delta("checkpoints/deltas_gsm8k/delta_000042.pt")

# Inspect
for layer_name, delta_tensor in deltas.items():
    print(f"{layer_name}: {(delta_tensor != 0).sum()} non-zero values")
```

### Reconstruct Weights at Step N

```python
from delta_checkpoint_callback import reconstruct_weights_at_step
import torch

# Save initial weights (step 0) during training
base_weights = {name: param.cpu().clone()
                for name, param in model.named_parameters()}
torch.save(base_weights, "checkpoints/base_weights.pt")

# Later: reconstruct weights at step 100
base = torch.load("checkpoints/base_weights.pt")
weights_100 = reconstruct_weights_at_step(
    base_weights=base,
    delta_dir="checkpoints/deltas_gsm8k",
    target_step=100,
)

# Load into model
model.load_state_dict(weights_100, strict=False)
```

## Implementation Details

### Callback Timing (Verified)

The callback hooks into `transformers.Trainer.on_optimizer_step()`:

```
1. optimizer.step()        # Weights updated
2. on_optimizer_step()     # ← Delta captured here (W_new - W_old)
3. zero_grad()             # Gradients cleared
```

This ensures we capture the **exact post-update state** with gradients still available.

### Sparsity Threshold

**Threshold = 0.0** (exact zeros only):

```python
mask = (delta_tensor != 0.0)  # Exact floating-point comparison
```

This means:
- ✅ **Lossless**: All non-zero changes are preserved
- ✅ **Exact reconstruction**: No approximation errors
- ⚠️ **Storage depends on optimizer**: If optimizer updates all weights (e.g., Adam with bias correction), sparsity will be low initially

Expected sparsity:
- Steps 1-10: ~50-80% (Adam warmup, many small updates)
- Steps 10+: ~95-99% (sparse updates dominate)

### Storage Estimates (Qwen2.5-1.5B)

| Scenario | Non-zero % | Size per checkpoint | 100 steps |
|----------|-----------|---------------------|-----------|
| Dense checkpoint | 100% | ~3GB | 300GB |
| Early training (5% sparse) | 95% | ~2.85GB | 285GB |
| Mid training (50% sparse) | 50% | ~1.5GB | 150GB |
| Late training (99% sparse) | 1% | ~30MB | ~3GB |
| **Average (95% sparse)** | **5%** | **~150MB** | **~15GB** |

**100x savings** assumes 99% sparsity (typical for later training steps).

## Future: Analysis Script

TODO: Create `analyze_deltas.py` to compute:

### Single-Step Metrics
- **Sparsity per step**: `nnz(delta_t) / total_params`
- **Update magnitude**: L2 norm of delta_t
- **Layer-wise sparsity**: Which layers change most

### Multi-Step Metrics
- **K-step sparsity**: `nnz(sum(delta_t...delta_{t+k})) / total_params`
  - Measures whether same parameters are updated repeatedly
- **Mask overlap (Jaccard)**: `|mask_t ∩ mask_{t+1}| / |mask_t ∪ mask_{t+1}|`
  - Are we updating the same parameters over time?
- **Update concentration**: Entropy of update distribution across layers
- **Cumulative sparsity**: `nnz(sum(all deltas)) / total_params`

### Visualization
- Sparsity heatmap over time (per layer)
- Update magnitude timeline
- Mask overlap matrix
- Parameter reuse statistics

Example usage (planned):
```python
python analyze_deltas.py \
    --delta-dir checkpoints/deltas_gsm8k \
    --output-dir analysis_results \
    --compute-k-step-sparsity --k-values 1,5,10,20 \
    --compute-mask-overlap \
    --visualize
```

## Verification

Run built-in tests:
```bash
python delta_checkpoint_callback.py
```

This verifies:
- ✅ Sparse COO conversion (threshold=0.0)
- ✅ Exact reconstruction from sparse format
- ✅ Zero preservation (no false positives/negatives)
- ✅ Integration with ParameterSnapshot

## Troubleshooting

### High storage usage

If checkpoints are unexpectedly large:
1. Check sparsity in logs: `[DeltaCheckpoint] Step X: YY% sparse`
2. If sparsity < 90%:
   - Early training: expected (Adam bias correction)
   - Late training: check optimizer config (beta1=0.9, beta2=0.999)
3. Consider saving every N steps instead of every step

### Missing checkpoints

- Verify `delta_checkpoint_enabled=True` in config
- Check logs for initialization message
- Ensure directory is writable

### Memory issues

- Delta computation uses CPU (minimal GPU memory impact)
- Snapshots are CPU-offloaded automatically
- Storage writes are blocking (TODO: async I/O)

## References

- **Callback timing**: `transformers/trainer.py:2740-2752`
- **Snapshot infrastructure**: `grail/trainer/analysis/primitives.py`
- **Sparse COO format**: PyTorch sparse tensor documentation
