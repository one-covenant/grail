# Model Analysis Framework - Quick Start

## 30-Second Integration

```python
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig

# Setup (once)
config = AnalysisConfig(interval=100)
analyzer = ModelAnalysisManager.create(config)

# Training loop
for batch in dataloader:
    loss.backward()
    optimizer.step()

    metrics = analyzer.on_optimizer_step(model, inputs=batch)
    if metrics:
        wandb.log(metrics)
```

## What You Get

Every 100 optimizer steps, get ~20 metrics answering:

**"How much did parameters change?"**
- `param_change/norm_l2`: Overall magnitude
- `param_change/sparsity_at_1e-06`: % unchanged

**"Can we use sparse updates (LoRA)?"**
- `sparse/kl_at_1e-06`: How different are outputs? (lower = better)
- `sparse/kept_ratio_at_1e-06`: Only updating X% of params (e.g., 15%)
- If KL is low (<0.01) and kept_ratio is small (e.g., 0.15), **sparse training would work well!**

## Configuration Presets

```python
# Minimal: Fast, basic metrics
config = AnalysisConfig.minimal()  # interval=500, param_change only

# Comprehensive: All features
config = AnalysisConfig.comprehensive()  # interval=50, all metrics + per-layer

# Custom
config = AnalysisConfig(
    interval=100,
    param_change_enabled=True,
    sparse_quality_enabled=True,  # Requires batch inputs
)
```

## TRL Integration (1 minute)

**Step 1:** Create callback

```python
# File: analysis_callback.py
from transformers import TrainerCallback
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig

class ModelAnalysisCallback(TrainerCallback):
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        inputs = kwargs.get("inputs")  # Optional

        metrics = self.analyzer.on_optimizer_step(model, inputs=inputs)
        if metrics:
            import wandb
            wandb.log({f"analysis/{k}": v for k, v in metrics.items()})
```

**Step 2:** Add to your training script

```python
# In train_trl_grpo.py
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig
from analysis_callback import ModelAnalysisCallback

# Before creating trainer
config = AnalysisConfig(interval=100)
analyzer = ModelAnalysisManager.create(config)
callback = ModelAnalysisCallback(analyzer)

# Add to trainer
trainer = GRPOTrainer(
    ...,
    callbacks=[vllm_eval_callback, callback],  # Add here
)

trainer.train()  # Metrics logged automatically!
```

## Key Files

| File | Purpose |
|------|---------|
| `grail/trainer/analysis/README.md` | Full documentation |
| `research/trl/analysis_integration_example.py` | Complete examples |
| `grail/trainer/analysis/` | Source code |
| `tests/unit/trainer/analysis/` | Unit tests |

## Common Patterns

### Disable Expensive Metrics

```python
config = AnalysisConfig(
    interval=100,
    param_change_enabled=True,
    sparse_quality_enabled=False,  # Disable (requires 3 forward passes)
)
```

### Add Custom Metric

```python
from grail.trainer.analysis import MetricComputer

class MyMetric(MetricComputer):
    def compute(self, delta, **kwargs):
        return {"my_metric": delta.statistics()["norm_l2"]}

analyzer = ModelAnalysisManager(config).add_metric(MyMetric())
```

### Troubleshooting

**No metrics appearing?**
- Check `interval` setting (default: 100)
- First measurement at step `interval` just captures snapshot
- Metrics appear at step `2 * interval`

**Memory issues?**
- Increase `interval` (e.g., 500)
- Disable `sparse_quality_enabled`
- Use `snapshot_dtype="float16"`

**Metrics in WandB but not grouped?**
- Metrics are prefixed: `param_change/*`, `sparse/*`
- In WandB UI, use grouping by prefix

## What Gets Measured?

```
Step 0:     [no measurement]
Step 100:   [snapshot captured] → no metrics
Step 200:   [metrics computed] → {param_change/norm_l2: 0.123, ...}
Step 300:   [metrics computed] → {param_change/norm_l2: 0.156, ...}
```

Each measurement compares step N to step N-100.

## Interpreting Key Metrics

| Metric | Good Value | Bad Value | Meaning |
|--------|------------|-----------|---------|
| `sparse/kl_at_1e-06` | < 0.01 | > 0.1 | Low = sparse updates work well |
| `sparse/kept_ratio_at_1e-06` | 0.10-0.30 | > 0.90 | Lower = more sparsity possible |
| `param_change/sparsity_at_1e-06` | 0.80-0.95 | < 0.50 | High = most params unchanged |
| `param_change/norm_l2` | Steady or decreasing | Exploding | Overall learning magnitude |

**Golden Scenario for LoRA:**
- `sparse/kl_at_1e-06` < 0.01
- `sparse/kept_ratio_at_1e-06` ~ 0.15
- → You can update only 15% of weights and get nearly identical outputs!

---

**Need more?** See full README: `grail/trainer/analysis/README.md`
