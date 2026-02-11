## Model Analysis Framework

A reusable, extensible framework for analyzing model behavior during training. Measures parameter changes, sparse update quality, and other training dynamics.

### Features

- **Parameter Change Tracking**: Magnitude, sparsity, per-layer, per-component statistics
- **Sparse Quality Analysis**: Can we get similar results with sparse updates? (LoRA feasibility)
- **Framework Agnostic**: Works with any training loop (GRPO, TRL, custom trainers)
- **Memory Efficient**: CPU-offloaded snapshots, streaming computation
- **Extensible**: Plugin architecture for custom metrics
- **Type-Safe**: Immutable primitives, clear interfaces

---

### Quick Start

```python
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    interval=100,  # Measure every 100 optimizer steps
    param_change_enabled=True,
    sparse_quality_enabled=True,
)

# Create manager
analyzer = ModelAnalysisManager.create(config)

# In training loop
for batch in dataloader:
    loss.backward()
    optimizer.step()

    # Analyze (returns {} if not at measurement interval)
    metrics = analyzer.on_optimizer_step(model, inputs=batch)
    if metrics:
        wandb.log(metrics)
```

---

### Architecture (Design 3: Layered Plugin)

```
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
│  (train_trl_grpo.py, grail GRPO, custom trainers)          │
└──────────────────┬──────────────────────────────────────────┘
                   │ uses
┌──────────────────▼──────────────────────────────────────────┐
│              ModelAnalysisManager                            │
│  • Lifecycle management (snapshots, intervals)              │
│  • Coordinates metric computers                             │
│  • Builder pattern: .add_metric()                           │
└──────────────────┬──────────────────────────────────────────┘
                   │ orchestrates
┌──────────────────▼──────────────────────────────────────────┐
│              MetricComputer (ABC)                            │
│  • Stateless, pure functions                                │
│  • compute(delta, snapshot, context) -> metrics             │
└──────────────────┬──────────────────────────────────────────┘
                   │ uses
┌──────────────────▼──────────────────────────────────────────┐
│          Primitives (Immutable)                              │
│  • ParameterSnapshot: W at time t                           │
│  • ParameterDelta: Δ = W_new - W_old                        │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1: Primitives**
- `ParameterSnapshot`: Immutable snapshot of model parameters (CPU, float32)
- `ParameterDelta`: Δ = W_new - W_old with built-in statistics

**Layer 2: Metric Computers** (Stateless)
- `ParameterChangeMetrics`: Magnitude, sparsity, per-layer/component stats
- `SparseQualityMetrics`: How well do sparse updates approximate full updates?
- Custom: Implement `MetricComputer` interface

**Layer 3: Manager** (Orchestration)
- `ModelAnalysisManager`: Lifecycle, snapshot management, coordination
- Factory method: `ModelAnalysisManager.create(config)`
- Builder pattern: `.add_metric(computer)`

---

### Directory Structure

```
grail/trainer/analysis/
├── __init__.py              # Public API exports
├── README.md                # This file
├── primitives.py            # ParameterSnapshot, ParameterDelta
├── config.py                # AnalysisConfig
├── manager.py               # ModelAnalysisManager
└── metrics/
    ├── __init__.py
    ├── base.py              # MetricComputer ABC, AnalysisContext
    ├── parameter_change.py  # ParameterChangeMetrics
    └── sparse_quality.py    # SparseQualityMetrics
```

---

### Configuration Options

```python
@dataclass
class AnalysisConfig:
    # Global
    interval: int = 100                    # Measure every N optimizer steps
    snapshot_device: str = "cpu"           # Where to store snapshots
    snapshot_dtype: str = "float32"        # Dtype for snapshots

    # Parameter Change Analysis
    param_change_enabled: bool = True
    param_change_thresholds: list[float] = [1e-8, 1e-6, 1e-4]
    param_change_per_layer: bool = False   # Per-layer breakdown
    param_change_track_components: bool = False  # Attention vs MLP

    # Sparse Quality Analysis
    sparse_quality_enabled: bool = True
    sparse_quality_thresholds: list[float] = [1e-8, 1e-6, 1e-4]
    sparse_quality_include_random: bool = True  # Random baseline

    # Future: Gradient, Momentum analysis
    gradient_enabled: bool = False
    momentum_enabled: bool = False
```

**Presets:**
- `AnalysisConfig.minimal()`: Param change only, high interval (low overhead)
- `AnalysisConfig.comprehensive()`: All metrics, frequent sampling

---

### Metrics Reference

#### Parameter Change Metrics (`param_change/*`)

**Global Statistics:**
- `norm_l2`: L2 norm of all parameter changes
- `norm_l1`: L1 norm (sum of absolute changes)
- `norm_max`: Maximum absolute change
- `mean`, `std`: Mean and stddev of changes

**Sparsity (multi-threshold):**
- `sparsity_at_1e-06`: Fraction of params with |Δ| ≤ 1e-6 (unchanged)
- `changed_ratio_at_1e-06`: Fraction changed (1 - sparsity)

**Per-Layer (if enabled):**
- `layer_N/mean_abs_delta`: Average change in layer N
- `layer_N/sparsity`: Sparsity in layer N

**Per-Component (if enabled):**
- `component/q_proj/mean_abs_delta`: Attention query projection
- `component/gate_proj/mean_abs_delta`: MLP gating

#### Sparse Quality Metrics (`sparse/*`)

For each threshold (e.g., `1e-06`):

**Question:** If we zero out all changes with |Δ| ≤ threshold, how close are outputs?

- `kl_at_1e-06`: KL divergence (lower = better, 0 = identical)
- `cosine_at_1e-06`: Cosine similarity (higher = better, 1 = identical)
- `mse_at_1e-06`: Mean squared error (lower = better)
- `top1_agree_at_1e-06`: Top-1 prediction agreement (higher = better, 1 = perfect)

**Sparsity:**
- `kept_ratio_at_1e-06`: Fraction of params kept (non-zero)
- `unchanged_ratio_at_1e-06`: Fraction dropped (1 - kept_ratio)

**Random Baseline:**
- `kl_at_1e-06_random`: Same metrics but with random mask at same sparsity
- If magnitude-based >> random, large changes are meaningful

---

### Integration Examples

#### 1. TRL GRPO Training

```python
from transformers import TrainerCallback
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig

class ModelAnalysisCallback(TrainerCallback):
    def __init__(self, analyzer: ModelAnalysisManager):
        self.analyzer = analyzer

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        inputs = kwargs.get("inputs")  # Optional

        metrics = self.analyzer.on_optimizer_step(model, inputs=inputs)
        if metrics:
            import wandb
            wandb.log({f"analysis/{k}": v for k, v in metrics.items()})

# Usage
config = AnalysisConfig(interval=100)
analyzer = ModelAnalysisManager.create(config)
callback = ModelAnalysisCallback(analyzer)

trainer = GRPOTrainer(..., callbacks=[callback])
trainer.train()
```

See `research/trl/analysis_integration_example.py` for full example.

#### 2. Custom Training Loop

```python
from grail.trainer.analysis import ModelAnalysisManager, AnalysisConfig

config = AnalysisConfig(interval=100)
analyzer = ModelAnalysisManager.create(config)

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()

        # Analyze
        metrics = analyzer.on_optimizer_step(model, inputs=batch)
        if metrics:
            print(f"Step {optimizer_step}: {metrics}")
```

#### 3. Custom Metrics

```python
from grail.trainer.analysis import MetricComputer, AnalysisConfig, ModelAnalysisManager

class GradientNormMetric(MetricComputer):
    def compute(self, **kwargs):
        context = kwargs.get("context")
        if context is None or context.model is None:
            return {}

        total_norm = sum(
            p.grad.norm(2).item() ** 2
            for p in context.model.parameters()
            if p.grad is not None
        )
        return {"gradient/total_norm": total_norm ** 0.5}

# Add to manager
config = AnalysisConfig(interval=50)
analyzer = (
    ModelAnalysisManager(config)
    .add_metric(GradientNormMetric())
)
```

---

### Testing

Run unit tests:
```bash
pytest tests/unit/trainer/analysis/ -v
```

---

### Design Rationale

#### Why Design 3 (Layered Plugin)?

**Compared to Monolithic (Design 1):**
- ✅ Single Responsibility Principle
- ✅ Independently testable components
- ✅ Extensible without modifying core

**Compared to Simple Strategy (Design 2):**
- ✅ Cleaner abstractions (immutable primitives)
- ✅ Better separation of concerns
- ✅ More maintainable long-term

**Compared to Event Hooks (Design 4):**
- ✅ Simpler control flow
- ✅ Easier to debug
- ✅ Not over-engineered for this use case

#### Key Principles

1. **Immutability**: Snapshots and deltas can't be modified
2. **Stateless Computation**: Metrics are pure functions
3. **Fail-Safe**: Errors in one metric don't break training
4. **Memory Efficiency**: CPU offloading, streaming computation
5. **Framework Agnostic**: No dependencies on specific trainers

---

### Migration from Old Code

**From `ParamChangeTracker` to `ParameterChangeMetrics`:**
```python
# Old
tracker = ParamChangeTracker(measure_interval=100)
if tracker.has_snapshot():
    metrics = tracker.compute_metrics(model)
tracker.capture_snapshot(model)

# New
config = AnalysisConfig(interval=100, param_change_enabled=True)
analyzer = ModelAnalysisManager.create(config)
metrics = analyzer.on_optimizer_step(model)  # Handles snapshot lifecycle
```

**From `SparseQualityAnalyzer` to `SparseQualityMetrics`:**
```python
# Old
analyzer = SparseQualityAnalyzer(tracker, enabled=True)
metrics = analyzer.analyze(model, input_ids, attention_mask)

# New
config = AnalysisConfig(interval=100, sparse_quality_enabled=True)
analyzer = ModelAnalysisManager.create(config)
metrics = analyzer.on_optimizer_step(model, inputs={"input_ids": ..., "attention_mask": ...})
```

---

### Future Extensions

Planned metric computers:

1. **GradientMetrics**: Gradient norm, variance, per-layer stats
2. **MomentumMetrics**: Optimizer state analysis
3. **ActivationMetrics**: Dead neurons, saturation
4. **LossLandscape**: Local curvature, sharpness

Adding is easy - just implement `MetricComputer` interface:
```python
class MyMetric(MetricComputer):
    def compute(self, delta, context, **kwargs):
        return {"my_metric/value": ...}

analyzer.add_metric(MyMetric())
```

---

### Performance Considerations

**Memory:**
- Snapshots stored on CPU in float32 (~2x model size)
- Deltas computed on-demand, not stored
- Sparse quality analysis requires temporary GPU memory for forward passes

**Computation:**
- Parameter change: ~0.1s for 1.5B model
- Sparse quality (3 thresholds): ~3s for 1.5B model (3 forward passes)
- Scales linearly with model size

**Recommendations:**
- Use `interval=100` or higher for large models
- Disable `sparse_quality` for frequent measurements
- Use `minimal()` config for production training

---

### Troubleshooting

**Metrics not appearing:**
- Check `interval` matches optimizer steps
- Ensure WandB is initialized before first measurement

**Memory errors:**
- Increase `interval` to reduce snapshot frequency
- Use `snapshot_dtype="float16"` (less precise but smaller)
- Disable `sparse_quality_enabled`

**Model parameter mismatch:**
- Snapshots capture parameter names; adding/removing layers breaks delta computation
- Call `analyzer.reset()` after architecture changes

---

### Citation

If you use this analysis framework in research, please cite:

```bibtex
@software{grail_analysis2024,
  title={Model Analysis Framework for Training Diagnostics},
  author={GRAIL Team},
  year={2024},
  url={https://github.com/yourusername/grail}
}
```
