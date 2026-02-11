# Sparsity Computation in GRPO Training Algorithm

## Overview
Sparsity in GRPO is computed at two levels:
1. **Parameter Change Sparsity** (`ParamChangeTracker`): Measures how many model parameters change after optimizer steps
2. **Sparse Quality Analysis** (`SparseQualityAnalyzer`): Evaluates how well sparse weight updates approximate full updates

---

## 1. Parameter Change Tracking (ParamChangeTracker)

### When it runs
- Every `measure_interval` optimizer steps (default: 100)
- Captures parameter snapshot at step t, measures diff at step t+k

### Pseudocode

```
class ParamChangeTracker:
    
    capture_snapshot(model):
        """Store current model weights on CPU"""
        snapshot = {}
        for each parameter in model:
            snapshot[param_name] = param.data.clone().cpu().float()
        return snapshot
    
    
    compute_metrics(model, snapshot):
        """Compute sparsity and param change statistics"""
        
        total_params = 0
        changed_counts = {} for each threshold in [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
        
        for each parameter in model:
            if param_name not in snapshot:
                continue
            
            # CRITICAL: Convert to float32 BEFORE subtraction (bfloat16 loses precision)
            current_fp32 = param.data.cpu().float()
            snapshot_fp32 = snapshot[param_name].float()
            
            # Compute per-element absolute delta
            delta = abs(current_fp32 - snapshot_fp32)
            n = delta.numel()
            total_params += n
            
            # Multi-threshold counting
            for threshold in thresholds:
                changed_counts[threshold] += count(delta > threshold)
            
            # Global aggregation
            abs_delta_sum += sum(delta)
            max_abs_delta = max(max_abs_delta, max(delta))
            relative_delta = delta / (abs(snapshot_fp32) + eps)
            relative_delta_sum += sum(relative_delta)
            
            # Sign flips: count params where sign changed
            if track_sign_flips:
                sign_flips += count((current_fp32 * snapshot_fp32) < 0)
            
            # Per-layer breakdown (optional)
            if track_per_layer:
                layer_idx = extract_layer_index(param_name)
                layer_stats[layer_idx].total_params += n
                layer_stats[layer_idx].changed_count += count(delta > primary_threshold)
            
            # Per-component breakdown (optional)
            if track_components:
                component = extract_component(param_name)  # e.g., "q_proj", "gate_proj"
                component_stats[component].total_params += n
                component_stats[component].changed_count += count(delta > primary_threshold)
        
        # Final metrics computation
        metrics.sparsity_ratio = (total_params - changed_at_primary) / total_params
            # ↑ unchanged / total = 1.0 means no changes, 0.0 means all params changed
        
        metrics.mean_abs_delta = abs_delta_sum / total_params
        metrics.max_abs_delta = max_abs_delta
        metrics.mean_relative_delta = relative_delta_sum / total_params
        metrics.sign_flip_ratio = sign_flip_count / total_params
        
        # Multi-threshold sparsity ratios
        for threshold in thresholds:
            unchanged_at_threshold = total_params - changed_counts[threshold]
            metrics.sparsity_by_threshold[threshold] = unchanged_at_threshold / total_params
        
        # Per-layer and per-component sparsity
        for layer_idx, stats in layer_stats.items():
            metrics.per_layer_sparsity[layer_idx] = (stats.total_params - stats.changed_count) / stats.total_params
        
        for component, stats in component_stats.items():
            metrics.per_component_sparsity[component] = (stats.total_params - stats.changed_count) / stats.total_params
        
        return metrics
```

---

## 2. Sparse Quality Analysis (SparseQualityAnalyzer)

### When it runs
- At the same interval as parameter tracking (every `measure_interval` optimizer steps)
- Uses the same snapshot as `ParamChangeTracker`

### Pseudocode

```
class SparseQualityAnalyzer:
    
    analyze(model, input_ids, attention_mask, snapshot):
        """Evaluate quality of sparse weight updates"""
        
        # Step 1: Compute weight deltas (full updates)
        deltas = {}
        total_params = 0
        
        for each parameter in model:
            if param_name not in snapshot:
                continue
            
            # Convert to float32 for precision
            current_fp32 = param.data.cpu().float()
            snapshot_fp32 = snapshot[param_name].float()
            deltas[param_name] = current_fp32 - snapshot_fp32
            total_params += deltas[param_name].numel()
        
        
        # Step 2: Get reference logits from current (fully updated) model
        with no_grad:
            logits_current = model(input_ids, attention_mask).logits
            logits_current_cpu = logits_current.cpu().float()
        
        
        # Step 3: For each sparsity threshold, compute quality metrics
        results = []
        
        for threshold in [0.0, 1e-8, 1e-6, 1e-4]:
            
            # Create magnitude-based mask: keep params with |delta| > threshold
            mask_dict = {}
            kept_params = 0
            
            for param_name, delta in deltas.items():
                mask = (abs(delta) > threshold)
                mask_dict[param_name] = mask
                kept_params += count(mask)
            
            kept_ratio = kept_params / total_params
            
            
            # Apply sparse delta: W_sparse = W_old + (delta * mask)
            logits_sparse = forward_pass_with_sparse_delta(
                model, snapshot, deltas, mask_dict, input_ids, attention_mask
            )
            logits_sparse_cpu = logits_sparse.cpu().float()
            
            
            # Create random baseline with same sparsity level
            random_mask_dict = {}
            for param_name, delta in deltas.items():
                # Random uniform, threshold to keep_ratio fraction
                rand = uniform_random_like(delta)
                random_mask_dict[param_name] = (rand < kept_ratio)
            
            logits_random = forward_pass_with_sparse_delta(
                model, snapshot, deltas, random_mask_dict, input_ids, attention_mask
            )
            logits_random_cpu = logits_random.cpu().float()
            
            
            # Compute quality metrics comparing sparse vs current
            metrics_magnitude = compute_quality_metrics(
                logits_current_cpu, logits_sparse_cpu, attention_mask
            )
            # Returns: (kl_div, cosine_similarity, mse, top1_agreement)
            
            # Compute quality metrics for random baseline
            metrics_random = compute_quality_metrics(
                logits_current_cpu, logits_random_cpu, attention_mask
            )
            
            results.append(ThresholdMetrics(
                threshold=threshold,
                kept_ratio=kept_ratio,
                kl_divergence=metrics_magnitude.kl_div,
                cosine_similarity=metrics_magnitude.cosine_sim,
                mse=metrics_magnitude.mse,
                top1_agreement=metrics_magnitude.top1_agree,
                kl_divergence_random=metrics_random.kl_div,
                cosine_similarity_random=metrics_random.cosine_sim,
                mse_random=metrics_random.mse,
                top1_agreement_random=metrics_random.top1_agree,
            ))
        
        return results


def forward_pass_with_sparse_delta(model, snapshot, deltas, mask_dict, input_ids, attention_mask):
    """Temporarily apply sparse delta, run forward pass, restore weights"""
    
    modified_params = set()
    
    # Apply sparse updates: W = W_old + (delta * mask)
    for param_name, param in model.named_parameters():
        if param_name not in snapshot:
            continue
        
        modified_params.add(param_name)
        sparse_delta = deltas[param_name] * mask_dict[param_name]
        snapshot_fp32 = snapshot[param_name].float()
        new_weight = snapshot_fp32 + sparse_delta
        param.data = new_weight.to(device=param.device, dtype=param.dtype)
    
    try:
        # Forward pass
        with no_grad:
            logits = model(input_ids, attention_mask).logits
    finally:
        # Restore original weights: W = W_old + delta (full update)
        for param_name, param in model.named_parameters():
            if param_name not in modified_params:
                continue
            
            snapshot_fp32 = snapshot[param_name].float()
            full_delta = deltas[param_name]
            original_weight = snapshot_fp32 + full_delta
            param.data = original_weight.to(device=param.device, dtype=param.dtype)
    
    return logits


def compute_quality_metrics(logits_current, logits_sparse, attention_mask):
    """Compute KL div, cosine similarity, MSE, top-1 agreement"""
    
    # KL Divergence: how much sparse logit distribution diverges from current
    log_probs_current = log_softmax(logits_current, dim=-1)
    log_probs_sparse = log_softmax(logits_sparse, dim=-1)
    kl_per_token = kl_div(log_probs_sparse, log_probs_current)
    kl_per_position = sum_over_vocab(kl_per_token)
    kl_mean = sum(kl_per_position * attention_mask) / sum(attention_mask)
    
    # Cosine Similarity: are logit vectors pointing in same direction?
    logits_current_norm = normalize(logits_current)
    logits_sparse_norm = normalize(logits_sparse)
    cosine_per_position = sum_over_vocab(logits_current_norm * logits_sparse_norm)
    cosine_mean = sum(cosine_per_position * attention_mask) / sum(attention_mask)
    
    # Mean Squared Error: raw logit differences
    mse_per_position = mean_over_vocab((logits_current - logits_sparse)^2)
    mse_mean = sum(mse_per_position * attention_mask) / sum(attention_mask)
    
    # Top-1 Agreement: do sparse and current agree on argmax token?
    top1_current = argmax(logits_current, dim=-1)
    top1_sparse = argmax(logits_sparse, dim=-1)
    agreement_per_position = (top1_current == top1_sparse)
    agreement_mean = sum(agreement_per_position * attention_mask) / sum(attention_mask)
    
    return (kl_mean, cosine_mean, mse_mean, agreement_mean)
```

---

## 3. Integration in GRPO Training Loop

### Pseudocode

```
class GRPOAlgorithm:
    
    train_epoch(model, ref_model, tokenizer, groups, optimizer, accelerator, monitor, window, config):
        
        # Initialize trackers
        param_tracker = ParamChangeTracker.from_config(config)
        sparse_analyzer = SparseQualityAnalyzer.from_config(param_tracker, config)
        
        optimizer_step_count = 0
        
        for each micro_batch in training_data:
            
            # Forward/backward as usual
            loss = compute_loss(...)
            optimizer.backward(loss / grad_accum_steps)
            grad_accum_counter += 1
            
            if grad_accum_counter >= grad_accum_steps:
                
                # Optimizer step
                optimizer.step()
                optimizer_step_count += 1
                
                # At measurement interval: capture/measure sparsity
                if param_tracker.should_measure(optimizer_step_count):
                    
                    if param_tracker.has_snapshot():
                        # Measure diff: current vs snapshot from k steps ago
                        param_metrics = param_tracker.compute_metrics(model)
                        await log_metrics(param_metrics, monitor, optimizer_step_count)
                        
                        # Analyze sparse quality using same snapshot
                        if sparse_analyzer.enabled:
                            sparse_metrics = sparse_analyzer.analyze(
                                model, input_ids, attention_mask
                            )
                            await log_sparse_metrics(sparse_metrics, monitor, optimizer_step_count)
                    
                    # Capture new snapshot for next measurement window
                    param_tracker.clear_snapshot()
                    param_tracker.capture_snapshot(model)
                
                grad_accum_counter = 0
```

---

## Key Insights

### Sparsity Definition
- **sparsity_ratio** = (unchanged_params / total_params)
  - 1.0 = no changes (all params unchanged)
  - 0.0 = all params changed
  - Typical values: 0.98-0.99 (very sparse updates)

### Multi-threshold Analysis
Tests multiple thresholds simultaneously to understand parameter change distribution:
- `0.0`: Truly unchanged (exact equality in float32)
- `1e-12`: Machine epsilon level changes (numerical noise)
- `1e-8`: Low precision threshold (suggests small learning rate)
- `1e-6`: Moderate threshold (typical for low LR training)

### Sparse Quality Metrics
Compares three distributions:
1. **Current (full update)**: Baseline model after all gradient updates
2. **Sparse update**: Model with only large-magnitude deltas applied
3. **Random baseline**: Same sparsity level but random parameter selection

**Metrics:**
- **KL Divergence**: Lower = sparse approximates current well
- **Cosine Similarity**: Higher = same logit direction
- **MSE**: Lower = smaller logit differences
- **Top-1 Agreement**: Higher = same predicted token (most important for LLMs)

### Memory-Efficient Design
- Snapshots stored on CPU to preserve GPU VRAM
- Deltas computed in float32 (bfloat16 cannot represent small changes like 1e-10)
- Streaming computation: one parameter at a time
- Temporary model patches instead of full clones
- Explicit cleanup with `torch.cuda.empty_cache()`

### Per-Layer and Per-Component Breakdown
Enables detailed analysis by:
- **Layer**: Extract layer index from parameter name (e.g., "model.layers.15...")
- **Component**: Identify projection type (q_proj, k_proj, gate_proj, etc.)

Helps identify which parts of model are learning (changing) vs. static.

---

## Typical Observed Patterns

| Metric | Healthy Training | Issues |
|--------|------------------|--------|
| sparsity_ratio | 0.98-0.99 | <0.90 (too much change) or 1.0 (no learning) |
| max_delta | 1e-4 to 1e-3 | 1e-10 (learning rate too low) or >1e-1 (instability) |
| sign_flip_ratio | <10% | >30% (parameter oscillation) |
| top1_agreement (sparse) | >95% | <80% (sparse approximation poor) |
| kl_div (sparse) | <0.1 | >1.0 (sparse very different from current) |


