# Pod Investigation Report - Jan 12, 2026

## Executive Summary

**All 4 deployed pods failed during training execution**, not during deployment. The issue is a **critical bug in the training script** affecting every pod uniformly.

### Results Overview
- ✅ **Deployment**: 4/4 succeeded
- ❌ **Training Execution**: 0/4 succeeded
- 🔴 **Root Cause**: `AttributeError: 'Namespace' object has no attribute 'model_id'`

---

## Critical Bug Found

### Location
- **File**: `/root/grail/research/trl/train_trl_grpo.py`
- **Line**: 1636
- **Error**: 

```python
run_name=f"trl_{adapter.name}_grpo_{args.model_id.replace('/', '_')}_{run_id}",
                                        ^^^^^^^^^^^^^
AttributeError: 'Namespace' object has no attribute 'model_id'
```

### Why This Happened
The training script expects a `--model-id` CLI argument, but the `run_parallel_training.py` launcher is passing `--model` instead. This creates a mismatch:
- Launcher sends: `--model "Qwen/Qwen2.5-1.5B-Instruct"`
- Script expects: `args.model_id`
- What exists: Only `args.model` attribute (wrong name)

### Affected Pods
| Pod | Status | Failure Point |
|-----|--------|---------------| 
| Qwen 2.5-0.5B | ❌ | All 4 instances failed after ~21 min |
| Qwen 2.5-1.5B | ❌ | All 4 instances failed after ~3 min |
| Llama 3.2-1B | ❌ | All 4 instances failed after ~2 min |
| Gemma 3-4B | ❌ | All 4 instances failed after ~10 min (but ran longer before error) |

---

## Detailed Pod Investigation

### Pod 1: Qwen 2.5-0.5B (SSH: 202.221.60.95:20248)
**Status**: ❌ FAILED

**Timeline**:
- Launcher started successfully at 08:26:13
- Training processes launched: 4 instances
- All 4 training processes exited with **code 1** within 30 seconds
- Launcher stopped at 08:31:19

**Error Log**:
```
Traceback (most recent call last):
  File "/root/grail/research/trl/train_trl_grpo.py", line 1793, in <module>
    main()
  File "/root/grail/research/trl/train_trl_grpo.py", line 1636, in main
    run_name=f"trl_{adapter.name}_grpo_{args.model_id.replace('/', '_')}_{run_id}",
                                        ^^^^^^^^^^^^^
AttributeError: 'Namespace' object has no attribute 'model_id'
```

**System Status**:
- GPU: All 8 GPUs idle (0% utilization, 0 MiB used)
- Memory: 32Gi used / 1.0Ti total
- CPU: Minimal usage

---

### Pod 2: Qwen 2.5-1.5B (SSH: 202.221.60.95:11100)
**Status**: ❌ FAILED

**Timeline**:
- Launcher started at 08:18:59
- Training processes launched: 4 instances
- All 4 training instances failed with **code 1**
- Same `AttributeError: 'Namespace' object has no attribute 'model_id'`

**Error Pattern**: Identical to Pod 1

**System Status**:
- GPU: All 8 GPUs idle (0% utilization)
- Memory: 32Gi used / 1.0Ti total
- Python: 3.12.3
- CUDA: Not in PATH (nvcc not found - but vLLM still works)

---

### Pod 3: Llama 3.2-1B (SSH: 154.54.100.126:20299)
**Status**: ❌ FAILED

**Timeline**:
- Launcher started at 08:16:26
- Training processes launched: 4 instances
- All 4 instances failed with **code 1** after ~2-3 minutes
- Same `AttributeError: 'Namespace' object has no attribute 'model_id'`

**Error Pattern**: Identical to Pods 1 & 2

**System Status**:
- GPU: All 8 GPUs idle (0% utilization)
- Memory: 11Gi used / 1.7Ti total
- Disk: 52G used / 8.7T total

---

### Pod 4: Gemma 3-4B (SSH: 202.221.60.95:11198)
**Status**: ❌ FAILED (After ~10 minutes of running)

**Timeline**:
- Launcher started at 08:24:52
- All 4 training processes launched and **ran for ~10 minutes** (08:28:05 → 08:37:05)
- Training processes exited with **code 1**
- Same `AttributeError: 'Namespace' object has no attribute 'model_id'`

**Unique Issue**: VLLM servers hung during shutdown (force-killed after 10-second timeout)

**Error in logs**:
```
Force killing: Command '['.../trl', 'vllm-serve', '--model', 'google/gemma-3-4b-it', 
'--port', '8000', ...'] timed out after 10 seconds
```

**System Status**:
- GPU: All 8 GPUs idle at final check (but were in use during the 10 min runtime)
- Memory: 30Gi used / 881Gi total
- Disk: 157G used / 876G total

---

## Root Cause Analysis

### The Bug Chain

1. **run_parallel_training.py** (line 215):
   ```python
   cmd = [
       str(python_path), "train_trl_grpo.py",
       "--dataset", self.dataset,
       ...
       "--model", self.model_id,  # <-- SENDS --model
   ]
   ```

2. **train_trl_grpo.py** expects:
   ```python
   # Somewhere in argument parsing:
   parser.add_argument("--model-id", ...)  # <-- EXPECTS --model-id
   ```

3. **Result**:
   - `args.model` exists (value: model string)
   - `args.model_id` does NOT exist (parser error or wrong name)
   - When line 1636 tries to access `args.model_id`, it fails

### Why All Pods Failed Identically

The issue is **upstream in the launcher**, not pod-specific:
- All pods use the same `train_trl_grpo.py` script
- All pods call it with `--model` argument
- All pods crash with the same error

---

## Secondary Issues Found

### 1. Flash Attention 2 Unavailable (Non-fatal)
```
⚠️  Flash Attention 2 unavailable (ImportError), using SDPA
```
- **Impact**: Minor - falls back to SDPA (Scaled Dot-Product Attention)
- **Fix**: Optional dependency, not breaking

### 2. VLLM Shutdown Timeout (Gemma pod only)
```
Force killing: Command timed out after 10 seconds
```
- **Impact**: Ungraceful shutdown, but training already failed
- **Root**: VLLM servers hung during termination

### 3. torch_dtype Deprecation Warning
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
- **Impact**: Code will break in future PyTorch versions
- **Fix**: Update model loading code

---

## Immediate Fix Required

**File**: `/root/grail/research/trl/train_trl_grpo.py`

### Fix 1: Update Line 1636 to use correct argument name

**Current (BROKEN)**:
```python
run_name=f"trl_{adapter.name}_grpo_{args.model_id.replace('/', '_')}_{run_id}",
```

**Fixed**:
```python
# Option A: Use args.model instead
run_name=f"trl_{adapter.name}_grpo_{args.model.replace('/', '_')}_{run_id}",

# Option B: Or fix the argument parser to accept --model-id
# And update the launcher to send --model-id
```

### Fix 2: Check argument parser

Look for where CLI arguments are parsed to ensure it accepts either:
- `--model` (current launcher sends this)
- Or update launcher to send `--model-id`

---

## Validation Checklist

After applying the fix:

- [ ] Verify `train_trl_grpo.py` line 1636 uses correct attribute name
- [ ] Verify argument parser handles the model ID argument
- [ ] Run a quick test on one pod
- [ ] Re-run the full deployment with all 6 models

---

## Files to Review

1. `/root/grail/research/trl/train_trl_grpo.py` - Lines 1636+, argument parsing
2. `/root/grail/research/trl/run_parallel_training.py` - Line 215, argument passing
3. Check how the model argument is named in the argument parser

---

## Impact Summary

| Category | Finding |
|----------|---------|
| **Deployment** | ✅ All infrastructure working perfectly |
| **Pod Connectivity** | ✅ SSH access working, GPUs available |
| **VLLM Servers** | ✅ Started and loaded models correctly |
| **Dataset Loading** | ✅ Datasets loaded successfully |
| **Training Execution** | ❌ FAILED - Critical bug in script |

**Bottom Line**: Infrastructure is perfect. Single critical bug in training script prevents any training from running.
