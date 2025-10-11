# Proof Testing Changes Summary

## Overview

Replaced hardcoded test prompts with realistic SAT prompts generated using the same production modules (`grail.environments.sat`) that miners use. This ensures test prompts match actual inference workloads.

## Changes Made

### 1. New Function: `generate_realistic_sat_prompt()` 
**Location:** `tests/proof_test_utils.py`

- **Purpose:** Generate production-quality SAT prompts for proof tests
- **Features:**
  - Uses `generate_sat_problem()` from `grail.environments.sat` (same as mining)
  - Applies Qwen chat template and system prompt when tokenizer provided
  - Deterministic: same seed → same prompt
  - Supports custom difficulty levels
  - Backward compatible (works without tokenizer)

**Signature:**
```python
def generate_realistic_sat_prompt(
    seed: str | None = None,
    difficulty: float = 0.5,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str
```

### 2. Session Fixture: `sat_prompts`
**Location:** `tests/conftest.py`

- **Purpose:** Pregenerate 5 SAT prompts for reuse across all proof tests
- **Scope:** Session-scoped for performance
- **Contents:**
  - 5 prompts with difficulties: 0.3, 0.5, 0.7, 0.5, 0.8
  - All include production chat template and system prompt
  - Deterministic seeds for reproducibility

### 3. Test File Updates

Replaced hardcoded prompts in:

#### `test_proof_cross_framework.py`
- ✅ Replaced 4 hardcoded prompts in `production_prompts` fixture
- ✅ Replaced determinism test prompt
- **Before:** 60+ lines of hardcoded SAT/math prompts
- **After:** 5 lines calling `generate_realistic_sat_prompt()`

#### `test_proof_model_mismatch.py`
- ✅ Replaced `prompt` fixture with realistic generator
- **Before:** Generic "Problem: Provide working..." string
- **After:** Realistic SAT prompt with proper difficulty

#### `test_proof_quantized_models.py`
- ✅ Replaced `prompt` fixture with realistic generator
- **Before:** Hardcoded SAT instance
- **After:** Realistic SAT prompt with proper difficulty

## Benefits

### ✅ Authenticity
- Prompts generated using **exact same code** as production mining
- Includes Qwen chat template: `build_qwen_chat_template()`
- Includes system prompt from `rollout_generator.py`
- SAT problems from `generate_sat_problem()` match miner workload

### ✅ Consistency
- Deterministic: same seed always produces same prompt
- Session fixture ensures all tests use same prompts
- No drift between test prompts and production prompts

### ✅ Maintainability
- **DRY:** No duplicate prompt strings across test files
- Single source of truth for prompt generation
- Easy to adjust difficulty, format, or problem types
- Changes to production prompt logic automatically propagate to tests

### ✅ Flexibility
- Optional tokenizer parameter for raw vs templated prompts
- Configurable difficulty levels (0.0 to 1.0)
- Deterministic or random seeds

## Usage Examples

### Using Pregenerated Fixtures
```python
def test_with_fixture(sat_prompts):
    prompt = sat_prompts[0]  # Easy prompt (difficulty 0.3)
    # ... test code
```

### Generating Custom Prompts
```python
def test_custom(tokenizer):
    # Production-like (with chat template)
    prompt = generate_realistic_sat_prompt(
        seed="my_test", 
        difficulty=0.8,
        tokenizer=tokenizer
    )
    
    # Raw SAT only (no template)
    raw = generate_realistic_sat_prompt(
        seed="my_test",
        difficulty=0.8
        # tokenizer=None
    )
```

## Prompt Format Example

With tokenizer:
```
You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>
Keep the reasoning succinct (≤25 steps, ≤500 tokens).<|im_end|>
SAT Problem:
SAT Problem (seed: test123...):
Variables: 6
Clauses:
  (NOT x1 OR x6 OR x3)
  (NOT x4 OR x3 OR NOT x6)
  (x6 OR x4 OR NOT x2)
  ...
<start_working_out>
```

Without tokenizer:
```
SAT Problem:
SAT Problem (seed: test123...):
Variables: 6
Clauses:
  (NOT x1 OR x6 OR x3)
  ...
Provide your final assignment between <SOLUTION></SOLUTION> as 
space-separated 0/1 values for x1..xN (e.g., <SOLUTION>0 1 0 1</SOLUTION>).
```

## Stats

- **Lines removed:** 85 (hardcoded prompts)
- **Lines added:** 127 (realistic generator + fixtures)
- **Net change:** +42 lines (most are reusable infrastructure)
- **Files modified:** 5
- **Hardcoded prompts eliminated:** ~8-10

## Verification

All changes pass:
- ✅ Ruff linting
- ✅ Type hints validation  
- ✅ Deterministic prompt generation
- ✅ Backward compatibility (tokenizer optional)
- ✅ Chat template application
- ✅ System prompt inclusion

## Migration Guide

For test authors:

1. **Replace hardcoded prompts:**
   ```python
   # OLD
   prompt = "Problem: SAT instance..."
   
   # NEW
   from tests.proof_test_utils import generate_realistic_sat_prompt
   prompt = generate_realistic_sat_prompt("my_seed", 0.5, tokenizer)
   ```

2. **Use fixtures when possible:**
   ```python
   def test_my_feature(sat_prompts):
       prompt = sat_prompts[1]  # Use pregenerated
   ```

3. **Pass tokenizer for production-like prompts:**
   - Fixtures already include tokenizer
   - Custom prompts should pass `tokenizer` parameter
   - Only omit for raw SAT problem text


