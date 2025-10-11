# Proof Testing with Realistic SAT Prompts

This guide explains how to use the realistic SAT prompt generation for proof tests.

## Quick Start

### Using Pregenerated Prompts (Recommended)

The `sat_prompts` fixture in `conftest.py` provides 5 pregenerated prompts:

```python
def test_my_proof_feature(sat_prompts):
    # Use one of the pregenerated prompts
    prompt = sat_prompts[0]  # Easy problem
    # ... your test code
```

### Generating Custom Prompts

Use `generate_realistic_sat_prompt()` from `proof_test_utils`:

```python
from tests.proof_test_utils import generate_realistic_sat_prompt

def test_with_custom_prompt(tokenizer):
    # With chat template (production-like, recommended)
    prompt = generate_realistic_sat_prompt(
        seed="my_test_seed_123",
        difficulty=0.7,
        tokenizer=tokenizer  # Applies Qwen chat template + system prompt
    )
    
    # Raw SAT problem only (no chat template)
    raw_prompt = generate_realistic_sat_prompt(
        seed="my_test_seed_123",
        difficulty=0.7
        # tokenizer=None (default)
    )
    
    # Random prompt (each run different)
    random_prompt = generate_realistic_sat_prompt()
```

## Why Use This?

- **Authentic**: Uses the same SAT generator, chat template, and system prompt as production mining
- **Consistent**: Deterministic seeds ensure reproducible tests
- **DRY**: No hardcoded prompts scattered across test files
- **Realistic**: Tests match actual inference workloads including Qwen-style formatting

## Available Fixtures

### `sat_prompts` (session-scoped)

Provides 5 prompts with varying difficulties:
- Index 0: Easy (difficulty 0.3)
- Index 1: Medium (difficulty 0.5)  
- Index 2: Hard (difficulty 0.7)
- Index 3: Medium compact (difficulty 0.5)
- Index 4: Large hard (difficulty 0.8)

## Example Usage

```python
import pytest
from tests.proof_test_utils import (
    create_proof,
    verify_proof,
    generate_realistic_sat_prompt,
)

class TestGRAILProofs:
    def test_proof_verification(self, model, tokenizer, device, sat_prompts):
        """Test using pregenerated prompts."""
        randomness = "feedbeef1234"
        
        # Use pregenerated prompt (includes chat template + system prompt)
        proof = create_proof(
            model, tokenizer, sat_prompts[1], randomness, device
        )
        valid, diag = verify_proof(model, proof, randomness, device)
        assert valid

    def test_custom_difficulty(self, model, tokenizer, device):
        """Test with specific difficulty level."""
        # With chat template (production-like)
        prompt = generate_realistic_sat_prompt(
            seed="hard_test", 
            difficulty=0.9,
            tokenizer=tokenizer
        )
        proof = create_proof(model, tokenizer, prompt, "abc123", device)
        # ... test code
```

### What Gets Generated

When you use `generate_realistic_sat_prompt()` with a tokenizer, you get:

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
  ...
<start_working_out>
```

This matches **exactly** what miners generate in production!

