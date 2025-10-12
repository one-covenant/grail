# Copycat Detection Test Suite Summary

## Overview
Comprehensive test suite for the copycat detection module using **realistic SAT prompts** from the conftest fixture and simulated completions.

## Test File Stats
- **File**: `tests/test_copycat_detection.py`
- **Lines**: 298 (including realistic SAT tests)
- **Test Count**: 12 tests
- **Test Classes**: 3
- **Execution Time**: ~2 seconds
- **Status**: ✅ All passing

## Test Coverage

### 1. TestCompletionDigest (2 tests)
Tests SHA-256 digest computation for completion tokens:
- `test_digest_basic_properties` - Deterministic, slices prompts, different completions
- `test_digest_edge_cases` - Missing metadata, empty tokens

### 2. TestCopycatTracker (7 tests)
Tests core overlap detection logic:
- `test_no_overlap_no_violations` - Unique completions → no flags
- `test_window_threshold_violation` - 60% overlap → flagged
- `test_window_below_threshold_no_violation` - 40% overlap → not flagged
- `test_interval_accumulation` - Accumulates across windows
- `test_interval_reset_clears_state` - Reset clears statistics
- `test_three_way_overlap` - Multiple miners sharing digests
- `test_asymmetric_counts` - Uses min(total_a, total_b) denominator

### 3. TestCopycatWithRealisticSAT (3 tests) ⭐ NEW
Tests with **actual SAT prompts from conftest**:
- `test_unique_sat_completions_no_violation` - Unique SAT solutions → no flags
- `test_copied_sat_completions_detected` - 80% copied SAT solutions → flagged
- `test_multiple_sat_prompts_mixed_overlap` - Different prompts, 30% overlap → not flagged

## Key Features

### Realistic SAT Integration
- ✅ Uses `sat_prompts` fixture from conftest (production-like prompts)
- ✅ Tokenizes with actual model tokenizer
- ✅ Generates realistic completion lengths (20-50 tokens)
- ✅ Simulates both copied and unique completions
- ✅ Tests across multiple SAT prompts with varying difficulties

### Helper Functions
```python
generate_sat_completion_tokens(prompt_tokens, seed, unique)
  → Generates realistic SAT completion tokens

make_rollout_data(prompt_tokens, seeds, unique_flags)
  → Batch generates rollout digests from seeds
```

### Test Scenarios Covered
1. **No overlap** - All miners have unique completions
2. **High overlap** - 60-80% shared completions (detected)
3. **Low overlap** - 30-40% shared completions (not detected)
4. **Interval accumulation** - Overlap accumulates across windows
5. **Three-way copying** - Multiple miners copying same completions
6. **Asymmetric counts** - Different total rollouts per miner
7. **Multiple SAT prompts** - Different problems across miners

## Production Alignment

### Uses Actual Mining Components
- SAT prompt generator (`generate_realistic_sat_prompt`)
- Tokenizer from `MODEL_NAME`
- Qwen chat template + system prompt
- Realistic completion characteristics

### Matches Validation Pipeline
- Digest computation matches production
- Tracker logic matches validator
- Threshold values (0.5 window, 0.75 interval)
- Pairwise overlap calculation

## Example Test Output

```bash
$ pytest tests/test_copycat_detection.py -v
======================== 12 passed in 2.23s =========================

TestCompletionDigest::test_digest_basic_properties PASSED
TestCompletionDigest::test_digest_edge_cases PASSED
TestCopycatTracker::test_no_overlap_no_violations PASSED
TestCopycatTracker::test_window_threshold_violation PASSED
TestCopycatTracker::test_window_below_threshold_no_violation PASSED
TestCopycatTracker::test_interval_accumulation PASSED
TestCopycatTracker::test_interval_reset_clears_state PASSED
TestCopycatTracker::test_three_way_overlap PASSED
TestCopycatTracker::test_asymmetric_counts PASSED
TestCopycatWithRealisticSAT::test_unique_sat_completions_no_violation PASSED ⭐
TestCopycatWithRealisticSAT::test_copied_sat_completions_detected PASSED ⭐
TestCopycatWithRealisticSAT::test_multiple_sat_prompts_mixed_overlap PASSED ⭐
```

## Benefits of Realistic SAT Tests

1. **Authentic**: Uses actual SAT prompts that miners see in production
2. **Comprehensive**: Tests digest computation with real tokenization
3. **Realistic**: Simulates actual copying patterns (partial overlap, etc.)
4. **Integration**: Verifies end-to-end flow from prompt → tokens → digest → detection
5. **Confidence**: Ensures copycat detection works with production data formats

## Quick Reference

### Running Tests
```bash
# All copycat tests
pytest tests/test_copycat_detection.py -v

# Only realistic SAT tests
pytest tests/test_copycat_detection.py::TestCopycatWithRealisticSAT -v

# Specific test
pytest tests/test_copycat_detection.py::TestCopycatWithRealisticSAT::test_copied_sat_completions_detected -v
```

### Adding New Tests
```python
def test_my_scenario(self, tracker, sat_prompt_tokens):
    # Generate completions
    digests = make_rollout_data(
        sat_prompt_tokens,
        seeds=[100, 101, 102],
        unique_flags=[True, True, False]
    )
    # Build miner rollouts
    miner_rollouts = {"miner1": (Counter(digests), len(digests))}
    # Ingest and assert
    w_cheat, *_ = tracker.ingest_window(100, miner_rollouts)
    assert ...
```
