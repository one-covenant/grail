# Miner Debugging and Optimization

This guide explains how to efficiently debug and optimize your miner implementation.

## Table of Contents

- [Why Not Debug on Mainnet First?](#why-not-debug-on-mainnet-first)
- [Recommended Approaches](#recommended-approaches)
  - [Method 1: Local Validator (Simple)](#method-1-local-validator-simple)
  - [Method 2: Full Local Pipeline (Advanced)](#method-2-full-local-pipeline-advanced)
- [Log Analysis](#log-analysis)
- [Avoiding the Ban Period](#avoiding-the-ban-period)

---

## Why Not Debug on Mainnet First?

**The Problem:**
When your submission fails on mainnet, validators exclude you from sampling for ~14 windows (see grail/shared/constants.py:223). At 30 blocks/window × 12 seconds/block, that's **~84 minutes** before you can test again.

This creates a slow feedback loop:
1. Submit to mainnet → Rejected
2. Wait 84 minutes
3. Try new fix → Maybe rejected again
4. Wait another 84 minutes...

**The Solution:**
Run a local validator where you control the ban period and get instant feedback.

---

## Recommended Approaches

### Method 1: Local Validator (Simple)

**Fastest way to debug.** Run a validator locally to check your miner submissions in real-time.

**Setup:**

⚠️ **Important:** Don't use `--test-mode` for the validator. That flag only validates the trainer UID (grail/shared/constants.py:132), not miner UIDs.

**Recommended: Run with nohup and log files** (see run_all.sh:1):

```bash
source .venv/bin/activate

# Start miner with logging
CUDA_VISIBLE_DEVICES=0 nohup grail -vv mine > mine.log 2>&1 &

# Start validator with logging (no --test-mode)
CUDA_VISIBLE_DEVICES=1 nohup grail -vv validate > validate.log 2>&1 &
```

**Why nohup + log files?**
- Runs in background (won't stop if terminal disconnects)
- All output saved to searchable log files
- Easy to grep/search logs for specific UIDs or errors
- Can monitor multiple sessions simultaneously

**Monitor your submissions:**
```bash
# Watch live validation results
tail -f validate.log | grep "uid=<your-uid>"

# Search for specific outcomes
grep "Rejected" validate.log | grep "uid=<your-uid>"
grep "Accepted" validate.log | grep "uid=<your-uid>"

# View miner activity
tail -f mine.log | grep -E "(Generated|Uploaded)"
```

### Method 2: Full Local Pipeline (Advanced)

Run trainer + validator + miner locally for end-to-end testing.

**Setup:**

1. **Update constants** (grail/shared/constants.py:132):
   ```python
   TRAINER_UID = <your-trainer-uid>  # Change from 80
   ```

2. **Run all components** (requires 3 GPUs):

   ```bash
   # Terminal 1: Trainer (with --test-mode)
   CUDA_VISIBLE_DEVICES=0 grail -vv train --test-mode > train.log 2>&1 &

   # Terminal 2: Miner (wait ~5 min for trainer init)
   CUDA_VISIBLE_DEVICES=1 grail -vv mine > mine.log 2>&1 &

   # Terminal 3: Validator (with --test-mode)
   CUDA_VISIBLE_DEVICES=2 grail -vv validate --test-mode > validate.log 2>&1 &
   ```

**Why this is harder:**
- Requires multiple GPUs
- Trainer initialization takes ~5 minutes
- More complex orchestration

**When to use:**
- Testing full training pipeline
- Debugging trainer-validator integration
- Research and development

---

## Log Analysis

### Finding Your Submissions

```bash
# Real-time monitoring
tail -f validate.log | grep "uid=42"

# Count validations
grep -c "uid=42" validate.log

# Find rejections with reasons
grep "Rejected" validate.log | grep "uid=42"
```

### Common Rejection Reasons

| Error | Cause | Fix |
|-------|-------|-----|
| `Sketch tolerance exceeded` | Model weights don't match checkpoint | Load correct checkpoint window |
| `Signature verification failed` | Hotkey mismatch or signing error | Verify wallet config |
| `Sampling distribution check failed` | Token sampling distribution wrong | Check generation params (temperature, top_p) |
| `Reward mismatch` | SAT solution evaluation differs | Verify reward calculation logic |

### Log Verbosity

```bash
grail validate      # Standard
grail -v validate   # Verbose (recommended)
grail -vv validate  # Very verbose (debug trace)
```

See grail/shared/constants.py:136 for trace logging constants.

---

## Avoiding the Ban Period

### The Ban Mechanism

Validators exclude miners with failures from sampling for `FAILURE_LOOKBACK_WINDOWS` (grail/shared/constants.py:223):

```python
# Default: 14 windows ≈ 84 minutes
FAILURE_LOOKBACK_WINDOWS = 14
```

### Solution: Reduce Ban Time for Local Testing

When running a local validator, reduce the lookback window for faster iteration:

**Edit grail/shared/constants.py:223:**
```python
# Temporary: reduce to 1-2 windows for local testing
FAILURE_LOOKBACK_WINDOWS = 1  # ~6 minutes instead of 84
```

**Workflow:**
1. Set `FAILURE_LOOKBACK_WINDOWS = 1` in constants
2. Run local validator with reduced ban time
3. Test changes → See results in ~6 minutes
4. Fix issues and retry immediately
5. Once stable, revert to `FAILURE_LOOKBACK_WINDOWS = 14` for production

⚠️ **Remember:** Mainnet validators still use the default 14-window ban. Local testing lets you iterate faster, but your first mainnet submission still matters.

### Best Practice

1. **Develop locally** with `FAILURE_LOOKBACK_WINDOWS = 1`
2. **Fix all issues** found in local validation
3. **Revert to default** `FAILURE_LOOKBACK_WINDOWS = 14`
4. **Deploy to mainnet** once everything passes locally

This way, you avoid multiple mainnet bans and iterate 10-20x faster.

---

## Quick Reference

### For Most Miners (Method 1)

```bash
# 1. Reduce ban time for testing
# Edit grail/shared/constants.py:223
FAILURE_LOOKBACK_WINDOWS = 1

# 2. Run miner
grail -vv mine > mine.log 2>&1 &

# 3. Run validator (no --test-mode)
grail -vv validate > validate.log 2>&1 &

# 4. Monitor
tail -f validate.log | grep "uid=<your-uid>"

# 5. Fix, iterate, repeat
# When ready: revert FAILURE_LOOKBACK_WINDOWS = 14
```

### For Advanced Testing (Method 2)

```bash
# 1. Update TRAINER_UID in grail/shared/constants.py:132
# 2. Reduce FAILURE_LOOKBACK_WINDOWS in grail/shared/constants.py:223

# 3. Run trainer with --test-mode
CUDA_VISIBLE_DEVICES=0 grail -vv train --test-mode > train.log 2>&1 &

# 4. Wait for init, then run miner
CUDA_VISIBLE_DEVICES=1 grail -vv mine > mine.log 2>&1 &

# 5. Run validator with --test-mode
CUDA_VISIBLE_DEVICES=2 grail -vv validate --test-mode > validate.log 2>&1 &

# 6. Monitor all logs
tail -f validate.log | grep "uid="
```

---

## Support

- **GitHub Issues**: https://github.com/one-covenant/grail/issues
- **Discord**: https://discord.com/channels/799672011265015819/1354089114189955102
- **Validator Docs**: [validator.md](validator.md)
- **Miner Docs**: [miner.md](miner.md)
