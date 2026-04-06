# Incentive Mechanism

This document explains how miners are scored, how rollouts are verified, and what determines your weight on-chain. Understanding this is essential for maximizing your mining rewards.

---

## How Weights Work

Validators process the previous complete window (30 blocks, ~6 minutes), download miner rollout files from R2, verify a sample, score each miner, and set weights on-chain every 360 blocks (~1 hour, 12 windows).

The only metric that matters for your weight is **unique rollouts**. A rollout is unique if its completion tokens (the part after the prompt) have a distinct SHA-256 hash. Duplicate completions within or across problems do not count extra.

---

## Two Scoring Modes

The subnet switches between two scoring modes depending on network conditions. The active mode is hardcoded in `grail/shared/constants.py` and applies to all validators identically.

### Capped Mode (current default)

Each miner's unique rollouts are capped at `UNIQUE_ROLLOUTS_CAP` (2,500) per window, with a period cap of 2,500 x 12 = 30,000 over the rolling submission interval.

Scoring:

```
score = min(unique_rollouts, period_cap) ^ SUPERLINEAR_EXPONENT
weight = score / (period_cap ^ SUPERLINEAR_EXPONENT)
```

Producing more than the cap gives no additional weight. Producing less than the cap gives proportionally less, and the "missing" weight goes to burn rather than being redistributed to other miners.

The burn mechanism works as follows. 80% of total emissions always go to the burn UID (UID 0). The remaining 20% is split among miners proportionally to their cap achievement. If miners collectively produce below the cap, the unallocated portion of that 20% also goes to burn. For example, a miner at 50% of cap gets `0.20 x 0.50 = 0.10` (10%) of emissions, with the remaining 90% going to burn.

### Uncapped Mode

No per-window or period cap. Unique rollouts are scored directly:

```
score = unique_rollouts ^ SUPERLINEAR_EXPONENT
weight = score / sum(all_scores)
```

Producing more rollouts yields proportionally more weight. Burn is fixed at 80% regardless of production level, miners always split the remaining 20%.

### What This Means for Miners

- **Capped mode**: Target 2,500 unique rollouts per window. Going above yields nothing, going below costs you proportionally. Optimize for consistency across windows, not peak throughput.
- **Uncapped mode**: Maximize throughput. More unique rollouts = more weight, no ceiling.
- **Both modes**: The superlinear exponent (4.0) heavily amplifies differences. A miner with 2x the unique rollouts of another gets 2^4 = 16x the weight. This also makes sybil splitting unprofitable: splitting into k identities yields k^(1-4) of the total reward.

---

## Verification Pipeline

Every rollout a validator checks passes through 9 sequential validators. There are two severity levels:

- **Hard**: A single failure immediately rejects the rollout. The pipeline stops, no further checks run.
- **Soft**: A single failure does not reject the rollout. Instead, soft failures accumulate across all checked rollouts for that miner. If more than 51% of checked rollouts fail the soft check (`STOCHASTIC_CHECK_FAILURE_THRESHOLD = 0.51`), the entire miner is rejected for that window, same as a hard failure. Below that threshold, the miner passes normally.

| # | Validator | Severity | What It Checks |
|---|-----------|----------|----------------|
| 1 | Schema | Hard | Required fields are present, correct types |
| 2 | Token | Hard | Token IDs are valid integers within vocab range |
| 3 | Environment Prompt | Hard | Prompt matches the canonical prompt derived from `sha256(hotkey + window_hash + group_index)` |
| 4 | GRAIL Proof | Hard | Cryptographic proof binding tokens to the model's hidden states (see below) |
| 5 | Termination | Hard | Sequence ends with EOS probability >= 0.02 |
| 6 | Environment Evaluation | Hard | Generated code is executed and evaluated for correctness (Triton kernel: compiled and run against reference) |
| 7 | Reward | Hard | Miner's claimed reward matches validator-recomputed reward (rel_tol=0.02) |
| 8 | Logprob | Hard | Miner's log-probability values match recomputed values |
| 9 | Distribution | Soft | Heuristic checks on token probability distribution (detects wrong model, prefill tricks, bimodal distributions) |

Currently the only soft check is Distribution. It flags rollouts where chosen-token probabilities look inconsistent with sampling from the expected model (e.g., suspiciously low median probability, extremely low minimum probability, low Q10 in the initial window). A few flagged rollouts are tolerated, but if the majority fail, the miner is gated.

### Sampling Strategy

Validators do not check every rollout. The sampling works as follows:

- If a miner has <= 20 rollouts, all are checked.
- Otherwise, 10% of GRPO groups are sampled (capped at 64 rollouts per miner per window).
- Sampling is deterministic per miner per window (seeded by hotkey + window randomness).
- Results are extrapolated: `estimated_unique = unique_in_sample / sample_rate`.

Additionally, validators only check a subset of miners each window: 25% of active miners (min 2, max 35). A miner's score is extrapolated across windows where they were active but not checked.

---

## GRAIL Proof

The GRAIL proof is the core anti-cheating mechanism. It cryptographically binds a miner's token outputs to the actual hidden states produced by the shared model, preventing token fabrication or generation with a different model.

How it works:

1. The validator picks 32 random positions (`CHALLENGE_K = 32`) in the completion.
2. At each position, the model's hidden activations at the last layer are extracted.
3. The top 16 activations (`PROOF_TOPK = 16`) are selected and mapped into logarithmic buckets (8 buckets per sign, 16 total).
4. A random linear projection (sketch) is computed over the bucketed values modulo a Mersenne prime (2^31 - 1).
5. The validator recomputes this independently and compares against the miner's commitment.

Tolerance is position-dependent to account for floating-point drift in causal attention:

```
tolerance(pos) = 6000 + 5.0 * sqrt(pos)
```

Early positions have tight tolerance, later positions allow more drift. The bucketing (coarse quantization) makes the proof robust across different backends (vLLM, HuggingFace, SGLang) and CUDA versions while remaining cryptographically binding.

Both miner and validator use Flash Attention 2 (`ATTN_IMPLEMENTATION = "flash_attention_2"`) to ensure consistent, padding-invariant floating-point accumulation across batch sizes.

---

## Copycat Detection

Validators detect miners copying each other's rollouts by hashing completion tokens and comparing digests across miners.

Two scopes are checked:

- **Per-window**: If shared rollouts between two miners exceed 5% of the smaller miner's total, both are flagged.
- **Per-interval**: Same check accumulated across the 12-window submission interval, with a tighter 3% threshold.

Flagged miners get their metrics zeroed and the failure flag set. Any failure in the last 14 windows results in zero weight.

---

## Failure Gating

Any of these conditions results in zero weight for a miner:

- No rollout files submitted in any recent window.
- Any verification failure or copycat flag in the last 14 windows (`FAILURE_LOOKBACK_WINDOWS`).
- Never selected for validation (no data to extrapolate from).

This means a single bad window can zero your weight for ~1.4 hours. Consistent, valid submissions are rewarded over high-volume but unreliable ones.

---

## Constants Reference

| Constant | Value | Description |
|----------|-------|-------------|
| `WINDOW_LENGTH` | 30 blocks | ~6 minutes per window |
| `ROLLOUTS_PER_PROBLEM` | 16 | GRPO group size |
| `MAX_NEW_TOKENS` | 8192 | Maximum completion length |
| `UNIQUE_ROLLOUTS_CAP` | 2,500 | Per-window cap (capped mode) |
| `SUPERLINEAR_EXPONENT` | 4.0 | Weight amplification exponent |
| `GRAIL_BURN_PERCENTAGE` | 80% | Base emissions to burn UID |
| `GRAIL_BURN_UID` | 0 | Burn destination |
| `WEIGHT_ROLLING_WINDOWS` | 12 | Windows per submission interval |
| `CHALLENGE_K` | 32 | Proof challenge positions |
| `PROOF_TOPK` | 16 | Activations per position |
| `PROOF_NUM_BUCKETS` | 8 | Log-magnitude buckets per sign |
| `PROOF_SKETCH_TOLERANCE_BASE` | 6000 | Base FP tolerance |
| `PROOF_SKETCH_TOLERANCE_GROWTH` | 5.0 | Sqrt-growth tolerance factor |
| `SAMPLE_RATE` | 10% | Rollout spot-check fraction |
| `MAX_SAMPLES_PER_MINER` | 64 | Max rollouts checked per miner per window |
| `MINER_SAMPLE_RATE` | 25% | Fraction of miners checked per window |
| `FAILURE_LOOKBACK_WINDOWS` | 14 | Windows to look back for failures |
| `STOCHASTIC_CHECK_FAILURE_THRESHOLD` | 51% | Soft-failure fraction to reject miner |
| `COPYCAT_WINDOW_THRESHOLD` | 5% | Per-window overlap threshold |
| `COPYCAT_INTERVAL_THRESHOLD` | 3% | Per-interval overlap threshold |
