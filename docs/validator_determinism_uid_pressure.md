### Validator determinism and UID-pressure hardening

This change set improves validator consistency across nodes and reduces incentives to split work across many UIDs ("UID pressure").

Key decisions and rationale:

- Deterministic sampling of rollouts
  - What: Spot-check GRPO groups using a `random.Random` instance seeded by `wallet_addr + target_window_hash` so all validators select the same groups.
  - Why: Unseeded RNG leads to divergent samples and inconsistent weights across validators.

- Deterministic challenge randomness with optional drand
  - What: Default to a deterministic `sha256(window_hash)` challenge. If `--use-drand` is enabled, mix drand using a round anchored by a hash of the window hash, with deterministic fallback.
  - Why: Using wall-clock time causes validators to disagree. Determinism is required for reproducible verification results.

- Offload heavy verification to a thread
  - What: Run `verifier.verify_rollout(...)` via `asyncio.to_thread`.
  - Why: Avoid blocking the event loop with CPU-bound verification, preventing watchdog timeouts and improving responsiveness.

- Weighting uses estimated_valid and adds success signal
  - What: Aggregate `estimated_valid` over the last 3 windows to reflect total validated volume with sampling. Include a non-zero success coefficient in base score: `0.6*unique + 0.2*success + 0.2*valid` before superlinear exponent.
  - Why: Better reflects throughput under sampling and applies pressure toward successful and unique solutions while still discouraging UID splitting via the superlinear curve.

- Zero-sum fallback
  - What: If all scores are zero, fall back to uniform weights over active miners.
  - Why: Prevents submitting all-zero weights which some chains reject and avoids degenerate distributions.

- State pruning
  - What: Keep only the last 3 windows of per-miner metrics in memory.
  - Why: Bound memory growth over long validator uptimes.

- Safer defaults
  - What: `--test-mode` defaults to False; `--use-drand` defaults to False.
  - Why: Production-safe defaults and deterministic behavior by default.

Impact:

- All validators should derive identical samples and challenge randomness for a given window, aligning verification and weight outcomes.
- UID-splitting remains disincentivized due to superlinear scoring, now with clearer success and volume signals.
- Runtime stability improves by avoiding event loop blocking and by pruning state.

Operational notes:

- To enable drand mixing deterministically, pass `--use-drand`. The code handles drand unavailability by falling back deterministically to `sha256(window_hash)`.
- If any future proof scheme needs to change binding assumptions, extend the signed challenge to include a commit hash.


