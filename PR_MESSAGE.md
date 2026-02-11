## feat(scoring,trust): competitive incentivization improvements

### Summary
- **Trust list via R2**: Validator publishes per-window trust lists to R2, cutting miner selection lag from ~2.5h (on-chain) to ~6min. Trainer auto-discovers the highest-stake validator's bucket with on-chain fallback.
- **Configurable rollouts cap**: New `GRAIL_UNIQUE_ROLLOUTS_CAP_ENABLED` toggle (default off) to disable the per-miner unique rollouts ceiling, allowing proportional rewards without underproduction burn.

### Test plan
- [x] 14 unit tests for trust list publish/read/validation/fallback
- [x] Unit tests for rollouts cap toggle (enabled vs disabled)
- [x] E2E verified: trust lists published to R2, trainer probe reads correctly
