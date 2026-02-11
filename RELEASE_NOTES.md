# GRAIL v0.0.48 Release Notes

**Release Date:** February 7, 2026

### Highlights

- **Trust List via R2**: Validator now publishes per-window trust lists to R2, reducing miner selection lag from ~2.5 hours (on-chain) to ~6 minutes. Trainer auto-discovers the highest-stake validator's bucket with on-chain fallback.
- **Configurable Rollouts Cap**: New `GRAIL_UNIQUE_ROLLOUTS_CAP_ENABLED` toggle (default off) to disable the per-miner unique rollouts ceiling, allowing proportional rewards without underproduction burn.
