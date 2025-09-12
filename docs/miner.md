# Miner Setup

This guide explains how to run a Grail miner. Miners generate GRPO rollouts with GRAIL proofs for SAT problems, upload them to object storage, and participate in decentralized scoring via validators.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Wallets](#wallets)
  - [Storage (R2/S3)](#storage-r2s3)
  - [Monitoring](#monitoring)
- [Running the Miner](#running-the-miner)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)

---

## Introduction

Grail miners:
- Connect to a Bittensor subnet and follow scoring windows.
- Derive per-window randomness from drand and the window’s block hash.
- Generate multiple GRPO rollouts per SAT problem using a HF model.
- Produce GRAIL proofs (Prover) binding tokens to the model and seed.
- Upload signed rollouts to object storage for validators to verify and score.

---

## Prerequisites

- Linux with a GPU driver (CUDA for NVIDIA) or CPU-only is acceptable if you can meet window timing
- Python (via `uv venv`) and Git
- Bittensor wallet (cold/hot) registered on the target subnet
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

Hardware guidance:
- Any hardware is acceptable as long as you can generate and upload within the window. The codebase has been primarily tested on NVIDIA A100/H100.
- Network bandwidth needs are modest; uploads are JSON rollouts.

---

## Quick Start

```bash
# Clone and enter
git clone https://github.com/tplr-ai/grail
cd grail

# Create venv and install
uv venv && source .venv/bin/activate

# Recommended (reproducible, uses lockfile and installs extras)
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run miner (uses Typer CLI)
grail -vv mine 
```

---

## Configuration

### Environment Variables

Set these in `.env` (see `.env.example` for full list and guidance):

- Network & subnet
  - `BT_NETWORK` (finney|test|custom)
  - `BT_CHAIN_ENDPOINT` (when `BT_NETWORK=custom`)
  - `NETUID` (target subnet id)
- Wallets
  - `BT_WALLET_COLD` (coldkey name)
  - `BT_WALLET_HOT` (hotkey name)
- Model & generation (read-only for miners in this release)
  - Do not override `GRAIL_MODEL_NAME` or `GRAIL_MAX_NEW_TOKENS`.
  - Validators assume the network default model and generation cap; changes may cause rollouts to be rejected.
  - You may adjust `GRAIL_ROLLOUTS_PER_PROBLEM` for throughput, but ensure you finish generation before the window upload buffer (last 2 blocks).
- Object storage (R2/S3)
  - `R2_BUCKET_ID`, `R2_ACCOUNT_ID`
  - Dual credentials (recommended):
    - Read-only: `R2_READ_ACCESS_KEY_ID`, `R2_READ_SECRET_ACCESS_KEY`
    - Write: `R2_WRITE_ACCESS_KEY_ID`, `R2_WRITE_SECRET_ACCESS_KEY`
- Monitoring
  - `GRAIL_MONITORING_BACKEND` (wandb|null)
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`

### Wallets

Create and fund a wallet, then register the hotkey on your subnet:

```bash
btcli wallet new_coldkey --wallet.name default
btcli wallet new_hotkey --wallet.name default --wallet.hotkey miner
btcli subnet register --wallet.name default --wallet.hotkey miner --netuid <NETUID> --subtensor.network <NETWORK>
```

Set `BT_WALLET_COLD` and `BT_WALLET_HOT` to these names in `.env`.

### Storage (R2/S3)

**Bucket requirement:** Name it the same as your account ID; set the region to ENAM.

Grail uses a dual-credential design:
- Write credentials stay local and are used by miners to upload.
- Read credentials are committed on-chain so validators can fetch your data.

Fill `R2_*` variables in `.env.example`.

### Monitoring

Set `GRAIL_MONITORING_BACKEND=wandb` and provide `WANDB_API_KEY` (or use `null`). Metrics include rollout counts, rewards, upload durations, and success rates.

---

## Running the Miner

From an activated venv with `.env` configured:

```bash
grail mine --use-drand   # default; mix drand + block-hash for randomness
# grail mine --no-drand  # fallback to block-hash only
```

Flags are minimal; most behavior is configured via `.env`. Increase verbosity with `-v` or `-vv`.

---

## Operations

High-level loop (see `grail/cli/mine.py`):

1. Load R2 credentials and initialize `GrailChainManager`; commit read credentials on-chain.
2. Connect to subtensor; compute `window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH`.
3. For the window:
   - Derive randomness: `sha256(block_hash + drand.randomness)` (or block hash only).
   - Generate SAT problems (difficulty ramps) and create GRPO batches via `SATRolloutGenerator`.
   - Use `Prover` to commit/open GRAIL proofs and package signed rollouts.
4. Upload the window’s rollouts to R2/S3 with write credentials.
5. Repeat on the next window.

Artifacts uploaded per rollout include:
- GRAIL commit (`tokens`, `s_vals`, signature, beacon)
- SAT problem metadata (seed, clauses, difficulty)
- GRPO data (reward, advantage, token logprobs, lengths, success)
- Miner signature over a challenge derived from seed/block/nonce

---

## Troubleshooting

- CUDA OOM or driver errors: reduce `GRAIL_MAX_NEW_TOKENS`; ensure drivers match CUDA runtime; periodically clear cache.
- No uploads: check `R2_*` variables and bucket permissions; verify network/firewall.
- Not receiving weights: ensure uploads succeed; validator will score the previous complete window.
- Drand failures: miner automatically falls back to block-hash; you can use `--no-drand`.
- Wallet not found: ensure `BT_WALLET_COLD`/`BT_WALLET_HOT` names exist in your `~/.bittensor/wallets`.

---

## Best Practices

- Keep model-related envs at network defaults; do not override `GRAIL_MODEL_NAME` or `GRAIL_MAX_NEW_TOKENS`.
- Reserve the final 2 blocks of each window for uploads; the miner does this automatically but avoid heavy generation near the end.
- Use `--use-drand` (default) for robust challenge derivation; fall back with `--no-drand` only if needed.
- Ensure R2 dual-credential setup: write locally, read credentials are committed on-chain by the miner.
- Monitor GPU memory (if using CUDA); the miner periodically empties cache, but size your rollouts to avoid OOM.
- Increase verbosity with `-vv` when diagnosing sampling, group sizes, or upload issues.

## Reference

- Miner entrypoint: `grail/cli/mine.py`
- Prover & protocol: `grail/grail.py`
- SAT environment & rewards: `grail/environments/sat.py`
- Rollout generation: `grail/mining/rollout_generator.py`
- Storage & uploads: `grail/infrastructure/comms.py`
- Credentials & chain: `grail/infrastructure/credentials.py`, `grail/infrastructure/chain.py`


