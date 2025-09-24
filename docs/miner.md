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

- Linux with NVIDIA A100 GPU and CUDA drivers installed
- Python (via `uv venv`) and Git
- Bittensor wallet (cold/hot) registered on the target subnet
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

Hardware requirements:
- **NVIDIA A100 GPU is required** for the current version
- The codebase has been optimized and tested on NVIDIA A100
- GPU-agnostic verification is coming soon, which will enable support for other hardware configurations
- Network bandwidth needs are modest; uploads are JSON rollouts

---

## Quick Start

```bash
# Clone and enter
git clone https://github.com/one-covenant/grail
cd grail

# Create venv and install
uv venv && source .venv/bin/activate

# Recommended (reproducible, uses lockfile and installs extras)
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run miner
grail mine 

# Run miner in debug mode
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
  - Use `Qwen/Qwen3-4B-Instruct-2507` as the network default model in the first version.
  - Set `GRAIL_MAX_NEW_TOKENS=1024` (mandatory in first version).
  - Validators assume the network default model and generation cap; changes may cause rollouts to be rejected.
  - `GRAIL_ROLLOUTS_PER_PROBLEM` is fixed at 4 in this release and must not be changed. Throughput tuning should be done via other parameters; still ensure generation finishes before the upload buffer (last 2 blocks).
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

Public dashboards:
- **WandB Dashboard**: set `WANDB_ENTITY=tplr` and `WANDB_PROJECT=grail` to log to the public W&B project for detailed metrics and historical data. View at https://wandb.ai/tplr/grail.
- **Grafana Dashboard**: Real-time system logs, validator performance metrics, and network statistics are available at https://grail-grafana.tplr.ai/.

---

## Running the Miner

From an activated venv with `.env` configured:

```bash
grail mine   # default; mix drand + block-hash for randomness
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

- CUDA OOM or driver errors: Ensure you're using an NVIDIA A100 GPU; verify drivers match CUDA runtime; periodically clear cache.
- GPU not detected: Currently requires NVIDIA A100. Check `nvidia-smi` output to verify GPU availability.
- No uploads: check `R2_*` variables and bucket permissions; verify network/firewall.
- Not receiving weights: ensure uploads succeed; validator will score the previous complete window.
- Drand failures: miner automatically falls back to block-hash; you can use `--no-drand`.
- Wallet not found: ensure `BT_WALLET_COLD`/`BT_WALLET_HOT` names exist in your `~/.bittensor/wallets`.

---

## Best Practices

- **Use NVIDIA A100 GPU** for optimal performance; this is currently required for proper GRAIL proof generation and verification.
- Keep model-related envs at network defaults; do not override `GRAIL_MODEL_NAME` or `GRAIL_MAX_NEW_TOKENS`. Use `Qwen/Qwen3-4B-Instruct-2507` and set `GRAIL_MAX_NEW_TOKENS=1024` in the first version.
- Reserve the final 2 blocks of each window for uploads; the miner does this automatically but avoid heavy generation near the end.
- Use `--use-drand` (default) for robust challenge derivation; fall back with `--no-drand` only if needed.
- Ensure R2 dual-credential setup: write locally, read credentials are committed on-chain by the miner.
- Monitor GPU memory on your A100; the miner periodically empties cache, but size your rollouts to avoid OOM.
- Increase verbosity with `-vv` when diagnosing sampling, group sizes, or upload issues.
- Note: GPU-agnostic verification is under development and will expand hardware support in future releases.


## Support

For issues or questions:
- GitHub Issues: https://github.com/one-covenant/grail/issues
- Discord: https://discord.com/channels/799672011265015819/1354089114189955102