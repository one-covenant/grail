# Validator Setup

This guide explains how to run a Grail validator. Validators verify miners’ SAT rollouts with GRAIL proofs, compute miner scores over rolling windows, and set weights on-chain.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Wallets](#wallets)
  - [Storage (R2/S3)](#storage-r2s3)
  - [Monitoring](#monitoring)
- [Running the Validator](#running-the-validator)
- [Operations](#operations)
  - [Windowing](#windowing)
  - [Discovery & Download](#discovery--download)
  - [Verification](#verification)
  - [Scoring & Weights](#scoring--weights)
  - [Publishing](#publishing)
- [Troubleshooting](#troubleshooting)
- [Reference](#reference)

---

## Introduction

Grail validators:
- Track Bittensor blocks and operate on complete windows.
- Fetch miners’ window files from object storage using per-miner read credentials committed on-chain.
- Verify each rollout with the Verifier (GRAIL proof + SAT reconstruction + signature checks).
- Score miners by unique successful solutions and estimated valid rollouts.
- Normalize and set weights on-chain each window.

---

## Prerequisites

- Linux, Python via `uv venv`, Git
- Bittensor wallet (cold/hot) registered on the target subnet
- Cloudflare R2 (or S3-compatible) bucket and credentials
- Optional: WandB account for monitoring

Hardware: CPU-only is fine. A GPU can speed up verification if the model forward is used more heavily, but the default verifier is optimized for practicality.

---

## Quick Start

```bash
# Clone and enter
git clone https://github.com/tplr-ai/grail
cd grail

# Create venv and install
uv venv && source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run validator (Typer CLI)
grail -vv validate --test-mode      # validate own files only
# grail -vv validate --no-test-mode # validate all miners on the subnet
```

---

## Configuration

### Environment Variables

Set these in `.env` (see `.env.example`):

- Network & subnet
  - `BT_NETWORK` (finney|test|custom)
  - `BT_CHAIN_ENDPOINT` (when `BT_NETWORK=custom`)
  - `NETUID` (target subnet id)
- Wallets
  - `BT_WALLET_COLD` (coldkey name)
  - `BT_WALLET_HOT` (hotkey name)
- Model (read-only for validators in this release)
  - `GRAIL_MODEL_NAME` (default network model used by Verifier)
- Object storage (R2/S3)
  - `R2_BUCKET_ID`, `R2_ACCOUNT_ID`
  - Dual credentials (recommended):
    - Read-only: `R2_READ_ACCESS_KEY_ID`, `R2_READ_SECRET_ACCESS_KEY`
    - Write: `R2_WRITE_ACCESS_KEY_ID`, `R2_WRITE_SECRET_ACCESS_KEY`
- Monitoring
  - `GRAIL_MONITORING_BACKEND` (wandb|null) and WandB fields

### Wallets

Use a registered wallet. Set `BT_WALLET_COLD`/`BT_WALLET_HOT` to the names you created with `btcli`.

### Storage (R2/S3)

Validators load local write credentials and use miners’ read credentials fetched from chain to download their files. See `docs/r2_credentials_design.md`.

### Monitoring

Set `GRAIL_MONITORING_BACKEND=wandb` to enable metrics; otherwise use `null`.

---

## Running the Validator

```bash
grail validate --use-drand      # default; use drand-derived challenge randomness
# grail validate --no-drand     # fallback to block-hash only
grail -vv validate --test-mode  # verify only self (useful locally)
```

Verbosity `-v/-vv` increases logging. Most behavior is configured via `.env`.

---

## Operations

The implementation lives in `grail/cli/validate.py`.

### Windowing

- Determine current block; compute windows of length `WINDOW_LENGTH`.
- Process the previous complete window: `target_window = current - WINDOW_LENGTH`.

### Discovery & Download

- For each hotkey (test mode: just self; otherwise all active hotkeys in metagraph):
  - Build expected file: `grail/windows/{hotkey}-window-{target_window}.json`.
  - Retrieve miner’s read credentials from chain (`GrailChainManager`).
  - Check existence and download with read creds; fallback to local creds if needed.

### Verification

For each downloaded inference:
- Required fields: `window_start`, `nonce`, `sat_seed`, `block_hash`, `commit`, `proof`, `challenge`, `hotkey`, `signature`.
- Window and block-hash must match the validator’s `target_window` and hash.
- Nonce must be unique within a miner’s window file.
- Signature check: hotkey verifies `challenge = sat_seed + block_hash + nonce`.
- SAT seed must equal `{wallet_addr}-{target_window_hash}-{nonce}` (for GRPO rollouts, base seed `{wallet_addr}-{target_window_hash}-{rollout_group}` is reconstructed).
- Challenge randomness for GRAIL proof: mix drand randomness (current round) with `target_window_hash`; fallback to hash-only on failures.
- Verifier (`grail/grail.py`) checks token validity, sketch proof, model identity, and SAT reconstruction.

Sampling and batching:
- If total rollouts ≤ MAX_SAMPLES_PER_MINER → verify all.
- Else sample complete GRPO groups (~10%) with early stopping if failures exceed threshold.

### Scoring & Weights

Per miner over recent 3 windows, compute:
- `unique` unique successful solutions
- `successful` count of successful rollouts
- `valid` count (or estimated valid) verified

Base score (in [0,1]):
```
base = 0.6 * min(1, unique/10) + 0.0 * min(1, successful/20) + 0.4 * min(1, valid/50)
```
Apply superlinear curve (from `SUPERLINEAR_EXPONENT`, default 1.5):
```
score = base ** SUPERLINEAR_EXPONENT
```
Normalize to weights across miners; set on-chain with `set_weights`.

### Publishing

- Upload all valid rollouts to R2/S3 for training (`upload_valid_rollouts`).
- Optionally export to a public Hugging Face dataset (`upload_to_huggingface`).

---

## Troubleshooting

- No files found: ensure miners committed read creds on-chain; verify bucket name and permissions.
- Frequent verification failures: check model alignment (`GRAIL_MODEL_NAME`), drand connectivity; fall back with `--no-drand`.
- Weight setting fails: wallet funding/permissions and network connectivity.
- Test mode confusion: `--test-mode` validates only your own files; disable for production.

---

## Reference

- Validator entrypoint: `grail/cli/validate.py`
- Verifier & protocol: `grail/grail.py`
- SAT environment & rewards: `grail/environments/sat.py`
- Storage & downloads: `grail/infrastructure/comms.py`
- Credentials & chain: `grail/infrastructure/credentials.py`, `grail/infrastructure/chain.py`


