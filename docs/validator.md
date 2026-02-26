# Validator Setup

This guide explains how to run a Grail validator. Validators verify miners’ rollouts with GRAIL proofs, compute miner scores over rolling windows, and set weights on-chain. The active environment is set network-wide (currently **Triton Kernel**).

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
- Verify each rollout with the Verifier (GRAIL proof + environment evaluation + signature checks).
- Score miners by unique successful solutions and estimated valid rollouts.
- Normalize and set weights on-chain each window.

---

## Prerequisites

- Linux with NVIDIA GPU drivers installed
- Docker and Docker Compose installed
- **2x NVIDIA GPUs** (A100 or H100 recommended) for Triton kernel evaluation
  - GPU 0: Model inference / proof verification
  - GPU 1: Triton kernel compilation + correctness evaluation
- At least 40GB RAM recommended for optimal performance
- Bittensor wallet (cold/hot) registered on the target subnet
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

For detailed hardware specifications, see [`compute.min.yaml`](../compute.min.yaml).

Hardware requirements:
- **Linux with 2x NVIDIA GPUs** (A100/H100 recommended for Triton JIT compatibility)
  - GPU 0: Model inference and proof verification
  - GPU 1: Triton kernel evaluation (compilation + correctness checks)
- At least 40GB RAM recommended for optimal performance

---

## Quick Start

### Docker (recommended)

```bash
# Clone and enter
git clone https://github.com/one-covenant/grail
cd grail

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run validator with Docker Compose
docker compose --env-file .env -f docker/docker-compose.validator.yml up -d
```

### Native (without Docker)

```bash
# Clone and enter
git clone https://github.com/one-covenant/grail
cd grail

# Create venv and install
uv venv && source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run validator
grail validate

# Run validator in debug mode
grail -vv validate

# Run in test mode (validates only your own files)
grail -vv validate --test-mode
```

---

## Configuration

### Environment Variables

Set these in `.env` (see `.env.example`). This file will be used with Docker Compose:

- Network & subnet
  - `BT_NETWORK` (finney|test|custom)
  - `BT_CHAIN_ENDPOINT` (when `BT_NETWORK=custom`)
  - `NETUID` (target subnet id)
- Wallets
  - `BT_WALLET_COLD` (coldkey name)
  - `BT_WALLET_HOT` (hotkey name)
- Model (dynamically loaded from R2 checkpoints)
  - The model is loaded automatically from R2 checkpoints and evolves through training
  - Validators automatically load the appropriate checkpoint for each validation window
  - Maximum new tokens is fixed at 8192 (hardcoded constant `MAX_NEW_TOKENS`)
  - Rollouts per problem is fixed at 16 (hardcoded constant `ROLLOUTS_PER_PROBLEM`)
  - Models are shared via R2 storage and updated by the trainer after each window
  - No manual model configuration required - checkpoints are loaded automatically
- Object storage (R2/S3)
  - `R2_BUCKET_ID`, `R2_ACCOUNT_ID`
  - Dual credentials (recommended):
    - Read-only: `R2_READ_ACCESS_KEY_ID`, `R2_READ_SECRET_ACCESS_KEY`
    - Write: `R2_WRITE_ACCESS_KEY_ID`, `R2_WRITE_SECRET_ACCESS_KEY`
- Monitoring
  - `GRAIL_MONITORING_BACKEND` (wandb|null)
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`
- Kernel evaluation (triton_kernel environment)
  - `GRAIL_GPU_EVAL` (true|false, default: true) — enable GPU-based kernel correctness evaluation
  - `KERNEL_EVAL_GPU_IDS` (default: 1) — physical GPU index for kernel eval (GPU 0 runs the model)
  - `KERNEL_EVAL_BACKEND` (persistent|subprocess|basilica, default: persistent) — evaluation backend. `persistent` reuses the CUDA context (~40x faster), `subprocess` isolates each eval, `basilica` uses Basilica cloud GPU workers (not yet implemented)
  - `KERNEL_EVAL_TIMEOUT` (default: 60) — per-kernel evaluation timeout in seconds
- Optional
  - `HF_TOKEN`, `HF_USERNAME` (for Hugging Face dataset publishing)

### Wallets

Use a registered wallet. Set `BT_WALLET_COLD`/`BT_WALLET_HOT` to the names you created with `btcli`.

### Storage (R2/S3)

**Bucket requirement:** Name it the same as your account ID; set the region to ENAM.

Validators load local write credentials and use miners’ read credentials fetched from chain to download their files.

### Monitoring

Set `GRAIL_MONITORING_BACKEND=wandb` to enable metrics; otherwise use `null`.

#### Grafana + Loki Logging (Optional)

For centralized logging, use Promtail to ship logs to Loki. This approach is more robust than in-process log shipping and prevents stalling under network pressure.

**Promtail is disabled by default.** To enable it, use Docker Compose profiles.

**Setup:**
1. Set environment variables in `.env`:
   ```bash
   PROMTAIL_ENABLE=true
   PROMTAIL_LOKI_URL=http://your-loki-server:3100/loki/api/v1/push
   GRAIL_ENV=prod
   GRAIL_LOG_FILE=/var/log/grail/grail.log

   # Optional: log rotation settings
   GRAIL_LOG_MAX_SIZE=100MB
   GRAIL_LOG_BACKUP_COUNT=5
   ```

2. Deploy **with Promtail enabled** using the `--profile promtail` flag:
   ```bash
   docker compose --env-file .env --profile promtail -f docker/docker-compose.validator.yml up -d
   ```

   Or set the profile in your environment:
   ```bash
   export COMPOSE_PROFILES=promtail
   docker compose --env-file .env -f docker/docker-compose.validator.yml up -d
   ```

3. To run **without Promtail** (default behavior):
   ```bash
   docker compose --env-file .env -f docker/docker-compose.validator.yml up -d
   ```

4. Verify logs appear in Grafana with the configured labels.

**Architecture:**
- App writes logs to console (Rich) and file (`GRAIL_LOG_FILE`)
- Promtail tails the log file and ships to Loki with buffering, backoff, and timeouts
- No network I/O in the app's logging path prevents stalling

**Troubleshooting:**
- Check Promtail logs: `docker logs grail-promtail`
- Verify log file exists: `docker exec grail-validator ls -la /var/log/grail/`
- Test Loki connectivity: `docker exec grail-promtail wget -O- ${PROMTAIL_LOKI_URL}`

Public dashboards:
- **WandB Dashboard**: set `WANDB_ENTITY=tplr` and `WANDB_PROJECT=grail` to publish validator logs to the public W&B project for detailed metrics and historical data. View at https://wandb.ai/tplr/grail.
- **Grafana Dashboard**: Real-time system logs, validator performance metrics, and network statistics are available at https://grail-grafana.tplr.ai/.

---

## Running the Validator

### Standard Deployment

```bash
# Start validator with Docker Compose (includes Watchtower for automatic updates)
docker compose --env-file .env -f docker/docker-compose.validator.yml up -d

# View validator logs
docker logs -f grail-validator

# View Watchtower logs (automatic updater)
docker logs -f watchtower
```

Most behavior is configured via `.env`.

### Automatic Updates with Watchtower

The Docker Compose configuration includes Watchtower, which automatically:
- Checks for new validator images every 30 seconds
- Pulls the latest `ghcr.io/one-covenant/grail:latest` image
- Gracefully restarts your validator with the new version
- Cleans up old images to save disk space

This ensures your validator always runs the latest stable version without manual intervention.

### Manual Control

```bash
# Stop validator and Watchtower
docker compose -f docker/docker-compose.validator.yml down

# Update manually (if Watchtower is stopped)
docker pull ghcr.io/one-covenant/grail:latest
docker compose --env-file .env -f docker/docker-compose.validator.yml up -d

# Monitor resources
docker stats grail-validator
nvidia-smi  # GPU usage
```

---

## Operations

The implementation lives in `grail/cli/validate.py`.

### Windowing

- Determine current block; compute windows of length `WINDOW_LENGTH`.
- Process the previous complete window: `target_window = current - WINDOW_LENGTH`.
- **Load checkpoint**: Download the appropriate model checkpoint for the target window from R2.

### Discovery & Download

- For each hotkey (test mode: just self; otherwise all active hotkeys in metagraph):
  - Build expected file: `grail/windows/{hotkey}-window-{target_window}.parquet`.
  - Retrieve miner’s read credentials from chain (`GrailChainManager`).
  - Check existence and download with read creds; fallback to local creds if needed.

### Verification

For each downloaded inference:
- Required fields: `window_start`, `nonce`, `block_hash`, `commit`, `proof`, `challenge`, `hotkey`, `signature`, plus environment-specific fields.
- Window and block-hash must match the validator's `target_window` and hash.
- Nonce must be unique within a miner's window file.
- Signature check: hotkey verifies `challenge = seed + block_hash + nonce`.
- Seed is reconstructed as `{wallet_addr}-{target_window_hash}-{rollout_group}`. GRPO group size is fixed at 16 (`ROLLOUTS_PER_PROBLEM`).
- Challenge randomness for GRAIL proof: mix drand randomness (current round) with `target_window_hash`; fallback to hash-only on failures.
- Verifier (`grail/grail.py`) checks token validity, sketch proof, and model identity.
- For Triton Kernel environment: kernel correctness is verified by re-executing the generated kernel on-GPU against reference implementations.

Sampling and batching:
- If total rollouts ≤ MAX_SAMPLES_PER_MINER → verify all.
- Else sample complete GRPO groups (~10%) with early stopping if failures exceed threshold.

### Scoring & Weights

Per miner, compute `estimated_unique` rollouts over recent windows (extrapolated to account for sampling). If `UNIQUE_ROLLOUTS_CAP` is enabled, unique rollouts are capped at this value.

Apply superlinear curve (`SUPERLINEAR_EXPONENT = 4.0`):
```
score = estimated_unique ** SUPERLINEAR_EXPONENT
```

Normalize to weights across miners; set on-chain with `set_weights`. An emission burn mechanism (`GRAIL_BURN_PERCENTAGE = 80%`) redirects a portion of emissions to the burn UID.

### Publishing

- Upload all valid rollouts to R2/S3 for training (`upload_valid_rollouts`).

---

## Troubleshooting

### Common Issues

- No files found: ensure miners committed read creds on-chain; verify bucket name (should be the same as your account ID) and permissions.
- Frequent verification failures: check drand connectivity; fall back with `--no-drand`.
- Weight setting fails: wallet funding/permissions and network connectivity.
- Test mode confusion: `--test-mode` validates only your own files; disable for production.

### Docker-Specific Issues

**Validator Not Starting:**
- Check logs: `docker logs grail-validator`
- Verify wallet path: Ensure `~/.bittensor` is accessible
- Check hardware support: Ensure your platform's floating point precision is within tolerance thresholds

**Watchtower Not Updating:**
- Check registry access: `
