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

- Linux, Docker and Docker Compose installed
- **NVIDIA A100 GPU with CUDA support (required)**
- At least 20GB RAM recommended
- Bittensor wallet (cold/hot) registered on the target subnet
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

Hardware requirements:
- **NVIDIA A100 GPU is required** for the current version to ensure consistent verification across the network
- GPU-agnostic verification is coming soon, which will enable support for other hardware configurations
- The verifier has been optimized and tested on NVIDIA A100

---

## Quick Start

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
- Model (read-only for validators in this release)
  - `GRAIL_MODEL_NAME` (mandatory: set to `Qwen/Qwen3-4B-Instruct-2507` in first version)
  - `GRAIL_MAX_NEW_TOKENS` (mandatory: set to `1024` in first version)
- Object storage (R2/S3)
  - `R2_BUCKET_ID`, `R2_ACCOUNT_ID`
  - Dual credentials (recommended):
    - Read-only: `R2_READ_ACCESS_KEY_ID`, `R2_READ_SECRET_ACCESS_KEY`
    - Write: `R2_WRITE_ACCESS_KEY_ID`, `R2_WRITE_SECRET_ACCESS_KEY`
- Monitoring
  - `GRAIL_MONITORING_BACKEND` (wandb|null)
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`
- Optional
  - `HF_TOKEN`, `HF_USERNAME` (for Hugging Face dataset publishing)

### Wallets

Use a registered wallet. Set `BT_WALLET_COLD`/`BT_WALLET_HOT` to the names you created with `btcli`.

### Storage (R2/S3)

**Bucket requirement:** Name it the same as your account ID; set the region to ENAM.

Validators load local write credentials and use miners’ read credentials fetched from chain to download their files.

### Monitoring

Set `GRAIL_MONITORING_BACKEND=wandb` to enable metrics; otherwise use `null`.

#### Grafana + Loki Logging (Recommended)

For centralized logging, use Promtail to ship logs to Loki. This approach is more robust than in-process log shipping and prevents stalling under network pressure.

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

2. Deploy with Promtail included:
   ```bash
   docker compose --env-file .env -f docker/docker-compose.validator.yml up -d
   ```

3. Verify logs appear in Grafana with the configured labels.

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
- SAT seed must equal `{wallet_addr}-{target_window_hash}-{nonce}` (for GRPO rollouts, base seed `{wallet_addr}-{target_window_hash}-{rollout_group}` is reconstructed). Note: GRPO group size is fixed at 4 (`GRAIL_ROLLOUTS_PER_PROBLEM=4`) in this release and must not be changed by miners.
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
Apply superlinear curve (from `SUPERLINEAR_EXPONENT`, default 2.5):
```
score = base ** SUPERLINEAR_EXPONENT
```
Normalize to weights across miners; set on-chain with `set_weights`.

### Publishing

- Upload all valid rollouts to R2/S3 for training (`upload_valid_rollouts`).

---

## Troubleshooting

### Common Issues

- No files found: ensure miners committed read creds on-chain; verify bucket name (should be the same as your account ID) and permissions.
- Frequent verification failures: check model alignment (`GRAIL_MODEL_NAME`), drand connectivity; fall back with `--no-drand`.
- Weight setting fails: wallet funding/permissions and network connectivity.
- Test mode confusion: `--test-mode` validates only your own files; disable for production.

### Docker-Specific Issues

**Validator Not Starting:**
- Check logs: `docker logs grail-validator`
- Verify wallet path: Ensure `~/.bittensor` is accessible
- Check GPU: Run `nvidia-smi` to verify NVIDIA A100 availability
- GPU requirement: Currently requires NVIDIA A100 for consistent verification

**Watchtower Not Updating:**
- Check registry access: `docker pull ghcr.io/one-covenant/grail:latest`
- Verify Watchtower logs: `docker logs watchtower`
- Ensure container has label `com.centurylinklabs.watchtower.enable=true`

**Network Issues:**
- Ensure ports are accessible if using external axon
- Check firewall rules for Docker networks
- Verify subtensor endpoint connectivity

**Container Health:**
```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.State}}"

# Inspect container details
docker inspect grail-validator
```

---

## Reference

- Validator entrypoint: `grail/cli/validate.py`
- Verifier & protocol: `grail/grail.py`
- SAT environment & rewards: `grail/environments/sat.py`
- Storage & downloads: `grail/infrastructure/comms.py`
- Credentials & chain: `grail/infrastructure/credentials.py`, `grail/infrastructure/chain.py`

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/one-covenant/grail/issues
- Discord: https://discord.com/channels/799672011265015819/1354089114189955102