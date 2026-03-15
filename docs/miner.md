# Miner Setup

This guide explains how to run a Grail miner. Miners generate GRPO rollouts with GRAIL proofs across multiple environments (Triton Kernel, SAT, GSM8K, MATH, MBPP, HumanEval, and more), upload them to object storage, and participate in decentralized scoring via validators.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Wallets](#wallets)
  - [Storage (R2/S3)](#storage-r2s3)
  - [Monitoring](#monitoring)
- [Supported Environments](#supported-environments)
  - [Basilica Kernel Evaluation Setup](#basilica-kernel-evaluation-setup)
  - [Triton Kernel Setup](#triton-kernel-setup)
- [Running the Miner](#running-the-miner)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)

---

## Introduction

Grail miners:
- Connect to a Bittensor subnet and follow scoring windows.
- Derive per-window randomness from drand and the window’s block hash.
- Generate multiple GRPO rollouts per problem using a HF model across various environments.
- Produce GRAIL proofs (Prover) binding tokens to the model and seed.
- Upload signed rollouts to object storage for validators to verify and score.

The active environment is set network-wide (currently **Triton Kernel**). The environment determines what problems are generated and how rewards are computed. See [Supported Environments](#supported-environments) below.

---

## Prerequisites

- Linux with NVIDIA GPU drivers installed (required for Triton Kernel environment; macOS/Windows may work for text-only environments but is untested)
- Python (via `uv venv`) and Git
- `flash-attn` package: `uv pip install flash-attn --no-build-isolation` (requires CUDA toolkit; included in Docker image)
- [Basilica CLI](https://cli.basilica.ai) installed and funded — required to deploy the kernel-bench evaluation service
- Bittensor wallet (cold/hot) registered on the target subnet (all neurons verify registration at startup and exit with a helpful error if not registered)
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

For detailed hardware specifications, see [`compute.min.yaml`](../compute.min.yaml).

Hardware requirements:
- **Text-only environments** (SAT, GSM8K, MATH, MBPP, HumanEval): 1 GPU with 24GB+ VRAM
- **Triton Kernel environment** (current production): 2 GPUs + Basilica:
  - **GPU 0** — Model inference / decoding (vLLM/SGLang backend)
  - **GPU 1** — Proof computation / logprob verification (HuggingFace model)
  - **Kernel evaluation** — Basilica cloud A100 GPUs. Set `KERNEL_EVAL_BACKEND=basilica` and `BASILICA_EVAL_URL`.
- At least 64GB RAM recommended
- Network bandwidth needs are modest; uploads are Parquet rollout files

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

# Deploy kernel-bench on Basilica (required for Triton Kernel environment)
# See "Basilica Kernel Evaluation Setup" below

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
- Model & generation (dynamically loaded from R2 checkpoints)
  - The model is loaded automatically from R2 checkpoints and evolves through training
  - Miners automatically load the latest checkpoint from the previous window
  - Maximum new tokens is fixed at 8192 (hardcoded constant `MAX_NEW_TOKENS`)
  - Rollouts per problem is fixed at 16 (hardcoded constant `ROLLOUTS_PER_PROBLEM`)
  - Models are shared via R2 storage and updated by the trainer after each window
  - No manual model configuration required - checkpoints are loaded automatically
- Performance tuning
  - `GRAIL_GENERATION_BATCH_SIZE` (default: 1): Number of rollouts to generate in parallel per batch. Higher values increase throughput but require more VRAM. Must be ≤ 16 and must divide evenly into `ROLLOUTS_PER_PROBLEM`. Valid options: 1, 2, 4, 8, 16. Start with 1 and gradually increase while monitoring GPU memory with `nvidia-smi`. Example: `export GRAIL_GENERATION_BATCH_SIZE=4` for ~3-4x throughput on A100.
- Kernel evaluation (Triton Kernel environment)
  - `GRAIL_GPU_EVAL=true`: Required. Enables GPU-based kernel correctness evaluation.
  - `KERNEL_EVAL_BACKEND=basilica`: Required. Uses Basilica cloud A100 GPUs for kernel eval.
  - `BASILICA_EVAL_URL`: URL of the Basilica kernel-bench service.
  - `KERNEL_EVAL_TIMEOUT` (default: 60): Per-kernel evaluation timeout in seconds.
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

## Supported Environments

The active environment is configured network-wide and determines the problem type and reward structure. Miners automatically use the current environment.

| Environment | ID | Description | GPU Requirement |
|---|---|---|---|
| **Triton Kernel** | `triton_kernel` | Generate GPU kernels in Triton; evaluated for correctness on-GPU | 2 GPUs + Basilica |
| **3-SAT** | `sat` | Deterministic 3-SAT constraint satisfaction problems | 1 GPU |
| **GSM8K** | `gsm8k` | Math word problems with step-by-step reasoning | 1 GPU |
| **MATH** | `math` | Competition-level mathematics (Hendrycks MATH) | 1 GPU |
| **MBPP** | `mbpp` | Python code generation (Mostly Basic Python Problems) | 1 GPU |
| **HumanEval** | `humaneval` | Function-level code generation (OpenAI HumanEval) | 1 GPU |
| **Affine Trace** | `affine_trace` | Affine type system trace reasoning | 1 GPU |
| **Affine Logic** | `affine_logic` | Affine type system logic reasoning | 1 GPU |

The current production environment is **Triton Kernel**. Pipeline mode is the **recommended** way to run miners. The miner pipeline uses GPUs in parallel:

1. **GPU 0 — Decoding**: vLLM/SGLang generates Triton kernel code from problem prompts.
2. **GPU 1 — Proof computation**: A HuggingFace model computes logprobs and GRAIL commitments for verification.
3. **Kernel evaluation**: Basilica cloud A100 GPUs check correctness against reference implementations.

Proof computation (GPU 1) and kernel evaluation (Basilica) run **in parallel** after decoding completes, since proofs only need token IDs and do not depend on evaluation results.

**Weight sync between windows:** When a new checkpoint is available, the pipeline reloads vLLM weights using the sleep/wake/reload API (~3-30 seconds) instead of a full server restart (~5 minutes). This is automatic and requires no configuration. If the fast path fails, it falls back to a full restart.

If checkpoints live on a different volume (e.g. ephemeral/tmpfs), set `GRAIL_PIPELINE_SYMLINK_DIR` to a writable directory on the same filesystem or any accessible path — the symlink used for weight reload will be placed there instead of next to the checkpoint.

### Basilica Kernel Evaluation Setup

The Triton Kernel environment requires a running **kernel-bench** service on Basilica for GPU-based kernel correctness evaluation. You must deploy this service **before** starting your miner.

**1. Install the Basilica CLI and log in:**

```bash
curl -fsSL https://cli.basilica.ai/install.sh | sh
basilica login
```

**2. Deploy kernel-bench:**

```bash
basilica deploy ghcr.io/erfanmhi/kernel-bench:latest \
  --gpu 2 --gpu-model A100 \
  --cpu 12 --memory 48Gi \
  --health-path /health \
  --startup-failure-threshold 60
```

The number of GPUs affects how many kernels can be evaluated in parallel. Recommendations:
- **1 GPU** — sufficient for validators and low-throughput miners
- **2 GPUs** — recommended starting point for miners
- **More GPUs** — useful if you generate many rollouts per window and want to overlap proof computation with kernel evaluation. Scale up based on your throughput needs.

**3. Wait for the service to be ready:**

The deploy command will stream progress. Once it shows a URL like `https://<id>.deployments.basilica.ai`, verify it's healthy:

```bash
curl https://<your-instance>.deployments.basilica.ai/health
```

You should see `"status": "healthy"` with `workers_alive` matching your GPU count.

**4. Set `BASILICA_EVAL_URL` in your `.env`:**

```bash
BASILICA_EVAL_URL=https://<your-instance>.deployments.basilica.ai
```

**Managing your deployment:**

```bash
basilica deploy ls                    # list all deployments
basilica deploy status <deployment-id> # check status
basilica deploy logs <deployment-id>   # view logs
basilica deploy restart <deployment-id> # restart if needed
basilica deploy delete <deployment-id>  # tear down when done
```

### Triton Kernel Setup

Once your Basilica kernel-bench service is running, configure your `.env`:

```bash
# Kernel evaluation (Basilica)
GRAIL_GPU_EVAL=true
KERNEL_EVAL_BACKEND=basilica
BASILICA_EVAL_URL=https://<your-instance>.deployments.basilica.ai

# Pipeline mode (GPU 0 = decoding, GPU 1 = proofs)
GRAIL_PIPELINE_ENABLED=true
GRAIL_PIPELINE_BACKEND=sglang
GRAIL_PIPELINE_GEN_GPU=0
GRAIL_PIPELINE_PROOF_GPU=1

CUDA_VISIBLE_DEVICES=0,1 grail -vv mine
```

`GRAIL_GPU_EVAL` must be `true` for production. This is required for full kernel correctness evaluation via Basilica.

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
3. **Load checkpoint**: Download the model checkpoint from the previous window (`window_start - WINDOW_LENGTH`) from R2.
4. For the window:
   - Derive randomness: `sha256(block_hash + drand.randomness)` (or block hash only).
   - Generate problems from the active environment and create GRPO batches using the loaded checkpoint.
   - Use `Prover` to commit/open GRAIL proofs and package signed rollouts.
5. Upload the window's rollouts to R2/S3 with write credentials.
6. Repeat on the next window (loading new checkpoint if available).

Artifacts uploaded per window (as a Parquet file) include per-rollout:
- GRAIL commit (`tokens`, `s_vals`, signature, beacon)
- Environment-specific problem metadata (e.g., kernel reference, problem ID)
- GRPO data (reward, advantage, token logprobs, lengths, success)
- Miner signature over a challenge derived from seed/block/nonce

---

## Troubleshooting

### Quick Fixes

- CUDA OOM or driver errors: Ensure you have adequate GPU VRAM (40GB+ recommended per GPU); verify drivers match CUDA runtime; periodically clear cache.
- GPU not detected: Check `nvidia-smi` output.
- Kernel eval failures: Run `curl $BASILICA_EVAL_URL/health` — check that `workers_alive` > 0 and `status` is `healthy`. If workers are dead, run `basilica deploy restart <deployment-id>`. If the deployment doesn't exist yet, see [Basilica Kernel Evaluation Setup](#basilica-kernel-evaluation-setup).
- No uploads: check `R2_*` variables and bucket permissions; verify network/firewall.
- Not receiving weights: ensure uploads succeed; validator will score the previous complete window.
- Drand failures: miner automatically falls back to block-hash; you can use `--no-drand`.
- Wallet not found: ensure `BT_WALLET_COLD`/`BT_WALLET_HOT` names exist in your `~/.bittensor/wallets`.

### Debugging and Optimization

**Don't debug on mainnet first!** The most efficient way to debug and optimize your miner is to run a local validator and monitor your submissions in real-time.

For a comprehensive guide on debugging techniques, local validation setup, and optimization workflows, see:
- **[Miner Debugging and Optimization Guide](miner-debugging.md)**

Key points:
- ✅ Run a local validator to test your miner submissions
- ✅ Monitor logs for detailed rejection reasons
- ✅ Fix issues before they impact mainnet performance
- ❌ Don't rely solely on Grafana/WandB dashboards for debugging

---

## Best Practices

- **Triton Kernel** (current production): Set `KERNEL_EVAL_BACKEND=basilica` and `BASILICA_EVAL_URL`. Only 2 local GPUs needed.
- **Text-only environments** (SAT, GSM8K, MATH, etc.): 1 GPU is sufficient; any CUDA-capable accelerator works.
- Models evolve through training: the initial base model is loaded from R2 and automatically updated with new checkpoints each window. Fixed at 8192 max new tokens and 16 rollouts per problem.
- Reserve the final 2 blocks of each window for uploads; the miner does this automatically but avoid heavy generation near the end.
- Use `--use-drand` (default) for robust challenge derivation; fall back with `--no-drand` only if needed.
- Ensure R2 dual-credential setup: write locally, read credentials are committed on-chain by the miner.
- Monitor GPU memory; the miner periodically empties cache, but size your rollouts to avoid OOM.
- Increase verbosity with `-vv` when diagnosing sampling, group sizes, or upload issues.


## Support

For issues or questions:
- **FAQ**: [Common questions and answers](FAQ.md)
- **Debugging Guide**: [Miner debugging and optimization](miner-debugging.md)
- GitHub Issues: https://github.com/one-covenant/grail/issues
- Discord: https://discord.com/channels/799672011265015819/1354089114189955102
