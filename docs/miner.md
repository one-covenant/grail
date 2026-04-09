# Miner Setup

This guide explains how to run a Grail miner. Miners generate GRPO rollouts with GRAIL proofs for the network's active environment, upload them to object storage, and participate in decentralized scoring via validators.

For this release the network focuses on **coding** (MBPP, HumanEval) and **math** (GSM8K, MATH) environments. The active environment is published per checkpoint by the trainer and the miner picks it up automatically.

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
- Derive per-window randomness from drand and the window's block hash.
- Run an SGLang generation server (default backend) to produce GRPO rollouts in parallel.
- Compute GRAIL proofs on a dedicated proof GPU (HuggingFace model).
- Upload signed rollouts to object storage for validators to verify and score.

The active environment, sampling policy (`max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`), and model are all published per-checkpoint by the trainer. The miner reads them from each checkpoint's metadata and drives the generation server with exactly those values, so the validator's hard checks (`termination_valid`, `logprobs_valid`, `proof_valid`) line up automatically.

The pipelined mining engine is the **only** generation path. The legacy single-GPU fallback was removed; you need at least two visible GPUs (one for SGLang, one for HF proofs).

---

## Prerequisites

- Linux with NVIDIA GPU drivers installed (required for Triton Kernel environment; macOS/Windows may work for text-only environments but is untested)
- Python (via `uv venv`) and Git
- Bittensor wallet (cold/hot) registered on the target subnet (all neurons verify registration at startup and exit with a helpful error if not registered)
- Cloudflare R2 (or S3-compatible) bucket and credentials
  - **Create a Bucket: Name it the same as your account ID and set the region to ENAM.**
- Optional: WandB account for monitoring

For detailed hardware specifications, see [`compute.min.yaml`](../compute.min.yaml).

Hardware requirements (pipeline mode is mandatory):
- **At least 2 visible GPUs** with 40GB+ VRAM each (A100 / H100 / B200 class):
  - **GPU 0** — SGLang generation server (decoding)
  - **GPU 1** — HuggingFace model for proof computation (logprobs + GRAIL commitments)
- Both GPUs must be in `CUDA_VISIBLE_DEVICES`. The miner exits with a clear `ProtocolViolationError` at startup if the configured `GRAIL_PIPELINE_VLLM_GPU` / `GRAIL_PIPELINE_PROOF_GPU` indices exceed the visible device count.
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
- Model & generation (driven by trainer-published checkpoints)
  - The model and the active environment are loaded automatically from the trainer's most recent R2 checkpoint.
  - Sampling policy (`max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`) is published per-checkpoint and consumed by the miner via `GenerationParams.from_checkpoint_metadata`. The miner drives the SGLang/vLLM backend with exactly those values, so the validator's `termination_valid` check (`min(metadata.max_tokens, MAX_NEW_TOKENS_PROTOCOL_CAP)`) lines up by construction.
  - `MAX_NEW_TOKENS_PROTOCOL_CAP = 8192` (in `grail/protocol/constants.py`) is the protocol cap; the trainer's per-checkpoint `max_tokens` is the actual limit and is typically much smaller.
  - Rollouts per problem is fixed at 16 (`ROLLOUTS_PER_PROBLEM`).
  - No manual model or sampling configuration required.
- Pipeline backend
  - `GRAIL_PIPELINE_BACKEND` (sglang|vllm, default: **sglang**): Generation server backend. SGLang is the production-validated path; it uses the native `/generate` endpoint with `input_ids` so there is no text re-tokenization.
  - `GRAIL_PIPELINE_VLLM_GPU` (default: 0): GPU index for the generation server, relative to `CUDA_VISIBLE_DEVICES`.
  - `GRAIL_PIPELINE_PROOF_GPU` (default: 1): GPU index for the HuggingFace proof model, relative to `CUDA_VISIBLE_DEVICES`. Must differ from `VLLM_GPU`.
  - `GRAIL_PIPELINE_VLLM_TP` (default: 1): Tensor-parallel size for the generation server. Set to 4 to spread the model across consecutive GPUs `[VLLM_GPU, VLLM_GPU+VLLM_TP)`.
  - `GRAIL_PIPELINE_MAX_MODEL_LEN` (default: 12288): Server context length. The miner clamps `prompt_len + max_new_tokens` to this value before issuing requests.
  - `GRAIL_PIPELINE_PROOF_FLASH_ATTN` (default: false): MUST be left at `false` so the proof path matches the validator's SDPA implementation. Setting it to `true` causes proof divergence.
- Performance tuning
  - `GRAIL_GENERATION_BATCH_SIZE` (default: 1): Number of rollouts to generate in parallel per batch. Higher values increase throughput but require more VRAM. Must be ≤ 16 and must divide evenly into `ROLLOUTS_PER_PROBLEM`. Valid options: 1, 2, 4, 8, 16.
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

For this release the network is focused on **coding** and **math** environments. The trainer publishes the active environment per checkpoint; the miner reads it from `CheckpointMetadata.env_id` and runs that environment automatically.

| Environment | ID | Description |
|---|---|---|
| **MBPP** | `mbpp` | Python code generation (Mostly Basic Python Problems) |
| **HumanEval** | `humaneval` | Function-level code generation (OpenAI HumanEval) |
| **GSM8K** | `gsm8k` | Math word problems with step-by-step reasoning |
| **MATH** | `math` | Competition-level mathematics (Hendrycks MATH) |

Other environments (`sat`, `affine_*`, `triton_kernel`, etc.) still exist in the codebase but are not the focus of this release.

### Pipeline architecture

The pipelined mining engine is mandatory and lays out work across at least two GPUs:

1. **GPU 0 — Generation server** (SGLang by default): Receives `input_ids`, generates completions in parallel, returns `output_ids` directly. No text re-tokenization.
2. **GPU 1 — Proof worker**: A HuggingFace model on a dedicated GPU computes the GRAIL commitments and chosen-token logprobs (GPU `log_softmax` + gather; only the chosen-token logprobs cross PCIe).

Generation and proof computation overlap across batches, so the pipeline produces several hundred rollouts per window on a 2-GPU setup.

**Weight sync between windows:** When the trainer publishes a new checkpoint, the miner reloads the SGLang/vLLM server weights using the sleep/wake/reload API (~3-30 seconds) instead of a full server restart. This is automatic and requires no configuration.

If checkpoints live on a different volume (e.g. ephemeral/tmpfs), set `GRAIL_PIPELINE_SYMLINK_DIR` to a writable directory on the same filesystem — the symlink used for weight reload will be placed there. (vLLM backend only; harmless when `GRAIL_PIPELINE_BACKEND=sglang`.)

### Two-GPU setup example

```bash
# .env configuration for the default coding/math envs with pipeline mode
GRAIL_PIPELINE_BACKEND=sglang   # Default and recommended
GRAIL_PIPELINE_VLLM_GPU=0       # Relative to CUDA_VISIBLE_DEVICES
GRAIL_PIPELINE_PROOF_GPU=1      # Relative to CUDA_VISIBLE_DEVICES
GRAIL_PIPELINE_VLLM_TP=1        # Bump only if you have spare GPUs
GRAIL_PIPELINE_MAX_MODEL_LEN=12288
GRAIL_PIPELINE_PROOF_FLASH_ATTN=false  # MUST stay false to match validator SDPA

# Run with GPU 0 for SGLang, GPU 1 for proofs
CUDA_VISIBLE_DEVICES=0,1 grail -vv mine
```

> **Note on GPU indices:** `GRAIL_PIPELINE_VLLM_GPU` and `GRAIL_PIPELINE_PROOF_GPU` are **relative** to `CUDA_VISIBLE_DEVICES`. The miner validates them at startup and exits with a clear `ProtocolViolationError` if they exceed the visible device count.

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

High-level loop (see `grail/cli/mine.py` and `grail/neurons/miner.py`):

1. Load R2 credentials and initialize `GrailChainManager`; commit read credentials on-chain.
2. Connect to subtensor; compute `window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH`.
3. **Load checkpoint**: Download the latest checkpoint from R2 (fast path if a delta on top of a cached checkpoint exists, full reconstruction otherwise).
4. **Initialize the pipeline (first window only)**: Start the SGLang generation server on `GRAIL_PIPELINE_VLLM_GPU` and load the proof model on `GRAIL_PIPELINE_PROOF_GPU`. Init failures `SystemExit(1)` so the supervisor restarts the process.
5. **Read sampling policy**: Pull `generation_params` from `CheckpointMetadata` and convert via `GenerationParams.from_checkpoint_metadata`. Malformed metadata raises `ProtocolViolationError`, which the outer loop catches and skips the window cleanly (no 10s retry storm).
6. For each problem in the window:
   - Derive randomness: `sha256(block_hash + drand.randomness)` (or block hash only).
   - Drive the generation server with the trainer's sampling policy to produce a GRPO group.
   - Compute proofs on the proof GPU; drop the entire group if any proof fails (so the validator's `group_size` check never sees a partial group).
   - Package signed rollouts.
7. Upload the window's rollouts to R2/S3 with write credentials.
8. Repeat on the next window. New checkpoints are detected automatically.

Artifacts uploaded per window (as a Parquet file) include per-rollout:
- GRAIL commit (`tokens`, `s_vals`, signature, beacon)
- Environment-specific problem metadata (e.g., kernel reference, problem ID)
- GRPO data (reward, advantage, token logprobs, lengths, success)
- Miner signature over a challenge derived from seed/block/nonce

---

## Troubleshooting

### Quick Fixes

- CUDA OOM or driver errors: Ensure you have adequate GPU VRAM (40GB+ recommended per GPU); verify drivers match CUDA runtime; periodically clear cache.
- GPU not detected: Check `nvidia-smi` output. For `triton_kernel`, ensure `KERNEL_EVAL_GPU_IDS` points to a valid **physical** device index (as shown by `nvidia-smi`).
- Kernel eval failures: CUDA sticky errors (illegal memory access, device-side assert) are automatically recovered via subprocess isolation and retry. Check logs for `CUDA sticky error` warnings.
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

- **Two GPUs are mandatory** for the pipelined miner. SGLang on GPU 0, HF proof model on GPU 1. Both must be in `CUDA_VISIBLE_DEVICES`.
- The trainer publishes the active environment, model, and sampling policy per checkpoint. **Do not override `max_tokens`, `temperature`, etc. locally** — let the miner read them from `CheckpointMetadata.generation_params` so the validator's hard checks pass by construction.
- `GRAIL_PIPELINE_PROOF_FLASH_ATTN` MUST stay `false`. The validator uses SDPA; switching the proof path to FA2 changes FP accumulation and causes sketch divergence.
- Reserve the final 2 blocks of each window for uploads; the miner does this automatically via its EMA-based time-budget gate.
- Use `--use-drand` (default) for robust challenge derivation; fall back with `--no-drand` only if needed.
- Ensure R2 dual-credential setup: write locally, read credentials are committed on-chain by the miner.
- Monitor GPU memory; the miner periodically empties cache, but size your rollouts (`GRAIL_GENERATION_BATCH_SIZE`) to avoid OOM.
- Increase verbosity with `-vv` when diagnosing sampling, group sizes, or upload issues.


## Support

For issues or questions:
- **FAQ**: [Common questions and answers](FAQ.md)
- **Debugging Guide**: [Miner debugging and optimization](miner-debugging.md)
- GitHub Issues: https://github.com/one-covenant/grail/issues
- Discord: https://discord.com/channels/799672011265015819/1354089114189955102
