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
- **No extra kernel evaluation GPU is required for this release.** The coding/math environments do not run on-GPU kernel eval; an additional GPU is only needed if you opt in to the `triton_kernel` environment.
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
  - `GRAIL_PIPELINE_PROOF_GPU` (default: 1): GPU index for the primary HuggingFace proof model, relative to `CUDA_VISIBLE_DEVICES`. Used as the single proof GPU when `GRAIL_PIPELINE_PROOF_GPU_IDS` is unset, and as the primary worker (the one that reuses the parent's loaded model) when the multi-worker pool is enabled.
  - `GRAIL_PIPELINE_PROOF_GPU_IDS` (default: empty): Comma-separated list of proof GPU indices for the multi-proof-worker pool, e.g. `"4,5,6,7"`. Empty falls back to a single in-process worker on `GRAIL_PIPELINE_PROOF_GPU`. The pool fans proofs out across all listed GPUs in parallel — this is what lifts proof throughput from the single-worker ceiling (~5 r/s on a 7B model) to ~20 r/s with 4 GPUs, and is required for any mining configuration that wants more than ~600 rollouts/window.
  - `GRAIL_PIPELINE_PROOF_SUBPROCESS` (default: false): Run the proof workers in persistent `mp.spawn` subprocesses instead of an in-process `ThreadPoolExecutor`. Each subprocess pins to one GPU via `CUDA_VISIBLE_DEVICES`, loads its own copy of the HF model on `cuda:0`, and computes proofs without a wallet (the parent process signs the returned commitment hashes). This eliminates GIL contention in the proof phase and is the recommended setting whenever `GRAIL_PIPELINE_PROOF_GPU_IDS` lists more than one GPU. Falls back to in-process workers on any startup error.
  - `GRAIL_PIPELINE_GROUPS_PER_BATCH` (default: 1): Number of GRPO groups dispatched to the SGLang/vLLM backend in a single `backend.generate()` call. Default 1 keeps the v0.0.59 baseline behaviour (one group of 16 prompts per request). Larger values (24, 48, 72) push many more in-flight sequences into SGLang's continuous batcher and lift TP=4 generation utilisation from ~6% (16 in flight) to ~80%+ (1152 in flight). Pair with `GRAIL_PIPELINE_PROOF_GPU_IDS` so the proof side can keep up with the higher gen rate.
  - `GRAIL_PIPELINE_VLLM_TP` (default: 1): Tensor-parallel size for the generation server. Set to 4 to spread the model across consecutive GPUs `[VLLM_GPU, VLLM_GPU+VLLM_TP)`. The SGLang launch path correctly wires this through to `--tp-size` and the corresponding `CUDA_VISIBLE_DEVICES` range, symmetric with the vLLM path.
  - `GRAIL_PIPELINE_MAX_MODEL_LEN` (default: 12288): Server context length. The miner clamps `prompt_len + max_new_tokens` to this value before issuing requests.
  - `GRAIL_PIPELINE_PROOF_FLASH_ATTN` (default: false): MUST be left at `false` so the proof path matches the validator's SDPA implementation. Setting it to `true` causes proof divergence.
  - `GRAIL_PIPELINE_DISABLE_CUDA_GRAPH` (default: false): Belt-and-suspenders SGLang stability flag. The launch already passes `--disable-piecewise-cuda-graph` and `--disable-custom-all-reduce` unconditionally as part of the v0.0.59 Blackwell workaround; this flag also disables the FULL CUDA graph capture, which only matters on GPU topologies where the partial workaround isn't enough.
- Performance tuning
  - `GRAIL_GENERATION_BATCH_SIZE` (default: 1): Legacy single-group batch size for the in-group dispatch. Pre-dates `GRAIL_PIPELINE_GROUPS_PER_BATCH`; on the modern multi-group path the per-group batching is determined by `ROLLOUTS_PER_PROBLEM=16` and the relevant tuning knob is `GRAIL_PIPELINE_GROUPS_PER_BATCH`.
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

The pipelined mining engine is mandatory and lays out work across at least two GPUs. The reference build supports three proof topologies, in order of capacity:

1. **Single proof worker (baseline)** — one in-process `ProofWorker` on `GRAIL_PIPELINE_PROOF_GPU`. Sequential one-group-at-a-time dispatch through one proof thread. Easy to set up; ~5 r/s ceiling on a 7B model so ~600 rollouts/window after wallet/upload overhead.
2. **N in-process workers** — set `GRAIL_PIPELINE_PROOF_GPU_IDS=4,5,6,7`. The engine spins up one `ProofWorker` per listed GPU; the multi-group dispatch fan-out splits work across them. Lifts proof ceiling to ~20 r/s with 4 GPUs but stays GIL-bound on Python-side proof prep.
3. **N persistent subprocess workers (recommended for production)** — the same `GRAIL_PIPELINE_PROOF_GPU_IDS` setting plus `GRAIL_PIPELINE_PROOF_SUBPROCESS=true`. Each worker runs in an `mp.spawn` subprocess with `CUDA_VISIBLE_DEVICES` pinned to one physical GPU, loads the HF model once on `cuda:0`, and computes proofs without a wallet. The parent process signs commitment hashes after the worker returns. GIL-free; the path used for the FINAL_ARCHITECTURE benchmark of 4,477 rollouts/window on 8x A100 SXM4 with Qwen2.5-7B.

The generation side is laid out as:

- **GPUs 0 to (TP-1)** — SGLang generation server with `--tp-size` set from `GRAIL_PIPELINE_VLLM_TP`. Sees `input_ids`, returns `output_ids` directly, no text re-tokenization. With `GRAIL_PIPELINE_GROUPS_PER_BATCH=N` the miner sends `N×16` prompts in a single `/generate` call so SGLang's continuous batcher can amortise prefill across hundreds of in-flight sequences (vs 16 in the v0.0.59 baseline).
- **GPUs TP to (TP + len(proof_gpu_ids) - 1)** — proof workers, one per GPU.

Across groups, the miner double-buffers: collect for batch N runs **while** batch N+1 is generating. Combined with multi-group dispatch and the subprocess pool, this is the architecture that pushes per-window throughput from the v0.0.59 single-worker baseline (~600 r/win) toward the FINAL_ARCHITECTURE measured ceiling.

**Weight sync between windows:** When the trainer publishes a new checkpoint, the miner reloads the SGLang/vLLM server weights using the sleep/wake/reload API (~3-30 seconds) instead of a full server restart. The in-process proof workers re-use the parent's freshly-applied state dict (fast path) or `load_model()` from disk (slow path). The subprocess pool reloads workers in-place over the pipe via a `ReloadCommand` instead of restarting the processes, which avoids the `mp.spawn` startup cost on every checkpoint roll. All of this is automatic and requires no configuration.

If checkpoints live on a different volume (e.g. ephemeral/tmpfs), set `GRAIL_PIPELINE_SYMLINK_DIR` to a writable directory on the same filesystem — the symlink used for weight reload will be placed there. (vLLM backend only; harmless when `GRAIL_PIPELINE_BACKEND=sglang`.)

### Two-GPU setup example (single proof worker)

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

### 8-GPU setup example (TP=4 SGLang + 4 subprocess proof workers)

For an 8-GPU host (e.g. 8x A100 SXM4) targeting the FINAL_ARCHITECTURE throughput:

```bash
# .env additions on top of the two-GPU example
GRAIL_PIPELINE_VLLM_GPU=0
GRAIL_PIPELINE_VLLM_TP=4               # GPUs 0-3 → SGLang TP=4 instance
GRAIL_PIPELINE_PROOF_GPU=4             # primary in-process worker
GRAIL_PIPELINE_PROOF_GPU_IDS=4,5,6,7   # GPUs 4-7 → proof workers
GRAIL_PIPELINE_PROOF_SUBPROCESS=true   # GIL-free subprocess pool
GRAIL_PIPELINE_GROUPS_PER_BATCH=72     # send 72×16=1152 prompts per SGLang call
GRAIL_PIPELINE_GPU_MEM_UTIL=0.88
GRAIL_PIPELINE_MAX_MODEL_LEN=4096
GRAIL_PIPELINE_MAX_NUM_SEQS=512
GRAIL_PIPELINE_MAX_CONCURRENT=512

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 grail -vv mine
```

Tune `GRAIL_PIPELINE_GROUPS_PER_BATCH` upward (24 → 48 → 72 → 96) until SGLang utilisation flattens; the proof side scales with the number of GPUs in `GRAIL_PIPELINE_PROOF_GPU_IDS`. Smaller boxes (4 GPUs) typically run TP=2 + 2 proof workers + `GROUPS_PER_BATCH=16`.

> **Note on GPU indices:** `GRAIL_PIPELINE_VLLM_GPU` and `GRAIL_PIPELINE_PROOF_GPU(_IDS)` are **relative** to `CUDA_VISIBLE_DEVICES`. The miner validates the full vLLM TP range and every entry in the proof list at startup and exits with a clear `ProtocolViolationError` if any index exceeds the visible device count.

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
- GPU not detected: Check `nvidia-smi` output. The coding/math envs in this release only need the SGLang and proof GPUs; `KERNEL_EVAL_GPU_IDS` is only consulted when the (inactive) `triton_kernel` environment is selected.
- Kernel eval failures: only relevant when `triton_kernel` is active. CUDA sticky errors (illegal memory access, device-side assert) are automatically recovered via subprocess isolation and retry. Check logs for `CUDA sticky error` warnings.
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

- **Two GPUs are the minimum** for the pipelined miner. SGLang on GPU 0, HF proof model on GPU 1. Both must be in `CUDA_VISIBLE_DEVICES`. To exceed the single-proof-worker baseline (~600 r/win), set `GRAIL_PIPELINE_PROOF_GPU_IDS` and `GRAIL_PIPELINE_PROOF_SUBPROCESS=true` to enable the multi-worker subprocess pool.
- **Tune `GRAIL_PIPELINE_GROUPS_PER_BATCH`** to push more work into SGLang's continuous batcher. Default 1 reproduces the v0.0.59 baseline; values of 24-72 are typical for production multi-GPU setups.
- The trainer publishes the active environment, model, and sampling policy per checkpoint. **Do not override `max_tokens`, `temperature`, etc. locally** — let the miner read them from `CheckpointMetadata.generation_params` so the validator's hard checks pass by construction.
- `GRAIL_PIPELINE_PROOF_FLASH_ATTN` MUST stay `false`. The validator uses SDPA; switching the proof path to FA2 changes FP accumulation and causes sketch divergence.
- Reserve the final 2 blocks of each window for uploads; the miner does this automatically via its EMA-based time-budget gate.
- Use `--use-drand` (default) for robust challenge derivation; fall back with `--no-drand` only if needed.
- Ensure R2 dual-credential setup: write locally, read credentials are committed on-chain by the miner.
- Monitor GPU memory; the miner periodically empties cache, but size your rollouts and `GRAIL_PIPELINE_GROUPS_PER_BATCH` to avoid OOM on the SGLang side.
- Increase verbosity with `-vv` when diagnosing sampling, group sizes, or upload issues.


## Support

For issues or questions:
- **FAQ**: [Common questions and answers](FAQ.md)
- **Debugging Guide**: [Miner debugging and optimization](miner-debugging.md)
- GitHub Issues: https://github.com/one-covenant/grail/issues
- Discord: https://discord.com/channels/799672011265015819/1354089114189955102
