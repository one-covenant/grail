<h1 align="center">grail: Verifiable Post-Training for LLMs</h1>

<div align="center">
  <pre>
   ✦  ✦  ✦  ✦  ✦  ✦  ✦
  ┌─┐┬─┐┌─┐┬┬
  │ ┬├┬┘├─┤││
  └─┘┴└─┴ ┴┴┴─┘
  ✦  ✦  ✦  ✦  ✦  ✦  ✦
  </pre>
</div>


<p align="center">
  Documentation:
  <a href="docs/miner.md">Miner</a> •
  <a href="docs/validator.md">Validator</a> •
  <a href="docs/FAQ.md">FAQ</a>
</p>

<p align="center">
  <!-- <a href="https://codecov.io/gh/one-covenant/grail">
    <img src="https://codecov.io/gh/one-covenant/grail/branch/main/graph/badge.svg" alt="Codecov" />
  </a>
  <a href="https://github.com/one-covenant/grail/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/one-covenant/grail" alt="License" />
  </a> -->
  <a href="https://wandb.ai/tplr/grail">
    <img src="https://img.shields.io/badge/W%26B-Public%20Dashboard-FFBE00?logo=weightsandbiases" alt="Weights & Biases" />
  </a>
  <a href="https://grail-grafana.tplr.ai/">
    <img src="https://img.shields.io/badge/Grafana-Real--time%20Logs-F46800?logo=grafana" alt="Grafana Dashboard" />
  </a>
  <a href="https://github.com/one-covenant/grail/commits/main">
    <img src="https://img.shields.io/github/last-commit/one-covenant/grail" alt="Last Commit" />
  </a>
</p>


## Overview

**grail** delivers post-training for language models with cryptographically verifiable inference. It implements the **GRAIL protocol** (Guaranteed Rollout Authenticity via Inference Ledger) so that rollouts produced during RL are tied to a specific model and input, and can be independently verified by validators.

### Key Distinction: grail vs GRAIL

- **grail** (lowercase): The Bittensor subnet implementation orchestrating miners, validators, and a trainer for verifiable post-training
- **GRAIL** (uppercase): The protocol that proves rollout authenticity and model identity

### Current Status

- Miners generate rollouts with GRAIL proofs across multiple environments (currently **Triton Kernel**)
- Validators verify proofs, evaluate kernel correctness on-GPU, and set weights on-chain
- The trainer runs GRPO-based reinforcement learning on validated rollouts, publishing updated checkpoints each window
- Model checkpoints are shared via R2 storage and automatically loaded by miners and validators

## Architecture

### Core Components

#### 1. GRAIL Protocol (`grail/grail.py`)
Prover/Verifier implementation with:
- PRF-based index derivation and sketch commitments for token-level verification
- Verifier-supplied challenge (drand + chain/window context)
- Token and model-config validation; structured signatures bound to model identity

#### 2. Mining Engine (`grail/mining/engine.py`, `grail/environments/loop.py`)
GRPO-style rollout system with:
- Multiple rollouts per problem (16 per group), token-level logprob tracking
- 3-GPU pipeline mode: vLLM generation, HuggingFace proof computation, and kernel evaluation in parallel
- Shared `forward_single_layer` function ensuring bit-identical results between miner and validator

#### 3. Environment System (`grail/environments/`)
Modular environments with a single active environment set network-wide:
- **Triton Kernel** (`gpu_kernel/`) — *current default*: GPU kernel generation and on-GPU correctness evaluation using Triton
- **3-SAT** (`sat.py`): Deterministic 3-SAT constraint satisfaction problems
- **GSM8K** (`gsm8k_env.py`): Math word problems with step-by-step reasoning verification
- **MATH** (`math_hendrycks_env.py`): Competition-level math from the Hendrycks MATH dataset
- **MBPP** (`python_code_env.py`): Python code generation from the MBPP benchmark
- **HumanEval** (`python_code_env.py`): Function-level code generation from OpenAI HumanEval
- **Affine Trace/Logic** (`affinetes/`): Affine type system trace and logic environments

#### 4. Trainer (`grail/trainer/`)
Asynchronous GRPO trainer with:
- Per-window training on validated rollouts fetched from R2
- Delta checkpoint publishing (~99% bandwidth reduction vs full checkpoints)
- Adaptive KL, importance sampling, and chunked logit computation for memory efficiency

#### 5. Communication & Storage (`grail/infrastructure/comms.py`)
Object-storage utilities for miner/validator/trainer coordination:
- Upload mined rollouts, publish validated rollouts, checkpoint management via R2

#### 6. Randomness & Chain
- Randomness (`grail/infrastructure/drand.py`): Robust drand v2-first client with fallbacks and a mock beacon for testing
- Chain & credentials (`grail/infrastructure/chain.py`): Manages R2 credential commitments and metagraph access

#### 7. CLI (`grail/cli/`)
Typer-based CLI with subcommands: `mine`, `validate`, `train`.

## How It Works

### Post-Training Flow

1. **Problem Generation**: The active environment generates problems using public randomness derived from drand and the window's block hash
2. **Rollout Collection**: Miners generate 16 GRPO rollouts per problem, tracking token ids and logprobs for proof construction
3. **GRAIL Verification**: Validators verify tokens, the GRAIL commitment/opening against the claimed model, and environment-specific evaluation (e.g., kernel correctness for Triton Kernel)
4. **Reward & Weights**: Validators score miners based on unique valid rollouts with a superlinear curve (`SUPERLINEAR_EXPONENT = 4.0`), then normalize and set weights on-chain
5. **Model Updates**: The trainer collects validated rollouts, runs GRPO training, and publishes updated model checkpoints to R2 each window

### Verifiable Inference

The GRAIL protocol ensures:
- Deterministic, publicly auditable challenges (drand + chain context)
- Model-binding proof of token processing; no substitution or replay
- Environment-agnostic verification: the protocol works across all supported environments

## Technical Details

### Protocol & Config (from `grail/shared/constants.py`)
- **PRIME_Q**: 2,147,483,647 (mod prime for sketches)
- **CHALLENGE_K**: 16 (minimum challenged positions)
- **PROOF_BATCH_SIZE**: 16 (fixed constant for miner/validator numerical consistency)
- **WINDOW_LENGTH**: 30 blocks per scoring window
- **ROLLOUTS_PER_PROBLEM**: 16

### Supported Environments
- **Triton Kernel** (current default): GPU kernel generation — the model writes Triton kernels evaluated for correctness on a dedicated GPU
- **3-SAT**: Variables 3–10, Clauses 5–20, Clause length 3; deterministic from seed
- **GSM8K**: Math word problems from the GSM8K dataset with step-by-step reasoning verification
- **MATH**: Competition-level mathematics from the Hendrycks MATH dataset
- **MBPP**: Python code generation from the Mostly Basic Python Problems benchmark
- **HumanEval**: Function-level code generation from the OpenAI HumanEval benchmark
- **Affine Trace/Logic**: Affine type system environments for trace and logic reasoning

### Model Requirements
- Hugging Face Transformers compatible, exposes token ids/logprobs
- **Text-only environments** (SAT, GSM8K, MATH, MBPP, HumanEval): 1 GPU minimum; any CUDA-capable accelerator
- **Triton Kernel environment**: 3 GPUs recommended — one for model inference (decoding), one for proof/logprob computation, and one for kernel evaluation. The kernel evaluation GPU should be A100 or H100 class to support Triton JIT compilation

For detailed hardware specifications, see [`compute.min.yaml`](compute.min.yaml).

## Setup

For detailed setup instructions, please refer to the appropriate documentation:

### Mining Setup
See [Miner Documentation](docs/miner.md) for comprehensive setup instructions including:
- Hardware and environment requirements
- Wallet and network configuration
- R2/S3 credentials setup
- Pipeline mode configuration (3-GPU)
- Running the miner

### Validation Setup
See [Validator Documentation](docs/validator.md) for comprehensive setup instructions including:
- Hardware and environment requirements
- Docker Compose or native deployment
- Wallet and network configuration
- Running the validator

### Quick Start
```bash
# Clone and install
git clone https://github.com/one-covenant/grail
cd grail
uv venv && source .venv/bin/activate
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your wallet names, network, and R2 credentials

# Run miner
grail mine

# Run validator
grail validate
```

**Important Notes:**
- Randomness is fetched from drand; miners mix it with the window's block hash
- Rollouts are uploaded to object storage (R2/S3); validators fetch, verify, score, and set weights
- Model checkpoints evolve through training and are automatically loaded each window
- For monitoring:
  - Miners and validators can log detailed metrics to the public W&B project: https://wandb.ai/tplr/grail
  - Real-time system logs and network statistics are available at the Grafana dashboard: https://grail-grafana.tplr.ai/

## Architecture Benefits

1. **Verifiable Training**: Cryptographic binding of rollouts to model and input
2. **Decentralized Post-Training**: Internet-scale contribution and evaluation
3. **Environment Agnostic**: Modular framework supports multiple problem domains
4. **Incentive Aligned**: On-chain weights reward sustained, verifiable improvements

## Contributing

We welcome contributions to:
- New environments and reward vectors
- Protocol robustness and verification
- Performance and throughput improvements
- Documentation and examples
