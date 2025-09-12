<h1 align="center">grail: Verifiable Post-Training for LLMs</h1>

<div align="center">
  <pre>
GGGGG RRRRR AAAAA IIIII L
    G     R   R A   A   I   L    
    G  GG RRRR  AAAAA   I   L    
    G   G R  R  A   A   I   L    
    GGGGG R   R A   A IIIII LLLLL
  </pre>
</div>

<p align="center">
  Documentation:
  <a href="docs/miner.md">Miner</a> •
  <a href="docs/validator.md">Validator</a> •
</p>

<p align="center">
  <a href="https://codecov.io/gh/tplr-ai/grail">
    <img src="https://codecov.io/gh/tplr-ai/grail/branch/main/graph/badge.svg" alt="Codecov" />
  </a>
  <a href="https://github.com/tplr-ai/grail/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/tplr-ai/grail" alt="License" />
  </a>
  <a href="https://github.com/tplr-ai/grail/commits/main">
    <img src="https://img.shields.io/github/last-commit/tplr-ai/grail" alt="Last Commit" />
  </a>
</p>


## Overview

**grail** delivers post-training for language models with cryptographically verifiable inference. It implements the **GRAIL protocol** (Guaranteed Rollout Authenticity via Inference Ledger) so that rollouts produced during RL are tied to a specific model and input, and can be independently verified by validators.

### Key Distinction: grail vs GRAIL

- **grail** (lowercase): The Bittensor subnet implementation orchestrating miners and validators for verifiable post-training
- **GRAIL** (uppercase): The protocol that proves rollout authenticity and model identity

### Current Status

- The current release is inference-only: miners generate rollouts and validators verify and score them.
- Reinforcement learning post-training (e.g., GRPO trainer and model updates) will be added in a future version.

## Architecture

### Core Components

#### 1. GRAIL Protocol (`grail/grail.py`)
Prover/Verifier implementation with:
- PRF-based index derivation and sketch commitments for token-level verification
- Verifier-supplied challenge (drand + chain/window context)
- Token and model-config validation; structured signatures bound to model identity
- SAT problem binding and solution checks for end-to-end rollout verification

#### 2. Rollout Generation (`grail/mining/rollout_generator.py`, `grail/environments/sat.py`)
GRPO-style rollout system with:
- Multiple rollouts per problem, token-level logprob tracking, advantage computation
- Qwen-style chat template injection for reasoning/solution tagging
- SAT-specific `SATRolloutGenerator` with modular reward vector composition

#### 3. Environment System (`grail/environments/`)
Modular environments, currently:
- **SAT Problems** (`sat.py`): Deterministic 3-SAT generation, parsing, reward shaping

#### 4. Communication & Storage (`grail/infrastructure/comms.py`)
Object-storage utilities for miner/validator coordination:
- Upload mined rollouts (`sink_window_inferences`), publish validated rollouts (`upload_valid_rollouts`)
- Optional dataset export to Hugging Face (`upload_to_huggingface`)

#### 5. Randomness & Chain
- Randomness (`grail/infrastructure/drand.py`): Robust drand v2-first client with fallbacks and a mock beacon for testing
- Chain & credentials (`grail/infrastructure/chain.py`): Manages R2 credential commitments and metagraph access

#### 6. CLI (`grail/cli/`)
Typer-based CLI with subcommands: `mine`, `validate` (and experimental `train`).

Best practices for miners:
- Do not override model-related environment variables (`GRAIL_MODEL_NAME`, `GRAIL_MAX_NEW_TOKENS`).
- Leave the final 2 blocks of each window for upload; generation should stop near the end automatically.
- Prefer `uv sync` for reproducible installs.

## How It Works

### Post-Training Flow

1. **Problem Generation**: Validators derive a SAT instance from a public seed that mixes drand randomness with the window’s block hash
2. **Rollout Collection**: Miners generate multiple GRPO rollouts, tracking token ids and logprobs for proof construction
3. **GRAIL Verification**: Validators verify tokens, the GRAIL commitment/opening against the claimed model, the deterministic SAT instance, and the reported solution
4. **Reward & Weights**: Validators score miners over recent windows using unique/valid/successful rollout metrics with a superlinear curve, then normalize and set weights on-chain
5. **Model Updates (planned)**: Validated rollouts will be used for post-training in a future release

### Verifiable Inference

The GRAIL protocol ensures:
- Deterministic, publicly auditable challenges (drand + chain context)
- Model-binding proof of token processing; no substitution or replay
- Deterministic SAT instance reconstruction and solution verification

## Technical Details

### Protocol & Config (from `grail/shared/constants.py`)
- **PRIME_Q**: 2,147,483,647 (mod prime for sketches)
- **CHALLENGE_K**: 16 (minimum challenged positions)
- **TOLERANCE**: 3 (numeric tolerance for comparisons)
- **MODEL_NAME**: default `Qwen/Qwen3-4B-Instruct-2507` (override via `GRAIL_MODEL_NAME`)
- **MAX_NEW_TOKENS**: configurable generation cap (default 1024 via env)
- **WINDOW_LENGTH**: 50 blocks per scoring window

### Supported Environments
- **3-SAT**: Variables 3–10, Clauses 5–20, Clause length 3; deterministic from seed

### Model Requirements
- Hugging Face Transformers compatible, exposes token ids/logprobs
- CUDA recommended for throughput

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone https://github.com/tplr-ai/grail
cd grail

# Create a venv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

## Usage

### Mining

```bash
# Copy then fill out env items (wallets, network, R2 credentials)
cp .env.example .env

# Run miner locally
grail mine
```

### Validating

```bash
# Copy then fill out env items
cp .env.example .env

# Run validator locally
grail validate
```

Notes:
- Randomness is fetched from drand; miners mix it with the window’s block hash
- Rollouts are uploaded to object storage (R2/S3); validators fetch, verify, score, and set weights
- Validated rollouts can be exported to a Hugging Face dataset for analysis

## Architecture Benefits

1. **Verifiable Training**: Cryptographic binding of rollouts to model and input
2. **Decentralized Post-Training**: Internet-scale contribution and evaluation
3. **Problem Agnostic**: Environment framework enables new domains beyond SAT
4. **Incentive Aligned**: On-chain weights reward sustained, verifiable improvements

## Contributing

We welcome contributions to:
- New environments and reward vectors
- Protocol robustness and verification
- Performance and throughput improvements
- Documentation and examples
