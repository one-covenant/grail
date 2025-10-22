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

#### 5. Randomness & Chain
- Randomness (`grail/infrastructure/drand.py`): Robust drand v2-first client with fallbacks and a mock beacon for testing
- Chain & credentials (`grail/infrastructure/chain.py`): Manages R2 credential commitments and metagraph access

#### 6. CLI (`grail/cli/`)
Typer-based CLI with subcommands: `mine`, `validate` (and experimental `train`).

 Best practices for miners:
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
- **WINDOW_LENGTH**: 50 blocks per scoring window

### Supported Environments
- **3-SAT**: Variables 3–10, Clauses 5–20, Clause length 3; deterministic from seed
- **GSM8K**: Math word problems from the GSM8K dataset with step-by-step reasoning verification

### Model Requirements
- Hugging Face Transformers compatible, exposes token ids/logprobs
- **OS and hardware-agnostic**: Runs on any platform with floating point precision within tolerance
- Accelerators (GPU/TPU) recommended for throughput

For detailed hardware specifications, see [`compute.min.yaml`](compute.min.yaml).

## Setup

For detailed setup instructions, please refer to the appropriate documentation:

### Mining Setup
See [Miner Documentation](docs/miner.md) for comprehensive setup instructions including:
- Hardware and environment requirements
- Wallet and network configuration
- R2/S3 credentials setup
- Dependency installation
- Running the miner

### Validation Setup
See [Validator Documentation](docs/validator.md) for comprehensive setup instructions including:
- Hardware and environment requirements
- Wallet and network configuration
- Dependency installation
- Running the validator

### Quick Start
```bash
# Install dependencies
uv sync

# Run miner
grail mine

# Run validator
grail validate
```

**Important Notes:**
- Randomness is fetched from drand; miners mix it with the window's block hash
- Rollouts are uploaded to object storage (R2/S3); validators fetch, verify, score, and set weights
- For monitoring:
  - Miners and validators can log detailed metrics to the public W&B project: https://wandb.ai/tplr/grail
  - Real-time system logs and network statistics are available at the Grafana dashboard: https://grail-grafana.tplr.ai/

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
