# grail - incentivising intelligence

## Overview

**grail** focuses on post-training language models with verifiable inference capabilities. It leverages the **GRAIL protocol** (Guaranteed Rollout Authenticity via Inference Ledger) to ensure cryptographically verifiable model outputs during reinforcement learning rollouts.

### Key Distinction: grail vs GRAIL

- **grail** (lowercase): The Bittensor subnet implementation for post-training models using verifiable inference
- **GRAIL** (uppercase): The underlying cryptographic protocol (Guaranteed Rollout Authenticity via Inference Ledger) that provides verifiable inference guarantees

## Architecture

### Core Components

#### 1. GRAIL Protocol (`grail/grail.py`)
The cryptographic protocol implementation that ensures verifiable inference through:
- Pseudorandom function (PRF) for deterministic token generation
- Sketch-based proof system for model output verification
- Integration with drand beacon for randomness
- Support for SAT problem generation and verification

#### 2. Rollout Generation (`grail/rollout.py`)
GRPO (Generalized Proximal Policy Optimization) rollout system for:
- Multiple rollouts per problem for robust training
- Token-level logprob tracking for GRAIL proofs
- Advantage computation for policy gradient training
- Trajectory recording for analysis

#### 3. Environment System (`grail/environments/`)
Modular environment framework supporting:
- **SAT Problems** (`sat.py`): 3-SAT problem generation and solving
- Extensible base for additional problem domains
- Text-based problem representation for LLM processing

#### 4. Communication Layer (`grail/comms.py`)
Handles subnet communication and coordination between validators and miners.

#### 5. Randomness (`grail/drand.py`)
Integration with drand beacon for verifiable randomness in problem generation.

## How It Works

### Post-Training Flow

1. **Problem Generation**: Validators generate verifiable problems (e.g., SAT instances) using drand beacon randomness
2. **Rollout Collection**: Miners generate multiple solution attempts using their models
3. **GRAIL Verification**: Each rollout is cryptographically verified using the GRAIL protocol
4. **Reward Computation**: Valid rollouts receive rewards based on solution quality
5. **Model Updates**: Miners update their models using GRPO with collected rollouts

### Verifiable Inference

The GRAIL protocol ensures that:
- Model outputs are deterministically verifiable
- Rollouts cannot be falsified or replayed
- Randomness is publicly verifiable through drand
- Token generation follows cryptographic constraints

## Technical Details

### GRAIL Protocol Parameters
- **PRIME_Q**: 2,147,483,647 (modular arithmetic prime)
- **CHALLENGE_K**: 16 (number of challenge tokens)
- **TOLERANCE**: 3 (allowed deviations in verification)

### Supported Environments
- **SAT Problems**: 3-SAT instances with configurable complexity
  - Variables: 3-10
  - Clauses: 5-20
  - Clause length: 3 (3-SAT)

### Model Requirements
- Compatible with HuggingFace transformers
- Support for logprob extraction
- CUDA-enabled for efficient inference

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

# Install GRAIL
uv pip install -e .
```

## Usage

### Mining

```bash
# Copy then fill out env items
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

## Architecture Benefits

1. **Verifiable Training**: All training data is cryptographically verifiable
2. **Decentralized Post-Training**: Distributed model improvement without centralized control
3. **Problem Agnostic**: Extensible to various problem domains beyond SAT
4. **Incentive Aligned**: Rewards genuine model improvements through verifiable performance

## Contributing

The grail subnet welcomes contributions for:
- New environment implementations
- Protocol optimizations
- Performance improvements
- Documentation enhancements
