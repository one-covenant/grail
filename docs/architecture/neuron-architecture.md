## GRAIL Neuron Architecture and Incremental Refactor Plan

### Purpose
This document defines a simple, neuron-based architecture for GRAIL that:
- Preserves the current behavior of miner, validator, and (future) trainer
- Improves maintainability and testability with minimal churn
- Aligns with Bittensor practices and sound software engineering principles

It also outlines the incremental refactor plan and the target repository structure.

---

## First principles and requirements

### Functional (Bittensor)
- Deterministic, window-aligned processing and safe throttling of on-chain `set_weights`
- Robust subtensor interactions with timeouts/retries/backoff
- Reliable object-storage I/O for uploads/downloads
- HF auth/usage for datasets and models
- GPU-safe operation where applicable (eval mode; no-grad; bounded memory)

### Non-functional (SE best practices)
- Clear lifecycles for long-running processes (start/stop/health)
- Separation of concerns: domain logic vs orchestration vs infrastructure
- Typed public interfaces; explicit configuration; structured logging
- Minimal dependencies; consistent model handling
- Easy unit/integration testing; incremental, low-risk changes

---

## Architecture overview

The architecture is organized into layers:

- Neurons (process lifecycle)
  - BaseNeuron lifecycle (signal handling, stop_event, optional window change event)
  - MinerNeuron, ValidatorNeuron, TrainerNeuron compose services and run forever loops

- Services (orchestration)
  - Mining: problem selection → rollout generation → packaging → upload
  - Validation: discovery → dedup → verify → score → publish/weights
  - Training: fetch valid data → train → save/checkpoint

- Protocol (pure logic)
  - Prover/Verifier, commitments, challenges, signatures
  - No I/O; deterministic, testable code

- Model provider
  - Centralized tokenizer/model initialization (dtype/device, pad token, chat templates)

- Infrastructure
  - Subtensor (network) adapters, chain/credential management, drand, object storage

- CLI
  - Thin Typer commands that parse options and start the appropriate neuron

- Monitoring
  - Backend-agnostic manager (e.g., Weights & Biases) with consistent context and metrics

---

## Neuron layer

### BaseNeuron (slim, shared)
- Responsibilities:
  - Install signal handlers
  - Maintain a shared `stop_event`
  - Optionally maintain `current_block` and `current_window` and an `asyncio.Event` (window_changed)
  - Provide `wait_until_window()` helper
  - Clean shutdown for background tasks/threads

- Optional extension:
  - Replace polling with a block-header subscription thread when needed

### Role neurons
- MinerNeuron
  - Window-driven rollout generation and uploads
- ValidatorNeuron
  - Window-driven discovery, verification, scoring, and weight submission (throttled)
- TrainerNeuron (future)
  - Window-driven training cycles and checkpoint uploads
  - Optional profiler hooks

Neurons should be very thin: they should host lifecycle and call into role-specific services.

---

## Services layer (extraction target)

- Mining service
  - Orchestrates window loop, uses rollout generator, packages proofs, uploads

- Validation service
  - Orchestrates window loop, uses copycat/dedup, verifier, scoring, publishing and on-chain weights

- Training service (later)
  - Pulls validated data, runs trainer, and uploads future-window checkpoints

Each service remains small, focused, and testable with fakes.

---

## Protocol layer (split target)
- Prover/Verifier (model-bound proof logic)
- Commitments/Challenges/Signatures
- Deterministic; no I/O; unit-test friendly

---

## Model provider
- One place to load `AutoTokenizer` and `AutoModelForCausalLM` with:
  - Device/dtype policy
  - Pad token handling
  - Chat template setup
- Ensures consistency across miner, validator, trainer

---

## Infrastructure
- Network/subtensor: creation and (optional) header subscription helper with backoff
- Chain manager + credential management
- Drand client
- Object storage I/O (read/write; chunking; retries)

---

## CLI
- Maintain Typer UX; CLI commands become thin wrappers that construct and run neurons

---

## Monitoring
- Uniform run contexts and namespaces (uid/netuid/block/window)
- Minimal required metrics; consistent names across roles

---

## Design tradeoffs and decisions

### Layering depth
- Tradeoff: Deeper layers → cleaner separation but more files; shallower → faster to change but more entanglement
- Decision: Stage 1 introduces Neurons only; Stage 2 extracts Services/Protocol/Model Provider

### Window handling: polling vs subscription
- Tradeoff: Subscription offers low latency but adds complexity; polling is simple and sufficient at 12s blocks
- Decision: Start with polling; add a subscription thread only when needed

### Central model provider
- Tradeoff: Centralization adds indirection; reduces config drift and subtle bugs
- Decision: Add in Stage 2, keep interfaces small and explicit

### Axon/Dendrite
- Tradeoff: Powerful but increases surface area and ops complexity
- Decision: Defer; neuron lifecycle leaves the door open

### Repo granularity
- Tradeoff: Many micro-packages vs pragmatic few
- Decision: Add only `neurons/` in Stage 1; expand to `services/`, `protocol/`, `model/` in Stage 2

### Import stability vs clarity
- Tradeoff: Moving files breaks imports; re-exports hide structure
- Decision: Use re-exports during migration; remove once downstream code is updated

---

## Target repository structure

### Stage 1 (minimal change; add neurons only)
- Behavior remains identical. Neurons delegate to existing code.

```
grail/
  cli/
    __init__.py
    mine.py                 # existing
    validate.py             # existing
    train.py                # existing (WIP)
  neurons/
    __init__.py
    base.py                 # lifecycle (signals, stop_event, optional window_changed)
    miner.py                # wraps current mining loop
    validator.py            # wraps current validation loop
    trainer.py              # scaffolding for future training loop
  environments/
  infrastructure/
  monitoring/
  shared/
  mining/
  validation/
  grail.py                  # legacy protocol (to be split later)
  # No other moves in Stage 1
```

### Stage 2 (clear boundaries; optional re-exports for stability)
- Extract orchestration into services; split protocol; centralize model handling.

```
grail/
  cli/
    mine.py                 # instantiate MinerNeuron().main()
    validate.py             # instantiate ValidatorNeuron().main()
    train.py                # instantiate TrainerNeuron().main()
  neurons/
    base.py
    miner.py
    validator.py
    trainer.py
    health.py               # optional health/readiness utilities
  services/
    mining/
      service.py            # outer loop orchestration
      pipeline.py           # problem → generate → package → upload
    validation/
      service.py            # outer loop orchestration
      pipeline.py           # discover → dedup → verify → score → publish
      dedup.py              # wraps validation/copycat.py
      scoring.py
    training/
      service.py
      pipeline.py
  protocol/
    prover.py
    verifier.py
    commitments.py
    challenges.py
    signatures.py
    __init__.py             # re-export for import stability
  model/
    provider.py             # tokenizer/model factory
    hf_compat.py            # move/re-export if needed
  environments/
  infrastructure/
    drand.py                # single source of truth (dedupe if needed)
  shared/
    logging.py              # single logging helper
    constants.py, types.py, schemas.py, subnet.py
  monitoring/
    manager.py, base.py, backends/
```

---

## Incremental refactor plan

### Stage 1 (neurons only; no behavior change)
- Add `neurons/` with `base.py`, `miner.py`, `validator.py`, `trainer.py`
- Keep CLI logic intact; neurons call existing functions
- Run linter/tests; ensure zero regressions

### Stage 2 (flip CLI to neurons)
- Update Typer commands to instantiate and run neurons
- Maintain existing flags and environment variables

### Stage 3 (extract services/protocol/model provider and consolidate duplicates)
- Move validation/mining loops to `services/`
- Split `grail/grail.py` into `protocol/` modules with re-exports
- Add `model/provider.py`; migrate Prover/Verifier
- Consolidate logging and drand duplicates
- Keep re-exports temporarily; remove later

Rollback: Each stage is independently reversible.

---

## Coding and dependency standards

- Package management: `uv`; pin versions in `pyproject.toml`; no manual lockfile edits
- Python: 3.9–3.11
- Allowed deps: torch, transformers, safetensors, numpy, typer, rich, python-dotenv, requests, aiobotocore/botocore, bittensor, huggingface_hub, datasets
- Style: PEP 8; max line length 100; typed public APIs; no wildcard imports
- Logging: `logging.getLogger(__name__)`; no prints; no secret logging
- Concurrency: `asyncio` for I/O; no event-loop blocking; timeouts on network calls
- Security: HTTPS, environment-based secrets loading, safetensors for model weights

---

## Testing approach

- Unit tests at the Protocol and Services layers (pure logic; fakes for infra)
- Integration tests for neuron lifecycles with stubbed network/storage
- Deterministic tests where feasible; seed RNGs when needed

---

## Bittensor-specific guidance

- Refresh metagraph only when block increases; cache by block number
- Throttle weight submissions (e.g., one per 360 blocks)
- Structured metrics with uid/netuid/block/window context
- Use eval mode and no-grad for validator/miner; manage GPU memory proactively

---

## Glossary

- Neuron: Long-running process (miner/validator/trainer) with a clean lifecycle
- Service: Orchestrated steps for a role (e.g., validation pipeline)
- Protocol: Model-proof logic (Prover/Verifier, commitments)
- Model provider: Centralized factory for tokenizer/model setup

---


