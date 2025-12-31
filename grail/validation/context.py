"""Validation context for GRAIL rollout verification.

The ValidationContext carries all state through the validation pipeline,
enabling validators to:
- Access model/tokenizer without coupling
- Cache intermediate results (logits, parsed problems)
- Accumulate check results and metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class ValidationContext:
    """Shared context passed through validation pipeline.

    Validators read inputs, use resources, cache results, and update checks.
    """

    # Inputs (immutable)
    commit: dict
    prover_address: str
    challenge_randomness: str

    # Resources (shared, immutable)
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device

    # Trusted validator-derived values (never trust miner data for these)
    window_hash: str = ""  # Window block hash from validator's chain query
    group_index: int = 0  # File-order group index for deterministic seed derivation

    # Optional identifiers / logging context (defaulted fields must follow non-defaults)
    miner_uid: str | None = None  # Miner UID for logging/metrics namespacing

    # Environment configuration from checkpoint metadata
    env_id: str | None = None  # Environment identifier from checkpoint
    env_params: dict[str, Any] = field(default_factory=dict)  # Environment params from checkpoint

    # Cached intermediate results (populated by validators)
    cached_logits: torch.Tensor | None = None
    verified_problem: Any | None = None  # SATProblem, CodeProblem, etc.

    # Results accumulators (updated by validators)
    checks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
