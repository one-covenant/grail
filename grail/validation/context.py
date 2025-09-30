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
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device

    # Cached intermediate results (populated by validators)
    cached_logits: torch.Tensor | None = None
    verified_problem: Any | None = None  # SATProblem, CodeProblem, etc.

    # Results accumulators (updated by validators)
    checks: dict[str, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
