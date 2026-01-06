"""Base validator interface for GRAIL validation pipeline."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

from .context import ValidationContext

logger = logging.getLogger(__name__)


class Validator(ABC):
    """Base interface for all validators.

    Each validator:
    - Implements one focused check
    - Updates ctx.checks with result
    - May cache results in ctx for downstream validators
    - Returns bool (pass/fail)
    - Declares severity (hard or soft)
    """

    # Subclasses must define these class attributes
    check_name: str
    """Name of this check (e.g., 'proof_valid', 'sat_problem_valid')."""

    severity: str = "hard"
    """Severity: 'hard' (immediate rejection) or 'soft' (threshold-based)."""

    @abstractmethod
    def validate(self, ctx: ValidationContext) -> bool:
        """Run validation check.

        Args:
            ctx: Validation context with commit, model, and accumulators

        Returns:
            True if check passes, False otherwise

        Side effects:
            Updates ctx.checks[self.check_name]
            May update ctx.cached_logits, ctx.verified_problem, ctx.metadata
        """
        pass

    def compute_logits(
        self, ctx: ValidationContext, full_sequence: bool = True
    ) -> torch.Tensor | None:
        """Compute logits via model inference (fallback when not cached).

        Args:
            ctx: Validation context
            full_sequence: If True, return full [seq_len, vocab_size] tensor.
                          If False, return only second-to-last token logits.

        Returns:
            Logits tensor or None if computation fails
        """
        try:
            tokens = ctx.commit.get("tokens", [])
            if not tokens or (not full_sequence and len(tokens) < 2):
                return None

            full_ids = torch.tensor(tokens, dtype=torch.long, device=ctx.device).unsqueeze(0)

            with torch.inference_mode():
                outs = ctx.model(full_ids)

            if full_sequence:
                return outs.logits[0].detach().to("cpu")
            else:
                # Return second-to-last token for EOS checks
                if outs.logits.size(1) < 2:
                    return None
                return outs.logits[0, -2, :].detach().to("cpu")

        except Exception as e:
            logger.debug(f"Logits computation failed: {e}")
            return None
