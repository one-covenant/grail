"""Base validator interface for GRAIL validation pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .context import ValidationContext


class Validator(ABC):
    """Base interface for all validators.

    Each validator:
    - Implements one focused check
    - Updates ctx.checks with result
    - May cache results in ctx for downstream validators
    - Returns bool (pass/fail)
    - Declares severity (hard or soft)
    """

    @property
    @abstractmethod
    def check_name(self) -> str:
        """Name of this check (e.g., 'proof_valid', 'sat_problem_valid')."""
        pass

    @property
    def severity(self) -> str:
        """Severity of this check: 'hard' or 'soft'.

        Hard checks: Failure causes immediate rejection
        Soft checks: Failure accumulates, threshold-based rejection
        """
        return "hard"  # Default to hard

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
