"""Validation pipeline orchestration for GRAIL."""

from __future__ import annotations

import logging

from .base import Validator
from .context import ValidationContext

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Sequential validation pipeline with early exit on failure.

    Runs validators in order. Stops at first failure (fail-fast).
    """

    def __init__(self, validators: list[Validator]):
        self.validators = validators

    def validate(self, ctx: ValidationContext) -> tuple[bool, dict[str, bool]]:
        """Run all validators with proper hard/soft check handling.

        Hard checks: Exit immediately on failure
        Soft checks: Continue to next validator, accumulate failures

        Args:
            ctx: Validation context with commit, model, etc.

        Returns:
            Tuple of (is_valid, checks_dict)
            is_valid is False if any hard check failed
        """
        for validator in self.validators:
            try:
                passed = validator.validate(ctx)

                if not passed:
                    # Check severity (default to hard if not specified)
                    severity = getattr(validator, "severity", "hard")

                    if severity == "hard":
                        # Hard check failed - exit immediately
                        logger.debug(f"HARD check failed: {validator.check_name}")
                        return False, ctx.checks
                    else:
                        # Soft check failed - log but continue
                        logger.debug(f"SOFT check failed: {validator.check_name} (continuing)")
                        # Continue to next validator

            except Exception as e:
                logger.error(f"Validator {validator.check_name} raised exception: {e}")
                ctx.checks[validator.check_name] = False
                # Treat crashes as hard failures
                return False, ctx.checks

        return True, ctx.checks
