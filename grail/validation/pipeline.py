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
        """Run all validators until one fails or all pass.

        Args:
            ctx: Validation context with commit, proof, model, etc.

        Returns:
            Tuple of (is_valid, checks_dict)
        """
        for validator in self.validators:
            try:
                passed = validator.validate(ctx)
                if not passed:
                    logger.debug(f"Validation failed at check: {validator.check_name}")
                    return False, ctx.checks
            except Exception as e:
                logger.error(f"Validator {validator.check_name} raised exception: {e}")
                ctx.checks[validator.check_name] = False
                return False, ctx.checks

        return True, ctx.checks
