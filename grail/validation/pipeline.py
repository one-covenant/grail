"""Validation pipeline orchestration for GRAIL."""

from __future__ import annotations

import logging

from .base import Validator
from .context import ValidationContext
from .validators import (
    DistributionValidator,
    GRAILProofValidator,
    SATProblemValidator,
    SATPromptValidator,
    SATSolutionValidator,
    SchemaValidator,
    TerminationValidator,
    TokenValidator,
)

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


def get_hard_check_keys(pipeline: ValidationPipeline) -> tuple[str, ...]:
    """Extract hard check names from pipeline validators.

    Args:
        pipeline: Validation pipeline instance

    Returns:
        Tuple of check names for validators with severity="hard"
    """
    return tuple(v.check_name for v in pipeline.validators if v.severity == "hard")


def get_soft_check_keys(pipeline: ValidationPipeline) -> tuple[str, ...]:
    """Extract soft check names from pipeline validators.

    Args:
        pipeline: Validation pipeline instance

    Returns:
        Tuple of check names for validators with severity="soft"
    """
    return tuple(v.check_name for v in pipeline.validators if v.severity == "soft")


def create_sat_validation_pipeline() -> ValidationPipeline:
    """Create the standard SAT validation pipeline.

    Returns a pipeline with validators in dependency order:
    1. Schema validation (basic structure)
    2. Token validation (token IDs are valid)
    3. GRAIL proof validation (cryptographic proof)
    4. SAT problem validation (problem regenerates correctly)
    5. SAT prompt validation (prompt matches canonical form)
    6. Termination validation (completion ends properly)
    7. SAT solution validation (solution is correct)
    8. Distribution validation (logprobs match model)

    Returns:
        ValidationPipeline configured for SAT environment
    """
    validators = [
        SchemaValidator(),  # FIRST - structure/types, no GPU
        TokenValidator(),  # SECOND - vocab/length check
        SATProblemValidator(),  # This always should be called before SATPromptValidator
        SATPromptValidator(),
        GRAILProofValidator(),  # Cryptographic proof validation
        TerminationValidator(),
        DistributionValidator(),
        SATSolutionValidator(),
    ]

    logger.info(f"Created SAT validation pipeline with {len(validators)} validators")
    return ValidationPipeline(validators)
