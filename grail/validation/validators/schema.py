"""Schema validation using Pydantic models.

Validates rollout structure, types, and value ranges before expensive
GPU-based validation. Catches malformed submissions early.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from ...schemas.rollout import Commit
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class SchemaValidator(Validator):
    """Validates rollout conforms to Pydantic schema.

    This validator runs FIRST (before GPU validators) to filter out:
    - Missing required fields
    - Wrong types (str instead of int, etc.)
    - Out-of-range values (completion_length > MAX_NEW_TOKENS)
    - Cross-field inconsistencies (commitments length != tokens length)

    Uses Pydantic for automatic validation with clear error messages.
    """

    check_name = "schema_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Validate commit against Pydantic schema with detailed error logging."""
        try:
            # Validate commit structure using Pydantic
            Commit(**ctx.commit)

            # Note: Full RolloutData validation would need top-level fields
            # Currently we only validate commit since that's what's passed in ctx
            # Top-level fields (window_start, nonce, etc.) validated in validate.py

            ctx.checks[self.check_name] = True
            return True

        except ValidationError as e:
            # Pydantic provides structured, detailed error information
            errors = []
            error_details = []

            for error in e.errors():
                # Build field path (e.g., "commit.tokens" or "commit.rollout.completion_length")
                field_path = ".".join(str(x) for x in error["loc"])
                error_type = error["type"]
                msg = error["msg"]
                input_val = error.get("input")

                # Create detailed error message
                if error_type == "missing":
                    detailed_msg = f"❌ MISSING FIELD: '{field_path}' is required but not provided"
                elif error_type == "int_parsing" or error_type == "int_type":
                    detailed_msg = (
                        f"❌ TYPE ERROR: '{field_path}' must be an integer, "
                        f"got {type(input_val).__name__}: {input_val}"
                    )
                elif error_type == "list_type":
                    detailed_msg = (
                        f"❌ TYPE ERROR: '{field_path}' must be a list, "
                        f"got {type(input_val).__name__}"
                    )
                elif error_type == "bool_type":
                    detailed_msg = (
                        f"❌ TYPE ERROR: '{field_path}' must be a boolean, "
                        f"got {type(input_val).__name__}: {input_val}"
                    )
                elif any(x in error_type for x in ["greater_than", "less_than", "range"]):
                    detailed_msg = f"❌ VALUE ERROR: '{field_path}' {msg}, got {input_val}"
                elif error_type == "value_error":
                    # Custom validator error (our cross-field checks)
                    detailed_msg = f"❌ VALIDATION ERROR: '{field_path}' - {msg}"
                else:
                    # Generic error
                    detailed_msg = f"❌ ERROR: '{field_path}' - {msg} (type: {error_type})"

                errors.append(f"{field_path}: {msg}")
                error_details.append(detailed_msg)

            ctx.checks[self.check_name] = False
            ctx.metadata["schema_errors"] = errors
            ctx.metadata["schema_error_details"] = error_details

            # Log clear, actionable errors for miners
            logger.warning(f"Schema validation FAILED with {len(errors)} error(s):")
            for detail in error_details[:10]:  # Log up to 10 detailed errors
                logger.warning(f"  {detail}")
            if len(error_details) > 10:
                logger.warning(f"  ... and {len(error_details) - 10} more errors")

            # Also log in debug mode with full Pydantic error context
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Full Pydantic validation errors:\n{e}")

            return False

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Schema validation crashed unexpectedly: {e}", exc_info=True)
            ctx.checks[self.check_name] = False
            ctx.metadata["schema_errors"] = [f"validation_crash: {e}"]
            return False
