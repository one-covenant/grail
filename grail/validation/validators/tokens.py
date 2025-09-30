"""Token validation for GRAIL rollouts.

Validates that token IDs are within vocabulary bounds and sequence
length is within model limits. Must run before proof validation.
"""

from __future__ import annotations

import logging

from ...protocol.tokens import verify_tokens
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class TokenValidator(Validator):
    """Validates token IDs against model vocabulary and sequence limits.

    Checks:
    - All token IDs are within vocabulary bounds (0 <= id < vocab_size)
    - Sequence length doesn't exceed model's max context length
    - Tokens list is non-empty

    This catches tokens generated with a different model/tokenizer.
    """

    check_name = "tokens_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Validate tokens against model config."""
        tokens = ctx.commit.get("tokens", [])

        if not tokens:
            logger.debug("Empty tokens list")
            ctx.checks[self.check_name] = False
            return False

        # Use protocol.tokens.verify_tokens() for vocab/length checks
        is_valid = verify_tokens(tokens, ctx.model.config)

        ctx.checks[self.check_name] = is_valid

        if not is_valid:
            logger.debug(
                f"Token validation failed: {len(tokens)} tokens, "
                f"vocab_size={getattr(ctx.model.config, 'vocab_size', 'unknown')}"
            )

        return is_valid
