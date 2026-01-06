"""Termination validation for GRAIL rollouts."""

from __future__ import annotations

import logging

import torch

from ...shared.constants import MAX_NEW_TOKENS, MIN_EOS_PROBABILITY
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class TerminationValidator(Validator):
    """Verifies rollout terminated via max length or confident EOS.

    Rollouts must terminate in one of two ways:
    1. Reached the expected max completion length (max length termination)
    2. Ended with EOS token having probability >= MIN_EOS_PROBABILITY
    """

    check_name = "termination_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify termination condition."""
        try:
            tokens = ctx.commit.get("tokens", [])
            if not tokens:
                logger.debug("No tokens for termination check")
                ctx.checks[self.check_name] = False
                return False

            rollout = ctx.commit.get("rollout", {})
            completion_len = rollout.get("completion_length")

            # Prefer checkpoint-provided generation max_tokens when available.
            # This avoids false failures when miners/validators intentionally use
            # shorter completions (e.g., max_tokens=512 for code tasks).
            expected_max_new_tokens = MAX_NEW_TOKENS
            max_tokens_raw = ctx.generation_params.get("max_tokens")
            if max_tokens_raw is not None:
                try:
                    max_tokens_int = int(max_tokens_raw)
                    if max_tokens_int > 0:
                        expected_max_new_tokens = min(max_tokens_int, MAX_NEW_TOKENS)
                except (TypeError, ValueError):
                    pass

            # Reject if exceeds max
            if isinstance(completion_len, int) and completion_len > expected_max_new_tokens:
                logger.debug("Exceeds max tokens: %s > %s", completion_len, expected_max_new_tokens)
                ctx.checks[self.check_name] = False
                return False

            # Check max length termination
            if completion_len == expected_max_new_tokens:
                ctx.checks[self.check_name] = True
                return True

            # Check EOS termination
            eos_id = getattr(ctx.tokenizer, "eos_token_id", None)
            if eos_id is None:
                logger.debug("Tokenizer missing eos_token_id")
                ctx.checks[self.check_name] = False
                return False
            if tokens[-1] != eos_id:
                logger.debug("Not EOS token and not max length")
                ctx.checks[self.check_name] = False
                return False

            # Use cached logits from proof validator (full sequence)
            if ctx.cached_logits is None:
                # Fallback: compute logits (uses base class method)
                logger.debug("Recomputing logits for EOS probability")
                logits = self.compute_logits(ctx, full_sequence=False)
                if logits is None:
                    logger.debug("Cannot verify EOS probability: no logits")
                    ctx.checks[self.check_name] = False
                    return False
            else:
                # Extract second-to-last token's logits from cached full tensor
                if ctx.cached_logits.dim() == 1:
                    # Already extracted (shouldn't happen with new caching)
                    logits = ctx.cached_logits
                elif len(tokens) >= 2:
                    logits = ctx.cached_logits[-2]
                else:
                    logger.debug("Insufficient tokens for EOS check")
                    ctx.checks[self.check_name] = False
                    return False

            # Check EOS probability
            probs = torch.softmax(logits, dim=-1)
            p_eos = float(probs[eos_id].item())

            if p_eos >= MIN_EOS_PROBABILITY:
                ctx.checks[self.check_name] = True
                return True

            logger.debug(f"EOS prob too low: {p_eos:.4f} < {MIN_EOS_PROBABILITY}")
            ctx.checks[self.check_name] = False
            return False

        except Exception as e:
            logger.debug(f"Termination check error: {e}")
            ctx.checks[self.check_name] = False
            return False
