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
    1. Reached exactly MAX_NEW_TOKENS (max length termination)
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

            # Reject if exceeds max
            if isinstance(completion_len, int) and completion_len > MAX_NEW_TOKENS:
                logger.debug(f"Exceeds max tokens: {completion_len} > {MAX_NEW_TOKENS}")
                ctx.checks[self.check_name] = False
                return False

            # Check max length termination
            if completion_len == MAX_NEW_TOKENS:
                ctx.checks[self.check_name] = True
                return True

            # Check EOS termination
            if tokens[-1] != ctx.tokenizer.eos_token_id:
                logger.debug("Not EOS token and not max length")
                ctx.checks[self.check_name] = False
                return False

            # Use cached logits from proof validator
            logits = ctx.cached_logits
            if logits is None:
                # Fallback: compute logits
                logger.debug("Recomputing logits for EOS probability")
                logits = self._compute_logits(ctx)
                if logits is None:
                    logger.debug("Cannot verify EOS probability: no logits")
                    ctx.checks[self.check_name] = False
                    return False

            # Check EOS probability
            probs = torch.softmax(logits, dim=-1)
            p_eos = float(probs[ctx.tokenizer.eos_token_id].item())

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

    def _compute_logits(self, ctx: ValidationContext) -> torch.Tensor | None:
        """Compute logits if not cached."""
        try:
            tokens = ctx.commit.get("tokens", [])
            if len(tokens) < 2:
                return None

            full_ids = torch.tensor(tokens, dtype=torch.long, device=ctx.device).unsqueeze(0)
            with torch.inference_mode():
                outs = ctx.model(full_ids)

            if outs.logits.size(1) < 2:
                return None

            return outs.logits[0, -2, :].detach().to("cpu")
        except Exception as e:
            logger.debug(f"Logits computation failed: {e}")
            return None
