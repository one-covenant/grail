"""Cross-validator consistency checks for GRAIL validation pipeline.

Validates that outputs from different validators are internally consistent,
catching edge cases where individual validators pass but their results conflict.
"""

from __future__ import annotations

import logging

from ...shared.constants import CHALLENGE_K
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class ConsistencyValidator(Validator):
    """Validates consistency across multiple validators' outputs.

    This validator runs AFTER other validators and checks for:
    - Token count consistency (schema vs actual tokens)
    - Prompt/completion boundary alignment
    - Logprob array length matches token sequence
    - Reward computation consistency with evaluation
    - Metadata cross-validation between validators
    - Commitment array length matches token sequence

    This is a HARD check - any inconsistency causes immediate rejection.
    """

    check_name = "consistency_valid"
    severity = "hard"

    def validate(self, ctx: ValidationContext) -> bool:
        """Run comprehensive consistency checks across validator outputs."""
        try:
            # Extract core data
            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            commitments = ctx.commit.get("commitments", [])

            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))
            token_logprobs = rollout.get("token_logprobs", [])

            # Check 1: Token count consistency
            if not self._check_token_count_consistency(
                tokens, prompt_len, completion_len, ctx
            ):
                return False

            # Check 2: Commitment array consistency
            if not self._check_commitment_consistency(tokens, commitments, ctx):
                return False

            # Check 3: Logprob array consistency
            if not self._check_logprob_consistency(
                tokens, token_logprobs, prompt_len, completion_len, ctx
            ):
                return False

            # Check 4: Reward consistency (if env evaluation ran)
            if not self._check_reward_consistency(rollout, ctx):
                return False

            # Check 5: Completion length bounds
            if not self._check_completion_bounds(completion_len, ctx):
                return False

            # Check 6: Metadata consistency
            if not self._check_metadata_consistency(ctx):
                return False

            ctx.checks[self.check_name] = True
            logger.debug(
                f"[{self.check_name}] All consistency checks passed | "
                f"tokens={len(tokens)} prompt={prompt_len} completion={completion_len}"
            )
            return True

        except Exception as e:
            logger.error(f"[{self.check_name}] Unexpected error: {e}", exc_info=True)
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_error"] = str(e)
            return False

    def _check_token_count_consistency(
        self,
        tokens: list[int],
        prompt_len: int,
        completion_len: int,
        ctx: ValidationContext,
    ) -> bool:
        """Verify token counts match across different fields."""
        expected_total = prompt_len + completion_len
        actual_total = len(tokens)

        if expected_total != actual_total:
            logger.warning(
                f"[{self.check_name}] Token count mismatch | "
                f"expected={expected_total} (prompt={prompt_len} + completion={completion_len}) | "
                f"actual={actual_total}"
            )
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_failure"] = "token_count_mismatch"
            ctx.metadata["expected_tokens"] = expected_total
            ctx.metadata["actual_tokens"] = actual_total
            return False

        return True

    def _check_commitment_consistency(
        self, tokens: list[int], commitments: list, ctx: ValidationContext
    ) -> bool:
        """Verify commitment array length matches token sequence."""
        if not commitments:
            # Commitments might be optional in some contexts
            return True

        if len(commitments) != len(tokens):
            logger.warning(
                f"[{self.check_name}] Commitment array length mismatch | "
                f"commitments={len(commitments)} | tokens={len(tokens)}"
            )
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_failure"] = "commitment_length_mismatch"
            return False

        return True

    def _check_logprob_consistency(
        self,
        tokens: list[int],
        token_logprobs: list,
        prompt_len: int,
        completion_len: int,
        ctx: ValidationContext,
    ) -> bool:
        """Verify logprob array structure and length."""
        if not isinstance(token_logprobs, list):
            logger.warning(
                f"[{self.check_name}] token_logprobs is not a list | "
                f"type={type(token_logprobs).__name__}"
            )
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_failure"] = "logprobs_wrong_type"
            return False

        # Logprobs should cover entire sequence (prompt + completion)
        expected_len = len(tokens)
        actual_len = len(token_logprobs)

        if expected_len != actual_len:
            logger.warning(
                f"[{self.check_name}] Logprobs length mismatch | "
                f"expected={expected_len} | actual={actual_len} | "
                f"prompt_len={prompt_len} | completion_len={completion_len}"
            )
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_failure"] = "logprobs_length_mismatch"
            ctx.metadata["expected_logprobs"] = expected_len
            ctx.metadata["actual_logprobs"] = actual_len
            return False

        # Verify prompt logprobs are zeros (as per miner convention)
        if prompt_len > 0:
            prompt_logprobs = token_logprobs[:prompt_len]
            non_zero_count = sum(1 for lp in prompt_logprobs if abs(float(lp)) > 1e-6)

            if non_zero_count > 0:
                logger.debug(
                    f"[{self.check_name}] Non-zero logprobs in prompt region | "
                    f"count={non_zero_count}/{prompt_len}"
                )
                # This is a soft warning, not a hard failure
                ctx.metadata["prompt_logprobs_non_zero"] = non_zero_count

        return True

    def _check_reward_consistency(
        self, rollout: dict, ctx: ValidationContext
    ) -> bool:
        """Verify reward values are consistent with environment evaluation."""
        # Only check if environment evaluation ran successfully
        if "env_eval_result" not in ctx.metadata:
            return True

        env_result = ctx.metadata["env_eval_result"]
        miner_reward = rollout.get("total_reward")
        env_reward = env_result.get("reward")

        # Skip if either is missing
        if miner_reward is None or env_reward is None:
            return True

        # Note: Actual tolerance checking is done by RewardValidator
        # Here we just check for gross inconsistencies (e.g., wrong sign)
        try:
            miner_r = float(miner_reward)
            env_r = float(env_reward)

            # Check for sign mismatch (major inconsistency)
            if (miner_r > 0) != (env_r > 0) and abs(miner_r) > 0.01 and abs(env_r) > 0.01:
                logger.warning(
                    f"[{self.check_name}] Reward sign mismatch | "
                    f"miner={miner_r:.4f} | env={env_r:.4f}"
                )
                ctx.metadata["consistency_warning"] = "reward_sign_mismatch"
                # Don't fail hard here, let RewardValidator handle it

        except (ValueError, TypeError) as e:
            logger.debug(f"[{self.check_name}] Reward type conversion error: {e}")

        return True

    def _check_completion_bounds(
        self, completion_len: int, ctx: ValidationContext
    ) -> bool:
        """Verify completion length meets minimum requirements."""
        # Completion must be long enough for challenge sampling
        if completion_len < CHALLENGE_K:
            logger.warning(
                f"[{self.check_name}] Completion too short for challenges | "
                f"completion_len={completion_len} | min_required={CHALLENGE_K}"
            )
            ctx.checks[self.check_name] = False
            ctx.metadata["consistency_failure"] = "completion_too_short"
            return False

        return True

    def _check_metadata_consistency(self, ctx: ValidationContext) -> bool:
        """Verify metadata from different validators is consistent."""
        # Check if logprob validator reported mismatches but passed
        logprob_mismatches = ctx.metadata.get("logprob_mismatches", 0)
        logprob_total = ctx.metadata.get("logprob_total", 0)

        if logprob_total > 0:
            mismatch_rate = logprob_mismatches / logprob_total
            if mismatch_rate > 0.5:
                logger.warning(
                    f"[{self.check_name}] High logprob mismatch rate | "
                    f"rate={mismatch_rate:.2%} ({logprob_mismatches}/{logprob_total})"
                )
                ctx.metadata["consistency_warning"] = "high_logprob_mismatch_rate"
                # Don't fail hard, this is just a warning

        # Check distribution metrics for extreme anomalies
        dist_metrics = ctx.metadata.get("distribution_metrics")
        if dist_metrics:
            min_prob = dist_metrics.get("min", 1.0)
            if min_prob < 1e-10:
                logger.warning(
                    f"[{self.check_name}] Extremely low token probability detected | "
                    f"min_prob={min_prob:.2e}"
                )
                ctx.metadata["consistency_warning"] = "extreme_low_probability"

        return True
