"""Environment-agnostic validators using the EnvRegistry."""

from __future__ import annotations

import logging
import math
from statistics import median

import torch

from ...environments.registry import get_adapter
from ...protocol.constants import (
    CHALLENGE_K,
    CURRENT_ENV_ID,
    REWARD_ABS_TOL,
    REWARD_REL_TOL,
)
from ...protocol.crypto import indices_from_root_in_range
from ...protocol.signatures import derive_env_seed
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


def _derive_canonical_seed(ctx: ValidationContext) -> int:
    """Derive canonical environment seed from trusted validator values.

    Uses validator's chain-derived window_hash and group_index to compute
    the canonical seed. NEVER trusts miner-provided data for this derivation.

    The miner's hotkey (prover_address) is already validated by signature checks
    before reaching this point.

    Returns:
        Integer seed for environment reset
    """
    # Use trusted validator-derived values: prover_address (validated hotkey),
    # window_hash (from validator's chain query), group_index (file-order)
    return derive_env_seed(ctx.prover_address, ctx.window_hash, ctx.group_index)


class EnvironmentPromptValidator(Validator):
    check_name = "env_prompt_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        try:
            # Use environment ID from checkpoint metadata (fallback to constant)
            env_id = ctx.env_id or CURRENT_ENV_ID

            if not env_id:
                logger.error(
                    "No environment ID available (both ctx.env_id and CURRENT_ENV_ID are None)"
                )
                ctx.checks[self.check_name] = False
                return False

            # Derive canonical seed from trusted validator values
            canonical_seed = _derive_canonical_seed(ctx)

            try:
                adapter = get_adapter(env_id)
            except ValueError as e:
                logger.error(f"Invalid environment ID '{env_id}': {e}")
                ctx.checks[self.check_name] = False
                return False

            canonical_ids = adapter.build_prompt_ids(
                canonical_seed, ctx.tokenizer, env_params=ctx.env_params or None
            )

            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))

            if prompt_len != len(canonical_ids):
                # Debug: show first/last tokens to identify mismatch pattern
                miner_first_10 = tokens[: min(10, len(tokens))]
                canonical_first_10 = canonical_ids[: min(10, len(canonical_ids))]
                logger.warning(
                    (
                        "Prompt length mismatch: miner=%d, validator=%d "
                        "(seed=%d, hotkey=%s, window=%s, group=%d)"
                    ),
                    prompt_len,
                    len(canonical_ids),
                    canonical_seed,
                    ctx.prover_address[:12] if ctx.prover_address else "?",
                    ctx.window_hash[:12] if ctx.window_hash else "?",
                    ctx.group_index,
                )
                logger.debug(
                    ("Token comparison: miner_first_10=%s, canonical_first_10=%s"),
                    miner_first_10,
                    canonical_first_10,
                )
                ctx.checks[self.check_name] = False
                return False
            if prompt_len + completion_len != len(tokens):
                logger.warning(
                    "Token length mismatch: prompt_len=%d, completion_len=%d, total_tokens=%d",
                    prompt_len,
                    completion_len,
                    len(tokens),
                )
                ctx.checks[self.check_name] = False
                return False
            if tokens[:prompt_len] != canonical_ids:
                # Find first mismatch position
                mismatch_pos = -1
                for i in range(min(len(tokens), len(canonical_ids))):
                    if tokens[i] != canonical_ids[i]:
                        mismatch_pos = i
                        break
                logger.warning(
                    "Prompt token mismatch at position %d: miner=%d, validator=%d (seed=%d, prompt_len=%d)",
                    mismatch_pos,
                    tokens[mismatch_pos] if 0 <= mismatch_pos < len(tokens) else -1,
                    canonical_ids[mismatch_pos] if 0 <= mismatch_pos < len(canonical_ids) else -1,
                    canonical_seed,
                    prompt_len,
                )
                ctx.checks[self.check_name] = False
                return False

            ctx.checks[self.check_name] = True
            return True
        except Exception as e:
            logger.warning(f"Env prompt validation error: {e}", exc_info=True)
            ctx.checks[self.check_name] = False
            return False


class EnvironmentEvaluationValidator(Validator):
    check_name = "env_eval_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        try:
            # Use environment ID from checkpoint metadata (fallback to constant)
            env_id = ctx.env_id or CURRENT_ENV_ID

            if not env_id:
                logger.error(
                    "No environment ID available (both ctx.env_id and CURRENT_ENV_ID are None)"
                )
                ctx.checks[self.check_name] = False
                return False

            # Derive canonical seed from trusted validator values
            canonical_seed = _derive_canonical_seed(ctx)

            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))
            end_idx = prompt_len + completion_len if completion_len > 0 else len(tokens)
            completion_ids = tokens[prompt_len:end_idx]
            completion_text = ctx.tokenizer.decode(completion_ids, skip_special_tokens=False)

            try:
                adapter = get_adapter(env_id)
            except ValueError as e:
                logger.error(f"Invalid environment ID '{env_id}': {e}")
                ctx.checks[self.check_name] = False
                return False
            result = adapter.evaluate_completion(
                canonical_seed, completion_text, ctx.tokenizer, env_params=ctx.env_params or None
            )
            ctx.metadata["env_eval_result"] = result
            ctx.checks[self.check_name] = True
            return True
        except Exception as e:
            logger.debug(f"Env evaluation error: {e}")
            ctx.checks[self.check_name] = False
            return False


class RewardValidator(Validator):
    check_name = "reward_valid"
    severity = "soft"
    soft_threshold = 0.20

    def validate(self, ctx: ValidationContext) -> bool:
        try:
            rollout = ctx.commit.get("rollout", {})
            miner_reward = float(rollout.get("total_reward"))
            env_res = ctx.metadata.get("env_eval_result", {})
            env_reward = float(env_res.get("reward"))

            ok = math.isclose(
                miner_reward, env_reward, rel_tol=REWARD_REL_TOL, abs_tol=REWARD_ABS_TOL
            )
            ctx.checks[self.check_name] = ok

            if not ok:
                # Extract detailed diagnostics for debugging reward mismatches
                miner_uid = ctx.miner_uid or "unknown"
                window_hash_short = ctx.window_hash[:12] if ctx.window_hash else "?"
                reward_diff = abs(miner_reward - env_reward)
                rel_diff = reward_diff / max(abs(miner_reward), abs(env_reward), 1e-9)

                # Get test execution details from env_eval_result
                tests_passed = env_res.get("tests_passed", "N/A")
                tests_total = env_res.get("tests_total", "N/A")
                env_success = env_res.get("success", "N/A")

                # Get miner's claimed test results from rollout
                miner_success = rollout.get("success", "N/A")

                logger.warning(
                    "[reward_valid] REWARD MISMATCH | "
                    "uid=%s | window_hash=%s | "
                    "miner_reward=%.6f | validator_reward=%.6f | "
                    "abs_diff=%.6f | rel_diff=%.4f | "
                    "tolerances=(rel=%.4f, abs=%.2e) | "
                    "validator_tests=%s/%s | validator_success=%s | miner_success=%s | "
                    "env_id=%s | "
                    "This indicates code execution produced different results. "
                    "Possible causes: (1) timeout boundary race, (2) non-deterministic code, "
                    "(3) execution pool issues, (4) resource limits",
                    miner_uid,
                    window_hash_short,
                    miner_reward,
                    env_reward,
                    reward_diff,
                    rel_diff,
                    REWARD_REL_TOL,
                    REWARD_ABS_TOL,
                    tests_passed,
                    tests_total,
                    env_success,
                    miner_success,
                    ctx.env_id or CURRENT_ENV_ID,
                )

                # Log full env_eval_result at DEBUG level for detailed analysis
                logger.debug(
                    "[reward_valid] Full env_eval_result for uid=%s: %s",
                    miner_uid,
                    env_res,
                )

            return ok
        except Exception as e:
            logger.warning(
                "[reward_valid] Validation error for uid=%s: %s",
                ctx.miner_uid or "unknown",
                e,
                exc_info=True,
            )
            ctx.checks[self.check_name] = False
            return False


class LogprobValidator(Validator):
    """Validate miner-claimed logprobs via a robust median IS-ratio check.

    Over the K=32 cryptographically chosen completion positions, computes the
    per-position importance-sampling deviation
        dev_i = exp(|model_lp_i - miner_lp_i|) - 1
    and rejects the rollout iff ``median(dev_i) > LOGPROB_IS_EPS``.

    The median is robust to a handful of bf16-noise outliers while still
    rejecting miners whose distribution diverges meaningfully from the
    canonical model. Empirically (430k honest cross-GPU/cross-attn/cross-batch
    trials, 0% FP) ``LOGPROB_IS_EPS=0.10`` leaves ~50% headroom over the worst
    observed honest pair (max median dev = 0.066).
    """

    check_name = "logprobs_valid"
    severity = "hard"

    # Maximum allowed median importance-sampling deviation per K-sample.
    # Equivalent PPO ratio bound is ``1 + LOGPROB_IS_EPS``.
    LOGPROB_IS_EPS = 0.10

    def validate(self, ctx: ValidationContext) -> bool:
        try:
            rollout = ctx.commit.get("rollout", {})
            claimed = rollout.get("token_logprobs", [])
            if not isinstance(claimed, list):
                ctx.checks[self.check_name] = False
                return False

            tokens = ctx.commit.get("tokens", [])
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))

            # Enforce minimum completion length so the K=32 challenge fits
            if completion_len < CHALLENGE_K:
                logger.debug(
                    "[logprobs_valid] Completion too short | required>=%d got=%d "
                    "| prompt_len=%d seq_len=%d",
                    CHALLENGE_K,
                    completion_len,
                    prompt_len,
                    len(tokens),
                )
                ctx.checks[self.check_name] = False
                return False

            # Miner emits token_logprobs as [0.0] * prompt_len + completion_lps
            if len(claimed) != len(tokens):
                logger.debug(
                    "[logprobs_valid] Length mismatch: expected %d, got %d",
                    len(tokens),
                    len(claimed),
                )
                ctx.checks[self.check_name] = False
                return False

            logits = ctx.cached_logits  # [seq_len, vocab] on CPU; may be None
            precomputed = ctx.precomputed_logprobs
            if logits is None and precomputed is None:
                logger.debug("[logprobs_valid] No cached logits or precomputed logprobs available")
                ctx.checks[self.check_name] = False
                return False

            challenged_idxs = indices_from_root_in_range(
                tokens,
                ctx.challenge_randomness,
                prompt_len,
                prompt_len + completion_len,
                CHALLENGE_K,
            )

            devs = self._compute_devs(
                challenged_idxs=challenged_idxs,
                tokens=tokens,
                claimed=claimed,
                logits=logits,
                precomputed=precomputed,
            )

            # Fail closed if any challenged position could not be resolved.
            if devs is None or len(devs) != CHALLENGE_K:
                resolved = -1 if devs is None else len(devs)
                logger.debug(
                    "[logprobs_valid] Could not resolve all challenged positions "
                    "| resolved=%d expected=%d",
                    resolved,
                    CHALLENGE_K,
                )
                ctx.checks[self.check_name] = False
                return False

            median_dev = median(devs)
            max_dev = max(devs)
            ok = median_dev <= self.LOGPROB_IS_EPS

            ctx.metadata["logprob_median_dev"] = median_dev
            ctx.metadata["logprob_max_dev"] = max_dev
            ctx.metadata["logprob_total"] = len(devs)

            if not ok:
                worst_idx = max(range(len(devs)), key=devs.__getitem__)
                worst_abs = challenged_idxs[worst_idx]
                logger.debug(
                    "[logprobs_valid] FAIL median_dev=%.6f max_dev=%.6f eps=%.4f "
                    "| worst pos=%d (rel=%d) token=%d miner_lp=%.6f",
                    median_dev,
                    max_dev,
                    self.LOGPROB_IS_EPS,
                    worst_abs,
                    worst_abs - prompt_len,
                    tokens[worst_abs],
                    float(claimed[worst_abs]),
                )

            ctx.checks[self.check_name] = ok
            return ok
        except Exception as e:
            logger.debug(f"Logprob validation error: {e}")
            ctx.checks[self.check_name] = False
            return False

    @staticmethod
    def _compute_devs(
        *,
        challenged_idxs: list[int],
        tokens: list[int],
        claimed: list[float],
        logits: torch.Tensor | None,
        precomputed: dict[int, float] | None,
    ) -> list[float] | None:
        """Return per-position IS deviation ``exp(|Δlp|) - 1`` or ``None`` on failure."""
        devs: list[float] = []
        for abs_idx in challenged_idxs:
            if precomputed is not None and abs_idx in precomputed:
                model_lp = precomputed[abs_idx]
            else:
                if logits is None:
                    return None
                pos = abs_idx - 1
                if pos < 0 or pos >= logits.size(0):
                    return None
                dist = torch.log_softmax(logits[pos].float(), dim=-1)
                model_lp = float(dist[tokens[abs_idx]].item())

            miner_lp = float(claimed[abs_idx])
            devs.append(math.exp(abs(model_lp - miner_lp)) - 1.0)
        return devs
