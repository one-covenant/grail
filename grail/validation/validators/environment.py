"""Environment-agnostic validators using the EnvRegistry."""

from __future__ import annotations

import logging
import math

import torch

from ...environments.registry import get_adapter
from ...protocol.signatures import derive_env_seed
from ...shared.constants import CURRENT_ENV_ID, REWARD_ABS_TOL, REWARD_REL_TOL
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
            # Use trusted environment ID constant (never trust miner data)
            env_id = CURRENT_ENV_ID

            # Derive canonical seed from trusted validator values
            canonical_seed = _derive_canonical_seed(ctx)

            adapter = get_adapter(env_id)

            # Pass integer seed through; adapter handles type as needed
            canonical_ids = adapter.build_prompt_ids(canonical_seed, ctx.tokenizer)

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
            # Use trusted environment ID constant (never trust miner data)
            env_id = CURRENT_ENV_ID

            # Derive canonical seed from trusted validator values
            canonical_seed = _derive_canonical_seed(ctx)

            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))
            end_idx = prompt_len + completion_len if completion_len > 0 else len(tokens)
            completion_ids = tokens[prompt_len:end_idx]
            completion_text = ctx.tokenizer.decode(completion_ids, skip_special_tokens=False)

            adapter = get_adapter(env_id)
            # Pass integer seed through; adapter handles type as needed
            result = adapter.evaluate_completion(canonical_seed, completion_text, ctx.tokenizer)
            ctx.metadata["env_eval_result"] = result
            ctx.checks[self.check_name] = True
            return True
        except Exception as e:
            logger.debug(f"Env evaluation error: {e}")
            ctx.checks[self.check_name] = False
            return False


class RewardValidator(Validator):
    check_name = "reward_valid"

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
            return ok
        except Exception as e:
            logger.debug(f"Reward validation error: {e}")
            ctx.checks[self.check_name] = False
            return False


class LogprobValidator(Validator):
    check_name = "logprobs_valid"
    severity = "hard"

    LOGPROB_ABS_TOL = 1e-3
    LOGPROB_REL_TOL = 0.02

    def validate(self, ctx: ValidationContext) -> bool:
        try:
            rollout = ctx.commit.get("rollout", {})
            claimed = rollout.get("token_logprobs", [])
            if not isinstance(claimed, list):
                ctx.checks[self.check_name] = False
                return False

            # Use cached logits from proof validator
            logits = ctx.cached_logits  # shape [seq_len, vocab]
            tokens = ctx.commit.get("tokens", [])
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))

            if logits is None or completion_len <= 0:
                ctx.checks[self.check_name] = True
                return True

            comp_ids = tokens[prompt_len : prompt_len + completion_len]

            # Miner expect to generate token_logprobs as: [0.0] * prompt_len + logprobs
            # So len(claimed) should equal len(tokens) = prompt_len + completion_len
            if len(claimed) != len(tokens):
                logger.debug(
                    "Logprob length mismatch: expected %d, got %d", len(tokens), len(claimed)
                )
                ctx.checks[self.check_name] = False
                return False

            # Extract completion logprobs (skip prompt region)
            claimed_comp = claimed[prompt_len : prompt_len + completion_len]

            # Debug: log alignment and first few claimed logprobs
            logger.debug(
                (
                    "VALIDATOR LOGPROB ALIGNMENT: claimed_len=%d completion_len=%d "
                    "prompt_len=%d tokens_len=%d claimed_comp_len=%d"
                ),
                len(claimed),
                completion_len,
                prompt_len,
                len(tokens),
                len(claimed_comp),
            )
            if claimed_comp:
                first_5_claimed = claimed_comp[: min(5, len(claimed_comp))]
                logger.debug(
                    "VALIDATOR CLAIMED LOGPROBS: first_5=%s",
                    [f"{lp:.6f}" for lp in first_5_claimed],
                )

            # logits correspond to next-token scores; align per-generation index
            mismatches = 0
            total = min(len(claimed_comp), len(comp_ids))
            first_mismatch_details = None

            for i in range(total):
                pos = prompt_len + i - 1
                if pos < 0 or pos >= logits.size(0):
                    continue
                dist = torch.log_softmax(logits[pos], dim=-1)
                model_lp = float(dist[comp_ids[i]].item())
                miner_lp = float(claimed_comp[i])
                if not self._close_lp(model_lp, miner_lp):
                    mismatches += 1
                    # Capture first mismatch for debugging
                    if first_mismatch_details is None:
                        first_mismatch_details = {
                            "pos": i,
                            "token_id": comp_ids[i],
                            "model_lp": model_lp,
                            "miner_lp": miner_lp,
                            "diff": abs(model_lp - miner_lp),
                        }

            # Soft check: allow some noise
            ok = mismatches <= max(1, total // 10)
            ctx.metadata["logprob_mismatches"] = mismatches
            ctx.metadata["logprob_total"] = total

            if not ok and first_mismatch_details:
                logger.debug(
                    "Logprob validation failed: %d/%d mismatches. First mismatch at pos %d: "
                    "token=%d, model_lp=%.6f, miner_lp=%.6f, diff=%.6f",
                    mismatches,
                    total,
                    first_mismatch_details["pos"],
                    first_mismatch_details["token_id"],
                    first_mismatch_details["model_lp"],
                    first_mismatch_details["miner_lp"],
                    first_mismatch_details["diff"],
                )

            ctx.checks[self.check_name] = ok
            return ok
        except Exception as e:
            logger.debug(f"Logprob validation error: {e}")
            ctx.checks[self.check_name] = False
            return False

    def _close_lp(self, a: float, b: float) -> bool:
        if abs(a - b) <= self.LOGPROB_ABS_TOL:
            return True
        denom = max(1e-6, abs(a))
        return abs(a - b) / denom <= self.LOGPROB_REL_TOL
