"""GRPO (Group Relative Policy Optimization) with consolidated data loading and training.

This module consolidates:
- Data classes for GRPO groups and rollouts
- Loader for fetching and validating miner data
- Computation utilities for logprobs and entropy
- Training algorithm implementation
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

try:
    from grail.infrastructure.miner_data import fetch_multiple_miners_data
except Exception:  # pragma: no cover - optional in offline mode

    async def fetch_multiple_miners_data(*args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        raise RuntimeError("Miner data fetching is unavailable in offline mode.")


from grail.shared.constants import (
    ROLLOUTS_PER_PROBLEM,
    TRAINER_ADAPTIVE_KL,
    TRAINER_ADV_CLIP_PERCENTILE,
    TRAINER_BATCH_SIZE,
    TRAINER_ENTROPY_COEF,
    TRAINER_GRAD_ACCUM_STEPS,
    TRAINER_GRAD_CLIP,
    TRAINER_IS_RATIO_MAX,
    TRAINER_KL_ADAPT_RATE,
    TRAINER_KL_COEF,
    TRAINER_KL_MAX,
    TRAINER_KL_MIN,
    TRAINER_KL_TARGET,
    TRAINER_LOGRATIO_CLAMP,
    TRAINER_MAX_LENGTH,
    TRAINER_PPO_CLIP_EPS,
    TRAINER_PPO_CLIP_EPS_UPPER,
    TRAINER_USE_IS,
)
from grail.trainer.metrics import (
    KMetricsAggregator,
    TaskReplicateResult,
    derive_k_values,
)

from .base import TrainingAlgorithm

if TYPE_CHECKING:
    from grail.infrastructure.chain import GrailChainManager
    from grail.shared.schemas import BucketCredentials
    from grail.trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


def _print_decoded_rollout_samples(
    miner_data: dict[str, dict], max_samples_per_miner: int = 100, max_tokens_to_show: int = 1000
) -> None:
    """Decode and print sample tokens from miner rollouts for inspection.

    Helpful for debugging: shows what tokens/text was sent by miners.

    Args:
        miner_data: Dict mapping hotkey -> window_data from miners
        max_samples_per_miner: Max number of rollouts to show per miner
        max_tokens_to_show: Max number of tokens to display from each rollout
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.debug("transformers not available; skipping token decoding")
        return

    if not miner_data:
        logger.info("No miner data to decode")
        return

    # Try to infer tokenizer from first rollout's model info
    tokenizer = None
    model_name = None
    for _miner_hotkey, window_data in miner_data.items():
        if isinstance(window_data, dict):
            inferences = window_data.get("inferences", [])
            if inferences and isinstance(inferences[0], dict):
                commit = inferences[0].get("commit", {})
                model_info = commit.get("model", {})
                model_name = model_info.get("name")
                break

    if not model_name:
        logger.debug("Could not infer model name from rollouts; skipping decoding")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.debug("Failed to load tokenizer for %s: %s", model_name, e)
        return

    logger.info("=" * 100)
    logger.info("DECODED ROLLOUT SAMPLES (Model: %s)", model_name)
    logger.info("=" * 100)

    for miner_hotkey, window_data in miner_data.items():
        if not isinstance(window_data, dict):
            continue

        inferences = window_data.get("inferences", [])
        if not isinstance(inferences, list):
            continue

        logger.info("\nMiner: %s | Total rollouts: %d", miner_hotkey[:16], len(inferences))

        for idx, rollout in enumerate(inferences[:max_samples_per_miner]):
            if not isinstance(rollout, dict):
                continue

            commit = rollout.get("commit", {})
            tokens = commit.get("tokens", [])
            rollout_data = commit.get("rollout", {})
            prompt_length = rollout_data.get("prompt_length", 0)
            completion_length = rollout_data.get("completion_length", 0)
            success = rollout_data.get("success", False)
            reward = rollout_data.get("total_reward", 0.0)

            logger.info(
                "  Rollout %d | Prompt: %d tokens | Completion: %d tokens | "
                "Success: %s | Reward: %.3f",
                idx,
                prompt_length,
                completion_length,
                success,
                reward,
            )

            if tokens and isinstance(tokens, list):
                # Decode prompt
                prompt_tokens = tokens[:prompt_length] if prompt_length <= len(tokens) else tokens
                try:
                    prompt_text = tokenizer.decode(
                        prompt_tokens[:max_tokens_to_show], skip_special_tokens=False
                    )
                    logger.info(
                        "    [PROMPT (first %d tokens)]:\n%s",
                        min(max_tokens_to_show, len(prompt_tokens)),
                        prompt_text,
                    )
                except Exception as e:
                    logger.debug("Failed to decode prompt: %s", e)

                # Decode completion
                if completion_length > 0 and prompt_length < len(tokens):
                    comp_tokens = tokens[
                        prompt_length : min(prompt_length + max_tokens_to_show, len(tokens))
                    ]
                    try:
                        comp_text = tokenizer.decode(comp_tokens, skip_special_tokens=False)
                        logger.info(
                            "    [COMPLETION (first %d tokens)]:\n%s",
                            len(comp_tokens),
                            comp_text[:500],
                        )
                    except Exception as e:
                        logger.debug("Failed to decode completion: %s", e)
            else:
                logger.info("    [No tokens available]")

    logger.info("=" * 100)


@dataclass
class GRPORollout:
    """Single rollout from a GRPO group."""

    tokens: list[int]
    prompt_length: int
    completion_length: int
    advantage: float
    reward: float
    success: bool
    nonce: int
    rollout_group: str
    token_logprobs: list[float] | None = None


@dataclass
class GRPOGroup:
    """Collection of rollouts associated with one SAT problem."""

    group_id: str
    rollouts: list[GRPORollout]

    def is_valid(
        self, advantage_tolerance: float, rollouts_per_problem: int = ROLLOUTS_PER_PROBLEM
    ) -> bool:
        """Validate group size and zero-sum advantage condition."""
        if len(self.rollouts) != rollouts_per_problem:
            return False
        advantage_sum = sum(r.advantage for r in self.rollouts)
        return abs(advantage_sum) < advantage_tolerance


def _has_advantage_variance(group: GRPOGroup) -> bool:
    """Check if a GRPO group has variance in advantage values.

    Groups with zero advantage variance (all rollouts have identical advantage)
    are filtered out as they provide no learning signal.

    Args:
        group: The GRPO group to check

    Returns:
        True if the group has advantage variance, False if all advantages are identical
    """
    if not group.rollouts:
        return False

    advantages = [r.advantage for r in group.rollouts]
    first_advantage = advantages[0]

    # Check if all advantages are the same
    return any(adv != first_advantage for adv in advantages)


def _is_valid_logprobs(logprobs: list[float] | None) -> bool:
    """Validate that logprobs list contains only finite numeric values.

    Args:
        logprobs: List of logprob values to validate

    Returns:
        True if logprobs is a non-empty list of numeric values with all finite floats, False otherwise
    """
    if not isinstance(logprobs, list) or not logprobs:
        return False
    try:
        arr = np.asarray(logprobs, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 1:
        return False
    return bool(np.isfinite(arr).all())


def _group_rollouts(raw_rollouts: list[dict[str, Any]]) -> dict[str, list[GRPORollout]]:
    """Group raw rollout dicts into rollout objects by group_id.

    Entire groups are filtered if any rollout has invalid logprobs to ensure
    data consistency and training stability.

    Args:
        raw_rollouts: List of raw rollout dictionaries from miners

    Returns:
        Dict mapping group_id to list of GRPORollout objects (only valid groups)
    """
    # First pass: collect all rollouts, tracking groups with invalid logprobs
    ungrouped: dict[str, list[GRPORollout]] = {}
    invalid_logprobs_groups: set[str] = set()

    for rollout_dict in raw_rollouts:
        group_id = str(rollout_dict.get("rollout_group", ""))
        if not group_id:
            continue

        commit = rollout_dict.get("commit", {})
        rollout_meta = commit.get("rollout", {})

        # Extract and validate token logprobs during loading
        tlp = rollout_meta.get("token_logprobs", None)
        # Require behavior logprobs to exist and be well-formed
        if tlp is None:
            logger.warning(
                "Missing token_logprobs; marking group for filter",
                extra={"group_id": group_id},
            )
            invalid_logprobs_groups.add(group_id)
        elif not isinstance(tlp, list):
            logger.warning(
                "Invalid token_logprobs type; marking group for filter",
                extra={
                    "group_id": group_id,
                    "logprobs_type": type(tlp).__name__,
                },
            )
            invalid_logprobs_groups.add(group_id)
        elif not _is_valid_logprobs(tlp):
            logger.warning(
                "Non-finite token_logprobs detected; marking group for filter",
                extra={
                    "group_id": group_id,
                    "logprobs_len": len(tlp),
                },
            )
            invalid_logprobs_groups.add(group_id)
        else:
            # Validate length consistency with provided lengths
            orig_prompt_len = int(rollout_meta.get("prompt_length", 0))
            orig_comp_len = int(rollout_meta.get("completion_length", 0) or 0)
            expected_total = max(0, orig_prompt_len) + max(0, orig_comp_len)
            tlp_len = len(tlp)
            if not (tlp_len >= expected_total or tlp_len == orig_comp_len):
                logger.warning(
                    "token_logprobs length inconsistent with prompt/completion; marking group for filter",
                    extra={
                        "group_id": group_id,
                        "logprobs_len": tlp_len,
                        "expected_total": expected_total,
                        "prompt_len": orig_prompt_len,
                        "completion_len": orig_comp_len,
                    },
                )
                invalid_logprobs_groups.add(group_id)

        try:
            rollout = GRPORollout(
                tokens=list(commit.get("tokens", [])),
                prompt_length=int(rollout_meta.get("prompt_length", 0)),
                completion_length=int(rollout_meta.get("completion_length", 0) or 0),
                advantage=float(rollout_meta.get("advantage", 0.0)),
                reward=float(rollout_meta.get("total_reward", 0.0)),
                success=bool(rollout_meta.get("success", False)),
                nonce=int(rollout_dict.get("nonce", 0)),
                rollout_group=group_id,
                token_logprobs=(list(tlp) if _is_valid_logprobs(tlp) else None),
            )
            ungrouped.setdefault(group_id, []).append(rollout)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Failed to parse rollout for group %s: %s",
                group_id,
                exc,
            )
            invalid_logprobs_groups.add(group_id)

    # Second pass: keep only groups with all valid logprobs
    grouped: dict[str, list[GRPORollout]] = {}
    for group_id, rollouts in ungrouped.items():
        if group_id in invalid_logprobs_groups:
            logger.info(
                "Filtering entire group due to invalid logprobs in any rollout",
                extra={
                    "group_id": group_id,
                    "num_rollouts": len(rollouts),
                },
            )
        else:
            grouped[group_id] = rollouts

    if invalid_logprobs_groups:
        logger.info(
            "Filtered %d groups with invalid logprobs during loading",
            len(invalid_logprobs_groups),
        )

    return grouped


def _compute_training_metrics(groups: list[GRPOGroup], window: int) -> dict[str, Any]:
    """Compute and log pre-training data quality metrics for GRPO groups.

    Args:
        groups: List of GRPO groups
        window: Training window number

    Returns:
        Dictionary of computed metrics keyed by metric name (with pass@k, mean@k, etc.)
    """
    try:
        report_ks = derive_k_values(ROLLOUTS_PER_PROBLEM)
    except Exception:  # noqa: BLE001
        report_ks = [1, 5, 10]
        if groups:
            report_ks.append(len(groups[0].rollouts))
        report_ks = sorted(set(report_ks))

    aggregator = KMetricsAggregator(report_ks=report_ks)
    for group in groups:
        # Define replicate order deterministically by nonce
        ordered = sorted(group.rollouts, key=lambda r: r.nonce)
        for idx, r in enumerate(ordered):
            aggregator.add(
                TaskReplicateResult(
                    task_id=group.group_id,
                    replicate_idx=idx,
                    reward=float(r.reward),
                    success=bool(r.success),
                )
            )

    prefilter_metrics = aggregator.summarize()
    if prefilter_metrics:
        # Log a concise, stable set of key indicators
        # Align k-values with ROLLOUTS_PER_PROBLEM for readability
        k_keys = [k for k in report_ks if f"pass@{k}" in prefilter_metrics]
        summary_bits = [f"pass@{k}={prefilter_metrics.get(f'pass@{k}', 0.0):.3f}" for k in k_keys]
        # Include ordered diagnostics for transparency (ordering-sensitive)
        summary_bits.extend(
            [
                f"pass_ordered@{k}={prefilter_metrics.get(f'pass_ordered@{k}', 0.0):.3f}"
                for k in k_keys
            ]
        )
        # Prefer full-group mean over first-RPP window for readability
        full_k = max(report_ks) if report_ks else 1
        mean_full = prefilter_metrics.get(f"mean@{full_k}", 0.0)
        summary_bits.append(f"mean@{full_k}={mean_full:.3f}")
        summary_bits.append(f"reward_mean={prefilter_metrics.get('reward_mean_all', 0.0):.3f}")
        summary_bits.append(f"reward_std={prefilter_metrics.get('reward_std_all', 0.0):.3f}")
        summary_bits.append(f"success_rate={prefilter_metrics.get('success_rate_all', 0.0):.3f}")
        logger.info(
            "Training data metrics (pre-filter, window %s): %s",
            window,
            ", ".join(summary_bits),
        )

    # Return metrics with metadata for async logging by caller
    return prefilter_metrics


def _group_reward_per_token(group: GRPOGroup) -> float:
    """Calculate mean reward per completion token for a GRPO group.

    Args:
        group: The GRPO group to calculate reward/token for

    Returns:
        Mean reward per completion token across all rollouts
    """
    totals: list[float] = []
    for rollout in group.rollouts:
        denominator: int = max(1, int(rollout.completion_length))
        totals.append(float(rollout.reward) / float(denominator))
    return float(sum(totals) / max(1, len(totals)))


def _filter_valid_groups(
    groups: list[GRPOGroup],
    advantage_tolerance: float,
    window: int,
    config: TrainingConfig | None = None,
) -> list[GRPOGroup]:
    """Filter and refine GRPO groups based on multiple validation stages.

    Applies a series of filtering stages:
    1. Structural validation (group size and zero-sum advantage condition)
    2. Variance check (groups must have advantage variance to provide learning signal)
    3. Completion token constraints (optional, from config)
    4. Refinement filters (success rate, reward/token thresholds, quantile dropping)
    5. Group count capping

    Args:
        groups: List of GRPO groups to filter
        advantage_tolerance: Maximum allowed sum of advantages in a group
        window: Training window number (for logging purposes)
        config: Optional training config with filtering parameters

    Returns:
        List of filtered and refined GRPO groups
    """
    # Stage 1: Fast structural validation
    valid_groups: list[GRPOGroup] = [
        group for group in groups if group.is_valid(advantage_tolerance)
    ]
    invalid_count: int = len(groups) - len(valid_groups)
    if invalid_count > 0:
        logger.warning(
            "Filtered out %s invalid GRPO groups for window %s",
            invalid_count,
            window,
        )

    # Stage 2: Variance check to ensure learning signal
    groups_with_variance: list[GRPOGroup] = [
        group for group in valid_groups if _has_advantage_variance(group)
    ]
    zero_variance_count: int = len(valid_groups) - len(groups_with_variance)
    if zero_variance_count > 0:
        logger.warning(
            "Filtered out %s GRPO groups with zero advantage variance for window %s",
            zero_variance_count,
            window,
        )

    # Stage 3: Optional structural cap on completion tokens (fast check)
    if config is not None and getattr(config, "grpo_max_completion_tokens", None):
        max_completion: int = int(config.grpo_max_completion_tokens)
        before: int = len(groups_with_variance)
        groups_with_variance = [
            group
            for group in groups_with_variance
            if all(0 <= rollout.completion_length <= max_completion for rollout in group.rollouts)
        ]
        if len(groups_with_variance) < before:
            logger.warning(
                "Filtered out %s groups exceeding max completion tokens (%s) for window %s",
                before - len(groups_with_variance),
                max_completion,
                window,
            )

    # Stage 4: Refinement filters (quality/efficiency)
    refined_groups: list[GRPOGroup] = groups_with_variance
    if config is not None:
        # Success fraction gate
        min_success_fraction: float = max(
            0.0, float(getattr(config, "grpo_min_success_fraction", 0.0))
        )
        if min_success_fraction > 0.0:
            before = len(refined_groups)
            refined_groups = [
                group
                for group in refined_groups
                if (
                    sum(1 for rollout in group.rollouts if rollout.success)
                    / max(1, len(group.rollouts))
                )
                >= min_success_fraction
            ]
            if len(refined_groups) < before:
                logger.warning(
                    "Filtered out %s groups below min success fraction=%.2f for window %s",
                    before - len(refined_groups),
                    min_success_fraction,
                    window,
                )

        # Reward per token threshold
        min_reward_per_token: float = float(getattr(config, "grpo_min_reward_per_token", 0.0))
        if min_reward_per_token > 0.0:
            before = len(refined_groups)
            refined_groups = [
                group
                for group in refined_groups
                if _group_reward_per_token(group) >= min_reward_per_token
            ]
            if len(refined_groups) < before:
                logger.warning(
                    "Filtered out %s groups below min reward/token=%.4f for window %s",
                    before - len(refined_groups),
                    min_reward_per_token,
                    window,
                )

        # Drop lowest quantile by reward/token if configured
        quantile_drop: float = float(getattr(config, "grpo_reward_per_token_drop_quantile", 0.0))
        if 0.0 < quantile_drop < 1.0 and refined_groups:
            scored: list[tuple[GRPOGroup, float]] = [
                (group, _group_reward_per_token(group)) for group in refined_groups
            ]
            # Sort ascending by score and drop lowest quantile
            scored.sort(key=lambda item: item[1])
            drop_count: int = int(len(scored) * quantile_drop)
            if drop_count > 0:
                refined_groups = [group for group, _ in scored[drop_count:]]
                logger.warning(
                    "Dropped %s groups (lowest %.0f%% by reward/token) for window %s",
                    drop_count,
                    quantile_drop * 100.0,
                    window,
                )

        # Stage 5: Cap groups by score (descending reward/token) to max_groups limit
        max_groups: int = int(getattr(config, "grpo_max_groups", 8))
        if len(refined_groups) > max_groups:
            scored = [(group, _group_reward_per_token(group)) for group in refined_groups]
            scored.sort(key=lambda item: item[1], reverse=True)
            refined_groups = [group for group, _ in scored[:max_groups]]
            logger.warning(
                "Limiting GRPO groups from %s to %s for window %s",
                len(scored),
                max_groups,
                window,
            )
    else:
        # Backward-compatible cap when no config provided
        max_groups = 8
        if len(refined_groups) > max_groups:
            refined_groups = refined_groups[:max_groups]
            logger.warning(
                "Limiting GRPO groups from %s to %s for window %s",
                len(groups_with_variance),
                max_groups,
                window,
            )

    logger.info(
        "Loaded %s valid GRPO groups for window %s",
        len(refined_groups),
        window,
    )

    return refined_groups


async def load_grpo_groups(
    window: int,
    advantage_tolerance: float,
    trusted_miner_hotkeys: set[str] | None = None,
    credentials: BucketCredentials | Any = None,
    chain_manager: GrailChainManager | None = None,
    uid_by_hotkey: dict[str, int] | None = None,
    config: TrainingConfig | None = None,
    monitor: Any | None = None,
) -> list[GRPOGroup]:
    """Load and validate GRPO groups directly from trusted miners.

    Args:
        window: Training window number
        advantage_tolerance: Maximum allowed sum of advantages in a group
        trusted_miner_hotkeys: Set of trusted miner hotkeys to load from
        credentials: R2 credentials for bucket access
        chain_manager: Chain manager for miner bucket discovery
        uid_by_hotkey: Mapping of hotkey to UID for readable logging
        monitor: Optional monitor for logging metrics to wandb

    Returns:
        List of valid GRPO groups
    """
    # Require trusted miners and credentials for direct miner fetching
    if not trusted_miner_hotkeys:
        logger.warning(
            "No trusted miners for window %s; skipping data load",
            window,
        )
        return []

    if credentials is None:
        logger.error("Credentials required for loading miner data")
        return []

    # Build UID list for logging
    trusted_uids = []
    if uid_by_hotkey:
        trusted_uids = sorted(
            [uid_by_hotkey[hk] for hk in trusted_miner_hotkeys if hk in uid_by_hotkey]
        )

    # Fetch window data from all trusted miners in parallel
    logger.info(
        "Fetching data from %d trusted miners (UIDs=%s) for window %s",
        len(trusted_miner_hotkeys),
        trusted_uids if trusted_uids else "N/A",
        window,
    )

    miner_data = await fetch_multiple_miners_data(
        miner_hotkeys=trusted_miner_hotkeys,
        window=window,
        credentials=credentials,
        chain_manager=chain_manager,
        max_concurrent=10,
    )

    if not miner_data:
        logger.warning(
            "No data fetched from any trusted miner for window %s",
            window,
        )
        return []

    # Debug: Decode and print sample tokens
    _print_decoded_rollout_samples(miner_data, max_samples_per_miner=100, max_tokens_to_show=1000)

    # Extract all rollouts from all miners
    raw_rollouts = []
    for miner_hotkey, window_data in miner_data.items():
        miner_uid = uid_by_hotkey.get(miner_hotkey) if uid_by_hotkey else None
        miner_ident = f"UID {miner_uid}" if miner_uid is not None else miner_hotkey[:12]

        if not isinstance(window_data, dict):
            logger.debug("Invalid window data format from miner %s", miner_ident)
            continue

        inferences = window_data.get("inferences", [])
        if not isinstance(inferences, list):
            logger.debug("Invalid inferences format from miner %s", miner_ident)
            continue

        # Tag each rollout with the miner hotkey
        for rollout in inferences:
            if isinstance(rollout, dict):
                rollout["hotkey"] = miner_hotkey
                raw_rollouts.append(rollout)

    logger.info(
        "Loaded %d raw rollouts from %d miners for window %s",
        len(raw_rollouts),
        len(miner_data),
        window,
    )

    grouped = _group_rollouts(raw_rollouts)

    # Construct GRPOGroup objects
    groups: list[GRPOGroup] = [
        GRPOGroup(group_id, rollouts) for group_id, rollouts in grouped.items()
    ]

    # Compute training-set metrics BEFORE filtering invalid groups
    prefilter_metrics = _compute_training_metrics(groups, window)

    # Log prefilter metrics to monitor with distinct namespace
    if monitor is not None and prefilter_metrics:
        for key, value in prefilter_metrics.items():
            try:
                await monitor.log_gauge(
                    f"training/prefilter/{key}",
                    float(value),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to log prefilter metric %s: %s", key, exc)

    # Apply comprehensive filtering stages
    refined_groups = _filter_valid_groups(groups, advantage_tolerance, window, config)

    return refined_groups


def _is_finite_tensor(tensor: torch.Tensor) -> bool:
    """Return True if all elements are finite (no NaN/Inf)."""
    return bool(torch.isfinite(tensor).all().item())


def compute_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    return_per_token: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute log-probabilities over completion tokens for GRPO.

    This function:
    1. Runs model forward pass on padded input_ids (right-padded to batch max length)
    2. Shifts logits left by 1 to align with next-token prediction (standard LM indexing)
    3. For each sequence, extracts logprobs for completion tokens only

    Key indexing detail: logits[i,j] predicts token[i,j+1], so to get logprob of
    token[i, prompt_len + k] (k-th completion token), we extract logprobs[i, prompt_len-1+k]

    Args:
        model: Language model
        input_ids: [batch_size, seq_len] right-padded token ids
        attention_mask: [batch_size, seq_len] mask (1=real, 0=pad)
        prompt_lengths: [batch_size] original prompt length per sample (before padding)
        completion_lengths: [batch_size] number of completion tokens per sample
        return_per_token: If True, return (sum_logprobs, per_token_logprobs_padded)

    Returns:
        If return_per_token=False: [batch_size] tensor of sum log-probabilities
        If return_per_token=True: ([batch_size], [batch_size, max_comp_len]) tuple
    """
    # Precision controlled by caller via accelerator.autocast
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2,
        shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    if not return_per_token:
        # Original behavior: return sequence sums
        seq_log_probs: list[torch.Tensor] = []
        seq_len_minus_1 = token_log_probs.shape[1]
        for idx, prompt_len in enumerate(prompt_lengths):
            completion_len = completion_lengths[idx]
            start_idx = max(0, prompt_len - 1)
            end_idx = min(seq_len_minus_1, start_idx + completion_len)
            if end_idx > start_idx:
                seq_log_probs.append(token_log_probs[idx, start_idx:end_idx].sum())
            else:
                seq_log_probs.append(torch.tensor(0.0, device=token_log_probs.device))
        return torch.stack(seq_log_probs)
    else:
        # Return both sums and per-token logprobs with masks
        max_comp_len = max(completion_lengths) if completion_lengths else 1
        batch_size = len(prompt_lengths)
        device = token_log_probs.device

        per_token_padded: torch.Tensor = torch.zeros(batch_size, max_comp_len, device=device)
        seq_log_probs_tensor: torch.Tensor = torch.zeros(batch_size, device=device)

        seq_len_minus_1 = token_log_probs.shape[1]
        for idx, prompt_len in enumerate(prompt_lengths):
            completion_len = completion_lengths[idx]
            start_idx = max(0, prompt_len - 1)
            end_idx = min(seq_len_minus_1, start_idx + completion_len)
            if end_idx > start_idx:
                comp_logps = token_log_probs[idx, start_idx:end_idx]
                per_token_padded[idx, : len(comp_logps)] = comp_logps
                seq_log_probs_tensor[idx] = comp_logps.sum()

        return seq_log_probs_tensor, per_token_padded


def compute_entropy(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute mean entropy over completion tokens for entropy regularization.

    Mirrors the indexing used in compute_logprobs to ensure entropy is computed
    only over the completion portion of each sequence.

    Args:
        model: Language model
        input_ids: [batch_size, seq_len] right-padded token ids
        attention_mask: [batch_size, seq_len] mask (1=real, 0=pad)
        prompt_lengths: [batch_size] original prompt length per sample (before padding)
        completion_lengths: [batch_size] number of completion tokens per sample

    Returns:
        [batch_size] tensor of mean entropy over completion tokens
    """
    # Precision is controlled by the caller via accelerator.autocast
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits[:, :-1, :].contiguous()

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)

    entropies: list[torch.Tensor] = []
    seq_len_minus_1 = entropy_per_token.shape[1]
    for idx, prompt_len in enumerate(prompt_lengths):
        completion_len = completion_lengths[idx]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(seq_len_minus_1, start_idx + completion_len)
        if end_idx > start_idx:
            entropies.append(entropy_per_token[idx, start_idx:end_idx].mean())
        else:
            entropies.append(torch.tensor(0.0, device=entropy_per_token.device))

    return torch.stack(entropies)


async def train_grpo_epoch(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    groups: list[GRPOGroup],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    monitor: Any,
    window: int,
    batch_size: int = TRAINER_BATCH_SIZE,
    algorithm: Any = None,  # TrainingAlgorithm instance for global counters
) -> dict[str, float]:
    """Run a single GRPO training epoch and return aggregated metrics."""
    model.train()
    ref_model.eval()

    all_rollouts: list[tuple] = []
    for group in groups:
        for rollout in group.rollouts:
            all_rollouts.append((rollout, group.group_id))

    all_rollouts.sort(key=lambda item: (item[1], item[0].nonce))

    # Use batch_size parameter (defaults to TRAINER_BATCH_SIZE if not provided)
    num_batches = math.ceil(len(all_rollouts) / batch_size)

    epoch_metrics: dict[str, list[float]] = defaultdict(list)
    grad_accum_counter = 0

    # Adaptive KL coefficient (mutable state for this epoch)
    current_kl_coef = float(TRAINER_KL_COEF)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_rollouts))
        batch_rollouts = [all_rollouts[i][0] for i in range(start_idx, end_idx)]

        batch_tokens: list[list[int]] = []
        batch_prompt_lens: list[int] = []
        batch_comp_lens: list[int] = []
        batch_advantages: list[float] = []
        batch_behavior_seq_logprobs: list[float] = []
        # Store miner-provided behavior per-token logprobs over completion for prompt-average PG
        batch_rewards: list[float] = []

        for rollout in batch_rollouts:
            tokens = rollout.tokens[:TRAINER_MAX_LENGTH]
            batch_tokens.append(tokens)

            # CRITICAL FIX: Recalculate completion_len after truncation
            # If sequence is truncated, actual completion tokens may be fewer
            actual_prompt_len = rollout.prompt_length
            actual_comp_len = min(
                rollout.completion_length,
                TRAINER_MAX_LENGTH - rollout.prompt_length,
            )
            batch_prompt_lens.append(actual_prompt_len)
            batch_comp_lens.append(actual_comp_len)

            batch_advantages.append(rollout.advantage)
            batch_rewards.append(rollout.reward)

            # Miner-provided per-token logprobs must exist; aggregate over truncated completion
            tlp: list[float] = list((rollout.token_logprobs or [])[:TRAINER_MAX_LENGTH])
            prompt_len = actual_prompt_len
            comp_len = actual_comp_len
            expected_len = prompt_len + comp_len

            if len(tlp) >= expected_len:
                # Extract completion logprobs: indices [prompt_len:prompt_len+comp_len]
                completion_logprobs = tlp[prompt_len : prompt_len + comp_len]
                provided = float(np.sum(completion_logprobs))
            else:
                # Legacy: miner provided only completion logprobs; respect truncation
                provided = float(np.sum(tlp[:comp_len]))

            batch_behavior_seq_logprobs.append(provided)

        # Basic structural sanity checks (length mismatches can cause degenerate grads)
        for i, tokens in enumerate(batch_tokens):
            expected_len = max(0, batch_prompt_lens[i]) + max(0, batch_comp_lens[i])
            if len(tokens) < expected_len:
                logger.warning(
                    "Sequence shorter than expected after truncation",
                    extra={
                        "idx": i,
                        "len_tokens": len(tokens),
                        "expected_len": expected_len,
                        "prompt_len": batch_prompt_lens[i],
                        "completion_len": batch_comp_lens[i],
                    },
                )

        max_len = max(len(tokens) for tokens in batch_tokens)
        # Ensure pad_token_id is set (fallback to eos_token_id if needed)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
            logger.warning("pad_token_id is None; using eos_token_id as fallback")

        input_ids = []
        attention_masks = []
        for tokens in batch_tokens:
            pad_length = max_len - len(tokens)
            input_ids.append(tokens + [pad_id] * pad_length)
            attention_masks.append([1] * len(tokens) + [0] * pad_length)

        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=accelerator.device,
        )
        attention_mask_tensor = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=accelerator.device,
        )
        advantages_tensor = torch.tensor(
            batch_advantages,
            dtype=torch.float32,
            device=accelerator.device,
        )

        # Compute current policy logprobs (with per-token for proper KL)
        logprobs_current_sum, logprobs_current_per_token = compute_logprobs(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
            return_per_token=True,
        )

        if not _is_finite_tensor(logprobs_current_sum):
            cur_min = torch.nan_to_num(logprobs_current_sum).min().item()
            cur_max = torch.nan_to_num(logprobs_current_sum).max().item()
            logger.warning(
                "Non-finite current logprobs; skipping batch",
                extra={"min": float(cur_min), "max": float(cur_max)},
            )
            continue

        # OLD/BEHAVIOR policy logprobs for importance sampling
        # At this point, behavior logprobs are guaranteed to be present and valid
        logprobs_old = torch.tensor(
            list(batch_behavior_seq_logprobs),
            dtype=torch.float32,
            device=accelerator.device,
        )

        if not _is_finite_tensor(logprobs_old):
            old_min = torch.nan_to_num(logprobs_old).min().item()
            old_max = torch.nan_to_num(logprobs_old).max().item()
            logger.warning(
                "Non-finite old/behavior logprobs; skipping batch",
                extra={"min": float(old_min), "max": float(old_max)},
            )
            continue

        # Compute REFERENCE model logprobs for KL divergence penalty
        # (separate from behavior policy - this is for regularization)
        if ref_model is not None:
            with torch.no_grad():
                logprobs_ref_sum, logprobs_ref_per_token = compute_logprobs(
                    ref_model,
                    input_ids_tensor,
                    attention_mask_tensor,
                    batch_prompt_lens,
                    batch_comp_lens,
                    return_per_token=True,
                )
            if not _is_finite_tensor(logprobs_ref_sum):
                ref_min = torch.nan_to_num(logprobs_ref_sum).min().item()
                ref_max = torch.nan_to_num(logprobs_ref_sum).max().item()
                logger.warning(
                    "Non-finite reference logprobs; skipping batch",
                    extra={"min": float(ref_min), "max": float(ref_max)},
                )
                continue
        else:
            # If no ref model, set ref logprobs to old logprobs (zero KL)
            logprobs_ref_sum = logprobs_old
            logprobs_ref_per_token = logprobs_current_per_token.detach()

        # Importance sampling log-ratio: π_current / π_old (behavior)
        log_ratio = logprobs_current_sum - logprobs_old
        if not _is_finite_tensor(log_ratio):
            logger.debug("Non-finite log-ratio before clamp; applying clamp")
        # Moderate clamp for numerical stability when exponentiating to ratios
        log_ratio_clamped = torch.clamp(
            log_ratio, min=-TRAINER_LOGRATIO_CLAMP, max=TRAINER_LOGRATIO_CLAMP
        )

        # Advantage normalization for stable gradients
        # IMPORTANT: Normalize advantages to have unit variance while preserving zero-sum
        # This prevents gradient explosion when advantages have large magnitude
        # Clip extreme outliers first, then standardize
        perc_val = float(TRAINER_ADV_CLIP_PERCENTILE)
        q = max(0.0, min(100.0, perc_val)) / 100.0
        if 0.0 < q < 1.0:
            clip_val = torch.quantile(advantages_tensor.abs(), q)
            if torch.isfinite(clip_val):
                advantages_tensor = advantages_tensor.clamp(-clip_val, clip_val)

        # We don't normalize advantages since they are already normalized
        advantages_normalized = advantages_tensor

        # Policy gradient loss with importance sampling and PPO-style clipping
        ratio_clip_frac_val: float = 0.0
        ratio_ceiling_frac_val: float = 0.0
        ratios_pre_ceiling = torch.exp(log_ratio_clamped)
        if TRAINER_USE_IS:
            # Importance sampling ratio with numerical stability clamp, then hard ceiling
            if TRAINER_IS_RATIO_MAX > 0.0:
                ceiling_mask = ratios_pre_ceiling > TRAINER_IS_RATIO_MAX
                ratios = torch.clamp(ratios_pre_ceiling, max=TRAINER_IS_RATIO_MAX)
            else:
                ceiling_mask = torch.zeros_like(ratios_pre_ceiling, dtype=torch.bool)
                ratios = ratios_pre_ceiling

            # Asymmetric PPO-style clipping (DAPO-style): tighter lower bound, relaxed upper bound
            lower = 1.0 - TRAINER_PPO_CLIP_EPS
            upper = 1.0 + TRAINER_PPO_CLIP_EPS_UPPER
            ratios_clipped = torch.clamp(ratios, lower, upper)

            # Track clipping statistics for monitoring
            clip_mask = (ratios < lower) | (ratios > upper)
            ratio_clip_frac_val = clip_mask.float().mean().item()
            ratio_ceiling_frac_val = ceiling_mask.float().mean().item()

            pg_unclipped = ratios * advantages_normalized
            pg_clipped = ratios_clipped * advantages_normalized
            loss_pg = -torch.min(pg_unclipped, pg_clipped).mean()
        else:
            # On-policy: no ratio, just advantage-weighted logprobs
            loss_pg = -(advantages_normalized * logprobs_current_sum).mean()

        # KL divergence penalty: KL(π_current || π_ref)
        # Proper per-token KL with completion masks
        max_comp_len = logprobs_current_per_token.shape[1]
        # Build completion mask: [batch_size, max_comp_len]
        # Vectorized construction of completion mask: [batch_size, max_comp_len]
        comp_lens_tensor = torch.as_tensor(batch_comp_lens, device=accelerator.device)
        token_positions = torch.arange(max_comp_len, device=accelerator.device)
        completion_mask = (token_positions.unsqueeze(0) < comp_lens_tensor.unsqueeze(1)).to(
            torch.float32
        )

        # Per-token log-ratio for KL: log(π/π_ref)
        per_token_log_ratio = logprobs_current_per_token - logprobs_ref_per_token
        # Approximate KL via: 0.5 * E[(log π - log π_ref)²]
        # This is a symmetric approximation that's numerically stable
        per_token_kl = 0.5 * per_token_log_ratio.pow(2) * completion_mask
        # Mean over valid tokens (token-weighted, not sequence-weighted)
        kl_tensor = per_token_kl.sum() / completion_mask.sum().clamp(min=1.0)
        loss_kl = current_kl_coef * kl_tensor
        kl_value = float(kl_tensor.detach().item())

        entropies = compute_entropy(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
        )
        if not _is_finite_tensor(entropies):
            logger.warning("Non-finite entropies; skipping batch")
            continue
        loss_entropy = -TRAINER_ENTROPY_COEF * entropies.mean()

        loss_total = loss_pg + loss_kl + loss_entropy

        # Skip backward on non-finite loss to avoid corrupting optimizer state
        if not torch.isfinite(loss_total):
            loss_pg_v = torch.nan_to_num(loss_pg).item()
            loss_kl_v = torch.nan_to_num(loss_kl).item()
            loss_ent_v = torch.nan_to_num(loss_entropy).item()
            logger.warning(
                "Non-finite total loss; skipping batch",
                extra={
                    "loss_pg": float(loss_pg_v),
                    "loss_kl": float(loss_kl_v),
                    "loss_entropy": float(loss_ent_v),
                },
            )
            continue

        # Zero gradients before backward pass (only at start of accumulation)
        if grad_accum_counter == 0:
            optimizer.zero_grad(set_to_none=True)

        # Scale loss by accumulation steps to keep effective LR stable
        scaled_loss = loss_total / float(TRAINER_GRAD_ACCUM_STEPS)
        accelerator.backward(scaled_loss)
        grad_accum_counter += 1

        grad_norm_scalar = None
        # Only step optimizer and clip gradients every N accumulation steps
        if grad_accum_counter >= TRAINER_GRAD_ACCUM_STEPS:
            # Clip gradients in fp32 (no mixed precision)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TRAINER_GRAD_CLIP,
            )
            grad_norm_scalar = grad_norm.item()

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                loss_tot_v = torch.nan_to_num(loss_total).item()
                lr_mean = torch.nan_to_num(log_ratio).mean().item()
                lr_std = torch.nan_to_num(log_ratio).std().item()
                adv_mean = advantages_tensor.mean().item()
                adv_std = advantages_tensor.std().item()
                logger.warning(
                    "NaN/Inf gradient norm detected; skipping batch",
                    extra={
                        "loss_total": float(loss_tot_v),
                        "log_ratio_mean": float(lr_mean),
                        "log_ratio_std": float(lr_std),
                        "adv_mean": float(adv_mean),
                        "adv_std": float(adv_std),
                    },
                )
                grad_accum_counter = 0
                continue

            # Standard optimizer step in fp32
            optimizer.step()
            
            logger.info(f"Optimizer step completed. Current KL coef: {current_kl_coef}")

            # Adaptive KL adjustment after each optimizer step based on observed KL
            if TRAINER_ADAPTIVE_KL:
                # Increase/decrease coefficient multiplicatively to target KL
                if kl_value > TRAINER_KL_TARGET * 1.2:
                    current_kl_coef = min(TRAINER_KL_MAX, current_kl_coef * TRAINER_KL_ADAPT_RATE)
                elif kl_value < TRAINER_KL_TARGET * 0.8:
                    current_kl_coef = max(TRAINER_KL_MIN, current_kl_coef / TRAINER_KL_ADAPT_RATE)
            grad_accum_counter = 0

        epoch_metrics["loss_total"].append(loss_total.item())
        epoch_metrics["loss_pg"].append(loss_pg.item())
        epoch_metrics["loss_kl"].append(loss_kl.item())
        epoch_metrics["loss_entropy"].append(loss_entropy.item())
        epoch_metrics["grad_norm"].append(grad_norm_scalar if grad_norm_scalar is not None else 0.0)
        epoch_metrics["advantage_mean"].append(advantages_tensor.mean().item())
        epoch_metrics["advantage_std"].append(advantages_tensor.std().item())
        epoch_metrics["entropy_mean"].append(entropies.mean().item())
        # advantages_normalized is now same as advantages_tensor (no batch normalization)
        epoch_metrics["advantage_mean_normalized"].append(advantages_normalized.mean().item())
        epoch_metrics["advantage_std_normalized"].append(advantages_normalized.std().item())
        # Track divergence metrics (use clamped log_ratio for safe exponentiation)
        epoch_metrics["kl_divergence"].append(kl_value)
        # Pre-ceiling ratio stats (consistent with historical logging)
        epoch_metrics["ratio_mean"].append(ratios_pre_ceiling.mean().item())
        epoch_metrics["ratio_std"].append(ratios_pre_ceiling.std().item())
        # Clipping diagnostics (0 when importance sampling disabled)
        epoch_metrics["ratio_clip_frac"].append(ratio_clip_frac_val)
        epoch_metrics["ratio_ceiling_frac"].append(ratio_ceiling_frac_val)
        epoch_metrics["kl_coef"].append(current_kl_coef)
        # Track how many samples have miner-provided behavior logprobs
        epoch_metrics["behavior_logprobs_frac"].append(1.0)
        # Track reward curve
        epoch_metrics["reward_mean"].append(torch.tensor(batch_rewards).mean().item())
        epoch_metrics["reward_std"].append(torch.tensor(batch_rewards).std().item())

        # Log per-batch metrics if monitor is available for fine-grained tracking
        if monitor is not None and algorithm is not None:
            # Use global batch counter for smooth, continuous x-axis across all windows
            algorithm.global_batch_counter += 1
            batch_log_metrics = {
                "loss_total": loss_total.item(),
                "loss_pg": loss_pg.item(),
                "loss_kl": loss_kl.item(),
                "loss_entropy": loss_entropy.item(),
                "grad_norm": grad_norm_scalar if grad_norm_scalar is not None else 0.0,
                "advantage_mean": advantages_tensor.mean().item(),
                "reward_mean": torch.tensor(batch_rewards).mean().item(),
                "kl_divergence": kl_value,
                "entropy_mean": entropies.mean().item(),
                "ratio_clip_frac": ratio_clip_frac_val,
                "ratio_ceiling_frac": ratio_ceiling_frac_val,
            }
            for key, value in batch_log_metrics.items():
                try:
                    await monitor.log_gauge(
                        f"training/batch/{key}",
                        value,
                        tags={"batch_step": str(algorithm.global_batch_counter)},  # Global counter
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to log batch metric %s: %s", key, exc)

    # Handle remaining accumulated gradients at epoch end
    if grad_accum_counter > 0:
        final_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            TRAINER_GRAD_CLIP,
        )
        if not (torch.isnan(final_grad_norm) or torch.isinf(final_grad_norm)):
            optimizer.step()

    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in epoch_metrics.items()
    }


class GRPOAlgorithm(TrainingAlgorithm):
    """GRPO algorithm implementation."""

    name: str = "grpo"

    async def train_epoch(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        groups: list[GRPOGroup],
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        monitor: Any | None,
        window: int,
        config: TrainingConfig,
    ) -> dict[str, float]:
        """Train for one epoch using GRPO algorithm."""
        # Increment global epoch counter for continuous tracking
        self.global_epoch_counter += 1

        return await train_grpo_epoch(
            model,
            ref_model,
            tokenizer,
            groups,
            optimizer,
            accelerator,
            monitor,
            window,
            algorithm=self,  # Pass self to access global counters
        )
