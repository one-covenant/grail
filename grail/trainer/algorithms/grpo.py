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

from grail.shared.constants import (
    GRPO_RANKING_REWARD_WEIGHT,
    GRPO_RANKING_VARIANCE_WEIGHT,
    IMPORTANCE_SAMPLING_LEVEL,
    ROLLOUTS_PER_PROBLEM,
    is_kl_enabled,
)
from grail.trainer.metrics import (
    KMetricsAggregator,
    TaskReplicateResult,
    derive_k_values,
)
from grail.trainer.param_tracker import (
    ParamChangeTracker,
    log_param_change_metrics,
)
from grail.trainer.sparse_quality import (
    SparseQualityAnalyzer,
    log_sparse_quality_metrics,
)

from .base import TrainingAlgorithm

if TYPE_CHECKING:
    from grail.infrastructure.chain import GrailChainManager
    from grail.shared.schemas import BucketCredentials
    from grail.trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


async def fetch_multiple_miners_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Fetch miner window data (lazy import).

    Keeping this import lazy prevents `grail.trainer.algorithms.grpo` from pulling in
    bittensor/networking dependencies during offline usage (e.g., research/offline_trainer).
    """
    try:
        from grail.infrastructure.miner_data import fetch_multiple_miners_data as _impl
    except Exception as exc:  # pragma: no cover - optional in offline mode
        raise RuntimeError("Miner data fetching is unavailable in offline mode.") from exc

    return await _impl(*args, **kwargs)


class AdaptiveKLController:
    """Stateful controller for adaptive KL coefficient.

    Maintains a persistent KL penalty coefficient and updates it after each
    optimizer step based on observed KL relative to a target range.
    """

    def __init__(
        self,
        *,
        initial: float,
        target: float,
        min_value: float,
        max_value: float,
        adapt_rate: float,
    ) -> None:
        self.value: float = float(initial)
        self.target: float = float(target)
        self.min_value: float = float(min_value)
        self.max_value: float = float(max_value)
        self.adapt_rate: float = float(adapt_rate)

    def update(self, observed_kl: float) -> float:
        """Update and return the current KL coefficient based on observed KL.

        The update is multiplicative outside a tolerance band around the target.
        """
        try:
            kl = float(observed_kl)
        except Exception:  # noqa: BLE001
            return self.value

        upper = self.target * 1.2
        lower = self.target * 0.8
        if kl > upper:
            self.value = min(self.max_value, self.value * self.adapt_rate)
        elif kl < lower:
            self.value = max(self.min_value, self.value / self.adapt_rate)
        return self.value

    def state_dict(self) -> dict[str, float]:
        return {"value": float(self.value)}

    def load_state_dict(self, state: dict[str, float]) -> None:
        try:
            self.value = float(state.get("value", self.value))
        except Exception:  # noqa: BLE001
            pass


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

    logger.info("=" * 100, extra={"markup": False})
    logger.info("DECODED ROLLOUT SAMPLES (Model: %s)", model_name, extra={"markup": False})
    logger.info("=" * 100, extra={"markup": False})

    for miner_hotkey, window_data in miner_data.items():
        if not isinstance(window_data, dict):
            continue

        inferences = window_data.get("inferences", [])
        if not isinstance(inferences, list):
            continue

        logger.info(
            "\nMiner: %s | Total rollouts: %d",
            miner_hotkey[:16],
            len(inferences),
            extra={"markup": False},
        )

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
                extra={"markup": False},
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
                        extra={"markup": False},
                    )
                except Exception as e:
                    logger.debug("Failed to decode prompt: %s", e, extra={"markup": False})

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
                            extra={"markup": False},
                        )
                    except Exception as e:
                        logger.debug("Failed to decode completion: %s", e, extra={"markup": False})
            else:
                logger.info("    [No tokens available]", extra={"markup": False})

    logger.info("=" * 100, extra={"markup": False})


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


def _compute_training_metrics(
    groups: list[GRPOGroup],
    window: int,
    eos_token_id: int | None = None,
    rollouts_per_problem: int | None = None,
) -> dict[str, Any]:
    """Compute and log pre-training data quality metrics for GRPO groups.

    Args:
        groups: List of GRPO groups
        window: Training window number
        eos_token_id: Optional EOS token ID to determine terminated completions
        rollouts_per_problem: Number of rollouts per problem (for k-metric derivation)

    Returns:
        Dictionary of computed metrics keyed by metric name (with pass@k, mean@k, etc.)
    """
    if rollouts_per_problem is None:
        rollouts_per_problem = 5  # Fallback default

    try:
        report_ks = derive_k_values(rollouts_per_problem)
    except Exception:  # noqa: BLE001
        report_ks = [1, 5, 10]
        if groups:
            report_ks.append(len(groups[0].rollouts))
        report_ks = sorted(set(report_ks))

    aggregator = KMetricsAggregator(report_ks=report_ks)

    # Track completion lengths
    all_completion_lengths: list[int] = []
    terminated_completion_lengths: list[int] = []

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

            # Track completion lengths
            completion_len = r.completion_length
            all_completion_lengths.append(completion_len)

            # Check if completion is terminated with EOS
            if eos_token_id is not None and completion_len > 0:
                # Get the last token of the completion
                prompt_len = r.prompt_length
                if len(r.tokens) >= prompt_len + completion_len:
                    last_completion_token = r.tokens[prompt_len + completion_len - 1]
                    if last_completion_token == eos_token_id:
                        terminated_completion_lengths.append(completion_len)

    prefilter_metrics = aggregator.summarize()

    # Add length-related metrics
    if all_completion_lengths:
        prefilter_metrics["mean_length"] = float(np.mean(all_completion_lengths))
        prefilter_metrics["min_length"] = float(np.min(all_completion_lengths))
        prefilter_metrics["max_length"] = float(np.max(all_completion_lengths))

    # Add terminated completion metrics if we have EOS token ID
    if terminated_completion_lengths:
        prefilter_metrics["mean_terminated_length"] = float(np.mean(terminated_completion_lengths))
        prefilter_metrics["min_terminated_length"] = float(np.min(terminated_completion_lengths))
        prefilter_metrics["max_terminated_length"] = float(np.max(terminated_completion_lengths))
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
        # Add length metrics to summary
        if "mean_length" in prefilter_metrics:
            summary_bits.append(f"mean_length={prefilter_metrics.get('mean_length', 0.0):.1f}")
        if "mean_terminated_length" in prefilter_metrics:
            summary_bits.append(
                f"mean_terminated_length={prefilter_metrics.get('mean_terminated_length', 0.0):.1f}"
            )
        logger.info(
            "Training data metrics (pre-filter, window %s): %s",
            window,
            ", ".join(summary_bits),
        )

    # Return metrics with metadata for async logging by caller
    return prefilter_metrics


def _group_reward_per_token(group: GRPOGroup) -> float:
    """Calculate mean reward per completion token for a GRPO group (GFPO).

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


def _group_advantage_variance(group: GRPOGroup) -> float:
    """Calculate advantage variance for a GRPO group (learning signal strength).

    Higher variance indicates stronger learning signal - diverse outcomes that
    provide clear policy gradient direction.

    Args:
        group: The GRPO group to calculate advantage variance for

    Returns:
        Variance of advantages across rollouts (0.0 if insufficient rollouts)
    """
    if len(group.rollouts) < 2:
        return 0.0
    advantages = [r.advantage for r in group.rollouts]
    mean_adv = sum(advantages) / len(advantages)
    return sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)


def _group_efficiency_score(
    group: GRPOGroup, reward_weight: float = 0.7, variance_weight: float = 0.3
) -> float:
    """Compute combined efficiency score for group ranking (GFPO + variance).

    Combines token efficiency (reward/token) with learning signal strength
    (advantage variance) using configurable weights. Groups are ranked by this
    score for top-k selection.

    Args:
        group: The GRPO group to score
        reward_weight: Weight for reward/token component (default: 0.7)
        variance_weight: Weight for advantage variance component (default: 0.3)

    Returns:
        Combined efficiency score (normalized, higher is better)
    """
    reward_per_token = _group_reward_per_token(group)
    adv_variance = _group_advantage_variance(group)
    return reward_weight * reward_per_token + variance_weight * adv_variance


def _filter_valid_groups(
    groups: list[GRPOGroup],
    advantage_tolerance: float,
    window: int,
    config: TrainingConfig | None = None,
) -> list[GRPOGroup]:
    """Filter and refine GRPO groups using multi-stage pipeline.

    Implements filtering techniques from recent research (DAPO, GFPO, 2025):
    1. Structural validation (group size and zero-sum advantage condition)
    2. DAPO filtering: Remove uninformative groups (zero advantage variance)
    3. Completion token constraints (optional, from config)
    4. GFPO refinement: Success rate, reward/token thresholds, quantile dropping
    5. Top-k ranking: Combined efficiency score (reward/token + advantage variance)

    Args:
        groups: List of GRPO groups to filter
        advantage_tolerance: Maximum allowed sum of advantages in a group
        window: Training window number (for logging purposes)
        config: Optional training config with filtering parameters

    Returns:
        List of filtered and refined GRPO groups (up to max_groups)
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

    # Stage 2: DAPO filtering - remove uninformative groups (zero advantage variance)
    # Groups where all rollouts have identical rewards provide no learning signal
    groups_with_variance: list[GRPOGroup] = [
        group for group in valid_groups if _has_advantage_variance(group)
    ]
    zero_variance_count: int = len(valid_groups) - len(groups_with_variance)
    if zero_variance_count > 0:
        logger.warning(
            "DAPO: Filtered out %s uninformative groups (zero variance) for window %s",
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

        # Stage 5: Rank by combined efficiency score and select top-k groups
        max_groups: int = int(getattr(config, "grpo_max_groups_per_window", 10000))
        if len(refined_groups) > max_groups:
            # Get ranking weights from config or fall back to constants
            reward_weight: float = float(
                getattr(config, "grpo_ranking_reward_weight", GRPO_RANKING_REWARD_WEIGHT)
            )
            variance_weight: float = float(
                getattr(config, "grpo_ranking_variance_weight", GRPO_RANKING_VARIANCE_WEIGHT)
            )
            # Score each group with combined efficiency metric (GFPO + variance)
            scored = [
                (group, _group_efficiency_score(group, reward_weight, variance_weight))
                for group in refined_groups
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            refined_groups = [group for group, _ in scored[:max_groups]]
            logger.info(
                "Selected top %s/%s groups by efficiency score (reward_wt=%.2f, var_wt=%.2f) "
                "for window %s",
                max_groups,
                len(scored),
                reward_weight,
                variance_weight,
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
    eos_token_id: int | None = None,
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
        eos_token_id: Optional EOS token ID to determine terminated completions

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
    rollouts_per_problem = config.rollouts_per_problem if config else 5
    prefilter_metrics = _compute_training_metrics(
        groups, window, eos_token_id, rollouts_per_problem
    )

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

    # Cast to float32 for precise log_softmax (even if model is bfloat16)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
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

    # Cast to float32 for precise softmax/log_softmax (even if model is bfloat16)
    logits_f32 = logits.float()
    probs = F.softmax(logits_f32, dim=-1)
    log_probs = F.log_softmax(logits_f32, dim=-1)
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


class GRPOAlgorithm(TrainingAlgorithm):
    """GRPO algorithm implementation with support for multiple loss variants.

    Supported variants:
    - 'grpo': Group Relative Policy Optimization (default) - averages per-sequence, then batch
    - 'bnpo': Batch Normalization Policy Optimization - averages over all tokens globally
    - 'dapo': Distributed Adaptive Policy Optimization - normalizes by total completion tokens
    - 'dr_grpo': Denominator-Reduced GRPO - normalizes by batch_size × max_completion_length

    Importance Sampling Levels:
    - 'sequence': Compute one importance sampling ratio per sequence (default)
        Sums log-probabilities over tokens first, then computes ratio and clips at sequence level.
        Standard PPO/GRPO approach with sequence-level clipping.
    - 'token': Compute importance sampling ratio per token independently
        Computes ratio for each token, then clips each token's contribution independently.
        More fine-grained clipping that can be more stable for highly variable sequences.
        Inspired by HuggingFace TRL GRPO implementation.
    """

    name: str = "grpo"

    def __init__(
        self,
        adaptive_kl_enabled: bool = False,
        grpo_variant: str | None = None,
        importance_sampling_level: str | None = None,
        config: TrainingConfig | None = None,
    ) -> None:
        super().__init__()

        # Use config if provided, otherwise construct from individual parameters
        if config is not None:
            self.config = config
        else:
            # Fallback: create minimal config with provided parameters (backward compatibility)
            from grail.trainer.config import TrainingConfig as TC

            self.config = TC(grpo_variant=grpo_variant or "grpo")

        # Optimizer step counter for parameter change tracking
        self.optimizer_step_count: int = 0

        # Parameter change tracker (memory-safe, CPU-offloaded)
        self.param_tracker = ParamChangeTracker.from_config(self.config)
        if self.param_tracker.is_enabled():
            logger.info(
                "Parameter change tracking enabled: measure every %d optimizer steps, "
                "threshold=%.0e, per_layer=%s, components=%s",
                self.param_tracker.measure_interval,
                self.param_tracker.threshold,
                self.param_tracker.track_per_layer,
                self.param_tracker.track_components,
            )

        # Sparse quality analyzer (uses param_tracker's snapshot, runs at same interval)
        self.sparse_analyzer = SparseQualityAnalyzer.from_config(self.param_tracker, self.config)
        if self.sparse_analyzer.enabled:
            logger.info("Sparse quality analysis enabled (runs with param tracking)")

        # Persistent adaptive KL controller across epochs/windows
        self.kl_controller: AdaptiveKLController = AdaptiveKLController(
            initial=self.config.kl_coef,
            target=self.config.kl_target,
            min_value=self.config.kl_min,
            max_value=self.config.kl_max,
            adapt_rate=self.config.kl_adapt_rate,
        )
        self.adaptive_kl_enabled: bool = bool(adaptive_kl_enabled)

        # Validate and set loss variant (with common aliases)
        # If not explicitly provided, use config's default
        raw_variant = (grpo_variant or self.config.grpo_variant).strip().lower()
        aliases = {
            "gspo": "dr_grpo",  # common misnomer maps to denominator-reduced GRPO
            "dr-grpo": "dr_grpo",
        }
        selected_variant = aliases.get(raw_variant, raw_variant)
        valid_variants = {"grpo", "bnpo", "dapo", "dr_grpo"}
        if selected_variant not in valid_variants:
            raise ValueError(
                f"Invalid grpo_variant '{grpo_variant}'. Must be one of {sorted(valid_variants)}"
            )
        self.grpo_variant: str = selected_variant
        logger.info(f"Using {self.grpo_variant.upper()} loss variant")

        # Validate and set importance sampling level
        # Controls whether clipping is applied at the sequence or token level
        # If not explicitly provided, use the environment-configured default
        raw_is_level = (importance_sampling_level or IMPORTANCE_SAMPLING_LEVEL).strip().lower()
        valid_is_levels = {"sequence", "token"}
        if raw_is_level not in valid_is_levels:
            raise ValueError(
                f"Invalid importance_sampling_level '{raw_is_level}'. "
                f"Must be one of {sorted(valid_is_levels)}"
            )
        self.importance_sampling_level: str = raw_is_level
        logger.info(f"Using {self.importance_sampling_level}-level importance sampling")

    def _prepare_batch_tensors(
        self,
        batch_rollouts: list[GRPORollout],
        tokenizer: Any,
        accelerator: Accelerator,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[int],
        list[float],
        list[list[float]],
        list[float],
    ]:
        """Prepare batch tensors from rollouts.

        Args:
            batch_rollouts: List of rollouts for this batch
            tokenizer: Tokenizer for padding
            accelerator: Accelerator for device placement

        Returns:
            Tuple of (input_ids, attention_mask, prompt_lengths, completion_lengths,
                     advantages, behavior_per_token_logprobs, rewards)
        """
        batch_tokens: list[list[int]] = []
        batch_prompt_lens: list[int] = []
        batch_comp_lens: list[int] = []
        batch_advantages: list[float] = []
        batch_behavior_per_token_logprobs: list[list[float]] = []
        batch_rewards: list[float] = []

        for rollout in batch_rollouts:
            tokens = rollout.tokens[: self.config.max_length]
            batch_tokens.append(tokens)

            # CRITICAL FIX: Recalculate prompt/completion lengths after truncation
            # Both values must be clamped to valid non-negative ranges
            #
            # Case 1: prompt_length > max_length
            #   - actual_prompt_len = max_length (entire sequence is prompt)
            #   - actual_comp_len = 0 (no completion tokens in truncated sequence)
            #
            # Case 2: prompt_length <= max_length
            #   - actual_prompt_len = prompt_length (unchanged)
            #   - actual_comp_len = min(completion_length, max_length - prompt_length)
            #
            # The max(0, ...) guards are critical to prevent negative lengths which
            # would corrupt mask construction and gradient computation.
            actual_prompt_len = min(rollout.prompt_length, self.config.max_length)
            actual_comp_len = max(
                0,
                min(
                    rollout.completion_length,
                    self.config.max_length - rollout.prompt_length,
                ),
            )
            batch_prompt_lens.append(actual_prompt_len)
            batch_comp_lens.append(actual_comp_len)

            batch_advantages.append(rollout.advantage)
            batch_rewards.append(rollout.reward)

            # Miner-provided per-token logprobs must exist; extract completion portion
            tlp: list[float] = list((rollout.token_logprobs or [])[: self.config.max_length])
            prompt_len = actual_prompt_len
            comp_len = actual_comp_len
            expected_len = prompt_len + comp_len

            if len(tlp) >= expected_len:
                # Extract completion logprobs: indices [prompt_len:prompt_len+comp_len]
                completion_logprobs = tlp[prompt_len : prompt_len + comp_len]
            else:
                # Legacy: miner provided only completion logprobs; respect truncation
                completion_logprobs = tlp[:comp_len]

            batch_behavior_per_token_logprobs.append(completion_logprobs)

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

        return (
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
            batch_advantages,
            batch_behavior_per_token_logprobs,
            batch_rewards,
        )

    def _normalize_advantages(self, advantages_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize advantages with clipping and standardization.

        Args:
            advantages_tensor: Raw advantages tensor

        Returns:
            Normalized advantages tensor
        """
        # Advantage normalization for stable gradients
        # IMPORTANT: Normalize advantages to have unit variance while preserving zero-sum
        # This prevents gradient explosion when advantages have large magnitude
        # Clip extreme outliers first, then standardize
        perc_val = float(self.config.adv_clip_percentile)
        q = max(0.0, min(100.0, perc_val)) / 100.0
        if 0.0 < q < 1.0:
            clip_val = torch.quantile(advantages_tensor.abs(), q)
            if torch.isfinite(clip_val):
                advantages_tensor = advantages_tensor.clamp(-clip_val, clip_val)

        # We don't normalize advantages since they are already normalized
        return advantages_tensor

    def _compute_policy_gradient_loss(
        self,
        logprobs_current_per_token: torch.Tensor,
        logprobs_old_per_token: torch.Tensor,
        advantages_normalized: torch.Tensor,
        completion_mask: torch.Tensor,
        batch_size: int,
        total_completion_tokens: int | None = None,
    ) -> tuple[torch.Tensor, float, float, torch.Tensor]:
        """Compute policy gradient loss with importance sampling and PPO clipping.

        Supports both sequence-level and token-level importance sampling based on
        self.importance_sampling_level:
        - "sequence": Compute one ratio per sequence (sum logprobs, then compute ratio)
        - "token": Compute ratio per token independently (more fine-grained clipping)

        Args:
            logprobs_current_per_token: [batch_size, max_comp_len] current policy per-token logprobs
            logprobs_old_per_token: [batch_size, max_comp_len] behavior policy per-token logprobs
            advantages_normalized: [batch_size] normalized advantages
            completion_mask: [batch_size, max_comp_len] mask (1=real token, 0=padding)
            batch_size: Number of sequences in batch
            total_completion_tokens: Total completion tokens across all processes (for DAPO)

        Returns:
            Tuple of (loss_pg, ratio_clip_frac, ratio_ceiling_frac, ratios_pre_ceiling)
        """
        # Compute log-ratios based on importance sampling level
        if self.importance_sampling_level == "token":
            # Token-level: compute ratio per token independently
            # Shape: [batch_size, max_comp_len]
            log_ratio = logprobs_current_per_token - logprobs_old_per_token

            if not _is_finite_tensor(log_ratio):
                logger.debug("Non-finite per-token log-ratio before clamp; applying clamp")
            # Moderate clamp for numerical stability when exponentiating to ratios
            log_ratio_clamped = torch.clamp(
                log_ratio, min=-self.config.logratio_clamp, max=self.config.logratio_clamp
            )

            # For reporting, compute sequence-level ratios (same as original for monitoring)
            logprobs_current_sum = (logprobs_current_per_token * completion_mask).sum(dim=1)
            logprobs_old_sum = (logprobs_old_per_token * completion_mask).sum(dim=1)
            log_ratio_seq = logprobs_current_sum - logprobs_old_sum
            log_ratio_seq_clamped = torch.clamp(
                log_ratio_seq, min=-self.config.logratio_clamp, max=self.config.logratio_clamp
            )
            ratios_pre_ceiling = torch.exp(log_ratio_seq_clamped)

        elif self.importance_sampling_level == "sequence":
            # Sequence-level: sum logprobs first, then compute ratio
            # Shape: [batch_size]
            logprobs_current_sum = (logprobs_current_per_token * completion_mask).sum(dim=1)
            logprobs_old_sum = (logprobs_old_per_token * completion_mask).sum(dim=1)
            log_ratio = logprobs_current_sum - logprobs_old_sum

            if not _is_finite_tensor(log_ratio):
                logger.debug("Non-finite log-ratio before clamp; applying clamp")
            # Moderate clamp for numerical stability when exponentiating to ratios
            log_ratio_clamped = torch.clamp(
                log_ratio, min=-self.config.logratio_clamp, max=self.config.logratio_clamp
            )
            ratios_pre_ceiling = torch.exp(log_ratio_clamped)
        else:
            raise ValueError(
                f"Invalid importance_sampling_level: {self.importance_sampling_level}. "
                "Must be 'sequence' or 'token'."
            )

        # Policy gradient loss with importance sampling and PPO-style clipping
        ratio_clip_frac_val: float = 0.0
        ratio_ceiling_frac_val: float = 0.0

        if self.config.use_is:
            # Compute ratios with optional ceiling
            ratios = torch.exp(log_ratio_clamped)

            # Apply hard ceiling if configured
            if self.config.is_ratio_max > 0.0:
                ceiling_mask = ratios > self.config.is_ratio_max
                ratios = torch.clamp(ratios, max=self.config.is_ratio_max)
                # Compute ceiling fraction from valid (non-padding) tokens
                if self.importance_sampling_level == "token":
                    ratio_ceiling_frac_val = (
                        (ceiling_mask * completion_mask).sum()
                        / completion_mask.sum().clamp(min=1.0)
                    ).item()
                else:
                    ratio_ceiling_frac_val = ceiling_mask.float().mean().item()
            else:
                ceiling_mask = None

            # Asymmetric PPO-style clipping (DAPO-style): tighter lower bound, relaxed upper bound
            lower = 1.0 - self.config.ppo_clip_eps
            upper = 1.0 + self.config.ppo_clip_eps_upper
            ratios_clipped = torch.clamp(ratios, lower, upper)

            # Track clipping statistics for monitoring
            clip_mask = (ratios < lower) | (ratios > upper)
            if self.importance_sampling_level == "token":
                # For token-level, compute clipping fraction over valid tokens
                ratio_clip_frac_val = (
                    (clip_mask * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                ).item()
            else:
                # For sequence-level, compute clipping fraction over sequences
                ratio_clip_frac_val = clip_mask.float().mean().item()

            # Compute per-token loss based on importance sampling level
            if self.importance_sampling_level == "token":
                # Token-level: ratios and advantages interact per-token
                # ratios: [batch_size, max_comp_len]
                # advantages: [batch_size] -> expand to [batch_size, 1] for broadcasting
                advantages_expanded = advantages_normalized.unsqueeze(1)  # [batch_size, 1]

                pg_unclipped = ratios * advantages_expanded  # [batch_size, max_comp_len]
                pg_clipped = ratios_clipped * advantages_expanded  # [batch_size, max_comp_len]
                per_token_loss = -torch.min(pg_unclipped, pg_clipped)  # [batch_size, max_comp_len]

            elif self.importance_sampling_level == "sequence":
                # Sequence-level: ratios are [batch_size], broadcast to token dimension
                # Expand advantages to [batch_size, 1] for broadcasting
                advantages_expanded = advantages_normalized.unsqueeze(1)  # [batch_size, 1]
                ratios_expanded = ratios.unsqueeze(1)  # [batch_size, 1]
                ratios_clipped_expanded = ratios_clipped.unsqueeze(1)  # [batch_size, 1]

                pg_unclipped = ratios_expanded * advantages_expanded  # [batch_size, 1]
                pg_clipped = ratios_clipped_expanded * advantages_expanded  # [batch_size, 1]
                per_token_loss = -torch.min(pg_unclipped, pg_clipped)  # [batch_size, 1]

                # Expand to [batch_size, max_comp_len] by broadcasting
                per_token_loss = per_token_loss.expand(-1, completion_mask.size(1))

            # Aggregate using variant-specific strategy
            max_completion_length = completion_mask.size(1)
            loss_pg = self._aggregate_policy_loss(
                per_token_loss,
                completion_mask,
                batch_size,
                max_completion_length,
                total_completion_tokens,
            )
        else:
            # On-policy: no ratio, just advantage-weighted logprobs
            if self.importance_sampling_level == "token":
                # Sum over tokens first for on-policy
                logprobs_current_sum = (logprobs_current_per_token * completion_mask).sum(dim=1)
            # else: logprobs_current_sum already computed for sequence-level
            loss_pg = -(advantages_normalized * logprobs_current_sum).mean()

        return loss_pg, ratio_clip_frac_val, ratio_ceiling_frac_val, ratios_pre_ceiling

    def _compute_kl_divergence_loss(
        self,
        logprobs_current_per_token: torch.Tensor,
        logprobs_ref_per_token: torch.Tensor,
        batch_comp_lens: list[int],
        current_kl_coef: float,
        accelerator: Accelerator,
    ) -> tuple[torch.Tensor, float]:
        """Compute KL divergence loss between current and reference policies.

        Args:
            logprobs_current_per_token: Per-token logprobs from current policy
            logprobs_ref_per_token: Per-token logprobs from reference policy
            batch_comp_lens: Completion lengths for each sequence
            current_kl_coef: Current KL coefficient
            accelerator: Accelerator for device placement

        Returns:
            Tuple of (loss_kl, kl_value)
        """
        max_comp_len = logprobs_current_per_token.shape[1]
        comp_lens_tensor = torch.as_tensor(batch_comp_lens, device=accelerator.device)
        token_positions = torch.arange(max_comp_len, device=accelerator.device)
        completion_mask = (token_positions.unsqueeze(0) < comp_lens_tensor.unsqueeze(1)).to(
            torch.float32
        )

        per_token_log_ratio = logprobs_current_per_token - logprobs_ref_per_token
        per_token_kl = 0.5 * per_token_log_ratio.pow(2) * completion_mask
        kl_tensor = per_token_kl.sum() / completion_mask.sum().clamp(min=1.0)
        loss_kl = current_kl_coef * kl_tensor
        kl_value = float(kl_tensor.detach().item())

        return loss_kl, kl_value

    def _aggregate_policy_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        batch_size: int,
        max_completion_length: int,
        total_completion_tokens: int | None = None,
    ) -> torch.Tensor:
        """Aggregate per-token policy loss using the selected variant's normalization strategy.

        Args:
            per_token_loss: [batch_size, max_comp_len] per-token loss values
            completion_mask: [batch_size, max_comp_len] mask (1=real token, 0=padding)
            batch_size: Number of sequences in batch
            max_completion_length: Maximum completion length in batch
            total_completion_tokens: Total completion tokens across all processes (for DAPO)

        Returns:
            Aggregated scalar loss

        Normalization strategies:
            - grpo: Average loss per sequence, then average over batch
                    → captures per-sequence quality, standard RL approach
            - bnpo: Average over all valid tokens globally
                    → treats all tokens equally, reduces variance
            - dr_grpo: Normalize by fixed denominator (batch_size × max_length)
                       → stable gradients regardless of actual token counts
            - dapo: Normalize by total completion tokens across all processes
                    → scales gradients consistently in distributed training
        """
        if self.grpo_variant == "grpo":
            # Average per-sequence, then batch: sum tokens per seq / seq_length, then mean over batch
            seq_losses = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(
                dim=1
            ).clamp(min=1.0)
            return seq_losses.mean()

        elif self.grpo_variant == "bnpo":
            # Global token average: sum all tokens / total valid tokens
            total_loss = (per_token_loss * completion_mask).sum()
            total_tokens = completion_mask.sum().clamp(min=1.0)
            return total_loss / total_tokens

        elif self.grpo_variant == "dr_grpo":
            # Fixed denominator normalization: independent of actual token counts
            total_loss = (per_token_loss * completion_mask).sum()
            denominator = batch_size * max_completion_length
            return total_loss / denominator

        elif self.grpo_variant == "dapo":
            # Distributed normalization: requires total tokens across all processes
            if total_completion_tokens is None:
                raise ValueError("DAPO variant requires total_completion_tokens parameter")
            total_loss = (per_token_loss * completion_mask).sum()
            return total_loss / total_completion_tokens

        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown variant: {self.grpo_variant}")

    def _collect_batch_metrics(
        self,
        epoch_metrics: dict[str, list[float]],
        loss_total: torch.Tensor,
        loss_pg: torch.Tensor,
        loss_kl: torch.Tensor,
        loss_entropy: torch.Tensor,
        grad_norm_scalar: float | None,
        advantages_tensor: torch.Tensor,
        advantages_normalized: torch.Tensor,
        entropies: torch.Tensor,
        kl_value: float,
        ratios_pre_ceiling: torch.Tensor,
        ratio_clip_frac_val: float,
        ratio_ceiling_frac_val: float,
        batch_rewards: list[float],
    ) -> None:
        """Collect batch metrics for epoch aggregation.

        Args:
            epoch_metrics: Dictionary to store metrics in
            loss_total: Total loss value
            loss_pg: Policy gradient loss
            loss_kl: KL divergence loss
            loss_entropy: Entropy loss
            grad_norm_scalar: Gradient norm (None if not computed)
            advantages_tensor: Raw advantages
            advantages_normalized: Normalized advantages
            entropies: Entropy values
            kl_value: KL divergence value
            ratios_pre_ceiling: Importance sampling ratios before ceiling
            ratio_clip_frac_val: Fraction of ratios clipped
            ratio_ceiling_frac_val: Fraction of ratios hitting ceiling
            batch_rewards: List of rewards for this batch
        """
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

        # Track reward curve
        epoch_metrics["reward_mean"].append(torch.tensor(batch_rewards).mean().item())
        epoch_metrics["reward_std"].append(torch.tensor(batch_rewards).std().item())

    def _finalize_actual_batch_metrics(
        self,
        tracker: dict[str, float],
        grad_norm_scalar: float | None,
    ) -> dict[str, float] | None:
        """Convert accumulated micro-batch stats into a full-batch metrics dict."""
        micro_batches = int(tracker.get("micro_batches", 0))
        if micro_batches == 0:
            return None

        def _safe_mean(total: float, denom: float) -> float:
            return float(total) / max(denom, 1.0)

        adv_mean = _safe_mean(tracker["adv_sum"], tracker["adv_count"])
        reward_mean = _safe_mean(tracker["reward_sum"], tracker["reward_count"])
        entropy_mean = _safe_mean(tracker["entropy_sum"], tracker["entropy_count"])
        ratio_denominator = (
            tracker["token_count"]
            if self.importance_sampling_level == "token"
            else tracker["sequence_count"]
        )

        return {
            "loss_total": _safe_mean(tracker["loss_total_sum"], micro_batches),
            "loss_pg": _safe_mean(tracker["loss_pg_sum"], micro_batches),
            "loss_kl": _safe_mean(tracker["loss_kl_sum"], micro_batches),
            "loss_entropy": _safe_mean(tracker["loss_entropy_sum"], micro_batches),
            "grad_norm": grad_norm_scalar if grad_norm_scalar is not None else 0.0,
            "advantage_mean": adv_mean,
            "reward_mean": reward_mean,
            "kl_divergence": _safe_mean(tracker["kl_sum"], micro_batches),
            "entropy_mean": entropy_mean,
            "ratio_clip_frac": _safe_mean(
                tracker["ratio_clip_sum"],
                ratio_denominator,
            ),
            "ratio_ceiling_frac": _safe_mean(
                tracker["ratio_ceiling_sum"],
                ratio_denominator,
            ),
        }

    async def _log_batch_metrics(
        self,
        monitor: Any,
        batch_metrics: dict[str, float] | None,
    ) -> None:
        """Log per-batch metrics to monitoring system.

        Args:
            monitor: Monitoring system instance
            batch_metrics: Aggregated metrics for the full (effective) batch
        """
        if monitor is None or not batch_metrics:
            return

        # Use global batch counter for smooth, continuous x-axis across all windows
        self.global_batch_counter += 1
        for key, value in batch_metrics.items():
            try:
                await monitor.log_gauge(
                    f"training/batch/{key}",
                    value,
                    tags={"batch_step": str(self.global_batch_counter)},  # Global counter
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to log batch metric %s: %s", key, exc)

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
        """Train for one epoch using GRPO algorithm.

        This method orchestrates the training loop by:
        1. Preparing batches from rollout groups
        2. Computing policy, reference, and behavior log probabilities
        3. Computing policy gradient loss with importance sampling and PPO clipping
        4. Computing KL divergence penalty and entropy regularization
        5. Performing gradient accumulation and optimization steps
        6. Tracking and logging training metrics

        Args:
            model: Current policy model to train
            ref_model: Reference policy model for KL divergence
            tokenizer: Tokenizer for batch preparation
            groups: List of GRPO groups containing rollouts
            optimizer: Optimizer for model updates
            accelerator: Accelerator for device management and distributed training
            monitor: Optional monitoring system for logging metrics
            window: Current training window number
            config: Training configuration

        Returns:
            Dictionary of averaged epoch metrics
        """
        # Increment global epoch counter for continuous tracking
        self.global_epoch_counter += 1

        micro_batch_size = max(1, int(config.batch_size))
        grad_accum_steps = max(1, self.config.grad_accum_steps)

        model.train()
        # KL gating: if base coefficient is zero, disable KL entirely
        kl_enabled: bool = is_kl_enabled()
        if ref_model is not None:
            ref_model.eval()

        # Flatten and sort all rollouts by group ID and nonce for deterministic batching
        all_rollouts: list[tuple] = []
        for group in groups:
            for rollout in group.rollouts:
                all_rollouts.append((rollout, group.group_id))

        all_rollouts.sort(key=lambda item: (item[1], item[0].nonce))

        # Use micro batch size parameter (defaults to TRAINER_BATCH_SIZE if not provided)
        num_micro_batches = math.ceil(len(all_rollouts) / micro_batch_size)

        epoch_metrics: dict[str, list[float]] = defaultdict(list)
        grad_accum_counter = 0
        actual_batch_tracker = {
            "micro_batches": 0,
            "sequence_count": 0.0,
            "token_count": 0.0,
            "loss_total_sum": 0.0,
            "loss_pg_sum": 0.0,
            "loss_kl_sum": 0.0,
            "loss_entropy_sum": 0.0,
            "adv_sum": 0.0,
            "adv_count": 0.0,
            "reward_sum": 0.0,
            "reward_count": 0.0,
            "entropy_sum": 0.0,
            "entropy_count": 0.0,
            "kl_sum": 0.0,
            "ratio_clip_sum": 0.0,
            "ratio_ceiling_sum": 0.0,
        }

        def reset_actual_batch_tracker() -> None:
            actual_batch_tracker.update(
                {
                    "micro_batches": 0,
                    "sequence_count": 0.0,
                    "token_count": 0.0,
                    "loss_total_sum": 0.0,
                    "loss_pg_sum": 0.0,
                    "loss_kl_sum": 0.0,
                    "loss_entropy_sum": 0.0,
                    "adv_sum": 0.0,
                    "adv_count": 0.0,
                    "reward_sum": 0.0,
                    "reward_count": 0.0,
                    "entropy_sum": 0.0,
                    "entropy_count": 0.0,
                    "kl_sum": 0.0,
                    "ratio_clip_sum": 0.0,
                    "ratio_ceiling_sum": 0.0,
                }
            )

        # Adaptive KL coefficient: prefer persistent controller when KL enabled
        if kl_enabled and self.adaptive_kl_enabled:
            current_kl_coef = self.kl_controller.value
        else:
            current_kl_coef = float(self.config.kl_coef)

        reset_actual_batch_tracker()

        for micro_idx in range(num_micro_batches):
            start_idx = micro_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, len(all_rollouts))
            if start_idx >= len(all_rollouts):
                break
            batch_rollouts = [all_rollouts[i][0] for i in range(start_idx, end_idx)]

            # Step 1: Prepare batch tensors from rollouts
            (
                input_ids_tensor,
                attention_mask_tensor,
                batch_prompt_lens,
                batch_comp_lens,
                batch_advantages,
                batch_behavior_per_token_logprobs,
                batch_rewards,
            ) = self._prepare_batch_tensors(batch_rollouts, tokenizer, accelerator)

            advantages_tensor = torch.tensor(
                batch_advantages,
                dtype=torch.float32,
                device=accelerator.device,
            )

            # Step 2: Compute current policy logprobs (with per-token for proper KL)
            with accelerator.autocast():
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

            # Step 3: Prepare behavior policy logprobs and completion mask for importance sampling
            # Convert per-token behavior logprobs to padded tensor matching completion shape
            max_comp_len = logprobs_current_per_token.shape[1]
            logprobs_old_per_token = torch.zeros(
                len(batch_rollouts), max_comp_len, device=accelerator.device
            )
            completion_mask = torch.zeros(
                len(batch_rollouts), max_comp_len, device=accelerator.device
            )

            for idx, (logprobs_list, comp_len) in enumerate(
                zip(batch_behavior_per_token_logprobs, batch_comp_lens, strict=True)
            ):
                actual_len = min(len(logprobs_list), max_comp_len)
                if actual_len > 0:
                    logprobs_old_per_token[idx, :actual_len] = torch.tensor(
                        logprobs_list[:actual_len], device=accelerator.device
                    )
                    completion_mask[idx, :comp_len] = 1.0

            # Validate behavior logprobs
            logprobs_old_sum = (logprobs_old_per_token * completion_mask).sum(dim=1)
            if not _is_finite_tensor(logprobs_old_sum):
                old_min = torch.nan_to_num(logprobs_old_sum).min().item()
                old_max = torch.nan_to_num(logprobs_old_sum).max().item()
                logger.warning(
                    "Non-finite old/behavior logprobs; skipping batch",
                    extra={"min": float(old_min), "max": float(old_max)},
                )
                continue

            # Step 4: Compute reference model logprobs for KL divergence penalty (only if enabled)
            logprobs_ref_per_token = None
            if kl_enabled:
                if ref_model is not None:
                    with torch.no_grad(), accelerator.autocast():
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
                        # Free tensors before continuing
                        del logprobs_ref_sum, logprobs_ref_per_token
                        continue
                    # Delete reference sum to free memory (only need per-token)
                    del logprobs_ref_sum
                else:
                    # If no ref model, set ref per-token logprobs equal to current (zero KL)
                    logprobs_ref_per_token = logprobs_current_per_token.detach()

            # Step 5: Normalize advantages for stable gradients
            advantages_normalized = self._normalize_advantages(advantages_tensor)

            # Step 6: Compute total completion tokens for DAPO variant
            total_completion_tokens = None
            if self.grpo_variant == "dapo":
                local_completion_tokens = completion_mask.sum()  # Keep as tensor
                # Gather across all processes with proper shape [1] for each process
                all_tokens = accelerator.gather(local_completion_tokens.unsqueeze(0))
                total_completion_tokens = int(all_tokens.sum().item())

            # Step 7: Compute policy gradient loss with importance sampling and PPO clipping
            loss_pg, ratio_clip_frac_val, ratio_ceiling_frac_val, ratios_pre_ceiling = (
                self._compute_policy_gradient_loss(
                    logprobs_current_per_token,
                    logprobs_old_per_token,
                    advantages_normalized,
                    completion_mask,
                    len(batch_rollouts),
                    total_completion_tokens,
                )
            )

            # Step 8: Compute KL divergence penalty (only when enabled)
            if kl_enabled:
                loss_kl, kl_value = self._compute_kl_divergence_loss(
                    logprobs_current_per_token,
                    logprobs_ref_per_token,
                    batch_comp_lens,
                    current_kl_coef,
                    accelerator,
                )
            else:
                loss_kl = torch.tensor(0.0, device=logprobs_current_sum.device)
                kl_value = 0.0

            # Step 9: Compute entropy regularization
            # Only compute entropy if coefficient is non-zero (saves memory)
            if self.config.entropy_coef > 0.0:
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
                loss_entropy = -self.config.entropy_coef * entropies.mean()
            else:
                # Skip entropy computation entirely when coefficient is zero
                entropies = torch.zeros(len(batch_rollouts), device=accelerator.device)
                loss_entropy = torch.tensor(0.0, device=accelerator.device)

            # Step 10: Aggregate total loss
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

            micro_sequence_count = len(batch_rollouts)
            micro_token_count = float(completion_mask.sum().item())
            reward_sum = float(sum(batch_rewards)) if batch_rewards else 0.0

            actual_batch_tracker["micro_batches"] += 1
            actual_batch_tracker["sequence_count"] += float(micro_sequence_count)
            actual_batch_tracker["token_count"] += micro_token_count
            actual_batch_tracker["loss_total_sum"] += float(loss_total.item())
            actual_batch_tracker["loss_pg_sum"] += float(loss_pg.item())
            actual_batch_tracker["loss_kl_sum"] += float(loss_kl.item())
            actual_batch_tracker["loss_entropy_sum"] += float(loss_entropy.item())
            actual_batch_tracker["adv_sum"] += float(advantages_tensor.sum().item())
            actual_batch_tracker["adv_count"] += float(advantages_tensor.numel())
            actual_batch_tracker["reward_sum"] += reward_sum
            actual_batch_tracker["reward_count"] += float(len(batch_rewards))
            actual_batch_tracker["entropy_sum"] += float(entropies.sum().item())
            actual_batch_tracker["entropy_count"] += float(entropies.numel())
            actual_batch_tracker["kl_sum"] += float(kl_value)
            ratio_weight = (
                micro_token_count
                if self.importance_sampling_level == "token"
                else float(micro_sequence_count)
            )
            actual_batch_tracker["ratio_clip_sum"] += ratio_clip_frac_val * ratio_weight
            actual_batch_tracker["ratio_ceiling_sum"] += ratio_ceiling_frac_val * ratio_weight

            # Step 11: Perform gradient accumulation and optimization
            # Zero gradients before backward pass (only at start of accumulation)
            if grad_accum_counter == 0:
                optimizer.zero_grad(set_to_none=True)

            # Scale loss by accumulation steps to keep effective LR stable
            scaled_loss = loss_total / float(grad_accum_steps)
            accelerator.backward(scaled_loss)
            grad_accum_counter += 1

            grad_norm_scalar = None
            # Only step optimizer and clip gradients every N accumulation steps
            if grad_accum_counter >= grad_accum_steps:
                # Clip gradients in fp32 (no mixed precision)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.grad_clip,
                )
                grad_norm_scalar = grad_norm.item()

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    # Build diagnostic info for NaN/Inf gradient detection
                    logprobs_current_sum = (logprobs_current_per_token * completion_mask).sum(dim=1)
                    logprobs_old_sum = (logprobs_old_per_token * completion_mask).sum(dim=1)
                    log_ratio = logprobs_current_sum - logprobs_old_sum
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
                    reset_actual_batch_tracker()
                    continue

                # Standard optimizer step in fp32
                optimizer.step()
                self.optimizer_step_count += 1

                # Parameter change tracking: measure diff between step t and step t-k
                # Snapshot persists across k steps, then we measure and update snapshot
                if (
                    self.param_tracker.is_enabled()
                    and self.optimizer_step_count % self.param_tracker.measure_interval == 0
                ):
                    # If we have a snapshot from k steps ago, compute metrics
                    if self.param_tracker.has_snapshot():
                        # Basic param change metrics
                        try:
                            param_metrics = self.param_tracker.compute_metrics(model)
                            await log_param_change_metrics(
                                param_metrics, monitor, self.optimizer_step_count
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to compute param change metrics: %s", exc)

                        # Sparse quality analysis (uses same snapshot)
                        if self.sparse_analyzer.enabled:
                            try:
                                sparse_metrics = self.sparse_analyzer.analyze(
                                    model, input_ids_tensor, attention_mask_tensor
                                )
                                await log_sparse_quality_metrics(
                                    sparse_metrics, monitor, self.optimizer_step_count
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("Failed to compute sparse quality metrics: %s", exc)

                    # Capture new snapshot for next k-step measurement
                    self.param_tracker.clear_snapshot()
                    self.param_tracker.capture_snapshot(model)

                logger.info(f"Optimizer step completed. Current KL coef: {current_kl_coef}")

                # Adaptive KL adjustment after each optimizer step based on observed KL
                if kl_enabled and self.adaptive_kl_enabled:
                    current_kl_coef = self.kl_controller.update(kl_value)
                grad_accum_counter = 0

                batch_metrics = self._finalize_actual_batch_metrics(
                    actual_batch_tracker,
                    grad_norm_scalar,
                )
                await self._log_batch_metrics(monitor, batch_metrics)
                reset_actual_batch_tracker()

            # Step 12: Collect batch metrics for epoch aggregation
            self._collect_batch_metrics(
                epoch_metrics,
                loss_total,
                loss_pg,
                loss_kl,
                loss_entropy,
                grad_norm_scalar,
                advantages_tensor,
                advantages_normalized,
                entropies,
                kl_value,
                ratios_pre_ceiling,
                ratio_clip_frac_val,
                ratio_ceiling_frac_val,
                batch_rewards,
            )

        # Handle remaining accumulated gradients at epoch end
        if grad_accum_counter > 0:
            final_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.grad_clip,
            )
            if not (torch.isnan(final_grad_norm) or torch.isinf(final_grad_norm)):
                optimizer.step()
                self.optimizer_step_count += 1

                # Parameter change tracking at epoch end (same logic as main loop)
                if (
                    self.param_tracker.is_enabled()
                    and self.optimizer_step_count % self.param_tracker.measure_interval == 0
                ):
                    if self.param_tracker.has_snapshot():
                        try:
                            param_metrics = self.param_tracker.compute_metrics(model)
                            await log_param_change_metrics(
                                param_metrics, monitor, self.optimizer_step_count
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to compute param change metrics: %s", exc)

                        # Sparse quality analysis (uses same snapshot)
                        if self.sparse_analyzer.enabled:
                            try:
                                sparse_metrics = self.sparse_analyzer.analyze(
                                    model, input_ids_tensor, attention_mask_tensor
                                )
                                await log_sparse_quality_metrics(
                                    sparse_metrics, monitor, self.optimizer_step_count
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning("Failed to compute sparse quality metrics: %s", exc)

                    self.param_tracker.clear_snapshot()
                    self.param_tracker.capture_snapshot(model)

                batch_metrics = self._finalize_actual_batch_metrics(
                    actual_batch_tracker,
                    final_grad_norm.item(),
                )
                await self._log_batch_metrics(monitor, batch_metrics)
            reset_actual_batch_tracker()
            grad_accum_counter = 0

        return {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in epoch_metrics.items()
        }
