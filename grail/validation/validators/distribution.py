"""Token distribution validation for detecting suspicious patterns."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import FloatTensor, LongTensor
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
)

from ...protocol.constants import (
    SAMPLING_HIGH_P,
    SAMPLING_INITIAL_WINDOW_STEPS,
    SAMPLING_LOW_P,
    SAMPLING_LOW_Q10_MAX,
    SAMPLING_MEDIAN_LOW_MAX,
    SAMPLING_MIN_STEPS,
    SAMPLING_MIN_TOKEN_PROB,
)
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class DistributionValidator(Validator):
    """Detects suspicious token probability distributions.

    Validates that chosen-token probabilities look consistent with sampling
    from the expected base model. Detects exploits like:
    - Using a different model for generation
    - Prefill tricks with low-probability prefixes
    - Bimodal distributions indicating model switching

    This is a SOFT check - failures accumulate and trigger rejection
    only if threshold exceeded (>26% of sampled rollouts).
    """

    check_name = "token_distribution_valid"
    severity = "soft"  # Soft check - doesn't cause immediate rejection
    soft_threshold = 0.51

    def validate(self, ctx: ValidationContext) -> bool:
        """Check token distribution for suspicious patterns."""
        probs = self._collect_probs(ctx)

        # Insufficient data
        if probs is None or len(probs) < SAMPLING_MIN_STEPS:
            ctx.checks[self.check_name] = False
            ctx.metadata["distribution_reason"] = "insufficient"
            return False

        # Compute distribution metrics
        metrics = self._compute_metrics(probs)

        # Detect suspicious patterns and collect reasons/initial-window metrics
        suspicious, reasons, initial_metrics = self._is_suspicious(metrics, probs)

        # Record results
        ctx.checks[self.check_name] = not suspicious
        ctx.metadata["distribution_metrics"] = metrics
        if initial_metrics is not None:
            ctx.metadata["distribution_initial_metrics"] = initial_metrics
        ctx.metadata["distribution_reasons"] = reasons

        # Note: W&B logging is done at miner level (aggregated across rollouts)
        # See MinerValidator._log_aggregated_distribution_metrics()

        return not suspicious

    def _collect_probs(self, ctx: ValidationContext) -> list[float] | None:
        """Collect chosen token probabilities from cached logits.

        Applies the drift-safe subset of HF's sample-mode pipeline
        (repetition_penalty, temperature) before reading the chosen-token
        probability. ``top_k`` and ``top_p`` are intentionally NOT applied
        here; see ``_build_hf_processors`` for the rationale.
        """
        try:
            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})

            prompt_len = int(rollout.get("prompt_length", 0) or 0)
            completion_len = int(rollout.get("completion_length", 0) or 0)

            if completion_len <= 0:
                return None

            # Use cached logits from proof validator (avoids redundant forward pass)
            if ctx.cached_logits is None:
                logger.debug("No cached logits available")
                return None

            logits = ctx.cached_logits
            device = ctx.model.device

            # Build HF logits processors from trusted generation_params.
            # generation_params is enforced upstream by
            # CheckpointMetadata.validate_metadata(); the required keys are
            # guaranteed to be present here.
            processors = self._build_hf_processors(ctx.generation_params)

            # Materialise the full token sequence on the model device once so
            # processors can slice prefix views without per-step host->device
            # transfers.
            full_input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

            probs: list[float] = []
            for t in range(prompt_len, min(prompt_len + completion_len, len(tokens))):
                if t == 0 or t - 1 >= logits.size(0):
                    continue

                # Logits the model produced when predicting token tokens[t].
                # HF processors mutate scores in place; pass a clone so the
                # cached_logits buffer is never touched. An empty processor
                # list is a no-op identity, so we always call through.
                step_logits = processors(
                    cast(LongTensor, full_input_ids[:, :t]),
                    cast(
                        FloatTensor,
                        logits[t - 1].to(device=device, dtype=torch.float32).unsqueeze(0).clone(),
                    ),
                )

                step_probs = torch.softmax(step_logits, dim=-1).squeeze(0)
                probs.append(float(step_probs[tokens[t]].item()))

            return probs if probs else None

        except Exception as e:
            logger.debug(f"Failed to collect token probs: {e}")
            return None

    @staticmethod
    def _build_hf_processors(generation_params: dict[str, Any]) -> LogitsProcessorList:
        """Construct the drift-safe subset of HF's sample-mode pipeline.

        Applies (in HF order):

        1. RepetitionPenaltyLogitsProcessor (when rep_penalty != 1.0)
        2. TemperatureLogitsWarper           (when temperature != 1.0)

        ``top_k`` and ``top_p`` are deliberately NOT applied. They are hard
        masks that set logits to ``-inf``, and the cutoff boundary is sensitive
        to bf16 prefill-vs-decode numerical drift between the miner's
        incremental decode and the validator's full prefill. In practice
        ~30-75% of legitimately-sampled tokens flip from "kept" to "masked"
        across the two runs and end up with ``softmax`` prob exactly ``0.0``,
        which trips the ``min_low`` rule on honest miners. ``temperature`` and
        ``repetition_penalty`` are smooth, monotonic, drift-tolerant
        operations and do not have this failure mode.

        TODO(distribution-validator): the consequence of dropping top_k/top_p
        is that the original false-positive case this validator was being
        rewritten to fix (high-T sampling makes raw chosen probs cluster
        below ``SAMPLING_MEDIAN_LOW_MAX``) is only partially addressed. The
        proper fix is per-position ratio normalization
        ``r_t = P_raw(chosen) / E_p_raw[t]`` (expected raw prob under the
        sampling distribution), which is drift-tolerant because it's a
        weighted sum, not a hard mask. Tracked in the original plan at
        ``~/.claude/plans/federated-sparking-flute.md``.

        ``generation_params`` is enforced upstream by
        ``CheckpointMetadata.validate_metadata()``; missing keys raise
        ``KeyError`` here so regressions are loud.
        """
        processors: list[Any] = []

        rep = float(generation_params["repetition_penalty"])
        if rep != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(rep))

        temperature = float(generation_params["temperature"])
        if temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))

        return LogitsProcessorList(processors)

    def _compute_metrics(self, probs: list[float]) -> dict:
        """Compute distribution statistics (quantiles and fractions)."""
        try:
            import numpy as np

            x = np.array(probs, dtype=np.float64)
            mean = float(x.mean())
            median = float(np.median(x))
            q10 = float(np.quantile(x, 0.10))
            min_p = float(x.min())

            # Fractions at extremes
            low_frac = float((x <= SAMPLING_LOW_P).mean())
            high_frac = float((x >= SAMPLING_HIGH_P).mean())
            mid_frac = max(0.0, 1.0 - low_frac - high_frac)

            # Bimodality coefficient
            d = x - mean
            variance = float((d * d).mean())

            if variance > 1e-12:
                m3 = float((d**3).mean())
                m4 = float((d**4).mean())
                skew = m3 / (variance**1.5 + 1e-12)
                kurt = m4 / (variance**2 + 1e-12)
                bc = (skew * skew + 1.0) / max(kurt, 1e-6)
            else:
                skew, kurt, bc = 0.0, 3.0, 0.0

            return {
                "mean": mean,
                "median": median,
                "q10": q10,
                "min": min_p,
                "low_frac": low_frac,
                "high_frac": high_frac,
                "mid_frac": mid_frac,
                "bc": bc,
            }

        except ImportError:
            # Fallback: torch-only implementation (simpler metrics)
            logger.debug("NumPy not available; using simplified metrics")
            t = torch.tensor(probs, dtype=torch.float64)
            return {
                "mean": float(t.mean().item()),
                "median": float(t.median().item()),
                "min": float(t.min().item()),
                "q10": 0.0,
                "low_frac": 0.0,
                "high_frac": 0.0,
                "mid_frac": 1.0,
                "bc": 0.0,
            }

    def _is_suspicious(self, metrics: dict, probs: list[float]) -> tuple[bool, dict, dict | None]:
        """Apply heuristics to detect exploits and return reasons.

        Returns:
            (suspicious, reasons_dict, initial_window_metrics_or_None)
        """
        reasons: dict[str, bool] = {
            "median_low": False,
            "min_low": False,
            "q10_low": False,
            "q10_low_initial": False,
        }

        # Unimodal low (all probs clustered low)
        if metrics["median"] <= SAMPLING_MEDIAN_LOW_MAX:
            logger.debug(f"Suspicious: median {metrics['median']} too low")
            reasons["median_low"] = True

        # Extremely low probability token (exploit)
        if metrics["min"] <= SAMPLING_MIN_TOKEN_PROB:
            logger.debug(f"Suspicious: min prob {metrics['min']} too low")
            reasons["min_low"] = True

        # Low Q10 distribution (full sequence)
        if metrics.get("q10", 0.0) <= SAMPLING_LOW_Q10_MAX:
            logger.debug(f"Suspicious: low q10 {metrics.get('q10')}")
            reasons["q10_low"] = True

        # Check initial window for prefix exploits
        initial_metrics: dict | None = None
        k = min(SAMPLING_INITIAL_WINDOW_STEPS, len(probs))
        if k > 0:
            initial_metrics = self._compute_metrics(probs[:k])
            if initial_metrics.get("q10", 0.0) <= SAMPLING_LOW_Q10_MAX:
                logger.debug("Suspicious: low q10 in initial window")
                reasons["q10_low_initial"] = True

        suspicious = any(reasons.values())
        return suspicious, reasons, initial_metrics
