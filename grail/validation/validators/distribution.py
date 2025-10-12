"""Token distribution validation for detecting suspicious patterns."""

from __future__ import annotations

import logging

import torch

from ...shared.constants import (
    SAMPLING_BC_THRESHOLD,
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
        """Collect chosen token probabilities from cached logits."""
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

            # Collect probabilities for completion tokens
            probs = []
            for t in range(prompt_len, min(prompt_len + completion_len, len(tokens))):
                if t > 0 and t - 1 < logits.size(0):
                    step_probs = torch.softmax(logits[t - 1], dim=-1)
                    probs.append(float(step_probs[tokens[t]].item()))

            return probs if probs else None

        except Exception as e:
            logger.debug(f"Failed to collect token probs: {e}")
            return None

    def _compute_metrics(self, probs: list[float]) -> dict:
        """Compute distribution statistics (bimodality, quantiles)."""
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
            "bimodal_full": False,
            "bimodal_initial": False,
        }

        # Unimodal low (all probs clustered low)
        if metrics["median"] <= SAMPLING_MEDIAN_LOW_MAX:
            logger.debug(f"Suspicious: median {metrics['median']} too low")
            reasons["median_low"] = True

        # Extremely low probability token (exploit)
        if metrics["min"] <= SAMPLING_MIN_TOKEN_PROB:
            logger.debug(f"Suspicious: min prob {metrics['min']} too low")
            reasons["min_low"] = True

        # Bimodal distribution (full sequence)
        low_q10 = metrics.get("q10", 0.0) <= SAMPLING_LOW_Q10_MAX
        high_bc = metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD
        if low_q10 and high_bc:
            logger.debug(f"Suspicious: bimodal (q10={metrics.get('q10')}, bc={metrics.get('bc')})")
            reasons["bimodal_full"] = True

        # Check initial window for prefix exploits
        initial_metrics: dict | None = None
        k = min(SAMPLING_INITIAL_WINDOW_STEPS, len(probs))
        if k > 0:
            initial_metrics = self._compute_metrics(probs[:k])
            init_low_q10 = initial_metrics.get("q10", 0.0) <= SAMPLING_LOW_Q10_MAX
            init_high_bc = initial_metrics.get("bc", 0.0) >= SAMPLING_BC_THRESHOLD
            if init_low_q10 and init_high_bc:
                logger.debug("Suspicious: bimodal in initial window")
                reasons["bimodal_initial"] = True

        suspicious = any(reasons.values())
        return suspicious, reasons, initial_metrics
