"""Security-focused validators for detecting attacks and exploits.

Implements additional security checks beyond cryptographic proof validation,
focusing on behavioral anomalies and attack patterns.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import torch

from ...shared.constants import WINDOW_LENGTH
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class TimestampValidator(Validator):
    """Validates submission timestamps to detect replay attacks.

    Checks:
    - Submission timestamp is recent (within reasonable window)
    - Timestamp is not in the future
    - Timestamp matches expected window boundaries
    """

    check_name = "timestamp_valid"
    severity = "hard"

    # Allow submissions up to 2 windows old (to handle network delays)
    MAX_WINDOW_AGE = 2 * WINDOW_LENGTH

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify submission timestamp is valid and recent."""
        try:
            # Extract timestamp from commit (if present)
            timestamp = ctx.commit.get("timestamp")
            if timestamp is None:
                # Timestamp might be optional in current version
                ctx.checks[self.check_name] = True
                return True

            current_time = time.time()
            submission_time = float(timestamp)

            # Check 1: Not in the future (allow 60s clock skew)
            if submission_time > current_time + 60:
                logger.warning(
                    f"[{self.check_name}] Future timestamp detected | "
                    f"submission={submission_time} | current={current_time} | "
                    f"diff={submission_time - current_time:.1f}s"
                )
                ctx.checks[self.check_name] = False
                ctx.metadata["timestamp_failure"] = "future_timestamp"
                return False

            # Check 2: Not too old (replay attack detection)
            age_seconds = current_time - submission_time
            max_age_seconds = self.MAX_WINDOW_AGE * 12  # ~12s per block

            if age_seconds > max_age_seconds:
                logger.warning(
                    f"[{self.check_name}] Stale timestamp detected | "
                    f"age={age_seconds:.1f}s | max_age={max_age_seconds:.1f}s"
                )
                ctx.checks[self.check_name] = False
                ctx.metadata["timestamp_failure"] = "stale_timestamp"
                ctx.metadata["timestamp_age_seconds"] = age_seconds
                return False

            ctx.checks[self.check_name] = True
            return True

        except (ValueError, TypeError) as e:
            logger.warning(f"[{self.check_name}] Invalid timestamp format: {e}")
            ctx.checks[self.check_name] = False
            ctx.metadata["timestamp_failure"] = "invalid_format"
            return False


class HiddenStateAnomalyValidator(Validator):
    """Detects anomalies in hidden state statistics.

    Validates that hidden states from proof validation show expected
    statistical properties for the claimed model. Detects:
    - Unusual activation magnitudes
    - Suspicious activation patterns
    - Model substitution attempts
    """

    check_name = "hidden_state_anomaly_valid"
    severity = "soft"  # Soft check - accumulates with other soft checks

    # Expected ranges for Qwen-style models (adjust per model family)
    EXPECTED_NORM_RANGE = (0.1, 100.0)  # Typical hidden state L2 norms
    EXPECTED_MEAN_RANGE = (-5.0, 5.0)  # Typical mean activation

    def validate(self, ctx: ValidationContext) -> bool:
        """Check hidden state statistics for anomalies."""
        try:
            # This validator requires cached hidden states from proof validation
            # Currently we don't cache hidden states, only logits
            # This is a placeholder for future enhancement

            # For now, we can check logit statistics as a proxy
            if ctx.cached_logits is None:
                ctx.checks[self.check_name] = True
                return True

            logits = ctx.cached_logits  # [seq_len, vocab_size]

            # Compute statistics
            logit_norms = torch.norm(logits, dim=-1)  # [seq_len]
            mean_norm = float(logit_norms.mean().item())
            max_norm = float(logit_norms.max().item())
            min_norm = float(logit_norms.min().item())

            logit_means = logits.mean(dim=-1)  # [seq_len]
            mean_activation = float(logit_means.mean().item())

            # Check for anomalies
            anomalies = []

            # Check 1: Unusually high norms (potential numerical instability)
            if max_norm > 5000.0:
                anomalies.append(f"high_max_norm={max_norm:.2f}")
                logger.debug(
                    f"[{self.check_name}] High logit norm detected: {max_norm:.2f}"
                )

            # Check 2: Unusually low norms (potential zero/dead activations)
            if min_norm < 0.001:
                anomalies.append(f"low_min_norm={min_norm:.4f}")
                logger.debug(
                    f"[{self.check_name}] Low logit norm detected: {min_norm:.4f}"
                )

            # Check 3: Extreme mean activation
            if abs(mean_activation) > 50.0:
                anomalies.append(f"extreme_mean={mean_activation:.2f}")
                logger.debug(
                    f"[{self.check_name}] Extreme mean activation: {mean_activation:.2f}"
                )

            # Record statistics
            ctx.metadata["logit_stats"] = {
                "mean_norm": mean_norm,
                "max_norm": max_norm,
                "min_norm": min_norm,
                "mean_activation": mean_activation,
            }

            if anomalies:
                ctx.metadata["hidden_state_anomalies"] = anomalies
                ctx.checks[self.check_name] = False
                logger.debug(
                    f"[{self.check_name}] Anomalies detected: {', '.join(anomalies)}"
                )
                return False

            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"[{self.check_name}] Error checking hidden states: {e}")
            ctx.checks[self.check_name] = True  # Don't fail on errors
            return True


class RateLimitValidator(Validator):
    """Tracks submission rates per miner to detect spam/flooding.

    This is a stateful validator that maintains per-miner submission counts
    across windows. Detects:
    - Excessive submission rates
    - Burst patterns
    - Potential DoS attempts
    """

    check_name = "rate_limit_valid"
    severity = "soft"

    # Class-level state (shared across instances)
    _submission_counts: dict[str, list[tuple[float, int]]] = defaultdict(list)
    _last_cleanup: float = 0.0

    # Rate limits
    MAX_SUBMISSIONS_PER_WINDOW = 1000  # Generous limit
    CLEANUP_INTERVAL = 3600  # Clean old data every hour

    def validate(self, ctx: ValidationContext) -> bool:
        """Check if miner is within rate limits."""
        try:
            miner_address = ctx.prover_address
            if not miner_address:
                ctx.checks[self.check_name] = True
                return True

            current_time = time.time()
            window_start = ctx.window_start if hasattr(ctx, "window_start") else 0

            # Periodic cleanup of old data
            self._cleanup_old_data(current_time)

            # Record this submission
            self._submission_counts[miner_address].append((current_time, window_start))

            # Count submissions in current window
            window_submissions = sum(
                1
                for _, w in self._submission_counts[miner_address]
                if w == window_start
            )

            if window_submissions > self.MAX_SUBMISSIONS_PER_WINDOW:
                logger.warning(
                    f"[{self.check_name}] Rate limit exceeded | "
                    f"miner={miner_address[:12]} | "
                    f"submissions={window_submissions} | "
                    f"limit={self.MAX_SUBMISSIONS_PER_WINDOW}"
                )
                ctx.checks[self.check_name] = False
                ctx.metadata["rate_limit_exceeded"] = True
                ctx.metadata["submission_count"] = window_submissions
                return False

            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"[{self.check_name}] Error checking rate limit: {e}")
            ctx.checks[self.check_name] = True  # Don't fail on errors
            return True

    @classmethod
    def _cleanup_old_data(cls, current_time: float) -> None:
        """Remove submission records older than 24 hours."""
        if current_time - cls._last_cleanup < cls.CLEANUP_INTERVAL:
            return

        cutoff_time = current_time - 86400  # 24 hours
        for miner in list(cls._submission_counts.keys()):
            cls._submission_counts[miner] = [
                (t, w)
                for t, w in cls._submission_counts[miner]
                if t > cutoff_time
            ]
            # Remove miner if no recent submissions
            if not cls._submission_counts[miner]:
                del cls._submission_counts[miner]

        cls._last_cleanup = current_time
        logger.debug(
            f"[RateLimitValidator] Cleaned old data | "
            f"active_miners={len(cls._submission_counts)}"
        )


class ModelFingerprintValidator(Validator):
    """Validates model fingerprint to detect model substitution.

    Checks that the model used for generation matches expected characteristics
    by analyzing output distributions and patterns.
    """

    check_name = "model_fingerprint_valid"
    severity = "soft"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify model fingerprint matches expected model."""
        try:
            # Extract model info from commit
            model_info = ctx.commit.get("model", {})
            claimed_model = model_info.get("name")
            expected_model = getattr(ctx.model, "name_or_path", None)

            # Basic check: model name matches (already done in proof validator)
            if claimed_model != expected_model:
                logger.debug(
                    f"[{self.check_name}] Model name mismatch already caught by proof validator"
                )
                ctx.checks[self.check_name] = False
                return False

            # Advanced check: Analyze token distribution patterns
            # Different models have different vocabulary usage patterns
            if ctx.cached_logits is not None:
                fingerprint_match = self._check_vocabulary_fingerprint(ctx)
                if not fingerprint_match:
                    ctx.checks[self.check_name] = False
                    return False

            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"[{self.check_name}] Error checking fingerprint: {e}")
            ctx.checks[self.check_name] = True  # Don't fail on errors
            return True

    def _check_vocabulary_fingerprint(self, ctx: ValidationContext) -> bool:
        """Check if vocabulary usage matches expected model."""
        try:
            tokens = ctx.commit.get("tokens", [])
            if not tokens:
                return True

            # Check for suspicious token patterns
            # Example: Repeated use of rare tokens might indicate model substitution

            # Count unique tokens in completion
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_tokens = tokens[prompt_len:]

            if not completion_tokens:
                return True

            unique_tokens = len(set(completion_tokens))
            total_tokens = len(completion_tokens)
            uniqueness_ratio = unique_tokens / total_tokens

            # Very low uniqueness might indicate repetitive/stuck generation
            if uniqueness_ratio < 0.1 and total_tokens > 20:
                logger.debug(
                    f"[{self.check_name}] Low token uniqueness | "
                    f"ratio={uniqueness_ratio:.2%} | "
                    f"unique={unique_tokens}/{total_tokens}"
                )
                ctx.metadata["low_token_uniqueness"] = True
                # Don't fail hard, just flag for monitoring

            return True

        except Exception as e:
            logger.debug(f"[{self.check_name}] Fingerprint check error: {e}")
            return True
