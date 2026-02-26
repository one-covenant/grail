"""Adam sign descent metrics computation.

Analyzes whether Adam behaves as sign descent during GRPO training.
Based on Balles & Hennig (ICML 2018) showing Adam ~ sign descent + variance-adaptive
magnitude, and Bernstein et al. (ICML 2018, signSGD) confirming signSGD with momentum
matches Adam.

Computes the Adam normalization ratio m_hat / (sqrt(v_hat) + eps) — the core quantity
that should be approximately +/-1 if sign descent holds. Combined with gradient SNR
analysis for characterizing RL gradient noise in GRPO.

Performance: CPU-per-layer computation with exact scalar statistics over all parameters
and reservoir-sampled histograms (1M samples for publication-quality distribution plots).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot

logger = logging.getLogger(__name__)

# Special prefix for histogram data (detected by logging callbacks via "/_histogram/" pattern)
HISTOGRAM_KEY_PREFIX = "adam_sign/_histogram/"

# Component name patterns for transformer architectures
_COMPONENT_PATTERNS = {
    "q_proj": ["q_proj"],
    "k_proj": ["k_proj"],
    "v_proj": ["v_proj"],
    "o_proj": ["o_proj"],
    "gate_proj": ["gate_proj"],
    "up_proj": ["up_proj"],
    "down_proj": ["down_proj"],
    "embed_tokens": ["embed_tokens"],
    "lm_head": ["lm_head"],
    "layernorm": ["layernorm", "layer_norm", "norm"],
}


def _extract_component(name: str) -> str | None:
    """Extract component type from parameter name."""
    name_lower = name.lower()
    for comp, patterns in _COMPONENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in name_lower:
                return comp
    return None


def _extract_layer_idx(name: str) -> int | None:
    """Extract layer index from parameter name (e.g., 'model.layers.5.self_attn...')."""
    import re

    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None


class _ComponentAccumulator:
    """Accumulates exact statistics for a single component type."""

    __slots__ = (
        "count",
        "sum_abs_ratio",
        "band_10pct",
        "sign_agree_count",
        "sign_agree_nonzero",
        "sum_snr",
    )

    def __init__(self) -> None:
        self.count = 0
        self.sum_abs_ratio = 0.0
        self.band_10pct = 0
        self.sign_agree_count = 0
        self.sign_agree_nonzero = 0
        self.sum_snr = 0.0

    def accumulate(
        self,
        abs_ratio: torch.Tensor,
        nz_mask: torch.Tensor,
        grad: torch.Tensor,
        m_hat: torch.Tensor,
        snr: torch.Tensor,
    ) -> None:
        n = abs_ratio.numel()
        self.count += n
        self.sum_abs_ratio += abs_ratio.sum().item()
        self.band_10pct += int(((abs_ratio >= 0.9) & (abs_ratio <= 1.1)).sum().item())
        nz_count = int(nz_mask.sum().item())
        self.sign_agree_nonzero += nz_count
        self.sign_agree_count += int(((grad.sign() == m_hat.sign()) & nz_mask).sum().item())
        self.sum_snr += snr.sum().item()


class _LayerAccumulator:
    """Accumulates exact statistics for a single layer."""

    __slots__ = ("count", "sum_abs_ratio", "band_10pct")

    def __init__(self) -> None:
        self.count = 0
        self.sum_abs_ratio = 0.0
        self.band_10pct = 0

    def accumulate(self, abs_ratio: torch.Tensor) -> None:
        n = abs_ratio.numel()
        self.count += n
        self.sum_abs_ratio += abs_ratio.sum().item()
        self.band_10pct += int(((abs_ratio >= 0.9) & (abs_ratio <= 1.1)).sum().item())


class AdamSignDescentMetrics(MetricComputer):
    """Compute Adam sign descent metrics for GRPO training analysis.

    Analyzes the Adam normalization ratio m_hat / (sqrt(v_hat) + eps) which should
    be approximately +/-1 if Adam behaves as sign descent. Also computes gradient
    SNR metrics relevant to RL training with high-variance advantage-weighted gradients.

    Scalar statistics are EXACT (computed over all parameters via CPU reductions).
    Histograms use reservoir sampling (default 1M samples) for memory efficiency.

    Example:
        >>> computer = AdamSignDescentMetrics(
        ...     track_per_component=True,
        ...     histogram_samples=1_000_000,
        ... )
        >>> metrics = computer.compute(
        ...     context=AnalysisContext(model=model, optimizer=optimizer)
        ... )
        >>> print(metrics["adam_sign/frac_within_10pct_of_lr"])
    """

    def __init__(
        self,
        track_per_component: bool = True,
        track_per_layer: bool = False,
        histogram_samples: int = 1_000_000,
        near_lr_tolerance: float = 0.1,
    ) -> None:
        """Initialize Adam sign descent metric computer.

        Args:
            track_per_component: Track per-component statistics (q_proj, k_proj, etc.)
            track_per_layer: Track per-layer statistics (more verbose)
            histogram_samples: Max samples for histograms via reservoir sampling (0 to disable)
            near_lr_tolerance: Tolerance for "near learning rate" fraction (default 0.1 = 10%)
        """
        self.track_per_component = track_per_component
        self.track_per_layer = track_per_layer
        self.histogram_samples = histogram_samples
        self.near_lr_tolerance = near_lr_tolerance

    def requires_model(self) -> bool:
        """Requires model to access parameters and gradients."""
        return True

    def requires_optimizer(self) -> bool:
        """Requires optimizer to access Adam state (exp_avg, exp_avg_sq)."""
        return True

    def compute(
        self,
        delta: ParameterDelta | None = None,
        old_snapshot: ParameterSnapshot | None = None,
        current_snapshot: ParameterSnapshot | None = None,
        context: AnalysisContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute Adam sign descent metrics.

        Args:
            delta: Unused (for compatibility)
            old_snapshot: Unused (for compatibility)
            current_snapshot: Unused (for compatibility)
            context: Must contain model and optimizer with Adam state
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metrics with "adam_sign/" prefix.
            Histogram data stored under HISTOGRAM_KEY_PREFIX keys as numpy arrays.
        """
        if context is None or context.model is None or context.optimizer is None:
            return {}

        model = context.model
        optimizer = context.optimizer

        # Extract optimizer hyperparameters from first param group
        param_groups = optimizer.param_groups
        if not param_groups:
            return {}
        pg = param_groups[0]
        lr = pg.get("lr", 0.0)
        betas = pg.get("betas", (0.9, 0.999))
        beta1, beta2 = betas[0], betas[1]
        eps = pg.get("eps", 1e-8)

        # Running accumulators (CPU scalars — exact over all parameters)
        total = 0
        sum_abs_ratio = 0.0
        sum_sq_ratio = 0.0
        sum_ratio_for_moments = 0.0  # signed ratio sum for skewness/kurtosis
        sum_ratio_sq_for_moments = 0.0
        sum_ratio_cu_for_moments = 0.0
        sum_ratio_qu_for_moments = 0.0
        band_10pct = 0
        band_50pct = 0
        band_2x = 0
        sign_agree_grad = 0
        sign_agree_momentum = 0
        count_nonzero = 0
        sum_snr = 0.0
        sum_sq_snr = 0.0
        sum_effective_lr = 0.0

        # Per-component accumulators
        comp_stats: dict[str, _ComponentAccumulator] = defaultdict(_ComponentAccumulator)

        # Per-layer accumulators
        layer_stats: dict[int, _LayerAccumulator] = defaultdict(_LayerAccumulator)

        # Reservoir samplers (CPU lists)
        reservoir_update: list[float] = []  # |ratio|
        reservoir_signed: list[float] = []  # signed ratio
        reservoir_snr: list[float] = []  # SNR
        reservoir_log: list[float] = []  # log10(|ratio|)
        reservoir_idx = 0

        step_t = None
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            state = optimizer.state.get(param)
            if not state or "exp_avg" not in state:
                continue

            step_t = state.get("step")
            if step_t is None:
                continue
            # Handle tensor step counts
            if isinstance(step_t, torch.Tensor):
                step_t = step_t.item()
            step_t = int(step_t)
            if step_t == 0:
                continue

            # Move to CPU float32 one layer at a time (safe for GPU memory)
            grad = param.grad.detach().to(device="cpu", dtype=torch.float32).flatten()
            m_t = state["exp_avg"].detach().to(device="cpu", dtype=torch.float32).flatten()
            v_t = state["exp_avg_sq"].detach().to(device="cpu", dtype=torch.float32).flatten()
            n = grad.numel()
            if n == 0:
                del grad, m_t, v_t
                continue

            # Bias correction
            bc_m = 1.0 / (1.0 - beta1**step_t)
            bc_v = 1.0 / (1.0 - beta2**step_t)
            m_hat = m_t * bc_m
            v_hat = v_t * bc_v

            # Core: Adam normalization ratio
            ratio = m_hat / (v_hat.sqrt() + eps)
            abs_ratio = ratio.abs()

            # EXACT scalar accumulation
            total += n
            sum_abs_ratio += abs_ratio.sum().item()
            sum_sq_ratio += abs_ratio.pow(2).sum().item()

            # Signed ratio moments for bimodal coefficient
            sum_ratio_for_moments += ratio.sum().item()
            sum_ratio_sq_for_moments += ratio.pow(2).sum().item()
            ratio_cu = ratio.pow(3)
            sum_ratio_cu_for_moments += ratio_cu.sum().item()
            del ratio_cu
            ratio_qu = ratio.pow(4)
            sum_ratio_qu_for_moments += ratio_qu.sum().item()
            del ratio_qu

            # Band counts
            band_10pct += int(((abs_ratio >= 0.9) & (abs_ratio <= 1.1)).sum().item())
            band_50pct += int(((abs_ratio >= 0.5) & (abs_ratio <= 1.5)).sum().item())
            band_2x += int(((abs_ratio >= 0.5) & (abs_ratio <= 2.0)).sum().item())

            # Sign agreement (exact)
            nz = (grad != 0) & (m_hat != 0)
            nz_count = int(nz.sum().item())
            count_nonzero += nz_count
            sign_agree_grad += int(((grad.sign() == m_hat.sign()) & nz).sum().item())

            # Sign agreement with theoretical update direction
            # Adam update = -lr * m_hat / (sqrt(v_hat) + eps), so sign(update) = -sign(ratio)
            # Agreement: sign(-ratio) == -sign(m_t) ↔ sign(ratio) == sign(m_t)
            sign_agree_momentum += int(((ratio.sign() == m_t.sign()) & (m_t != 0)).sum().item())

            # SNR (exact): |m_hat| / sqrt(v_hat - m_hat^2 + eps)
            variance = (v_hat - m_hat.pow(2)).clamp(min=0)
            snr = m_hat.abs() / (variance.sqrt() + eps)
            sum_snr += snr.sum().item()
            sum_sq_snr += snr.pow(2).sum().item()

            # Effective learning rate
            sum_effective_lr += (abs_ratio * lr).sum().item()

            # Reservoir sampling (vectorized, same algorithm as GradientSparsityMetrics)
            if self.histogram_samples > 0:
                abs_np = abs_ratio.numpy()
                ratio_np = ratio.numpy()
                snr_np = snr.numpy()
                # Log of absolute ratio (clamp to avoid log(0))
                log_np = np.log10(np.clip(abs_np, 1e-38, None))

                _reservoir_sample_batch(
                    abs_np, reservoir_update, reservoir_idx, self.histogram_samples
                )
                _reservoir_sample_batch(
                    ratio_np, reservoir_signed, reservoir_idx, self.histogram_samples
                )
                _reservoir_sample_batch(
                    snr_np, reservoir_snr, reservoir_idx, self.histogram_samples
                )
                _reservoir_sample_batch(
                    log_np, reservoir_log, reservoir_idx, self.histogram_samples
                )
                reservoir_idx += n

            # Per-component (if enabled)
            if self.track_per_component:
                comp = _extract_component(name)
                if comp:
                    comp_stats[comp].accumulate(abs_ratio, nz, grad, m_hat, snr)

            # Per-layer (if enabled)
            if self.track_per_layer:
                layer_idx = _extract_layer_idx(name)
                if layer_idx is not None:
                    layer_stats[layer_idx].accumulate(abs_ratio)

            # Free CPU tensors before next layer
            del grad, m_t, v_t, m_hat, v_hat, ratio, abs_ratio, snr, variance, nz

        if total == 0 or step_t is None:
            return {}

        # Finalize scalar metrics
        metrics: dict[str, Any] = {}

        mean_abs_ratio = sum_abs_ratio / total
        mean_sq_ratio = sum_sq_ratio / total
        std_ratio = max(0.0, mean_sq_ratio - mean_abs_ratio**2) ** 0.5

        metrics["adam_sign/update_ratio_mean"] = mean_abs_ratio
        metrics["adam_sign/update_ratio_std"] = std_ratio
        metrics["adam_sign/frac_within_10pct_of_lr"] = band_10pct / total
        metrics["adam_sign/frac_within_50pct_of_lr"] = band_50pct / total
        metrics["adam_sign/frac_within_2x_of_lr"] = band_2x / total

        # Sign agreement
        if count_nonzero > 0:
            metrics["adam_sign/grad_sign_agreement"] = sign_agree_grad / count_nonzero
        else:
            metrics["adam_sign/grad_sign_agreement"] = 0.0
        metrics["adam_sign/sign_agreement_with_momentum"] = sign_agree_momentum / total

        # Effective learning rate
        metrics["adam_sign/effective_lr_mean"] = sum_effective_lr / total
        metrics["adam_sign/lr"] = lr

        # Bias correction values
        bc_m = 1.0 / (1.0 - beta1**step_t)
        bc_v = 1.0 / (1.0 - beta2**step_t)
        metrics["adam_sign/bias_correction_m"] = bc_m
        metrics["adam_sign/bias_correction_v"] = bc_v
        metrics["adam_sign/optimizer_step_count"] = float(step_t)

        # Gradient SNR
        mean_snr = sum_snr / total
        metrics["adam_sign/gradient_snr_mean"] = mean_snr

        # Median from reservoir samples
        if reservoir_update:
            metrics["adam_sign/update_ratio_median"] = float(np.median(reservoir_update))
        if reservoir_snr:
            metrics["adam_sign/gradient_snr_median"] = float(np.median(reservoir_snr))

        # Update magnitude entropy from reservoir
        if reservoir_update:
            arr = np.array(reservoir_update, dtype=np.float32)
            n_bins = 200
            hist_counts, _ = np.histogram(arr, bins=n_bins, range=(0.0, max(3.0, float(arr.max()))))
            # Normalize to probabilities
            probs = hist_counts / hist_counts.sum()
            probs = probs[probs > 0]
            entropy = -float(np.sum(probs * np.log2(probs)))
            metrics["adam_sign/update_magnitude_entropy"] = entropy

            # Effective bits per update
            metrics["adam_sign/effective_bits_per_update"] = entropy

        # Bimodal coefficient: (skewness^2 + 1) / kurtosis of signed ratio
        # BC > 5/9 suggests bimodality
        if total > 3:
            mean_signed = sum_ratio_for_moments / total
            var_signed = (sum_ratio_sq_for_moments / total) - mean_signed**2
            if var_signed > 1e-30:
                std_signed = var_signed**0.5
                # Standardized moments
                m3 = (
                    (sum_ratio_cu_for_moments / total)
                    - 3 * mean_signed * (sum_ratio_sq_for_moments / total)
                    + 2 * mean_signed**3
                )
                skewness = m3 / (std_signed**3)
                m4 = (
                    (sum_ratio_qu_for_moments / total)
                    - 4 * mean_signed * (sum_ratio_cu_for_moments / total)
                    + 6 * mean_signed**2 * (sum_ratio_sq_for_moments / total)
                    - 3 * mean_signed**4
                )
                kurtosis = m4 / (std_signed**4)
                if kurtosis > 1e-10:
                    metrics["adam_sign/bimodal_coefficient"] = (skewness**2 + 1) / kurtosis

        # Histograms (reservoir-sampled)
        if reservoir_update:
            metrics[f"{HISTOGRAM_KEY_PREFIX}update_ratio"] = np.array(
                reservoir_update, dtype=np.float32
            )
        if reservoir_signed:
            metrics[f"{HISTOGRAM_KEY_PREFIX}norm_ratio"] = np.array(
                reservoir_signed, dtype=np.float32
            )
        if reservoir_snr:
            metrics[f"{HISTOGRAM_KEY_PREFIX}gradient_snr"] = np.array(
                reservoir_snr, dtype=np.float32
            )
        if reservoir_log:
            metrics[f"{HISTOGRAM_KEY_PREFIX}log_update_ratio"] = np.array(
                reservoir_log, dtype=np.float32
            )

        # Per-component metrics
        if self.track_per_component:
            for comp_name, acc in comp_stats.items():
                if acc.count == 0:
                    continue
                prefix = f"adam_sign/component/{comp_name}"
                metrics[f"{prefix}/frac_within_10pct_of_lr"] = acc.band_10pct / acc.count
                metrics[f"{prefix}/norm_ratio_mean_abs"] = acc.sum_abs_ratio / acc.count
                if acc.sign_agree_nonzero > 0:
                    metrics[f"{prefix}/grad_sign_agreement"] = (
                        acc.sign_agree_count / acc.sign_agree_nonzero
                    )
                metrics[f"{prefix}/gradient_snr_mean"] = acc.sum_snr / acc.count

        # Per-layer metrics
        if self.track_per_layer:
            for layer_idx in sorted(layer_stats.keys()):
                acc = layer_stats[layer_idx]
                if acc.count == 0:
                    continue
                prefix = f"adam_sign/layer_{layer_idx}"
                metrics[f"{prefix}/norm_ratio_mean_abs"] = acc.sum_abs_ratio / acc.count
                metrics[f"{prefix}/frac_within_10pct_of_lr"] = acc.band_10pct / acc.count

        return metrics

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "AdamSignDescentMetrics"


def _reservoir_sample_batch(
    values_np: np.ndarray,
    reservoir: list[float],
    current_idx: int,
    max_samples: int,
) -> None:
    """Vectorized reservoir sampling into a list.

    Same algorithm as GradientSparsityMetrics (gradient_sparsity.py:130-157).

    Args:
        values_np: New values to potentially add (numpy array)
        reservoir: Existing reservoir list (modified in place)
        current_idx: Number of elements seen so far (before this batch)
        max_samples: Maximum reservoir size
    """
    n_new = len(values_np)
    if n_new == 0:
        return

    if current_idx < max_samples:
        # Still filling reservoir - take what we can
        take = min(n_new, max_samples - current_idx)
        reservoir.extend(values_np[:take].tolist())
        # Handle remaining elements with reservoir sampling
        remaining = values_np[take:]
        if len(remaining) > 0:
            start_idx = current_idx + take
            indices = np.arange(start_idx, start_idx + len(remaining))
            probs = max_samples / (indices + 1)
            mask = np.random.random(len(remaining)) < probs
            replace_indices = np.random.randint(0, max_samples, size=mask.sum())
            for idx, val in zip(replace_indices, remaining[mask], strict=False):
                reservoir[idx] = float(val)
    else:
        # Reservoir full - vectorized replacement
        indices = np.arange(current_idx, current_idx + n_new)
        probs = max_samples / (indices + 1)
        mask = np.random.random(n_new) < probs
        replace_indices = np.random.randint(0, max_samples, size=mask.sum())
        for idx, val in zip(replace_indices, values_np[mask], strict=False):
            reservoir[idx] = float(val)
