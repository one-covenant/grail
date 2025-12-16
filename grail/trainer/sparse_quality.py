"""Sparse update quality analysis for training.

Measures how well sparse weight updates approximate full updates by comparing
model outputs (logits) between:
- Full update: W_current
- Sparse update: W_old + sparse_delta (where small deltas are zeroed)

This helps understand if LoRA/PEFT methods would work well and identifies
redundant gradient updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn

    from grail.trainer.param_tracker import ParamChangeTracker

logger = logging.getLogger(__name__)

# Thresholds for sparse quality analysis (realistic for fp32 precision)
SPARSE_QUALITY_THRESHOLDS: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4)


@dataclass
class ThresholdMetrics:
    """Metrics for a single sparsity threshold."""

    threshold: float
    kept_ratio: float  # Fraction of params kept (non-zero in sparse delta), 0.0-1.0

    # Magnitude-based sparse metrics
    kl_divergence: float
    cosine_similarity: float
    mse: float
    top1_agreement: float

    # Random baseline metrics (same sparsity level, random mask)
    kl_divergence_random: float
    cosine_similarity_random: float
    mse_random: float
    top1_agreement_random: float


@dataclass
class SparseQualityMetrics:
    """Aggregated sparse quality metrics across all thresholds."""

    threshold_metrics: list[ThresholdMetrics] = field(default_factory=list)

    def to_log_dict(self, prefix: str = "param_change/sparse") -> dict[str, float]:
        """Convert to flat dictionary for logging.

        Args:
            prefix: Prefix for all metric keys

        Returns:
            Flat dictionary suitable for W&B logging
        """
        result: dict[str, float] = {}

        for tm in self.threshold_metrics:
            thresh_str = f"{tm.threshold:.0e}"

            # Magnitude-based metrics
            result[f"{prefix}/kl_div_at_{thresh_str}"] = tm.kl_divergence
            result[f"{prefix}/cosine_at_{thresh_str}"] = tm.cosine_similarity
            result[f"{prefix}/mse_at_{thresh_str}"] = tm.mse
            result[f"{prefix}/top1_agree_at_{thresh_str}"] = tm.top1_agreement
            result[f"{prefix}/kept_ratio_at_{thresh_str}"] = tm.kept_ratio
            result[f"{prefix}/unchanged_ratio_at_{thresh_str}"] = 1.0 - tm.kept_ratio

            # Random baseline metrics
            result[f"{prefix}/kl_div_at_{thresh_str}_random"] = tm.kl_divergence_random
            result[f"{prefix}/cosine_at_{thresh_str}_random"] = tm.cosine_similarity_random
            result[f"{prefix}/mse_at_{thresh_str}_random"] = tm.mse_random
            result[f"{prefix}/top1_agree_at_{thresh_str}_random"] = tm.top1_agreement_random

        return result


def _compute_kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute KL divergence between two logit distributions.

    Args:
        logits_a: Reference logits [B, T, V]
        logits_b: Comparison logits [B, T, V]
        mask: Attention mask [B, T] (1 = valid, 0 = padding)

    Returns:
        Mean KL divergence over valid positions
    """
    # Use F.kl_div for numerical stability
    # F.kl_div expects log(input) and target, computes: target * (log(target) - log(input))
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)

    # KL(P_a || P_b): how much P_a diverges from P_b
    # F.kl_div(log_b, log_a.exp(), reduction='none', log_target=False) = P_a * (log_a - log_b)
    kl_per_token = F.kl_div(log_probs_b, log_probs_a, reduction="none", log_target=True)
    kl_per_position = kl_per_token.sum(dim=-1)  # [B, T]

    # Average over valid positions
    valid_count = mask.sum().clamp(min=1.0)
    kl_mean = (kl_per_position * mask).sum() / valid_count

    return float(kl_mean.item())


def _compute_cosine_similarity(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute mean cosine similarity between logit vectors.

    Args:
        logits_a: Reference logits [B, T, V]
        logits_b: Comparison logits [B, T, V]
        mask: Attention mask [B, T]

    Returns:
        Mean cosine similarity over valid positions
    """
    # Normalize along vocab dimension (eps prevents NaN for zero vectors)
    a_norm = F.normalize(logits_a, p=2, dim=-1, eps=1e-8)
    b_norm = F.normalize(logits_b, p=2, dim=-1, eps=1e-8)

    # Cosine similarity per position
    cos_per_position = (a_norm * b_norm).sum(dim=-1)  # [B, T]

    # Average over valid positions
    valid_count = mask.sum().clamp(min=1.0)
    cos_mean = (cos_per_position * mask).sum() / valid_count

    return float(cos_mean.item())


def _compute_mse(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute mean squared error between logits.

    Args:
        logits_a: Reference logits [B, T, V]
        logits_b: Comparison logits [B, T, V]
        mask: Attention mask [B, T]

    Returns:
        Mean MSE over valid positions
    """
    # MSE per position (mean over vocab dim)
    mse_per_position = ((logits_a - logits_b) ** 2).mean(dim=-1)  # [B, T]

    # Average over valid positions
    valid_count = mask.sum().clamp(min=1.0)
    mse_mean = (mse_per_position * mask).sum() / valid_count

    return float(mse_mean.item())


def _compute_top1_agreement(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute top-1 prediction agreement rate.

    Args:
        logits_a: Reference logits [B, T, V]
        logits_b: Comparison logits [B, T, V]
        mask: Attention mask [B, T]

    Returns:
        Fraction of positions where top-1 predictions match
    """
    top1_a = logits_a.argmax(dim=-1)  # [B, T]
    top1_b = logits_b.argmax(dim=-1)  # [B, T]

    agreement = (top1_a == top1_b).float()  # [B, T]

    # Average over valid positions
    valid_count = mask.sum().clamp(min=1.0)
    agreement_mean = (agreement * mask).sum() / valid_count

    return float(agreement_mean.item())


def _compute_all_metrics(
    logits_current: torch.Tensor,
    logits_sparse: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Compute all comparison metrics.

    Args:
        logits_current: Reference logits from current model
        logits_sparse: Logits from sparse-updated model
        mask: Attention mask

    Returns:
        Tuple of (kl_div, cosine_sim, mse, top1_agree)
    """
    kl = _compute_kl_divergence(logits_current, logits_sparse, mask)
    cos = _compute_cosine_similarity(logits_current, logits_sparse, mask)
    mse = _compute_mse(logits_current, logits_sparse, mask)
    top1 = _compute_top1_agreement(logits_current, logits_sparse, mask)

    return kl, cos, mse, top1


class SparseQualityAnalyzer:
    """Analyzes quality of sparse weight updates.

    Uses ParamChangeTracker's snapshot to avoid duplicate memory usage.
    Runs at the same interval as param tracking.

    Memory-safe design:
    - Deltas computed on CPU in float32 for precision
    - Model temporarily patched for forward pass, then restored
    - Logits moved to CPU for metric computation
    """

    def __init__(
        self,
        tracker: ParamChangeTracker,
        enabled: bool = True,
        thresholds: tuple[float, ...] = SPARSE_QUALITY_THRESHOLDS,
    ) -> None:
        """Initialize the analyzer.

        Args:
            tracker: ParamChangeTracker instance (provides snapshot)
            enabled: Whether sparse quality analysis is enabled
            thresholds: Sparsity thresholds to test
        """
        self.tracker = tracker
        self.enabled = enabled
        self.thresholds = thresholds

    @classmethod
    def from_config(
        cls,
        tracker: ParamChangeTracker,
        config: Any,
    ) -> SparseQualityAnalyzer:
        """Create analyzer from TrainingConfig.

        Args:
            tracker: ParamChangeTracker instance
            config: TrainingConfig instance

        Returns:
            Configured SparseQualityAnalyzer
        """
        return cls(
            tracker=tracker,
            enabled=getattr(config, "sparse_quality_enabled", False),
        )

    def analyze(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SparseQualityMetrics:
        """Analyze sparse update quality.

        Args:
            model: Current model (with updated weights)
            input_ids: Batch input token IDs [B, T]
            attention_mask: Attention mask [B, T]

        Returns:
            SparseQualityMetrics with results for all thresholds
        """
        if not self.tracker.has_snapshot():
            raise RuntimeError("No snapshot available. Cannot analyze sparse quality.")

        snapshot = self.tracker.get_snapshot()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Step 1: Compute deltas on CPU in float32 for precision
        deltas: dict[str, torch.Tensor] = {}
        total_params = 0

        for name, param in model.named_parameters():
            if name not in snapshot:
                continue
            # Convert to float32 BEFORE subtraction for precision
            current_fp32 = param.data.cpu().float()
            snapshot_fp32 = snapshot[name].float()
            deltas[name] = current_fp32 - snapshot_fp32
            total_params += deltas[name].numel()

        # Step 2: Get reference logits from current model
        with torch.no_grad():
            logits_current = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits_current_cpu = logits_current.cpu().float()
            del logits_current  # Free GPU memory immediately
            torch.cuda.empty_cache()

        # Step 3: For each threshold, compute metrics
        results = SparseQualityMetrics()

        for threshold in self.thresholds:
            # Create magnitude-based mask
            mask_dict: dict[str, torch.Tensor] = {}
            kept_params = 0

            for name, delta in deltas.items():
                mask = delta.abs() > threshold
                mask_dict[name] = mask
                kept_params += mask.sum().item()

            kept_ratio = (kept_params / total_params) if total_params > 0 else 0.0

            # Apply sparse delta and get logits
            logits_sparse_cpu = self._get_sparse_logits(
                model, snapshot, deltas, mask_dict, input_ids, attention_mask, device, dtype
            )
            torch.cuda.empty_cache()  # Free any GPU memory from forward pass

            # Create random mask with same sparsity level
            random_mask_dict = self._create_random_mask(deltas, kept_ratio)

            # Apply random sparse delta and get logits
            logits_random_cpu = self._get_sparse_logits(
                model, snapshot, deltas, random_mask_dict, input_ids, attention_mask, device, dtype
            )
            torch.cuda.empty_cache()  # Free any GPU memory from forward pass

            # Compute metrics (on CPU)
            mask_2d = attention_mask.cpu().float()

            kl, cos, mse, top1 = _compute_all_metrics(
                logits_current_cpu, logits_sparse_cpu, mask_2d
            )
            kl_r, cos_r, mse_r, top1_r = _compute_all_metrics(
                logits_current_cpu, logits_random_cpu, mask_2d
            )

            results.threshold_metrics.append(
                ThresholdMetrics(
                    threshold=threshold,
                    kept_ratio=kept_ratio,
                    kl_divergence=kl,
                    cosine_similarity=cos,
                    mse=mse,
                    top1_agreement=top1,
                    kl_divergence_random=kl_r,
                    cosine_similarity_random=cos_r,
                    mse_random=mse_r,
                    top1_agreement_random=top1_r,
                )
            )

            # Free intermediate CPU tensors
            del logits_sparse_cpu, logits_random_cpu, mask_dict, random_mask_dict

        # Cleanup
        del deltas, logits_current_cpu
        torch.cuda.empty_cache()

        return results

    def _get_sparse_logits(
        self,
        model: nn.Module,
        snapshot: dict[str, torch.Tensor],
        deltas: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get logits from model with sparse delta applied.

        Temporarily patches model weights, runs forward pass, then restores.
        Memory-efficient: restores using snapshot + full_delta instead of cloning.

        Args:
            model: The model
            snapshot: Original weights (on CPU)
            deltas: Weight deltas (on CPU, float32)
            mask_dict: Sparsity masks for each parameter
            input_ids: Input tokens
            attention_mask: Attention mask
            device: Target device
            dtype: Target dtype

        Returns:
            Logits on CPU as float32
        """
        # Track which params we modified (for restoration)
        modified_params: set[str] = set()

        for name, param in model.named_parameters():
            if name not in snapshot:
                continue

            modified_params.add(name)

            # Compute sparse delta (float32 on CPU)
            sparse_delta = deltas[name] * mask_dict[name]

            # Apply: W = W_old + sparse_delta (in float32, then convert)
            snapshot_fp32 = snapshot[name].float()
            new_weight = snapshot_fp32 + sparse_delta
            param.data.copy_(new_weight.to(device=device, dtype=dtype))

        try:
            # Forward pass
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                logits_cpu = logits.cpu().float()
                del logits  # Free GPU memory immediately
        finally:
            # Restore original weights using snapshot + full_delta (no GPU clone needed!)
            for name, param in model.named_parameters():
                if name in modified_params:
                    # W_current = W_old + delta, so restore by computing on CPU and moving to GPU
                    snapshot_fp32 = snapshot[name].float()
                    full_delta = deltas[name]  # Full delta (not sparse)
                    original_weight = snapshot_fp32 + full_delta
                    param.data.copy_(original_weight.to(device=device, dtype=dtype))

        return logits_cpu

    def _create_random_mask(
        self,
        deltas: dict[str, torch.Tensor],
        keep_fraction: float,
    ) -> dict[str, torch.Tensor]:
        """Create random mask with specified keep fraction.

        Args:
            deltas: Weight deltas (for shape reference)
            keep_fraction: Fraction of parameters to keep (0.0 to 1.0)

        Returns:
            Dictionary of random boolean masks
        """
        random_mask: dict[str, torch.Tensor] = {}

        for name, delta in deltas.items():
            # Random uniform -> threshold to get desired sparsity
            rand_tensor = torch.rand_like(delta)
            random_mask[name] = rand_tensor < keep_fraction

        return random_mask


async def log_sparse_quality_metrics(
    metrics: SparseQualityMetrics,
    monitor: Any,
    step: int,
) -> None:
    """Log sparse quality metrics to monitoring system.

    Args:
        metrics: Computed metrics
        monitor: Monitoring system instance
        step: Optimizer step count
    """
    if monitor is None:
        return

    log_dict = metrics.to_log_dict()

    for key, value in log_dict.items():
        try:
            await monitor.log_gauge(
                key,
                value,
                tags={"optimizer_step": str(step)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to log sparse quality metric %s: %s", key, exc)

    # Log summary
    if metrics.threshold_metrics:
        # Log at 1e-6 threshold as representative
        for tm in metrics.threshold_metrics:
            if tm.threshold == 1e-6:
                logger.info(
                    "Sparse quality at step %d (τ=1e-6, keep=%.2f%%): "
                    "KL=%.4f (rand=%.4f), top1=%.2f%% (rand=%.2f%%)",
                    step,
                    tm.kept_ratio * 100,
                    tm.kl_divergence,
                    tm.kl_divergence_random,
                    tm.top1_agreement * 100,
                    tm.top1_agreement_random * 100,
                )
                break
