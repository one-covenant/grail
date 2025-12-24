"""Sparse update quality metrics computation.

Measures how well sparse weight updates (keeping only large changes) approximate
full updates by comparing model outputs. This helps determine if LoRA-style
sparse training would be effective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from grail.trainer.analysis.metrics.base import AnalysisContext, MetricComputer
from grail.trainer.analysis.primitives import ParameterDelta, ParameterSnapshot

if TYPE_CHECKING:
    from torch import nn


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
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)

    # KL(P_a || P_b): how much P_a diverges from P_b
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
    # Normalize along vocab dimension
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


class SparseQualityMetrics(MetricComputer):
    """Compute sparse update quality metrics.

    Tests how well sparse updates (keeping only large parameter changes)
    approximate full updates by comparing model outputs.

    For each threshold:
    1. Create sparse delta (zero out small changes)
    2. Apply to model: W = W_old + sparse_delta
    3. Run forward pass and compare logits to full update
    4. (Optional) Compare to random baseline with same sparsity

    This answers: "If we only updated the top X% of weights by magnitude,
    how close would the model behavior be to updating all weights?"

    Requires:
    - Model in context
    - Batch inputs in context

    Example:
        >>> computer = SparseQualityMetrics(
        ...     thresholds=[1e-6, 1e-4],
        ...     include_random_baseline=True,
        ... )
        >>> metrics = computer.compute(
        ...     delta=delta,
        ...     old_snapshot=old_snap,
        ...     context=AnalysisContext(model=model, inputs=batch),
        ... )
        >>> print(metrics["sparse/kl_at_1e-06"])  # How much do outputs differ?
    """

    def __init__(
        self,
        thresholds: list[float] | None = None,
        include_random_baseline: bool = True,
    ) -> None:
        """Initialize sparse quality metric computer.

        Args:
            thresholds: Sparsity thresholds to test (default: [1e-8, 1e-6, 1e-4])
            include_random_baseline: Compute random baseline for comparison
        """
        self.thresholds = thresholds if thresholds is not None else [1e-8, 1e-6, 1e-4]
        self.include_random_baseline = include_random_baseline

    def requires_model(self) -> bool:
        """Requires model for forward passes."""
        return True

    def requires_inputs(self) -> bool:
        """Requires batch inputs for forward passes."""
        return True

    def compute(
        self,
        delta: ParameterDelta | None = None,
        old_snapshot: ParameterSnapshot | None = None,
        current_snapshot: ParameterSnapshot | None = None,
        context: AnalysisContext | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute sparse quality metrics.

        Args:
            delta: Parameter delta to analyze
            old_snapshot: Old snapshot (for applying sparse deltas)
            current_snapshot: Current snapshot (unused, for symmetry)
            context: Must contain model and inputs
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metrics with "sparse/" prefix
        """
        if delta is None or old_snapshot is None or context is None:
            return {}

        if not self.validate_context(context):
            return {}

        model = context.model
        inputs = context.inputs

        if model is None or inputs is None:
            return {}

        metrics: dict[str, float] = {}

        # Get device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Get reference logits from current model (full update)
        with torch.no_grad():
            logits_full = model(**inputs).logits
            logits_full_cpu = logits_full.cpu().float()
            del logits_full  # Free GPU memory
            torch.cuda.empty_cache()

        # Create attention mask for metric computation
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].cpu().float()
        else:
            # Assume all positions are valid if no mask provided
            batch_size, seq_len = logits_full_cpu.shape[:2]
            mask = torch.ones(batch_size, seq_len)

        # Test each threshold
        for threshold in self.thresholds:
            # Compute sparsity info
            sparsity_info = delta.sparsity_at_threshold(threshold)
            kept_ratio = sparsity_info["kept_ratio"]

            # Create sparse delta
            sparse_delta = delta.apply_sparse_mask(threshold, mask_type="magnitude")

            # Get logits with sparse update
            logits_sparse_cpu = self._get_sparse_logits(
                model, old_snapshot, sparse_delta, inputs, device, dtype
            )
            torch.cuda.empty_cache()

            # Compute quality metrics
            kl = _compute_kl_divergence(logits_full_cpu, logits_sparse_cpu, mask)
            cos = _compute_cosine_similarity(logits_full_cpu, logits_sparse_cpu, mask)
            mse = _compute_mse(logits_full_cpu, logits_sparse_cpu, mask)
            top1 = _compute_top1_agreement(logits_full_cpu, logits_sparse_cpu, mask)

            # Store metrics
            thresh_str = f"{threshold:.0e}"
            metrics[f"sparse/kl_at_{thresh_str}"] = kl
            metrics[f"sparse/cosine_at_{thresh_str}"] = cos
            metrics[f"sparse/mse_at_{thresh_str}"] = mse
            metrics[f"sparse/top1_agree_at_{thresh_str}"] = top1
            metrics[f"sparse/kept_ratio_at_{thresh_str}"] = kept_ratio
            metrics[f"sparse/unchanged_ratio_at_{thresh_str}"] = 1.0 - kept_ratio

            # Random baseline comparison
            if self.include_random_baseline:
                random_delta = delta.apply_sparse_mask(threshold, mask_type="random")
                logits_random_cpu = self._get_sparse_logits(
                    model, old_snapshot, random_delta, inputs, device, dtype
                )
                torch.cuda.empty_cache()

                kl_r = _compute_kl_divergence(logits_full_cpu, logits_random_cpu, mask)
                cos_r = _compute_cosine_similarity(logits_full_cpu, logits_random_cpu, mask)
                mse_r = _compute_mse(logits_full_cpu, logits_random_cpu, mask)
                top1_r = _compute_top1_agreement(logits_full_cpu, logits_random_cpu, mask)

                metrics[f"sparse/kl_at_{thresh_str}_random"] = kl_r
                metrics[f"sparse/cosine_at_{thresh_str}_random"] = cos_r
                metrics[f"sparse/mse_at_{thresh_str}_random"] = mse_r
                metrics[f"sparse/top1_agree_at_{thresh_str}_random"] = top1_r

                del logits_random_cpu

            del logits_sparse_cpu

        # Cleanup
        del logits_full_cpu
        torch.cuda.empty_cache()

        return metrics

    def _get_sparse_logits(
        self,
        model: nn.Module,
        old_snapshot: ParameterSnapshot,
        sparse_delta: ParameterDelta,
        inputs: dict,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get logits from model with sparse delta applied.

        Temporarily patches model weights, runs forward pass, then restores.

        Args:
            model: The model
            old_snapshot: Original weights (W_old)
            sparse_delta: Sparse delta to apply
            inputs: Batch inputs
            device: Target device
            dtype: Target dtype

        Returns:
            Logits on CPU as float32
        """
        # Track which params we modified (for restoration)
        modified_params: set[str] = set()
        original_weights: dict[str, torch.Tensor] = {}

        # Apply sparse update: W = W_old + sparse_delta
        for name, param in model.named_parameters():
            if name not in old_snapshot.data or name not in sparse_delta.deltas:
                continue

            modified_params.add(name)

            # Save original for restoration
            original_weights[name] = param.data.clone()

            # Apply: W = W_old + sparse_delta
            old_weight = old_snapshot.data[name].float()
            delta_sparse = sparse_delta.deltas[name]
            new_weight = old_weight + delta_sparse

            param.data.copy_(new_weight.to(device=device, dtype=dtype))

        try:
            # Forward pass
            with torch.no_grad():
                logits = model(**inputs).logits
                logits_cpu = logits.cpu().float()
                del logits  # Free GPU memory immediately
        finally:
            # Restore original weights
            for name in modified_params:
                param = dict(model.named_parameters())[name]
                param.data.copy_(original_weights[name])

        return logits_cpu

    @property
    def name(self) -> str:
        """Human-readable name."""
        return "SparseQualityMetrics"
