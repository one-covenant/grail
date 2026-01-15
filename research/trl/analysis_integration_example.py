"""Integration example: Model Analysis with TRL GRPO Training.

This script demonstrates how to integrate the grail.trainer.analysis module
into TRL's GRPO training loop. It shows both basic and advanced usage patterns.

The analysis module measures:
- Parameter change statistics (magnitude, sparsity)
- Sparse update quality (can we get similar results with sparse updates?)

Key Integration Points:
1. Create AnalysisConfig and ModelAnalysisManager
2. Create a TrainerCallback that calls analyzer on optimizer steps
3. Log metrics to WandB with proper namespacing

Run this alongside train_trl_grpo.py to add analysis:
    python train_trl_grpo.py --dataset gsm8k  # with analysis integrated
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

# Import the analysis framework
from grail.trainer.analysis import AnalysisConfig, ModelAnalysisManager
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainingArguments

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS CALLBACK FOR TRL TRAINER
# ════════════════════════════════════════════════════════════════════════════
class ModelAnalysisCallback(TrainerCallback):
    """Callback for running model analysis during TRL GRPO training.

    This callback integrates grail.trainer.analysis.ModelAnalysisManager
    into the TRL training loop, measuring parameter changes and sparse
    update quality at regular intervals.

    Example:
        >>> config = AnalysisConfig(interval=100, sparse_quality_enabled=True)
        >>> analyzer = ModelAnalysisManager.create(config)
        >>> callback = ModelAnalysisCallback(analyzer)
        >>>
        >>> trainer = GRPOTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_ds,
        ...     callbacks=[callback],  # Add analysis callback
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        analyzer: ModelAnalysisManager,
        log_to_wandb: bool = True,
        log_prefix: str = "analysis",
    ) -> None:
        """Initialize the callback.

        Args:
            analyzer: Configured ModelAnalysisManager instance
            log_to_wandb: Whether to log metrics to WandB
            log_prefix: Prefix for WandB metrics (default: "analysis")
        """
        self.analyzer = analyzer
        self.log_to_wandb = log_to_wandb
        self.log_prefix = log_prefix
        self._last_logged_step = -1

        logger.info(
            "ModelAnalysisCallback initialized: interval=%d, metrics=%d, log_to_wandb=%s",
            analyzer.config.interval,
            len(analyzer),
            log_to_wandb,
        )

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Called after optimizer.step() but before gradients are zeroed.

        Args:
            args: Training arguments
            state: Trainer state (contains global_step)
            control: Trainer control object
            **kwargs: Additional arguments (may contain model, inputs, etc.)
        """
        # Get model from kwargs (TRL passes it)
        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "Model not available in callback kwargs at step %d. Skipping analysis.",
                state.global_step,
            )
            return

        # Avoid duplicate WandB logs in distributed runs.
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not is_world_process_zero:
            return

        # Try to get batch inputs (may not always be available)
        # TRL might pass this in kwargs, or we can skip it
        inputs = kwargs.get("inputs")
        optimizer = kwargs.get("optimizer")

        # Run analysis (returns empty dict if not at measurement interval)
        try:
            metrics = self.analyzer.on_optimizer_step(model, inputs=inputs, optimizer=optimizer)
        except Exception as e:
            logger.error(
                "Analysis failed at step %d: %s",
                state.global_step,
                e,
                exc_info=True,
            )
            return

        # Log to WandB if metrics were computed
        if metrics and self.log_to_wandb:
            self._log_metrics_to_wandb(metrics)

    def _log_metrics_to_wandb(self, metrics: dict[str, float]) -> None:
        """Log analysis metrics to WandB with proper namespacing.

        Args:
            metrics: Dictionary of metrics from analyzer
        """
        try:
            import wandb

            if wandb.run is not None:
                # Add prefix to all metric keys
                wandb_metrics = {f"{self.log_prefix}/{k}": v for k, v in metrics.items()}

                # Don't pass step= explicitly; let WandB use its internal step counter
                wandb.log(wandb_metrics)

                logger.info(
                    "Logged %d analysis metrics to WandB",
                    len(metrics),
                )
        except Exception as e:
            logger.warning("Failed to log to WandB: %s", e)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Called at the end of training.

        Logs final analysis summary.
        """
        logger.info(
            "Analysis complete: Measured at %d steps (interval: %d)",
            self.analyzer.step_count,
            self.analyzer.config.interval,
        )


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION EXAMPLES
# ════════════════════════════════════════════════════════════════════════════


def example_minimal_integration() -> ModelAnalysisCallback:
    """Minimal integration: Just parameter change tracking.

    This is the simplest setup - tracks basic parameter statistics
    with minimal overhead.

    Returns:
        ModelAnalysisCallback ready to use with trainer
    """
    # Create minimal configuration (high interval, param change only)
    config = AnalysisConfig.minimal()

    # Create manager with factory method
    analyzer = ModelAnalysisManager.create(config)

    # Create callback
    callback = ModelAnalysisCallback(analyzer, log_to_wandb=True)

    return callback


def example_comprehensive_integration() -> ModelAnalysisCallback:
    """Comprehensive integration: All metrics enabled.

    Use this for detailed analysis during research. Includes:
    - Parameter change statistics (multi-threshold)
    - Per-layer breakdowns
    - Per-component breakdowns (attention, MLP, etc.)
    - Sparse quality analysis with random baselines

    Returns:
        ModelAnalysisCallback ready to use with trainer
    """
    # Create comprehensive configuration
    config = AnalysisConfig.comprehensive()

    # Create manager
    analyzer = ModelAnalysisManager.create(config)

    # Create callback with custom prefix
    callback = ModelAnalysisCallback(
        analyzer,
        log_to_wandb=True,
        log_prefix="model_analysis",  # Custom namespace in WandB
    )

    return callback


def example_custom_configuration() -> ModelAnalysisCallback:
    """Custom configuration: Tailored to specific needs.

    This example shows how to customize analysis intervals,
    thresholds, and feature flags.

    Returns:
        ModelAnalysisCallback ready to use with trainer
    """
    # Create custom configuration
    config = AnalysisConfig(
        # Measurement interval: every 50 optimizer steps
        interval=50,
        # Parameter change settings
        param_change_enabled=True,
        param_change_thresholds=[1e-8, 1e-6, 1e-4, 1e-2],  # Custom thresholds
        param_change_per_layer=True,  # Per-layer stats
        param_change_track_components=True,  # Component breakdown
        # Sparse quality settings
        sparse_quality_enabled=True,
        sparse_quality_thresholds=[1e-6, 1e-4],  # Only 2 thresholds (faster)
        sparse_quality_include_random=True,  # Include random baseline
        # Snapshot settings
        snapshot_device="cpu",  # Store snapshots on CPU
        snapshot_dtype="float32",  # Full precision for deltas
    )

    # Create manager
    analyzer = ModelAnalysisManager.create(config)

    # Create callback
    callback = ModelAnalysisCallback(analyzer)

    return callback


def example_custom_metrics() -> ModelAnalysisCallback:
    """Advanced: Adding custom metric computers.

    This example shows how to extend the analysis framework
    with your own custom metrics.

    Returns:
        ModelAnalysisCallback ready to use with trainer
    """
    from grail.trainer.analysis import MetricComputer, ParameterDelta

    # Define custom metric computer
    class GradientNormMetric(MetricComputer):
        """Example: Track gradient norms."""

        def compute(self, delta: ParameterDelta | None = None, **kwargs) -> dict[str, float]:
            """Compute gradient statistics."""
            context = kwargs.get("context")
            if context is None or context.model is None:
                return {}

            # Compute total gradient norm
            total_norm = 0.0
            for p in context.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm(2).item() ** 2

            return {
                "gradient/total_norm": total_norm**0.5,
            }

    # Create base configuration
    config = AnalysisConfig(
        interval=100,
        param_change_enabled=True,
        sparse_quality_enabled=False,  # Disable for speed
    )

    # Create manager and add custom metric
    analyzer = ModelAnalysisManager(config).add_metric(GradientNormMetric())

    # Or use builder pattern
    # analyzer = (
    #     ModelAnalysisManager(config)
    #     .add_metric(ParameterChangeMetrics(thresholds=[1e-6]))
    #     .add_metric(GradientNormMetric())
    # )

    # Create callback
    callback = ModelAnalysisCallback(analyzer)

    return callback


# ════════════════════════════════════════════════════════════════════════════
# USAGE IN train_trl_grpo.py
# ════════════════════════════════════════════════════════════════════════════


def integrate_with_trl_training():
    """Example showing how to integrate into train_trl_grpo.py.

    Add this code to your training script:
    """
    # STEP 1: Import at top of train_trl_grpo.py
    # from grail.trainer.analysis import AnalysisConfig, ModelAnalysisManager
    # from analysis_integration_example import ModelAnalysisCallback

    # STEP 2: Create analysis callback (before creating trainer)
    # Option A: Minimal (fast, basic metrics)
    # analysis_callback = example_minimal_integration()

    # Option B: Comprehensive (detailed analysis)
    # analysis_callback = example_comprehensive_integration()

    # Option C: Custom configuration
    # config = AnalysisConfig(
    #     interval=100,  # Measure every 100 optimizer steps
    #     param_change_enabled=True,
    #     sparse_quality_enabled=True,
    # )
    # analyzer = ModelAnalysisManager.create(config)
    # analysis_callback = ModelAnalysisCallback(analyzer)

    # STEP 3: Add to trainer callbacks list
    # trainer = GRPOTrainer(
    #     model=model,
    #     reward_funcs=reward_tracker,
    #     args=grpo_config,
    #     train_dataset=train_ds,
    #     processing_class=tokenizer,
    #     callbacks=[
    #         vllm_eval_callback,  # Existing callback
    #         analysis_callback,   # ADD THIS
    #     ],
    # )

    # STEP 4: Train (analysis runs automatically)
    # trainer.train()

    # That's it! Metrics will be logged to WandB under "analysis/*" namespace
    pass


# ════════════════════════════════════════════════════════════════════════════
# INTERPRETING METRICS
# ════════════════════════════════════════════════════════════════════════════


def interpreting_metrics_guide():
    """Guide to interpreting analysis metrics in WandB.

    PARAMETER CHANGE METRICS (analysis/param_change/*):
    ────────────────────────────────────────────────────────
    - norm_l2: L2 norm of all parameter changes (overall magnitude)
    - norm_l1: L1 norm (sum of absolute changes)
    - norm_max: Maximum absolute change in any parameter
    - mean: Mean parameter change
    - std: Standard deviation of changes

    - sparsity_at_1e-06: Fraction of params with |delta| <= 1e-6 (unchanged)
    - changed_ratio_at_1e-06: Fraction of params with |delta| > 1e-6 (changed)

    → High sparsity (e.g., 0.95) = most parameters didn't change much
    → Low sparsity (e.g., 0.10) = most parameters changed significantly

    PER-LAYER METRICS (analysis/param_change/layer_N/*):
    ────────────────────────────────────────────────────────
    - mean_abs_delta: Average magnitude of change in layer N
    - sparsity: Fraction unchanged in layer N

    → Compare across layers: which layers learn fastest?

    PER-COMPONENT METRICS (analysis/param_change/component/*/):
    ────────────────────────────────────────────────────────
    - mean_abs_delta: Average change in attention (q_proj, k_proj, etc.) or MLP
    - sparsity: Fraction unchanged per component

    → Compare attention vs MLP: where is learning happening?


    SPARSE QUALITY METRICS (analysis/sparse/*):
    ────────────────────────────────────────────────────────
    These answer: "If we only updated the top X% of weights (by magnitude),
    how close would model outputs be to updating all weights?"

    At each threshold (e.g., 1e-06):
    - kl_at_1e-06: KL divergence between full and sparse updates (lower = better)
    - cosine_at_1e-06: Cosine similarity of logits (higher = better, range 0-1)
    - mse_at_1e-06: Mean squared error (lower = better)
    - top1_agree_at_1e-06: Fraction where top-1 prediction matches (higher = better)

    - kept_ratio_at_1e-06: What % of params were kept (not zeroed)
    - unchanged_ratio_at_1e-06: What % were dropped

    Random baselines (kl_at_1e-06_random, etc.):
    - If magnitude-based >> random, then large changes matter!
    - If magnitude-based ≈ random, then update is noisy/random

    Example interpretation:
        kl_at_1e-06 = 0.001 (very low)
        kept_ratio_at_1e-06 = 0.15 (only 15% of params)
        → We could achieve nearly identical outputs by updating only 15% of weights!
        → LoRA-style sparse training would work well here


    WHAT TO LOOK FOR:
    ────────────────────────────────────────────────────────
    1. Decreasing sparsity over time?
       → Model learning (parameters changing more)

    2. High kept_ratio but low KL divergence?
       → Sparse updates work well (LoRA-friendly)

    3. Different sparsity across layers/components?
       → Uneven learning (some layers frozen?)

    4. Magnitude-based much better than random baseline?
       → Large changes are meaningful (not just noise)
    """
    pass


if __name__ == "__main__":
    # Print examples
    print("=" * 80)
    print("MODEL ANALYSIS INTEGRATION EXAMPLES")
    print("=" * 80)

    print("\n1. MINIMAL INTEGRATION")
    print("-" * 80)
    callback1 = example_minimal_integration()
    print(f"   Created: {callback1.analyzer}")

    print("\n2. COMPREHENSIVE INTEGRATION")
    print("-" * 80)
    callback2 = example_comprehensive_integration()
    print(f"   Created: {callback2.analyzer}")

    print("\n3. CUSTOM CONFIGURATION")
    print("-" * 80)
    callback3 = example_custom_configuration()
    print(f"   Created: {callback3.analyzer}")

    print("\n4. CUSTOM METRICS")
    print("-" * 80)
    callback4 = example_custom_metrics()
    print(f"   Created: {callback4.analyzer}")

    print("\n" + "=" * 80)
    print("See docstrings for usage in train_trl_grpo.py")
    print("=" * 80)
