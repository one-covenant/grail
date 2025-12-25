"""Configuration for model analysis system.

Centralizes all configuration for parameter change tracking, sparse quality
analysis, and future metric extensions (gradients, momentum, activations).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration for model analysis during training.

    This configuration controls when and what metrics are computed during training.
    All analysis is triggered at regular intervals to minimize overhead.

    Attributes:
        interval: Measure every N optimizer steps (default: 100)
        snapshot_device: Device for storing snapshots (default: "cpu")
        snapshot_dtype: Dtype for snapshot storage (default: "float32")

        # Parameter Change Analysis
        param_change_enabled: Enable parameter change tracking
        param_change_thresholds: Sparsity thresholds for counting changed params
        param_change_per_layer: Track per-layer statistics (more verbose)
        param_change_track_components: Track by component (attention, mlp, embed)

        # Sparse Quality Analysis
        sparse_quality_enabled: Enable sparse update quality measurement
        sparse_quality_thresholds: Thresholds for sparse approximation testing
        sparse_quality_include_random: Include random baseline comparison

        # Gradient Analysis (future)
        gradient_enabled: Enable gradient statistics tracking
        gradient_track_per_layer: Track gradient statistics per layer

        # Momentum Analysis (future)
        momentum_enabled: Enable optimizer momentum tracking

    Example:
        >>> config = AnalysisConfig(
        ...     interval=100,
        ...     param_change_enabled=True,
        ...     sparse_quality_enabled=True,
        ...     sparse_quality_thresholds=[1e-6, 1e-4],
        ... )
        >>> analyzer = ModelAnalysisManager.create(config)
    """

    # Global settings
    interval: int = 100
    snapshot_device: str = "cpu"
    snapshot_dtype: str = "float32"

    # Parameter change analysis
    param_change_enabled: bool = True
    param_change_thresholds: list[float] = field(default_factory=lambda: [0.0])
    param_change_per_layer: bool = False
    param_change_track_components: bool = False

    # Sparse quality analysis
    sparse_quality_enabled: bool = True
    sparse_quality_thresholds: list[float] = field(default_factory=lambda: [0.0])
    sparse_quality_include_random: bool = True

    # Gradient analysis (future)
    gradient_enabled: bool = False
    gradient_track_per_layer: bool = False

    # Momentum analysis (future)
    momentum_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")

        if self.snapshot_device not in ["cpu", "cuda"]:
            raise ValueError(
                f"snapshot_device must be 'cpu' or 'cuda', got '{self.snapshot_device}'"
            )

        valid_dtypes = ["float32", "float16", "bfloat16"]
        if self.snapshot_dtype not in valid_dtypes:
            raise ValueError(
                f"snapshot_dtype must be one of {valid_dtypes}, got '{self.snapshot_dtype}'"
            )

        # Validate thresholds are positive
        for threshold in self.param_change_thresholds:
            if threshold < 0:
                raise ValueError(f"param_change_thresholds must be non-negative, got {threshold}")

        for threshold in self.sparse_quality_thresholds:
            if threshold < 0:
                raise ValueError(f"sparse_quality_thresholds must be non-negative, got {threshold}")

    @classmethod
    def minimal(cls) -> AnalysisConfig:
        """Create minimal configuration (parameter change only, high interval).

        Use this for minimal overhead during training.

        Returns:
            AnalysisConfig with only basic parameter tracking enabled
        """
        return cls(
            interval=500,
            param_change_enabled=True,
            param_change_per_layer=False,
            sparse_quality_enabled=False,
            gradient_enabled=False,
        )

    @classmethod
    def comprehensive(cls) -> AnalysisConfig:
        """Create comprehensive configuration (all metrics, frequent sampling).

        Use this for detailed analysis during research/debugging.

        Returns:
            AnalysisConfig with all available metrics enabled
        """
        return cls(
            interval=50,
            param_change_enabled=True,
            param_change_per_layer=True,
            param_change_track_components=True,
            sparse_quality_enabled=True,
            sparse_quality_include_random=True,
            gradient_enabled=False,  # Will enable when implemented
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> AnalysisConfig:
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            AnalysisConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "interval": self.interval,
            "snapshot_device": self.snapshot_device,
            "snapshot_dtype": self.snapshot_dtype,
            "param_change_enabled": self.param_change_enabled,
            "param_change_thresholds": self.param_change_thresholds,
            "param_change_per_layer": self.param_change_per_layer,
            "param_change_track_components": self.param_change_track_components,
            "sparse_quality_enabled": self.sparse_quality_enabled,
            "sparse_quality_thresholds": self.sparse_quality_thresholds,
            "sparse_quality_include_random": self.sparse_quality_include_random,
            "gradient_enabled": self.gradient_enabled,
            "gradient_track_per_layer": self.gradient_track_per_layer,
            "momentum_enabled": self.momentum_enabled,
        }
