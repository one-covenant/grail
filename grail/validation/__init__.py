"""Validation service and components for GRAIL protocol.

This package contains the refactored validation logic, separated from CLI concerns.

Main Components:
- ValidationService: Main orchestration service
- MinerValidator: Single miner validation logic
- WindowProcessor: Single window processing orchestration
- MinerSampler: Miner discovery and sampling
- CopycatService: Copycat detection and gating

Types:
- ValidationContext: Shared context for validation operations
- WindowResults: Aggregated window validation results
- MinerResults: Single miner validation results
"""

from ..shared.digest import compute_completion_digest
from .context import ValidationContext
from .copycat_service import (
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_SERVICE,
    COPYCAT_TRACKER,
    COPYCAT_WINDOW_THRESHOLD,
    CopycatService,
    CopycatViolation,
)
from .miner_validator import MinerValidator
from .pipeline import (
    ValidationPipeline,
    create_env_validation_pipeline,
    get_hard_check_keys,
    get_soft_check_keys,
)
from .sampling import MinerSampler
from .service import ValidationService
from .types import MinerResults, WindowResults
from .window_processor import WindowProcessor

__all__ = [
    # Service layer
    "ValidationService",
    "MinerSampler",
    "MinerValidator",
    "WindowProcessor",
    "CopycatService",
    "COPYCAT_SERVICE",
    # Pipeline
    "ValidationPipeline",
    "create_env_validation_pipeline",
    "get_hard_check_keys",
    "get_soft_check_keys",
    # Copycat detection
    "COPYCAT_INTERVAL_THRESHOLD",
    "COPYCAT_TRACKER",
    "COPYCAT_WINDOW_THRESHOLD",
    "CopycatViolation",
    "compute_completion_digest",
    # Types
    "ValidationContext",
    "WindowResults",
    "MinerResults",
]
