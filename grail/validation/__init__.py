"""GRAIL validation package.

Provides:
- Context-based validation architecture with composable validators (NEW)
- Copycat detection for anti-gaming (EXISTING)
"""

from __future__ import annotations

# Core validation architecture
from .base import Validator
from .context import ValidationContext

# Existing copycat detection
from .copycat import (
    COPYCAT_INTERVAL_THRESHOLD,
    COPYCAT_TRACKER,
    COPYCAT_WINDOW_THRESHOLD,
    CopycatViolation,
    compute_completion_digest,
)
from .pipeline import ValidationPipeline

# All validators
from .validators import (
    DistributionValidator,
    GRAILProofValidator,
    SATProblemValidator,
    SATPromptValidator,
    SATSolutionValidator,
    TerminationValidator,
)


def create_sat_validation_pipeline() -> ValidationPipeline:
    """Create validation pipeline for SAT rollouts.

    Pipeline order (fail-fast):
    1. GRAIL proof (cryptographic, caches logits)
    2. SAT problem (regeneration from seed)
    3. SAT prompt (canonical prefix matching)
    4. Termination (max length or EOS)
    5. Distribution (anti-gaming heuristic)
    6. SAT solution (if success claimed)
    """
    return ValidationPipeline(
        [
            GRAILProofValidator(),
            SATProblemValidator(),
            SATPromptValidator(),
            TerminationValidator(),
            DistributionValidator(),
            SATSolutionValidator(),
        ]
    )


__all__ = [
    # Core
    "Validator",
    "ValidationContext",
    "ValidationPipeline",
    # Pipeline factory
    "create_sat_validation_pipeline",
    # Validators
    "GRAILProofValidator",
    "SATProblemValidator",
    "SATPromptValidator",
    "SATSolutionValidator",
    "TerminationValidator",
    "DistributionValidator",
    # Copycat
    "COPYCAT_INTERVAL_THRESHOLD",
    "COPYCAT_TRACKER",
    "COPYCAT_WINDOW_THRESHOLD",
    "CopycatViolation",
    "compute_completion_digest",
]
