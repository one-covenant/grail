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
    SchemaValidator,
    TerminationValidator,
    TokenValidator,
)


def create_sat_validation_pipeline() -> ValidationPipeline:
    """Create validation pipeline for SAT rollouts.

    Pipeline order (fail-fast):
    1. Schema (structure/types, no GPU)
    2. Tokens (vocab bounds, sequence length)
    3. GRAIL proof (GPU/framework-agnostic cryptographic proof, caches logits)
    4. SAT problem (regeneration from seed)
    5. SAT prompt (canonical prefix matching)
    6. Termination (max length or EOS)
    7. Distribution (anti-gaming heuristic)
    8. SAT solution (if success claimed)
    """
    return ValidationPipeline(
        [
            SchemaValidator(),  # FIRST - structure/types, no GPU
            TokenValidator(),  # SECOND - vocab/length check
            GRAILProofValidator(),  # Cryptographic proof validation
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
    "SchemaValidator",
    "TokenValidator",
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
