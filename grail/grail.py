#!/usr/bin/env python3
"""GRAIL protocol re-exports.

Stage 3: This module now serves as a re-export hub for protocol functions.
The monolithic Verifier class (1,161 lines) has been deleted and replaced with:
- grail.validation.validators.* - Individual validator classes
- grail.validation.create_sat_validation_pipeline() - Pipeline composition
- grail.scoring.* - Scoring and weight computation

Original grail.py was 1,722 lines. Now 124 lines (93% reduction).
"""

from __future__ import annotations

# Re-export protocol functions for backward compatibility
from .protocol.crypto import (  # noqa: F401
    create_proof,
    dot_mod_q,
    indices_from_root,
    prf,
    r_vec_from_randomness,
)
from .protocol.signatures import (  # noqa: F401
    build_commit_binding,
    derive_env_seed,
    hash_commitments,
    sign_commit_binding,
    verify_commit_signature,
)
from .protocol.tokens import (  # noqa: F401
    hash_tokens,
    int_to_bytes,
    verify_tokens,
)

__all__ = [
    # Crypto
    "prf",
    "r_vec_from_randomness",
    "indices_from_root",
    "dot_mod_q",
    "create_proof",
    # Signatures
    "build_commit_binding",
    "sign_commit_binding",
    "verify_commit_signature",
    "derive_env_seed",
    "hash_commitments",
    # Tokens
    "int_to_bytes",
    "hash_tokens",
    "verify_tokens",
]
