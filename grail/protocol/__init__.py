"""GRAIL protocol: cryptographic primitives, signatures, and proof verification.

This package contains the core GRAIL protocol implementation extracted from
the monolithic grail.py module for better testability and maintainability.
"""

from __future__ import annotations

# Re-export crypto primitives
from .crypto import (
    create_proof,
    dot_mod_q,
    indices_from_root,
    prf,
    r_vec_from_randomness,
)

# Re-export signature functions
from .signatures import (
    build_commit_binding,
    derive_canonical_sat,
    sign_commit_binding,
    sign_s_vals,
    verify_commit_signature,
    verify_s_vals_signature,
)

# Re-export token utilities
from .tokens import (
    hash_s_vals,
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
    "sign_s_vals",
    "verify_s_vals_signature",
    "build_commit_binding",
    "sign_commit_binding",
    "verify_commit_signature",
    "derive_canonical_sat",
    # Tokens
    "int_to_bytes",
    "hash_tokens",
    "hash_s_vals",
    "verify_tokens",
]
