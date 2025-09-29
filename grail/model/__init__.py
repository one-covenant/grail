"""Model loading and configuration for GRAIL.

Centralized model and tokenizer initialization to ensure consistency
across Prover, Verifier, and Trainer components.
"""

from __future__ import annotations

from .provider import get_model, get_tokenizer

__all__ = ["get_model", "get_tokenizer"]
