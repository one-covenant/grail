"""Token validation and hashing utilities for GRAIL protocol.

Pure functions for:
- Token serialization and hashing
- Token list validation against model config
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    try:
        from transformers import PretrainedConfig
    except ImportError:
        PretrainedConfig = Any  # type: ignore

from ..shared.hf_compat import resolve_max_context_length, resolve_vocab_size

logger = logging.getLogger(__name__)


def int_to_bytes(i: int) -> bytes:
    """Convert integer to 4-byte big-endian representation."""
    return struct.pack(">I", i & 0xFFFFFFFF)


def hash_tokens(tokens: list[int]) -> bytes:
    """Compute SHA-256 hash of tokens for integrity checking."""
    tokens_bytes = b"".join(int_to_bytes(t) for t in tokens)
    return hashlib.sha256(tokens_bytes).digest()


def verify_tokens(tokens: list[int], model_config: PretrainedConfig | Any) -> bool:
    """Verify token list validity for model processing.

    Args:
        tokens: List of token IDs to verify
        model_config: Model configuration object with vocab_size and max sequence length attributes

    Returns:
        True if tokens are valid, False otherwise
    """
    # Check empty tokens
    if not tokens:
        logger.warning("Empty token list in commit")
        return False

    # Validate token IDs (best-effort if vocab size available)
    vocab_size = resolve_vocab_size(model_config)
    if vocab_size is not None:
        if not _validate_token_ids(tokens, vocab_size):
            return False
    else:
        logger.debug("Model config lacks vocab_size; skipping token-id bounds check")

    # Validate sequence length
    if not _validate_sequence_length(tokens, model_config):
        return False

    return True


def _validate_token_ids(tokens: list[int], vocab_size: int) -> bool:
    """Check that all token IDs are within vocabulary bounds."""
    invalid_tokens = [t for t in tokens if not isinstance(t, int) or t < 0 or t >= vocab_size]
    if invalid_tokens:
        logger.warning(
            f"Invalid token IDs found in verification: {invalid_tokens[:10]}... "
            f"(vocab_size={vocab_size})"
        )
        return False
    return True


def _validate_sequence_length(tokens: list[int], model_config: PretrainedConfig | Any) -> bool:
    """Check that token sequence doesn't exceed model's max length."""
    max_length = resolve_max_context_length(model_config)

    if len(tokens) > max_length:
        logger.warning(f"Token sequence ({len(tokens)}) exceeds model max length ({max_length})")
        return False
    return True
