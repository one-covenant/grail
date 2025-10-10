"""Digest utilities for GRAIL validation.

Contains functions for computing stable hashes of rollout data
for copycat detection and other validation purposes.
"""

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def compute_completion_digest(commit_data: dict, rollout_meta: dict) -> str | None:
    """Return a SHA-256 digest of completion token IDs for copycat detection.

    We canonically slice completion tokens as tokens[prompt_length:] so identical
    completions map to the same digest across validators regardless of prompt.
    On failure to slice, we fall back to hashing the full token list.

    Args:
        commit_data: Dictionary containing 'tokens' key with token IDs
        rollout_meta: Dictionary containing 'prompt_length' key

    Returns:
        SHA-256 hex digest of completion tokens, or None if computation fails

    Example:
        >>> commit = {"tokens": [1, 2, 3, 4, 5]}
        >>> meta = {"prompt_length": 2}
        >>> digest = compute_completion_digest(commit, meta)
        >>> isinstance(digest, str) and len(digest) == 64
        True
    """
    try:
        tokens = commit_data.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            return None

        try:
            prompt_len = int(rollout_meta.get("prompt_length", 0) or 0)
        except Exception:  # pragma: no cover - defensive fallback
            prompt_len = 0

        completion_ids = tokens[prompt_len:]
        digest_input = json.dumps(
            completion_ids, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(digest_input).hexdigest()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Completion digest computation failed (%s)", exc)
        try:
            digest_input = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode()
            return hashlib.sha256(digest_input).hexdigest()
        except Exception as fallback_exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to hash tokens for copycat digest (%s)", fallback_exc)
            return None


__all__ = ["compute_completion_digest"]
