"""Base classes and utilities for text generation backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from ...shared.constants import MAX_NEW_TOKENS

logger = logging.getLogger(__name__)


def _decode_prompts(tokenizer: Any, prompt_ids_batch: list[list[int]]) -> list[str]:
    """Decode batch of token IDs to text prompts.

    Args:
        tokenizer: HuggingFace tokenizer instance
        prompt_ids_batch: List of tokenized prompts

    Returns:
        List of decoded text prompts
    """
    return [tokenizer.decode(p, skip_special_tokens=False) for p in prompt_ids_batch]


def _tokenize_completion(tokenizer: Any, completion_text: str, fallback: list[int]) -> list[int]:
    """Tokenize completion text with error handling.

    Args:
        tokenizer: HuggingFace tokenizer instance
        completion_text: Generated completion text
        fallback: Token IDs to return on error (typically prompt IDs)

    Returns:
        List of completion token IDs
    """
    try:
        comp_ids = tokenizer(completion_text, return_tensors=None, return_attention_mask=False)[
            "input_ids"
        ]

        # Handle nested list structure
        if isinstance(comp_ids, list) and len(comp_ids) > 0 and isinstance(comp_ids[0], list):
            comp_ids = comp_ids[0]

        return cast(list[int], comp_ids if isinstance(comp_ids, list) else [])
    except (AttributeError, IndexError, TypeError, KeyError) as e:
        logger.debug("Failed to tokenize completion: %s", e)
        return fallback


def _clamp(value: float | int, min_val: float | int, max_val: float | int) -> float | int:
    """Clamp *value* to [min_val, max_val]."""
    return max(min_val, min(value, max_val))


# Fields that MUST be present in checkpoint metadata's generation_params dict.
_REQUIRED_GENERATION_FIELDS = ("max_tokens", "temperature", "top_p", "top_k", "repetition_penalty")


@dataclass
class GenerationParams:
    """Text generation parameters passed to backends.

    Supports per-sample deterministic generation via seeds when the backend
    implementation allows it (HF via torch.Generator list; vLLM via seed field).
    """

    max_new_tokens: int = MAX_NEW_TOKENS
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int | None = 20
    repetition_penalty: float | None = 1.1
    trim_right_padding: bool = False

    @classmethod
    def from_checkpoint_metadata(cls, meta: dict[str, Any]) -> GenerationParams:
        """Build GenerationParams from checkpoint metadata dict.

        All generation fields must be present in *meta*. Raises ValueError if
        any required field is missing so the miner fails fast rather than
        silently generating with wrong parameters.

        Values are clamped to safe ranges to prevent obviously broken configs
        from reaching the backend.
        """
        missing = [f for f in _REQUIRED_GENERATION_FIELDS if f not in meta]
        if missing:
            raise ValueError(
                f"Checkpoint metadata missing required generation params: {', '.join(missing)}. "
                f"Available keys: {list(meta.keys())}. "
                f"The trainer must include all of: {list(_REQUIRED_GENERATION_FIELDS)}."
            )

        return cls(
            max_new_tokens=int(_clamp(meta["max_tokens"], 1, 16384)),
            temperature=float(_clamp(meta["temperature"], 0.01, 2.0)),
            top_p=float(_clamp(meta["top_p"], 0.0, 1.0)),
            top_k=int(_clamp(meta["top_k"], 0, 1000)) or None,
            repetition_penalty=float(_clamp(meta["repetition_penalty"], 1.0, 2.0)),
        )


class TextGenBackend(ABC):
    """Abstract interface for batched text generation backends.

    All backends must return tuples: (tokens, chosen_logprobs_or_none).
    The second element may be None when logprobs are not requested or unsupported.
    """

    _max_model_len: int | None = None

    def _cap_max_tokens(self, prompt_len: int, requested: int) -> int:
        """Cap *requested* max_new_tokens so prompt + completion fits in context.

        Returns the effective max_new_tokens. Logs a warning when capping occurs
        because the capped completion may fail validator checks (termination,
        token distribution) that expect the full max_tokens from checkpoint
        metadata.
        """
        if self._max_model_len is None:
            return requested
        headroom = self._max_model_len - prompt_len
        if headroom <= 0:
            logger.warning(
                "Prompt length (%d) exceeds max_model_len (%d), no room for "
                "completion. This request will produce 0 tokens and likely "
                "fail validation. Increase GRAIL_PIPELINE_MAX_MODEL_LEN.",
                prompt_len,
                self._max_model_len,
            )
            return 0
        if requested > headroom:
            logger.warning(
                "Capping max_new_tokens %d -> %d (prompt_len=%d, "
                "max_model_len=%d). The shorter completion may fail validator "
                "checks. Increase GRAIL_PIPELINE_MAX_MODEL_LEN to at least %d.",
                requested,
                headroom,
                prompt_len,
                self._max_model_len,
                prompt_len + requested,
            )
            return headroom
        return requested

    @abstractmethod
    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate completions for a batch of prompts.

        Args:
            prompt_ids_batch: Batch of tokenized prompts
            params: Generation parameters (temperature, top_p, etc)
            seeds: Optional per-sample seeds for deterministic sampling

        Returns:
            List of tuples per prompt: (full_sequence, chosen_logprobs_or_none).
            - full_sequence: Token IDs for prompt + completion
            - chosen_logprobs_or_none: List of logprobs for chosen completion tokens,
              or None if not requested/supported

        Backends must left-pad internally and remove left padding before
        returning, so returned sequences are aligned as [prompt + completion].
        If params.trim_right_padding is True, they should trim any trailing pad
        tokens in the completion region as appropriate for the tokenizer.
        """
        ...
