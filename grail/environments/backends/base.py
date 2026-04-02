"""Base classes and utilities for text generation backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from ...protocol.constants import MAX_NEW_TOKENS

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


class TextGenBackend(ABC):
    """Abstract interface for batched text generation backends.

    All backends must return tuples: (tokens, chosen_logprobs_or_none).
    The second element may be None when logprobs are not requested or unsupported.
    """

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
