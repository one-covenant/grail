"""Base classes and utilities for text generation backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast

from ...protocol.constants import MAX_NEW_TOKENS_PROTOCOL_CAP

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

    max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int | None = 20
    repetition_penalty: float | None = 1.1
    trim_right_padding: bool = False

    @classmethod
    def from_checkpoint_metadata(cls, generation_params: dict[str, Any]) -> GenerationParams:
        """Build a GenerationParams from trainer-published checkpoint metadata.

        The trainer commits the per-checkpoint sampling policy via
        ``generation_params``. The validator's ``TerminationValidator`` caps
        completion length at ``min(generation_params["max_tokens"],
        MAX_NEW_TOKENS_PROTOCOL_CAP)``, so the miner MUST drive its backend with the same
        value or rollouts hard-fail ``termination_valid``.

        Fail loud at the trust boundary if required fields are missing or
        invalid: silently falling back to ``MAX_NEW_TOKENS_PROTOCOL_CAP`` was the source
        of the ``Exceeds max tokens: 8192 > 2048`` rejections in production.

        Args:
            generation_params: Raw dict from
                ``CheckpointMetadata.generation_params``.

        Returns:
            Typed ``GenerationParams`` ready to hand to a backend.

            ``top_k=None`` and ``repetition_penalty=None`` are intentionally
            sentinel values meaning "do not constrain"; backends omit the
            corresponding sampling field so the server's own default applies.
            ``top_k=0`` from the trainer is normalised to ``None`` for the
            same reason (sampling APIs use ``0`` as "disabled").

        Raises:
            ProtocolViolationError: any field is missing, malformed, or out
                of range. The miner outer loop catches this and skips the
                window cleanly.
        """
        from ...protocol.errors import ProtocolViolationError

        if not isinstance(generation_params, dict) or "max_tokens" not in generation_params:
            raise ProtocolViolationError(
                "Checkpoint metadata is missing 'max_tokens' in generation_params; "
                "cannot drive miner without it."
            )

        try:
            raw_max = int(generation_params["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.max_tokens must be an int; "
                f"got {generation_params['max_tokens']!r}"
            ) from exc
        if raw_max <= 0:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.max_tokens must be > 0; got {raw_max}"
            )
        max_new = min(raw_max, MAX_NEW_TOKENS_PROTOCOL_CAP)

        # All sampling fields are part of the same trust boundary as
        # max_tokens. Each field is REQUIRED — silent defaults would let
        # a publisher bug ship a half-specified policy that the validator
        # would interpret differently. Each range matches the trainer
        # publisher in grail/trainer/checkpoint_publisher.py exactly.
        for required in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if required not in generation_params:
                raise ProtocolViolationError(
                    f"Checkpoint metadata generation_params is missing "
                    f"required field {required!r}; trainer publisher must "
                    f"emit all of {{max_tokens, temperature, top_p, top_k, "
                    f"repetition_penalty}}."
                )

        try:
            temperature = float(generation_params["temperature"])
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.temperature must be a "
                f"float; got {generation_params['temperature']!r}"
            ) from exc
        if not (0.01 <= temperature <= 2.0):
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.temperature must be in "
                f"[0.01, 2.0]; got {temperature}"
            )

        try:
            top_p = float(generation_params["top_p"])
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.top_p must be a float; "
                f"got {generation_params['top_p']!r}"
            ) from exc
        if not (0.0 <= top_p <= 1.0):
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.top_p must be in [0.0, 1.0]; got {top_p}"
            )

        top_k_raw = generation_params["top_k"]
        try:
            top_k_int = int(top_k_raw)
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.top_k must be an int; got {top_k_raw!r}"
            ) from exc
        if top_k_int < 0 or top_k_int > 1000:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.top_k must be in [0, 1000]; got {top_k_int}"
            )
        # top_k=0 is the sampling-API "disabled" sentinel; backends omit
        # the field entirely when top_k is None, which makes the server
        # use its full vocabulary (no top-k filter).
        top_k: int | None = top_k_int if top_k_int > 0 else None

        repetition_penalty_raw = generation_params["repetition_penalty"]
        try:
            repetition_penalty_val = float(repetition_penalty_raw)
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.repetition_penalty must "
                f"be a float; got {repetition_penalty_raw!r}"
            ) from exc
        if not (1.0 <= repetition_penalty_val <= 2.0):
            raise ProtocolViolationError(
                f"Checkpoint metadata generation_params.repetition_penalty must "
                f"be in [1.0, 2.0]; got {repetition_penalty_val}"
            )

        return cls(
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty_val,
        )


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
