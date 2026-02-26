"""Environment loop for GRPO rollout generation with GRAIL proofs.

Provides AgentEnvLoop class that:
  - Wraps model/tokenizer with sampling config
  - Runs single-turn episodes with logprob tracking
  - Generates GRAIL proof commitments
  - Supports both sequential and vectorized GRPO group generation
  - Returns GRPORollout dataclass compatible with mining packaging
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import random
import time
import traceback as _tb
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import numpy as np
import torch

from ..model.forward import forward_single_layer
from ..shared.chat_templates import apply_chat_template as _apply_chat_template
from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX, MAX_NEW_TOKENS
from ..shared.hf_compat import resolve_hidden_size
from .core import ChatMessage, MultiTurnEnv

logger = logging.getLogger(__name__)


def _shutdown_engine_and_free_gpu(engine_ref: Any | None, engine_name: str = "engine") -> None:
    """Shared GPU cleanup logic for inference engines (vLLM, SGLang, etc).

    Args:
        engine_ref: Engine instance to shutdown (will be set to None by caller)
        engine_name: Name for logging (e.g., "vLLM", "SGLang async")
    """
    import gc

    if engine_ref is None:
        return

    try:
        logger.info("Shutting down %s engine...", engine_name)

        # Try graceful shutdown if available
        if hasattr(engine_ref, "shutdown"):
            engine_ref.shutdown()

        # Force garbage collection to release resources
        gc.collect()

        # Synchronize GPU and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("%s engine shutdown complete", engine_name)
    except Exception as e:
        logger.warning("Error during %s shutdown: %s", engine_name, e)


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


def _set_global_seed(seed: int) -> None:
    """Seed all relevant RNGs for deterministic generation."""
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


class TextGenBackend(Protocol):
    """Abstract interface for batched text generation backends.

    All backends must return tuples: (tokens, chosen_logprobs_or_none).
    The second element may be None when logprobs are not requested or unsupported.
    """

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


class HFBackend:
    """HuggingFace generation backend using a provided model/tokenizer instance.

    This backend does not own the model; it reuses the instance passed in to
    maintain a single copy in memory when used for both generation and proofs.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        return_chosen_logprobs: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._return_chosen_logprobs = bool(return_chosen_logprobs)

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        batch_size = len(prompt_ids_batch)
        if batch_size == 0:
            return []

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        eos_id = self._tokenizer.eos_token_id

        # Left-pad inputs to max length
        max_len = max(len(p) for p in prompt_ids_batch)
        padded_inputs: list[list[int]] = []
        attention_masks: list[list[int]] = []
        left_pads: list[int] = []
        for p in prompt_ids_batch:
            pad_len = max_len - len(p)
            padded_inputs.append([pad_id] * pad_len + p)
            attention_masks.append([0] * pad_len + [1] * len(p))
            left_pads.append(pad_len)

        input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.max_new_tokens),
            "temperature": float(params.temperature),
            "do_sample": bool(params.do_sample),
            "top_p": float(params.top_p),
            "return_dict_in_generate": True,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if params.top_k is not None:
            gen_kwargs["top_k"] = int(params.top_k)
        if params.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(params.repetition_penalty)

        # Seeds are currently ignored to favor batched generation efficiency
        if seeds is not None and seeds:
            logger.debug("HFBackend: ignoring seeds for batched generation")

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        results: list[tuple[list[int], list[float] | None]] = []
        for b in range(batch_size):
            seq = outputs.sequences[b]
            # Remove left padding
            seq_wo_left = seq[left_pads[b] :]
            all_ids = seq_wo_left.tolist()

            if params.trim_right_padding:
                # Trim trailing padding conservatively
                if pad_id is not None and pad_id != eos_id:
                    # Find last non-pad index
                    last_non_pad = len(all_ids) - 1
                    while last_non_pad >= 0 and all_ids[last_non_pad] == pad_id:
                        last_non_pad -= 1
                    all_ids = all_ids[: max(0, last_non_pad + 1)]
            # HF backend doesn't currently support logprobs extraction
            results.append((all_ids, None))

        return results


class VLLMServerBackend:
    """vLLM server (OpenAI-compatible) backend over HTTP with async API.

    Uses AsyncOpenAI client to interact with a running vLLM server. Deterministic
    generation is achieved by passing a per-request seed when provided.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: Any,
        timeout: float = 300.0,
        max_concurrent_requests: int = 32,
        return_chosen_logprobs: bool = False,
        warn_on_missing_token_ids: bool = True,
        strict_token_ids: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)
        self._max_concurrent_requests = max_concurrent_requests
        self._return_chosen_logprobs = bool(return_chosen_logprobs)
        self._warn_on_missing_token_ids = bool(warn_on_missing_token_ids)
        self._strict_token_ids = bool(strict_token_ids)

        # Lazy client creation — defer to first use within the running event loop
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="EMPTY",
                timeout=self._timeout,
            )
        return self._client

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate completions using vLLM server async API.

        Now properly async - called directly from async context without nested event loops.
        """
        return await self._async_generate_batch(prompt_ids_batch, params, seeds)

    async def _async_generate_batch(
        self,
        prompt_ids_batch: list[list[int]],
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        import asyncio
        import time

        batch_start = time.time()
        batch_size = len(prompt_ids_batch)
        logger.info("vLLMServer: Starting ASYNC batch of %d prompts", batch_size)

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Use configurable semaphore to control client-side concurrency
        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int, prompt: str, rnd_seed: int | None
        ) -> tuple[int, str, list[float] | None, list[int] | None]:
            max_retries = 3
            base_backoff = 1.0
            async with sem:
                for attempt in range(max_retries):
                    req_start = time.time()
                    try:
                        # In strict_token_ids mode, send prompt as token IDs
                        prompt_value: Any = prompt
                        if self._strict_token_ids:
                            prompt_value = prompt_ids_batch[idx]

                        completion_kwargs: dict[str, Any] = {
                            "model": self._model_name,
                            "prompt": prompt_value,
                            "max_tokens": int(params.max_new_tokens),
                            "temperature": float(params.temperature),
                            "top_p": float(params.top_p),
                            # Ensure single completion per request
                            "n": 1,
                        }
                        # Provide vendor extensions via extra_body for vLLM
                        extra_body: dict[str, Any] = {}
                        if params.top_k is not None:
                            extra_body["top_k"] = int(params.top_k)
                        if params.repetition_penalty is not None:
                            extra_body["repetition_penalty"] = float(params.repetition_penalty)

                        # CRITICAL: Request token IDs to avoid re-tokenization mismatch
                        # This ensures the token IDs we use match the logprobs from vLLM
                        # Note: vLLM 0.10.2+ returns both text AND token_ids when this is set
                        if self._return_chosen_logprobs or self._strict_token_ids:
                            extra_body["return_token_ids"] = True
                            # Ensure special tokens are preserved for exact alignment
                            extra_body["skip_special_tokens"] = False
                            extra_body["spaces_between_special_tokens"] = False
                            # Include stop string to ensure logprobs length matches tokens
                            extra_body["include_stop_str_in_output"] = True

                        if extra_body:
                            completion_kwargs["extra_body"] = extra_body
                        if rnd_seed is not None:
                            completion_kwargs["seed"] = int(rnd_seed)
                        if self._return_chosen_logprobs:
                            # Request logprobs. We only store chosen-token logprobs; top alternatives are ignored.
                            completion_kwargs["logprobs"] = 1

                        response = await self._get_client().completions.create(**completion_kwargs)
                        text = response.choices[0].text if response.choices else ""
                        chosen_logprobs: list[float] | None = None
                        chosen_token_ids: list[int] | None = None
                        try:
                            # Extract logprobs from response
                            lp = getattr(response.choices[0], "logprobs", None)
                            if lp is not None and hasattr(lp, "token_logprobs"):
                                # OpenAI-compatible field; list of floats for chosen tokens
                                chosen_logprobs = list(lp.token_logprobs or [])

                            # Extract token IDs from response (vLLM 0.10.2+)
                            # When return_token_ids=True, vLLM returns token_ids in the choice
                            choice = response.choices[0]
                            if hasattr(choice, "token_ids") and choice.token_ids is not None:
                                # Direct token_ids field (vLLM 0.10.2+)
                                chosen_token_ids = list(choice.token_ids)
                            elif lp is not None and hasattr(lp, "tokens"):
                                # Fallback: try to extract from logprobs.tokens field
                                tokens_field = lp.tokens
                                if tokens_field:
                                    try:
                                        # tokens might be integers or strings
                                        parsed_ids: list[int] = []
                                        valid = True
                                        for t in tokens_field:
                                            if isinstance(t, int):
                                                parsed_ids.append(t)
                                            elif isinstance(t, str) and t.isdigit():
                                                parsed_ids.append(int(t))
                                            else:
                                                valid = False
                                                break
                                        chosen_token_ids = parsed_ids if valid else None
                                    except (ValueError, TypeError):
                                        chosen_token_ids = None
                        except Exception as e:
                            logger.debug("Failed to extract logprobs/token_ids: %s", e)
                            chosen_logprobs = None
                            chosen_token_ids = None
                        _ = time.time() - req_start
                        return (idx, text, chosen_logprobs, chosen_token_ids)
                    except Exception as e:
                        if attempt < max_retries - 1:
                            backoff = base_backoff * (2**attempt)
                            logger.warning(
                                "  vLLMServer req %d failed (attempt %d/%d), retrying in %.1fs: %s",
                                idx + 1,
                                attempt + 1,
                                max_retries,
                                backoff,
                                type(e).__name__,
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.warning(
                                "  vLLMServer req %d failed after %d attempts: %s",
                                idx + 1,
                                max_retries,
                                type(e).__name__,
                            )
                            return (idx, "", None, None)
            return (idx, "", None, None)

        tasks = []
        for idx, prompt in enumerate(prompts):
            seed_val = seeds[idx] if seeds and idx < len(seeds) else None
            tasks.append(_call_one_async(idx, prompt, seed_val))

        results_tuples = await asyncio.gather(*tasks, return_exceptions=False)
        completions: dict[int, str] = {}
        chosen_lp_map: dict[int, list[float] | None] = {}
        chosen_token_ids_map: dict[int, list[int] | None] = {}
        for idx, text, chosen_lp, chosen_tok_ids in results_tuples:
            completions[idx] = text
            chosen_lp_map[idx] = chosen_lp
            chosen_token_ids_map[idx] = chosen_tok_ids

        results: list[tuple[list[int], list[float] | None]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            completion_text = completions.get(idx, "")
            vllm_token_ids = chosen_token_ids_map.get(idx)

            # CRITICAL FIX: Use actual token IDs from vLLM if available
            # This ensures token IDs match the logprobs returned by vLLM
            if vllm_token_ids is not None:
                comp_ids = vllm_token_ids
                logger.debug(
                    "Using vLLM token IDs directly (count=%d) to avoid re-tokenization mismatch",
                    len(comp_ids),
                )
            else:
                if self._strict_token_ids:
                    raise RuntimeError(
                        "vLLM did not return token IDs but strict_token_ids is enabled. "
                        "Ensure vLLM version supports return_token_ids (>=0.10.2)."
                    )
                # Fallback: re-tokenize (may cause logprob mismatch)
                comp_ids = _tokenize_completion(self._tokenizer, completion_text, [])
                if self._warn_on_missing_token_ids:
                    logger.warning(
                        "vLLM did not return token IDs; falling back to re-tokenization. "
                        "This may cause importance sampling ratio mismatch!"
                    )

            all_ids = p_ids + comp_ids
            chosen_lp = chosen_lp_map.get(idx)
            results.append((all_ids, chosen_lp))

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "vLLMServer: ASYNC batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
            batch_time,
            batch_size,
            throughput,
        )
        return results


class SGLangServerBackend:
    """sgLang server (OpenAI-compatible) backend over HTTP with async API.

    Uses AsyncOpenAI client for concurrent, non-blocking requests to a running
    SGLang server. Runs in a separate subprocess, avoiding Gloo socket corruption.

    Reference: https://docs.sglang.ai/basic_usage/openai_api.html
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        tokenizer: Any,
        timeout: float = 300.0,
        max_concurrent_requests: int = 4,
        return_chosen_logprobs: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)
        self._max_concurrent_requests = max_concurrent_requests
        self._return_chosen_logprobs = bool(return_chosen_logprobs)

        # Lazy client creation — defer to first use within the running event loop
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="EMPTY",
                timeout=self._timeout,
            )
        return self._client

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate using async OpenAI API to SGLang server.

        Now properly async - called directly from async context without nested event loops.
        """
        return await self._async_generate_batch(prompt_ids_batch, params, seeds)

    async def _async_generate_batch(
        self,
        prompt_ids_batch: list[list[int]],
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Async batch generation with concurrent requests and retry logic."""
        import asyncio

        batch_start = time.time()
        batch_size = len(prompt_ids_batch)
        logger.info("SGLangServer: Starting ASYNC batch of %d prompts", batch_size)

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Use configurable semaphore to control client-side concurrency
        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int, prompt: str, random_seed: int | None
        ) -> tuple[int, str]:
            """Make async OpenAI API request with retries and backoff."""
            max_retries: int = 3
            base_backoff: float = 1.0

            async with sem:  # Limit concurrency
                for attempt in range(max_retries):
                    req_start = time.time()
                    try:
                        # Build completion kwargs
                        completion_kwargs: dict[str, Any] = {
                            "model": self._model_name,
                            "prompt": prompt,
                            "max_tokens": int(params.max_new_tokens),
                            "temperature": float(params.temperature),
                            "top_p": float(params.top_p),
                            # Ensure single completion per request
                            "n": 1,
                        }
                        # Provide vendor extensions via extra_body for SGLang
                        extra_body: dict[str, Any] = {}
                        if params.top_k is not None:
                            extra_body["top_k"] = int(params.top_k)
                        if params.repetition_penalty is not None:
                            extra_body["repetition_penalty"] = float(params.repetition_penalty)

                        if extra_body:
                            completion_kwargs["extra_body"] = extra_body

                        # Add seed (SGLang supports this parameter)
                        if random_seed is not None:
                            completion_kwargs["seed"] = int(random_seed)

                        # Async call to server
                        response = await self._get_client().completions.create(**completion_kwargs)

                        req_time = time.time() - req_start
                        text = response.choices[0].text if response.choices else ""
                        logger.debug(
                            "  Request %d/%d took %.2fs, output_len=%d",
                            idx + 1,
                            batch_size,
                            req_time,
                            len(text),
                        )
                        return (idx, text)

                    except Exception as e:
                        req_time = time.time() - req_start
                        if attempt < max_retries - 1:
                            backoff = base_backoff * (2**attempt)
                            logger.warning(
                                "  Request %d/%d failed (attempt %d/%d), retrying in %.1fs: %s",
                                idx + 1,
                                batch_size,
                                attempt + 1,
                                max_retries,
                                backoff,
                                type(e).__name__,
                            )
                            await asyncio.sleep(backoff)
                        else:
                            logger.warning(
                                "  Request %d/%d failed after %d attempts (%.2fs): %s",
                                idx + 1,
                                batch_size,
                                max_retries,
                                req_time,
                                type(e).__name__,
                            )
                            return (idx, "")

                return (idx, "")

        # Execute all requests concurrently with gather
        tasks = []
        for idx, prompt in enumerate(prompts):
            seed_val = seeds[idx] if seeds and idx < len(seeds) else None
            tasks.append(_call_one_async(idx, prompt, seed_val))

        # Wait for all completions
        results_tuples = await asyncio.gather(*tasks, return_exceptions=False)

        # Build completion dict from results
        completions: dict[int, str] = {}
        for idx, text in results_tuples:
            completions[idx] = text

        # Reconstruct results in original order
        results: list[tuple[list[int], list[float] | None]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            completion_text = completions.get(idx, "")
            comp_ids = _tokenize_completion(self._tokenizer, completion_text, [])
            all_ids = p_ids + comp_ids
            # SGLang backend doesn't currently support logprobs extraction
            results.append((all_ids, None))

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "SGLangServer: ASYNC batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
            batch_time,
            batch_size,
            throughput,
        )

        return results


@dataclass
class GRPORollout:
    """Single rollout for GRPO training with GRAIL proof support."""

    tokens: list[int]
    token_logprobs: list[float]
    prompt_length: int
    completion_length: int
    reward: float
    advantage: float
    trajectory: list[tuple[Any, Any, float]]
    success: bool
    commitments: list[dict]
    signature: bytes
    beacon: dict
    proof_version: str


def compute_advantages(rewards: list[float]) -> list[float]:
    """GRPO advantages: zero-mean within group, variance-normalized."""
    n = len(rewards)
    if n == 0:
        return []
    mean_reward = sum(rewards) / n
    centered = [r - mean_reward for r in rewards]
    std = (sum(a * a for a in centered) / n) ** 0.5
    denom = max(std, 1e-8)
    return [a / denom for a in centered]


def _batched_forward_pass(
    model: Any,
    device: str,
    all_token_ids_batch: list[list[int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run sub-batched forward passes with right-padding.

    Uses ``forward_single_layer`` — the same function the validator uses —
    so miner and validator produce identical hidden states and logits for
    GRAIL proof generation and verification.

    On OOM the sub-batch size is halved automatically (down to 1) before
    propagating the error.

    Args:
        model: HuggingFace causal LM
        device: Device string (e.g. ``"cuda:1"``)
        all_token_ids_batch: Variable-length token sequences

    Returns:
        ``(per_seq_hidden, per_seq_logits)`` — lists of tensors, one per
        sequence.  hidden: ``[seq_len, hidden_dim]`` on *device*.
        logits: ``[seq_len, vocab]`` on CPU.
    """
    from ..shared.constants import PROOF_BATCH_SIZE

    batch_size = len(all_token_ids_batch)
    seq_lens = [len(seq) for seq in all_token_ids_batch]
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", 0)

    per_seq_hidden: list[torch.Tensor] = []
    per_seq_logits: list[torch.Tensor] = []

    sub_batch_size = PROOF_BATCH_SIZE
    pos = 0

    while pos < batch_size:
        sub_end = min(pos + sub_batch_size, batch_size)
        sub_seqs = all_token_ids_batch[pos:sub_end]
        sub_lens = seq_lens[pos:sub_end]
        sub_max = max(sub_lens)
        sub_bs = len(sub_seqs)

        try:
            # Right-pad variable-length sequences to sub-batch max length
            input_ids = torch.full((sub_bs, sub_max), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros(sub_bs, sub_max, dtype=torch.long, device=device)
            for i, (seq, slen) in enumerate(zip(sub_seqs, sub_lens, strict=True)):
                input_ids[i, :slen] = torch.tensor(seq, dtype=torch.long, device=device)
                attn_mask[i, :slen] = 1

            # Shared forward path with validator (use_cache=False, single-layer)
            with torch.inference_mode():
                h_layer, logits = forward_single_layer(model, input_ids, attn_mask, LAYER_INDEX)

            # Extract per-sequence results, trimming padding.
            # .clone() decouples from the batched tensor so del frees GPU memory.
            for i, slen in enumerate(sub_lens):
                per_seq_hidden.append(h_layer[i, :slen, :].clone())
                per_seq_logits.append(logits[i, :slen, :].detach().to("cpu"))

            del h_layer, logits, input_ids, attn_mask
            torch.cuda.empty_cache()
            pos = sub_end

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if (
                not isinstance(e, torch.cuda.OutOfMemoryError)
                and "out of memory" not in str(e).lower()
            ):
                raise
            if sub_batch_size <= 1:
                raise

            new_size = max(1, sub_batch_size // 2)
            logger.warning(
                "OOM at sub-batch %d, halving sub-batch size: %d -> %d",
                sub_batch_size,
                sub_batch_size,
                new_size,
            )
            sub_batch_size = new_size
            # Clear traceback frame locals — they hold references to
            # intermediate GPU tensors from the failed forward pass.
            if e.__traceback__ is not None:
                _tb.clear_frames(e.__traceback__)
            del e
            gc.collect()
            torch.cuda.empty_cache()
            # Do NOT advance pos — retry the same chunk with smaller sub-batch

    logger.info(
        "Batched forward pass: %d seqs, sub-batch %d->%d (max_len=%d, min_len=%d)",
        batch_size,
        PROOF_BATCH_SIZE,
        sub_batch_size,
        max(seq_lens),
        min(seq_lens),
    )
    return per_seq_hidden, per_seq_logits


def compute_proofs(
    model: Any,
    device: str,
    hidden_dim: int,
    all_token_ids_batch: list[list[int]],
    prompt_lens: list[int],
    randomness_hex: str,
    wallet: Any,
) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
    """Compute GRAIL commitments and logprobs.  Used by AgentEnvLoop and ProofWorker.

    Two-phase design:
      Phase 1 — Forward passes via ``_batched_forward_pass`` (which internally
        calls ``forward_single_layer``, the same function the validator uses).
        Sub-batch size is the constant PROOF_BATCH_SIZE (16).
        Falls back to sequential single-sequence passes on OOM.
      Phase 2 — Per-sequence commitment + logprob extraction from cached
        hidden states and logits.

    Args:
        model: The loaded PyTorch model
        device: Device string (e.g. "cuda:1")
        hidden_dim: Model hidden dimension
        all_token_ids_batch: List of full token sequences (prompt + completion)
        prompt_lens: List of prompt lengths corresponding to each sequence
        randomness_hex: Hex string for randomness beacon
        wallet: Bittensor wallet for signing commitments

    Returns:
        List of tuples: (commitments, logprobs, signature, beacon, proof_version)
        one per rollout in the batch.
    """
    batch_size = len(all_token_ids_batch)
    if batch_size == 0:
        return []

    from ..protocol.grail_verifier import GRAILVerifier

    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    # --- Phase 1: Forward passes (batched with OOM fallback) ---
    use_batched = True
    per_seq_hidden: list[torch.Tensor | None] = [None] * batch_size
    per_seq_logits: list[torch.Tensor | None] = [None] * batch_size

    try:
        hidden_list, logits_list = _batched_forward_pass(model, device, all_token_ids_batch)
        for i in range(batch_size):
            per_seq_hidden[i] = hidden_list[i]
            per_seq_logits[i] = logits_list[i]
        del hidden_list, logits_list
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        logger.warning(
            "Batched proof OOM even at sub-batch=1 (total=%d), falling back to sequential",
            batch_size,
        )
        # Free any partial results from earlier sub-batches before fallback
        per_seq_hidden = [None] * batch_size
        per_seq_logits = [None] * batch_size
        if e.__traceback__ is not None:
            _tb.clear_frames(e.__traceback__)
        del e
        gc.collect()
        torch.cuda.empty_cache()
        use_batched = False

    # --- Phase 2: Per-sequence commitment and logprob computation ---
    results: list[tuple[list[dict], list[float], bytes, dict, str]] = []

    for idx, all_token_ids in enumerate(all_token_ids_batch):
        prompt_len = prompt_lens[idx]

        if use_batched:
            h_layer = per_seq_hidden[idx]
            logits = per_seq_logits[idx]
            assert h_layer is not None and logits is not None
        else:
            # Sequential fallback: one sequence at a time using the same
            # forward_single_layer path for numerical consistency.
            if idx == 0:
                logger.debug(
                    "SEQUENTIAL FALLBACK: seq_len=%d prompt_len=%d",
                    len(all_token_ids),
                    prompt_len,
                )
            token_tensor = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(token_tensor)
            with torch.inference_mode():
                h_batch, logits_batch = forward_single_layer(
                    model, token_tensor, attn_mask, LAYER_INDEX
                )
                h_layer = h_batch[0]
                logits = logits_batch[0].detach().to("cpu")
            del token_tensor, attn_mask, h_batch, logits_batch

        if idx == 0:
            logger.debug(
                "PROOF COMPUTATION: seq_len=%d prompt_len=%d batched=%s "
                "tokens_first_4=%s tokens_last_4=%s",
                len(all_token_ids),
                prompt_len,
                use_batched,
                all_token_ids[:4],
                all_token_ids[-4:] if len(all_token_ids) >= 4 else all_token_ids,
            )

        # --- Vectorized commitment computation (all positions at once) ---
        commitments = verifier.create_commitments_batch(h_layer, r_vec)

        if idx == 0:
            for pos in [0, prompt_len - 1, prompt_len, len(all_token_ids) - 1]:
                if 0 <= pos < len(commitments):
                    commitment = commitments[pos]
                    logger.debug(
                        "MINER COMMITMENT pos=%d token_id=%d "
                        "sketch_hash=%s rank_hash=%s hidden_norm=%.6f",
                        pos,
                        all_token_ids[pos],
                        commitment.get("sketch_hash", "")[:16],
                        commitment.get("rank_hash", "")[:16],
                        float(h_layer[pos].norm().item()),
                    )

        # --- Vectorized logprob computation ---
        completion_ids = all_token_ids[prompt_len:]
        num_completion = len(completion_ids)
        logprobs: list[float] = []

        if num_completion > 0:
            start_logit = prompt_len - 1
            end_logit = start_logit + num_completion
            valid_start = max(0, start_logit)
            valid_end = min(logits.size(0), end_logit)

            if valid_start < valid_end:
                skip_front = valid_start - start_logit
                n_valid = valid_end - valid_start
                valid_token_ids = completion_ids[skip_front : skip_front + n_valid]
                token_tensor = torch.tensor(valid_token_ids, dtype=torch.long)

                # Chunked log_softmax to cap CPU memory (~600MB per chunk)
                LOGPROB_CHUNK = 512
                chunk_logprobs: list[float] = []
                for c_start in range(0, n_valid, LOGPROB_CHUNK):
                    c_end = min(c_start + LOGPROB_CHUNK, n_valid)
                    logit_slice = logits[valid_start + c_start : valid_start + c_end]
                    log_probs_chunk = torch.log_softmax(logit_slice.float(), dim=-1)
                    tok_slice = token_tensor[c_start:c_end]
                    selected = log_probs_chunk[torch.arange(c_end - c_start), tok_slice]
                    chunk_logprobs.extend(selected.tolist())

                logprobs = (
                    [float("-inf")] * skip_front
                    + chunk_logprobs
                    + [float("-inf")] * (num_completion - skip_front - n_valid)
                )
            else:
                logprobs = [float("-inf")] * num_completion
                logger.warning(
                    "All completion logit positions out of range: start=%d end=%d logits_size=%d",
                    start_logit,
                    end_logit,
                    logits.size(0),
                )

        commitment_data = json.dumps(commitments, sort_keys=True)
        commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
        if wallet is None:
            raise RuntimeError(
                "GRAIL proof generation requires bittensor wallet (unavailable in offline mode)"
            )
        signature = wallet.hotkey.sign(commitment_hash)

        beacon = {"randomness": randomness_hex}
        proof_version = GRAIL_PROOF_VERSION

        results.append((commitments, logprobs, signature, beacon, proof_version))

    logger.debug(
        "Completed proof computation for %d rollout(s) (batched=%s)",
        len(all_token_ids_batch),
        use_batched,
    )
    return results


def assemble_rollouts(
    batch_data: list[tuple[list[int], int, float, dict]],
    proof_results: list[tuple[list[dict], list[float], bytes, dict, str]],
) -> list[GRPORollout]:
    """Assemble GRPORollout objects from generation + proof results.

    Args:
        batch_data: List of (all_ids, prompt_len, reward, info) tuples
        proof_results: Matching list of (commitments, logprobs, signature, beacon, proof_version)

    Returns:
        List of GRPORollout objects (advantages not yet computed)
    """
    rollouts: list[GRPORollout] = []
    for (all_ids, prompt_len, reward, info), (
        commitments,
        logprobs,
        signature,
        beacon,
        proof_version,
    ) in zip(batch_data, proof_results, strict=False):
        completion_ids = all_ids[prompt_len:]
        action_val = info.get("assignment", [])
        trajectory = [(0, action_val, reward)]

        rollout = GRPORollout(
            tokens=all_ids,
            token_logprobs=[0.0] * prompt_len + logprobs,
            prompt_length=int(prompt_len),
            completion_length=int(len(completion_ids)),
            reward=reward,
            advantage=0.0,
            trajectory=trajectory,
            success=bool(info.get("success", False)),
            commitments=commitments,
            signature=signature,
            beacon=beacon,
            proof_version=proof_version,
        )
        logger.debug("Prompt length: %d", rollout.prompt_length)
        rollouts.append(rollout)
    return rollouts


class AgentEnvLoop:
    """Stateful episode driver for step-only environments.

    Handles prompt rendering, model generation with logprobs, GRAIL
    commitments, and GRPO advantage computation. Supports single and
    vectorized execution.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = 0.6,
        *,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int | None = 20,
        repetition_penalty: float | None = 1.1,
        gen_backend: TextGenBackend | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        self._gen_params = GenerationParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            trim_right_padding=False,
        )

        # Default backend: reuse the same HF model instance for generation
        # to avoid increasing memory usage when also computing proofs.
        self._backend: TextGenBackend = gen_backend or HFBackend(model, tokenizer, device)

        # Hidden dim is only needed for GRAIL proof computation (not for evaluation)
        # Lazy-resolve when needed; server backends don't require it
        self._hidden_dim: int | None = None
        if model is not None:
            self._hidden_dim = resolve_hidden_size(model)

        # Log tokenizer version information for debugging
        try:
            import tokenizers  # type: ignore
            import transformers

            logger.info(
                "MINER TOKENIZER INFO: transformers=%s, tokenizers=%s, name_or_path=%s",
                transformers.__version__,
                tokenizers.__version__,  # type: ignore[attr-defined]  # tokenizers has __version__
                getattr(tokenizer, "name_or_path", "unknown"),
            )
        except Exception as e:
            logger.debug("Failed to log tokenizer version info: %s", e)

    def generate_batch_for_eval(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        *,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Lightweight batch generation for evaluation (no GRAIL proofs/commitments).

        This is ~2x faster than run_grpo_group() as it skips expensive proof computation,
        wallet signing, and advantage calculation.

        Args:
            env_factory: Factory function to create environment instances
            count: Number of rollouts to generate
            batch_size: Number of rollouts to process per batch
            seed: Optional seed for environment reset

        Returns:
            List of (reward, success) tuples for each rollout
        """
        results: list[tuple[float, bool]] = []

        # Process in batches for efficient generation
        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_count = batch_end - batch_start

            # Create and reset batch of environments
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            # Collect prompts for batch
            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            # Batch generate tokens (no logprobs/commitments needed for eval)
            batch_results = asyncio.run(
                self._batch_generate_tokens(prompts_list, include_logprobs=False)
            )

            # Process each rollout in the batch
            for env, (all_ids, prompt_len, _chosen_lp) in zip(envs, batch_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                # Step environment to get reward
                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )

                results.append((float(reward), bool(info.get("success", False))))

        return results

    def generate_and_eval(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        *,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> list[tuple[list[int], int, float, dict]]:
        """Generate completions and evaluate them in environments.

        Separable first stage of run_grpo_group: creates envs, generates via
        backend, steps envs to obtain rewards. Does NOT compute proofs.

        Args:
            env_factory: Factory for environment instances
            count: Number of rollouts to generate
            batch_size: Batch size for generation
            seed: Optional seed for environment reset

        Returns:
            List of (all_ids, prompt_len, reward, info) tuples
        """
        all_batch_data: list[tuple[list[int], int, float, dict]] = []

        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_count = batch_end - batch_start

            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            batch_results = asyncio.run(
                self._batch_generate_tokens(prompts_list, include_logprobs=False)
            )

            for env, (all_ids, prompt_len, _chosen_lp) in zip(envs, batch_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )
                all_batch_data.append((all_ids, prompt_len, float(reward), info))

        return all_batch_data

    def run_grpo_group(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        randomness_hex: str,
        wallet: Any,  # bt.wallet, but optional in offline mode
        *,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> list[GRPORollout]:
        """Generate multiple rollouts for GRPO with proofs and compute advantages."""
        # Stage 1: Generate and evaluate
        batch_data = self.generate_and_eval(env_factory, count, batch_size=batch_size, seed=seed)

        # Stage 2: Compute proofs
        all_ids_batch = [data[0] for data in batch_data]
        prompt_lens = [data[1] for data in batch_data]

        proof_results = self._batch_compute_commitments_and_logprobs(
            all_ids_batch, prompt_lens, randomness_hex, wallet
        )

        # Stage 3: Assemble rollouts
        rollouts = assemble_rollouts(batch_data, proof_results)

        # Stage 4: Compute advantages
        advantages = self._compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages, strict=False):
            rollout.advantage = float(adv)

        return rollouts

    # ---------------------- Shared eval helpers ----------------------
    def render_prompt_ids_batch(self, messages_list: list[list[dict[str, str]]]) -> list[list[int]]:
        """Render a batch of chat messages to token IDs using the tokenizer's template."""
        results: list[list[int]] = []
        for messages in messages_list:
            _rendered, prompt_ids = self._render_chat(messages)
            results.append(prompt_ids)
        return results

    async def generate_from_prompt_ids_batch(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        seeds: list[int] | None = None,
        trim_right_padding: bool = False,
        include_logprobs: bool = False,
    ) -> list[tuple[list[int], int, list[float] | None]]:
        """Generate sequences for a batch of tokenized prompts.

        Args:
            prompt_ids_batch: Batch of tokenized prompts (already templated).
            seeds: Optional per-sample seeds for deterministic sampling.
            trim_right_padding: If True, trims trailing right padding from completions.
            include_logprobs: If True and supported by backend, returns chosen-token
                log probabilities (one per completion token) as the third tuple element.

        Returns:
            List of triples per sample: (all_token_ids, prompt_len, chosen_logprobs_or_none).
            - all_token_ids: Full prompt+completion token ids
            - prompt_len: Length of the prompt portion
            - chosen_logprobs_or_none: List of logprobs for chosen completion tokens, or None
              when not requested or unavailable
        """
        params = replace(self._gen_params, trim_right_padding=trim_right_padding)
        backend_results = await self._backend.generate(prompt_ids_batch, params=params, seeds=seeds)

        # Backend returns tuples of (sequence, chosen_logprobs_or_none)
        results: list[tuple[list[int], int, list[float] | None]] = []
        for (seq, chosen_lp), p_ids in zip(backend_results, prompt_ids_batch, strict=False):
            prompt_len = len(p_ids)
            # If include_logprobs is False, discard any logprobs the backend may have returned
            final_lp = chosen_lp if include_logprobs else None
            results.append((seq, prompt_len, final_lp))
        return results

    def _render_chat(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[int]]:
        rendered = _apply_chat_template(self.tokenizer, messages)
        toks = self.tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
        prompt_ids = toks.input_ids[0].tolist()

        return rendered, prompt_ids

    async def _generate_tokens(
        self,
        prompt_ids: list[int],
        *,
        include_logprobs: bool = False,
    ) -> tuple[list[int], int, list[float] | None]:
        """Generate completion tokens; optionally return chosen-token logprobs.

        This method returns tokens trimmed of right padding. When include_logprobs
        is True and the backend supports chosen-token logprob reporting (e.g.,
        vLLM OpenAI server with logprobs enabled), the third element contains a
        list of log probabilities for the chosen completion tokens; otherwise None.
        """
        # Delegate to batch method to avoid duplication
        results = await self.generate_from_prompt_ids_batch(
            [prompt_ids],
            seeds=None,
            trim_right_padding=True,
            include_logprobs=include_logprobs,
        )
        return results[0]

    async def _batch_generate_tokens(
        self,
        prompts_list: list[list[dict[str, str]]],
        *,
        include_logprobs: bool = False,
    ) -> list[tuple[list[int], int, list[float] | None]]:
        """Batch generate completion tokens; optionally include chosen-token logprobs.

        Uses left-padding to handle variable-length prompts efficiently. When
        include_logprobs is True and supported by the backend, returns a list of
        triples (all_ids, prompt_len, chosen_logprobs_or_none) per sample.
        """
        # Render all prompts to token IDs
        prompt_ids_list: list[list[int]] = []
        for prompts in prompts_list:
            _rendered, prompt_ids = self._render_chat(prompts)
            prompt_ids_list.append(prompt_ids)

        # Delegate to generate_from_prompt_ids_batch to avoid duplication
        return await self.generate_from_prompt_ids_batch(
            prompt_ids_list,
            seeds=None,
            trim_right_padding=True,
            include_logprobs=include_logprobs,
        )

    def _trim_right_padding(
        self,
        seq: torch.Tensor,
        prompt_len: int,
    ) -> tuple[list[int], int]:
        """Trim trailing padding from sequence, preserving EOS semantics.

        Args:
            seq: Full token sequence [prompt_len + completion]
            prompt_len: Length of prompt portion

        Returns:
            (trimmed_token_ids, effective_completion_len)
        """
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Extract completion portion
        completion = seq[prompt_len:]

        # Case 1: pad_id != eos_id (separate tokens)
        if pad_id is not None and pad_id != eos_id:
            # Trim trailing pad_id tokens
            eff_comp = int((completion != pad_id).sum().item())
        # Case 2: pad_id == eos_id or no pad_id (same token)
        else:
            # Keep up to and including first EOS in completion
            eos_hits = (completion == eos_id).nonzero(as_tuple=False)
            if eos_hits.numel() > 0:
                eff_comp = int(eos_hits[0].item()) + 1
            else:
                eff_comp = completion.size(0)

        # Trim and convert to list
        trimmed = seq[: prompt_len + eff_comp]
        return trimmed.tolist(), eff_comp

    def _batch_compute_commitments_and_logprobs(
        self,
        all_token_ids_batch: list[list[int]],
        prompt_lens: list[int],
        randomness_hex: str,
        wallet: Any,  # bt.wallet, but optional in offline mode
    ) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
        """Compute GRAIL commitments and token logprobs using unbatched forward passes.

        Delegates to the standalone ``compute_proofs()`` function so that
        both AgentEnvLoop and ProofWorker share the same implementation.
        """
        if self._hidden_dim is None:
            raise RuntimeError(
                "Cannot compute GRAIL proofs: hidden_dim not initialized. "
                "This likely means AgentEnvLoop was created with a "
                "server backend for evaluation only."
            )

        return compute_proofs(
            self.model,
            self.device,
            self._hidden_dim,
            all_token_ids_batch,
            prompt_lens,
            randomness_hex,
            wallet,
        )

    def _compute_advantages(self, rewards: list[float]) -> list[float]:
        """GRPO advantages: zero-mean within group, variance-normalized."""
        return compute_advantages(rewards)
