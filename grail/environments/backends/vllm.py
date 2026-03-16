"""vLLM server (OpenAI-compatible) generation backend."""

from __future__ import annotations

import logging
from typing import Any

from .base import GenerationParams, TextGenBackend, _decode_prompts, _tokenize_completion

logger = logging.getLogger(__name__)


class VLLMServerBackend(TextGenBackend):
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
        max_model_len: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)
        self._max_concurrent_requests = max_concurrent_requests
        self._return_chosen_logprobs = bool(return_chosen_logprobs)
        self._warn_on_missing_token_ids = bool(warn_on_missing_token_ids)
        self._strict_token_ids = bool(strict_token_ids)
        self._max_model_len = max_model_len

        # Lazy client creation -- defer to first use within the running event loop
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

        # Pre-compute effective max_tokens per prompt to stay within context window.
        effective_max_per_prompt = [
            self._cap_max_tokens(len(p_ids), params.max_new_tokens) for p_ids in prompt_ids_batch
        ]

        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int,
            prompt: str,
            max_tokens: int,
            rnd_seed: int | None,
        ) -> tuple[int, str, list[float] | None, list[int] | None]:
            if max_tokens <= 0:
                logger.warning(
                    "  vLLMServer req %d: skipping, prompt exceeds "
                    "max_model_len (%s). Will produce 0 tokens.",
                    idx + 1,
                    self._max_model_len,
                )
                return (idx, "", None, None)

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
                            "max_tokens": max_tokens,
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

        tasks = [
            _call_one_async(
                idx,
                prompt,
                effective_max_per_prompt[idx],
                seeds[idx] if seeds and idx < len(seeds) else None,
            )
            for idx, prompt in enumerate(prompts)
        ]

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
