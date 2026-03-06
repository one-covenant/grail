"""SGLang server (OpenAI-compatible) generation backend."""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import GenerationParams, TextGenBackend, _decode_prompts, _tokenize_completion

logger = logging.getLogger(__name__)


class SGLangServerBackend(TextGenBackend):
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
