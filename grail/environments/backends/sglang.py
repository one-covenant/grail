"""SGLang server backend using native /generate endpoint.

Uses httpx.AsyncClient for concurrent, non-blocking requests to a running
SGLang server. Passes input_ids directly and receives output_ids, avoiding
all text re-tokenization issues (BPE mismatches, stop token stripping).

Reference: https://docs.sglang.io/basic_usage/native_api.html
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .base import GenerationParams, TextGenBackend

logger = logging.getLogger(__name__)


class SGLangServerBackend(TextGenBackend):
    """SGLang server backend via native /generate endpoint.

    Sends prompt token IDs directly and receives exact output token IDs
    (including stop tokens). No text round-trip, no re-tokenization.

    Reference: https://docs.sglang.io/basic_usage/native_api.html
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
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
            )
        return self._client

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        """Generate using SGLang native /generate endpoint.

        Passes input_ids directly and extracts output_ids from the response,
        avoiding text re-tokenization entirely.
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

        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _call_one_async(
            idx: int, prompt_ids: list[int], random_seed: int | None
        ) -> tuple[int, list[int]]:
            """POST to /generate with retries and backoff."""
            max_retries: int = 3
            base_backoff: float = 1.0

            async with sem:
                for attempt in range(max_retries):
                    req_start = time.time()
                    try:
                        sampling_params: dict[str, Any] = {
                            "max_new_tokens": int(params.max_new_tokens),
                            "temperature": float(params.temperature),
                            "top_p": float(params.top_p),
                        }
                        if params.top_k is not None:
                            sampling_params["top_k"] = int(params.top_k)
                        if params.repetition_penalty is not None:
                            sampling_params["repetition_penalty"] = float(params.repetition_penalty)
                        if random_seed is not None:
                            sampling_params["seed"] = int(random_seed)

                        body: dict[str, Any] = {
                            "input_ids": prompt_ids,
                            "sampling_params": sampling_params,
                            "return_token_ids": True,
                        }

                        resp = await self._get_client().post("/generate", json=body)
                        resp.raise_for_status()
                        data = resp.json()

                        output_ids = data.get("output_ids", [])
                        if not isinstance(output_ids, list):
                            logger.warning(
                                "  Request %d/%d: output_ids is not a list, using empty",
                                idx + 1,
                                batch_size,
                            )
                            output_ids = []

                        req_time = time.time() - req_start
                        logger.debug(
                            "  Request %d/%d took %.2fs, output_tokens=%d",
                            idx + 1,
                            batch_size,
                            req_time,
                            len(output_ids),
                        )
                        return (idx, output_ids)

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
                            return (idx, [])

                return (idx, [])

        tasks = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            seed_val = seeds[idx] if seeds and idx < len(seeds) else None
            tasks.append(_call_one_async(idx, p_ids, seed_val))

        results_tuples = await asyncio.gather(*tasks, return_exceptions=False)

        completion_ids_map: dict[int, list[int]] = {}
        for idx, output_ids in results_tuples:
            completion_ids_map[idx] = output_ids

        results: list[tuple[list[int], list[float] | None]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            comp_ids = completion_ids_map.get(idx, [])
            all_ids = p_ids + comp_ids
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
