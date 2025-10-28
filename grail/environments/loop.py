"""Environment loop for GRPO rollout generation with GRAIL proofs.

Provides AgentEnvLoop class that:
  - Wraps model/tokenizer with sampling config
  - Runs single-turn episodes with logprob tracking
  - Generates GRAIL proof commitments
  - Supports both sequential and vectorized GRPO group generation
  - Returns GRPORollout dataclass compatible with mining packaging
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import bittensor as bt
import numpy as np
import torch

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
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.95
    top_k: int | None = 50
    repetition_penalty: float | None = 1.1
    trim_right_padding: bool = False


class TextGenBackend(Protocol):
    """Abstract interface for batched text generation backends."""

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        """Return full sequences (prompt + completion) for each prompt in batch.

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

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
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

        results: list[list[int]] = []
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
            results.append(all_ids)

        return results


class VLLMBackend:
    """vLLM generation backend.

    Expects an engine object with a `generate(prompts, **kwargs)` API that
    returns results per prompt, each with an `.outputs[0]` containing either
    `token_ids` (preferred) or `text`. We reconstruct full sequences by
    concatenating prompt token IDs with completion token IDs if available.
    """

    def __init__(self, engine: Any, tokenizer: Any) -> None:
        self._engine = engine
        self._tokenizer = tokenizer

    def shutdown(self) -> None:
        """Release vLLM engine resources and free GPU memory."""
        _shutdown_engine_and_free_gpu(self._engine, "vLLM")
        self._engine = None

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        from vllm import SamplingParams  # type: ignore

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Map shared params to vLLM SamplingParams
        sp_kwargs: dict[str, Any] = {
            "max_tokens": int(params.max_new_tokens),
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
        }
        if params.top_k is not None:
            sp_kwargs["top_k"] = int(params.top_k)
        if params.repetition_penalty is not None:
            sp_kwargs["repetition_penalty"] = float(params.repetition_penalty)

        # Create per-prompt or shared SamplingParams depending on seeds
        if seeds is not None and len(seeds) == len(prompts):
            params_list = [SamplingParams(**sp_kwargs, seed=int(s)) for s in seeds]
            results_raw = self._engine.generate(prompts, params_list)
        else:
            sp = SamplingParams(**sp_kwargs, seed=int(seeds[0]) if seeds else None)
            results_raw = self._engine.generate(prompts, sp)

        results: list[list[int]] = []
        for p_ids, out in zip(prompt_ids_batch, results_raw, strict=False):
            try:
                # vLLM returns completion token IDs directly
                completion_ids = cast(list[int], out.outputs[0].token_ids)
                results.append(p_ids + completion_ids)
            except (AttributeError, IndexError, TypeError):
                # Fallback: re-tokenize text output
                try:
                    text = cast(str, out.outputs[0].text)
                    comp_ids = _tokenize_completion(self._tokenizer, text, [])
                    results.append(p_ids + comp_ids)
                except (AttributeError, IndexError, TypeError, KeyError) as e:
                    logger.debug("Failed to extract vLLM completion: %s", e)
                    results.append(list(p_ids))

        return results


class SGLangBackend:
    """sgLang generation backend for high-throughput inference.

    Expects an engine object with a `generate` API compatible with sgLang's LLM.
    Constructs SamplingParams and generates completions efficiently.
    """

    def __init__(self, engine: Any, tokenizer: Any) -> None:
        self._engine = engine
        self._tokenizer = tokenizer

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Map shared params to sgLang sampling_params
        sp_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.max_new_tokens),
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
        }
        if params.top_k is not None:
            sp_kwargs["top_k"] = int(params.top_k)
        if params.repetition_penalty is not None:
            sp_kwargs["frequency_penalty"] = float(params.repetition_penalty)

        # Generate with per-prompt seeds if provided
        if seeds is not None and len(seeds) == len(prompts):
            results_raw = []
            for prompt, seed in zip(prompts, seeds, strict=True):
                sp = {**sp_kwargs, "random_seed": int(seed)}
                out = self._engine.generate([prompt], sampling_params=sp)
                results_raw.extend(out)
        else:
            sp = {**sp_kwargs, "random_seed": int(seeds[0])} if seeds else sp_kwargs
            results_raw = self._engine.generate(prompts, sampling_params=sp)

        results: list[list[int]] = []
        for p_ids, out in zip(prompt_ids_batch, results_raw, strict=False):
            # Extract completion text from SGLang output
            comp_text = out.text if hasattr(out, "text") and isinstance(out.text, str) else str(out)
            comp_ids = _tokenize_completion(self._tokenizer, comp_text, [])
            results.append(p_ids + comp_ids)

        return results


class SGLangAsyncBackend:
    """SGLang offline async engine backend (no HTTP server required).

    Uses SGLang's async Engine API for direct, in-process generation.
    This eliminates server process management complexity and timeouts.
    Reference: https://docs.sglang.ai/basic_usage/offline_engine_api.html
    """

    def __init__(self, *, model_path: str, tokenizer: Any, **engine_kwargs: Any) -> None:
        """Initialize async backend with SGLang engine.

        Args:
            model_path: Path to model (local or HuggingFace model ID)
            tokenizer: HuggingFace tokenizer for decoding completions
            **engine_kwargs: Additional args passed to sgl.Engine (e.g., dtype, tp_size)
        """
        import sglang as sgl

        self._tokenizer = tokenizer

        # Initialize SGLang offline engine
        logger.info(f"Initializing SGLang offline engine: {model_path}")
        self._engine = sgl.Engine(model_path=model_path, **engine_kwargs)
        logger.info("SGLang offline engine ready")

        # Create a dedicated background asyncio loop to avoid nested-loop deadlocks
        import asyncio as _asyncio
        import threading as _threading

        self._io_loop: _asyncio.AbstractEventLoop = _asyncio.new_event_loop()

        def _run_loop() -> None:
            _asyncio.set_event_loop(self._io_loop)
            self._io_loop.run_forever()

        self._io_thread: _threading.Thread = _threading.Thread(
            target=_run_loop,
            name="sglang-async-loop",
            daemon=True,
        )
        self._io_thread.start()

        # Submission semaphore to cap concurrent async_generate calls
        self._submit_sem: _threading.Semaphore = _threading.Semaphore(value=2)

    def shutdown(self) -> None:
        """Shutdown async backend, release memory, and stop private loop.

        Attempts a best-effort engine shutdown in a short-lived thread.
        Always drops references, runs GC, and clears CUDA cache. Finally,
        stops the private asyncio loop and joins its thread.
        """
        import threading as _threading

        try:
            # Attempt engine shutdown in a guarded thread (best effort)
            engine_ref = getattr(self, "_engine", None)
            self._engine = None

            def _do_shutdown() -> None:
                try:
                    if engine_ref is not None and hasattr(engine_ref, "shutdown"):
                        engine_ref.shutdown()
                except Exception:
                    pass

            t = _threading.Thread(target=_do_shutdown, name="sglang-engine-shutdown", daemon=True)
            t.start()
            t.join(timeout=10.0)

            # Force Python GC to reclaim any reachable objects
            import gc

            gc.collect()

            # Free CUDA cache to release unused GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("SGLangAsyncBackend.shutdown cleanup issue: %s", e)

        # Stop background event loop thread
        try:
            if hasattr(self, "_io_loop") and self._io_loop.is_running():
                self._io_loop.call_soon_threadsafe(self._io_loop.stop)
            if hasattr(self, "_io_thread"):
                self._io_thread.join(timeout=2.0)
        except Exception as e:
            logger.debug("SGLangAsync backend loop stop issue: %s", e)

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        """Generate via private loop to avoid nested event-loop deadlocks.

        Submits coroutine to a background event loop using a submission
        semaphore to bound concurrency, and enforces a generous timeout
        to fail fast on hangs.
        """
        import asyncio as _asyncio

        # Bounded submission to avoid overloading the engine
        acquired = self._submit_sem.acquire(timeout=60.0)
        if not acquired:
            logger.warning("SGLangAsync: submission semaphore acquire timed out; dropping request")
            return list(prompt_ids_batch)
        try:

            async def _with_timeout() -> list[list[int]]:
                return await _asyncio.wait_for(
                    self._async_generate(prompt_ids_batch, params, seeds),
                    timeout=300.0,
                )

            future = _asyncio.run_coroutine_threadsafe(_with_timeout(), self._io_loop)
            return future.result()
        finally:
            self._submit_sem.release()

    async def _async_generate(
        self,
        prompt_ids_batch: list[list[int]],
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        """Async generation implementation using SGLang engine."""
        batch_size = len(prompt_ids_batch)
        batch_start = time.time()
        logger.info("SGLangAsync: Starting batch of %d prompts", batch_size)

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        # Map sampling params to SGLang format
        sampling_params: dict[str, Any] = {
            "temperature": float(params.temperature),
            "top_p": float(params.top_p),
            "max_new_tokens": int(params.max_new_tokens),
        }
        if params.top_k is not None:
            sampling_params["top_k"] = int(params.top_k)

        # Generate completions
        try:
            outputs = await self._engine.async_generate(prompts, sampling_params)
        except Exception as e:
            logger.error("SGLang async generation failed: %s", e, exc_info=True)
            return prompt_ids_batch

        # Reconstruct full sequences
        results: list[list[int]] = []
        for p_ids, output in zip(prompt_ids_batch, outputs, strict=False):
            # Extract completion text
            if isinstance(output, dict) and "text" in output:
                comp_text = output["text"]
            elif hasattr(output, "text"):
                comp_text = output.text
            else:
                comp_text = str(output)

            comp_ids = _tokenize_completion(self._tokenizer, comp_text, [])
            results.append(p_ids + comp_ids if comp_ids else p_ids)

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "SGLangAsync: Batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
            batch_time,
            batch_size,
            throughput,
        )

        return results


class SGLangServerBackend:
    """sgLang server (OpenAI-compatible) backend over HTTP.

    Sends prompts to a running sgLang server and reconstructs full sequences
    by concatenating prompt token IDs with completion token IDs derived from
    the returned text.

    Uses the official OpenAI client library for robust communication.
    """

    def __init__(
        self, *, base_url: str, model_name: str, tokenizer: Any, timeout: float = 120.0
    ) -> None:
        import openai

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._tokenizer = tokenizer
        self._timeout = float(timeout)

        # Initialize OpenAI client pointing to SGLang server
        self._client = openai.Client(
            base_url=f"{self._base_url}/v1",
            api_key="EMPTY",  # SGLang doesn't require authentication
            timeout=self._timeout,
        )

    def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[list[int]]:
        batch_start = time.time()
        batch_size = len(prompt_ids_batch)
        logger.info("SGLangServer: Starting PARALLEL batch of %d prompts", batch_size)

        prompts = _decode_prompts(self._tokenizer, prompt_ids_batch)

        def _call_one(idx: int, prompt: str, random_seed: int | None) -> tuple[int, str]:
            """Make OpenAI API request with retries and exponential backoff."""
            req_start = time.time()
            max_retries: int = 3
            base_backoff: float = 1.0

            for attempt in range(max_retries):
                try:
                    # Build completion kwargs
                    completion_kwargs: dict[str, Any] = {
                        "model": self._model_name,
                        "prompt": prompt,
                        "max_tokens": int(params.max_new_tokens),
                        "temperature": float(params.temperature),
                        "top_p": float(params.top_p),
                    }

                    # Map repetition_penalty to frequency_penalty if provided
                    if params.repetition_penalty is not None:
                        completion_kwargs["frequency_penalty"] = float(params.repetition_penalty)

                    # Add seed (SGLang supports this parameter)
                    if random_seed is not None:
                        completion_kwargs["seed"] = int(random_seed)

                    # Call OpenAI client (points to SGLang server)
                    response = self._client.completions.create(**completion_kwargs)

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
                        backoff: float = base_backoff * (2**attempt)
                        logger.warning(
                            "  Request %d/%d failed (attempt %d/%d), retrying in %.1fs: %s: %s",
                            idx + 1,
                            batch_size,
                            attempt + 1,
                            max_retries,
                            backoff,
                            type(e).__name__,
                            str(e)[:100],
                        )
                        time.sleep(backoff)
                    else:
                        logger.warning(
                            "  Request %d/%d failed after %d attempts (%.2fs total): %s",
                            idx + 1,
                            batch_size,
                            max_retries,
                            req_time,
                            type(e).__name__,
                        )
                        return (idx, "")

            return (idx, "")

        # Parallel execution with ThreadPoolExecutor
        # Use conservative worker count to avoid overwhelming server
        max_workers: int = min(batch_size, 4)
        completions: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests concurrently
            futures = []
            for idx, prompt in enumerate(prompts):
                seed_val = None
                if seeds is not None and idx < len(seeds):
                    seed_val = int(seeds[idx])
                future = executor.submit(_call_one, idx, prompt, seed_val)
                futures.append(future)

            # Collect results as they complete
            for future in as_completed(futures):
                idx, completion_text = future.result()
                completions[idx] = completion_text

        # Reconstruct results in original order
        results: list[list[int]] = []
        for idx, p_ids in enumerate(prompt_ids_batch):
            completion_text = completions.get(idx, "")
            comp_ids = _tokenize_completion(self._tokenizer, completion_text, [])
            results.append(p_ids + comp_ids)

        batch_time = time.time() - batch_start
        throughput = batch_size / batch_time if batch_time > 0 else 0
        logger.info(
            "SGLangServer: PARALLEL batch complete in %.2fs (%d prompts, %.1f prompts/sec)",
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
        temperature: float = 0.7,
        batch_size: int = 1,
        *,
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int | None = 50,
        repetition_penalty: float | None = 1.1,
        gen_backend: TextGenBackend | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)

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
        if gen_backend is None and model is not None:
            self._hidden_dim = resolve_hidden_size(model)

        # Log tokenizer version information for debugging
        try:
            import tokenizers  # type: ignore
            import transformers

            logger.info(
                "MINER TOKENIZER INFO: transformers=%s, tokenizers=%s, name_or_path=%s",
                transformers.__version__,
                tokenizers.__version__,
                getattr(tokenizer, "name_or_path", "unknown"),
            )
        except Exception as e:
            logger.debug("Failed to log tokenizer version info: %s", e)

    def run_single_turn(
        self,
        env: MultiTurnEnv,
        randomness_hex: str,
        wallet: bt.wallet,
        *,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> GRPORollout:
        """Run one episode and return a GRPORollout with proof."""
        obs = env.reset(task_id=task_id, seed=seed)
        rendered, prompt_ids = self._render_chat(
            [{"role": m.role, "content": m.content} for m in obs.messages]
        )
        all_ids, prompt_len = self._generate_tokens(prompt_ids)
        completion_ids = all_ids[prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        next_obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )

        commitments, logprobs, signature, beacon, proof_version = (
            self._compute_commitments_and_logprobs(
                all_ids,
                prompt_len,
                randomness_hex,
                wallet,
            )
        )

        # Build trajectory for compatibility (step_idx, action, reward)
        # For single-turn, extract assignment from info if available
        action_val = info.get("assignment", [])
        trajectory = [(0, action_val, float(reward))]

        return GRPORollout(
            tokens=all_ids,
            token_logprobs=[0.0] * prompt_len + logprobs,
            prompt_length=int(prompt_len),
            completion_length=int(len(completion_ids)),
            reward=float(reward),
            advantage=0.0,
            trajectory=trajectory,
            success=bool(info.get("success", False)),
            commitments=commitments,
            signature=signature,
            beacon=beacon,
            proof_version=proof_version,
        )

    def generate_batch_for_eval(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        *,
        seed: int | None = None,
    ) -> list[tuple[float, bool]]:
        """Lightweight batch generation for evaluation (no GRAIL proofs/commitments).

        This is ~2x faster than run_grpo_group() as it skips expensive proof computation,
        wallet signing, and advantage calculation.

        Args:
            env_factory: Factory function to create environment instances
            count: Number of rollouts to generate
            seed: Optional seed for environment reset

        Returns:
            List of (reward, success) tuples for each rollout
        """
        results: list[tuple[float, bool]] = []

        # Process in batches for efficient generation
        for batch_start in range(0, count, self.batch_size):
            batch_end = min(batch_start + self.batch_size, count)
            batch_count = batch_end - batch_start

            # Create and reset batch of environments
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            # Collect prompts for batch
            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            # Batch generate tokens (no logprobs/commitments needed for eval)
            batch_results = self._batch_generate_tokens(prompts_list)

            # Process each rollout in the batch
            for env, (all_ids, prompt_len) in zip(envs, batch_results, strict=False):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                # Step environment to get reward
                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )

                results.append((float(reward), bool(info.get("success", False))))

        return results

    def run_grpo_group(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        randomness_hex: str,
        wallet: bt.wallet,
        *,
        seed: int | None = None,
    ) -> list[GRPORollout]:
        """Generate multiple rollouts for GRPO with proofs and compute advantages."""
        rollouts: list[GRPORollout] = []

        # Process in batches for efficient generation
        for batch_start in range(0, count, self.batch_size):
            batch_end = min(batch_start + self.batch_size, count)
            batch_count = batch_end - batch_start

            # Create and reset batch of environments
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]

            # Collect prompts for batch
            prompts_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]

            # Batch generate tokens (logprobs computed later with commitments)
            batch_results = self._batch_generate_tokens(prompts_list)

            # Process each rollout in the batch
            for env, _obs, (all_ids, prompt_len) in zip(
                envs, obs_list, batch_results, strict=False
            ):
                completion_ids = all_ids[prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

                next_obs, reward, terminated, truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )

                commitments, logprobs, signature, beacon, proof_version = (
                    self._compute_commitments_and_logprobs(
                        all_ids,
                        prompt_len,
                        randomness_hex,
                        wallet,
                    )
                )

                action_val = info.get("assignment", [])
                trajectory = [(0, action_val, float(reward))]

                rollout = GRPORollout(
                    tokens=all_ids,
                    token_logprobs=[0.0] * prompt_len + logprobs,
                    prompt_length=int(prompt_len),
                    completion_length=int(len(completion_ids)),
                    reward=float(reward),
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

        advantages = self._compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages, strict=False):
            rollout.advantage = float(adv)

        return rollouts

    def run_grpo_group_vec(
        self,
        envs: list[MultiTurnEnv],
        randomness_hex: str,
        wallet: bt.wallet,
    ) -> list[GRPORollout]:
        """Vectorized GRPO group generation (batched prompts and generation).

        Uses sequential fallback through run_single_turn, which now efficiently
        computes logprobs and commitments in a single forward pass.
        """
        rollouts: list[GRPORollout] = []
        for env in envs:
            rollout = self.run_single_turn(env, randomness_hex, wallet)
            rollouts.append(rollout)

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

    def generate_from_prompt_ids_batch(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        seeds: list[int] | None = None,
        trim_right_padding: bool = False,
    ) -> list[tuple[list[int], int]]:
        """Generate sequences for a batch of tokenized prompts.

        Returns list of (all_token_ids, prompt_len) pairs.
        """
        params = replace(self._gen_params, trim_right_padding=trim_right_padding)
        sequences = self._backend.generate(prompt_ids_batch, params=params, seeds=seeds)
        results: list[tuple[list[int], int]] = []
        for p_ids, seq in zip(prompt_ids_batch, sequences, strict=False):
            results.append((seq, len(p_ids)))
        return results

    def _render_chat(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[int]]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        toks = self.tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
        prompt_ids = toks.input_ids[0].tolist()

        return rendered, prompt_ids

    def _generate_tokens(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[int], int]:
        """Generate completion tokens without computing logprobs.

        Logprobs will be computed in a single forward pass with commitments.
        Returns tokens trimmed of right padding.
        """
        batch = [prompt_ids]
        params = replace(self._gen_params, trim_right_padding=True)
        sequences = self._backend.generate(batch, params=params, seeds=None)
        # Left padding is removed by backend, so prompt_len is original length
        return sequences[0], len(prompt_ids)

    def _batch_generate_tokens(
        self,
        prompts_list: list[list[dict[str, str]]],
    ) -> list[tuple[list[int], int]]:
        """Batch generate completion tokens without computing logprobs.

        Uses left-padding to handle variable-length prompts efficiently.
        Logprobs will be computed in a single forward pass with commitments.
        """
        batch_size = len(prompts_list)

        # Fast path for single prompt
        if batch_size == 1:
            _, prompt_ids = self._render_chat(prompts_list[0])
            seq, prompt_len = self._generate_tokens(prompt_ids)
            return [(seq, prompt_len)]

        # Render all prompts and collect token IDs
        prompt_ids_list: list[list[int]] = []
        for prompts in prompts_list:
            _rendered, prompt_ids = self._render_chat(prompts)
            prompt_ids_list.append(prompt_ids)

        params = replace(self._gen_params, trim_right_padding=True)
        sequences = self._backend.generate(prompt_ids_list, params=params, seeds=None)

        results: list[tuple[list[int], int]] = []
        for p_ids, seq in zip(prompt_ids_list, sequences, strict=False):
            results.append((seq, len(p_ids)))
        return results

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

    def _compute_commitments_and_logprobs(
        self,
        all_token_ids: list[int],
        prompt_len: int,
        randomness_hex: str,
        wallet: bt.wallet,
    ) -> tuple[list[dict], list[float], bytes, dict, str]:
        """Compute GRAIL commitments and token logprobs in a single forward pass.

        This is more efficient than computing them separately, as it requires
        only one forward pass through the model.
        """
        if self._hidden_dim is None:
            raise RuntimeError(
                "Cannot compute GRAIL proofs: hidden_dim not initialized. "
                "This likely means AgentEnvLoop was created with a "
                "server backend for evaluation only."
            )

        from ..protocol.grail_verifier import GRAILVerifier

        verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        r_vec = verifier.generate_r_vec(randomness_hex)

        commitments: list[dict] = []
        logprobs: list[float] = []

        with torch.inference_mode():
            token_tensor = torch.tensor([all_token_ids], dtype=torch.long).to(self.device)
            model_outputs = self.model(token_tensor, output_hidden_states=True)

            # Extract hidden states for commitments
            h_layer = model_outputs.hidden_states[LAYER_INDEX][0]
            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    commitments.append(verifier.create_commitment(h_layer[pos], r_vec, pos))

            # Extract logits for logprobs computation
            # model_outputs.logits shape: [batch_size, seq_len, vocab_size]
            # We need logprobs for completion tokens (positions prompt_len onwards)
            logits = model_outputs.logits[0]  # [seq_len, vocab_size]
            completion_ids = all_token_ids[prompt_len:]

            for i, token_id in enumerate(completion_ids):
                # Position in logits is (prompt_len - 1 + i) because logits[i] predicts token[i+1]
                logit_pos = prompt_len - 1 + i
                if logit_pos < logits.size(0):
                    log_probs_dist = torch.log_softmax(logits[logit_pos], dim=-1)
                    logprobs.append(log_probs_dist[token_id].item())
                else:
                    logger.warning(
                        "Missing logits for completion token %d/%d; setting logprob to -inf",
                        i,
                        len(completion_ids),
                    )
                    logprobs.append(float("-inf"))

            # Debug: log first few logprobs to verify they're non-zero
            if logprobs:
                first_5_logprobs = logprobs[: min(5, len(logprobs))]
                logger.debug(
                    "MINER LOGPROBS: completion_len=%d first_5=%s",
                    len(logprobs),
                    [f"{lp:.6f}" for lp in first_5_logprobs],
                )

        commitment_data = json.dumps(commitments, sort_keys=True)
        commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
        signature = wallet.hotkey.sign(commitment_hash)

        beacon = {"randomness": randomness_hex}
        proof_version = GRAIL_PROOF_VERSION

        return commitments, logprobs, signature, beacon, proof_version

    def _compute_advantages(self, rewards: list[float]) -> list[float]:
        """GRPO advantages: zero-mean within group, variance-normalized."""
        n = len(rewards)
        if n == 0:
            return []
        mean_reward = sum(rewards) / n
        centered = [r - mean_reward for r in rewards]
        std = (sum(a * a for a in centered) / n) ** 0.5
        denom = max(std, 1e-8)
        return [a / denom for a in centered]
