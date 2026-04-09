"""AgentEnvLoop: stateful episode driver for step-only environments."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ..protocol.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
from ..shared.chat_templates import apply_chat_template as _apply_chat_template
from ..shared.hf_compat import resolve_hidden_size
from .advantages import compute_advantages
from .backends.base import GenerationParams, TextGenBackend
from .backends.hf import HFBackend
from .core import ChatMessage, MultiTurnEnv
from .proofs import compute_proofs
from .rollout import GRPORollout, assemble_rollouts

logger = logging.getLogger(__name__)


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
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
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
