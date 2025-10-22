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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import bittensor as bt
import torch

from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX, MAX_NEW_TOKENS
from ..shared.hf_compat import resolve_hidden_size
from .core import ChatMessage, MultiTurnEnv

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)

        self._hidden_dim = resolve_hidden_size(self.model)

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

        # Debug: log rendered prompt text for comparison
        logger.debug(
            "MINER RENDERED PROMPT: length=%d chars, tokens=%d\n%s",
            len(rendered),
            len(prompt_ids),
            rendered,
        )

        return rendered, prompt_ids

    def _generate_tokens(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[int], int]:
        """Generate completion tokens without computing logprobs.

        Logprobs will be computed in a single forward pass with commitments.
        Returns tokens trimmed of right padding.
        """
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        prompt_len = int(input_ids.shape[1])

        # Use proper pad_token_id; only fallback to eos if no pad token exists
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
            logger.debug("Using eos_token_id as pad_token_id fallback")

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                return_dict_in_generate=True,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Trim right padding before returning
        all_token_ids, _ = self._trim_right_padding(outputs.sequences[0], prompt_len)
        return all_token_ids, prompt_len

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
            return [self._generate_tokens(prompt_ids)]

        # Render all prompts and collect token IDs
        prompt_ids_list = []
        for prompts in prompts_list:
            _, prompt_ids = self._render_chat(prompts)
            prompt_ids_list.append(prompt_ids)

        # Store original prompt lengths (before padding)
        original_prompt_lens = [len(p) for p in prompt_ids_list]
        max_prompt_len = max(original_prompt_lens)

        # Left-pad all prompts to same length (standard for causal LM)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded_inputs = []
        attention_masks = []

        for prompt_ids in prompt_ids_list:
            pad_len = max_prompt_len - len(prompt_ids)
            padded = [pad_id] * pad_len + prompt_ids
            mask = [0] * pad_len + [1] * len(prompt_ids)
            padded_inputs.append(padded)
            attention_masks.append(mask)

        input_ids = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(self.device)

        # Batch generate (no logits output for efficiency)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                return_dict_in_generate=True,
                pad_token_id=pad_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Extract results for each sequence
        results = []
        for batch_idx in range(batch_size):
            original_prompt_len = original_prompt_lens[batch_idx]
            pad_len = max_prompt_len - original_prompt_len

            # Get full generated sequence (includes: padding + original_prompt + completion)
            full_seq = outputs.sequences[batch_idx]

            # Strip left padding to recover [original_prompt + completion]
            seq_without_left_pad = full_seq[pad_len:]

            # Trim right padding before returning
            all_token_ids, _ = self._trim_right_padding(seq_without_left_pad, original_prompt_len)

            results.append((all_token_ids, original_prompt_len))

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
