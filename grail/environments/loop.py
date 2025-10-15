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

from ..shared.chat_templates import build_qwen_chat_template
from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX, MAX_NEW_TOKENS
from ..shared.hf_compat import resolve_hidden_size
from ..shared.prompt_constants import REASONING_START, SYSTEM_PROMPT
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
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)

        self._hidden_dim = resolve_hidden_size(self.model)

        # Inject Qwen chat template
        # TODO: this should be removed later on and we need to make sure
        # we directly use the checkpoint's template shared by the trainer
        tpl = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        try:
            current_tpl = getattr(self.tokenizer, "chat_template", None)
            if current_tpl != tpl:
                logger.warning("MINER: Setting Qwen chat template (was different)")
                self.tokenizer.chat_template = tpl
            else:
                logger.debug("MINER: Chat template already matches Qwen template")
        except Exception:
            pass

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
        all_ids, logprobs, prompt_len = self._generate_with_logprobs(prompt_ids)
        completion_ids = all_ids[prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        next_obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion_text)
        )

        commitments, signature, beacon, proof_version = self._compute_commitments(
            all_ids,
            randomness_hex,
            wallet,
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

    def run_grpo_group(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        randomness_hex: str,
        wallet: bt.wallet,
        *,
        seed: int | None = None,
    ) -> list[GRPORollout]:
        """Generate multiple rollouts for GRPO and compute advantages."""
        rollouts: list[GRPORollout] = []
        for _ in range(count):
            env = env_factory()
            rollout = self.run_single_turn(env, randomness_hex, wallet, seed=seed)
            rollouts.append(rollout)

        advantages = self._compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages):
            rollout.advantage = float(adv)

        return rollouts

    def run_grpo_group_vec(
        self,
        envs: list[MultiTurnEnv],
        randomness_hex: str,
        wallet: bt.wallet,
    ) -> list[GRPORollout]:
        """Vectorized GRPO group generation (batched prompts and generation).

        Initial implementation: sequential fallback. Optimization with batched
        generate can be added later without API change.
        """
        rollouts: list[GRPORollout] = []
        for env in envs:
            rollout = self.run_single_turn(env, randomness_hex, wallet)
            rollouts.append(rollout)

        advantages = self._compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages):
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

    def _generate_with_logprobs(
        self,
        prompt_ids: list[int],
    ) -> tuple[list[int], list[float], int]:
        # Use pre-computed prompt_ids from _render_chat to ensure consistency
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        prompt_len = int(input_ids.shape[1])

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
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        all_token_ids = outputs.sequences[0].tolist()
        completion_ids = all_token_ids[prompt_len:]

        logprobs = []
        for i, token_id in enumerate(completion_ids):
            if i < len(outputs.scores):
                # Use log_softmax for numerical stability (matches validator)
                log_probs_dist = torch.log_softmax(outputs.scores[i][0], dim=-1)
                logprobs.append(log_probs_dist[token_id].item())
            else:
                # This should never happen in normal generation
                # If it does, it indicates a generation issue (e.g., early EOS)
                logger.warning(
                    "Missing score for completion token %d/%d; setting logprob to -inf",
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

        return all_token_ids, logprobs, prompt_len

    def _compute_commitments(
        self,
        all_token_ids: list[int],
        randomness_hex: str,
        wallet: bt.wallet,
    ) -> tuple[list[dict], bytes, dict, str]:
        from ..protocol.grail_verifier import GRAILVerifier

        verifier = GRAILVerifier(hidden_dim=self._hidden_dim)
        r_vec = verifier.generate_r_vec(randomness_hex)

        commitments: list[dict] = []
        with torch.inference_mode():
            token_tensor = torch.tensor([all_token_ids], dtype=torch.long).to(self.device)
            model_outputs = self.model(token_tensor, output_hidden_states=True)
            h_layer = model_outputs.hidden_states[LAYER_INDEX][0]
            for pos in range(len(all_token_ids)):
                if pos < h_layer.size(0):
                    commitments.append(verifier.create_commitment(h_layer[pos], r_vec, pos))

        commitment_data = json.dumps(commitments, sort_keys=True)
        commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
        signature = wallet.hotkey.sign(commitment_hash)

        beacon = {"randomness": randomness_hex}
        proof_version = GRAIL_PROOF_VERSION

        return commitments, signature, beacon, proof_version

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
