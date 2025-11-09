from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.gsm8k_env import GSM8KEnv
from grail.environments.loop import AgentEnvLoop, SGLangServerBackend, VLLMServerBackend
from grail.environments.sat_env import SATEnv
from grail.shared.constants import ROLLOUTS_PER_PROBLEM, TRAINER_MAX_LENGTH
from grail.trainer.algorithms.grpo import GRPOGroup, GRPORollout

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutGenConfig:
    backend: str  # "sglang_server" | "vllm_server"
    base_url: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float | None
    # Optional/defaulted fields must come after non-default fields
    model_name: str | None = None  # Served model name for OpenAI-compatible API
    rollouts_per_problem: int = ROLLOUTS_PER_PROBLEM
    return_logprobs: bool = True  # Enable behavior policy logprobs for IS
    environment: str = "sat"  # "sat" | "gsm8k"


class OfflineRolloutGenerator:
    """Server-backed rollout generator that produces GRPO groups.

    Uses AgentEnvLoop only for prompt rendering and server-backed generation.
    No proofs or wallet interactions are used.
    """

    def __init__(
        self, *, tokenizer: Any, config: RolloutGenConfig, hf_model: Any | None = None
    ) -> None:
        self._tokenizer = tokenizer
        self._cfg = config
        self._hf_model = hf_model  # Optional: for computing logprobs from HF instead of vLLM

        # Choose server backend
        backend_name = (self._cfg.backend or "sglang_server").lower()
        return_logprobs = bool(getattr(self._cfg, "return_logprobs", True))
        served_model_name = (
            self._cfg.model_name
            if getattr(self._cfg, "model_name", None)
            else getattr(tokenizer, "name_or_path", "model")
        )

        logger.info(
            "Initializing rollout generator",
            extra={
                "backend": backend_name,
                "base_url": self._cfg.base_url,
                "model_name": served_model_name,
                "rollouts_per_problem": self._cfg.rollouts_per_problem,
                "environment": self._cfg.environment,
                "return_logprobs": return_logprobs,
            },
        )

        if backend_name == "sglang_server":
            gen_backend = SGLangServerBackend(
                base_url=self._cfg.base_url,
                model_name=served_model_name,
                tokenizer=tokenizer,
                timeout=300.0,
                return_chosen_logprobs=return_logprobs,
            )
        elif backend_name == "vllm_server":
            # OpenAI-compatible vLLM server
            gen_backend = VLLMServerBackend(
                base_url=self._cfg.base_url,
                model_name=served_model_name,
                tokenizer=tokenizer,
                timeout=300.0,
                return_chosen_logprobs=return_logprobs,
            )
        else:
            logger.error("Unsupported generation backend", extra={"backend": self._cfg.backend})
            raise ValueError(f"Unsupported generation backend: {self._cfg.backend}")

        # model=None (server-backed); device string irrelevant for server calls
        self._loop = AgentEnvLoop(
            model=None,
            tokenizer=tokenizer,
            device="cpu",
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            do_sample=True,
            top_p=self._cfg.top_p,
            top_k=self._cfg.top_k,
            repetition_penalty=self._cfg.repetition_penalty,
            gen_backend=gen_backend,
        )

    def _create_env(self) -> MultiTurnEnv:
        """Factory method to create environment based on config."""
        env_type = getattr(self._cfg, "environment", "sat").lower()
        if env_type == "gsm8k":
            return GSM8KEnv()
        elif env_type == "sat":
            return SATEnv()
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")

    def _compute_hf_logprobs(
        self, token_ids: list[int], prompt_len: int, completion_len: int
    ) -> list[float] | None:
        """Compute logprobs using HuggingFace model for completion tokens.

        This provides ground-truth logprobs (before any vLLM sampling transforms).

        Args:
            token_ids: Full sequence (prompt + completion) token IDs
            prompt_len: Length of prompt portion
            completion_len: Length of completion portion

        Returns:
            List of logprobs for completion tokens, or None if HF model unavailable
        """
        if self._hf_model is None:
            return None

        try:
            # Prepare input tensor
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._hf_model.device)

            # Forward pass (no gradients)
            with torch.inference_mode():
                outputs = self._hf_model(input_ids)
                # Cast logits to float32 for precise log_softmax computation
                # This ensures numerical precision even when model is in bfloat16
                logits = outputs.logits[0].float()  # [seq_len, vocab_size] in float32

            # Extract logprobs for completion tokens
            # logits[i] predicts token[i+1], so for completion starting at prompt_len:
            # we extract logits[prompt_len-1:prompt_len-1+completion_len]
            completion_logprobs = []
            for i in range(completion_len):
                logit_pos = prompt_len - 1 + i
                if logit_pos < logits.shape[0]:
                    token_id = token_ids[prompt_len + i]
                    # log_softmax in float32 for maximum precision
                    log_probs_dist = F.log_softmax(logits[logit_pos], dim=-1)
                    completion_logprobs.append(log_probs_dist[token_id].item())
                else:
                    # Out of bounds - shouldn't happen, but handle gracefully
                    completion_logprobs.append(0.0)

            return completion_logprobs
        except Exception as e:
            logger.warning("Failed to compute HF logprobs: %s", e)
            return None

    async def generate_groups(
        self,
        seeds: Iterable[int],
        *,
        batch_size: int = 4,
    ) -> list[GRPOGroup]:
        """Generate GRPO groups for the provided problem seeds.

        Each seed yields exactly `rollouts_per_problem` rollouts with deterministic
        per-replicate random seeds passed to the server backend. Seeds are processed
        in batches to maximize generation efficiency.

        Args:
            seeds: Iterable of problem seeds
            batch_size: Number of problem seeds to process together (e.g., 4 seeds × 16 rollouts = 64 total)
        """
        seeds_list: list[int] = list(seeds)
        logger.info(
            "Starting rollout generation",
            extra={
                "num_seeds": len(seeds_list),
                "rollouts_per_problem": self._cfg.rollouts_per_problem,
                "generation_batch_size": batch_size,
                "rollouts_per_batch": batch_size * self._cfg.rollouts_per_problem,
            },
        )

        groups: list[GRPOGroup] = []
        rollouts_per_problem: int = int(self._cfg.rollouts_per_problem)

        # Process seeds in batches for efficient generation
        for batch_idx in range(0, len(seeds_list), batch_size):
            batch_seeds: list[int] = seeds_list[batch_idx : batch_idx + batch_size]
            logger.debug(
                "Processing seed batch",
                extra={"batch_index": batch_idx // batch_size, "batch_size": len(batch_seeds)},
            )

            # Step 1: Render prompts for all seeds in batch
            batch_prompts: list[list[dict[str, str]]] = []
            seed_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) for each seed's rollouts

            for seed in batch_seeds:
                env_for_prompt: MultiTurnEnv = self._create_env()
                obs = env_for_prompt.reset(task_id=str(seed), seed=int(seed))
                prompts: list[list[dict[str, str]]] = [
                    [{"role": m.role, "content": m.content} for m in obs.messages]
                ]
                # Replicate prompt list per rollout
                prompts_list: list[list[dict[str, str]]] = prompts * rollouts_per_problem
                start_idx: int = len(batch_prompts)
                batch_prompts.extend(prompts_list)
                seed_ranges.append((start_idx, start_idx + rollouts_per_problem))

            # Step 2: Render and generate all prompts in batch
            prompt_ids = self._loop.render_prompt_ids_batch(batch_prompts)
            logger.debug(
                "Batch prompts rendered",
                extra={"total_prompts": len(batch_prompts), "seeds_in_batch": len(batch_seeds)},
            )

            # Step 3: Create replicate seeds for all rollouts
            replicate_seeds: list[int] = []
            for seed in batch_seeds:
                seed_replicate_seeds: list[int] = [
                    int(seed) * 10_000 + i for i in range(rollouts_per_problem)
                ]
                replicate_seeds.extend(seed_replicate_seeds)

            include_logprobs: bool = bool(getattr(self._cfg, "return_logprobs", True))
            seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
                prompt_ids,
                seeds=replicate_seeds,
                trim_right_padding=True,
                include_logprobs=include_logprobs,
            )
            logger.debug(
                "Batch generation complete",
                extra={"total_sequences": len(seq_with_prompt_lens)},
            )

            # Step 4: Process results for each seed in batch
            for seed_idx, seed in enumerate(batch_seeds):
                start_idx, end_idx = seed_ranges[seed_idx]
                seed_results = seq_with_prompt_lens[start_idx:end_idx]

                # Decode and step environment for rewards
                rewards: list[float] = []
                successes: list[bool] = []
                seqs: list[list[int]] = []
                prompt_lens: list[int] = []
                comp_lens: list[int] = []
                all_logprobs: list[list[float] | None] = []

                for seq, prompt_len, chosen_logprobs in seed_results:
                    # Enforce max training length to avoid overflow
                    if TRAINER_MAX_LENGTH and len(seq) > int(TRAINER_MAX_LENGTH):
                        seq = seq[: int(TRAINER_MAX_LENGTH)]
                        # Also truncate logprobs if present
                        if chosen_logprobs is not None:
                            chosen_logprobs = chosen_logprobs[: int(TRAINER_MAX_LENGTH)]

                    completion_ids: list[int] = seq[prompt_len:]
                    completion_len: int = len(completion_ids)
                    text: str = self._tokenizer.decode(completion_ids, skip_special_tokens=False)

                    # Recreate env per replicate for clean single-turn stepping
                    env_r: MultiTurnEnv = self._create_env()
                    _ = env_r.reset(task_id=str(seed), seed=int(seed))
                    _obs_r, reward_r, _t, _u, info_r = env_r.step(
                        ChatMessage(role="assistant", content=text)
                    )

                    rewards.append(float(reward_r))
                    successes.append(bool(info_r.get("success", False)))
                    seqs.append(seq)
                    prompt_lens.append(int(prompt_len))
                    comp_lens.append(completion_len)

                    # Compute logprobs: prefer HF model for ground-truth, fallback to vLLM
                    final_logprobs: list[float] | None = None

                    if self._hf_model is not None:
                        # Use HF model to compute ground-truth logprobs
                        hf_logprobs = self._compute_hf_logprobs(seq, prompt_len, completion_len)
                        if hf_logprobs is not None:
                            final_logprobs = hf_logprobs

                            # Compare with vLLM logprobs if available
                            if chosen_logprobs is not None:
                                vllm_comp_lps = chosen_logprobs[:completion_len]
                                diffs = [
                                    abs(h - v)
                                    for h, v in zip(hf_logprobs, vllm_comp_lps, strict=False)
                                ]
                                mean_diff = np.mean(diffs) if diffs else 0.0
                                max_diff = np.max(diffs) if diffs else 0.0
                                min_diff = np.min(diffs) if diffs else 0.0
                                std_diff = np.std(diffs) if diffs else 0.0

                                # Compute log-ratio difference (sequence level)
                                hf_sum = sum(hf_logprobs)
                                vllm_sum = sum(vllm_comp_lps)
                                log_ratio_diff = abs(hf_sum - vllm_sum)

                                logger.info(
                                    "Logprob comparison HF vs vLLM: "
                                    "mean_diff=%.4f, max_diff=%.4f, min_diff=%.4f, std_diff=%.4f, "
                                    "seq_log_ratio_diff=%.4f, comp_len=%d",
                                    mean_diff,
                                    max_diff,
                                    min_diff,
                                    std_diff,
                                    log_ratio_diff,
                                    completion_len,
                                )

                                if mean_diff > 0.1:
                                    logger.warning(
                                        "⚠️  Large logprob mismatch detected! "
                                        "HF and vLLM compute logprobs from DIFFERENT distributions. "
                                        "mean_diff=%.4f > 0.1 threshold",
                                        mean_diff,
                                    )
                                elif mean_diff > 0.01:
                                    logger.warning(
                                        "Moderate logprob mismatch: mean_diff=%.4f "
                                        "(likely token ID differences, not distribution)",
                                        mean_diff,
                                    )
                                else:
                                    logger.debug(
                                        "✓ Logprobs match well: mean_diff=%.4f (token ID alignment OK)",
                                        mean_diff,
                                    )

                    # Fallback: use vLLM logprobs if HF not available
                    if final_logprobs is None and chosen_logprobs is not None:
                        final_logprobs = chosen_logprobs
                        logger.debug(
                            "Using vLLM logprobs (HF model unavailable), comp_len=%d",
                            completion_len,
                        )

                    # Store behavior policy logprobs for importance sampling
                    # Format: full sequence logprobs (prompt + completion)
                    if final_logprobs is not None:
                        # Pad with zeros for prompt tokens, keep completion logprobs
                        full_logprobs: list[float] = [0.0] * prompt_len + list(final_logprobs)
                        all_logprobs.append(full_logprobs)
                    else:
                        all_logprobs.append(None)

                # Compute GRPO advantages (zero-mean, variance-normalized)
                advantages: list[float] = self._compute_advantages(rewards)

                # Package into GRPOGroup
                group_id: str = str(seed)
                rollouts: list[GRPORollout] = []
                for i in range(rollouts_per_problem):
                    rollouts.append(
                        GRPORollout(
                            tokens=list(seqs[i]),
                            prompt_length=int(prompt_lens[i]),
                            completion_length=int(comp_lens[i]),
                            advantage=float(advantages[i]),
                            reward=float(rewards[i]),
                            success=bool(successes[i]),
                            nonce=int(i),
                            rollout_group=group_id,
                            token_logprobs=all_logprobs[i],
                        )
                    )
                groups.append(GRPOGroup(group_id=group_id, rollouts=rollouts))

                # Log stats for this group
                group_rewards: list[float] = [r.reward for r in rollouts]
                group_successes: list[bool] = [r.success for r in rollouts]
                logger.debug(
                    "Group generated",
                    extra={
                        "group_id": group_id,
                        "num_rollouts": len(rollouts),
                        "mean_reward": float(np.mean(group_rewards)),
                        "success_rate": float(np.mean(group_successes)),
                    },
                )

        logger.info(
            "Rollout generation complete",
            extra={
                "num_groups": len(groups),
                "total_rollouts": len(groups) * rollouts_per_problem,
            },
        )
        return groups

    @staticmethod
    def _compute_advantages(rewards: list[float]) -> list[float]:
        if not rewards:
            return []
        mean = float(np.mean(rewards))
        centered = [r - mean for r in rewards]
        var = float(np.mean([c * c for c in centered]))
        std = math.sqrt(var) if var > 0.0 else 1e-8
        return [c / std for c in centered]
