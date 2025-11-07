from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.loop import AgentEnvLoop, SGLangServerBackend, VLLMServerBackend
from grail.environments.gsm8k_env import GSM8KEnv
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
    rollouts_per_problem: int = ROLLOUTS_PER_PROBLEM
    return_logprobs: bool = True  # Enable behavior policy logprobs for IS
    environment: str = "sat"  # "sat" | "gsm8k"


class OfflineRolloutGenerator:
    """Server-backed rollout generator that produces GRPO groups.

    Uses AgentEnvLoop only for prompt rendering and server-backed generation.
    No proofs or wallet interactions are used.
    """

    def __init__(self, *, tokenizer: Any, config: RolloutGenConfig) -> None:
        self._tokenizer = tokenizer
        self._cfg = config

        # Choose server backend
        backend_name = (self._cfg.backend or "sglang_server").lower()
        return_logprobs = bool(getattr(self._cfg, "return_logprobs", True))
        
        logger.info(
            "Initializing rollout generator",
            extra={
                "backend": backend_name,
                "base_url": self._cfg.base_url,
                "rollouts_per_problem": self._cfg.rollouts_per_problem,
                "environment": self._cfg.environment,
                "return_logprobs": return_logprobs,
            },
        )
        
        if backend_name == "sglang_server":
            gen_backend = SGLangServerBackend(
                base_url=self._cfg.base_url,
                model_name=getattr(tokenizer, "name_or_path", "model"),
                tokenizer=tokenizer,
                timeout=300.0,
                return_chosen_logprobs=return_logprobs,
            )
        elif backend_name == "vllm_server":
            # OpenAI-compatible vLLM server
            gen_backend = VLLMServerBackend(
                base_url=self._cfg.base_url,
                model_name=getattr(tokenizer, "name_or_path", "model"),
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

    async def generate_groups(
        self,
        seeds: Iterable[int],
        *,
        batch_size: int = 1,
    ) -> list[GRPOGroup]:
        """Generate GRPO groups for the provided problem seeds.

        Each seed yields exactly `rollouts_per_problem` rollouts with deterministic
        per-replicate random seeds passed to the server backend.

        Args:
            seeds: Iterable of problem seeds
            batch_size: Number of rollouts to generate per batch
        """
        seeds_list = list(seeds)
        logger.info(
            "Starting rollout generation",
            extra={
                "num_seeds": len(seeds_list),
                "rollouts_per_problem": self._cfg.rollouts_per_problem,
                "batch_size": batch_size,
            },
        )
        
        groups: list[GRPOGroup] = []
        rollouts_per_problem = int(self._cfg.rollouts_per_problem)
        for idx, seed in enumerate(seeds_list):
            logger.debug("Generating rollouts for seed", extra={"seed": seed, "index": idx})
            # Create a temporary env to render the prompt deterministically for this seed
            env_for_prompt = self._create_env()
            obs = env_for_prompt.reset(task_id=str(seed), seed=int(seed))
            prompts = [[{"role": m.role, "content": m.content} for m in obs.messages]]
            # Replicate prompt list per rollout
            prompts_list = prompts * rollouts_per_problem

            # Render and generate with per-replicate seeds
            prompt_ids = self._loop.render_prompt_ids_batch(prompts_list)
            replicate_seeds = [int(seed) * 10_000 + i for i in range(rollouts_per_problem)]
            include_logprobs = bool(getattr(self._cfg, "return_logprobs", True))
            seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
                prompt_ids,
                seeds=replicate_seeds,
                trim_right_padding=True,
                include_logprobs=include_logprobs,
            )

            # Decode and step environment for rewards
            rewards: list[float] = []
            successes: list[bool] = []
            seqs: list[list[int]] = []
            prompt_lens: list[int] = []
            comp_lens: list[int] = []
            all_logprobs: list[list[float] | None] = []

            for seq, prompt_len, chosen_logprobs in seq_with_prompt_lens:
                # Enforce max training length to avoid overflow
                if TRAINER_MAX_LENGTH and len(seq) > int(TRAINER_MAX_LENGTH):
                    seq = seq[: int(TRAINER_MAX_LENGTH)]
                    # Also truncate logprobs if present
                    if chosen_logprobs is not None:
                        chosen_logprobs = chosen_logprobs[: int(TRAINER_MAX_LENGTH)]
                
                completion_ids = seq[prompt_len:]
                text = self._tokenizer.decode(completion_ids, skip_special_tokens=False)

                # Recreate env per replicate for clean single-turn stepping
                env_r = self._create_env()
                _ = env_r.reset(task_id=str(seed), seed=int(seed))
                _obs_r, reward_r, _t, _u, info_r = env_r.step(
                    ChatMessage(role="assistant", content=text)
                )

                rewards.append(float(reward_r))
                successes.append(bool(info_r.get("success", False)))
                seqs.append(seq)
                prompt_lens.append(int(prompt_len))
                comp_lens.append(int(len(completion_ids)))
                
                # Store behavior policy logprobs for importance sampling
                # Format: full sequence logprobs (prompt + completion)
                if chosen_logprobs is not None:
                    # Pad with zeros for prompt tokens, keep completion logprobs
                    full_logprobs = [0.0] * prompt_len + list(chosen_logprobs)
                    all_logprobs.append(full_logprobs)
                else:
                    all_logprobs.append(None)

            # Compute GRPO advantages (zero-mean, variance-normalized)
            advantages = self._compute_advantages(rewards)

            # Package into GRPOGroup
            group_id = str(seed)
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
            group_rewards = [r.reward for r in rollouts]
            group_successes = [r.success for r in rollouts]
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
        len(rewards)
        mean = float(np.mean(rewards))
        centered = [r - mean for r in rewards]
        var = float(np.mean([c * c for c in centered]))
        std = math.sqrt(var) if var > 0.0 else 1e-8
        return [c / std for c in centered]
