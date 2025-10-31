from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from grail.environments.core import ChatMessage
from grail.environments.loop import AgentEnvLoop, SGLangServerBackend, VLLMServerBackend
from grail.environments.sat_env import SATEnv
from grail.shared.constants import ROLLOUTS_PER_PROBLEM, TRAINER_MAX_LENGTH
from grail.trainer.algorithms.grpo import GRPOGroup, GRPORollout


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
        if backend_name == "sglang_server":
            gen_backend = SGLangServerBackend(
                base_url=self._cfg.base_url,
                model_name=getattr(tokenizer, "name_or_path", "model"),
                tokenizer=tokenizer,
                timeout=300.0,
            )
        elif backend_name == "vllm_server":
            # OpenAI-compatible vLLM server
            gen_backend = VLLMServerBackend(
                base_url=self._cfg.base_url,
                model_name=getattr(tokenizer, "name_or_path", "model"),
                tokenizer=tokenizer,
                timeout=300.0,
            )
        else:
            raise ValueError(f"Unsupported generation backend: {self._cfg.backend}")

        # model=None (server-backed); device string irrelevant for server calls
        self._loop = AgentEnvLoop(
            model=None,
            tokenizer=tokenizer,
            device="cpu",
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            batch_size=self._cfg.batch_size,
            do_sample=True,
            top_p=self._cfg.top_p,
            top_k=self._cfg.top_k,
            repetition_penalty=self._cfg.repetition_penalty,
            gen_backend=gen_backend,
        )

    async def generate_groups(self, seeds: Iterable[int]) -> list[GRPOGroup]:
        """Generate GRPO groups for the provided problem seeds.

        Each seed yields exactly `rollouts_per_problem` rollouts with deterministic
        per-replicate random seeds passed to the server backend.
        """
        groups: list[GRPOGroup] = []
        rollouts_per_problem = int(self._cfg.rollouts_per_problem)
        for seed in seeds:
            # Create a temporary env to render the prompt deterministically for this seed
            env_for_prompt = SATEnv()
            obs = env_for_prompt.reset(task_id=str(seed), seed=int(seed))
            prompts = [[{"role": m.role, "content": m.content} for m in obs.messages]]
            # Replicate prompt list per rollout
            prompts_list = prompts * rollouts_per_problem

            # Render and generate with per-replicate seeds
            prompt_ids = self._loop.render_prompt_ids_batch(prompts_list)
            replicate_seeds = [int(seed) * 10_000 + i for i in range(rollouts_per_problem)]
            seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
                prompt_ids,
                seeds=replicate_seeds,
                trim_right_padding=True,
            )

            # Decode and step environment for rewards
            rewards: list[float] = []
            successes: list[bool] = []
            seqs: list[list[int]] = []
            prompt_lens: list[int] = []
            comp_lens: list[int] = []

            for seq, prompt_len in seq_with_prompt_lens:
                # Enforce max training length to avoid overflow
                if TRAINER_MAX_LENGTH and len(seq) > int(TRAINER_MAX_LENGTH):
                    seq = seq[: int(TRAINER_MAX_LENGTH)]
                completion_ids = seq[prompt_len:]
                text = self._tokenizer.decode(completion_ids, skip_special_tokens=False)

                # Recreate env per replicate for clean single-turn stepping
                env_r = SATEnv()
                _ = env_r.reset(task_id=str(seed), seed=int(seed))
                _obs_r, reward_r, _t, _u, info_r = env_r.step(
                    ChatMessage(role="assistant", content=text)
                )

                rewards.append(float(reward_r))
                successes.append(bool(info_r.get("success", False)))
                seqs.append(seq)
                prompt_lens.append(int(prompt_len))
                comp_lens.append(int(len(completion_ids)))

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
                        token_logprobs=None,
                    )
                )

            groups.append(GRPOGroup(group_id=group_id, rollouts=rollouts))

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
