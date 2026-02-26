from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.gsm8k_env import GSM8KEnv
from grail.environments.loop import AgentEnvLoop, SGLangServerBackend, VLLMServerBackend
from grail.environments.sat_env import SATEnv
from grail.shared.constants import ROLLOUTS_PER_PROBLEM, TRAINER_MAX_LENGTH
from grail.trainer.algorithms.grpo import GRPOGroup, GRPORollout

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Constants for rollout generation
REPLICATE_SEED_MULTIPLIER = 10_000  # Used to generate unique seeds per rollout replicate
LOGPROB_MISMATCH_WARN_THRESHOLD = 0.1  # Threshold for warning on HF vs vLLM logprob differences
LOGPROB_MISMATCH_INFO_THRESHOLD = 0.01  # Threshold for info-level logprob difference logging


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

    This generator uses vLLM/SGLang server for efficient batch generation,
    but computes logprobs using the local HuggingFace model when available.
    This ensures ground-truth logprobs for importance sampling (IS) without
    vLLM's sampling transformations (top-k/top-p/temperature).

    Key design decisions:
    - Generation: Always uses vLLM/SGLang server (fast, batched)
    - Logprobs: Uses HF model when provided (ground truth), falls back to vLLM
    - GRPO: Computes zero-mean, unit-variance advantages per problem group

    No proofs or wallet interactions are used - this is for offline training only.
    """

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase,
        config: RolloutGenConfig,
        hf_model: PreTrainedModel | None = None,
    ) -> None:
        """Initialize the offline rollout generator.

        Args:
            tokenizer: HuggingFace tokenizer for the model
            config: Rollout generation configuration
            hf_model: Optional HuggingFace model for computing ground-truth logprobs.
                     If provided, logprobs will be computed from this model instead of vLLM.
        """
        self._tokenizer = tokenizer
        self._cfg = config
        self._hf_model = hf_model

        # Choose server backend
        backend_name = (self._cfg.backend or "sglang_server").lower()

        # OPTIMIZATION: Only request logprobs from vLLM if HF model is NOT provided.
        # When HF model is available, we compute ground-truth logprobs locally,
        # so requesting them from vLLM wastes bandwidth and computation.
        request_server_logprobs = bool(getattr(self._cfg, "return_logprobs", True))
        if hf_model is not None and request_server_logprobs:
            logger.info(
                "HF model provided - will compute logprobs locally instead of from server. "
                "Disabling server logprob requests for efficiency."
            )
            request_server_logprobs = False

        # Store decision for use in generate_groups
        self._request_server_logprobs = request_server_logprobs

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
                "logprob_source": "hf_model" if hf_model is not None else "server",
                "request_server_logprobs": request_server_logprobs,
            },
        )

        if backend_name == "sglang_server":
            gen_backend = SGLangServerBackend(
                base_url=self._cfg.base_url,
                model_name=served_model_name,
                tokenizer=tokenizer,
                timeout=300.0,
                return_chosen_logprobs=request_server_logprobs,
            )
        elif backend_name == "vllm_server":
            # OpenAI-compatible vLLM server
            gen_backend = VLLMServerBackend(
                base_url=self._cfg.base_url,
                model_name=served_model_name,
                tokenizer=tokenizer,
                timeout=300.0,
                return_chosen_logprobs=request_server_logprobs,
            )
        else:
            msg = f"Unsupported generation backend: {self._cfg.backend}"
            logger.error("Unsupported generation backend", extra={"backend": self._cfg.backend})
            raise ValueError(msg)

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
        """Factory method to create environment instance based on config.

        Returns:
            MultiTurnEnv: Fresh environment instance (GSM8K or SAT)

        Raises:
            ValueError: If environment type is not supported
        """
        env_type = getattr(self._cfg, "environment", "sat").lower()
        if env_type == "gsm8k":
            return GSM8KEnv()
        elif env_type == "sat":
            return SATEnv()
        else:
            msg = f"Unsupported environment type: {env_type}"
            raise ValueError(msg)

    def _compute_hf_logprobs(
        self, token_ids: list[int], prompt_len: int, completion_len: int
    ) -> list[float] | None:
        """Compute ground-truth logprobs using local HuggingFace model.

        This provides logprobs directly from the model's distribution WITHOUT any
        sampling transformations (temperature/top-k/top-p). These are the true
        log probabilities under the model's policy, ideal for importance sampling.

        The computation process:
        1. Cast logits to float32 to ensure numerical precision (model may use bfloat16)
        2. Extract logits for each completion token (logits[i] predicts token[i+1])
        3. Compute log_softmax and extract logprob for the chosen token

        Args:
            token_ids: Full sequence token IDs (prompt + completion)
            prompt_len: Number of prompt tokens
            completion_len: Number of completion tokens

        Returns:
            List of logprobs for completion tokens only, or None if computation fails.
            Length of returned list equals completion_len.
        """
        model = self._hf_model
        if model is None:
            return None
        if completion_len <= 0:
            return []
        if prompt_len <= 0:
            logger.warning(
                "Invalid prompt_len for HF logprob computation; returning None",
                extra={"prompt_len": prompt_len, "completion_len": completion_len},
            )
            return None

        # Prepare input tensor on model's device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)

        # Forward pass without gradients for efficiency
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                outputs = model(input_ids)
        except RuntimeError as exc:
            logger.error(
                "Failed to compute HF logprobs due to model forward error",
                extra={
                    "error": str(exc),
                    "prompt_len": prompt_len,
                    "completion_len": completion_len,
                    "seq_len": len(token_ids),
                },
            )
            return None
        finally:
            if was_training:
                model.train()

        # Extract logprobs for completion tokens only
        # Note: logits[i] predicts token[i+1], so completion logprobs start at logits[prompt_len-1].
        if not hasattr(outputs, "logits"):
            logger.error(
                "HF model output missing logits; cannot compute logprobs",
                extra={"prompt_len": prompt_len, "completion_len": completion_len},
            )
            return None

        logits = outputs.logits[0]  # [seq_len, vocab_size] (dtype may be bf16/fp16)
        seq_len = int(logits.shape[0])

        completion_logprobs: list[float] = []
        for i in range(completion_len):
            logit_pos = (prompt_len - 1) + i
            token_pos = prompt_len + i

            if logit_pos >= seq_len or token_pos >= len(token_ids):
                logger.warning(
                    "HF logprob index out of bounds; padding remaining positions with 0.0",
                    extra={
                        "logit_pos": logit_pos,
                        "token_pos": token_pos,
                        "seq_len_logits": seq_len,
                        "seq_len_tokens": len(token_ids),
                        "prompt_len": prompt_len,
                        "completion_len": completion_len,
                    },
                )
                completion_logprobs.extend([0.0] * (completion_len - i))
                break

            token_id = int(token_ids[token_pos])
            # Compute logprob(token) = logit(token) - logsumexp(logits) in float32 for stability.
            row = logits[logit_pos].float()
            log_denom = torch.logsumexp(row, dim=-1)
            completion_logprobs.append(float((row[token_id] - log_denom).item()))

        return completion_logprobs

    @staticmethod
    def _generate_replicate_seeds(problem_seed: int, num_rollouts: int) -> list[int]:
        """Generate deterministic per-rollout seeds from a problem seed.

        Args:
            problem_seed: Base seed for the problem
            num_rollouts: Number of rollout replicates to generate

        Returns:
            List of unique seeds for each rollout replicate
        """
        return [problem_seed * REPLICATE_SEED_MULTIPLIER + i for i in range(num_rollouts)]

    def _log_logprob_comparison(
        self,
        hf_logprobs: list[float],
        vllm_logprobs: list[float],
        completion_len: int,
    ) -> None:
        """Log comparison statistics between HF and vLLM logprobs.

        This helps diagnose discrepancies between ground-truth (HF) and server (vLLM)
        logprobs, which can indicate token ID misalignment or sampling transform issues.

        Args:
            hf_logprobs: Ground-truth logprobs from HuggingFace model
            vllm_logprobs: Logprobs returned from vLLM server
            completion_len: Length of the completion sequence
        """
        # Truncate vLLM logprobs to completion length for fair comparison
        vllm_comp_lps = vllm_logprobs[:completion_len]

        # Compute token-level differences
        diffs = [abs(h - v) for h, v in zip(hf_logprobs, vllm_comp_lps, strict=False)]
        if not diffs:
            return

        mean_diff = float(np.mean(diffs))
        max_diff = float(np.max(diffs))
        min_diff = float(np.min(diffs))
        std_diff = float(np.std(diffs))

        # Compute sequence-level log-ratio difference
        hf_sum = sum(hf_logprobs)
        vllm_sum = sum(vllm_comp_lps)
        log_ratio_diff = abs(hf_sum - vllm_sum)

        # Log at appropriate level based on severity
        if mean_diff > LOGPROB_MISMATCH_WARN_THRESHOLD:
            logger.warning(
                "⚠️  LARGE logprob mismatch! HF and vLLM compute from DIFFERENT distributions. "
                "mean_diff=%.4f (>%.2f threshold), max_diff=%.4f, std_diff=%.4f, "
                "seq_log_ratio_diff=%.4f, comp_len=%d",
                mean_diff,
                LOGPROB_MISMATCH_WARN_THRESHOLD,
                max_diff,
                std_diff,
                log_ratio_diff,
                completion_len,
            )
        elif mean_diff > LOGPROB_MISMATCH_INFO_THRESHOLD:
            logger.info(
                "Moderate logprob mismatch (likely token ID differences, not distribution): "
                "mean_diff=%.4f, max_diff=%.4f, std_diff=%.4f, "
                "seq_log_ratio_diff=%.4f, comp_len=%d",
                mean_diff,
                max_diff,
                std_diff,
                log_ratio_diff,
                completion_len,
            )
        else:
            logger.debug(
                "✓ Logprobs match well (token ID alignment OK): "
                "mean_diff=%.4f, max_diff=%.4f, min_diff=%.4f, std_diff=%.4f",
                mean_diff,
                max_diff,
                min_diff,
                std_diff,
            )

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

            # Step 3: Create deterministic replicate seeds for all rollouts
            replicate_seeds: list[int] = []
            for seed in batch_seeds:
                replicate_seeds.extend(self._generate_replicate_seeds(seed, rollouts_per_problem))

            # Step 4: Generate completions using server backend
            # Only request server logprobs if HF model is not available (decided in __init__)
            seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
                prompt_ids,
                seeds=replicate_seeds,
                trim_right_padding=True,
                include_logprobs=self._request_server_logprobs,
            )
            logger.debug(
                "Batch generation complete",
                extra={"total_sequences": len(seq_with_prompt_lens)},
            )

            # Step 5: Process results for each seed in batch
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
                    # Enforce max training sequence length to prevent OOM during training
                    max_len = int(TRAINER_MAX_LENGTH) if TRAINER_MAX_LENGTH else None
                    if max_len and len(seq) > max_len:
                        seq = seq[:max_len]

                    # Ensure prompt_len is consistent with any truncation
                    effective_prompt_len = min(int(prompt_len), len(seq))
                    completion_ids: list[int] = seq[effective_prompt_len:]
                    completion_len: int = len(completion_ids)
                    # chosen_logprobs are completion-only; truncate to match any completion truncation
                    if chosen_logprobs is not None and len(chosen_logprobs) > completion_len:
                        chosen_logprobs = chosen_logprobs[:completion_len]
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
                    prompt_lens.append(int(effective_prompt_len))
                    comp_lens.append(completion_len)

                    # Compute logprobs: prefer HF model for ground-truth, fallback to vLLM
                    final_logprobs: list[float] | None = None

                    if self._hf_model is not None:
                        # Use HF model to compute ground-truth logprobs
                        hf_logprobs = self._compute_hf_logprobs(
                            seq, effective_prompt_len, completion_len
                        )
                        if hf_logprobs is not None:
                            final_logprobs = hf_logprobs

                            # Compare with vLLM logprobs if available (for diagnostics only)
                            if chosen_logprobs is not None:
                                self._log_logprob_comparison(
                                    hf_logprobs, chosen_logprobs, completion_len
                                )

                    # Fallback: use vLLM logprobs if HF not available
                    if final_logprobs is None and chosen_logprobs is not None:
                        final_logprobs = chosen_logprobs
                        logger.debug(
                            "Using vLLM logprobs (HF model unavailable), comp_len=%d",
                            completion_len,
                        )

                    # Package logprobs in full-sequence format for GRPO
                    # GRPO expects logprobs aligned with token sequences (prompt + completion)
                    # We pad prompt positions with 0.0 since we only compute/care about completion logprobs
                    if final_logprobs is not None:
                        # final_logprobs contains completion-only logprobs (from HF or vLLM)
                        # Pad with zeros for prompt tokens to create full sequence alignment
                        full_logprobs: list[float] = [0.0] * effective_prompt_len + list(
                            final_logprobs
                        )
                        all_logprobs.append(full_logprobs)
                    else:
                        # No logprobs available - this rollout cannot be used for IS
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
        """Compute GRPO advantages from rewards within a problem group.

        GRPO (Group Relative Policy Optimization) uses advantages that are:
        1. Zero-mean: Centered by subtracting group mean reward
        2. Unit-variance: Normalized by group standard deviation

        This normalization ensures:
        - Positive advantages → reward above group average → increase probability
        - Negative advantages → reward below group average → decrease probability
        - Advantages sum to zero within each group (no bias)

        Args:
            rewards: List of rewards for all rollouts in a problem group

        Returns:
            List of normalized advantages (same length as rewards)
        """
        if not rewards:
            return []

        # Center rewards by subtracting mean
        mean = float(np.mean(rewards))
        centered = [r - mean for r in rewards]

        # Normalize by standard deviation (with small epsilon for stability)
        var = float(np.mean([c * c for c in centered]))
        std = math.sqrt(var) if var > 0.0 else 1e-8

        return [c / std for c in centered]
