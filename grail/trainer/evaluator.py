"""Evaluator service: vectorized, deterministic evaluation cycles.

Coordinates planning, generation, environment stepping, aggregation,
and monitoring. Designed to be called from the trainer loop, possibly
across multiple windows until a cycle completes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.vector import EnvVector
from grail.trainer.config import EvalConfig
from grail.trainer.eval_aggregator import EvalAggregator, TaskReplicateResult
from grail.trainer.eval_planner import EvaluationPlan

logger = logging.getLogger(__name__)


@dataclass
class EvalProgress:
    cycle_index: int
    offset: int  # number of task IDs fully processed so far


class EvaluatorService:
    """Run evaluation cycles in batches with deterministic seeds and replicates."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        env_factory: Callable[[], MultiTurnEnv],
        config: EvalConfig,
        monitor: Any | None = None,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._env_factory = env_factory
        self._cfg = config
        self._monitor = monitor
        self._device = device

        self._vector = EnvVector(env_factory, batch_size=self._cfg.batch_size)

    def _render_prompts(self, prompts_list: list[list[dict[str, str]]]) -> list[list[int]]:
        # Reuse AgentEnvLoop logic for chat rendering and tokenization
        input_ids: list[list[int]] = []
        for msgs in prompts_list:
            rendered = self._tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            toks = self._tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
            input_ids.append(toks.input_ids[0].tolist())
        return input_ids

    def _generate_batch(
        self,
        batch_prompt_ids: list[list[int]],
        generators: list[torch.Generator] | None,
    ) -> list[list[int]]:
        # Similar to AgentEnvLoop._batch_generate_tokens but returns trimmed sequences
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        max_len = max(len(p) for p in batch_prompt_ids)
        padded_inputs = []
        attention_masks = []
        for p in batch_prompt_ids:
            pad_len = max_len - len(p)
            padded_inputs.append([pad_id] * pad_len + p)
            attention_masks.append([0] * pad_len + [1] * len(p))

        input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._cfg.max_new_tokens,
            "temperature": self._cfg.temperature,
            "do_sample": self._cfg.do_sample,
            "top_p": self._cfg.top_p,
            "return_dict_in_generate": True,
            "pad_token_id": pad_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if generators is not None and len(generators) == input_ids.size(0):
            gen_kwargs["generator"] = generators

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        results: list[list[int]] = []
        for b in range(input_ids.size(0)):
            # Remove left padding and trim right padding similar to existing helper
            seq = outputs.sequences[b]
            left_pad = max_len - len(batch_prompt_ids[b])
            seq_wo_left = seq[left_pad:]
            # Trim right padding
            all_ids = seq_wo_left.tolist()
            results.append(all_ids)
        return results

    def _build_generators(self, seeds: list[int]) -> list[torch.Generator]:
        gens: list[torch.Generator] = []
        for s in seeds:
            g = torch.Generator(device=self._device)
            g.manual_seed(int(s))
            gens.append(g)
        return gens

    async def run_cycle(
        self,
        plan: EvaluationPlan,
        *,
        start_offset: int = 0,
        wallet: Any | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> dict[str, float]:
        """Run an evaluation cycle from the given offset. Returns summary metrics."""
        aggregator = EvalAggregator(report_ks=self._cfg.report_ks)

        total_ids = len(plan.ids)
        batch_size = self._cfg.batch_size
        t0 = time.monotonic()

        # Iterate over task IDs, expanding to replicates via per-replicate seeds
        for offset in range(start_offset, total_ids, batch_size):
            if heartbeat is not None:
                heartbeat()
            batch_ids = plan.ids[offset : min(offset + batch_size, total_ids)]

            # For pass@k, we generate up to plan.replicates completions per task
            # using batch expansion: duplicate each task id k times and use distinct seeds.
            expanded_ids: list[str] = []
            expanded_msgs: list[list[dict[str, str]]] = []
            expanded_seeds: list[int] = []

            # Reset envs once per task (not per replicate) for correct info context
            self._vector.reset_ids(batch_ids)

            # Build prompts per replicate
            for task_id in batch_ids:
                # Use the reset obs from the corresponding env
                # Grab initial observation messages directly from envs
                env_idx = batch_ids.index(task_id)
                # Re-reset to fetch observation deterministically for clarity
                obs = self._vector.envs[env_idx].reset(task_id=task_id)
                prompts = [{"role": m.role, "content": m.content} for m in obs.messages]
                for r_idx in range(plan.replicates):
                    expanded_ids.append(task_id)
                    expanded_msgs.append(prompts)
                    expanded_seeds.append(plan.seed_for(task_id, r_idx))

            prompt_ids = self._render_prompts(expanded_msgs)
            generators = self._build_generators(expanded_seeds)
            sequences = self._generate_batch(prompt_ids, generators)

            # Decode and step per replicate in groups matching each original task
            decoded: list[str] = [
                self._tokenizer.decode(seq[len(p_ids) :], skip_special_tokens=False)
                for seq, p_ids in zip(sequences, prompt_ids, strict=False)
            ]

            # Step envs per replicate, accumulate results
            # We step using the same env index for a task; since SingleTurnEnv terminates
            # after one step, we re-reset before each replicate to keep context identical.
            cursor = 0
            for env_idx, task_id in enumerate(batch_ids):
                for r_idx in range(plan.replicates):
                    text = decoded[cursor]
                    _ = self._vector.envs[env_idx].reset(task_id=task_id)
                    _obs, reward, _terminated, _truncated, info = self._vector.envs[env_idx].step(
                        ChatMessage(role="assistant", content=text)
                    )
                    success = bool(info.get("success", False))
                    aggregator.add(
                        TaskReplicateResult(
                            task_id=task_id,
                            replicate_idx=r_idx,
                            reward=float(reward),
                            success=success,
                            components=info.get("reward_components"),
                        )
                    )
                    cursor += 1

            if heartbeat is not None:
                heartbeat()
            if self._monitor:
                await self._monitor.log_counter("eval/batches_completed")

        metrics = aggregator.summarize()

        if self._monitor:
            duration = time.monotonic() - t0
            await self._monitor.log_gauge("profiling/eval_duration", duration)
            for key, val in metrics.items():
                await self._monitor.log_gauge(f"eval/{key}", float(val))

        # Optionally persist metrics
        return metrics
