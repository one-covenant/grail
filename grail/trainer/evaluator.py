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

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.loop import AgentEnvLoop
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

        # Reuse AgentEnvLoop for rendering and generation to avoid duplication
        self._loop = AgentEnvLoop(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            batch_size=self._cfg.batch_size,
            do_sample=self._cfg.do_sample,
            top_p=self._cfg.top_p,
        )

    def _render_prompts(self, prompts_list: list[list[dict[str, str]]]) -> list[list[int]]:
        return self._loop.render_prompt_ids_batch(prompts_list)

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
            # Generate sequences deterministically per replicate using seeds
            seq_with_prompt_lens = self._loop.generate_from_prompt_ids_batch(
                prompt_ids,
                seeds=expanded_seeds,
                trim_right_padding=True,
            )

            # Decode and step per replicate in groups matching each original task
            decoded: list[str] = []
            for seq, prompt_len in seq_with_prompt_lens:
                decoded.append(self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=False))

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
