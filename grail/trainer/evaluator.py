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
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from grail.environments.core import ChatMessage, MultiTurnEnv
from grail.environments.loop import AgentEnvLoop
from grail.environments.vector import EnvVector
from grail.trainer.config import EvalConfig
from grail.trainer.eval_planner import EvaluationPlan
from grail.trainer.metrics import KMetricsAggregator as EvalAggregator
from grail.trainer.metrics import TaskReplicateResult

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
        server_base_url: str | None = None,
        server_model_name: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._env_factory = env_factory
        self._cfg = config
        self._monitor = monitor
        self._device = device

        self._vector = EnvVector(env_factory, batch_size=self._cfg.batch_size)

        # Initialize generation backend based on server configuration
        gen_backend = None
        backend_name = (self._cfg.backend or "hf").lower()

        if backend_name == "sglang" and server_base_url:
            gen_backend = self._create_sglang_backend(server_base_url, server_model_name)
        elif backend_name == "vllm" and server_base_url:
            gen_backend = self._create_vllm_backend(server_base_url, server_model_name)
        elif backend_name not in ("hf", "sglang", "vllm"):
            logger.warning("Unknown backend '%s', falling back to HF", backend_name)

        self._loop = AgentEnvLoop(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            max_new_tokens=self._cfg.max_new_tokens,
            temperature=self._cfg.temperature,
            batch_size=self._cfg.batch_size,
            do_sample=self._cfg.do_sample,
            top_p=self._cfg.top_p,
            gen_backend=gen_backend,
        )

        # Store backend reference for cleanup
        self._gen_backend = gen_backend

    def _create_sglang_backend(self, base_url: str, model_name: str | None) -> Any | None:
        """Create SGLang server backend with fallback handling."""
        try:
            from grail.environments.loop import SGLangServerBackend

            model_id = model_name or getattr(self._model, "name_or_path", "model")
            backend = SGLangServerBackend(
                base_url=base_url,
                model_name=str(model_id),
                tokenizer=self._tokenizer,
                timeout=300.0,
                max_concurrent_requests=self._cfg.sglang_max_concurrent_requests,
            )
            logger.info(
                "Evaluator using SGLang server backend: url=%s model=%s concurrency=%d",
                base_url,
                model_id,
                self._cfg.sglang_max_concurrent_requests,
            )
            return backend
        except Exception:
            logger.exception("Failed to initialize SGLang backend; falling back to HF")
            return None

    def _create_vllm_backend(self, base_url: str, model_name: str | None) -> Any | None:
        """Create vLLM server backend with fallback handling."""
        try:
            from grail.environments.loop import VLLMServerBackend

            model_id = model_name or getattr(self._model, "name_or_path", "model")
            backend = VLLMServerBackend(
                base_url=base_url,
                model_name=str(model_id),
                tokenizer=self._tokenizer,
                timeout=300.0,
                max_concurrent_requests=self._cfg.vllm_max_concurrent_requests,
            )
            logger.info(
                "Evaluator using vLLM server backend: url=%s model=%s concurrency=%d",
                base_url,
                model_id,
                self._cfg.vllm_max_concurrent_requests,
            )
            return backend
        except Exception:
            logger.exception("Failed to initialize vLLM backend; falling back to HF")
            return None

    def shutdown(self) -> None:
        """Release evaluation backend resources and model references.

        Critical for freeing GPU memory before reloading training models.
        Must be called after evaluation completes.
        """
        import gc

        # Shutdown specialized backends (vLLM/SGLang engines)
        if self._gen_backend is not None and hasattr(self._gen_backend, "shutdown"):
            try:
                logger.info("Shutting down evaluation backend...")
                self._gen_backend.shutdown()
                self._gen_backend = None
            except Exception as e:
                logger.warning(f"Error shutting down evaluation backend: {e}")

        # Release all model references (important for HF backend)
        # The evaluator holds references to the model in multiple places:
        # - self._model (direct reference)
        # - self._loop.model (AgentEnvLoop reference)
        # - self._loop._backend (HFBackend if gen_backend is None)
        try:
            self._model = None
            self._loop = None
            self._vector = None

            # Force garbage collection to release references immediately
            gc.collect()

            # Clear CUDA cache to free GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            logger.info("Evaluator resources released")
        except Exception as e:
            logger.warning(f"Error releasing evaluator resources: {e}")

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

        # Progress bar for evaluation
        with Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            prog_task = progress.add_task(
                f"Eval {total_ids} tasks × {plan.replicates} reps",
                total=total_ids,
            )

            # Iterate over task IDs, expanding to replicates via per-replicate seeds
            for offset in range(start_offset, total_ids, batch_size):
                batch_start = time.monotonic()
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
                seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
                    prompt_ids,
                    seeds=expanded_seeds,
                    trim_right_padding=True,
                )

                # Decode and step per replicate in groups matching each original task
                decoded: list[str] = []
                for seq, prompt_len in seq_with_prompt_lens:
                    decoded.append(
                        self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=False)
                    )

                # Optional: log a few sample completions for visibility
                if getattr(self._cfg, "log_completions_n", 0) > 0:
                    try:
                        max_chars = int(getattr(self._cfg, "log_completions_max_chars", 300))
                        sample_count = min(int(self._cfg.log_completions_n), len(decoded))
                        replicate_counter: dict[str, int] = {}
                        samples: list[tuple[str, int, str]] = []
                        for i, text in enumerate(decoded):
                            task_id = expanded_ids[i]
                            r_idx = replicate_counter.get(task_id, 0)
                            replicate_counter[task_id] = r_idx + 1
                            samples.append((task_id, r_idx, text))
                            if len(samples) >= sample_count:
                                break

                        for task_id, r_idx, text in samples:
                            out = text if len(text) <= max_chars else text[:max_chars] + "…"
                            logger.info(
                                "Eval sample completion task=%s rep=%d:\n%s", task_id, r_idx, out
                            )
                    except Exception:
                        # Never let logging interfere with evaluation
                        pass

                # Step envs per replicate, accumulate results
                # We step using the same env index for a task; since SingleTurnEnv terminates
                # after one step, we re-reset before each replicate to keep context identical.
                cursor = 0
                for env_idx, task_id in enumerate(batch_ids):
                    for r_idx in range(plan.replicates):
                        text = decoded[cursor]
                        _ = self._vector.envs[env_idx].reset(task_id=task_id)
                        _obs, reward, _terminated, _truncated, info = self._vector.envs[
                            env_idx
                        ].step(ChatMessage(role="assistant", content=text))
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

                # Update progress
                progress.update(prog_task, advance=len(batch_ids))

                batch_elapsed = time.monotonic() - batch_start
                batch_prompts = len(batch_ids) * plan.replicates
                throughput = batch_prompts / batch_elapsed if batch_elapsed > 0 else 0
                logger.info(
                    f"Batch {offset // batch_size + 1}: {len(batch_ids)} tasks × {plan.replicates} reps "
                    f"({batch_prompts} prompts) in {batch_elapsed:.2f}s ({throughput:.1f} prompts/sec)"
                )

                if heartbeat is not None:
                    heartbeat()
                if self._monitor:
                    await self._monitor.log_counter("eval/batches_completed")

        metrics = aggregator.summarize()

        # Calculate and log duration
        duration = time.monotonic() - t0
        logger.info(
            "✅ Evaluation complete: %d tasks × %d reps in %.2fs (%.2f tasks/sec)",
            total_ids,
            plan.replicates,
            duration,
            total_ids / duration if duration > 0 else 0,
        )

        if self._monitor:
            await self._monitor.log_gauge("profiling/eval_duration", duration)
            throughput = total_ids / duration if duration > 0 else 0
            await self._monitor.log_gauge("profiling/eval_tasks_per_sec", throughput)
            for key, val in metrics.items():
                await self._monitor.log_gauge(f"eval/{key}", float(val))

        # Optionally persist metrics
        return metrics
