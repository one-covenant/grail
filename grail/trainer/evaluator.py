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
from grail.environments.execution import CodeExecutionPool, set_global_execution_pool
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
            do_sample=self._cfg.do_sample,
            top_p=self._cfg.top_p,
            gen_backend=gen_backend,
        )

        # Store backend reference for cleanup
        self._gen_backend = gen_backend

        # Initialize fast code execution pool
        # Workers are spawned once and reused across all evaluations
        # This eliminates ~2s spawn overhead per code execution
        self._execution_pool: CodeExecutionPool | None = None
        self._init_execution_pool()

    def _init_execution_pool(self) -> None:
        """Initialize the fast code execution pool.

        Workers are spawned immediately and warmed up to eliminate first-call latency.
        The pool is set as the global execution pool so environments can use it.
        """
        try:
            # Use 8 workers - enough parallelism without exhausting resources
            # max_tasks_per_child=50 recycles workers to prevent memory leaks
            self._execution_pool = CodeExecutionPool(
                num_workers=8,
                max_tasks_per_child=50,
            )
            self._execution_pool.start()
            set_global_execution_pool(self._execution_pool)
            logger.info(
                "Fast code execution pool initialized: %d workers",
                self._execution_pool.num_workers,
            )
        except Exception as e:
            logger.warning("Failed to initialize execution pool, falling back to slow path: %s", e)
            self._execution_pool = None

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
                return_chosen_logprobs=bool(
                    getattr(self._cfg, "vllm_return_chosen_logprobs", False)
                ),
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
            return_chosen_lp = bool(getattr(self._cfg, "vllm_return_chosen_logprobs", False))
            backend = VLLMServerBackend(
                base_url=base_url,
                model_name=str(model_id),
                tokenizer=self._tokenizer,
                timeout=300.0,
                max_concurrent_requests=self._cfg.vllm_max_concurrent_requests,
                return_chosen_logprobs=return_chosen_lp,
                warn_on_missing_token_ids=return_chosen_lp,
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

        # Shutdown fast code execution pool first
        if self._execution_pool is not None:
            try:
                logger.info("Shutting down code execution pool...")
                set_global_execution_pool(None)  # Clear global reference
                self._execution_pool.shutdown()
                self._execution_pool = None
                logger.info("Code execution pool shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down execution pool: {e}")

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
        window_number: int | None = None,
    ) -> dict[str, float]:
        """Run an evaluation cycle from the given offset. Returns summary metrics.

        Args:
            plan: Evaluation plan with task IDs and seeds
            start_offset: Offset to start from (for resumption)
            wallet: Optional wallet for credentials
            heartbeat: Optional heartbeat callback
            window_number: Current window number for context tracking
        """
        logger.info("=" * 60)
        logger.info("🚀 EVALUATION CYCLE START")
        logger.info(f"   Total tasks: {len(plan.ids)}")
        logger.info(f"   Replicates: {plan.replicates}")
        logger.info(f"   Batch size: {self._cfg.batch_size}")
        logger.info(f"   Total prompts: {len(plan.ids) * plan.replicates}")
        logger.info(f"   Backend: {self._cfg.backend or 'hf'}")
        logger.info("=" * 60)

        aggregator = EvalAggregator(report_ks=self._cfg.report_ks)

        total_ids = len(plan.ids)
        batch_size = self._cfg.batch_size
        t0 = time.monotonic()
        self._current_window_number = window_number

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
            num_batches = (total_ids - start_offset + batch_size - 1) // batch_size
            logger.info(f"📦 Processing {num_batches} batches...")

            for batch_idx, offset in enumerate(range(start_offset, total_ids, batch_size), 1):
                batch_start = time.monotonic()
                logger.info("")
                logger.info(f"{'=' * 60}")
                logger.info(f"📦 BATCH {batch_idx}/{num_batches} (offset={offset})")
                logger.info(f"{'=' * 60}")

                if heartbeat is not None:
                    heartbeat()
                batch_ids = plan.ids[offset : min(offset + batch_size, total_ids)]
                logger.info(f"   Tasks in batch: {len(batch_ids)}")
                logger.info(f"   Task IDs: {batch_ids[:3]}{'...' if len(batch_ids) > 3 else ''}")

                # Prepare batch: reset envs, expand to replicates, render prompts
                logger.info(
                    "⚙️  [1/4] Preparing batch (reset envs, expand to replicates, render prompts)..."
                )
                prep_start = time.monotonic()
                expanded_ids, expanded_msgs, prompt_ids = self._prepare_batch(batch_ids, plan)
                prep_time = time.monotonic() - prep_start
                logger.info(f"   ✓ Prepared {len(expanded_ids)} prompts in {prep_time:.2f}s")

                # Generate sequences deterministically per replicate using seeds
                logger.info(f"🤖 [2/4] Generating completions ({len(prompt_ids)} prompts)...")
                gen_start = time.monotonic()
                seq_with_prompt_lens = await self._generate_batch(prompt_ids, expanded_ids, plan)
                gen_time = time.monotonic() - gen_start
                logger.info(
                    f"   ✓ Generated {len(seq_with_prompt_lens)} completions in {gen_time:.2f}s ({gen_time / len(seq_with_prompt_lens):.2f}s/prompt)"
                )

                # Decode completions
                logger.info("📝 [3/4] Decoding completions...")
                decode_start = time.monotonic()
                decoded = self._decode_completions(seq_with_prompt_lens)
                decode_time = time.monotonic() - decode_start
                logger.info(f"   ✓ Decoded {len(decoded)} completions in {decode_time:.2f}s")

                # Log sample completions with full templated prompts
                self._log_completion_samples(expanded_ids, prompt_ids, decoded, plan.replicates)

                # Step environments and accumulate results
                logger.info("🧪 [4/4] Executing code tests (running in parallel)...")
                step_start = time.monotonic()
                self._step_and_aggregate(batch_ids, decoded, plan.replicates, aggregator)
                step_time = time.monotonic() - step_start
                logger.info(
                    f"   ✓ Executed tests for {len(batch_ids)} tasks in {step_time:.2f}s ({step_time / len(batch_ids):.2f}s/task)"
                )

                # Update progress and log metrics
                progress.update(prog_task, advance=len(batch_ids))
                batch_total = time.monotonic() - batch_start
                logger.info("")
                logger.info(f"📊 Batch {batch_idx} timing breakdown:")
                logger.info(
                    f"   Prepare:  {prep_time:6.2f}s ({100 * prep_time / batch_total:5.1f}%)"
                )
                logger.info(f"   Generate: {gen_time:6.2f}s ({100 * gen_time / batch_total:5.1f}%)")
                logger.info(
                    f"   Decode:   {decode_time:6.2f}s ({100 * decode_time / batch_total:5.1f}%)"
                )
                logger.info(
                    f"   Execute:  {step_time:6.2f}s ({100 * step_time / batch_total:5.1f}%) ← CODE EXECUTION"
                )
                logger.info(f"   Total:    {batch_total:6.2f}s")

                self._log_batch_metrics(
                    offset, batch_size, len(batch_ids), plan.replicates, batch_start
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

    def _prepare_batch(
        self, batch_ids: list[str], plan: EvaluationPlan
    ) -> tuple[list[str], list[list[dict[str, str]]], list[list[int]]]:
        """Prepare batch: reset environments, expand to replicates, render prompts.

        Args:
            batch_ids: List of task IDs for this batch
            plan: Evaluation plan containing replicates and seed info

        Returns:
            Tuple of (expanded_ids, expanded_msgs, prompt_ids)
        """
        expanded_ids: list[str] = []
        expanded_msgs: list[list[dict[str, str]]] = []
        expanded_seeds: list[int] = []

        # Reset envs once per task (not per replicate) for correct info context
        self._vector.reset_ids(batch_ids)

        # Build prompts per replicate
        for task_id in batch_ids:
            # Use the reset obs from the corresponding env
            env_idx = batch_ids.index(task_id)
            # Re-reset to fetch observation deterministically for clarity
            obs = self._vector.envs[env_idx].reset(task_id=task_id)
            prompts = [{"role": m.role, "content": m.content} for m in obs.messages]
            for r_idx in range(plan.replicates):
                expanded_ids.append(task_id)
                expanded_msgs.append(prompts)
                expanded_seeds.append(plan.seed_for(task_id, r_idx))

        prompt_ids = self._render_prompts(expanded_msgs)
        return expanded_ids, expanded_msgs, prompt_ids

    async def _generate_batch(
        self,
        prompt_ids: list[list[int]],
        expanded_ids: list[str],
        plan: EvaluationPlan,
    ) -> list[tuple[list[int], int]]:
        """Generate sequences deterministically using seeds.

        Args:
            prompt_ids: Tokenized prompts with chat template applied
            expanded_ids: Expanded task IDs (one per replicate)
            plan: Evaluation plan containing seed and replication info

        Returns:
            List of tuples per sample. Each tuple contains at least
            (sequence, prompt_length). A third element may be present with
            chosen-token logprobs if the backend provides them and they were
            requested upstream.
        """
        # Reconstruct seeds for generation
        expanded_seeds: list[int] = []
        task_id_to_replicates: dict[str, int] = {}
        for task_id in expanded_ids:
            r_idx = task_id_to_replicates.get(task_id, 0)
            task_id_to_replicates[task_id] = r_idx + 1
            expanded_seeds.append(plan.seed_for(task_id, r_idx))

        seq_with_prompt_lens = await self._loop.generate_from_prompt_ids_batch(
            prompt_ids,
            seeds=expanded_seeds,
            trim_right_padding=True,
        )
        return seq_with_prompt_lens

    def _decode_completions(
        self,
        seq_with_prompt_lens: list[
            tuple[list[int], int] | tuple[list[int], int, list[float] | None]
        ],
    ) -> list[str]:
        """Decode generated sequences to text, excluding prompt tokens.

        Args:
            seq_with_prompt_lens: List of tuples: (sequence, prompt_length)
                and optionally a third element with chosen-token logprobs.

        Returns:
            List of decoded completion strings
        """
        decoded: list[str] = []
        for item in seq_with_prompt_lens:
            seq = item[0]
            prompt_len = item[1]
            decoded.append(self._tokenizer.decode(seq[prompt_len:], skip_special_tokens=False))
        return decoded

    def _log_completion_samples(
        self,
        expanded_ids: list[str],
        prompt_ids: list[list[int]],
        decoded: list[str],
        replicates: int,
    ) -> None:
        """Log sample completions with full templated prompts for visibility.

        Args:
            expanded_ids: Expanded task IDs (one per replicate)
            prompt_ids: Tokenized prompts with chat template applied
            decoded: Decoded completion strings
            replicates: Number of replicates per task
        """
        if getattr(self._cfg, "log_completions_n", 0) <= 0:
            return

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
                # Find the index of this sample in decoded
                idx = expanded_ids.index(task_id) + r_idx
                # Decode the actual prompt with chat template applied
                prompt_str = self._tokenizer.decode(prompt_ids[idx], skip_special_tokens=False)
                out = text if len(text) <= max_chars else text[:max_chars] + "…"
                logger.info(
                    "Eval sample task=%s rep=%d\nPROMPT (with chat template):\n%s\nCOMPLETION:\n%s",
                    task_id,
                    r_idx,
                    prompt_str,
                    out,
                )
        except Exception:
            # Never let logging interfere with evaluation
            pass

    def _step_and_aggregate(
        self,
        batch_ids: list[str],
        decoded: list[str],
        replicates: int,
        aggregator: EvalAggregator,
    ) -> None:
        """Step environments per replicate and accumulate results.

        Args:
            batch_ids: List of task IDs for this batch
            decoded: Decoded completion strings
            replicates: Number of replicates per task
            aggregator: Metrics aggregator to accumulate results
        """
        cursor = 0
        for env_idx, task_id in enumerate(batch_ids):
            for r_idx in range(replicates):
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

    def _log_batch_metrics(
        self,
        offset: int,
        batch_size: int,
        num_batch_ids: int,
        replicates: int,
        batch_start: float,
    ) -> None:
        """Log batch-level performance metrics.

        Args:
            offset: Starting offset in evaluation
            batch_size: Configured batch size
            num_batch_ids: Actual number of task IDs in this batch
            replicates: Number of replicates per task
            batch_start: Start time of batch (monotonic)
        """
        batch_elapsed = time.monotonic() - batch_start
        batch_prompts = num_batch_ids * replicates
        throughput = batch_prompts / batch_elapsed if batch_elapsed > 0 else 0
        logger.info(
            f"Batch {offset // batch_size + 1}: {num_batch_ids} tasks × {replicates} reps "
            f"({batch_prompts} prompts) in {batch_elapsed:.2f}s ({throughput:.1f} prompts/sec)"
        )
