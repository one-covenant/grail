"""Pipelined mining engine.

After generation completes, proof computation and kernel eval run **in parallel**
since proofs only need token IDs, not env rewards. Kernel eval dispatches env.step()
calls across multiple GPUs via ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ..environments.advantages import compute_advantages
from ..environments.backends import GenerationParams
from ..environments.core import ChatMessage, MultiTurnEnv
from ..environments.rollout import GRPORollout, assemble_rollouts
from .config import PipelineConfig
from .proof_worker import ProofWorker
from .weight_sync import WeightSyncStrategy

logger = logging.getLogger(__name__)
timing_logger = logging.getLogger("grail.miner.timing")


class PipelinedMiningEngine:
    """Orchestrates 3-GPU pipelined mining.

    Composes a ``WeightSyncStrategy`` (generation on GPU 0) with a
    ``ProofWorker`` (HF model on GPU 1). Kernel eval runs on GPU 2/3
    inside the environment's ``step()`` method.

    After generation, proof computation and kernel eval run in parallel
    on different GPUs, then results are combined for rollout assembly.
    """

    def __init__(
        self,
        config: PipelineConfig,
        weight_sync: WeightSyncStrategy,
        proof_worker: ProofWorker,
        *,
        gen_params: GenerationParams | None = None,
    ) -> None:
        self._config = config
        self._weight_sync = weight_sync
        self._proof_worker = proof_worker
        self._gen_params = gen_params or GenerationParams()
        self._proof_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="proof")
        self._eval_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="eval")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_grpo_group(
        self,
        env_factory: Callable[[], MultiTurnEnv],
        count: int,
        randomness_hex: str,
        wallet: Any,
        *,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> list[GRPORollout]:
        """Generate a full GRPO group with parallel proof + kernel eval.

        Flow:
        1. Create & reset envs (CPU)
        2. Generate via backend (GPU 0)
        3. Submit proofs immediately (GPU 1, non-blocking — only needs token IDs)
        4. Step environments for kernel eval (GPU 2/3, runs in parallel with proofs)
        5. Collect proofs
        6. Assemble rollouts + compute advantages

        Args:
            env_factory: Factory for environment instances
            count: Number of rollouts in the group (e.g. 16)
            randomness_hex: Randomness beacon hex string
            wallet: Bittensor wallet for signing
            batch_size: If None, uses ``count`` (vLLM handles parallelism)
            seed: Optional seed for environment reset

        Returns:
            List of GRPORollout with advantages computed
        """
        if batch_size is None:
            batch_size = count

        backend = self._weight_sync.get_backend()
        tokenizer = self._proof_worker.tokenizer
        all_batch_data: list[tuple[list[int], int, float, dict]] = []
        all_proof_results: list[tuple[list[dict], list[float], bytes, dict, str]] = []

        group_start = time.time()
        total_gen_time = 0.0
        total_eval_time = 0.0
        total_proof_time = 0.0
        total_proof_wait = 0.0
        total_tokens = 0

        for batch_start in range(0, count, batch_size):
            batch_end = min(batch_start + batch_size, count)
            batch_count = batch_end - batch_start

            # 1. Create and reset environments
            t0 = time.time()
            envs = [env_factory() for _ in range(batch_count)]
            obs_list = [env.reset(seed=seed) for env in envs]
            env_create_time = time.time() - t0

            # 2. Render prompts
            t0 = time.time()
            messages_list = [
                [{"role": m.role, "content": m.content} for m in obs.messages] for obs in obs_list
            ]
            prompt_ids_batch: list[list[int]] = []
            for messages in messages_list:
                from ..shared.chat_templates import apply_chat_template

                rendered = apply_chat_template(tokenizer, messages)
                toks = tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
                prompt_ids_batch.append(toks.input_ids[0].tolist())
            render_time = time.time() - t0

            prompt_lengths = [len(ids) for ids in prompt_ids_batch]
            logger.info(
                "TIMING env_create=%.2fs render=%.2fs prompt_lens=%s",
                env_create_time,
                render_time,
                prompt_lengths,
            )

            # 3. Generate via backend (GPU 0)
            gen_start = time.time()
            gen_results = await backend.generate(prompt_ids_batch, params=self._gen_params)
            gen_time = time.time() - gen_start
            total_gen_time += gen_time

            # Extract all_ids and prompt_lens for proof submission
            gen_all_ids: list[list[int]] = []
            gen_prompt_lens: list[int] = []
            completion_lengths: list[int] = []
            for env_idx in range(batch_count):
                all_ids, _chosen_lp = gen_results[env_idx]
                prompt_len = len(prompt_ids_batch[env_idx])
                gen_all_ids.append(all_ids)
                gen_prompt_lens.append(prompt_len)
                completion_lengths.append(len(all_ids) - prompt_len)

            total_completion_tokens = sum(completion_lengths)
            total_tokens += total_completion_tokens
            logger.info(
                "TIMING gen=%.2fs (%d prompts, %d total_completion_tokens, "
                "min/avg/max=%d/%.0f/%d tokens, %.0f tok/s)",
                gen_time,
                batch_count,
                total_completion_tokens,
                min(completion_lengths),
                total_completion_tokens / max(len(completion_lengths), 1),
                max(completion_lengths),
                total_completion_tokens / max(gen_time, 0.001),
            )

            # 4. Submit proofs IMMEDIATELY (GPU 1, non-blocking)
            #    Proofs only need token IDs + prompt lens, NOT env rewards.
            #    This lets proofs run IN PARALLEL with kernel eval below.
            proof_start = time.time()
            proof_future = self._proof_executor.submit(
                self._proof_worker.compute_commitments_and_logprobs,
                gen_all_ids,
                gen_prompt_lens,
                randomness_hex,
                wallet,
            )

            # 5. Parallel kernel eval (runs in parallel with proofs)
            eval_start = time.time()

            def _eval_single(
                env_idx: int,
                _envs: list = envs,  # noqa: B006
                _all_ids: list = gen_all_ids,  # noqa: B006
                _prompt_lens: list = gen_prompt_lens,  # noqa: B006
            ) -> tuple[list[int], int, float, dict, float]:
                env = _envs[env_idx]
                all_ids = _all_ids[env_idx]
                prompt_len = _prompt_lens[env_idx]
                completion_ids = all_ids[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
                t0 = time.time()
                _next_obs, reward, _terminated, _truncated, info = env.step(
                    ChatMessage(role="assistant", content=completion_text)
                )
                return all_ids, prompt_len, float(reward), info, time.time() - t0

            eval_futures = [self._eval_executor.submit(_eval_single, i) for i in range(batch_count)]

            batch_data: list[tuple[list[int], int, float, dict]] = []
            per_eval_times: list[float] = []
            for future in eval_futures:
                all_ids, prompt_len, reward, info, eval_dur = future.result()
                batch_data.append((all_ids, prompt_len, reward, info))
                per_eval_times.append(eval_dur)

            eval_time = time.time() - eval_start
            total_eval_time += eval_time
            rewards = [d[2] for d in batch_data]
            logger.info(
                "TIMING kernel_eval_total=%.2fs (per_eval=[%s]) "
                "min/avg/max=%.2f/%.2f/%.2fs rewards=%s",
                eval_time,
                ", ".join(f"{t:.2f}" for t in per_eval_times),
                min(per_eval_times),
                sum(per_eval_times) / len(per_eval_times),
                max(per_eval_times),
                [f"{r:.2f}" for r in rewards],
            )

            all_batch_data.extend(batch_data)

            # 6. Collect proofs (wait for GPU 1 to finish)
            try:
                proof_results = proof_future.result(timeout=600)
                proof_time = time.time() - proof_start
                proof_wait = max(0, proof_time - eval_time)
                total_proof_time += proof_time
                total_proof_wait += proof_wait
                logger.info(
                    "TIMING proof=%.2fs (eval_overlap=%.2fs, extra_wait=%.2fs)",
                    proof_time,
                    min(proof_time, eval_time),
                    proof_wait,
                )
                all_proof_results.extend(proof_results)
            except Exception:
                logger.exception("Proof computation failed")
                # Remove the batch data for failed proofs
                all_batch_data = all_batch_data[: -len(batch_data)]

        # Check for infrastructure eval errors — if ANY rollout in the group
        # has an infra error, discard the ENTIRE group. Reducing group size
        # would cause validator group_size mismatch and hard failure.
        infra_errors = sum(
            1 for _, _, _, info in all_batch_data if info.get("eval_infra_error", False)
        )
        if infra_errors:
            logger.warning(
                "Discarding entire group: %d/%d rollouts had eval infrastructure errors",
                infra_errors,
                len(all_batch_data),
            )
            return []

        # Assemble rollouts
        t0 = time.time()
        rollouts = assemble_rollouts(all_batch_data, all_proof_results)

        # Compute advantages
        advantages = compute_advantages([r.reward for r in rollouts])
        for rollout, adv in zip(rollouts, advantages, strict=False):
            rollout.advantage = float(adv)
        assemble_time = time.time() - t0

        group_time = time.time() - group_start
        logger.info(
            "TIMING GROUP_TOTAL=%.2fs assemble=%.2fs | %d rollouts (%.1f rollouts/sec) "
            "| rewards: mean=%.3f min=%.2f max=%.2f",
            group_time,
            assemble_time,
            len(rollouts),
            len(rollouts) / group_time if group_time > 0 else 0,
            sum(r.reward for r in rollouts) / max(len(rollouts), 1),
            min((r.reward for r in rollouts), default=0),
            max((r.reward for r in rollouts), default=0),
        )

        # Emit structured timing log
        try:
            timing_entry = {
                "event": "group_timing",
                "gen_sec": round(total_gen_time, 2),
                "eval_sec": round(total_eval_time, 2),
                "proof_sec": round(total_proof_time, 2),
                "proof_wait_sec": round(total_proof_wait, 2),
                "group_sec": round(group_time, 2),
                "num_rollouts": len(rollouts),
                "total_completion_tokens": total_tokens,
                "tokens_per_sec": round(total_tokens / total_gen_time, 1)
                if total_gen_time > 0
                else 0.0,
            }
            timing_logger.info(json.dumps(timing_entry))
        except Exception:
            logger.debug("Failed to emit timing log", exc_info=True)

        return rollouts

    def submit_proofs(
        self,
        batch_data: list[tuple[list[int], int, float, dict]],
        randomness_hex: str,
        wallet: Any,
    ) -> Future[list[tuple[list[dict], list[float], bytes, dict, str]]]:
        """Non-blocking proof submission to GPU 1."""
        all_ids_batch = [data[0] for data in batch_data]
        prompt_lens = [data[1] for data in batch_data]

        return self._proof_executor.submit(
            self._proof_worker.compute_commitments_and_logprobs,
            all_ids_batch,
            prompt_lens,
            randomness_hex,
            wallet,
        )

    def shutdown(self) -> None:
        """Release resources."""
        self._proof_executor.shutdown(wait=False)
        self._eval_executor.shutdown(wait=False)
        self._proof_worker.shutdown()
        logger.info("PipelinedMiningEngine shutdown complete")
