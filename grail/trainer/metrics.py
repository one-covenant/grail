"""Centralized k-metrics computation for training and evaluation.

This module provides a single, DRY implementation of k-metrics used in both
evaluation and training data inspection. It computes:

- pass@k (unbiased): Unbiased estimator of probability that at least one of k
  samples succeeds, given n total replicates with c successes for a task.
  We use the estimator from Chen et al. (Evaluating LLMs trained on code):
  pass@k = 1 - C(n - c, k) / C(n, k) for k ≤ n; if k > n we clamp k = n.
  This avoids bias from replicate ordering and is the recommended pass@k.

- pass_ordered@k: Fraction of tasks with any success in the first k replicates
  in replicate order. Useful for "first-k attempts" diagnostics but sensitive
  to ordering; not unbiased. Included for operational visibility.

- mean@k / best@k: Per-task reward mean/best over the first k replicates,
  then averaged across tasks. Diagnostic metrics (ordering-sensitive).

- reward_mean_all / reward_std_all / success_rate_all: Global, replicate-level
  aggregates across all tasks and replicates (not task-averaged). These are
  stable summaries of rollout quality.
- reward_mean_taskavg_all / success_rate_taskavg_all: Task-averaged versions of
  the above (each task contributes equally regardless of its replicate count).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class TaskReplicateResult:
    task_id: str
    replicate_idx: int
    reward: float
    success: bool
    components: dict[str, float] | None = None


def derive_k_values(rollouts_per_problem: int) -> list[int]:
    """Derive standard k values given rollouts-per-problem.

    Ensures values are unique, sorted, and bounded by rollouts_per_problem.
    """
    rpp = max(1, int(rollouts_per_problem))
    ks = {1, min(5, rpp), min(10, rpp), rpp}
    return sorted(ks)


class KMetricsAggregator:
    """Accumulates per-task replicate results and computes k-metrics.

    Metrics:
    - pass@k: unbiased estimator per Chen et al. (order-invariant)
    - pass_ordered@k: fraction of tasks with any success in the first k replicates (diagnostic)
    - mean@k: per-task mean over first k replicates, averaged across tasks
    - best@k: per-task best over first k replicates, averaged across tasks
    - reward_mean_all: mean reward across all replicates (global)
    - reward_std_all: std deviation of rewards across all replicates (global, population)
    - success_rate_all: fraction of successful replicates across all replicates (global)
    - tasks: number of distinct tasks
    - replicates: total number of replicate results
    """

    def __init__(self, *, report_ks: Iterable[int]) -> None:
        self._report_ks = sorted({int(k) for k in report_ks})
        self._by_task: dict[str, list[TaskReplicateResult]] = defaultdict(list)

    def add(self, result: TaskReplicateResult) -> None:
        """Add a single replicate result for a task.

        Each task can have multiple replicate results; replicate_idx should be
        contiguous from 0..n-1 to allow deterministic prefix windows.
        """
        self._by_task[result.task_id].append(result)

    def add_group(self, task_id: str, rewards: list[float], successes: list[bool]) -> None:
        """Add a group of replicate results for one task.

        The order of input lists defines the replicate order (used by prefix metrics).
        Lengths may differ; we iterate over the shortest with zip(strict=False).
        """
        for idx, (rw, ok) in enumerate(zip(rewards, successes, strict=False)):
            self.add(
                TaskReplicateResult(
                    task_id=task_id,
                    replicate_idx=idx,
                    reward=float(rw),
                    success=bool(ok),
                )
            )

    def _ensure_sorted(self, results: list[TaskReplicateResult]) -> None:
        results.sort(key=lambda r: r.replicate_idx)

    def summarize(self) -> dict[str, float]:
        """Summarize all added results into a metrics dictionary.

        Returns keys:
        - pass@k (unbiased), pass_ordered@k, mean@k, best@k for k in report_ks
        - reward_mean_all, reward_std_all, success_rate_all
        - reward_mean_taskavg_all, success_rate_taskavg_all
        - tasks, replicates
        """
        if not self._by_task:
            return {}

        # Task-averaged accumulators
        pass_unbiased_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore
        pass_ordered_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore
        mean_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore
        best_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore

        num_tasks = len(self._by_task)

        # Global accumulators for overall stats
        all_rewards: list[float] = []
        all_successes: list[float] = []

        for _task_id, reps in self._by_task.items():
            self._ensure_sorted(reps)
            # Per-task arrays in replicate order
            successes = [1.0 if r.success else 0.0 for r in reps]
            rewards = [float(r.reward) for r in reps]

            all_rewards.extend(rewards)
            all_successes.extend(successes)

            n = len(reps)
            c = int(sum(successes))
            for k in self._report_ks:
                if n == 0:
                    continue
                # Clamp k to at most n for combinatorics
                kk = min(k, n)

                # Unbiased pass@k per Chen et al.: 1 - C(n-c, kk)/C(n, kk)
                if c <= 0:
                    unbiased = 0.0
                elif kk <= 0:
                    unbiased = 0.0
                elif kk > n:
                    unbiased = 1.0 if c > 0 else 0.0
                else:
                    try:
                        numerator = math.comb(n - c, kk)
                        denominator = math.comb(n, kk)
                        unbiased = 1.0 - (numerator / denominator if denominator > 0 else 0.0)
                    except ValueError:
                        # Defensive: if comb inputs invalid (shouldn't happen after clamps)
                        unbiased = 0.0
                pass_unbiased_at_k[k] += float(unbiased)

                # Ordering-sensitive diagnostics
                k_slice = slice(0, kk)
                window_success = 1.0 if any(successes[k_slice]) else 0.0
                denom = max(1, kk)
                window_mean = sum(rewards[k_slice]) / denom
                window_best = max(rewards[k_slice]) if rewards[k_slice] else 0.0

                pass_ordered_at_k[k] += window_success
                mean_at_k[k] += window_mean
                best_at_k[k] += window_best

        # Normalize task-averaged metrics
        out: dict[str, float] = {}
        for k in self._report_ks:
            # Preferred, order-invariant estimate
            out[f"pass@{k}"] = pass_unbiased_at_k[k] / num_tasks

            # Diagnostic, order-sensitive metrics
            out[f"pass_ordered@{k}"] = pass_ordered_at_k[k] / num_tasks
            out[f"mean@{k}"] = mean_at_k[k] / num_tasks
            out[f"best@{k}"] = best_at_k[k] / num_tasks

        # Global stats across all replicates
        n = len(all_rewards)
        if n > 0:
            mean_all = sum(all_rewards) / n
            # population variance
            var_all = sum((x - mean_all) ** 2 for x in all_rewards) / n
            std_all = var_all**0.5
            success_rate = (sum(all_successes) / len(all_successes)) if all_successes else 0.0
        else:
            mean_all = 0.0
            std_all = 0.0
            success_rate = 0.0

        out["reward_mean_all"] = float(mean_all)
        out["reward_std_all"] = float(std_all)
        out["success_rate_all"] = float(success_rate)
        out["tasks"] = float(num_tasks)
        out["replicates"] = float(n)

        # Task-averaged global stats (avoid overweighting tasks with more replicates)
        # For GRPO groups with fixed RPP, this equals global, but we include it for generality.
        if num_tasks > 0:
            means_per_task = []
            success_rates_per_task = []
            for _task_id, reps in self._by_task.items():
                if not reps:
                    continue
                rewards = [float(r.reward) for r in reps]
                successes = [1.0 if r.success else 0.0 for r in reps]
                means_per_task.append(sum(rewards) / len(rewards))
                success_rates_per_task.append(sum(successes) / len(successes))

            if means_per_task:
                out["reward_mean_taskavg_all"] = sum(means_per_task) / len(means_per_task)
            else:
                out["reward_mean_taskavg_all"] = 0.0

            if success_rates_per_task:
                out["success_rate_taskavg_all"] = sum(success_rates_per_task) / len(
                    success_rates_per_task
                )
            else:
                out["success_rate_taskavg_all"] = 0.0
        else:
            out["reward_mean_taskavg_all"] = 0.0
            out["success_rate_taskavg_all"] = 0.0

        return out
