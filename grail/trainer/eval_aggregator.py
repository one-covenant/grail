"""Streaming aggregation for evaluation metrics (pass@k, mean@k, best@k)."""

from __future__ import annotations

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


class EvalAggregator:
    """Accumulates per-task replicate results and computes k-metrics."""

    def __init__(self, *, report_ks: Iterable[int]) -> None:
        self._report_ks = sorted({int(k) for k in report_ks})
        self._by_task: dict[str, list[TaskReplicateResult]] = defaultdict(list)

    def add(self, result: TaskReplicateResult) -> None:
        self._by_task[result.task_id].append(result)

    def _ensure_sorted(self, results: list[TaskReplicateResult]) -> None:
        results.sort(key=lambda r: r.replicate_idx)

    def summarize(self) -> dict[str, float]:
        """Compute aggregated metrics across all tasks for requested k values."""
        if not self._by_task:
            return {}

        # Per-k accumulators
        pass_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore
        mean_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore
        best_at_k: dict[int, float] = dict.fromkeys(self._report_ks, 0.0)  # type: ignore

        num_tasks = len(self._by_task)

        for _task_id, reps in self._by_task.items():
            self._ensure_sorted(reps)
            # Precompute prefix windows
            successes = [1.0 if r.success else 0.0 for r in reps]
            rewards = [float(r.reward) for r in reps]

            for k in self._report_ks:
                k_slice = slice(0, min(k, len(reps)))
                window_success = 1.0 if any(successes[k_slice]) else 0.0
                window_mean = sum(rewards[k_slice]) / max(1, min(k, len(reps)))
                window_best = max(rewards[k_slice]) if reps[k_slice] else 0.0

                pass_at_k[k] += window_success
                mean_at_k[k] += window_mean
                best_at_k[k] += window_best

        # Normalize across tasks
        out: dict[str, float] = {}
        for k in self._report_ks:
            out[f"pass@{k}"] = pass_at_k[k] / num_tasks
            out[f"mean@{k}"] = mean_at_k[k] / num_tasks
            out[f"best@{k}"] = best_at_k[k] / num_tasks

        return out
