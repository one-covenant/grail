"""Task providers for environments (dataset-backed and procedural)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import TaskSource


@dataclass(frozen=True)
class TaskSpec:
    """Generic container for a task instance."""

    id: str
    payload: Any
    metadata: dict[str, Any]


class SATTaskSource(TaskSource):
    """Procedural SAT task provider wrapping the existing generator."""

    def __init__(self, *, difficulty: float = 0.5) -> None:
        self._difficulty = float(difficulty)

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        if task_id is not None:
            seed_material = task_id
        elif seed is not None:
            seed_material = str(seed)
        else:
            # Fallback deterministic material if not provided (not random here)
            seed_material = "sat-default-seed"

        # Lazy import to avoid circular dependency with sat_env
        from .sat_env import generate_sat_problem

        problem = generate_sat_problem(seed_material, self._difficulty)
        return TaskSpec(
            id=str(seed_material),
            payload=problem,
            metadata={"difficulty": self._difficulty},
        )


class GSM8KTaskSource(TaskSource):
    """HF datasets-backed GSM8K provider.

    Lazily loads the dataset on first call to avoid import overhead on
    module import.
    """

    def __init__(self, *, split: str = "train") -> None:
        self._split = split
        self._ds = None

    def _ensure_dataset(self) -> None:
        if self._ds is None:
            from datasets import load_dataset  # local import per workspace rules

            self._ds = load_dataset("openai/gsm8k", "main", split=self._split)

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        self._ensure_dataset()
        assert self._ds is not None

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except Exception:
                idx = 0
        elif seed is not None:
            # Deterministic index from seed
            idx = int(seed) % len(self._ds)
        else:
            idx = 0

        sample = self._ds[int(idx) % len(self._ds)]
        # GSM8K fields: {question, answer}
        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={"question": sample["question"], "answer": sample["answer"]},
            metadata={"split": self._split, "index": int(idx)},
        )
