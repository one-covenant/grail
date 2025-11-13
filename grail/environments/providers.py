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
    """Procedural SAT task provider wrapping the existing generator.

    Difficulty is derived deterministically from the seed to ensure
    miner and validator generate identical problems.
    """

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        if task_id is not None:
            seed_material = task_id
        elif seed is not None:
            seed_material = str(seed)
        else:
            # Fallback deterministic material if not provided (not random here)
            seed_material = "sat-default-seed"

        # Derive difficulty deterministically from seed
        # Use modulo to map seed to difficulty range [0.3, 0.7]
        seed_int = int(seed) if seed is not None else 12345
        difficulty = 0.3 + (seed_int % 100) / 250.0  # Maps to [0.3, 0.7]

        # Lazy import to avoid circular dependency with sat_env
        from .sat_env import generate_sat_problem

        problem = generate_sat_problem(seed_material, difficulty)
        return TaskSpec(
            id=str(seed_material),
            payload=problem,
            metadata={"difficulty": difficulty},
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
            # Uniform sampling using floating-point scaling to avoid modular bias
            # Convert seed to [0,1) then scale to dataset size
            seed_normalized = (int(seed) % (2**32)) / (2**32)
            idx = int(seed_normalized * len(self._ds))
        else:
            idx = 0

        sample = self._ds[int(idx) % len(self._ds)]
        # GSM8K fields: {question, answer}
        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={"question": sample["question"], "answer": sample["answer"]},
            metadata={"split": self._split, "index": int(idx)},
        )

    # --- Evaluation helpers ---
    def size(self) -> int:
        self._ensure_dataset()
        assert self._ds is not None
        return len(self._ds)

    def iter_ids(self) -> list[str]:
        self._ensure_dataset()
        assert self._ds is not None
        return [f"{self._split}_{i}" for i in range(len(self._ds))]


class MATHTaskSource(TaskSource):
    """HF datasets-backed Hendrycks MATH provider with filtering support.

    Supports filtering by:
    - level: int (1-5) - difficulty level
    - subject: str - mathematical domain (Algebra, Geometry, etc.)

    Lazily loads and filters dataset on first call.
    """

    def __init__(self, *, split: str = "train") -> None:
        self._split = split
        self._ds = None
        self._filtered_indices: dict[str, list[int]] = {}

    def _ensure_dataset(self) -> None:
        if self._ds is None:
            from datasets import load_dataset

            self._ds = load_dataset("nlile/hendrycks-MATH-benchmark", split=self._split)

    def _get_filtered_indices(
        self, level: int | None = None, subject: str | None = None
    ) -> list[int]:
        """Get dataset indices matching filter criteria.

        Caches filtered indices to avoid repeated filtering.

        Args:
            level: Difficulty level (1-5)
            subject: Mathematical subject

        Returns:
            List of dataset indices matching filters
        """
        self._ensure_dataset()
        assert self._ds is not None

        # Build cache key
        cache_key = f"level={level},subject={subject}"
        if cache_key in self._filtered_indices:
            return self._filtered_indices[cache_key]

        # Filter dataset
        indices = []
        for i in range(len(self._ds)):
            sample = self._ds[i]
            if level is not None and sample["level"] != level:
                continue
            if subject is not None and sample["subject"] != subject:
                continue
            indices.append(i)

        self._filtered_indices[cache_key] = indices
        return indices

    def next(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
        level: int | None = None,
        subject: str | None = None,
    ) -> TaskSpec:
        """Sample task with optional filtering.

        Args:
            seed: Random seed for sampling
            task_id: Specific task ID (format: split_index)
            level: Filter by difficulty level (1-5)
            subject: Filter by mathematical subject

        Returns:
            TaskSpec with problem, solution, and metadata
        """
        self._ensure_dataset()
        assert self._ds is not None

        # Get filtered indices if filters provided
        if level is not None or subject is not None:
            valid_indices = self._get_filtered_indices(level, subject)
            if not valid_indices:
                raise ValueError(f"No samples match filters: level={level}, subject={subject}")
        else:
            valid_indices = list(range(len(self._ds)))

        # Select index
        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except Exception:
                idx = valid_indices[0]
        elif seed is not None:
            # Uniform sampling within filtered set
            seed_normalized = (int(seed) % (2**32)) / (2**32)
            idx = valid_indices[int(seed_normalized * len(valid_indices))]
        else:
            idx = valid_indices[0]

        # Get sample
        sample = self._ds[int(idx) % len(self._ds)]

        # MATH fields: {problem, solution, answer, subject, level, unique_id}
        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={
                "question": sample["problem"],
                "solution": sample["solution"],
                "answer": sample["answer"],
            },
            metadata={
                "split": self._split,
                "index": int(idx),
                "level": int(sample["level"]),
                "subject": str(sample["subject"]),
                "unique_id": str(sample["unique_id"]),
            },
        )

    def size(self, level: int | None = None, subject: str | None = None) -> int:
        """Get dataset size with optional filtering.

        Args:
            level: Filter by difficulty level
            subject: Filter by subject

        Returns:
            Number of samples matching filters
        """
        if level is not None or subject is not None:
            return len(self._get_filtered_indices(level, subject))
        self._ensure_dataset()
        assert self._ds is not None
        return len(self._ds)

    def iter_ids(self, level: int | None = None, subject: str | None = None) -> list[str]:
        """Get all task IDs with optional filtering.

        Args:
            level: Filter by difficulty level
            subject: Filter by subject

        Returns:
            List of task IDs matching filters
        """
        if level is not None or subject is not None:
            indices = self._get_filtered_indices(level, subject)
            return [f"{self._split}_{i}" for i in indices]
        self._ensure_dataset()
        assert self._ds is not None
        return [f"{self._split}_{i}" for i in range(len(self._ds))]
