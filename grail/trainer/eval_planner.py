"""Deterministic evaluation planning.

Produces ordered task IDs for dataset-backed evaluation or deterministic
synthetic IDs for generative providers, and derives per-replicate seeds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass


def _hash64(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class EvaluationPlan:
    """Static description of a single evaluation cycle."""

    ids: list[str]
    replicates: int
    cycle_index: int
    seed_base: int

    def seed_for(self, task_id: str, replicate_idx: int) -> int:
        material = f"{self.seed_base}:{self.cycle_index}:{task_id}:{replicate_idx}"
        return _hash64(material) & ((1 << 63) - 1)


class EvaluationPlanner:
    """Planner that creates EvaluationPlan objects per cycle.

    For dataset-backed providers, pass `enumerate_ids` returning all task IDs
    for a split. For generative providers, pass a callable producing deterministic
    IDs given cycle index and subset size.
    """

    def __init__(
        self,
        *,
        replicates: int,
        seed_base: int,
        enumerate_ids: Callable[[], Iterable[str]] | None = None,
        generate_ids: Callable[[int, int], Iterable[str]] | None = None,
    ) -> None:
        self._replicates = int(replicates)
        self._seed_base = int(seed_base)
        self._enumerate_ids = enumerate_ids
        self._generate_ids = generate_ids

    def for_cycle(self, cycle_index: int, *, subset_size: int | None = None) -> EvaluationPlan:
        if self._enumerate_ids is not None:
            ids = list(self._enumerate_ids())
        elif self._generate_ids is not None:
            if subset_size is None:
                raise ValueError("subset_size is required for generative ID generation")
            ids = list(self._generate_ids(cycle_index, subset_size))
        else:
            raise ValueError("No ID source provided for EvaluationPlanner")

        return EvaluationPlan(
            ids=ids,
            replicates=self._replicates,
            cycle_index=int(cycle_index),
            seed_base=self._seed_base,
        )
