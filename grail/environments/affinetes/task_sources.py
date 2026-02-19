"""Task sources backed by affinetes environment generators."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core import TaskSource
from ..providers import TaskSpec


class TraceTaskSource(TaskSource):
    """HF dataset-backed trace/print task source via affinetes.

    Loads satpalsr/rl-python dataset, injects deterministic print statements
    via AST, executes code to pre-compute expected stdout.
    Ground truth is computed at generate() time -- no code runs at step() time.
    """

    _instance = None  # Singleton TraceTask (holds HF dataset)

    def _ensure_task(self) -> None:
        if TraceTaskSource._instance is None:
            from ._loader import load_trace_task

            TraceTaskSource._instance = load_trace_task()()

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        self._ensure_task()
        effective_id = int(task_id) if task_id else (seed if seed is not None else 0)
        challenge = asyncio.run(self._instance.generate(task_id=effective_id))
        return TaskSpec(
            id=str(effective_id),
            payload={
                "question": challenge.prompt,
                "expected_output": challenge.extra.get("expected_output", ""),
            },
            metadata={"env": "affine_trace", "task_id": effective_id},
        )


class LogicTaskSource(TaskSource):
    """Procedural logic puzzle task source via affinetes lgc-v2.

    8 task types (dyck_language, game_of_24, operation, cryptarithm, etc.)
    encoded in task_id: task_id = task_type_id * 100_000_000 + seed.
    """

    _instance = None  # Singleton LogicTaskV2

    def _ensure_task(self) -> None:
        if LogicTaskSource._instance is None:
            from ._loader import load_logic_task

            LogicTaskSource._instance = load_logic_task()()

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        self._ensure_task()
        effective_id = int(task_id) if task_id else (seed if seed is not None else 0)
        challenge = asyncio.run(self._instance.generate(task_id=effective_id))
        return TaskSpec(
            id=str(effective_id),
            payload={
                "question": challenge.prompt,
                "ground_truth": challenge.extra,  # {game_data, task_type, seed, ...}
            },
            metadata={
                "env": "affine_logic",
                "task_id": effective_id,
                "task_type": challenge.extra.get("task_type", "unknown"),
            },
        )
