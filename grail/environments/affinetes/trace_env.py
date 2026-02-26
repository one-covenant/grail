"""Affinetes trace (print) environment -- predict stdout of Python code."""

from __future__ import annotations

from typing import Any

from ..base import RewardVector
from ..core import ChatMessage, Observation, SingleTurnEnv
from ..providers import TaskSpec
from ..reward_components import answer_format_reward, no_trailing_reward, thinking_format_reward
from ..rubric import RewardVectorRubric
from .parsers import TraceCompletionParser
from .rewards import _compare_outputs_normalized, trace_correctness_reward
from .task_sources import TraceTaskSource


class AffineTraceEnv(SingleTurnEnv):
    """Single-turn env: predict stdout of Python code with print statements."""

    def __init__(self, *, task_source: TraceTaskSource | None = None) -> None:
        super().__init__()
        self._source = task_source or TraceTaskSource()
        self._parser = TraceCompletionParser()
        self._rubric = RewardVectorRubric(self._create_reward_vector())
        self._task: TaskSpec | None = None

    def _create_reward_vector(self) -> RewardVector:
        return RewardVector(
            reward_functions=[
                trace_correctness_reward,
                thinking_format_reward,
                answer_format_reward,
                no_trailing_reward,
            ],
            weights=[0.6, 0.15, 0.15, 0.1],
            parser=self._parser,
            bounds=[(0, 1), (0, 0.5), (0, 0.3), (0, 0.2)],
        )

    def _do_reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        self._task = self._source.next(seed=seed, task_id=task_id)
        return Observation(
            messages=[ChatMessage(role="user", content=self._task.payload["question"])],
            available_tools=[],
            turn_index=0,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        assert self._task is not None, "Must call reset() before step()"

        completion = action.content or ""
        expected = self._task.payload["expected_output"]

        # Compute decomposed reward via rubric (context = expected_output string)
        reward, components = self._rubric.step_reward(
            parsed=completion,
            context=expected,
            turn_index=1,
        )

        # Correctness for info dict (inline, no affinetes call)
        answer = self._parser._get_answer_block(completion)
        success = _compare_outputs_normalized(answer, expected) if answer else False

        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["question"]),
                ChatMessage(role="assistant", content=completion),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return (
            obs,
            float(reward),
            False,
            {
                "reward_components": components,
                "success": success,
                "termination_cause": "final",
            },
        )
