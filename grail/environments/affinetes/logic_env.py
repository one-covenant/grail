"""Affinetes lgc-v2 logic environment -- solve logic puzzles."""

from __future__ import annotations

from typing import Any

from ..base import RewardVector
from ..core import ChatMessage, Observation, SingleTurnEnv
from ..providers import TaskSpec
from ..reward_components import answer_format_reward, no_trailing_reward, thinking_format_reward
from ..rubric import RewardVectorRubric
from .parsers import LogicCompletionParser
from .rewards import logic_correctness_reward
from .task_sources import LogicTaskSource


class AffineLogicEnv(SingleTurnEnv):
    """Single-turn env: solve logic puzzles (8 task types from lgc-v2)."""

    def __init__(self, *, task_source: LogicTaskSource | None = None) -> None:
        super().__init__()
        self._source = task_source or LogicTaskSource()
        self._parser = LogicCompletionParser()
        self._rubric = RewardVectorRubric(self._create_reward_vector())
        self._task: TaskSpec | None = None

    def _create_reward_vector(self) -> RewardVector:
        return RewardVector(
            reward_functions=[
                logic_correctness_reward,
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
        ground_truth = self._task.payload["ground_truth"]

        # Compute decomposed reward via rubric (context = ground_truth dict)
        reward, components = self._rubric.step_reward(
            parsed=completion, context=ground_truth, turn_index=1,
        )

        # Correctness for info dict (uses cached verifiers, inline)
        answer = self._parser._get_answer_block(completion)
        success = False
        if answer and isinstance(ground_truth, dict):
            from .rewards import _get_verifier

            task_type = ground_truth.get("task_type", "")
            verifier = _get_verifier(task_type)
            game_data = ground_truth.get("game_data")
            if verifier and game_data:
                try:
                    from ._loader import load_logic_verifiers

                    _, Data = load_logic_verifiers()
                    data = (
                        Data.from_json(game_data) if isinstance(game_data, dict) else game_data
                    )
                    success = verifier.verify(data, answer)
                except Exception:
                    pass

        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["question"]),
                ChatMessage(role="assistant", content=completion),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return obs, float(reward), False, {
            "reward_components": components,
            "success": success,
            "termination_cause": "final",
            "task_type": self._task.metadata.get("task_type", "unknown"),
        }
