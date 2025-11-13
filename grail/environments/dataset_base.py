"""Base template class for single-turn math dataset environments.

This module provides an abstract base class that captures common patterns across
math dataset environments (GSM8K, MATH, FinanceQA, etc.) while allowing
dataset-specific customization through template methods.

Key extension points:
- Answer extraction from dataset records
- Answer extraction from model completions
- Answer validation strategies
- Metadata filtering (level, subject, etc.)
- Reward vector construction
"""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any

from .base import Parser, RewardVector
from .core import ChatMessage, Observation, Rubric, SingleTurnEnv
from .providers import TaskSource, TaskSpec
from .rubric import RewardVectorRubric


class MathDatasetEnv(SingleTurnEnv):
    """Abstract base class for single-turn math dataset environments.

    Template Method Pattern: Provides skeleton algorithm for environment lifecycle
    (reset, step, reward) while delegating dataset-specific logic to subclasses.

    Subclasses must implement:
    - _extract_dataset_answer(): Parse gold answer from dataset
    - _extract_completion_answer(): Parse predicted answer from completion
    - _default_task_source(): Return dataset-specific task source
    - _create_reward_vector(): Return dataset-specific reward configuration

    Subclasses may override:
    - _validate_answer(): Custom validation logic
    - _normalize_answer(): Custom normalization
    - _build_task_filter(): Support metadata filtering
    - _create_parser(): Custom completion parser
    """

    def __init__(
        self,
        *,
        task_source: TaskSource | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
    ) -> None:
        """Initialize environment with optional component overrides.

        Args:
            task_source: Dataset provider. Uses _default_task_source() if None.
            parser: Completion parser. Uses _create_parser() if None.
            rubric: Reward rubric. Uses RewardVectorRubric if None.
        """
        super().__init__()
        self._source = task_source or self._default_task_source()
        self._parser = parser or self._create_parser()
        self._rubric = rubric or RewardVectorRubric(self._create_reward_vector())
        self._task: TaskSpec | None = None

    # =========================================================================
    # Public API overrides
    # =========================================================================

    def reset(  # type: ignore[override]
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        **filter_kwargs: Any,
    ) -> Observation:
        """Reset environment with optional dataset-specific filters."""
        self._has_stepped = False
        return self._do_reset(task_id=task_id, seed=seed, **filter_kwargs)

    # =========================================================================
    # Template Methods - Dataset-Specific Logic (override in subclasses)
    # =========================================================================

    @abstractmethod
    def _extract_dataset_answer(self, task_payload: dict[str, Any]) -> str:
        """Extract gold answer from dataset record.

        Called during step() to get ground truth for validation.

        Examples:
            GSM8K: Parse '#### 42' from solution field
            MATH: Return payload['answer'] directly
            FinanceQA: Extract from structured answer field

        Args:
            task_payload: Raw dataset record (problem, solution, metadata)

        Returns:
            Gold answer string (pre-normalization)
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_completion_answer(self, completion: str, context: dict[str, Any]) -> str | None:
        """Extract predicted answer from model completion.

        Called during step() to parse model's final answer.

        Examples:
            GSM8K: Extract number from <SOLUTION>...</SOLUTION>
            MATH: Extract from \\boxed{...}
            Custom: Any regex or parsing logic

        Args:
            completion: Full model completion text
            context: Task payload for context-aware parsing

        Returns:
            Predicted answer string if found, None if extraction failed
        """
        raise NotImplementedError

    @abstractmethod
    def _default_task_source(self) -> TaskSource:
        """Create default task source for this dataset.

        Called during __init__ if no task_source provided.

        Returns:
            Dataset-specific TaskSource instance
        """
        raise NotImplementedError

    @abstractmethod
    def _create_reward_vector(self) -> RewardVector:
        """Create dataset-specific reward vector.

        Called during __init__ to configure decomposed rewards.

        Returns:
            RewardVector with dataset-appropriate reward functions and weights
        """
        raise NotImplementedError

    def _validate_answer(self, predicted: str, gold: str) -> bool:
        """Compare predicted answer against gold answer.

        Default: Normalized exact string match.
        Override for dataset-specific validation (symbolic, numeric, fuzzy).

        Args:
            predicted: Extracted model answer (raw)
            gold: Dataset gold answer (raw)

        Returns:
            True if answers are equivalent
        """
        pred_norm = self._normalize_answer(predicted)
        gold_norm = self._normalize_answer(gold)
        return pred_norm == gold_norm

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison.

        Default: Strip whitespace, lowercase, remove trailing punctuation.
        Override for dataset-specific normalization.

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer for comparison
        """
        return re.sub(r"[\s\.]+$", "", answer.strip().lower())

    def _build_task_filter(self, **filter_kwargs) -> dict[str, Any]:
        """Build filtering kwargs for task source based on metadata.

        Called during reset() to enable metadata-based task selection.

        Examples:
            GSM8K: {} (no filtering)
            MATH: {'level': 5, 'subject': 'Algebra'}
            AIME: {'year': 2023, 'difficulty': 'hard'}

        Args:
            **filter_kwargs: User-provided filter arguments from reset()

        Returns:
            Dictionary passed to task_source.next(**filters)
        """
        return {}

    def _create_parser(self) -> Parser:
        """Create dataset-specific completion parser.

        Override to provide custom parser for this dataset.
        Default returns None (use rubric's parser).

        Returns:
            Parser instance or None
        """
        return None  # type: ignore[return-value]

    # =========================================================================
    # Concrete Implementation - Reusable Logic
    # =========================================================================

    def _do_reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        **filter_kwargs: Any,
    ) -> Observation:
        """Reset environment with optional task selection and filtering.

        Args:
            task_id: Specific task ID to load (if supported by source)
            seed: Random seed for task sampling
            **filter_kwargs: Dataset-specific filters (level, subject, etc.)

        Returns:
            Initial observation with user message (problem statement)
        """
        # Build dataset-specific filters
        task_filters = self._build_task_filter(**filter_kwargs)
        task_filters.update({"seed": seed, "task_id": task_id})

        # Sample task from source
        self._task = self._source.next(**task_filters)

        # Create initial observation
        obs = Observation(
            messages=[ChatMessage(role="user", content=self._task.payload["question"])],
            available_tools=[],
            turn_index=0,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return obs

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute single turn: validate answer and compute reward.

        Args:
            action: Model's completion as ChatMessage

        Returns:
            Tuple of (observation, reward, truncated, info)
        """
        assert self._task is not None, "Must call reset() before step()"

        completion_text = action.content or ""

        # Extract answers using template methods
        predicted_answer = self._extract_completion_answer(completion_text, self._task.payload)
        gold_answer = self._extract_dataset_answer(self._task.payload)

        # Validate answer
        success = False
        if predicted_answer is not None:
            success = self._validate_answer(predicted_answer, gold_answer)

        # Compute decomposed reward via rubric
        reward, components = self._rubric.step_reward(
            parsed=completion_text,
            context=self._task.payload,
            turn_index=1,
        )

        # Build final observation
        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["question"]),
                ChatMessage(role="assistant", content=completion_text),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )

        # Info dict with validation results
        info = {
            "reward_components": components,
            "termination_cause": "final",
            "success": success,
            "gold_answer": gold_answer,
            "pred_answer": predicted_answer or "",
        }

        truncated = False
        return obs, float(reward), truncated, info
