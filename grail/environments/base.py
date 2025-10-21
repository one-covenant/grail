"""Base classes for reward computation and parsing."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Parser(ABC):
    """Base class for parsing completions into structured outputs."""

    @abstractmethod
    def parse(self, completion: str, context: Any) -> Any:
        """Parse completion text into structured output.

        Args:
            completion: The raw text completion from the model
            context: Additional context (e.g., problem instance) needed for
                parsing

        Returns:
            Parsed structured output that can be consumed by reward functions
        """
        pass


class ThinkingParser(Parser):
    """Base parser for environments using thinking + answer tags.

    Provides shared logic for detecting thinking and answer blocks using
    standard GRAIL prompt constants:
    - Thinking: <start_working_out>...</end_working_out>
    - Answer: <SOLUTION>...</SOLUTION>

    Validates proper ordering: thinking should come before answer.
    Subclasses should override parse() to define complete env-specific parsing.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded prompt constants."""
        super().__init__()
        self._constants_loaded = False
        self._tag_think_open: str = ""
        self._tag_think_close: str = ""
        self._tag_solution_open: str = ""
        self._tag_solution_close: str = ""

    def _ensure_constants_loaded(self) -> None:
        """Lazy load prompt constants to avoid circular import."""
        if self._constants_loaded:
            return

        from ..shared.prompt_constants import (
            REASONING_END_TOKEN,
            REASONING_START_TOKEN,
            SOLUTION_END_TOKEN,
            SOLUTION_START_TOKEN,
        )

        self._tag_think_open = REASONING_START_TOKEN
        self._tag_think_close = REASONING_END_TOKEN
        self._tag_solution_open = SOLUTION_START_TOKEN
        self._tag_solution_close = SOLUTION_END_TOKEN
        self._constants_loaded = True

    def _get_thinking_pattern(self) -> re.Pattern[str]:
        """Get compiled thinking block pattern."""
        self._ensure_constants_loaded()
        return re.compile(
            (rf"<{self._tag_think_open}>" r".*?" rf"</{self._tag_think_close}>"),
            re.IGNORECASE | re.DOTALL,
        )

    def _get_answer_pattern(self) -> re.Pattern[str]:
        """Get compiled answer block pattern."""
        self._ensure_constants_loaded()
        return re.compile(
            (
                rf"<{self._tag_solution_open}>"
                r"(?P<content>.*?)"
                rf"</{self._tag_solution_close}>"
            ),
            re.IGNORECASE | re.DOTALL,
        )

    def _get_think_then_answer_pattern(self) -> re.Pattern[str]:
        """Get compiled thinking→answer pattern (strict ordering)."""
        self._ensure_constants_loaded()
        return re.compile(
            (
                rf"<{self._tag_think_open}>"
                r".*?"
                rf"</{self._tag_think_close}>"
                r".*?"
                rf"<{self._tag_solution_open}>"
                r"(?P<content>.*?)"
                rf"</{self._tag_solution_close}>"
            ),
            re.IGNORECASE | re.DOTALL,
        )

    def _detect_thinking_block(self, text: str) -> bool:
        """Check if text contains thinking block. Returns True if found."""
        return bool(self._get_thinking_pattern().search(text or ""))

    def _detect_answer_block(self, text: str) -> bool:
        """Check if text contains answer block. Returns True if found."""
        return bool(self._get_answer_pattern().search(text or ""))

    def _get_thinking_block(self, text: str) -> str | None:
        """Extract thinking block content if present. Returns None otherwise."""
        match = self._get_thinking_pattern().search(text or "")
        if match:
            return match.group(0)
        return None

    def _get_answer_block(self, text: str) -> str | None:
        """Extract answer block content if present. Returns None otherwise."""
        match = self._get_answer_pattern().search(text or "")
        if match:
            return match.group("content")
        return None

    def _get_answer_with_thinking_check(self, text: str) -> tuple[str | None, int, bool]:
        """Extract answer, preferring thinking→answer ordering.

        Returns:
            (answer_content, trailing_chars, has_proper_ordering)
            - answer_content: None if no answer found
            - trailing_chars: chars after </SOLUTION> tag
            - has_proper_ordering: True if thinking comes before answer
        """
        text = text or ""

        # Check for proper ordering: thinking→answer
        if self._detect_thinking_block(text):
            match = self._get_think_then_answer_pattern().search(text)
            if match:
                return match.group("content"), len(text) - match.end(), True

        # Fallback: answer without thinking
        match = self._get_answer_pattern().search(text)
        if match:
            return match.group("content"), len(text) - match.end(), False

        return None, 0, False


class RewardVector:
    """Combines multiple reward functions with weights."""

    def __init__(
        self,
        reward_functions: list[Callable[[Any, Any], float]],
        weights: list[float],
        parser: Parser | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ):
        """Initialize reward vector.

        Args:
            reward_functions: List of functions that take (parsed_output,
                            context) and return float rewards
            weights: Weights for each reward function (should sum to 1.0)
            parser: Optional parser to preprocess completions before reward
                   computation
            bounds: Optional per-function bounds [(min, max), ...] aligned to
                    reward_functions. If provided, can be used to compose a
                    total reward bound.
        """
        if len(reward_functions) != len(weights):
            raise ValueError("Number of reward functions must match number of weights")

        self.reward_functions = reward_functions
        self.weights = weights
        self.parser = parser
        self._bounds = bounds

    def compute_reward(self, completion: str, context: Any) -> float:
        """Compute weighted sum of all reward functions.

        Args:
            completion: Raw text completion from model
            context: Problem context passed to parser and reward functions

        Returns:
            Weighted sum of all reward function outputs
        """
        if self.parser:
            parsed_output = self.parser.parse(completion, context)
        else:
            parsed_output = completion

        total_reward = 0.0
        for reward_fn, weight in zip(self.reward_functions, self.weights, strict=False):
            reward = reward_fn(parsed_output, context)
            total_reward += weight * reward

        return total_reward

    def compute_individual_rewards(self, completion: str, context: Any) -> list[float]:
        """Compute individual rewards from each function (useful for analysis).

        Args:
            completion: Raw text completion from model
            context: Problem context passed to parser and reward functions

        Returns:
            List of individual reward values (before weighting)
        """
        if self.parser:
            parsed_output = self.parser.parse(completion, context)
        else:
            parsed_output = completion

        rewards = []
        for reward_fn in self.reward_functions:
            reward = reward_fn(parsed_output, context)
            rewards.append(reward)

        return rewards

    # --------------------------------- Bounds --------------------------------
    def has_bounds(self) -> bool:
        """Return True if per-function bounds metadata was provided."""
        return self._bounds is not None

    def reward_bounds(self) -> tuple[float, float]:
        """Compose total reward bounds from per-function bounds and weights.

        Returns:
            (min_total, max_total)

        Raises:
            ValueError: if bounds metadata is missing or malformed.
        """
        if self._bounds is None:
            raise ValueError("No bounds metadata available for this RewardVector")
        if len(self._bounds) != len(self.reward_functions) or len(self._bounds) != len(
            self.weights
        ):
            raise ValueError("Bounds must align with reward functions and weights")

        min_total = 0.0
        max_total = 0.0
        for (mn, mx), w in zip(self._bounds, self.weights, strict=False):
            # Allow any real weights; composition follows linearity
            min_total += w * mn
            max_total += w * mx
        # Ensure ordering (if negative weights used, min/max might flip)
        low = min(min_total, max_total)
        high = max(min_total, max_total)
        return low, high
