"""Base classes for reward computation and parsing."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


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


class RewardVector:
    """Combines multiple reward functions with weights."""

    def __init__(
        self,
        reward_functions: list[Callable[[Any, Any], float]],
        weights: list[float],
        parser: Optional[Parser] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
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
        for reward_fn, weight in zip(self.reward_functions, self.weights):
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
        for (mn, mx), w in zip(self._bounds, self.weights):
            # Allow any real weights; composition follows linearity
            min_total += w * mn
            max_total += w * mx
        # Ensure ordering (if negative weights used, min/max might flip)
        low = min(min_total, max_total)
        high = max(min_total, max_total)
        return low, high
