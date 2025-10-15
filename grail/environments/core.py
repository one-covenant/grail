"""Core types and base classes for step-only multi-turn environments.

This module defines the minimal, Gymnasium-style API used across single-turn
and multi-turn environments, plus tiny DI interfaces for task sourcing and
rewarding. Parsing reuses the existing `Parser` from `grail.environments.base`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Observation:
    messages: list[ChatMessage]
    available_tools: list[str]
    turn_index: int
    task_meta: dict[str, Any]


@dataclass(frozen=True)
class RewardBreakdown:
    components: dict[str, float]
    scalar: float


class TaskSource(Protocol):
    def next(self, *, seed: int | None = None, task_id: str | None = None) -> Any:
        """Return the next task instance given an optional seed or ID."""
        ...


class Rubric(Protocol):
    def step_reward(
        self, *, parsed: Any, context: Any, turn_index: int
    ) -> tuple[float, dict[str, float]]:
        """Compute step reward and component dictionary for diagnostics."""
        ...


class MultiTurnEnv(ABC):
    """Abstract base for step-only multi-turn environments.

    Concrete implementations must provide reset and step methods.
    Single-turn tasks can configure max_turns=1 or inherit SingleTurnEnv.
    """

    @abstractmethod
    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: ChatMessage) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Take one step; return (observation, reward, terminated, truncated, info)."""
        pass


class SingleTurnEnv(MultiTurnEnv):
    """Template-method base for single-turn environments.

    Enforces single-step termination. Subclasses implement _do_reset and
    _do_step; base wraps them to force terminated=True after first step.
    """

    def __init__(self) -> None:
        self._has_stepped = False

    def reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        """Reset and delegate to child's _do_reset."""
        self._has_stepped = False
        return self._do_reset(task_id=task_id, seed=seed)

    def step(self, action: ChatMessage) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Step once and force terminated=True."""
        obs, reward, truncated, info = self._do_step(action)
        self._has_stepped = True
        terminated = True
        return obs, reward, terminated, truncated, info

    @abstractmethod
    def _do_reset(self, *, task_id: str | None = None, seed: int | None = None) -> Observation:
        """Child implements task selection and initial observation."""
        pass

    @abstractmethod
    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Child implements step logic; returns (obs, reward, truncated, info).

        Base will force terminated=True; child returns truncated if budget hit.
        """
        pass
