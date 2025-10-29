"""Vectorized environment utilities for batched evaluation.

Provides EnvVector that manages a batch of MultiTurnEnv instances and offers
reset/step helpers operating on lists. This keeps evaluation code concise and
efficient while preserving the existing environment contract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .core import ChatMessage, MultiTurnEnv, Observation


class EnvVector:
    """Lightweight vector wrapper around a set of environments.

    Environments are instantiated lazily via the provided factory and reused
    between calls. The wrapper is stateless with respect to tasks and only
    helps coordinate batch reset and step calls.
    """

    def __init__(self, env_factory: Callable[[], MultiTurnEnv], batch_size: int) -> None:
        self._env_factory = env_factory
        self._batch_size = int(batch_size)
        self._envs: list[MultiTurnEnv] = []

    @property
    def envs(self) -> list[MultiTurnEnv]:
        return self._envs

    def _ensure_envs(self, count: int) -> None:
        while len(self._envs) < count:
            self._envs.append(self._env_factory())

    def reset_ids(self, ids: Sequence[str], *, seed: int | None = None) -> list[Observation]:
        """Reset environments for the provided task IDs.

        If more IDs than the internal pool size are provided, new envs are
        created (up to len(ids)).
        """
        num = len(ids)
        self._ensure_envs(num)
        obs_list: list[Observation] = []
        for i in range(num):
            obs = self._envs[i].reset(task_id=ids[i], seed=seed)
            obs_list.append(obs)
        return obs_list

    def step_texts(
        self, texts: Sequence[str]
    ) -> list[tuple[Observation, float, bool, bool, dict[str, Any]]]:
        """Step environments with assistant messages constructed from texts."""
        assert len(texts) <= len(self._envs)
        results: list[tuple[Observation, float, bool, bool, dict[str, Any]]] = []
        for i, text in enumerate(texts):
            action = ChatMessage(role="assistant", content=text)
            results.append(self._envs[i].step(action))
        return results
