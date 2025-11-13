"""Environment factory for clean instantiation and dependency injection."""

from __future__ import annotations

import logging
from typing import Any

from ..shared.constants import CURRENT_ENV_ID
from .core import MultiTurnEnv

logger = logging.getLogger(__name__)


def create_env(
    env_id: str | None = None,
    *,
    task_source: Any | None = None,
    split: str = "train",
) -> MultiTurnEnv:
    """Create environment instance with clean dependency injection.

    Args:
        env_id: Environment identifier (defaults to CURRENT_ENV_ID)
        task_source: Optional task source for dataset-backed envs
        split: Dataset split for GSM8K (train/test/validation)

    Returns:
        Initialized environment instance

    Raises:
        ValueError: If env_id is unknown

    Examples:
        >>> env = create_env()  # Uses CURRENT_ENV_ID
        >>> env = create_env("sat")
        >>> env = create_env("gsm8k", split="test")
    """
    env_id = env_id or CURRENT_ENV_ID

    if env_id == "sat":
        from .sat_env import SATEnv

        return SATEnv()

    if env_id == "gsm8k":
        from .gsm8k_env import GSM8KEnv
        from .providers import GSM8KTaskSource

        source = task_source or GSM8KTaskSource(split=split)
        return GSM8KEnv(task_source=source)

    raise ValueError(f"Unknown environment ID: {env_id}")


def create_env_factory(
    env_id: str | None = None,
    *,
    task_source: Any | None = None,
    split: str = "train",
) -> Any:
    """Create environment factory function for deferred instantiation.

    Useful when you need to pass a factory to components that create
    multiple environment instances (e.g., vectorized evaluation).

    Args:
        env_id: Environment identifier (defaults to CURRENT_ENV_ID)
        task_source: Optional task source for dataset-backed envs
        split: Dataset split for GSM8K

    Returns:
        Callable that creates new environment instances

    Examples:
        >>> factory = create_env_factory("gsm8k", split="test")
        >>> env1 = factory()
        >>> env2 = factory()
    """
    env_id = env_id or CURRENT_ENV_ID

    def _factory() -> MultiTurnEnv:
        return create_env(env_id=env_id, task_source=task_source, split=split)

    return _factory
