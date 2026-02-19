"""Integration tests for affinetes environment adapters.

Requires affinetes submodule to be initialized.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grail.environments import ChatMessage, create_env
from grail.shared.prompt_constants import (
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _ensure_affinetes():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "affinetes"
    if not (vendor / "environments").exists():
        pytest.skip(
            "affinetes submodule not initialized "
            "(git submodule update --init vendor/affinetes)"
        )


def _build_completion(answer: str) -> str:
    return (
        f"{REASONING_START}Working through it.{REASONING_END}\n"
        f"{SOLUTION_START}{answer}{SOLUTION_END}"
    )


# --- Trace tests ---


def test_trace_env_reset_returns_prompt():
    env = create_env("affine_trace")
    obs = env.reset(seed=42)
    assert len(obs.messages) == 1
    assert obs.messages[0].role == "user"
    assert len(obs.messages[0].content) > 0


def test_trace_env_step_returns_valid_structure():
    env = create_env("affine_trace")
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=_build_completion("test output"))
    )
    assert terminated is True
    assert truncated is False
    assert isinstance(reward, float)
    assert "reward_components" in info
    assert "success" in info


def test_trace_env_deterministic():
    obs1 = create_env("affine_trace").reset(seed=123)
    obs2 = create_env("affine_trace").reset(seed=123)
    assert obs1.messages[0].content == obs2.messages[0].content


def test_trace_env_format_rewards_without_tags():
    """No tags -> no format bonuses, only base reward."""
    env = create_env("affine_trace")
    env.reset(seed=42)
    _, reward, _, _, info = env.step(
        ChatMessage(role="assistant", content="plain text no tags")
    )
    assert reward < 0.3  # No format bonuses, no correctness


# --- Logic tests ---


def test_logic_env_reset_returns_prompt():
    env = create_env("affine_logic")
    obs = env.reset(seed=500)
    assert len(obs.messages[0].content) > 0


def test_logic_env_step_returns_valid_structure():
    env = create_env("affine_logic")
    env.reset(seed=500)
    _, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=_build_completion("answer"))
    )
    assert terminated is True
    assert isinstance(reward, float)
    assert "task_type" in info


def test_logic_env_task_type_encoding():
    """task_id 100_000_500 should select game_of_24 task type."""
    env = create_env("affine_logic")
    obs = env.reset(task_id="100000500")
    assert obs.task_meta.get("task_type") == "game_of_24"


def test_logic_env_deterministic():
    obs1 = create_env("affine_logic").reset(task_id="500")
    obs2 = create_env("affine_logic").reset(task_id="500")
    assert obs1.messages[0].content == obs2.messages[0].content
