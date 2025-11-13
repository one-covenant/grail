from __future__ import annotations

import pytest

from grail.environments import ChatMessage, create_env
from grail.environments.gsm8k_env import _parse_gsm8k_golden
from grail.shared.prompt_constants import (
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _ensure_datasets() -> None:
    """Skip tests if HuggingFace datasets dependency is missing."""
    pytest.importorskip("datasets")


def _pull_task(env: object) -> dict[str, object]:
    task = getattr(env, "_task", None)
    assert task is not None, "Expected environment to hold current task after reset"
    return task.payload  # type: ignore[return-value]


def _build_completion(answer_block: str) -> str:
    reasoning = f"{REASONING_START}Working through the solution.{REASONING_END}"
    return f"{reasoning}\n{answer_block}"


def test_gsm8k_env_success_case() -> None:
    env = create_env("gsm8k")
    env.reset(seed=17)
    payload = _pull_task(env)
    gold_numeric = _parse_gsm8k_golden(str(payload["answer"]))

    completion = _build_completion(f"{SOLUTION_START}{gold_numeric}{SOLUTION_END}")

    _obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["pred_answer"] == gold_numeric
    # Correctness component contributes >= 0.6 when answer is correct.
    assert reward > 0.6


def test_gsm8k_env_incorrect_case() -> None:
    env = create_env("gsm8k")
    env.reset(seed=23)

    completion = _build_completion(f"{SOLUTION_START}0{SOLUTION_END}")

    _obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    # Only format rewards (no correctness) -> strictly less than correctness weight.
    assert reward < 0.6


def test_math_env_success_case() -> None:
    env = create_env("math")
    env.reset(seed=7, level=5, subject="Algebra")
    payload = _pull_task(env)
    gold_answer = str(payload["answer"])

    completion = _build_completion(f"\\boxed{{{gold_answer}}}")

    _obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is True
    assert info["pred_answer"] == gold_answer
    # Correctness weight is 0.7, so reward with format bonuses should exceed 0.75.
    assert reward > 0.75


def test_math_env_incorrect_case() -> None:
    env = create_env("math")
    env.reset(seed=9, level=1, subject="Prealgebra")

    completion = _build_completion("\\boxed{0}")

    _obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    # Reward should miss the correctness component entirely.
    assert reward < 0.75
