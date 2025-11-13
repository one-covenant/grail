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

    completion = _build_completion(f"{SOLUTION_START}{gold_answer}{SOLUTION_END}")

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

    completion = _build_completion(f"{SOLUTION_START}0{SOLUTION_END}")

    _obs, reward, terminated, truncated, info = env.step(
        ChatMessage(role="assistant", content=completion)
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    # Reward should miss the correctness component entirely.
    assert reward < 0.75


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        (r"\dfrac{7}{20}", r"\frac{7}{20}", True),
        (r"\frac{2 \sqrt{149}}{3}", r"\frac{2\sqrt{149}}{3}", True),
        (
            r"\begin{pmatrix} 8 & 12 \\ -4 & 20 \end{pmatrix}",
            r"\begin{pmatrix}8&12\\-4&20\end{pmatrix}",
            True,
        ),
        ("18\\text{ ways.}", "18\\text{ways.}", True),
        ("y^4-2y^3+7y^2+y-5", "y^4-2y^3+7y^2+y-5", True),
        (r"\frac{7}{20}", r"\frac{7}{21}", False),
    ],
)
def test_math_answer_normalization_examples(predicted: str, gold: str, expected: bool) -> None:
    """Verify normalization handles representative answers supplied by the user."""
    from grail.environments.math_hendrycks_env import _math_answers_equal

    assert _math_answers_equal(predicted, gold) is expected
