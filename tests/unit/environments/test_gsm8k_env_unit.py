from __future__ import annotations

import re
from typing import Any, cast

import pytest

from grail.environments.gsm8k_env import (
    GSM8KCompletionParser,
    _create_gsm8k_reward_vector,
    _parse_gsm8k_golden,
)

# ============================================================================
# REAL GSM8K TEST DATA - 5 Problems from openai/gsm8k dataset
# ============================================================================

TEST_PROBLEMS = [
    {
        "id": 0,
        "name": "Natalia's Clips",
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?"
        ),
        "gold_answer": (
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
            "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
            "#### 72"
        ),
        "expected_numeric": "72",
    },
    {
        "id": 1,
        "name": "Weng's Babysitting",
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
            "minutes of babysitting. How much did she earn?"
        ),
        "gold_answer": (
            "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
            "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n"
            "#### 10"
        ),
        "expected_numeric": "10",
    },
    {
        "id": 2,
        "name": "Betty's Wallet",
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty has only "
            "half of the money she needs. Her parents decided to give her $15 for that "
            "purpose, and her grandparents twice as much as her parents. How much more "
            "money does Betty need to buy the wallet?"
        ),
        "gold_answer": (
            "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\n"
            "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\n"
            "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n"
            "#### 5"
        ),
        "expected_numeric": "5",
    },
    {
        "id": 3,
        "name": "Julie's Book",
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 "
            "pages and today, she read twice as many pages as yesterday. If she wants "
            "to read half of the remaining pages tomorrow, how many pages should she read?"
        ),
        "gold_answer": (
            "Maila read 12 x 2 = <<12*2=24>>24 pages today.\n"
            "So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since "
            "yesterday.\n"
            "There are 120 - 36 = <<120-36=84>>84 pages left to be read.\n"
            "Since she wants to read half of the remaining pages tomorrow, then she "
            "should read 84/2 = <<84/2=42>>42 pages.\n"
            "#### 42"
        ),
        "expected_numeric": "42",
    },
    {
        "id": 4,
        "name": "James' Letters",
        "question": (
            "James writes a 3-page letter to 2 different friends twice a week. "
            "How many pages does he write a year?"
        ),
        "gold_answer": (
            "He writes each friend 3*2=<<3*2=6>>6 pages a week\n"
            "So he writes 6*2=<<6*2=12>>12 pages every week\n"
            "That means he writes 12*52=<<12*52=624>>624 pages a year\n"
            "#### 624"
        ),
        "expected_numeric": "624",
    },
]


@pytest.fixture(scope="module")
def parser() -> GSM8KCompletionParser:
    """Parser instance for all tests."""
    return GSM8KCompletionParser()


@pytest.fixture(scope="module")
def reward_vector() -> Any:
    """Reward vector instance for all tests."""
    return _create_gsm8k_reward_vector()


@pytest.fixture(scope="module", params=TEST_PROBLEMS, ids=[p["name"] for p in TEST_PROBLEMS])
def test_problem(request: pytest.FixtureRequest) -> dict[str, str]:
    """Parametrized fixture providing each of the 5 real GSM8K problems."""
    return cast(dict[str, str], request.param)


def _normalize_answer(s: str) -> str:
    """Normalize answer for comparison (matches gsm8k_env logic)."""
    return re.sub(r"[\s\.]+$", "", (s or "").strip().lower())


def _build_completion(thinking: str, answer: str, trailing: str = "") -> str:
    """Build properly formatted completion with thinking and solution tags."""
    from grail.shared.thinking import get_thinking_config

    cfg = get_thinking_config()
    return (
        f"{cfg.thinking_open}\n{thinking}\n{cfg.thinking_close}\n"
        f"{cfg.solution_open}{answer}{cfg.solution_close}{trailing}"
    )


def test_gsm8k_perfect_format_with_thinking(
    parser: GSM8KCompletionParser,
    reward_vector: Any,
    test_problem: dict[str, str],
) -> None:
    """Test perfect format: thinking + correct numeric answer, no trailing.

    For each of 5 real GSM8K problems, generate a completion with:
    - Proper thinking tags (mode-dependent)
    - Correct numeric answer inside <SOLUTION>...</SOLUTION>
    - No trailing text after solution tag

    Expected rewards:
    - correctness: 1.0 (exact match)
    - strict_format: 0.3 (numeric-only + no trailing)
    - thinking: 0.5 (thinking block present)
    - answer: 0.3 (answer block present)
    - no_trailing: 0.2 (no trailing text)
    """
    problem_data = {"question": test_problem["question"], "answer": test_problem["gold_answer"]}
    gold_num = test_problem["expected_numeric"]

    # Realistic thinking for math problem
    thinking = (
        "I need to identify the quantities and relationships in this problem, "
        "set up equations, solve step by step, and verify the answer makes sense."
    )

    # Perfect format: correct answer, no trailing
    completion = _build_completion(thinking, gold_num)

    # Parse
    parsed = parser.parse(completion, problem_data)
    assert parsed["has_thinking"] is True, "Should detect thinking block"
    assert parsed["has_answer"] is True, "Should detect answer block"
    assert parsed["is_numeric_only"] is True, "Answer should be numeric-only"
    assert parsed["trailing_after_answer"] == 0, "Should have no trailing text"
    assert parsed["answer_text"] == gold_num, f"Should extract correct answer: {gold_num}"

    # Rewards
    rewards = reward_vector.compute_individual_rewards(completion, problem_data)
    r_correct, r_strict, r_think, r_answer, r_no_trailing = rewards

    assert r_correct == pytest.approx(1.0, abs=1e-6), "Perfect format should get correctness=1.0"
    assert r_strict == pytest.approx(0.3, abs=1e-6), (
        "Strict format (numeric + no trailing) should be 0.3"
    )
    assert r_think == pytest.approx(0.5, abs=1e-6), "Should reward thinking block"
    assert r_answer == pytest.approx(0.3, abs=1e-6), "Should reward answer block"
    assert r_no_trailing == pytest.approx(0.2, abs=1e-6), "No trailing should give max reward"


def test_gsm8k_trailing_text_penalty(
    parser: GSM8KCompletionParser,
    reward_vector: Any,
    test_problem: dict[str, str],
) -> None:
    """Test trailing text penalty: correct answer but with explanation after tag.

    For each of 5 real GSM8K problems, generate:
    - Thinking block (present)
    - Correct numeric answer
    - Trailing text after </SOLUTION> tag

    Expected:
    - correctness: 1.0 (answer is correct)
    - strict_format: 0.0 (trailing text violates strict format requirement)
    - no_trailing: reduced based on trailing character count
    """
    problem_data = {"question": test_problem["question"], "answer": test_problem["gold_answer"]}
    gold_num = test_problem["expected_numeric"]

    thinking = "Let me work through this step by step."
    trailing_text = "\nSo the answer is indeed correct."

    completion = _build_completion(thinking, gold_num, trailing=trailing_text)

    # Parse
    parsed = parser.parse(completion, problem_data)
    assert parsed["has_thinking"] is True
    assert parsed["has_answer"] is True
    assert parsed["is_numeric_only"] is True, "Answer itself should still be numeric"
    assert parsed["trailing_after_answer"] == len(trailing_text), "Should count trailing chars"
    assert parsed["answer_text"] == gold_num

    # Rewards
    rewards = reward_vector.compute_individual_rewards(completion, problem_data)
    r_correct, r_strict, r_think, r_answer, r_no_trailing = rewards

    assert r_correct == pytest.approx(1.0, abs=1e-6), "Answer is still correct despite trailing"
    assert r_strict == 0.0, "Strict format penalizes trailing text"
    assert r_think == pytest.approx(0.5, abs=1e-6)
    assert r_answer == pytest.approx(0.3, abs=1e-6)
    # no_trailing = max(0.0, 0.2 - 0.001 * trailing_chars)
    expected_no_trailing = max(0.0, 0.2 - 0.001 * len(trailing_text))
    assert r_no_trailing == pytest.approx(expected_no_trailing, abs=1e-6), (
        f"Trailing penalty: 0.2 - 0.001*{len(trailing_text)} = {expected_no_trailing}"
    )


def test_gsm8k_units_inside_solution_violates_numeric(
    parser: GSM8KCompletionParser,
    reward_vector: Any,
    test_problem: dict[str, str],
) -> None:
    """Test non-numeric content inside solution: answer + units.

    For each of 5 real GSM8K problems, include units inside the solution:
    - Thinking block present
    - Answer with units: "{number} dollars" or similar
    - No trailing text after solution tag

    Expected:
    - Parser still extracts the numeric part (first number)
    - is_numeric_only: False (violates strict format)
    - correctness: 1.0 (numeric part matches)
    - strict_format: 0.0 (not numeric-only)
    - no_trailing: 0.2 (no trailing after tag)
    """
    problem_data = {"question": test_problem["question"], "answer": test_problem["gold_answer"]}
    gold_num = test_problem["expected_numeric"]

    thinking = "I'll calculate the result."
    # Include units: violates numeric-only constraint
    answer_with_units = f"{gold_num} dollars"

    completion = _build_completion(thinking, answer_with_units)

    # Parse
    parsed = parser.parse(completion, problem_data)
    assert parsed["has_thinking"] is True
    assert parsed["has_answer"] is True
    assert parsed["is_numeric_only"] is False, "Should detect non-numeric content"
    assert parsed["trailing_after_answer"] == 0, "No trailing after solution tag"
    # Parser extracts first number from inside tags
    assert parsed["answer_text"] == gold_num, "Should extract numeric part"

    # Rewards
    rewards = reward_vector.compute_individual_rewards(completion, problem_data)
    r_correct, r_strict, r_think, r_answer, r_no_trailing = rewards

    assert r_correct == pytest.approx(1.0, abs=1e-6), "Numeric part is correct"
    assert r_strict == 0.0, "Strict format requires numeric-only inside solution"
    assert r_think == pytest.approx(0.5, abs=1e-6)
    assert r_answer == pytest.approx(0.3, abs=1e-6)
    assert r_no_trailing == pytest.approx(0.2, abs=1e-6), "No trailing text"


def test_gsm8k_all_five_problems_summary(test_problem: dict[str, str]) -> None:
    """Summary test: validate all 5 problems are well-formed.

    This is a sanity check that all test data is valid and extractable.
    """
    assert "id" in test_problem
    assert "name" in test_problem
    assert "question" in test_problem
    assert "gold_answer" in test_problem
    assert "expected_numeric" in test_problem

    assert len(test_problem["question"]) > 0, "Question should not be empty"
    assert len(test_problem["gold_answer"]) > 0, "Gold answer should not be empty"
    assert len(test_problem["expected_numeric"]) > 0, "Expected numeric should not be empty"

    # Verify expected numeric matches gold answer extraction
    gold_parsed = _parse_gsm8k_golden(test_problem["gold_answer"])
    extracted = _normalize_answer(gold_parsed)
    expected_norm = _normalize_answer(test_problem["expected_numeric"])
    assert extracted == expected_norm, (
        f"Expected numeric mismatch for {test_problem['name']}: "
        f"extracted={extracted}, expected={expected_norm}"
    )
