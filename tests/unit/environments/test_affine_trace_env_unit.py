"""Unit tests for AffineTraceEnv parser and reward components.

No external dependencies -- tests parser + reward functions with embedded test data.
"""

from __future__ import annotations

import pytest

from grail.environments.affinetes.parsers import TraceCompletionParser
from grail.environments.affinetes.rewards import (
    _compare_outputs_normalized,
    trace_correctness_reward,
)
from grail.shared.thinking import get_thinking_config

_cfg = get_thinking_config()

# Embedded test data (no HF dataset or submodule needed)
TEST_CASES = [
    {"expected": "5\nhello", "correct_answer": "5\nhello", "wrong_answer": "6\nhello"},
    {"expected": "42", "correct_answer": "42", "wrong_answer": "43"},
    {"expected": "True\nFalse", "correct_answer": "True\nFalse", "wrong_answer": "True"},
]


@pytest.fixture(scope="module")
def parser():
    return TraceCompletionParser()


class TestTraceCompletionParser:
    def test_extracts_answer_from_solution_tags(self, parser):
        completion = (
            f"{_cfg.thinking_open}thinking{_cfg.thinking_close}\n{_cfg.solution_open}5{_cfg.solution_close}"
        )
        parsed = parser.parse(completion, {})
        assert parsed["answer_text"] == "5"
        assert parsed["has_thinking"] is True
        assert parsed["has_answer"] is True

    def test_no_answer_returns_none(self, parser):
        parsed = parser.parse("just text no tags", {})
        assert parsed["answer_text"] is None
        assert parsed["has_answer"] is False

    def test_trailing_chars(self, parser):
        parsed = parser.parse("<SOLUTION>5</SOLUTION>trailing", {})
        assert parsed["trailing_after_answer"] == len("trailing")

    def test_multiline_answer(self, parser):
        completion = "<SOLUTION>5\nhello</SOLUTION>"
        parsed = parser.parse(completion, {})
        assert parsed["answer_text"] == "5\nhello"
        assert parsed["has_answer"] is True

    def test_answer_without_thinking(self, parser):
        parsed = parser.parse("<SOLUTION>42</SOLUTION>", {})
        assert parsed["answer_text"] == "42"
        assert parsed["has_thinking"] is False
        assert parsed["has_answer"] is True

    def test_empty_completion(self, parser):
        parsed = parser.parse("", {})
        assert parsed["answer_text"] is None
        assert parsed["has_answer"] is False
        assert parsed["has_thinking"] is False

    def test_none_completion(self, parser):
        parsed = parser.parse(None, {})
        assert parsed["answer_text"] is None
        assert parsed["has_answer"] is False


class TestCompareOutputsNormalized:
    def test_exact_match(self):
        assert _compare_outputs_normalized("5", "5") is True

    def test_whitespace_normalized(self):
        assert _compare_outputs_normalized("5\n  hello", "5 hello") is True

    def test_mismatch(self):
        assert _compare_outputs_normalized("5", "6") is False

    def test_empty_strings(self):
        assert _compare_outputs_normalized("", "") is True

    def test_leading_trailing_whitespace(self):
        assert _compare_outputs_normalized("  42  ", "42") is True

    def test_multiline_vs_spaces(self):
        assert _compare_outputs_normalized("a\nb\nc", "a b c") is True


class TestTraceCorrectnessReward:
    @pytest.mark.parametrize("case", TEST_CASES)
    def test_correct_answer(self, case):
        parsed = {"answer_text": case["correct_answer"], "has_answer": True}
        assert trace_correctness_reward(parsed, case["expected"]) == 1.0

    @pytest.mark.parametrize("case", TEST_CASES)
    def test_wrong_answer(self, case):
        parsed = {"answer_text": case["wrong_answer"], "has_answer": True}
        assert trace_correctness_reward(parsed, case["expected"]) == 0.0

    def test_no_answer(self):
        parsed = {"answer_text": None, "has_answer": False}
        assert trace_correctness_reward(parsed, "5") == 0.0

    def test_non_dict_parsed(self):
        assert trace_correctness_reward("not a dict", "5") == 0.0

    def test_empty_answer(self):
        """Empty string answer should still match empty expected."""
        parsed = {"answer_text": "", "has_answer": True}
        assert trace_correctness_reward(parsed, "") == 1.0

    def test_non_string_context(self):
        """Non-string context is converted to string."""
        parsed = {"answer_text": "42", "has_answer": True}
        assert trace_correctness_reward(parsed, 42) == 1.0
