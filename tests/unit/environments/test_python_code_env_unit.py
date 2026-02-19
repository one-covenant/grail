"""Unit tests for Python code generation environment.

Tests PythonCodeParser and reward functions in isolation without
requiring dataset access or code execution.
"""

from __future__ import annotations

from typing import Any

import pytest

from grail.environments.python_code_env import (
    PythonCodeParser,
    _python_correctness_reward_from_parsed,
    _python_solution_format_reward,
    _python_syntax_reward,
    _python_thinking_format_reward,
)

# =============================================================================
# Test Data
# =============================================================================


def _build_completion(thinking: str, code: str, trailing: str = "") -> str:
    """Build properly formatted completion with thinking and solution tags."""
    from grail.shared.thinking import get_thinking_config

    cfg = get_thinking_config()
    return (
        f"{cfg.thinking_open}\n{thinking}\n{cfg.thinking_close}\n"
        f"{cfg.solution_open}\n{code}\n{cfg.solution_close}{trailing}"
    )


VALID_PYTHON_CODE = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""

SYNTAX_ERROR_CODE = """def broken_func(:
    return "missing closing paren"
"""


# =============================================================================
# Parser Tests
# =============================================================================


@pytest.fixture(scope="module")
def parser() -> PythonCodeParser:
    """Parser instance for all tests."""
    return PythonCodeParser()


class TestPythonCodeParser:
    """Test PythonCodeParser extraction and validation."""

    def test_parse_full_completion(self, parser: PythonCodeParser) -> None:
        """Test parsing a complete well-formatted completion."""
        completion = _build_completion("I need to implement fibonacci.", VALID_PYTHON_CODE)
        result = parser.parse(completion, {})

        assert result["has_thinking"] is True
        assert result["has_solution"] is True
        assert result["syntax_valid"] is True
        assert result["trailing_after_solution"] == 0
        assert "def fibonacci" in result["code"]

    def test_parse_no_thinking_block(self, parser: PythonCodeParser) -> None:
        """Test parsing completion without thinking block."""
        completion = f"<SOLUTION>\n{VALID_PYTHON_CODE}\n</SOLUTION>"
        result = parser.parse(completion, {})

        assert result["has_thinking"] is False
        assert result["has_solution"] is True
        assert result["syntax_valid"] is True

    def test_parse_no_solution_tags(self, parser: PythonCodeParser) -> None:
        """Test parsing completion without solution tags."""
        completion = VALID_PYTHON_CODE
        result = parser.parse(completion, {})

        assert result["has_thinking"] is False
        assert result["has_solution"] is False
        assert result["code"] == ""
        assert result["syntax_valid"] is False

    def test_parse_syntax_error(self, parser: PythonCodeParser) -> None:
        """Test parsing code with syntax errors."""
        completion = f"<SOLUTION>\n{SYNTAX_ERROR_CODE}\n</SOLUTION>"
        result = parser.parse(completion, {})

        assert result["has_solution"] is True
        assert result["syntax_valid"] is False

    def test_parse_trailing_text(self, parser: PythonCodeParser) -> None:
        """Test detection of trailing text after solution tag."""
        trailing = "\nI hope this helps!"
        completion = _build_completion("Thinking...", VALID_PYTHON_CODE, trailing)
        result = parser.parse(completion, {})

        assert result["trailing_after_solution"] == len(trailing)

    def test_parse_empty_completion(self, parser: PythonCodeParser) -> None:
        """Test handling of empty completion."""
        result = parser.parse("", {})

        assert result["has_thinking"] is False
        assert result["has_solution"] is False
        assert result["code"] == ""
        assert result["syntax_valid"] is False
        assert result["trailing_after_solution"] == 0

    def test_parse_none_completion(self, parser: PythonCodeParser) -> None:
        """Test handling of None completion."""
        result = parser.parse(None, {})  # type: ignore[arg-type]

        assert result["has_thinking"] is False
        assert result["has_solution"] is False
        assert result["code"] == ""

    def test_parse_empty_solution_tags(self, parser: PythonCodeParser) -> None:
        """Test handling of empty solution tags."""
        completion = "<SOLUTION></SOLUTION>"
        result = parser.parse(completion, {})

        assert result["has_solution"] is True
        assert result["code"] == ""
        assert result["syntax_valid"] is False


# =============================================================================
# Reward Function Tests
# =============================================================================


class TestRewardFunctions:
    """Test individual reward functions."""

    def test_correctness_reward_all_passed(self) -> None:
        """Test correctness reward when all tests pass."""
        parsed: dict[str, Any] = {"test_result": {"passed": 5, "total": 5, "status": "all_passed"}}
        reward = _python_correctness_reward_from_parsed(parsed, {})
        assert reward == pytest.approx(1.0)

    def test_correctness_reward_partial(self) -> None:
        """Test correctness reward with partial pass."""
        parsed: dict[str, Any] = {"test_result": {"passed": 3, "total": 5, "status": "partial"}}
        reward = _python_correctness_reward_from_parsed(parsed, {})
        assert reward == pytest.approx(0.6)

    def test_correctness_reward_no_tests(self) -> None:
        """Test correctness reward with zero tests."""
        parsed: dict[str, Any] = {"test_result": {"passed": 0, "total": 0, "status": "no_tests"}}
        reward = _python_correctness_reward_from_parsed(parsed, {})
        assert reward == 0.0

    def test_correctness_reward_missing_result(self) -> None:
        """Test correctness reward without test_result."""
        parsed: dict[str, Any] = {}
        reward = _python_correctness_reward_from_parsed(parsed, {})
        assert reward == 0.0

    def test_syntax_reward_valid(self) -> None:
        """Test syntax reward for valid code."""
        parsed: dict[str, Any] = {"syntax_valid": True}
        reward = _python_syntax_reward(parsed, {})
        assert reward == 1.0

    def test_syntax_reward_invalid(self) -> None:
        """Test syntax reward for invalid code."""
        parsed: dict[str, Any] = {"syntax_valid": False}
        reward = _python_syntax_reward(parsed, {})
        assert reward == 0.0

    def test_format_reward_correct(self) -> None:
        """Test format reward with proper tags."""
        parsed: dict[str, Any] = {"has_solution": True, "trailing_after_solution": 0}
        reward = _python_solution_format_reward(parsed, {})
        assert reward == 1.0

    def test_format_reward_no_tags(self) -> None:
        """Test format reward without solution tags."""
        parsed: dict[str, Any] = {"has_solution": False, "trailing_after_solution": 0}
        reward = _python_solution_format_reward(parsed, {})
        assert reward == 0.0

    def test_format_reward_excessive_trailing(self) -> None:
        """Test format reward with too much trailing text."""
        parsed: dict[str, Any] = {"has_solution": True, "trailing_after_solution": 100}
        reward = _python_solution_format_reward(parsed, {})
        assert reward == 0.0

    def test_format_reward_minimal_trailing(self) -> None:
        """Test format reward with acceptable trailing text."""
        parsed: dict[str, Any] = {"has_solution": True, "trailing_after_solution": 10}
        reward = _python_solution_format_reward(parsed, {})
        assert reward == 1.0

    def test_thinking_reward_present(self) -> None:
        """Test thinking reward when block present."""
        parsed: dict[str, Any] = {"has_thinking": True}
        reward = _python_thinking_format_reward(parsed, {})
        assert reward == 1.0

    def test_thinking_reward_absent(self) -> None:
        """Test thinking reward when block absent."""
        parsed: dict[str, Any] = {"has_thinking": False}
        reward = _python_thinking_format_reward(parsed, {})
        assert reward == 0.0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and defensive behavior."""

    def test_non_dict_parsed_returns_zero(self) -> None:
        """Test that reward functions handle non-dict inputs."""
        assert _python_correctness_reward_from_parsed("not a dict", {}) == 0.0  # type: ignore[arg-type]
        assert _python_syntax_reward("not a dict", {}) == 0.0  # type: ignore[arg-type]
        assert _python_solution_format_reward("not a dict", {}) == 0.0  # type: ignore[arg-type]
        assert _python_thinking_format_reward("not a dict", {}) == 0.0  # type: ignore[arg-type]

    def test_parser_handles_malformed_tags(self, parser: PythonCodeParser) -> None:
        """Test parser with malformed tags."""
        # Missing closing tag - parser handles gracefully
        result = parser.parse("<SOLUTION>code here", {})
        assert result["has_solution"] is False
        assert result["code"] == ""

        # Lowercase tags are accepted (case-insensitive detection)
        result = parser.parse("<solution>code</solution>", {})
        assert result["has_solution"] is True
        assert result["code"] == "code"

        # Random text without tags
        result = parser.parse("def foo(): return 42", {})
        assert result["has_solution"] is False

    def test_parser_handles_unicode(self, parser: PythonCodeParser) -> None:
        """Test parser with unicode content."""
        code = 'def greet(): return "Hello, 世界! 🌍"'
        completion = f"<SOLUTION>\n{code}\n</SOLUTION>"
        result = parser.parse(completion, {})

        assert result["has_solution"] is True
        assert result["syntax_valid"] is True
        assert "世界" in result["code"]
