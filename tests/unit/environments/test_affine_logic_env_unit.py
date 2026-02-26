"""Unit tests for AffineLogicEnv parser and reward components.

No external dependencies -- tests parser + reward function with mock verifier.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grail.environments.affinetes.parsers import LogicCompletionParser
from grail.shared.thinking import get_thinking_config

_cfg = get_thinking_config()


@pytest.fixture(scope="module")
def parser():
    return LogicCompletionParser()


class TestLogicCompletionParser:
    def test_extracts_answer(self, parser):
        completion = f"{_cfg.thinking_open}reason{_cfg.thinking_close}\n{_cfg.solution_open})))({_cfg.solution_close}"
        parsed = parser.parse(completion, {})
        assert parsed["answer_text"] == ")))("
        assert parsed["has_thinking"] is True
        assert parsed["has_answer"] is True

    def test_no_answer(self, parser):
        parsed = parser.parse("no tags here", {})
        assert parsed["answer_text"] is None
        assert parsed["has_answer"] is False

    def test_answer_without_thinking(self, parser):
        parsed = parser.parse("<SOLUTION>42</SOLUTION>", {})
        assert parsed["answer_text"] == "42"
        assert parsed["has_thinking"] is False

    def test_trailing_chars(self, parser):
        parsed = parser.parse("<SOLUTION>ans</SOLUTION>extra", {})
        assert parsed["trailing_after_answer"] == len("extra")

    def test_empty_completion(self, parser):
        parsed = parser.parse("", {})
        assert parsed["answer_text"] is None
        assert parsed["has_answer"] is False
        assert parsed["has_thinking"] is False


class TestLogicCorrectnessReward:
    def test_correct_with_mock_verifier(self):
        """Test correctness reward with a mock verifier (no affinetes needed)."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True
        mock_data_cls = MagicMock()
        mock_data_cls.from_json.return_value = MagicMock()

        with (
            patch(
                "grail.environments.affinetes.rewards._verifier_cache",
                {"dyck_language": mock_verifier},
            ),
            patch(
                "grail.environments.affinetes._loader.load_logic_verifiers",
                return_value=({}, mock_data_cls),
            ),
        ):
            from grail.environments.affinetes.rewards import logic_correctness_reward

            parsed = {"answer_text": ")))(", "has_answer": True}
            context = {
                "task_type": "dyck_language",
                "game_data": {"question": "...", "answer": "..."},
            }
            result = logic_correctness_reward(parsed, context)
            assert result == 1.0

    def test_incorrect_with_mock_verifier(self):
        """Test correctness reward returns 0.0 when verifier rejects."""
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = False
        mock_data_cls = MagicMock()
        mock_data_cls.from_json.return_value = MagicMock()

        with (
            patch(
                "grail.environments.affinetes.rewards._verifier_cache",
                {"dyck_language": mock_verifier},
            ),
            patch(
                "grail.environments.affinetes._loader.load_logic_verifiers",
                return_value=({}, mock_data_cls),
            ),
        ):
            from grail.environments.affinetes.rewards import logic_correctness_reward

            parsed = {"answer_text": "wrong", "has_answer": True}
            context = {
                "task_type": "dyck_language",
                "game_data": {"question": "...", "answer": "..."},
            }
            result = logic_correctness_reward(parsed, context)
            assert result == 0.0

    def test_no_answer(self):
        from grail.environments.affinetes.rewards import logic_correctness_reward

        parsed = {"answer_text": None, "has_answer": False}
        context = {"task_type": "dyck_language", "game_data": {}}
        assert logic_correctness_reward(parsed, context) == 0.0

    def test_non_dict_parsed(self):
        from grail.environments.affinetes.rewards import logic_correctness_reward

        assert logic_correctness_reward("not a dict", {}) == 0.0

    def test_non_dict_context(self):
        from grail.environments.affinetes.rewards import logic_correctness_reward

        parsed = {"answer_text": "ans", "has_answer": True}
        assert logic_correctness_reward(parsed, "not a dict") == 0.0

    def test_missing_game_data(self):
        """Missing game_data in context should return 0.0."""
        mock_verifier = MagicMock()
        with patch(
            "grail.environments.affinetes.rewards._verifier_cache",
            {"dyck_language": mock_verifier},
        ):
            from grail.environments.affinetes.rewards import logic_correctness_reward

            parsed = {"answer_text": "ans", "has_answer": True}
            context = {"task_type": "dyck_language"}  # no game_data
            assert logic_correctness_reward(parsed, context) == 0.0

    def test_unknown_task_type(self):
        """Unknown task_type should return 0.0 (no matching verifier)."""
        with patch(
            "grail.environments.affinetes.rewards._verifier_cache",
            {"dyck_language": MagicMock()},
        ):
            from grail.environments.affinetes.rewards import logic_correctness_reward

            parsed = {"answer_text": "ans", "has_answer": True}
            context = {"task_type": "nonexistent", "game_data": {}}
            assert logic_correctness_reward(parsed, context) == 0.0
