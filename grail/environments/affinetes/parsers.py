"""Completion parsers for affinetes environments."""

from __future__ import annotations

from typing import Any

from ..base import ThinkingParser


class TraceCompletionParser(ThinkingParser):
    """Parser for trace/print environment completions."""

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        answer, trailing, has_ordering = self._get_answer_with_thinking_check(completion)
        return {
            "answer_text": answer,
            "has_thinking": self._detect_thinking_block(completion),
            "has_answer": answer is not None,
            "trailing_after_answer": trailing,
        }


class LogicCompletionParser(ThinkingParser):
    """Parser for lgc-v2 logic environment completions."""

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        answer, trailing, has_ordering = self._get_answer_with_thinking_check(completion)
        return {
            "answer_text": answer,
            "has_thinking": self._detect_thinking_block(completion),
            "has_answer": answer is not None,
            "trailing_after_answer": trailing,
        }
