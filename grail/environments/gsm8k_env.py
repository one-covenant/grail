"""Single-turn GSM8K environment using HF datasets backend.

This environment serves a single math word problem per episode and computes
reward by exact-match against the dataset answer (with light normalization).
Uses RewardVector for decomposed reward components: correctness, strict format,
thinking format, answer format, and no trailing text penalties.

Refactored to extend MathDatasetEnv base class for better extensibility.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from .base import Parser, RewardVector, ThinkingParser
from .dataset_base import MathDatasetEnv
from .providers import GSM8KTaskSource, TaskSource
from .reward_components import (
    answer_format_reward,
    no_trailing_reward,
    thinking_format_reward,
)


class GSM8KCompletionParser(ThinkingParser):
    """Strict parser for GSM8K completions with thinking + answer format detection.

    Inherits thinking tag detection from ThinkingParser base class.
    Provides GSM8K-specific answer parsing for math problems.

    Detects:
    - Thinking blocks: <start_working_out>...</end_working_out> (inherited)
    - Answer blocks: <SOLUTION>...</SOLUTION>
    - Numeric-only validation: ensures answer contains only digits, +/-, decimal point
    - Trailing text: tracks chars after </SOLUTION>
    """

    _SOLUTION_PATTERN = re.compile(r"<SOLUTION>\s*(.+?)\s*</SOLUTION>", re.DOTALL)
    _NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
    _NUMERIC_ONLY_PATTERN = re.compile(r"^[-+]?[\d.,]+$")

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        """Parse completion for thinking tags, answer tags, and numeric validation.

        Returns dict with:
        - answer_text: extracted numeric answer
        - has_thinking: bool, True if thinking block present (uses inherited method)
        - has_answer: bool, True if answer block present
        - trailing_after_answer: int, chars after </SOLUTION>
        - is_numeric_only: bool, True if answer contains only numbers/+/-/decimal
        """
        text = completion or ""

        # Use inherited methods from ThinkingParser
        has_thinking = self._detect_thinking_block(text)
        has_answer = self._detect_answer_block(text)

        # GSM8K-specific: extract answer text and validate numeric content
        answer_text = ""
        trailing_after_answer = 0
        is_numeric_only = False

        if has_answer:
            # Prefer thinking→answer ordering, fall back to any answer block
            content, trailing, _ = self._get_answer_with_thinking_check(text)
            if content is not None:
                inside_tags = content.strip()
                trailing_after_answer = trailing

                # Extract first number from inside tags
                num_match = self._NUMBER_PATTERN.search(inside_tags)
                if num_match:
                    answer_text = num_match.group(0).replace(",", "").strip()
                    # Check if entire inside_tags content is numeric-only
                    is_numeric_only = bool(
                        self._NUMERIC_ONLY_PATTERN.match(inside_tags.replace(" ", ""))
                    )

        return {
            "answer_text": answer_text,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "trailing_after_answer": trailing_after_answer,
            "is_numeric_only": is_numeric_only,
        }


def _parse_gsm8k_golden(text: str) -> str:
    """Parse golden GSM8K answer from dataset solution text.

    Accepts standard GSM8K format with '#### answer' or, as a fallback,
    extracts the last number in the text.
    """
    _hash_pattern = re.compile(r"####\s*(?P<ans>.+)")
    _number_pattern = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")

    match = None
    for m in _hash_pattern.finditer(text or ""):
        match = m
    if match is not None:
        return match.group("ans").strip()

    nums = list(_number_pattern.finditer(text or ""))
    if nums:
        return nums[-1].group(0).replace(",", "").strip()

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _normalize_answer(s: str) -> str:
    return re.sub(r"[\s\.]+$", "", s.strip().lower())


def _gsm8k_correctness_reward(parsed: dict[str, Any], context: Any) -> float:
    """GSM8K-specific correctness reward.

    Compares parsed answer against gold answer from dataset context.
    Returns 1.0 if match, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    answer_text = str(parsed.get("answer_text", ""))
    if not answer_text:
        return 0.0

    # Get gold answer from context payload
    gold = context.get("answer", "") if isinstance(context, dict) else ""
    gold_parsed = _parse_gsm8k_golden(gold)

    pred_n = _normalize_answer(answer_text)
    gold_n = _normalize_answer(str(gold_parsed))

    return 1.0 if pred_n == gold_n else 0.0


def _gsm8k_strict_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """GSM8K-specific strict format reward.

    Validates that:
    - Answer block is present
    - Answer contains ONLY numeric characters (int or float)
    - No trailing text after answer

    Returns 0.3 if all conditions met, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    has_answer = parsed.get("has_answer", False)
    is_numeric_only = parsed.get("is_numeric_only", False)
    trailing = int(parsed.get("trailing_after_answer", 0))

    if has_answer and is_numeric_only and trailing == 0:
        return 0.3
    return 0.0


def _create_gsm8k_reward_vector() -> RewardVector:
    """Create GSM8K reward vector with 5 decomposed components.

    Components:
    1. Correctness (0.6): Exact match with gold answer
    2. Strict format (0.15): Numeric-only validation + no trailing
    3. Thinking (0.1): Presence of thinking block
    4. Answer (0.1): Presence of answer block
    5. No trailing (0.05): Penalty for text after answer

    Total weight: 1.0
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            _gsm8k_correctness_reward,
            _gsm8k_strict_format_reward,
            thinking_format_reward,
            answer_format_reward,
            no_trailing_reward,
        ],
    )
    weights = [0.6, 0.15, 0.1, 0.1, 0.05]

    return RewardVector(
        reward_functions,
        weights,
        parser=GSM8KCompletionParser(),
        bounds=[
            (0.0, 1.0),  # correctness
            (0.0, 0.3),  # strict_format
            (0.0, 0.5),  # thinking
            (0.0, 0.3),  # answer
            (0.0, 0.2),  # no_trailing
        ],
    )


class GSM8KEnv(MathDatasetEnv):
    """GSM8K single-turn environment with RewardVector support.

    Extends MathDatasetEnv with GSM8K-specific logic:
    - Answer extraction: <SOLUTION>numeric</SOLUTION> format
    - Gold parsing: #### answer format from dataset
    - Validation: Numeric exact match after normalization

    Uses decomposed rewards:
    - Correctness: exact match against gold
    - Strict format: numeric-only answer validation
    - Thinking: presence of reasoning
    - Answer: presence of solution tags
    - No trailing: penalize verbose completions
    """

    # =========================================================================
    # Template Method Implementations (GSM8K-specific)
    # =========================================================================

    def _extract_dataset_answer(self, task_payload: dict[str, Any]) -> str:
        """Extract gold answer from GSM8K dataset format (#### answer)."""
        return _parse_gsm8k_golden(task_payload.get("answer", ""))

    def _extract_completion_answer(self, completion: str, context: dict[str, Any]) -> str | None:
        """Extract numeric answer from <SOLUTION>...</SOLUTION> tags."""
        # Use parser to extract answer
        parsed = self._parser.parse(completion, context)
        answer_text = parsed.get("answer_text", "")
        return answer_text if answer_text else None

    def _default_task_source(self) -> TaskSource:
        """Create GSM8K task source."""
        return GSM8KTaskSource()

    def _create_parser(self) -> Parser:
        """Create GSM8K-specific completion parser."""
        return GSM8KCompletionParser()

    def _create_reward_vector(self) -> RewardVector:
        """Create GSM8K-specific reward vector."""
        return _create_gsm8k_reward_vector()
