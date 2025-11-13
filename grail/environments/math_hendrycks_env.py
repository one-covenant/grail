"""Single-turn Hendrycks MATH environment using HF datasets backend.

This environment serves mathematical problems from the Hendrycks MATH benchmark
spanning 7 subjects (Algebra, Geometry, Precalculus, etc.) and 5 difficulty
levels (elementary through college calculus).

Key features:
- Multi-strategy answer validation (exact, symbolic via sympy, numeric)
- Metadata filtering by level (1-5) and subject
- LaTeX answer format support (fractions, radicals, matrices, text)
- Rich reasoning traces (average 88-word solutions)

Answer extraction uses \\boxed{...} notation (100% reliable across dataset).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, cast

from .base import Parser, RewardVector, ThinkingParser
from .dataset_base import MathDatasetEnv
from .providers import MATHTaskSource, TaskSource
from .reward_components import (
    no_trailing_reward,
    thinking_format_reward,
)


def _math_answers_equal(predicted: str, gold: str) -> bool:
    """Compare answers using exact, symbolic, and numeric strategies.

    Args:
        predicted: Model-predicted answer (raw string)
        gold: Dataset gold answer (raw string)

    Returns:
        True if answers are equivalent under any strategy.
    """
    pred_norm = predicted.strip()
    gold_norm = gold.strip()

    if not pred_norm or not gold_norm:
        return False

    # Strategy 1: Exact match
    if pred_norm == gold_norm:
        return True

    # Strategy 2: Symbolic equivalence via sympy (fractions, radicals)
    try:
        import sympy

        expr_pred = sympy.parse_expr(pred_norm)
        expr_gold = sympy.parse_expr(gold_norm)
        if sympy.simplify(expr_pred - expr_gold) == 0:
            return True
    except Exception:
        pass

    # Strategy 3: Numeric comparison (floats)
    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        if abs(pred_val - gold_val) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


class MATHCompletionParser(ThinkingParser):
    """Parser for Hendrycks MATH completions with \\boxed{} answer detection.

    Inherits thinking tag detection from ThinkingParser base class.
    Provides MATH-specific answer parsing for boxed notation.

    Detects:
    - Thinking blocks: <start_working_out>...</end_working_out> (inherited)
    - Answer blocks: \\boxed{...} (LaTeX standard)
    - Trailing text: tracks chars after answer
    """

    _BOXED_OPEN_PATTERN = re.compile(r"\\boxed\s*{", re.DOTALL)

    def _extract_boxed_content(self, text: str) -> tuple[str | None, int]:
        """Extract boxed LaTeX content handling nested braces."""
        match = self._BOXED_OPEN_PATTERN.search(text)
        if not match:
            return None, 0

        start = match.end()
        depth = 1
        idx = start
        length = len(text)

        while idx < length and depth > 0:
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1

        if depth != 0:
            return None, 0

        content = text[start : idx - 1].strip()
        trailing = length - idx
        return content, trailing

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        """Parse completion for thinking tags and boxed answer.

        Returns dict with:
        - answer_text: extracted answer from \\boxed{}
        - has_thinking: bool, True if thinking block present
        - has_answer: bool, True if \\boxed{} present
        - trailing_after_answer: int, chars after answer
        """
        text = completion or ""

        # Use inherited method from ThinkingParser
        has_thinking = self._detect_thinking_block(text)

        # Extract answer from \\boxed{}
        answer_text = ""
        trailing_after_answer = 0
        has_answer = False

        answer_content, trailing = self._extract_boxed_content(text)
        if answer_content is not None:
            has_answer = True
            answer_text = answer_content
            trailing_after_answer = trailing

        return {
            "answer_text": answer_text,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "trailing_after_answer": trailing_after_answer,
        }


def _math_correctness_reward(parsed: dict[str, Any], context: Any) -> float:
    """MATH-specific correctness reward (0.0 or 1.0).

    Uses the same comparison logic as the environment to keep reward and
    success flag aligned. Returns 1.0 on exact/symbolic/numeric match.
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    answer = parsed.get("answer_text", "")
    gold = ""
    if isinstance(context, dict):
        gold = str(context.get("answer", ""))

    if not answer or not gold:
        return 0.0

    return 1.0 if _math_answers_equal(str(answer), gold) else 0.0


def _math_answer_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """MATH-specific answer format reward.

    Validates that:
    - \\boxed{} notation is present
    - No excessive trailing text after answer

    Returns 0.3 if conditions met, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0  # type: ignore[unreachable]

    has_answer = parsed.get("has_answer", False)
    trailing = int(parsed.get("trailing_after_answer", 0))

    # Allow small amount of trailing text (closing remarks, etc.)
    if has_answer and trailing < 50:
        return 0.3
    return 0.0


def _create_math_reward_vector() -> RewardVector:
    """Create MATH reward vector with 4 decomposed components.

    Components:
    1. Correctness (0.7): Handled at env level via multi-strategy validation
    2. Answer format (0.15): Presence of \\boxed{} + minimal trailing
    3. Thinking (0.1): Presence of thinking block
    4. No trailing (0.05): Penalty for excessive text after answer

    Total weight: 1.0
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            _math_correctness_reward,
            _math_answer_format_reward,
            thinking_format_reward,
            no_trailing_reward,
        ],
    )
    weights = [0.7, 0.15, 0.1, 0.05]

    return RewardVector(
        reward_functions,
        weights,
        parser=MATHCompletionParser(),
        bounds=[
            (0.0, 1.0),  # correctness (handled at env level)
            (0.0, 0.3),  # answer_format
            (0.0, 0.5),  # thinking
            (0.0, 0.2),  # no_trailing
        ],
    )


class MATHEnv(MathDatasetEnv):
    """Hendrycks MATH single-turn environment with multi-strategy validation.

    Extends MathDatasetEnv with MATH-specific logic:
    - Answer extraction: \\boxed{...} notation (LaTeX standard)
    - Gold answer: Direct from dataset['answer'] field
    - Validation: Multi-strategy (exact, symbolic via sympy, numeric)
    - Filtering: Supports level (1-5) and subject (7 domains)

    Answer types supported:
    - Numeric: 2, 18, 1.36, -5 (60% of dataset)
    - Fractions: \\frac{416}{27} (20% of dataset)
    - Radicals: 3\\sqrt{3}, \\frac{2\\sqrt{149}}{3} (19% of dataset)
    - Text/Special: \\text{June 20}, matrices (1% of dataset)

    Usage:
        env = MATHEnv()
        obs = env.reset(seed=42, level=5, subject="Algebra")
        obs, reward, done, info = env.step(ChatMessage(role="assistant", content=completion))
        print(f"Success: {info['success']}")
    """

    # =========================================================================
    # Template Method Implementations (MATH-specific)
    # =========================================================================

    def _extract_dataset_answer(self, task_payload: dict[str, Any]) -> str:
        """Extract gold answer from MATH dataset (direct field access)."""
        return task_payload.get("answer", "")

    def _extract_completion_answer(self, completion: str, context: dict[str, Any]) -> str | None:
        """Extract answer from \\boxed{...} notation."""
        parsed = self._parser.parse(completion, context)
        answer = parsed.get("answer_text", "")
        return answer if answer else None

    def _validate_answer(self, predicted: str, gold: str) -> bool:
        """Reuse shared comparison logic for reward + success consistency."""
        return _math_answers_equal(predicted, gold)

    def _build_task_filter(self, **filter_kwargs) -> dict[str, Any]:
        """Build filtering kwargs for MATH task source.

        Supports:
        - level: int (1-5) - difficulty level
        - subject: str - mathematical domain

        Args:
            **filter_kwargs: User-provided filters from reset()

        Returns:
            Dictionary passed to MATHTaskSource.next()
        """
        filters = {}
        if "level" in filter_kwargs and filter_kwargs["level"] is not None:
            filters["level"] = filter_kwargs["level"]
        if "subject" in filter_kwargs and filter_kwargs["subject"] is not None:
            filters["subject"] = filter_kwargs["subject"]
        return filters

    def _default_task_source(self) -> TaskSource:
        """Create MATH task source."""
        return MATHTaskSource()

    def _create_parser(self) -> Parser:
        """Create MATH-specific completion parser."""
        return MATHCompletionParser()

    def _create_reward_vector(self) -> RewardVector:
        """Create MATH-specific reward vector."""
        return _create_math_reward_vector()
