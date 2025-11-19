"""Single-turn Hendrycks MATH environment using HF datasets backend.

This environment serves mathematical problems from the Hendrycks MATH benchmark
spanning 7 subjects (Algebra, Geometry, Precalculus, etc.) and 5 difficulty
levels (elementary through college calculus).

Key features:
- Multi-strategy answer validation (exact, symbolic via sympy, numeric)
- Metadata filtering by level (1-5) and subject
- LaTeX answer format support (fractions, radicals, matrices, text)
- Rich reasoning traces (average 88-word solutions)

    Answer extraction uses <SOLUTION>...</SOLUTION> tags (consistent with GSM8K).
    Dataset gold answers are extracted from the 'answer' field or parsed from \\boxed{...} in the solution.
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


def _extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text.

    Handles nested braces by counting depth.
    Returns None if no \\boxed{} found.
    """
    if not text:
        return None

    # Find all indices of \boxed{
    boxed_indices = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not boxed_indices:
        return None

    # Use the last one
    start = boxed_indices[-1]
    # Skip \boxed{ (7 chars)
    content_start = start + 7
    
    depth = 1
    for i, char in enumerate(text[content_start:], start=content_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:i]
    
    return None


def _normalize_latex_answer(s: str) -> str:
    """Normalize LaTeX answer for robust comparison.

    Handles variations found in MATH dataset (1.8% of 12,500 answers):
    - \\dfrac vs \\frac (125 cases)
    - \\tfrac vs \\frac (17 cases)
    - \\left/\\right delimiters (122 cases)
    - LaTeX spacing commands (93 cases)
    - Whitespace variations (common)

    Args:
        s: LaTeX string to normalize

    Returns:
        Normalized string in canonical form
    """
    s = s.strip()

    # Normalize fraction commands (1.1% of dataset)
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\cfrac", r"\frac")

    # Remove sizing delimiters (1.0% of dataset)
    s = re.sub(r"\\left\b", "", s)
    s = re.sub(r"\\right\b", "", s)

    # Remove LaTeX spacing commands (0.7% of dataset)
    s = s.replace(r"\,", "").replace(r"\!", "")
    s = s.replace(r"\;", "").replace(r"\:", "")

    # Remove ALL whitespace (LaTeX meaning preserved)
    # "2 \sqrt{x}" and "2\sqrt{x}" are mathematically identical
    s = re.sub(r"\s+", "", s)

    # Lowercase for case-insensitive matching
    return s.lower()


def _math_answers_equal(predicted: str, gold: str) -> bool:
    """Compare answers using exact, symbolic, and numeric strategies.

    Uses LaTeX normalization to handle formatting variations before comparison.
    Tries three strategies in order: exact → symbolic → numeric.

    Args:
        predicted: Model-predicted answer (raw string)
        gold: Dataset gold answer (raw string)

    Returns:
        True if answers are equivalent under any strategy.
    """
    if not predicted or not gold:
        return False

    # Normalize LaTeX formatting first (handles 1.8% of dataset variations)
    pred_norm = _normalize_latex_answer(predicted)
    gold_norm = _normalize_latex_answer(gold)

    # Strategy 1: Exact match after normalization (fast path)
    if pred_norm == gold_norm:
        return True

    # Strategy 2: Symbolic equivalence via sympy (fractions, radicals, algebra)
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
    """Parser for Hendrycks MATH completions with <SOLUTION> tag detection.

    Inherits thinking and answer tag detection from ThinkingParser base class.
    Uses same format as GSM8K for training consistency.

    Expected format:
    - Thinking blocks: <start_working_out>...</end_working_out> (inherited)
    - Answer blocks: <SOLUTION>...</SOLUTION> (inherited)
    - Trailing text: tracks chars after answer

    Note: Dataset gold answers use LaTeX (fractions, radicals, etc.) but
    model completions should wrap them in <SOLUTION> tags.
    """

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        """Parse completion for thinking tags and solution answer.

        Returns dict with:
        - answer_text: extracted answer from <SOLUTION>...</SOLUTION>
        - has_thinking: bool, True if thinking block present
        - has_answer: bool, True if <SOLUTION> tags present
        - trailing_after_answer: int, chars after closing tag
        """
        text = completion or ""

        # Use inherited methods from ThinkingParser
        has_thinking = self._detect_thinking_block(text)
        has_answer = self._detect_answer_block(text)

        # Extract answer using inherited method
        answer_text = ""
        trailing_after_answer = 0

        if has_answer:
            content, trailing, _ = self._get_answer_with_thinking_check(text)
            if content is not None:
                answer_text = content.strip()
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
    - Completion format: <SOLUTION>answer</SOLUTION> (consistent with GSM8K)
    - Gold answer: Direct from dataset['answer'] field (LaTeX format)
    - Validation: Multi-strategy (exact, symbolic via sympy, numeric)
    - Filtering: Supports level (1-5) and subject (7 domains)

    Answer types supported in dataset:
    - Numeric: 2, 18, 1.36, -5 (60% of dataset)
    - Fractions: \\frac{416}{27} (20% of dataset)
    - Radicals: 3\\sqrt{3}, \\frac{2\\sqrt{149}}{3} (19% of dataset)
    - Text/Special: \\text{June 20}, matrices (1% of dataset)

    Model completion format:
        <start_working_out>Step-by-step reasoning</end_working_out>
        <SOLUTION>\\frac{3}{4}</SOLUTION>

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
        """Extract gold answer from MATH dataset.
        
        Tries 'answer' field first, falls back to parsing \\boxed{...} from 'solution'.
        """
        # 1. Try direct field (pre-processed datasets)
        answer = task_payload.get("answer", "")
        if answer:
            return str(answer)

        # 2. Fallback: Parse from solution text
        solution = task_payload.get("solution", "")
        if solution:
            boxed = _extract_boxed_answer(str(solution))
            if boxed:
                return boxed

        return ""

    def _extract_completion_answer(self, completion: str, context: dict[str, Any]) -> str | None:
        """Extract answer from <SOLUTION>...</SOLUTION> tags."""
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
