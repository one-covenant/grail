"""AIME 2024 evaluation utilities for thinking models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and uses robust
integer comparison for AIME answers (which are always 0-999).
"""

import sys
from pathlib import Path

# Add parent directory to path for _common import
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import (
    extract_answer_cascade,
    is_equiv_combined,
    strip_string_basic,
)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process model output and compare with target answer.

    AIME answers are always integers from 000-999.
    """
    response = results[0]

    # Extract answer using cascade (SOLUTION tag -> boxed -> dollar -> raw)
    answer = extract_answer_cascade(
        response,
        try_solution_tag=True,
        try_boxed=True,
        try_dollar=True,
    )

    # Get target answer
    answer_key = next((k for k in doc.keys() if k.lower() == "answer"), None)
    if answer_key is None:
        return {"exact_match": 0}

    target = str(doc[answer_key])

    # AIME answers are integers 0-999, use integer comparison
    is_correct = is_equiv_combined(
        answer,
        target,
        normalizer=_strip_string_aime,
        try_numeric=True,
        integer_only=True,
    )

    return {"exact_match": 1 if is_correct else 0}


def _strip_string_aime(string: str) -> str:
    """Normalize string for AIME comparison.

    Extends basic normalization with leading zero removal.
    """
    string = strip_string_basic(string)
    # Remove leading zeros for integer comparison (but keep "0")
    string = string.lstrip("0") or "0"
    return string
