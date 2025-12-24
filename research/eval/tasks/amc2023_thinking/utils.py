"""AMC 2023 evaluation utilities for thinking models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and uses robust
numeric comparison for AMC answers.
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
    """Process model output and compare with target answer."""
    response = results[0]

    # Extract answer using cascade (SOLUTION tag first for thinking models)
    answer = extract_answer_cascade(
        response,
        try_solution_tag=True,
        try_boxed=True,
        try_dollar=True,
    )

    # Get target answer
    target = str(doc.get("answer", ""))

    # Compare with numeric fallback
    is_correct = is_equiv_combined(
        answer,
        target,
        normalizer=_strip_string_amc,
        try_numeric=True,
        tolerance=0.01,
    )

    return {"exact_match": 1 if is_correct else 0}


def _strip_string_amc(string: str) -> str:
    """Normalize string for AMC comparison.

    Extends basic normalization with float->int conversion.
    """
    string = strip_string_basic(string)

    # Handle float formatting (e.g., "27.0" -> "27")
    try:
        num = float(string)
        if num == int(num):
            string = str(int(num))
    except ValueError:
        pass

    return string
