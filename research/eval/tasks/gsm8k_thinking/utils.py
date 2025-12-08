"""GSM8K evaluation utilities for thinking models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and compares with
the ground truth answer (after ####).
"""

import re
import sys
from pathlib import Path

# Add parent directory to path for _common import
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import (
    extract_solution_tag,
    is_equiv_combined,
)


def doc_to_target(doc: dict) -> str:
    """Convert document to target format for thinking."""
    answer = doc["answer"]

    # Extract final answer after ####
    if "####" in answer:
        final_answer = answer.split("####")[-1].strip()
        reasoning = answer.split("####")[0].strip()
    else:
        final_answer = answer.strip()
        reasoning = ""

    return (
        f"<start_working_out>\n{reasoning}\n</end_working_out>\n<SOLUTION>{final_answer}</SOLUTION>"
    )


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process model output and compare with target answer."""
    response = results[0]

    # Extract answer from <SOLUTION> tags first (reasoning models)
    answer = extract_solution_tag(response)
    if answer is None:
        # Fallback: try to extract number from the end
        answer = _extract_last_number(response)

    # Get target answer from document
    target_answer = doc["answer"]
    if "####" in target_answer:
        target = target_answer.split("####")[-1].strip()
    else:
        target = target_answer.strip()

    # Compare answers
    is_correct = is_equiv_combined(
        answer,
        target,
        normalizer=_clean_answer,
        try_numeric=True,
        tolerance=0.001,
    )

    return {"exact_match": 1 if is_correct else 0}


def _clean_answer(s: str) -> str:
    """Clean answer string for GSM8K comparison."""
    if s is None:
        return ""

    s = s.strip()
    # Remove dollar signs, commas, and common formatting
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Remove trailing period
    s = s.rstrip(".")

    return s


def _extract_last_number(s: str) -> str:
    """Extract the last number from a string."""
    if s is None:
        return ""

    # Look for #### pattern first (GSM8K format)
    if "####" in s:
        return s.split("####")[-1].strip()

    # Find all numbers
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s)
    if matches:
        # Return last number, removing commas
        return matches[-1].replace(",", "")

    return s.strip()
