"""Custom utils for thinking model evaluation on MATH.

Extracts answers from <SOLUTION>...</SOLUTION> tags instead of \\boxed{}.
"""

import sys
from pathlib import Path

import datasets

# Add parent directory to path for _common import
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import (
    extract_solution_tag,
    is_equiv_string,
    last_boxed_only_string,
    remove_boxed,
    strip_string_math,
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process dataset docs - extract ground truth answer from \\boxed{}."""

    def _process_doc(doc: dict) -> dict:
        boxed = last_boxed_only_string(doc["solution"])
        answer = remove_boxed(boxed) if boxed else ""
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": answer,
        }

    return dataset.map(_process_doc)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process results - extract answer from <SOLUTION> tags and compare."""
    # Extract from <SOLUTION> tags
    model_answer = extract_solution_tag(results[0])
    if model_answer is None:
        model_answer = results[0].strip()

    # Get ground truth (already extracted from \boxed{} in process_docs)
    ground_truth = doc.get("answer")
    if ground_truth is None:
        boxed = last_boxed_only_string(doc["solution"])
        ground_truth = remove_boxed(boxed) if boxed else ""

    # Compare using full math normalization
    is_correct = is_equiv_string(model_answer, ground_truth, strip_string_math)

    return {"exact_match": 1 if is_correct else 0}
