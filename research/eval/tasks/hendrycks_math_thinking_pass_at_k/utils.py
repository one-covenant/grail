"""Pass@k utilities for MATH benchmark evaluation.

Implements the standard pass@k metric: given n samples, compute the probability
that at least one of k random samples is correct.

Formula: pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples, k = samples to consider
"""

import sys
from math import comb
from pathlib import Path
from typing import Any

# Add parent directory to path for _common import
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import (
    extract_answer_cascade,
    is_equiv_string,
    strip_string_math,
)


def process_results(doc: dict, results: list[str]) -> dict[str, Any]:
    """Process results for a single document (used by lm-eval).

    This function is called for EACH sample. The pass@k aggregation
    happens in the aggregation function.
    """
    answer = str(doc.get("answer", ""))

    # Check each result
    correct_list = []
    for result in results:
        extracted = extract_answer_cascade(
            result,
            try_solution_tag=True,
            try_boxed=True,
            try_dollar=True,
        )
        is_correct = 1 if is_equiv_string(extracted, answer, strip_string_math) else 0
        correct_list.append(is_correct)

    return {
        "pass": correct_list,
        "num_correct": sum(correct_list),
        "num_samples": len(correct_list),
    }


def _pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k for a single problem.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider

    Returns:
        Probability that at least one of k samples is correct
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate_pass_at_k(results: list[dict], k: int = 5) -> float:
    """Aggregate pass@k across all documents.

    Args:
        results: List of result dicts from process_results
        k: Number of samples to consider for pass@k

    Returns:
        Average pass@k score
    """
    scores = []
    for r in results:
        n = r["num_samples"]
        c = r["num_correct"]
        if n < k:
            score = 1.0 if c > 0 else 0.0
        else:
            score = _pass_at_k(n, c, k)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


# Convenience aggregation functions for different k values
def aggregate_pass_at_1(results: list[dict]) -> float:
    """Aggregate pass@1 score."""
    return aggregate_pass_at_k(results, k=1)


def aggregate_pass_at_5(results: list[dict]) -> float:
    """Aggregate pass@5 score."""
    return aggregate_pass_at_k(results, k=5)


def aggregate_pass_at_10(results: list[dict]) -> float:
    """Aggregate pass@10 score."""
    return aggregate_pass_at_k(results, k=10)
