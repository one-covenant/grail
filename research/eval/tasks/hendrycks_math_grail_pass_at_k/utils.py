"""Pass@k utilities for MATH benchmark evaluation.

Implements the standard pass@k metric: given n samples, compute the probability
that at least one of k random samples is correct.

Formula: pass@k = 1 - C(n-c, k) / C(n, k)
where n = total samples, c = correct samples, k = samples to consider
"""

import re
from math import comb
from typing import Any


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | None = None,
) -> dict[str, float]:
    """Compute pass@k for math problems.

    Args:
        references: List of ground truth answers
        predictions: List of lists of model predictions (n samples per problem)
        k: List of k values to compute (e.g., [1, 5, 10])

    Returns:
        Dictionary with pass@k scores for each k value
    """
    if k is None:
        k = [1, 5]
    if isinstance(k, int):
        k = [k]

    results = {}
    for k_val in k:
        pass_at_k_scores = []
        for ref, preds in zip(references, predictions, strict=False):
            n = len(preds)
            if n < k_val:
                # If we have fewer samples than k, use what we have
                c = sum(1 for p in preds if is_equiv(extract_answer(p), ref))
                score = 1.0 if c > 0 else 0.0
            else:
                c = sum(1 for p in preds if is_equiv(extract_answer(p), ref))
                score = _pass_at_k(n, c, k_val)
            pass_at_k_scores.append(score)

        avg_score = sum(pass_at_k_scores) / len(pass_at_k_scores) if pass_at_k_scores else 0.0
        results[f"pass@{k_val}"] = avg_score

    return results


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


def extract_answer(text: str) -> str:
    """Extract answer from model output.

    Tries multiple extraction patterns in order:
    1. <SOLUTION>...</SOLUTION> tags (for reasoning models)
    2. \\boxed{...} (LaTeX boxed answers)
    3. $...$ (LaTeX inline math at end)
    4. Last number in text
    """
    if text is None:
        return ""

    # Try <SOLUTION> tags first (reasoning models)
    solution_match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()

    # Try \boxed{}
    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed)
        except (AssertionError, IndexError):
            pass

    # Try $...$ at end
    indices = [pos for pos, char in enumerate(text) if char == "$"]
    if len(indices) >= 2:
        return text[indices[-2] + 1 : indices[-1]]

    # Fallback: return cleaned text
    return text.strip()


def is_equiv(str1: str, str2: str) -> bool:
    """Check if two answers are mathematically equivalent."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{} content from string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def strip_string(string: str) -> str:
    """Normalize string for comparison."""
    if string is None:
        return ""

    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove common LaTeX
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("$", "")
    string = string.replace("\\$", "")

    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # Remove spaces
    string = string.replace(" ", "")

    return string


# For lm-eval metric interface
def process_results(doc: dict, results: list[str]) -> dict[str, Any]:
    """Process results for a single document (used by lm-eval).

    This function is called for EACH sample. The pass@k aggregation
    happens in the aggregation function.
    """
    answer = doc.get("answer", "")

    # Check each result
    correct_list = []
    for result in results:
        extracted = extract_answer(result)
        is_correct = 1 if is_equiv(extracted, str(answer)) else 0
        correct_list.append(is_correct)

    return {
        "pass": correct_list,  # List of 0/1 for each sample
        "num_correct": sum(correct_list),
        "num_samples": len(correct_list),
    }


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
    return aggregate_pass_at_k(results, k=1)


def aggregate_pass_at_5(results: list[dict]) -> float:
    return aggregate_pass_at_k(results, k=5)


def aggregate_pass_at_10(results: list[dict]) -> float:
    return aggregate_pass_at_k(results, k=10)
