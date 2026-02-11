#!/usr/bin/env python3
"""Custom reward functions for VeRL GRPO training.

This module provides reward functions that match the GRAIL environment
implementations for GSM8K, MATH, and MBPP datasets.

VeRL calls these functions with:
    compute_score(data_source, solution_str, ground_truth, extra_info=None)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

# Add project root to path for GRAIL imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from grail.environments.execution import check_code_executes
from grail.environments.math_hendrycks_env import _math_answers_equal

# ════════════════════════════════════════════════════════════════════════════
# TAGS AND PATTERNS
# ════════════════════════════════════════════════════════════════════════════
REASONING_START_TOKEN = "start_working_out"
REASONING_END_TOKEN = "end_working_out"
SOLUTION_START_TOKEN = "SOLUTION"
SOLUTION_END_TOKEN = "SOLUTION"

_HASH_PATTERN = re.compile(r"####\s*(?P<ans>.+)")
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
_NUMERIC_ONLY_PATTERN = re.compile(r"^[-+]?[\d.,]+$")


# ════════════════════════════════════════════════════════════════════════════
# GSM8K REWARD COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
def _parse_gsm8k_gold(raw_answer: str) -> str:
    """Parse GSM8K gold answer from #### format."""
    match = None
    for m in _HASH_PATTERN.finditer(raw_answer or ""):
        match = m
    if match is not None:
        return match.group("ans").strip()
    nums = list(_NUMBER_PATTERN.finditer(raw_answer or ""))
    if nums:
        return nums[-1].group(0).replace(",", "").strip()
    return ""


def _parse_gsm8k_completion(text: str) -> dict[str, Any]:
    """Parse GSM8K completion for answer and format."""
    flags = re.DOTALL | re.IGNORECASE
    has_thinking = bool(
        re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
    )
    answer_match = re.search(
        rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
    )

    answer_text = ""
    has_answer = bool(answer_match)
    is_numeric_only = False
    trailing = 0

    if answer_match:
        inside = answer_match.group(1).strip()
        num_match = _NUMBER_PATTERN.search(inside)
        if num_match:
            answer_text = num_match.group(0).replace(",", "").strip()
            is_numeric_only = bool(_NUMERIC_ONLY_PATTERN.match(inside.replace(" ", "")))
        trailing = len(text) - answer_match.end()

    return {
        "answer_text": answer_text,
        "has_thinking": has_thinking,
        "has_answer": has_answer,
        "is_numeric_only": is_numeric_only,
        "trailing": trailing,
    }


def compute_gsm8k_reward(solution_str: str, ground_truth: str) -> float:
    """Compute reward for GSM8K completion.

    Components (total 1.0):
    - Correctness (0.6): exact numeric match
    - Strict format (0.15): numeric-only + no trailing
    - Thinking (0.1): has thinking block
    - Answer (0.1): has answer block
    - No trailing (0.05): penalty for trailing text
    """
    parsed = _parse_gsm8k_completion(solution_str)
    gold_parsed = _parse_gsm8k_gold(ground_truth)

    # Validate answer
    pred_norm = re.sub(r"[\s\.]+$", "", parsed["answer_text"].strip().lower())
    gold_norm = re.sub(r"[\s\.]+$", "", gold_parsed.strip().lower())
    is_correct = pred_norm == gold_norm

    correctness = 0.6 if is_correct else 0.0
    strict_format = (
        0.15
        if (parsed["has_answer"] and parsed["is_numeric_only"] and parsed["trailing"] == 0)
        else 0.0
    )
    thinking = 0.1 if parsed["has_thinking"] else 0.0
    answer = 0.1 if parsed["has_answer"] else 0.0
    no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

    return correctness + strict_format + thinking + answer + no_trailing


# ════════════════════════════════════════════════════════════════════════════
# MATH REWARD COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
def _parse_math_completion(text: str) -> dict[str, Any]:
    """Parse MATH completion for answer and format."""
    flags = re.DOTALL | re.IGNORECASE
    has_thinking = bool(
        re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
    )
    answer_match = re.search(
        rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
    )

    answer_text = ""
    has_answer = bool(answer_match)
    trailing = 0

    if answer_match:
        answer_text = answer_match.group(1).strip()
        trailing = len(text) - answer_match.end()

    return {
        "answer_text": answer_text,
        "has_thinking": has_thinking,
        "has_answer": has_answer,
        "trailing": trailing,
    }


def compute_math_reward(solution_str: str, ground_truth: str) -> float:
    """Compute reward for MATH completion.

    Components (total 1.0):
    - Correctness (0.7): Multi-strategy validation (exact, symbolic, numeric)
    - Answer format (0.15): Has answer + minimal trailing
    - Thinking (0.1): Has thinking block
    - No trailing (0.05): Penalty for excessive trailing
    """
    parsed = _parse_math_completion(solution_str)

    is_correct = _math_answers_equal(parsed["answer_text"], ground_truth)

    correctness = 0.7 if is_correct else 0.0
    answer_format = 0.15 if (parsed["has_answer"] and parsed["trailing"] < 50) else 0.0
    thinking = 0.1 if parsed["has_thinking"] else 0.0
    no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

    return correctness + answer_format + thinking + no_trailing


# ════════════════════════════════════════════════════════════════════════════
# MBPP REWARD COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
def _parse_mbpp_completion(text: str) -> dict[str, Any]:
    """Parse MBPP completion for code and format."""
    flags = re.DOTALL | re.IGNORECASE
    has_thinking = bool(
        re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
    )
    solution_match = re.search(
        rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
    )

    code = ""
    has_solution = bool(solution_match)
    syntax_valid = False
    trailing = 0

    if solution_match:
        code = solution_match.group(1).strip()
        trailing = len(text) - solution_match.end()

        if code:
            try:
                compile(code, "<string>", "exec")
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False

    return {
        "code": code,
        "has_thinking": has_thinking,
        "has_solution": has_solution,
        "syntax_valid": syntax_valid,
        "trailing": trailing,
    }


def compute_mbpp_reward(solution_str: str, ground_truth: str) -> float:
    """Compute reward for MBPP completion.

    Components (total 1.0):
    - Correctness (0.7): Test pass rate
    - Syntax (0.1): Code compiles
    - Format (0.1): Has solution tags + minimal trailing
    - Thinking (0.1): Has thinking block
    """
    parsed = _parse_mbpp_completion(solution_str)

    # Parse test data from JSON
    try:
        test_data = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        test_data = {}

    correctness = 0.0
    if parsed["code"] and isinstance(test_data, dict):
        test_list = test_data.get("test_list", [])
        if test_list:
            test_setup = test_data.get("test_setup_code", "")
            test_imports = test_data.get("test_imports", [])

            setup_code = "\n".join(test_imports) if test_imports else ""
            if test_setup:
                setup_code += f"\n{test_setup}"

            test_cases = []
            for test in test_list:
                if setup_code:
                    test_cases.append(f"{setup_code}\n{test}")
                else:
                    test_cases.append(test)

            result = check_code_executes(parsed["code"], test_cases, timeout=5.0)
            if result["total"] > 0:
                pass_rate = result["passed"] / result["total"]
                correctness = 0.7 * pass_rate

    syntax = 0.1 if parsed["syntax_valid"] else 0.0
    solution_format = 0.1 if (parsed["has_solution"] and parsed["trailing"] < 50) else 0.0
    thinking = 0.1 if parsed["has_thinking"] else 0.0

    return correctness + syntax + solution_format + thinking


# ════════════════════════════════════════════════════════════════════════════
# MAIN REWARD FUNCTION (VeRL Entry Point)
# ════════════════════════════════════════════════════════════════════════════
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Compute reward score for a completion.

    This is the main entry point called by VeRL's reward system.
    Routes to the appropriate dataset-specific reward function.

    Args:
        data_source: Dataset identifier (e.g., "grail_gsm8k", "grail_math", "grail_mbpp")
        solution_str: The model's generated completion
        ground_truth: The expected answer/test data
        extra_info: Optional metadata dict

    Returns:
        Reward score between 0.0 and 1.0
    """
    # Route based on data_source
    if "gsm8k" in data_source.lower():
        return compute_gsm8k_reward(solution_str, ground_truth)
    elif "math" in data_source.lower():
        return compute_math_reward(solution_str, ground_truth)
    elif "mbpp" in data_source.lower():
        return compute_mbpp_reward(solution_str, ground_truth)
    else:
        # Default: return 0 for unknown datasets
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# TESTING / STANDALONE USAGE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Test the reward functions
    print("Testing reward functions...")

    # Test GSM8K
    gsm8k_completion = """<start_working_out>
Let me solve this step by step.
If John has 5 apples and gets 3 more, he has 5 + 3 = 8 apples.
</end_working_out>
<SOLUTION>8</SOLUTION>"""

    gsm8k_gold = "#### 8"
    gsm8k_reward = compute_score("grail_gsm8k", gsm8k_completion, gsm8k_gold)
    print(f"GSM8K reward: {gsm8k_reward:.2f} (expected: 1.0)")

    # Test MATH
    math_completion = """<start_working_out>
We need to find x such that x^2 = 4.
x = 2 or x = -2
</end_working_out>
<SOLUTION>2</SOLUTION>"""

    math_gold = "2"
    math_reward = compute_score("grail_math", math_completion, math_gold)
    print(f"MATH reward: {math_reward:.2f} (expected: 1.0)")

    # Test MBPP
    mbpp_completion = """<start_working_out>
I need to write a function that adds two numbers.
</end_working_out>
<SOLUTION>
def add(a, b):
    return a + b
</SOLUTION>"""

    mbpp_gold = json.dumps({
        "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        "test_setup_code": "",
        "test_imports": [],
    })
    mbpp_reward = compute_score("grail_mbpp", mbpp_completion, mbpp_gold)
    print(f"MBPP reward: {mbpp_reward:.2f} (expected: ~1.0)")
