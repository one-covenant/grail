"""Shared utilities for thinking model evaluation tasks.

This module contains common functions for answer extraction and comparison
used across multiple evaluation tasks (AIME, AMC, GSM8K, MATH, etc.).

Following DRY principles - extract once, reuse everywhere.
"""

import re
from collections.abc import Callable

# =============================================================================
# Answer Extraction Functions
# =============================================================================


def extract_solution_tag(text: str) -> str | None:
    """Extract content from <SOLUTION>...</SOLUTION> tags.

    Args:
        text: Model output text

    Returns:
        Content inside SOLUTION tags, or None if not found
    """
    match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_dollar_sign_answer(text: str) -> str | None:
    """Extract answer from $...$ format (last pair).

    Args:
        text: Model output text

    Returns:
        Content between last pair of dollar signs, or None if not found
    """
    indices = [pos for pos, char in enumerate(text) if char == "$"]
    if len(indices) >= 2:
        return text[indices[-2] + 1 : indices[-1]]
    return None


def remove_boxed(s: str) -> str | None:
    """Remove \\boxed{} wrapper from string.

    Args:
        s: String potentially wrapped in \\boxed{}

    Returns:
        Unwrapped content, or original string if no valid wrapper found
    """
    if s is None:
        return None

    # Handle "\\boxed " format (space after boxed)
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    # Handle "\\boxed{...}" format
    left = "\\boxed{"
    if s[: len(left)] == left and s.endswith("}"):
        return s[len(left) : -1]

    return s


def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{} or \\fbox{} content from a string.

    Handles nested braces correctly.

    Args:
        string: Text containing potential boxed content

    Returns:
        The last boxed expression (including \\boxed{} wrapper), or None
    """
    if not string:
        return None

    idx = string.rfind("\\boxed")

    # Handle "\\boxed " format
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    # Find matching closing brace
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


def extract_answer_cascade(
    text: str,
    try_solution_tag: bool = True,
    try_boxed: bool = True,
    try_dollar: bool = True,
) -> str:
    """Extract answer using cascade of methods.

    Tries extraction methods in order until one succeeds:
    1. <SOLUTION>...</SOLUTION> tags (optional)
    2. \\boxed{...} (optional)
    3. $...$ format (optional)
    4. Original text (fallback)

    Args:
        text: Model output text
        try_solution_tag: Whether to try SOLUTION tag extraction
        try_boxed: Whether to try boxed extraction
        try_dollar: Whether to try dollar sign extraction

    Returns:
        Extracted answer string
    """
    if not text:
        return ""

    # Try SOLUTION tags (reasoning models)
    if try_solution_tag:
        result = extract_solution_tag(text)
        if result:
            return result

    # Try boxed format
    if try_boxed:
        boxed = last_boxed_only_string(text)
        if boxed:
            unboxed = remove_boxed(boxed)
            if unboxed:
                return unboxed

    # Try dollar sign format
    if try_dollar:
        result = extract_dollar_sign_answer(text)
        if result:
            return result

    return text.strip()


# =============================================================================
# String Normalization Functions
# =============================================================================


def strip_string_basic(string: str) -> str:
    """Basic string normalization for comparison.

    Removes common formatting that doesn't affect mathematical meaning:
    - Linebreaks, spaces
    - LaTeX commands: \\!, \\left, \\right
    - Dollar signs

    Args:
        string: String to normalize

    Returns:
        Normalized string
    """
    if string is None:
        return ""

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("$", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")

    return string


def fix_fracs(string: str) -> str:
    """Fix fraction formatting (\\frac12 -> \\frac{1}{2})."""
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr or substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr

    return new_str


def fix_sqrt(string: str) -> str:
    """Fix sqrt formatting (\\sqrt2 -> \\sqrt{2})."""
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")
    new_string = splits[0]

    for split in splits[1:]:
        if split and split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr

    return new_string


def fix_a_slash_b(string: str) -> str:
    """Convert simple fractions a/b to \\frac{a}{b}."""
    if len(string.split("/")) != 2:
        return string

    a_str, b_str = string.split("/")
    try:
        a = int(a_str)
        b = int(b_str)
        if string == f"{a}/{b}":
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except ValueError:
        pass

    return string


def remove_right_units(string: str) -> str:
    """Remove units on the right side (e.g., '5 \\text{ meters}')."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def strip_string_math(string: str) -> str:
    """Full math string normalization for MATH benchmark.

    Includes all basic normalization plus:
    - tfrac/dfrac -> frac
    - Degrees removal
    - Units removal
    - Fraction normalization
    - Leading decimal fixes

    Args:
        string: String to normalize

    Returns:
        Normalized string
    """
    if string is None:
        return ""

    # Basic cleanup
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove degrees
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")

    # Remove units
    string = remove_right_units(string)

    # Remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # Fix leading decimals
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Handle "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # Fix sqrt formatting
    string = fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fractions
    string = fix_fracs(string)

    # Special case: 0.5 -> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Convert a/b to \frac{a}{b}
    string = fix_a_slash_b(string)

    return string


# =============================================================================
# Number Extraction Functions
# =============================================================================


def extract_integer(s: str) -> int | None:
    """Extract integer from string, handling common formats.

    Args:
        s: String potentially containing an integer

    Returns:
        Extracted integer, or None if not found
    """
    if s is None:
        return None

    s = s.strip()

    # Remove common wrappers
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.strip()

    # Try direct parse
    try:
        return int(s)
    except ValueError:
        pass

    # Find integers in the string (return last one)
    matches = re.findall(r"-?\d+", s)
    if matches:
        return int(matches[-1])

    return None


def extract_float(s: str) -> float | None:
    """Extract float from string, handling common formats.

    Args:
        s: String potentially containing a number

    Returns:
        Extracted float, or None if not found
    """
    if s is None:
        return None

    s = s.strip()

    # Remove common wrappers
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.strip()

    # Try direct parse
    try:
        return float(s)
    except ValueError:
        pass

    # Find numbers in the string (return last one)
    matches = re.findall(r"-?\d+\.?\d*", s)
    if matches:
        return float(matches[-1])

    return None


# =============================================================================
# Equivalence Checking Functions
# =============================================================================


def is_equiv_string(
    str1: str,
    str2: str,
    normalizer: Callable[[str], str] = strip_string_basic,
) -> bool:
    """Check if two strings are equivalent after normalization.

    Args:
        str1: First string
        str2: Second string
        normalizer: Function to normalize strings before comparison

    Returns:
        True if equivalent, False otherwise
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = normalizer(str1)
        ss2 = normalizer(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def is_equiv_numeric(
    str1: str,
    str2: str,
    tolerance: float = 0.01,
    integer_only: bool = False,
) -> bool:
    """Check if two strings represent equivalent numbers.

    Args:
        str1: First string
        str2: Second string
        tolerance: Absolute tolerance for float comparison
        integer_only: If True, only compare as integers

    Returns:
        True if equivalent, False otherwise
    """
    if str1 is None or str2 is None:
        return str1 is None and str2 is None

    if integer_only:
        int1 = extract_integer(str1)
        int2 = extract_integer(str2)
        if int1 is not None and int2 is not None:
            return int1 == int2
    else:
        num1 = extract_float(str1)
        num2 = extract_float(str2)
        if num1 is not None and num2 is not None:
            return abs(num1 - num2) < tolerance

    return False


def is_equiv_combined(
    str1: str,
    str2: str,
    normalizer: Callable[[str], str] = strip_string_basic,
    try_numeric: bool = True,
    tolerance: float = 0.01,
    integer_only: bool = False,
) -> bool:
    """Check equivalence using both string and numeric comparison.

    Args:
        str1: First string
        str2: Second string
        normalizer: Function to normalize strings
        try_numeric: Whether to try numeric comparison
        tolerance: Tolerance for float comparison
        integer_only: If True, only do integer comparison

    Returns:
        True if equivalent by any method
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        # Try string comparison first
        ss1 = normalizer(str1)
        ss2 = normalizer(str2)
        if ss1 == ss2:
            return True

        # Try numeric comparison
        if try_numeric:
            return is_equiv_numeric(ss1, ss2, tolerance, integer_only)

        return False
    except Exception:
        return str1 == str2
