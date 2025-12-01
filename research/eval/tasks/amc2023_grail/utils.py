"""AMC 2023 evaluation utilities for GRAIL reasoning models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and uses robust
numeric comparison for AMC answers.
"""

import re


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process model output and compare with target answer."""
    retval = 0
    response = results[0]

    # Extract answer from <SOLUTION>...</SOLUTION> tags first (for reasoning models)
    solution_match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", response, re.DOTALL)
    if solution_match:
        answer = solution_match.group(1).strip()
    else:
        # Fallback: try to extract from $...$ format
        indices = [pos for pos, char in enumerate(response) if char == "$"]
        if len(indices) >= 2:
            answer = response[indices[0] + 1 : indices[-1]]
        else:
            # Fallback: try to extract from \boxed{}
            boxed_answer = last_boxed_only_string(response)
            if boxed_answer is not None:
                try:
                    answer = remove_boxed(boxed_answer)
                except (AssertionError, IndexError):
                    answer = response
            else:
                answer = response

    # Get target answer
    target = str(doc.get("answer", ""))

    # Compare answers
    if is_equiv(answer, target):
        retval = 1

    return {"exact_match": retval}


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    """Check if two answers are equivalent."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)

        if verbose:
            print(f"Comparing: '{ss1}' vs '{ss2}'")

        # Direct string comparison
        if ss1 == ss2:
            return True

        # Try numeric comparison
        try:
            num1 = extract_number(ss1)
            num2 = extract_number(ss2)
            if num1 is not None and num2 is not None:
                # Compare as floats with tolerance
                return abs(num1 - num2) < 0.01
        except (ValueError, TypeError):
            pass

        return False
    except Exception:
        return str1 == str2


def extract_number(s: str) -> float:
    """Extract number from string."""
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

    # Try to find numbers in the string
    matches = re.findall(r"-?\d+\.?\d*", s)
    if matches:
        return float(matches[-1])

    return None


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


def last_boxed_only_string(string: str) -> str:
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

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("$", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")

    # Handle float formatting (e.g., "27.0" -> "27")
    try:
        num = float(string)
        if num == int(num):
            string = str(int(num))
    except ValueError:
        pass

    return string
