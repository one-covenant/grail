"""AIME 2024 evaluation utilities for GRAIL reasoning models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and uses robust
integer comparison for AIME answers (which are always 0-999).
"""

import re


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process model output and compare with target answer.

    AIME answers are always integers from 000-999.
    """
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
    answer_key = next((k for k in doc.keys() if k.lower() == "answer"), None)
    if answer_key is None:
        return {"exact_match": 0}

    target = str(doc[answer_key])

    # AIME answers are integers 0-999, so try integer comparison
    if is_equiv(answer, target):
        retval = 1

    return {"exact_match": retval}


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    """Check if two answers are equivalent.

    For AIME, answers are integers 0-999. We try to extract and compare integers.
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        # Clean and normalize strings
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)

        if verbose:
            print(f"Comparing: '{ss1}' vs '{ss2}'")

        # Direct string comparison
        if ss1 == ss2:
            return True

        # Try integer comparison (AIME answers are always integers)
        try:
            int1 = extract_integer(ss1)
            int2 = extract_integer(ss2)
            if int1 is not None and int2 is not None:
                return int1 == int2
        except (ValueError, TypeError):
            pass

        return False
    except Exception:
        return str1 == str2


def extract_integer(s: str) -> int:
    """Extract integer from string, handling common formats."""
    if s is None:
        return None

    s = s.strip()

    # Remove common wrappers
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = s.strip()

    # Try direct integer parse
    try:
        return int(s)
    except ValueError:
        pass

    # Try to find integers in the string
    matches = re.findall(r"-?\d+", s)
    if matches:
        # Return the last integer found (usually the final answer)
        return int(matches[-1])

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

    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove common LaTeX
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove dollar signs
    string = string.replace("$", "")
    string = string.replace("\\$", "")

    # Remove spaces
    string = string.replace(" ", "")

    # Remove leading zeros for integer comparison
    string = string.lstrip("0") or "0"

    return string
