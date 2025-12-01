"""GSM8K evaluation utilities for GRAIL reasoning models.

Extracts answers from <SOLUTION>...</SOLUTION> tags and compares with
the ground truth answer (after ####).
"""

import re


def doc_to_target(doc: dict) -> str:
    """Convert document to target format for GRAIL reasoning."""
    answer = doc["answer"]
    # Extract final answer after ####
    if "####" in answer:
        final_answer = answer.split("####")[-1].strip()
    else:
        final_answer = answer.strip()

    # Extract the reasoning part (before ####)
    if "####" in answer:
        reasoning = answer.split("####")[0].strip()
    else:
        reasoning = ""

    return (
        f"<start_working_out>\n{reasoning}\n</end_working_out>\n<SOLUTION>{final_answer}</SOLUTION>"
    )


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Process model output and compare with target answer."""
    retval = 0
    response = results[0]

    # Extract answer from <SOLUTION>...</SOLUTION> tags first (for reasoning models)
    solution_match = re.search(r"<SOLUTION>(.*?)</SOLUTION>", response, re.DOTALL)
    if solution_match:
        answer = solution_match.group(1).strip()
    else:
        # Fallback: try to extract number from the end of response
        answer = extract_last_number(response)

    # Get target answer from document
    target_answer = doc["answer"]
    if "####" in target_answer:
        target = target_answer.split("####")[-1].strip()
    else:
        target = target_answer.strip()

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
        # Clean both strings
        s1 = clean_answer(str1)
        s2 = clean_answer(str2)

        if verbose:
            print(f"Comparing: '{s1}' vs '{s2}'")

        # Direct string comparison
        if s1 == s2:
            return True

        # Try numeric comparison
        try:
            num1 = extract_number(s1)
            num2 = extract_number(s2)
            if num1 is not None and num2 is not None:
                return abs(num1 - num2) < 0.001
        except (ValueError, TypeError):
            pass

        return False
    except Exception:
        return str1 == str2


def clean_answer(s: str) -> str:
    """Clean answer string for comparison."""
    if s is None:
        return ""

    s = s.strip()
    # Remove dollar signs, commas, and common formatting
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Remove trailing period
    s = s.rstrip(".")

    return s


def extract_number(s: str) -> float:
    """Extract number from string."""
    if s is None:
        return None

    s = clean_answer(s)

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


def extract_last_number(s: str) -> str:
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
