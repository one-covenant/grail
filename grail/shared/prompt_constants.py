"""Single source of truth for reasoning/solution tags and system prompt.

Centralizes the tags used across parsing, prompting, and validation, and
provides a canonical SYSTEM_PROMPT string that references these tags.
"""

from __future__ import annotations

# Unbracketed tag tokens (used by parsers and regex)
REASONING_START_TOKEN = "start_working_out"
REASONING_END_TOKEN = "end_working_out"
SOLUTION_START_TOKEN = "SOLUTION"
SOLUTION_END_TOKEN = "SOLUTION"

# Bracketed forms (used in prompts/templates)
REASONING_START = f"<{REASONING_START_TOKEN}>"
REASONING_END = f"</{REASONING_END_TOKEN}>"
SOLUTION_START = f"<{SOLUTION_START_TOKEN}>"
SOLUTION_END = f"</{SOLUTION_END_TOKEN}>"

# Canonical system prompt referencing the tags above
SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your code between {SOLUTION_START} and {SOLUTION_END}."
)

__all__ = [
    # tokens
    "REASONING_START_TOKEN",
    "REASONING_END_TOKEN",
    "SOLUTION_START_TOKEN",
    "SOLUTION_END_TOKEN",
    # bracketed
    "REASONING_START",
    "REASONING_END",
    "SOLUTION_START",
    "SOLUTION_END",
    # prompt
    "SYSTEM_PROMPT",
]
