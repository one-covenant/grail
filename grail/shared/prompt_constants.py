"""Backward-compatible exports. All values derived from active ThinkingConfig.

New code should import from grail.shared.thinking directly.
"""

from __future__ import annotations

from .thinking import get_thinking_config

_cfg = get_thinking_config()

# Unbracketed tokens (used by parsers/regex)
REASONING_START_TOKEN = _cfg.thinking_open_token  # "think" or "start_working_out"
REASONING_END_TOKEN = _cfg.thinking_close_token  # "think" or "end_working_out"
SOLUTION_START_TOKEN = _cfg.solution_open_token  # "SOLUTION"
SOLUTION_END_TOKEN = _cfg.solution_close_token  # "SOLUTION"

# Bracketed forms (derived from tokens above)
REASONING_START = _cfg.thinking_open  # "<think>" or "<start_working_out>"
REASONING_END = _cfg.thinking_close  # "</think>" or "</end_working_out>"
SOLUTION_START = _cfg.solution_open  # "<SOLUTION>"
SOLUTION_END = _cfg.solution_close  # "</SOLUTION>"

SYSTEM_PROMPT = _cfg.system_prompt

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
