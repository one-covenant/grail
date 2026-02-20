"""Single source of truth for thinking mode configuration.

Supports two modes:
- NATIVE: Model has built-in thinking (e.g., Qwen3 <think>/<think>)
- INSTRUCTED: System prompt instructs custom thinking tags (e.g., Qwen2.5)

Usage:
    from grail.shared.thinking import get_thinking_config, ThinkingMode
    cfg = get_thinking_config()           # reads GRAIL_THINKING_MODE env
    cfg = get_thinking_config("native")   # explicit mode
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class ThinkingMode(str, Enum):
    NATIVE = "native"  # Model has built-in thinking (Qwen3 <think>)
    INSTRUCTED = "instructed"  # System prompt instructs custom thinking tags


@dataclass(frozen=True)
class ThinkingConfig:
    """Configuration for a thinking mode.

    Token names are stored without brackets; bracketed forms are derived.
    """

    mode: ThinkingMode
    thinking_open_token: str  # e.g. "think" or "start_working_out"
    thinking_close_token: str  # e.g. "think" or "end_working_out"
    solution_open_token: str  # e.g. "SOLUTION"
    solution_close_token: str  # e.g. "SOLUTION"
    system_prompt: str  # Mode-appropriate system prompt
    use_custom_template: bool  # True=instructed (build ChatML), False=native (use model's)
    enable_thinking: bool  # Passed to apply_chat_template for native models

    @property
    def thinking_open(self) -> str:
        return f"<{self.thinking_open_token}>"

    @property
    def thinking_close(self) -> str:
        return f"</{self.thinking_close_token}>"

    @property
    def solution_open(self) -> str:
        return f"<{self.solution_open_token}>"

    @property
    def solution_close(self) -> str:
        return f"</{self.solution_close_token}>"


INSTRUCTED_CONFIG = ThinkingConfig(
    mode=ThinkingMode.INSTRUCTED,
    thinking_open_token="start_working_out",
    thinking_close_token="end_working_out",
    solution_open_token="SOLUTION",
    solution_close_token="SOLUTION",
    system_prompt=(
        "You are given a problem. Think step by step.\n"
        "Place your reasoning between <start_working_out> and </end_working_out>.\n"
        "You may write and reason about code during your thinking.\n"
        "After thinking, you MUST provide your final solution wrapped in <SOLUTION> and </SOLUTION> tags.\n"
        "Any code outside <SOLUTION> tags will be ignored. Your answer will score 0 without these tags."
    ),
    use_custom_template=True,
    enable_thinking=False,
)

NATIVE_CONFIG = ThinkingConfig(
    mode=ThinkingMode.NATIVE,
    thinking_open_token="think",
    thinking_close_token="think",
    solution_open_token="SOLUTION",
    solution_close_token="SOLUTION",
    system_prompt=(
        "You are given a problem. Think step by step.\n"
        "You may write and reason about code during your thinking.\n"
        "After thinking, you MUST provide your final solution wrapped in <SOLUTION> and </SOLUTION> tags.\n"
        "Any code outside <SOLUTION> tags will be ignored. Your answer will score 0 without these tags."
    ),
    use_custom_template=False,
    enable_thinking=True,
)


def get_thinking_config(mode: ThinkingMode | str | None = None) -> ThinkingConfig:
    """Return config for given mode. Reads GRAIL_THINKING_MODE env var if None."""
    if mode is None:
        mode = os.getenv("GRAIL_THINKING_MODE", "native")
    if isinstance(mode, str):
        mode = ThinkingMode(mode.lower())
    return {
        ThinkingMode.INSTRUCTED: INSTRUCTED_CONFIG,
        ThinkingMode.NATIVE: NATIVE_CONFIG,
    }[mode]
