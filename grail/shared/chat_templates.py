#!/usr/bin/env python3
"""
Shared chat template utilities for GRAIL.

Provides reusable chat template functions to avoid duplication across modules.

Chat Template Format (Qwen2.5 ChatML-style):
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_response}<|im_end|>

When add_generation_prompt=True, the template ends with:
    <|im_start|>assistant
    (model generates from here)

This signals the model that it's the assistant's turn to respond.
"""

from .prompt_constants import SYSTEM_PROMPT

# Qwen2.5 special tokens (ChatML format)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_qwen_chat_template(system_prompt: str | None = None) -> str:
    """
    Build Qwen2.5 ChatML-style chat template with proper turn markers.

    Args:
        system_prompt: The system prompt to inject. If None, uses SYSTEM_PROMPT
                       from prompt_constants.py.

    Returns:
        Jinja2 template string for Qwen2.5 chat formatting

    Template structure:
        - Each turn starts with <|im_start|>{role}\\n and ends with <|im_end|>
        - System prompt is injected if no system message in conversation
        - add_generation_prompt=True adds <|im_start|>assistant\\n to signal generation

    Note:
        The model is expected to generate <start_working_out> as part of its completion,
        not receive it in the prompt. This allows proper reward computation on the
        model's generated reasoning tokens.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    # Escape single quotes in system prompt for safe embedding in template
    escaped_prompt = system_prompt.replace("'", "\\'")

    # ─────────────────────────────────────────────────────────────────────────
    # Qwen2.5 ChatML Template
    # ─────────────────────────────────────────────────────────────────────────
    # Format: <|im_start|>role\n{content}<|im_end|>
    #
    # CRITICAL: add_generation_prompt must output <|im_start|>assistant\n
    # Without this, the model doesn't know it's the assistant's turn and may
    # immediately output <|im_end|> (EOS), resulting in 1-token completions.
    # ─────────────────────────────────────────────────────────────────────────
    chat_template = (
        # Handle system message: use provided or inject default
        "{% if messages[0]['role'] == 'system' %}"
        f"{{{{ '{IM_START}system\\n' + messages[0]['content'] + '{IM_END}\\n' }}}}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        f"{{{{ '{IM_START}system\\n{escaped_prompt}{IM_END}\\n' }}}}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        # Process user/assistant messages
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        f"{{{{ '{IM_START}user\\n' + message['content'] + '{IM_END}\\n' }}}}"
        "{% elif message['role'] == 'assistant' %}"
        f"{{{{ '{IM_START}assistant\\n' + message['content'] + '{IM_END}\\n' }}}}"
        "{% endif %}"
        "{% endfor %}"
        # CRITICAL: Signal assistant turn for generation
        "{% if add_generation_prompt %}"
        f"{{{{ '{IM_START}assistant\\n' }}}}"
        "{% endif %}"
    )

    return chat_template
