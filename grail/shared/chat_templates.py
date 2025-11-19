#!/usr/bin/env python3
"""
Shared chat template utilities for GRAIL.

Provides reusable chat template functions to avoid duplication across modules.
"""


def build_qwen_chat_template(system_prompt: str) -> str:
    """
    Build Qwen-style chat template with system prompt.

    Args:
        system_prompt: The system prompt to inject
        reasoning_start: DEPRECATED - No longer used. The model generates reasoning tokens.

    Returns:
        Jinja2 template string for Qwen-style chat formatting

    Note:
        The model is expected to generate <start_working_out> as part of its completion,
        not receive it in the prompt. This allows proper reward computation on the
        model's generated reasoning tokens.
    """

    # TODO: later support jinja files for different system prompts, etc
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '{system_prompt}' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '' }}"
        "{% endif %}"
    )

    chat_template = chat_template.replace(
        "'{system_prompt}'",
        f"'{system_prompt}'",
    )

    return chat_template
