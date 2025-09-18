#!/usr/bin/env python3
"""
Shared chat template utilities for GRAIL.

Provides reusable chat template functions to avoid duplication across modules.
"""


def build_qwen_chat_template(system_prompt: str, reasoning_start: str) -> str:
    """
    Build Qwen-style chat template with system prompt and reasoning start.

    Args:
        system_prompt: The system prompt to inject
        reasoning_start: The reasoning start token (e.g., "<start_working>")

    Returns:
        Jinja2 template string for Qwen-style chat formatting
    """
    # Keep Jinja2 template exactly as specified; substitute via string replace
    # TODO: later support jinja files for different system prompts,
    # prompts, etc
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
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
        "{% endif %}"
    )

    chat_template = chat_template.replace(
        "'{system_prompt}'",
        f"'{system_prompt}'",
    )
    chat_template = chat_template.replace(
        """'{reasoning_start}'""",
        f"'{reasoning_start}'",
    )
    return chat_template
