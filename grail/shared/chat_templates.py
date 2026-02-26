"""Chat template utilities for GRAIL.

Supports two modes via ThinkingConfig:
- Instructed: installs a generic ChatML template; system prompt instructs thinking tags
- Native: leaves the model's own template untouched; passes enable_thinking=True

Template strategy:
    configure_tokenizer()  -- set up the right template for the mode
    prepare_messages()     -- inject system prompt at message level
    apply_chat_template()  -- unified helper combining both steps
"""

from __future__ import annotations

from typing import Any

from .thinking import ThinkingConfig, get_thinking_config

# ChatML special tokens
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def prepare_messages(
    messages: list[dict[str, str]],
    config: ThinkingConfig | None = None,
) -> list[dict[str, str]]:
    """Prepend system message if not present. Works for both modes."""
    config = config or get_thinking_config()
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": config.system_prompt}] + list(messages)


def build_instructed_chat_template() -> str:
    """ChatML Jinja2 template for instructed-thinking models.

    Used when config.use_custom_template is True. No baked-in system prompt.
    """
    return (
        "{% for message in messages %}"
        f"{{{{ '{IM_START}' + message['role'] + '\\n' + message['content'] + '{IM_END}\\n' }}}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{{{{ '{IM_START}assistant\\n' }}}}"
        "{% endif %}"
    )


def configure_tokenizer(tokenizer: Any, config: ThinkingConfig | None = None) -> Any:
    """Set up tokenizer template based on thinking mode.

    Instructed: installs custom ChatML template.
    Native: leaves model's own template untouched.
    """
    config = config or get_thinking_config()
    if config.use_custom_template:
        tokenizer.chat_template = build_instructed_chat_template()
    return tokenizer


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: ThinkingConfig | None = None,
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Unified helper: prepare_messages + apply_chat_template + enable_thinking."""
    config = config or get_thinking_config()
    prepared = prepare_messages(messages, config)
    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if config.enable_thinking:
        kwargs["enable_thinking"] = True
    return tokenizer.apply_chat_template(prepared, **kwargs)
