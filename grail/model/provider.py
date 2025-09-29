"""Model and tokenizer provider for GRAIL.

Centralized loading functions to ensure consistent configuration across
all components (Prover, Verifier, Trainer).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_name: str,
    *,
    pad_token_strategy: str = "eos",
    chat_template: str | None = None,
) -> AutoTokenizer:
    """Load tokenizer with consistent configuration.

    Args:
        model_name: HuggingFace model identifier
        pad_token_strategy: How to handle missing pad_token ("eos" or "none")
        chat_template: Optional chat template string to install

    Returns:
        Configured AutoTokenizer instance
    """
    logger.debug(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad_token is set to prevent model confusion
    if pad_token_strategy == "eos" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token to eos_token")

    # Install custom chat template if provided
    if chat_template is not None:
        try:
            tokenizer.chat_template = chat_template
            logger.debug("Installed custom chat template")
        except Exception as e:
            logger.warning(f"Failed to set chat template: {e}")

    return tokenizer


def get_model(
    model_name: str,
    *,
    device: str | None = None,
    use_safetensors: bool = True,
    eval_mode: bool = True,
) -> AutoModelForCausalLM:
    """Load model with consistent configuration.

    Args:
        model_name: HuggingFace model identifier
        device: Target device ("cuda", "cpu", or None for auto-detect)
        use_safetensors: Whether to prefer safetensors format
        eval_mode: Whether to set model to eval() mode

    Returns:
        Configured AutoModelForCausalLM instance
    """
    logger.debug(f"Loading model: {model_name}")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Auto-detected device: {device}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=use_safetensors)

    # Move to device
    model = model.to(device)

    # Set eval mode if requested
    if eval_mode:
        model.eval()
        logger.debug("Model set to eval mode")

    return model
