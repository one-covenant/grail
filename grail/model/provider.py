"""Model and tokenizer provider for GRAIL.

Centralized loading functions to ensure consistent configuration across
all components (Prover, Verifier, Trainer).
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_name: str,
    *,
    chat_template: str | None = None,
) -> Any:
    """Load tokenizer with consistent configuration.

    Args:
        model_name: HuggingFace model identifier
        chat_template: Optional chat template string to install

    Returns:
        Configured AutoTokenizer instance
    """
    logger.debug(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad_token_id only if missing (avoid conflating pad/eos semantics)
    # Required for batching; fallback to eos_token_id if no dedicated pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug("Set pad_token_id to eos_token_id (model had no pad token)")

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
) -> Any:
    """Load model with consistent configuration.

    Args:
        model_name: HuggingFace model identifier or local checkpoint path
        device: Target device ("cuda", "cpu", or None for auto-detect)
        use_safetensors: Whether to prefer safetensors format
        eval_mode: Whether to set model to eval() mode

    Returns:
        Configured model instance with preserved original name
    """
    logger.debug(f"Loading model: {model_name}")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Auto-detected device: {device}")

    # Check if this is a local checkpoint path with metadata
    original_model_name = model_name
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                original_model_name = metadata.get("model_name", model_name)
                logger.debug(f"Found checkpoint: {original_model_name}")
            except Exception as e:
                logger.debug(f"Failed to read checkpoint metadata: {e}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=use_safetensors)

    # Preserve original model name for GRAIL proof validation
    if hasattr(model, "name_or_path") and original_model_name != model_name:
        model.name_or_path = original_model_name
        logger.debug(f"Preserved original model name: {original_model_name}")

    # Move to device
    model = model.to(device)

    # Set eval mode if requested
    if eval_mode:
        model.eval()
        logger.debug("Model set to eval mode")

    return model


def clear_model_and_tokenizer(model: Any | None, tokenizer: Any | None) -> tuple[None, None]:
    """Release references and aggressively reclaim GPU memory.

    Returns a pair of Nones so callers can assign:
        model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
    """
    try:
        # Drop strong refs in caller by returning (None, None).
        # Local deletions here allow earlier collection if no other refs exist.
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return None, None
