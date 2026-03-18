"""Model and tokenizer provider for GRAIL.

Centralized loading functions to ensure consistent configuration across
all components (Prover, Verifier, Trainer).
"""

from __future__ import annotations

import gc
import json
import logging
import os
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
    checkpoint_window: int | None = None,
) -> Any:
    """Load model with consistent configuration.

    Args:
        model_name: HuggingFace model identifier or local checkpoint path
        device: Target device (e.g., "cuda", "cuda:0", "cpu", or None for auto-detect)
        use_safetensors: Whether to prefer safetensors format
        eval_mode: Whether to set model to eval() mode
        checkpoint_window: Optional checkpoint window number. If not provided, will be
                          extracted from metadata.json or parsed from the path.

    Returns:
        Configured model instance with preserved original name and checkpoint_window attribute
    """
    logger.debug(f"Loading model: {model_name}")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Auto-detected device: {device}")

    device_is_cuda = str(device).startswith("cuda")

    # Check if this is a local checkpoint path with metadata
    original_model_name = model_name
    resolved_checkpoint_window = checkpoint_window
    model_path = Path(model_name)
    if model_path.exists() and model_path.is_dir():
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                original_model_name = metadata.get("model_name", model_name)
                # Extract checkpoint_window from metadata if not explicitly provided
                if resolved_checkpoint_window is None and "window" in metadata:
                    resolved_checkpoint_window = int(metadata["window"])
                logger.debug(
                    f"Found checkpoint: {original_model_name}, window={resolved_checkpoint_window}"
                )
            except Exception as e:
                logger.debug(f"Failed to read checkpoint metadata: {e}")

        # Fallback: parse checkpoint-{window} from path if still not set
        if resolved_checkpoint_window is None and "checkpoint-" in model_name:
            try:
                checkpoint_segment = model_name.split("checkpoint-")[-1].split("/")[0]
                resolved_checkpoint_window = int(checkpoint_segment)
                logger.debug(f"Parsed checkpoint window from path: {resolved_checkpoint_window}")
            except (ValueError, IndexError):
                pass

    # Configure attention implementation from protocol constant.
    # On CUDA: use ATTN_IMPLEMENTATION (FA2), fail loudly if flash-attn is missing.
    # On CPU (tests, dev): use default SDPA (no proof computation happens on CPU).
    # GRAIL_TRAINER_ATTN_IMPL overrides for training only (e.g., "sdpa" for torch.compile).
    from ..shared.constants import ATTN_IMPLEMENTATION

    attn_implementation = None
    trainer_attn_override = os.getenv("GRAIL_TRAINER_ATTN_IMPL")
    if trainer_attn_override:
        if trainer_attn_override == "flash_attention_4":
            # Load with FA2 (HF's import validation requires it), then swap the
            # attention handler to FA4 via AttentionInterface after model creation.
            attn_implementation = "flash_attention_2"
            try:
                from .fa4_attention import register_fa4_attention

                register_fa4_attention()
            except ImportError:
                logger.warning("flash-attn-4 not installed, FA4 will not be used")
        else:
            attn_implementation = trainer_attn_override
        logger.info("Using attention implementation: %s (trainer override)", trainer_attn_override)
    elif device_is_cuda and ATTN_IMPLEMENTATION:
        attn_implementation = ATTN_IMPLEMENTATION
        if ATTN_IMPLEMENTATION == "flash_attention_2":
            try:
                import flash_attn  # type: ignore[import-not-found]  # noqa: F401
            except ImportError as err:
                raise RuntimeError(
                    "flash-attn is required but not installed. "
                    "GRAIL requires Flash Attention 2 for consistent proof verification. "
                    "Install with: uv pip install flash-attn --no-build-isolation"
                ) from err
        logger.info("Using attention implementation: %s", attn_implementation)

    # Apply Liger kernel optimizations (fused RMSNorm, RoPE, SwiGLU, CrossEntropy)
    # Must be called BEFORE model loading as it monkey-patches the model class.
    if os.getenv("GRAIL_TRAINER_USE_LIGER_KERNEL", "0") == "1":
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3  # type: ignore[import-not-found]

            apply_liger_kernel_to_qwen3(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
            )
            logger.info("Liger kernel applied to Qwen3 (rope, rms_norm, swiglu)")
        except ImportError:
            logger.warning("GRAIL_TRAINER_USE_LIGER_KERNEL=1 but liger-kernel not installed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to apply Liger kernel: %s", exc)

    # Load model with optimized attention if available
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=use_safetensors,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16 if device_is_cuda else torch.float32,
    )

    # Swap attention dispatch to FA4 after loading (must be after from_pretrained
    # because HF validates attn_implementation during loading).
    if trainer_attn_override == "flash_attention_4":
        try:
            model.config._attn_implementation = "flash_attention_4"  # type: ignore[reportPrivateUsage]
            logger.info("Model attention dispatch set to flash_attention_4 (FA4 native)")
        except Exception:  # noqa: BLE001
            pass

    # Preserve original model name for GRAIL proof validation
    model.name_or_path = original_model_name  # type: ignore[attr-defined]

    # Store checkpoint window for validation (avoids parsing path strings)
    model.grail_checkpoint_window = resolved_checkpoint_window  # type: ignore[attr-defined]

    # Move to device
    model = model.to(device)  # type: ignore[call-overload]

    # Set eval mode if requested
    if eval_mode:
        model.eval()
        logger.debug("Model set to eval mode")

    # Log model metadata
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_dtype = model.dtype
        model_config = model.config

        logger.info(
            f"✅ Model loaded: {original_model_name} | "
            f"Params: {total_params:,} (trainable: {trainable_params:,}) | "
            f"Dtype: {model_dtype} | Device: {device}"
        )
        logger.debug(
            f"Model config: vocab_size={getattr(model_config, 'vocab_size', '?')}, "
            f"hidden_size={getattr(model_config, 'hidden_size', '?')}, "
            f"num_hidden_layers={getattr(model_config, 'num_hidden_layers', '?')}, "
            f"num_attention_heads={getattr(model_config, 'num_attention_heads', '?')}"
        )
    except Exception as e:
        logger.debug(f"Failed to log model metadata: {e}")

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
