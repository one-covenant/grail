from typing import Any, Optional, Union

from transformers import PretrainedConfig, PreTrainedModel


def resolve_hidden_size(model: PreTrainedModel) -> int:
    """Return model hidden size across HF variants (Gemma/Qwen/etc.).

    Order of attempts:
    1) config.hidden_size, d_model, n_embd, embed_dim, hidden_dim
    2) config.text_config.hidden_size
    3) input embedding dimension
    """
    cfg = getattr(model, "config", None)

    for attr in (
        "hidden_size",
        "d_model",
        "n_embd",
        "embed_dim",
        "hidden_dim",
    ):
        try:
            if cfg is not None and hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    try:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            val = text_cfg.hidden_size
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    try:
        if hasattr(model, "get_input_embeddings") and callable(model.get_input_embeddings):
            emb = model.get_input_embeddings()
            weight = getattr(emb, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
    except Exception:
        pass

    raise AttributeError("Could not determine model hidden size from config or embeddings")


def resolve_vocab_size(
    model_config: Union[PretrainedConfig, Any],
) -> Optional[int]:
    """Return vocab size if present in config (direct or nested)."""
    try:
        for attr in ("vocab_size", "n_vocab", "vocabulary_size"):
            if hasattr(model_config, attr):
                val = getattr(model_config, attr)
                if isinstance(val, int) and val > 0:
                    return val

        text_cfg = getattr(model_config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "vocab_size"):
            val = text_cfg.vocab_size
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass

    return None


def resolve_max_context_length(
    model_config: Union[PretrainedConfig, Any],
) -> int:
    """Return best-effort max context length (max sequence length) with common fallbacks."""
    candidates = (
        "max_position_embeddings",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "max_sequence_length",
        "model_max_length",
    )

    for name in candidates:
        try:
            if hasattr(model_config, name):
                val = getattr(model_config, name)
                if isinstance(val, int) and val > 0:
                    return val
        except Exception:
            pass

    return 16384
