"""FlashAttention 4 integration via HF Transformers AttentionInterface.

Registers FA4 as a native attention backend ('flash_attention_4') that HF's
Qwen3 attention module dispatches to directly, bypassing the FA2 wrapper code.

FA4 (flash-attn-4) provides Blackwell-native (SM100) attention kernels using
CuTeDSL, UMMA instructions, and TMEM for up to 71% GPU utilization on B200.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_fa4_dispatch_logged = False


def register_fa4_attention() -> None:
    """Register FlashAttention 4 as 'flash_attention_4' in HF's attention dispatch.

    Must be called BEFORE model loading so that the model's attention layers
    dispatch to our handler at forward time.

    Tensors arrive as ``[B, H, S, D]`` from HF and are transposed to
    ``[B, S, H, D]`` for FA4. The output stays ``[B, S, H, D]`` (no transpose
    back), matching HF's convention where the caller reshapes via
    ``attn_output.reshape(*input_shape, -1)`` with ``input_shape = [B, S]``.
    """
    from flash_attn.cute import flash_attn_func as fa4_func  # type: ignore[import-not-found]
    from flash_attn.cute import (  # type: ignore[import-not-found]
        flash_attn_varlen_func as fa4_varlen,
    )
    from transformers.modeling_flash_attention_utils import (
        prepare_fa_kwargs_from_position_ids,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def _fa4_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        fa4_kwargs: dict[str, Any] = {"causal": True, "softmax_scale": scaling}
        if softcap is not None and softcap > 0.0:
            fa4_kwargs["softcap"] = softcap

        cu_seqlens_q, cu_seqlens_k = _detect_packing(
            kwargs, query.shape[0], prepare_fa_kwargs_from_position_ids
        )

        _log_first_dispatch(query, attention_mask, cu_seqlens_q, kwargs)

        if cu_seqlens_q is not None:
            out = _varlen_path(
                query, key, value, cu_seqlens_q, cu_seqlens_k, fa4_varlen, fa4_kwargs
            )
        elif attention_mask is not None:
            out = _padded_path(query, key, value, attention_mask, fa4_varlen, fa4_kwargs)
        else:
            out, _ = fa4_func(query, key, value, **fa4_kwargs)

        return out, None

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", _fa4_forward)
    logger.info("Registered flash_attention_4 via AttentionInterface")


def _detect_packing(
    kwargs: dict[str, Any],
    batch_size: int,
    prepare_fn: Any,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Detect packed sequences from kwargs, returning cu_seqlens or (None, None)."""
    cu_seqlens_q = kwargs.get("cu_seq_lens_q")
    cu_seqlens_k = kwargs.get("cu_seq_lens_k")

    if cu_seqlens_q is not None:
        return cu_seqlens_q, cu_seqlens_k

    position_ids = kwargs.get("position_ids")
    if position_ids is not None and batch_size == 1:
        try:
            (cu_seqlens_q, cu_seqlens_k), _ = prepare_fn(position_ids)
            return cu_seqlens_q, cu_seqlens_k
        except Exception:  # noqa: BLE001
            logger.debug("Failed to detect packing from position_ids, using dense path")

    return None, None


def _log_first_dispatch(
    query: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> None:
    """Log the first FA4 dispatch for debugging (once per process)."""
    global _fa4_dispatch_logged  # noqa: PLW0603
    if _fa4_dispatch_logged:
        return
    _fa4_dispatch_logged = True

    position_ids = kwargs.get("position_ids")
    logger.info(
        "FA4 dispatch: query=%s mask=%s cu_seqlens=%s pos_ids=%s",
        list(query.shape),
        list(attention_mask.shape) if attention_mask is not None else None,
        cu_seqlens_q is not None,
        list(position_ids.shape) if position_ids is not None else None,
    )


def _varlen_path(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None,
    fa4_varlen: Any,
    fa4_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Packed/variable-length attention via flash_attn_varlen_func."""
    B, S, H, D = query.shape
    out, _ = fa4_varlen(
        query.reshape(-1, H, D),
        key.reshape(-1, key.shape[2], D),
        value.reshape(-1, value.shape[2], D),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        **fa4_kwargs,
    )
    return out.view(B, S, H, D)


def _padded_path(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    fa4_varlen: Any,
    fa4_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Padded attention: unpad real tokens, run varlen, repad."""
    from flash_attn.bert_padding import pad_input  # type: ignore[import-not-found]

    batch_size = attention_mask.shape[0]
    query_length = query.shape[1]

    q, k, v, indices_q, cu_q, cu_k = _unpad_qkv(
        query, key, value, attention_mask, query_length, batch_size
    )
    out_unpad, _ = fa4_varlen(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, **fa4_kwargs)
    return pad_input(out_unpad, indices_q, batch_size, query_length)


def _unpad_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove padding from Q/K/V using attention_mask, return varlen-ready tensors."""
    indices_k = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    cu_seqlens_k = torch.nn.functional.pad(
        attention_mask.sum(dim=1, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        (1, 0),
    )
    seq_len = attention_mask.shape[1]

    if query_length == seq_len:
        indices_q = indices_k
        cu_seqlens_q = cu_seqlens_k
        q = query.reshape(-1, query.shape[-2], query.shape[-1])[indices_k]
    elif query_length == 1:
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)
        indices_q = torch.arange(batch_size, dtype=torch.long, device=query.device)
        q = query.squeeze(1)
    else:
        attention_mask_q = attention_mask[:, -query_length:]
        indices_q = attention_mask_q.flatten().nonzero(as_tuple=False).flatten()
        cu_seqlens_q = torch.nn.functional.pad(
            attention_mask_q.sum(dim=1, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            (1, 0),
        )
        q = query.reshape(-1, query.shape[-2], query.shape[-1])[indices_q]

    k = key.reshape(-1, key.shape[-2], key.shape[-1])[indices_k]
    v = value.reshape(-1, value.shape[-2], value.shape[-1])[indices_k]

    return q, k, v, indices_q, cu_seqlens_q, cu_seqlens_k
