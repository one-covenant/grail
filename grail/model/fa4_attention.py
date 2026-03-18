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


def register_fa4_attention() -> None:
    """Register FlashAttention 4 as 'flash_attention_4' in HF's attention dispatch.

    Must be called BEFORE model loading so that
    ``AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_4")``
    resolves to our handler.

    The handler accepts the same interface as HF's built-in attention functions:
    ``(module, query, key, value, attention_mask, **kwargs) -> (output, None)``

    Tensors arrive as ``[B, H, S, D]`` from HF and must be transposed to
    ``[B, S, H, D]`` for FA4's API. The output is returned as ``[B, S, H, D]``
    (no transpose back), matching HF's convention: the caller reshapes via
    ``attn_output.reshape(*input_shape, -1)`` where ``input_shape = [B, S]``.
    """
    from flash_attn.cute import flash_attn_func as _fa4_func
    from flash_attn.cute import flash_attn_varlen_func as _fa4_varlen
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
        # HF attention modules always pass [B, H, S, D]. FA4 expects [B, S, H, D].
        # The caller (Qwen3Attention.forward) expects the output back as [B, S, H, D]
        # so it can reshape to [B, S, H*D] via .reshape(*input_shape, -1).
        # We therefore transpose on input and do NOT transpose back on output,
        # matching the convention used by HF's own flash_attention_forward handler.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        fa4_kwargs: dict[str, Any] = {
            "causal": True,
            "softmax_scale": scaling,
        }
        if softcap is not None and softcap > 0.0:
            fa4_kwargs["softcap"] = softcap

        # Detect packed sequences. HF may pass cu_seqlens directly in kwargs, or
        # we derive them from position_ids (resets to 0 at sequence boundaries).
        cu_seqlens_q = kwargs.get("cu_seq_lens_q")
        cu_seqlens_k = kwargs.get("cu_seq_lens_k")
        position_ids = kwargs.get("position_ids")

        # If no cu_seqlens but position_ids indicate packing (B=1, non-monotonic),
        # compute cu_seqlens from position_ids resets.
        if cu_seqlens_q is None and position_ids is not None and query.shape[0] == 1:
            from transformers.modeling_flash_attention_utils import (
                prepare_fa_kwargs_from_position_ids,
            )

            try:
                (cu_seqlens_q, cu_seqlens_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
                    position_ids
                )
            except Exception:  # noqa: BLE001
                pass  # Not packed, fall through to dense path

        # Debug: log which path is taken (first call only)
        if not hasattr(_fa4_forward, "_logged"):
            _fa4_forward._logged = True
            logger.info(
                "FA4 dispatch: query=%s mask=%s cu_seqlens=%s pos_ids=%s",
                list(query.shape),
                list(attention_mask.shape) if attention_mask is not None else None,
                cu_seqlens_q is not None,
                list(position_ids.shape) if position_ids is not None else None,
            )

        if cu_seqlens_q is not None:
            # Packed/varlen path
            B, S, H, D = query.shape
            out, _ = _fa4_varlen(
                query.reshape(-1, H, D),
                key.reshape(-1, key.shape[2], D),
                value.reshape(-1, value.shape[2], D),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                **fa4_kwargs,
            )
            out = out.view(B, S, H, D)
        elif attention_mask is not None:
            # Padded path: unpad -> varlen -> repad
            from flash_attn.bert_padding import pad_input, unpad_input

            query_length = query.shape[1]
            q, k, v, indices_q, (cu_q, cu_k), (max_q, max_k) = _upad_input(
                query, key, value, attention_mask, query_length, unpad_input
            )
            out_unpad, _ = _fa4_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                **fa4_kwargs,
            )
            out = pad_input(out_unpad, indices_q, query.shape[0], query_length)
        else:
            # Dense path: no padding, no packing
            out, _ = _fa4_func(query, key, value, **fa4_kwargs)

        return out, None

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", _fa4_forward)
    logger.info("Registered flash_attention_4 via AttentionInterface")


def _upad_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    unpad_fn: Any,
) -> tuple:
    """Unpad tensors using attention_mask, returning varlen-compatible format."""
    batch_size, seq_len = attention_mask.shape
    indices_k = attention_mask.flatten().nonzero(as_tuple=False).flatten()
    cu_seqlens_k = torch.nn.functional.pad(
        attention_mask.sum(dim=1, dtype=torch.int32).cumsum(0, dtype=torch.int32),
        (1, 0),
    )
    max_seqlen_k = attention_mask.sum(dim=1).max().item()

    if query_length == seq_len:
        q = query.reshape(-1, query.shape[-2], query.shape[-1])[indices_k]
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
        indices_q = indices_k
    elif query_length == 1:
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)
        max_seqlen_q = 1
        indices_q = torch.arange(batch_size, dtype=torch.long, device=query.device)
        q = query.squeeze(1)
    else:
        attention_mask_q = attention_mask[:, -query_length:]
        indices_q = attention_mask_q.flatten().nonzero(as_tuple=False).flatten()
        cu_seqlens_q = torch.nn.functional.pad(
            attention_mask_q.sum(dim=1, dtype=torch.int32).cumsum(0, dtype=torch.int32),
            (1, 0),
        )
        max_seqlen_q = attention_mask_q.sum(dim=1).max().item()
        q = query.reshape(-1, query.shape[-2], query.shape[-1])[indices_q]

    k = key.reshape(-1, key.shape[-2], key.shape[-1])[indices_k]
    v = value.reshape(-1, value.shape[-2], value.shape[-1])[indices_k]

    return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)
