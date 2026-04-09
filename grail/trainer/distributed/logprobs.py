"""TP-aware chunked log-probability computation for GRPO training.

When the lm_head is tensor-parallel (ColwiseParallel with Shard(-1) output),
each TP rank holds ``vocab_size / tp_size`` logits.  This module computes
per-token log-probabilities without all-gathering the full vocab logits by
using a distributed logsumexp reduction (the Megatron pattern):

Communication per chunk: 3 all-reduces of shape ``[B, chunk]`` (tiny),
instead of a 1.86 GB vocab all-gather.

Uses ``torch.distributed.nn.functional.all_reduce`` (the differentiable
version) so that gradients flow correctly through the softmax normalisation
during the current-policy forward pass.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    pass


def _differentiable_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,  # type: ignore[assignment]
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable all-reduce that participates in autograd.

    Uses ``torch.distributed.nn.functional.all_reduce`` which registers
    autograd hooks, ensuring correct gradient flow through softmax
    normalisation in the current-policy forward pass.
    """
    import torch.distributed.nn.functional as dist_fn

    result = dist_fn.all_reduce(tensor, op=op, group=group)
    # dist_fn.all_reduce returns a tuple; extract the tensor
    if isinstance(result, tuple):
        return result[0]
    return result  # type: ignore[return-value]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: TP-aware chunked logprobs via distributed logsumexp
# ---------------------------------------------------------------------------


def tp_chunked_logprobs(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    tp_group: dist.ProcessGroup,
    return_entropy: bool = False,
    prompt_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-token log-probs when lm_head output is TP-sharded.

    Each TP rank holds a shard of the vocabulary logits (``V / tp_size``
    columns).  Instead of all-gathering the full ``[B, chunk, V]`` tensor we
    perform three lightweight all-reduces per chunk (max, sum-exp, target
    logit) to reconstruct exact log-probabilities.

    Args:
        lm_head: Language model head (linear projection to vocab).  Under TP
            its output is ``[B, S, V_local]`` where ``V_local = V / tp_size``
            (possibly with a remainder on the last rank).
        hidden_states: ``[B, S, H]`` hidden states (already shifted if the
            caller handled next-token alignment).
        labels: ``[B, S]`` global token IDs aligned with *hidden_states*.
        chunk_size: Number of sequence positions per LM-head chunk.
        tp_group: The process group spanning the TP ranks.
        return_entropy: Whether to also compute per-token entropy.
        prompt_lengths: If provided, prompt positions are zeroed out in the
            returned tensors (optional convenience; the caller can also mask
            afterwards).

    Returns:
        token_log_probs: ``[B, S]`` per-token log-probabilities.
        entropy_per_token: ``[B, S]`` per-token entropy, or ``None``.
    """
    batch_size, seq_len = labels.shape
    device = hidden_states.device

    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)

    token_log_probs = torch.zeros(batch_size, seq_len, device=device)
    entropy_per_token: torch.Tensor | None = None
    if return_entropy:
        entropy_per_token = torch.zeros(batch_size, seq_len, device=device)

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)

        chunk_hidden = hidden_states[:, chunk_start:chunk_end, :]
        chunk_labels = labels[:, chunk_start:chunk_end]  # [B, C] global IDs

        # (a) Project through lm_head -> local logits [B, C, V_local]
        chunk_logits = lm_head(chunk_hidden)

        # If the output is a DTensor (torch.distributed.tensor), unwrap.
        # Use the public to_local() API when available, fall back to _local_tensor.
        if hasattr(chunk_logits, "to_local"):
            chunk_logits = chunk_logits.to_local()
        elif hasattr(chunk_logits, "_local_tensor"):
            chunk_logits = chunk_logits._local_tensor

        vocab_local = chunk_logits.shape[-1]

        # (b) Cast to fp32 for numerically stable logsumexp
        chunk_logits = chunk_logits.float()  # [B, C, V_local]

        # (c) Local max, then differentiable all-reduce MAX across TP group.
        # We use torch.distributed.nn.functional.all_reduce which participates
        # in autograd, ensuring correct gradients through the softmax
        # normalisation for the current policy forward pass.
        local_max, _ = chunk_logits.max(dim=-1)  # [B, C]
        global_max = _differentiable_all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)

        # (d) Shift logits, compute local exp sum, differentiable all-reduce SUM
        shifted_logits = chunk_logits - global_max.unsqueeze(-1)  # [B, C, V_local]
        local_exp_sum = shifted_logits.exp().sum(dim=-1)  # [B, C]
        global_exp_sum = _differentiable_all_reduce(
            local_exp_sum, op=dist.ReduceOp.SUM, group=tp_group
        )

        # (e) Global logsumexp
        global_logsumexp = global_max + global_exp_sum.log()  # [B, C]

        # (f) Extract target token's logit.
        # Labels are global vocab indices. DTensor Shard(-1) may distribute
        # unevenly: the first (full_vocab % tp_size) ranks get ceil(V/tp),
        # the rest get floor(V/tp). Compute correct partition boundaries.
        # We cannot assume all ranks have the same vocab_local. Compute
        # this rank's partition start by summing sizes of preceding ranks.
        # Since DTensor Shard distributes as: first (V%N) ranks get (V//N)+1,
        # rest get V//N, and vocab_local on this rank is already correct,
        # we use a simple offset formula.
        # For rank r with tp_size N: partition_start = sum_{i=0}^{r-1} vocab_local_i
        # With DTensor Shard(0): rank i gets (V//N + (1 if i < V%N else 0))
        # We don't know full_vocab here, but we can reconstruct it:
        # Gather vocab_local from all ranks.
        local_vocab_tensor = torch.tensor([vocab_local], device=device, dtype=torch.long)
        all_vocab_sizes = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(tp_size)]
        dist.all_gather(all_vocab_sizes, local_vocab_tensor, group=tp_group)
        partition_start = sum(v.item() for v in all_vocab_sizes[:tp_rank])
        partition_end = partition_start + vocab_local

        # Map global label to local index; out-of-range labels get index 0
        # (the gathered value will be zeroed out below).
        local_label = chunk_labels - partition_start  # [B, C]
        in_partition = (chunk_labels >= partition_start) & (chunk_labels < partition_end)
        # Clamp to valid range so gather doesn't fail
        local_label = local_label.clamp(0, vocab_local - 1)

        # Gather the logit at the local index
        target_logit = chunk_logits.gather(2, local_label.unsqueeze(-1)).squeeze(-1)  # [B, C]

        # Zero out contributions from ranks that don't own this label
        target_logit = target_logit * in_partition.float()

        # (g) Differentiable all-reduce SUM (exactly one rank has the real value)
        target_logit = _differentiable_all_reduce(
            target_logit, op=dist.ReduceOp.SUM, group=tp_group
        )

        # (h) logprob = target_logit - global_logsumexp
        token_log_probs[:, chunk_start:chunk_end] = target_logit - global_logsumexp

        # (i) Entropy: -sum(softmax * logits) across full vocab.
        # Each rank computes its local piece, then all-reduce SUM.
        if return_entropy and entropy_per_token is not None:
            softmax_local = shifted_logits.exp() / global_exp_sum.unsqueeze(-1)  # [B, C, V_local]
            log_probs_local = shifted_logits - global_exp_sum.log().unsqueeze(-1)
            local_entropy = -(softmax_local * log_probs_local).sum(dim=-1)  # [B, C]
            local_entropy = _differentiable_all_reduce(
                local_entropy, op=dist.ReduceOp.SUM, group=tp_group
            )
            entropy_per_token[:, chunk_start:chunk_end] = local_entropy
            del softmax_local, log_probs_local, local_entropy

        del chunk_logits, shifted_logits, local_exp_sum, target_logit
        del local_max, global_max, global_exp_sum, global_logsumexp

    # Optionally zero out prompt positions
    if prompt_lengths is not None:
        for idx, plen in enumerate(prompt_lengths):
            if plen > 0:
                token_log_probs[idx, :plen] = 0.0
                if entropy_per_token is not None:
                    entropy_per_token[idx, :plen] = 0.0

    return token_log_probs, entropy_per_token


# ---------------------------------------------------------------------------
# Wrapper: dispatch between TP and non-TP paths
# ---------------------------------------------------------------------------


def _unwrap_for_chunked_logits(
    model: Any,
) -> tuple[Any, Any] | None:
    """Extract ``(base_model, lm_head)`` from a causal LM.

    Returns ``None`` if the model structure is incompatible (e.g. DDP/FSDP
    wrappers or non-standard architectures).
    """
    try:
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            return None
    except ImportError:
        pass
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if isinstance(model, FullyShardedDataParallel):
            return None
    except ImportError:
        pass

    base_model_prefix = getattr(model, "base_model_prefix", "")
    base = getattr(model, base_model_prefix, None) if base_model_prefix else None
    lm_head = getattr(model, "lm_head", None)

    if base is not None and lm_head is not None:
        return base, lm_head

    return None


def compute_logprobs_distributed(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    *,
    chunked: bool = True,
    chunk_size: int = 256,
    tp_group: dist.ProcessGroup | None = None,
    return_per_token: bool = True,
    return_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Compute completion-token log-probs, dispatching to TP or standard path.

    When *tp_group* is provided, uses :func:`tp_chunked_logprobs` to avoid
    the costly all-gather of full-vocab logits.  Otherwise, falls back to the
    standard chunked implementation from ``grpo.py``.

    Returns:
        A 3-tuple ``(logprobs_sum, logprobs_per_token, entropies)`` matching
        the interface expected by ``GRPOAlgorithm``:

        * ``logprobs_sum`` -- ``[B]`` sum of log-probs per sequence.
        * ``logprobs_per_token`` -- ``[B, max_comp_len]`` padded per-token
          log-probs if *return_per_token* is ``True``, else ``None``.
        * ``entropies`` -- ``[B]`` mean entropy per sequence if
          *return_entropy* is ``True``, else ``None``.
    """
    # ---- TP path: distributed logsumexp without full-vocab all-gather ----
    if tp_group is not None:
        parts = _unwrap_for_chunked_logits(model)
        if parts is None:
            raise RuntimeError(
                "Cannot extract (base_model, lm_head) from model for TP logprobs. "
                "Ensure the model is a standard HuggingFace CausalLM (not DDP/FSDP wrapped)."
            )
        base_model, lm_head = parts

        # Forward through the base model to get hidden states
        base_out = base_model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = base_out.last_hidden_state  # [B, S, H]

        # Apply the same prompt-skip optimisation as the non-TP path:
        # shift for next-token prediction, skipping prompt-only positions.
        min_prompt_len = min(prompt_lengths)
        keep_from = max(0, min_prompt_len - 1)

        shift_hidden = hidden_states[:, keep_from:-1, :]
        shift_labels = input_ids[:, keep_from + 1 :]

        token_log_probs, entropy_per_token = tp_chunked_logprobs(
            lm_head,
            shift_hidden,
            shift_labels,
            chunk_size,
            tp_group,
            return_entropy=return_entropy,
        )

        # Extract completion-token results, adjusting for the keep_from offset
        adjusted_prompt_lengths = [p - keep_from for p in prompt_lengths]
        return _extract_completion_logprobs_tp(
            token_log_probs,
            entropy_per_token,
            adjusted_prompt_lengths,
            completion_lengths,
            return_per_token,
            return_entropy,
        )

    # ---- Non-TP path: delegate to existing implementation ----
    from grail.trainer.algorithms.grpo import compute_logprobs

    result = compute_logprobs(
        model,
        input_ids,
        attention_mask if attention_mask is not None else torch.ones_like(input_ids),
        prompt_lengths,
        completion_lengths,
        return_per_token=return_per_token,
        return_entropy=return_entropy,
        chunked=chunked,
        chunk_size=chunk_size,
    )

    # Normalise return format to a consistent 3-tuple
    if return_per_token and return_entropy:
        if isinstance(result, tuple) and len(result) == 3:
            return result[0], result[1], result[2]
    if return_per_token and not return_entropy:
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], None
    # return_per_token=False (sum only)
    if isinstance(result, torch.Tensor):
        return result, None, None
    # Fallback for unexpected shapes
    if isinstance(result, tuple):
        logprobs_sum = result[0]
        logprobs_pt = result[1] if len(result) > 1 else None
        entropies = result[2] if len(result) > 2 else None
        return logprobs_sum, logprobs_pt, entropies

    return result, None, None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Completion extraction (mirrors grpo._extract_completion_logprobs)
# ---------------------------------------------------------------------------


def _extract_completion_logprobs_tp(
    token_log_probs: torch.Tensor,
    entropy_per_token: torch.Tensor | None,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    return_per_token: bool,
    return_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract completion-token logprobs/entropy and return a consistent 3-tuple.

    This mirrors ``grpo._extract_completion_logprobs`` but always returns a
    3-tuple ``(logprobs_sum, logprobs_per_token | None, entropies | None)``
    for a uniform interface.

    Args:
        token_log_probs: ``[B, shifted_seq_len]`` per-token log-probs.
        entropy_per_token: ``[B, shifted_seq_len]`` per-token entropy (or ``None``).
        prompt_lengths: Per-sample prompt lengths (already adjusted for any
            ``keep_from`` offset).
        completion_lengths: Per-sample completion lengths.
        return_per_token: Whether to return padded per-token logprobs.
        return_entropy: Whether to return mean entropy per sequence.
    """
    seq_len_minus_1 = token_log_probs.shape[1]
    device = token_log_probs.device
    batch_size = len(prompt_lengths)

    max_comp_len = max(completion_lengths) if completion_lengths else 1
    seq_log_probs = torch.zeros(batch_size, device=device)
    per_token_padded: torch.Tensor | None = None
    if return_per_token:
        per_token_padded = torch.zeros(batch_size, max_comp_len, device=device)

    entropies_list: list[torch.Tensor] = []

    for idx, prompt_len in enumerate(prompt_lengths):
        completion_len = completion_lengths[idx]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(seq_len_minus_1, start_idx + completion_len)
        if end_idx > start_idx:
            comp_logps = token_log_probs[idx, start_idx:end_idx]
            seq_log_probs[idx] = comp_logps.sum()
            if per_token_padded is not None:
                per_token_padded[idx, : len(comp_logps)] = comp_logps
            if return_entropy and entropy_per_token is not None:
                entropies_list.append(entropy_per_token[idx, start_idx:end_idx].mean())
        else:
            if return_entropy and entropy_per_token is not None:
                entropies_list.append(torch.tensor(0.0, device=device))

    entropies: torch.Tensor | None = None
    if return_entropy and entropy_per_token is not None:
        entropies = torch.stack(entropies_list) if entropies_list else torch.zeros(0, device=device)

    return seq_log_probs, per_token_padded, entropies
