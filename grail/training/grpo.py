"""GRPO (Group Relative Policy Optimization) utilities."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from ..shared.constants import (
    TRAINER_BATCH_SIZE,
    TRAINER_ENTROPY_COEF,
    TRAINER_GRAD_CLIP,
    TRAINER_KL_COEF,
    TRAINER_MAX_LENGTH,
)
from .data import GRPOGroup

logger = logging.getLogger(__name__)


def compute_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute sum log-probabilities over completion tokens for GRPO."""

    # Precision (fp16/bf16) is controlled by the caller via accelerator.autocast
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2,
        shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    seq_log_probs: list[torch.Tensor] = []
    seq_len_minus_1 = token_log_probs.shape[1]
    for idx, prompt_len in enumerate(prompt_lengths):
        completion_len = completion_lengths[idx]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(seq_len_minus_1, start_idx + completion_len)
        if end_idx > start_idx:
            seq_log_probs.append(token_log_probs[idx, start_idx:end_idx].sum())
        else:
            seq_log_probs.append(torch.tensor(0.0, device=token_log_probs.device))

    return torch.stack(seq_log_probs)


def compute_entropy(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute mean entropy over completion tokens."""

    # Precision is controlled by the caller via accelerator.autocast
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)

    entropies: list[torch.Tensor] = []
    seq_len_minus_1 = entropy_per_token.shape[1]
    for idx, prompt_len in enumerate(prompt_lengths):
        completion_len = completion_lengths[idx]
        start_idx = max(0, prompt_len - 1)
        end_idx = min(seq_len_minus_1, start_idx + completion_len)
        if end_idx > start_idx:
            entropies.append(entropy_per_token[idx, start_idx:end_idx].mean())
        else:
            entropies.append(torch.tensor(0.0, device=entropy_per_token.device))

    return torch.stack(entropies)


async def train_grpo_epoch(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    groups: list[GRPOGroup],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    monitor: Any,
    window: int,
) -> dict[str, float]:
    """Run a single GRPO training epoch and return aggregated metrics."""

    model.train()
    ref_model.eval()

    all_rollouts: list[tuple] = []
    for group in groups:
        for rollout in group.rollouts:
            all_rollouts.append((rollout, group.group_id))

    all_rollouts.sort(key=lambda item: (item[1], item[0].nonce))

    batch_size = TRAINER_BATCH_SIZE
    num_batches = math.ceil(len(all_rollouts) / batch_size)

    epoch_metrics: dict[str, list[float]] = defaultdict(list)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_rollouts))
        batch_rollouts = [all_rollouts[i][0] for i in range(start_idx, end_idx)]

        batch_tokens: list[list[int]] = []
        batch_prompt_lens: list[int] = []
        batch_comp_lens: list[int] = []
        batch_advantages: list[float] = []
        batch_behavior_seq_logprobs: list[float | None] = []

        for rollout in batch_rollouts:
            tokens = rollout.tokens[:TRAINER_MAX_LENGTH]
            batch_tokens.append(tokens)
            batch_prompt_lens.append(rollout.prompt_length)
            batch_comp_lens.append(rollout.completion_length)
            batch_advantages.append(rollout.advantage)
            # If miner provided per-token logprobs, aggregate over completion
            provided = None
            if getattr(rollout, "token_logprobs", None):
                tlp: list[float] = list(rollout.token_logprobs or [])
                # Two supported shapes: per-completion only, or per-shifted-seq
                if len(tlp) == max(0, rollout.completion_length):
                    provided = float(sum(tlp))
                else:
                    start_idx = max(0, rollout.prompt_length - 1)
                    end_idx = min(len(tlp), start_idx + rollout.completion_length)
                    if end_idx > start_idx:
                        provided = float(sum(tlp[start_idx:end_idx]))
            batch_behavior_seq_logprobs.append(provided)

        max_len = max(len(tokens) for tokens in batch_tokens)
        pad_id = tokenizer.pad_token_id
        input_ids = []
        attention_masks = []
        for tokens in batch_tokens:
            pad_length = max_len - len(tokens)
            input_ids.append(tokens + [pad_id] * pad_length)
            attention_masks.append([1] * len(tokens) + [0] * pad_length)

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=accelerator.device)
        attention_mask_tensor = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=accelerator.device,
        )
        advantages_tensor = torch.tensor(
            batch_advantages,
            dtype=torch.float32,
            device=accelerator.device,
        )

        # Forward pass under autocast for mixed precision
        with accelerator.autocast():
            logprobs_current = compute_logprobs(
                model,
                input_ids_tensor,
                attention_mask_tensor,
                batch_prompt_lens,
                batch_comp_lens,
            )

        # Behavior/reference logprobs: prefer miner-provided values when present
        all_have_behavior = all(x is not None for x in batch_behavior_seq_logprobs)
        any_have_behavior = any(x is not None for x in batch_behavior_seq_logprobs)
        if all_have_behavior:
            logprobs_ref = torch.tensor(
                list(batch_behavior_seq_logprobs),
                dtype=torch.float32,
                device=accelerator.device,
            )
        else:
            with torch.no_grad():
                with accelerator.autocast():
                    logprobs_ref = compute_logprobs(
                        ref_model,
                        input_ids_tensor,
                        attention_mask_tensor,
                        batch_prompt_lens,
                        batch_comp_lens,
                    )
            if any_have_behavior:
                provided_vals = [(x if x is not None else 0.0) for x in batch_behavior_seq_logprobs]
                provided_tensor = torch.tensor(
                    provided_vals,
                    dtype=logprobs_ref.dtype,
                    device=accelerator.device,
                )
                mask_tensor = torch.tensor(
                    [x is not None for x in batch_behavior_seq_logprobs],
                    dtype=torch.bool,
                    device=accelerator.device,
                )
                logprobs_ref = torch.where(mask_tensor, provided_tensor, logprobs_ref)

        # Use log-probability ratio for a symmetric KL-like penalty
        log_ratio = logprobs_current - logprobs_ref

        if advantages_tensor.std() > 1e-8:
            advantages_normalized = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std() + 1e-8
            )
        else:
            advantages_normalized = advantages_tensor

        loss_pg = -(advantages_normalized * logprobs_current).mean()
        loss_kl = TRAINER_KL_COEF * log_ratio.pow(2).mean()

        with accelerator.autocast():
            entropies = compute_entropy(
                model,
                input_ids_tensor,
                attention_mask_tensor,
                batch_prompt_lens,
                batch_comp_lens,
            )
        loss_entropy = -TRAINER_ENTROPY_COEF * entropies.mean()

        loss_total = loss_pg + loss_kl + loss_entropy

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss_total)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINER_GRAD_CLIP)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning("NaN/Inf gradient norm detected; skipping batch")
            continue

        optimizer.step()

        epoch_metrics["loss_total"].append(loss_total.item())
        epoch_metrics["loss_pg"].append(loss_pg.item())
        epoch_metrics["loss_kl"].append(loss_kl.item())
        epoch_metrics["loss_entropy"].append(loss_entropy.item())
        epoch_metrics["grad_norm"].append(grad_norm.item())
        epoch_metrics["advantage_mean"].append(advantages_tensor.mean().item())
        epoch_metrics["advantage_std"].append(advantages_tensor.std().item())
        epoch_metrics["entropy_mean"].append(entropies.mean().item())
        epoch_metrics["advantage_mean_normalized"].append(advantages_normalized.mean().item())
        epoch_metrics["advantage_std_normalized"].append(advantages_normalized.std().item())
        epoch_metrics["kl_divergence"].append(log_ratio.pow(2).mean().item())
        if any_have_behavior:
            frac = float(sum(1 for x in batch_behavior_seq_logprobs if x is not None))
            epoch_metrics["behavior_frac"].append(frac / max(1, len(batch_behavior_seq_logprobs)))

    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in epoch_metrics.items()
    }
