"""GRPO (Group Relative Policy Optimization) utilities."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute sum log-probabilities over completion tokens for GRPO."""

    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

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
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    completion_lengths: list[int],
) -> torch.Tensor:
    """Compute mean entropy over completion tokens."""

    with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
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
    model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    groups: list[GRPOGroup],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
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

        for rollout in batch_rollouts:
            tokens = rollout.tokens[:TRAINER_MAX_LENGTH]
            batch_tokens.append(tokens)
            batch_prompt_lens.append(rollout.prompt_length)
            batch_comp_lens.append(rollout.completion_length)
            batch_advantages.append(rollout.advantage)

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

        logprobs_current = compute_logprobs(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
        )

        with torch.no_grad():
            logprobs_ref = compute_logprobs(
                ref_model,
                input_ids_tensor,
                attention_mask_tensor,
                batch_prompt_lens,
                batch_comp_lens,
            )

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

        entropies = compute_entropy(
            model,
            input_ids_tensor,
            attention_mask_tensor,
            batch_prompt_lens,
            batch_comp_lens,
        )
        loss_entropy = -TRAINER_ENTROPY_COEF * entropies.mean()

        loss_total = loss_pg + loss_kl + loss_entropy

        optimizer.zero_grad()
        scaler.scale(loss_total).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINER_GRAD_CLIP)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning("NaN/Inf gradient norm detected; skipping batch")
            continue

        scaler.step(optimizer)
        scaler.update()

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

    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in epoch_metrics.items()
    }
