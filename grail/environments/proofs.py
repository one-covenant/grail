"""GRAIL proof computation: commitments and logprobs."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import traceback as _tb
from typing import Any

import torch

from ..model.forward import forward_single_layer
from ..shared.constants import GRAIL_PROOF_VERSION, LAYER_INDEX

logger = logging.getLogger(__name__)


def _batched_forward_pass(
    model: Any,
    device: str,
    all_token_ids_batch: list[list[int]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run sub-batched forward passes with right-padding.

    Uses ``forward_single_layer`` -- the same function the validator uses --
    so miner and validator produce identical hidden states and logits for
    GRAIL proof generation and verification.

    On OOM the sub-batch size is halved automatically (down to 1) before
    propagating the error.

    Args:
        model: HuggingFace causal LM
        device: Device string (e.g. ``"cuda:1"``)
        all_token_ids_batch: Variable-length token sequences

    Returns:
        ``(per_seq_hidden, per_seq_logits)`` -- lists of tensors, one per
        sequence.  hidden: ``[seq_len, hidden_dim]`` on *device*.
        logits: ``[seq_len, vocab]`` on CPU.
    """
    from ..shared.constants import PROOF_BATCH_SIZE

    batch_size = len(all_token_ids_batch)
    seq_lens = [len(seq) for seq in all_token_ids_batch]
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", 0)

    per_seq_hidden: list[torch.Tensor] = []
    per_seq_logits: list[torch.Tensor] = []

    sub_batch_size = PROOF_BATCH_SIZE
    pos = 0

    while pos < batch_size:
        sub_end = min(pos + sub_batch_size, batch_size)
        sub_seqs = all_token_ids_batch[pos:sub_end]
        sub_lens = seq_lens[pos:sub_end]
        sub_max = max(sub_lens)
        sub_bs = len(sub_seqs)

        try:
            # Right-pad variable-length sequences to sub-batch max length
            input_ids = torch.full((sub_bs, sub_max), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros(sub_bs, sub_max, dtype=torch.long, device=device)
            for i, (seq, slen) in enumerate(zip(sub_seqs, sub_lens, strict=True)):
                input_ids[i, :slen] = torch.tensor(seq, dtype=torch.long, device=device)
                attn_mask[i, :slen] = 1

            # Shared forward path with validator (use_cache=False, single-layer)
            with torch.inference_mode():
                h_layer, logits = forward_single_layer(model, input_ids, attn_mask, LAYER_INDEX)

            # Extract per-sequence results, trimming padding.
            # .clone() decouples from the batched tensor so del frees GPU memory.
            for i, slen in enumerate(sub_lens):
                per_seq_hidden.append(h_layer[i, :slen, :].clone())
                per_seq_logits.append(logits[i, :slen, :].detach().to("cpu"))

            del h_layer, logits, input_ids, attn_mask
            torch.cuda.empty_cache()
            pos = sub_end

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if (
                not isinstance(e, torch.cuda.OutOfMemoryError)
                and "out of memory" not in str(e).lower()
            ):
                raise
            if sub_batch_size <= 1:
                raise

            new_size = max(1, sub_batch_size // 2)
            logger.warning(
                "OOM at sub-batch %d, halving sub-batch size: %d -> %d",
                sub_batch_size,
                sub_batch_size,
                new_size,
            )
            sub_batch_size = new_size
            # Clear traceback frame locals -- they hold references to
            # intermediate GPU tensors from the failed forward pass.
            if e.__traceback__ is not None:
                _tb.clear_frames(e.__traceback__)
            del e
            gc.collect()
            torch.cuda.empty_cache()
            # Do NOT advance pos -- retry the same chunk with smaller sub-batch

    logger.info(
        "Batched forward pass: %d seqs, sub-batch %d->%d (max_len=%d, min_len=%d)",
        batch_size,
        PROOF_BATCH_SIZE,
        sub_batch_size,
        max(seq_lens),
        min(seq_lens),
    )
    return per_seq_hidden, per_seq_logits


def compute_proofs(
    model: Any,
    device: str,
    hidden_dim: int,
    all_token_ids_batch: list[list[int]],
    prompt_lens: list[int],
    randomness_hex: str,
    wallet: Any,
) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
    """Compute GRAIL commitments and logprobs.  Used by AgentEnvLoop and ProofWorker.

    Two-phase design:
      Phase 1 -- Forward passes via ``_batched_forward_pass`` (which internally
        calls ``forward_single_layer``, the same function the validator uses).
        Sub-batch size is the constant PROOF_BATCH_SIZE (16).
        Falls back to sequential single-sequence passes on OOM.
      Phase 2 -- Per-sequence commitment + logprob extraction from cached
        hidden states and logits.

    Args:
        model: The loaded PyTorch model
        device: Device string (e.g. "cuda:1")
        hidden_dim: Model hidden dimension
        all_token_ids_batch: List of full token sequences (prompt + completion)
        prompt_lens: List of prompt lengths corresponding to each sequence
        randomness_hex: Hex string for randomness beacon
        wallet: Bittensor wallet for signing commitments

    Returns:
        List of tuples: (commitments, logprobs, signature, beacon, proof_version)
        one per rollout in the batch.
    """
    batch_size = len(all_token_ids_batch)
    if batch_size == 0:
        return []

    from ..protocol.grail_verifier import GRAILVerifier

    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    # --- Phase 1: Forward passes (batched with OOM fallback) ---
    use_batched = True
    per_seq_hidden: list[torch.Tensor | None] = [None] * batch_size
    per_seq_logits: list[torch.Tensor | None] = [None] * batch_size

    try:
        hidden_list, logits_list = _batched_forward_pass(model, device, all_token_ids_batch)
        for i in range(batch_size):
            per_seq_hidden[i] = hidden_list[i]
            per_seq_logits[i] = logits_list[i]
        del hidden_list, logits_list
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        logger.warning(
            "Batched proof OOM even at sub-batch=1 (total=%d), falling back to sequential",
            batch_size,
        )
        # Free any partial results from earlier sub-batches before fallback
        per_seq_hidden = [None] * batch_size
        per_seq_logits = [None] * batch_size
        if e.__traceback__ is not None:
            _tb.clear_frames(e.__traceback__)
        del e
        gc.collect()
        torch.cuda.empty_cache()
        use_batched = False

    # --- Phase 2: Per-sequence commitment and logprob computation ---
    results: list[tuple[list[dict], list[float], bytes, dict, str]] = []

    for idx, all_token_ids in enumerate(all_token_ids_batch):
        prompt_len = prompt_lens[idx]

        if use_batched:
            h_layer = per_seq_hidden[idx]
            logits = per_seq_logits[idx]
            assert h_layer is not None and logits is not None
        else:
            # Sequential fallback: one sequence at a time using the same
            # forward_single_layer path for numerical consistency.
            if idx == 0:
                logger.debug(
                    "SEQUENTIAL FALLBACK: seq_len=%d prompt_len=%d",
                    len(all_token_ids),
                    prompt_len,
                )
            token_tensor = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(token_tensor)
            with torch.inference_mode():
                h_batch, logits_batch = forward_single_layer(
                    model, token_tensor, attn_mask, LAYER_INDEX
                )
                h_layer = h_batch[0]
                logits = logits_batch[0].detach().to("cpu")
            del token_tensor, attn_mask, h_batch, logits_batch

        if idx == 0:
            logger.debug(
                "PROOF COMPUTATION: seq_len=%d prompt_len=%d batched=%s "
                "tokens_first_4=%s tokens_last_4=%s",
                len(all_token_ids),
                prompt_len,
                use_batched,
                all_token_ids[:4],
                all_token_ids[-4:] if len(all_token_ids) >= 4 else all_token_ids,
            )

        # --- Vectorized commitment computation (all positions at once) ---
        commitments = verifier.create_commitments_batch(h_layer, r_vec)

        if idx == 0:
            for pos in [0, prompt_len - 1, prompt_len, len(all_token_ids) - 1]:
                if 0 <= pos < len(commitments):
                    commitment = commitments[pos]
                    logger.debug(
                        "MINER COMMITMENT pos=%d token_id=%d "
                        "sketch_hash=%s rank_hash=%s hidden_norm=%.6f",
                        pos,
                        all_token_ids[pos],
                        commitment.get("sketch_hash", "")[:16],
                        commitment.get("rank_hash", "")[:16],
                        float(h_layer[pos].norm().item()),
                    )

        # --- Vectorized logprob computation ---
        completion_ids = all_token_ids[prompt_len:]
        num_completion = len(completion_ids)
        logprobs: list[float] = []

        if num_completion > 0:
            start_logit = prompt_len - 1
            end_logit = start_logit + num_completion
            valid_start = max(0, start_logit)
            valid_end = min(logits.size(0), end_logit)

            if valid_start < valid_end:
                skip_front = valid_start - start_logit
                n_valid = valid_end - valid_start
                valid_token_ids = completion_ids[skip_front : skip_front + n_valid]
                token_tensor = torch.tensor(valid_token_ids, dtype=torch.long)

                # Chunked log_softmax to cap CPU memory (~600MB per chunk)
                LOGPROB_CHUNK = 512
                chunk_logprobs: list[float] = []
                for c_start in range(0, n_valid, LOGPROB_CHUNK):
                    c_end = min(c_start + LOGPROB_CHUNK, n_valid)
                    logit_slice = logits[valid_start + c_start : valid_start + c_end]
                    log_probs_chunk = torch.log_softmax(logit_slice.float(), dim=-1)
                    tok_slice = token_tensor[c_start:c_end]
                    selected = log_probs_chunk[torch.arange(c_end - c_start), tok_slice]
                    chunk_logprobs.extend(selected.tolist())

                logprobs = (
                    [float("-inf")] * skip_front
                    + chunk_logprobs
                    + [float("-inf")] * (num_completion - skip_front - n_valid)
                )
            else:
                logprobs = [float("-inf")] * num_completion
                logger.warning(
                    "All completion logit positions out of range: start=%d end=%d logits_size=%d",
                    start_logit,
                    end_logit,
                    logits.size(0),
                )

        commitment_data = json.dumps(commitments, sort_keys=True)
        commitment_hash = hashlib.sha256(commitment_data.encode()).digest()
        if wallet is None:
            raise RuntimeError(
                "GRAIL proof generation requires bittensor wallet (unavailable in offline mode)"
            )
        signature = wallet.hotkey.sign(commitment_hash)

        beacon = {"randomness": randomness_hex}
        proof_version = GRAIL_PROOF_VERSION

        results.append((commitments, logprobs, signature, beacon, proof_version))

    logger.debug(
        "Completed proof computation for %d rollout(s) (batched=%s)",
        len(all_token_ids_batch),
        use_batched,
    )
    return results
