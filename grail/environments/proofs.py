# GRAIL proof computation: commitments and logprobs

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
import traceback as _tb
from typing import Any

import torch

from ..model.forward import forward_single_layer
from ..protocol.constants import GRAIL_PROOF_VERSION, LAYER_INDEX

logger = logging.getLogger(__name__)


def _get_env_config() -> tuple[int, str]:
    """Get quant bits and deterministic mode from environment."""
    quant_bits_str = os.getenv("GRAIL_PROOF_QUANT_BITS", "8")
    try:
        quant_bits = int(quant_bits_str)
        if quant_bits not in (8, 16):
            logger.warning("GRAIL_PROOF_QUANT_BITS must be 8 or 16, got %s. Defaulting to 8.", quant_bits_str)
            quant_bits = 8
    except ValueError:
        logger.warning("Invalid GRAIL_PROOF_QUANT_BITS: %s. Defaulting to 8.", quant_bits_str)
        quant_bits = 8

    deterministic_mode = os.getenv("GRAIL_PROOF_DETERMINISTIC_MODE", "fixed_point").lower()
    if deterministic_mode not in ("fixed_point", "fp32_deterministic", "off"):
        logger.warning(
            "Invalid GRAIL_PROOF_DETERMINISTIC_MODE: %s. Defaulting to 'fixed_point'.",
            deterministic_mode,
        )
        deterministic_mode = "fixed_point"

    return quant_bits, deterministic_mode


def _project_fixed_point(h_layer: torch.Tensor, r_vec: torch.Tensor, bits: int) -> torch.Tensor:
    """Project using fixed-point integer arithmetic via float64."""
    assert r_vec.dim() in (1, 2), "r_vec must be 1D or 2D (batch x hidden)"
    if r_vec.dim() == 2:
        assert r_vec.size(0) == 1 or r_vec.size(1) == 1, "r_vec 2D must have a singleton dimension of size 1"

    with torch.no_grad():
        device = h_layer.device
        r_vec = r_vec.to(device)

        max_int = (1 << (bits - 1)) - 1

        # Avoid asymmetric saturation
        r_max = torch.max(torch.abs(r_vec))
        scale_r = r_max / max_int
        if scale_r == 0:
            scale_r = torch.tensor(1.0, dtype=r_vec.dtype, device=device)

        h_max = torch.max(torch.abs(h_layer))
        scale_h = h_max / max_int
        if scale_h == 0:
            scale_h = torch.tensor(1.0, dtype=h_layer.dtype, device=device)

        dtype = torch.int16 if bits <= 8 else torch.int32
        r_vec_int = (r_vec / scale_r).round().to(dtype)
        h_layer_int = (h_layer / scale_h).round().to(dtype)

        # Force contiguous float64 to avoid CUDA integer matmul and improve coalescing
        r_vec_f64 = r_vec_int.to(dtype=torch.float64, device=device).contiguous()
        h_layer_f64 = h_layer_int.to(dtype=torch.float64, device=device).contiguous()

        hidden_dim = h_layer.size(1)
        CHUNK_SIZE_HD = min(hidden_dim, 8192)  # dynamic chunking

        if hidden_dim > CHUNK_SIZE_HD:
            if r_vec_f64.dim() == 1:
                s_vals_f64 = torch.zeros(h_layer_f64.size(0), dtype=torch.float64, device=device)
                for start_idx in range(0, hidden_dim, CHUNK_SIZE_HD):
                    end_idx = min(start_idx + CHUNK_SIZE_HD, hidden_dim)
                    h_chunk = h_layer_f64[:, start_idx:end_idx]
                    r_chunk = r_vec_f64[start_idx:end_idx]
                    s_vals_f64 += torch.matmul(h_chunk, r_chunk)
            else:
                if r_vec_f64.size(0) == 1:
                    s_vals_f64 = torch.zeros((h_layer_f64.size(0), 1), dtype=torch.float64, device=device)
                    for start_idx in range(0, hidden_dim, CHUNK_SIZE_HD):
                        end_idx = min(start_idx + CHUNK_SIZE_HD, hidden_dim)
                        h_chunk = h_layer_f64[:, start_idx:end_idx]
                        r_chunk = r_vec_f64[:, start_idx:end_idx]
                        s_vals_f64 += torch.matmul(h_chunk, r_chunk.t())
                else:
                    s_vals_f64 = torch.zeros((h_layer_f64.size(0), r_vec_f64.size(1)), dtype=torch.float64, device=device)
                    for start_idx in range(0, hidden_dim, CHUNK_SIZE_HD):
                        end_idx = min(start_idx + CHUNK_SIZE_HD, hidden_dim)
                        h_chunk = h_layer_f64[:, start_idx:end_idx]
                        r_chunk = r_vec_f64[start_idx:end_idx, :]
                        s_vals_f64 += torch.matmul(h_chunk, r_chunk)
        else:
            if r_vec_f64.dim() == 1:
                s_vals_f64 = torch.matmul(h_layer_f64, r_vec_f64)
            elif r_vec_f64.size(0) == 1:
                s_vals_f64 = torch.matmul(h_layer_f64, r_vec_f64.t())
            else:
                s_vals_f64 = torch.matmul(h_layer_f64, r_vec_f64)

        # de-quantize correctly
        s_vals_fp32 = (s_vals_f64 * (scale_h * scale_r)).float()

        if r_vec.dim() == 1 and s_vals_fp32.dim() == 2:
            s_vals_fp32 = s_vals_fp32.squeeze(-1)

        return s_vals_fp32


def _batched_forward_pass(
    model: Any,
    device: str,
    all_token_ids_batch: list[list[int]],
    *,
    keep_logits_on_gpu: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run sub‑batched forward passes with right‑padding."""
    from ..protocol.constants import PROOF_BATCH_SIZE

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
            input_ids = torch.full((sub_bs, sub_max), pad_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros(sub_bs, sub_max, dtype=torch.long, device=device)
            for i, (seq, slen) in enumerate(zip(sub_seqs, sub_lens, strict=True)):
                input_ids[i, :slen] = torch.tensor(seq, dtype=torch.long, device=device)
                attn_mask[i, :slen] = 1

            with torch.inference_mode():
                h_layer, logits = forward_single_layer(model, input_ids, attn_mask, LAYER_INDEX)

            for i, slen in enumerate(sub_lens):
                per_seq_hidden.append(h_layer[i, :slen, :].clone())
                if keep_logits_on_gpu:
                    per_seq_logits.append(logits[i, :slen, :].clone())
                else:
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
            if e.__traceback__ is not None:
                _tb.clear_frames(e.__traceback__)
            del e
            gc.collect()
            torch.cuda.empty_cache()

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
    *,
    gpu_logprobs: bool = True,
) -> list[tuple[list[dict], list[float], bytes, dict, str]]:
    """Compute GRAIL commitments and logprobs for the miner pipeline."""
    batch_size = len(all_token_ids_batch)
    if batch_size == 0:
        return []

    from ..protocol.grail_verifier import GRAILVerifier

    verifier = GRAILVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    # --- Phase 1: forward passes ---
    use_batched = True
    per_seq_hidden: list[torch.Tensor | None] = [None] * batch_size
    per_seq_logits: list[torch.Tensor | None] = [None] * batch_size

    try:
        hidden_list, logits_list = _batched_forward_pass(
            model, device, all_token_ids_batch, keep_logits_on_gpu=gpu_logprobs
        )
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
        per_seq_hidden = [None] * batch_size
        per_seq_logits = [None] * batch_size
        if e.__traceback__ is not None:
            _tb.clear_frames(e.__traceback__)
        del e
        gc.collect()
        torch.cuda.empty_cache()
        use_batched = False

    # --- Phase 2: commitments and logprobs per sequence ---
    results: list[tuple[list[dict], list[float], bytes, dict, str]] = []

    for idx, all_token_ids in enumerate(all_token_ids_batch):
        prompt_len = prompt_lens[idx]

        if use_batched:
            h_layer = per_seq_hidden[idx]
            logits = per_seq_logits[idx]
            if h_layer is None or logits is None:
                from ..protocol.errors import ProtocolViolationError

                raise ProtocolViolationError(
                    f"Proof batched forward returned None for sequence {idx} "
                    f"(batch_size={batch_size}); cannot continue"
                )
        else:
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

        # deterministic projection
        quant_bits, deterministic_mode = _get_env_config()
        s_vals = None

        if deterministic_mode == "fixed_point":
            s_vals = _project_fixed_point(h_layer, r_vec, quant_bits)
        elif deterministic_mode == "fp32_deterministic":
            prev_deterministic = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                if r_vec.dim() == 1:
                    s_vals = torch.matmul(h_layer, r_vec.to(h_layer.device))
                elif r_vec.dim() == 2:
                    if r_vec.size(0) == 1:
                        s_vals = torch.matmul(h_layer, r_vec.to(h_layer.device).t())
                    else:
                        s_vals = torch.matmul(h_layer, r_vec.to(h_layer.device))
                else:
                    s_vals = torch.matmul(h_layer, r_vec.to(h_layer.device).view(-1))

                if r_vec.dim() == 1 and s_vals.dim() == 2:
                    s_vals = s_vals.squeeze(-1)
            except RuntimeError as e:
                logger.warning(
                    "Deterministic fp32 algorithms not fully supported on this platform: %s. "
                    "Falling back to standard float32 projection.",
                    e,
                )
                s_vals = None
            finally:
                torch.use_deterministic_algorithms(prev_deterministic)

        # commitments
        if s_vals is not None:
            # check signature compatibility
            has_projected_arg = False
            try:
                sig = inspect.signature(verifier.create_commitments_batch)
                if "projected_s_vals" in sig.parameters:
                    has_projected_arg = True
            except (ValueError, TypeError):
                has_projected_arg = False

            if has_projected_arg:
                commitments = verifier.create_commitments_batch(h_layer, r_vec, projected_s_vals=s_vals)
            else:
                raise RuntimeError(
                    "Verifier must support 'projected_s_vals' argument. Please update GRAILVerifier."
                )
        else:
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

        # logprobs
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

                if gpu_logprobs and logits.is_cuda:
                    token_tensor_gpu = torch.tensor(
                        valid_token_ids, dtype=torch.long, device=logits.device
                    )
                    LOGPROB_CHUNK = 512
                    chunk_logprobs: list[float] = []
                    for c_start in range(0, n_valid, LOGPROB_CHUNK):
                        c_end = min(c_start + LOGPROB_CHUNK, n_valid)
                        logit_slice = logits[valid_start + c_start : valid_start + c_end]
                        log_probs_chunk = torch.log_softmax(logit_slice.float(), dim=-1)
                        tok_slice = token_tensor_gpu[c_start:c_end]
                        selected = log_probs_chunk[
                            torch.arange(c_end - c_start, device=logits.device), tok_slice
                        ]
                        chunk_logprobs.extend(selected.tolist())
                        del log_probs_chunk
                    del token_tensor_gpu
                else:
                    token_tensor = torch.tensor(valid_token_ids, dtype=torch.long)
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

        if gpu_logprobs and use_batched and per_seq_logits[idx] is not None:
            per_seq_logits[idx] = None

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
