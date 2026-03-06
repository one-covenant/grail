"""HuggingFace generation backend."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .base import GenerationParams, TextGenBackend

logger = logging.getLogger(__name__)


class HFBackend(TextGenBackend):
    """HuggingFace generation backend using a provided model/tokenizer instance.

    This backend does not own the model; it reuses the instance passed in to
    maintain a single copy in memory when used for both generation and proofs.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        *,
        return_chosen_logprobs: bool = False,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._return_chosen_logprobs = bool(return_chosen_logprobs)

    async def generate(
        self,
        prompt_ids_batch: list[list[int]],
        *,
        params: GenerationParams,
        seeds: list[int] | None = None,
    ) -> list[tuple[list[int], list[float] | None]]:
        batch_size = len(prompt_ids_batch)
        if batch_size == 0:
            return []

        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        eos_id = self._tokenizer.eos_token_id

        # Left-pad inputs to max length
        max_len = max(len(p) for p in prompt_ids_batch)
        padded_inputs: list[list[int]] = []
        attention_masks: list[list[int]] = []
        left_pads: list[int] = []
        for p in prompt_ids_batch:
            pad_len = max_len - len(p)
            padded_inputs.append([pad_id] * pad_len + p)
            attention_masks.append([0] * pad_len + [1] * len(p))
            left_pads.append(pad_len)

        input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=self._device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=self._device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.max_new_tokens),
            "temperature": float(params.temperature),
            "do_sample": bool(params.do_sample),
            "top_p": float(params.top_p),
            "return_dict_in_generate": True,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if params.top_k is not None:
            gen_kwargs["top_k"] = int(params.top_k)
        if params.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(params.repetition_penalty)

        # Seeds are currently ignored to favor batched generation efficiency
        if seeds is not None and seeds:
            logger.debug("HFBackend: ignoring seeds for batched generation")

        with torch.inference_mode():
            outputs = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        results: list[tuple[list[int], list[float] | None]] = []
        for b in range(batch_size):
            seq = outputs.sequences[b]
            # Remove left padding
            seq_wo_left = seq[left_pads[b] :]
            all_ids = seq_wo_left.tolist()

            if params.trim_right_padding:
                # Trim trailing padding conservatively
                if pad_id is not None and pad_id != eos_id:
                    # Find last non-pad index
                    last_non_pad = len(all_ids) - 1
                    while last_non_pad >= 0 and all_ids[last_non_pad] == pad_id:
                        last_non_pad -= 1
                    all_ids = all_ids[: max(0, last_non_pad + 1)]
            # HF backend doesn't currently support logprobs extraction
            results.append((all_ids, None))

        return results
