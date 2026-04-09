"""Qwen left-padded batching produces divergent hidden states.

Regression test proving why GRAIL proof computation must use unbatched forward
passes. Float16/bfloat16 accumulation order differs between batched and unbatched
paths, causing hidden state divergence of ~0.3-0.5 (300-500x tolerance).
Only float32+eager achieves equivalence, which is too slow for production.
"""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Qwen batching equivalence test requires a GPU",
    ),
]

MODEL_NAME = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Sequences with different lengths to force left-padding.
PROMPT_SHORT = "You are given a math problem. Show work."
PROMPT_LONG = "You are given a different math problem. Think carefully and produce the final tags."


@pytest.fixture(scope="module")
def tokenizer():
    """Shared tokenizer for the module."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@torch.inference_mode()
def _hidden_divergence(model, tokenizer, device):
    """Measure max hidden-state divergence between unbatched and left-padded batched forward."""
    seq_a = tokenizer(PROMPT_SHORT, return_tensors="pt")["input_ids"][0].tolist()
    seq_b = tokenizer(PROMPT_LONG, return_tensors="pt")["input_ids"][0].tolist()
    assert len(seq_b) > len(seq_a)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Unbatched
    ids_single = torch.tensor(seq_a, dtype=torch.long, device=device).unsqueeze(0)
    single_hidden = model(ids_single, output_hidden_states=True).hidden_states[-1][0]

    # Left-padded batch
    pad_len = len(seq_b) - len(seq_a)
    input_ids = torch.tensor([[pad_id] * pad_len + seq_a, seq_b], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [[0] * pad_len + [1] * len(seq_a), [1] * len(seq_b)],
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.tensor(
        [[0] * pad_len + list(range(len(seq_a))), list(range(len(seq_b)))],
        dtype=torch.long,
        device=device,
    )

    batch_hidden = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
    ).hidden_states[-1][0, pad_len : pad_len + len(seq_a)]

    return torch.max(torch.abs(single_hidden - batch_hidden)).item()


def test_left_padded_batch_diverges_from_unbatched(tokenizer):
    """Regression: default float16 batching diverges, proving unbatched proofs are required."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    ).eval()

    diff = _hidden_divergence(model, tokenizer, "cuda")
    assert diff > 1e-3, (
        f"Expected meaningful divergence (>1e-3), got {diff:.6f}. "
        "If this passes, batched proofs may now be viable."
    )


@pytest.mark.parametrize(
    "dtype,attn_impl",
    [
        (torch.float16, "eager"),
        (torch.bfloat16, None),
        (torch.bfloat16, "eager"),
        (torch.float32, None),
        (torch.float32, "eager"),
    ],
    ids=["fp16-eager", "bf16-default", "bf16-eager", "fp32-default", "fp32-eager"],
)
def test_batch_equivalence_across_configs(tokenizer, dtype, attn_impl):
    """Exploratory: which (dtype, attention) combos achieve batching equivalence?

    Known results: only float32 configs achieve < 1e-3 divergence.
    This test documents the behavior without asserting pass/fail, since
    the outcome depends on hardware and driver versions.
    """
    kwargs = {"torch_dtype": dtype, "device_map": "cuda"}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs).eval()
    diff = _hidden_divergence(model, tokenizer, "cuda")

    del model
    torch.cuda.empty_cache()

    label = f"{dtype} + {attn_impl or 'default'}"
    if diff < 1e-3:
        print(f"\n  {label}: divergence {diff:.6f} -- equivalence achieved")
    else:
        print(f"\n  {label}: divergence {diff:.6f} -- still diverges")
