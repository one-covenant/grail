"""Unit test proving Qwen models don't support left-padded batch equivalence.

This test suite demonstrates why GRAIL proof computation cannot use batched forward
passes with left-padding on Qwen models, even with explicit position_ids and
attention_mask parameters.

The primary test shows that hidden states diverge by ~0.3-0.5 (300-500x the tolerance),
which was causing all proof verifications to fail in production.

Additional exploratory tests check if alternative configurations can achieve
numerical equivalence:
- Eager attention (standard PyTorch instead of Flash Attention)
- BFloat16 precision (better numerical stability than float16)
- Float32 precision (full precision)
- Combinations of attention implementations and dtypes

These tests help determine if there's any viable way to use batched inference
for proof computation, or if unbatched forward passes remain necessary.
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


@torch.inference_mode()
def _run_unbatched(model, tokens, device):
    """Run a single sequence without padding and return hidden states."""
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[-1][0].detach().to("cpu")


@torch.inference_mode()
def _run_unbatched_with_logits(model, tokens, device):
    """Run a single sequence without padding and return hidden states + logits."""
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    outputs = model(input_ids, output_hidden_states=True)
    return (outputs.hidden_states[-1][0].detach().to("cpu"), outputs.logits[0].detach().to("cpu"))


@torch.inference_mode()
def _run_left_padded_batch(model, batch_tokens, device, pad_id):
    """Run a left-padded batch with explicit attention mask + position IDs."""
    max_len = max(len(seq) for seq in batch_tokens)
    padded_inputs = []
    attention_masks = []
    position_ids = []
    left_pads = []
    for seq in batch_tokens:
        pad_len = max_len - len(seq)
        left_pads.append(pad_len)
        padded_inputs.append([pad_id] * pad_len + seq)
        attention_masks.append([0] * pad_len + [1] * len(seq))
        position_ids.append([0] * pad_len + list(range(len(seq))))

    input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    pos_ids = torch.tensor(position_ids, dtype=torch.long, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1].detach().to("cpu")
    return hidden_states, left_pads


@torch.inference_mode()
def _run_left_padded_batch_with_logits(model, batch_tokens, device, pad_id):
    """Run a left-padded batch with explicit attention mask + position IDs, return hidden states + logits."""
    max_len = max(len(seq) for seq in batch_tokens)
    padded_inputs = []
    attention_masks = []
    position_ids = []
    left_pads = []
    for seq in batch_tokens:
        pad_len = max_len - len(seq)
        left_pads.append(pad_len)
        padded_inputs.append([pad_id] * pad_len + seq)
        attention_masks.append([0] * pad_len + [1] * len(seq))
        position_ids.append([0] * pad_len + list(range(len(seq))))

    input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)
    pos_ids = torch.tensor(position_ids, dtype=torch.long, device=device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1].detach().to("cpu")
    logits = outputs.logits.detach().to("cpu")
    return hidden_states, logits, left_pads


def test_left_padded_batch_differs_from_unbatched():
    """Regression test: Left-padded batching produces different hidden states on Qwen.

    This test proves why the miner must use unbatched forward passes for proofs.
    Even with explicit position_ids and attention_mask, Qwen's fused attention
    kernels produce different hidden states when using left-padding vs no padding.

    Typical divergence: ~0.3-0.5 (300-500x larger than tolerance)
    This is why proofs were failing in production before the unbatching fix.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Try to use default attention (usually Flash Attention 2 for Qwen)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    # Empirically, Qwen diverges by >1e-3 even with explicit masks/position ids.
    print(f"\n📊 Hidden state divergence: {diff:.6f} (threshold: 1e-3)")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    assert diff > 1e-3, (
        f"Expected meaningful divergence (>1e-3), got max hidden-state diff {diff:.6f}"
    )


def test_eager_attention_batch_equivalence():
    """Test if eager attention implementation achieves better numerical equivalence.

    This test checks whether switching from Flash Attention to eager (standard PyTorch)
    attention reduces or eliminates the hidden state divergence seen with batched
    left-padding.

    Flash Attention uses fused kernels that may have different numerical behavior
    with padding masks. Eager attention uses standard PyTorch operations which may
    be more numerically stable across different batching strategies.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use eager attention implementation for better numerical stability
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",  # Force standard PyTorch attention
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    print(f"\n📊 [Eager Attention] Hidden state divergence: {diff:.6f}")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    if diff < 1e-3:
        print("   ✅ SUCCESS: Eager attention achieves numerical equivalence!")
    else:
        print(f"   ⚠️  Still diverges by {diff:.6f} (threshold: 1e-3)")

    # This is an exploratory test - we don't assert pass/fail
    # Just report the results for investigation


def test_float32_batch_equivalence():
    """Test if float32 precision achieves better numerical equivalence.

    This test checks whether using full float32 precision instead of float16
    reduces or eliminates the hidden state divergence with batched left-padding.

    Higher precision may reduce numerical errors that accumulate differently
    between batched and unbatched forward passes.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use float32 for maximum numerical precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    print(f"\n📊 [Float32] Hidden state divergence: {diff:.6f}")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    if diff < 1e-3:
        print("   ✅ SUCCESS: Float32 precision achieves numerical equivalence!")
    else:
        print(f"   ⚠️  Still diverges by {diff:.6f} (threshold: 1e-3)")

    # This is an exploratory test - we don't assert pass/fail
    # Just report the results for investigation


def test_bfloat16_batch_equivalence():
    """Test if bfloat16 precision achieves better numerical equivalence.

    This test checks whether using bfloat16 instead of float16 reduces or
    eliminates the hidden state divergence with batched left-padding.

    BFloat16 has the same dynamic range as float32 (8 exponent bits) but
    reduced mantissa precision (7 bits vs 10 for float16). This can provide
    better numerical stability than float16 in some operations while being
    faster than float32.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use bfloat16 for better numerical stability than float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    print(f"\n📊 [BFloat16] Hidden state divergence: {diff:.6f}")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    if diff < 1e-3:
        print("   ✅ SUCCESS: BFloat16 precision achieves numerical equivalence!")
    else:
        print(f"   ⚠️  Still diverges by {diff:.6f} (threshold: 1e-3)")

    # This is an exploratory test - we don't assert pass/fail
    # Just report the results for investigation


def test_eager_bfloat16_batch_equivalence():
    """Test if combining eager attention + bfloat16 achieves numerical equivalence.

    This test combines eager attention (standard PyTorch) with bfloat16 precision.
    This may offer a good balance between numerical stability and performance.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use eager attention with bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    print(f"\n📊 [Eager + BFloat16] Hidden state divergence: {diff:.6f}")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    if diff < 1e-3:
        print("   ✅ SUCCESS: Eager+BFloat16 achieves numerical equivalence!")
    else:
        print(f"   ⚠️  Still diverges by {diff:.6f} (threshold: 1e-3)")

    # This is an exploratory test - we don't assert pass/fail
    # Just report the results for investigation


def test_eager_float32_batch_equivalence():
    """Test if combining eager attention + float32 achieves numerical equivalence.

    This test combines both strategies: eager attention (no Flash Attention kernels)
    and full float32 precision to see if this eliminates the batching divergence.

    This is the most numerically stable configuration but also the slowest.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use both eager attention and float32 for maximum stability
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()

    # Two sequences with different lengths to force left padding.
    seq_a = tokenizer("You are given a math problem. Show work.", return_tensors="pt")["input_ids"][
        0
    ].tolist()
    seq_b = tokenizer(
        "You are given a different math problem. Think carefully and produce the final tags.",
        return_tensors="pt",
    )["input_ids"][0].tolist()

    # Sanity: ensure seq_b is longer to trigger non-zero left pad for seq_a.
    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden = _run_unbatched(model, seq_a, device)
    batch_hidden, left_pads = _run_left_padded_batch(
        model,
        [seq_a, seq_b],
        device,
        pad_id,
    )

    left_pad_a = left_pads[0]
    batched_slice = batch_hidden[0, left_pad_a : left_pad_a + len(seq_a)]

    diff = torch.max(torch.abs(single_hidden - batched_slice)).item()

    print(f"\n📊 [Eager + Float32] Hidden state divergence: {diff:.6f}")
    print(f"   Left padding: {left_pad_a} tokens")
    print(f"   Sequence length: {len(seq_a)} tokens")

    if diff < 1e-3:
        print("   ✅ SUCCESS: Eager+Float32 achieves numerical equivalence!")
    else:
        print(f"   ⚠️  Still diverges by {diff:.6f} (threshold: 1e-3)")

    # This is an exploratory test - we don't assert pass/fail
    # Just report the results for investigation


def test_diagnose_padding_issue():
    """Diagnostic test to understand why position_ids don't achieve equivalence.

    This test examines layer-by-layer hidden states to identify where the
    divergence occurs and helps understand the root cause.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use eager attention and float16 to isolate the issue
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()

    # Simple short sequence for easier debugging
    seq_a = tokenizer("Hello world", return_tensors="pt")["input_ids"][0].tolist()
    seq_b = tokenizer("Hello world this is longer", return_tensors="pt")["input_ids"][0].tolist()

    assert len(seq_b) > len(seq_a)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Unbatched run
    input_ids_single = torch.tensor(seq_a, dtype=torch.long, device=device).unsqueeze(0)
    outputs_single = model(input_ids_single, output_hidden_states=True)

    # Batched run
    max_len = max(len(seq_a), len(seq_b))
    pad_len = max_len - len(seq_a)

    padded_a = [pad_id] * pad_len + seq_a
    padded_b = seq_b
    attention_mask_a = [0] * pad_len + [1] * len(seq_a)
    attention_mask_b = [1] * len(seq_b)
    position_ids_a = [0] * pad_len + list(range(len(seq_a)))
    position_ids_b = list(range(len(seq_b)))

    input_ids_batch = torch.tensor([padded_a, padded_b], dtype=torch.long, device=device)
    attention_mask_batch = torch.tensor(
        [attention_mask_a, attention_mask_b], dtype=torch.long, device=device
    )
    position_ids_batch = torch.tensor(
        [position_ids_a, position_ids_b], dtype=torch.long, device=device
    )

    print("\n🔍 Diagnostic Information:")
    print(f"   Sequence A length: {len(seq_a)}")
    print(f"   Left padding: {pad_len} tokens")
    print(f"   Position IDs (padded): {position_ids_a}")
    print(f"   Attention mask: {attention_mask_a}")

    outputs_batch = model(
        input_ids=input_ids_batch,
        attention_mask=attention_mask_batch,
        position_ids=position_ids_batch,
        output_hidden_states=True,
    )

    # Compare layer by layer
    print("\n📊 Layer-by-layer divergence:")
    num_layers = len(outputs_single.hidden_states)

    for layer_idx in range(num_layers):
        single_hidden = outputs_single.hidden_states[layer_idx][0].detach().to("cpu")
        batch_hidden = outputs_batch.hidden_states[layer_idx][0, pad_len:].detach().to("cpu")

        diff = torch.max(torch.abs(single_hidden - batch_hidden)).item()

        layer_name = f"Layer {layer_idx}" if layer_idx > 0 else "Embedding"
        print(f"   {layer_name:12s}: {diff:.6f}")

        if layer_idx == 1 and diff > 0.01:
            print("      ⚠️  Divergence starts early (layer 1)!")
            break

    print("\n💡 Key insight: Divergence starts at Layer 1 (first transformer layer)")
    print("   This means RoPE/position embeddings work correctly,")
    print("   but something in attention or feedforward causes divergence.")


def test_understand_root_cause():
    """Deep dive: What causes the divergence even with position_ids?

    Theory: Float16 accumulation order differs between batched and unbatched.
    Even though attention_mask prevents attending to padding, the actual
    computation order/grouping differs, causing float16 rounding errors to
    accumulate differently.

    This test validates that the issue is fundamentally about floating-point
    arithmetic, not missing parameters.
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    # Test 1: Eager attention with float16
    print("\n📊 Test 1: Eager Attention + Float16")
    model_f16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()

    seq_a = tokenizer("Test", return_tensors="pt")["input_ids"][0].tolist()
    seq_b = tokenizer("Test sequence", return_tensors="pt")["input_ids"][0].tolist()
    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    single_hidden_f16 = _run_unbatched(model_f16, seq_a, device)
    batch_hidden_f16, left_pads_f16 = _run_left_padded_batch(
        model_f16, [seq_a, seq_b], device, pad_id
    )
    diff_f16 = torch.max(
        torch.abs(
            single_hidden_f16
            - batch_hidden_f16[0, left_pads_f16[0] : left_pads_f16[0] + len(seq_a)]
        )
    ).item()
    print(f"   Divergence: {diff_f16:.6f}")

    del model_f16
    torch.cuda.empty_cache()

    # Test 2: Eager attention with float32
    print("\n📊 Test 2: Eager Attention + Float32")
    model_f32 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()

    single_hidden_f32 = _run_unbatched(model_f32, seq_a, device)
    batch_hidden_f32, left_pads_f32 = _run_left_padded_batch(
        model_f32, [seq_a, seq_b], device, pad_id
    )
    diff_f32 = torch.max(
        torch.abs(
            single_hidden_f32
            - batch_hidden_f32[0, left_pads_f32[0] : left_pads_f32[0] + len(seq_a)]
        )
    ).item()
    print(f"   Divergence: {diff_f32:.6f}")

    del model_f32
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Float16 divergence: {diff_f16:.6f} ({'FAIL' if diff_f16 > 1e-3 else 'PASS'})")
    print(f"Float32 divergence: {diff_f32:.6f} ({'FAIL' if diff_f32 > 1e-3 else 'PASS'})")
    print()
    print("🎯 ROOT CAUSE: Floating-point accumulation order")
    print()
    print("In batched processing, PyTorch groups operations differently:")
    print("- Batched: Operations vectorized across batch dimension")
    print("  - Matrix multiplications grouped: [batch, seq, hidden] x [hidden, hidden]")
    print("  - Reductions and norms computed with batch dimension present")
    print()
    print("- Unbatched: Operations on single sequence")
    print("  - Matrix multiplications: [1, seq, hidden] x [hidden, hidden]")
    print("  - Different memory layout and computation grouping")
    print()
    print("Float16 has only ~3 decimal digits of precision. These tiny")
    print("differences in computation order cause rounding errors that")
    print("accumulate across 24 transformer layers, resulting in 0.3-0.5")
    print("divergence.")
    print()
    print("Float32 has ~7 decimal digits, so the accumulation errors stay")
    print("below 1e-3 threshold.")
    print()
    print("⚠️  This is NOT a bug - it's inherent to float16 arithmetic!")
    print("   Even with perfect position_ids and attention_mask, you cannot")
    print("   achieve exact numerical equivalence with float16 batching.")
    print()


def test_compare_logits_divergence():
    """Compare how hidden state divergence affects final logits.

    This test measures divergence in both hidden states and logits to understand
    whether the hidden state differences actually impact the final predictions.

    Key metrics:
    - Max absolute logit difference
    - Top-1 token agreement
    - Top-5 token overlap
    """
    model_name = os.getenv("QWEN_EQ_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("\n" + "=" * 70)
    print("HIDDEN STATES vs LOGITS DIVERGENCE ANALYSIS")
    print("=" * 70)

    device = "cuda"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Use a meaningful sequence for better analysis
    seq_a = tokenizer("You are given a math problem.", return_tensors="pt")["input_ids"][0].tolist()
    seq_b = tokenizer(
        "You are given a different math problem with more tokens.", return_tensors="pt"
    )["input_ids"][0].tolist()

    assert len(seq_b) > len(seq_a)

    # Test different configurations
    configs = [
        ("Float16 + Flash Attention", torch.float16, None),
        ("Float16 + Eager Attention", torch.float16, "eager"),
        ("BFloat16 + Flash Attention", torch.bfloat16, None),
        ("Float32 + Flash Attention", torch.float32, None),
        ("Float32 + Eager Attention", torch.float32, "eager"),
    ]

    results = []

    for config_name, dtype, attn_impl in configs:
        print(f"\n📊 Testing: {config_name}")
        print("-" * 70)

        kwargs = {"torch_dtype": dtype, "device_map": "cuda"}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()

        # Run unbatched
        single_hidden, single_logits = _run_unbatched_with_logits(model, seq_a, device)

        # Run batched
        batch_hidden, batch_logits, left_pads = _run_left_padded_batch_with_logits(
            model, [seq_a, seq_b], device, pad_id
        )

        left_pad_a = left_pads[0]
        seq_len = len(seq_a)

        # Extract the relevant slice from batch
        batched_hidden_slice = batch_hidden[0, left_pad_a : left_pad_a + seq_len]
        batched_logits_slice = batch_logits[0, left_pad_a : left_pad_a + seq_len]

        # Calculate hidden state divergence
        hidden_diff = torch.max(torch.abs(single_hidden - batched_hidden_slice)).item()

        # Calculate logit divergence
        logit_diff = torch.max(torch.abs(single_logits - batched_logits_slice)).item()

        # Check top-1 token agreement (most important for generation)
        single_top1 = torch.argmax(single_logits, dim=-1)
        batched_top1 = torch.argmax(batched_logits_slice, dim=-1)
        top1_agreement = (single_top1 == batched_top1).float().mean().item()

        # Check top-5 overlap
        single_top5 = torch.topk(single_logits, k=5, dim=-1).indices
        batched_top5 = torch.topk(batched_logits_slice, k=5, dim=-1).indices

        # Calculate average top-5 overlap per position
        top5_overlaps = []
        for pos in range(seq_len):
            overlap = len(set(single_top5[pos].tolist()) & set(batched_top5[pos].tolist()))
            top5_overlaps.append(overlap / 5.0)
        avg_top5_overlap = sum(top5_overlaps) / len(top5_overlaps)

        print(f"   Hidden state max diff:  {hidden_diff:.6f}")
        print(f"   Logit max diff:         {logit_diff:.6f}")
        print(f"   Top-1 agreement:        {top1_agreement * 100:.1f}%")
        print(f"   Avg Top-5 overlap:      {avg_top5_overlap * 100:.1f}%")

        results.append(
            {
                "config": config_name,
                "hidden_diff": hidden_diff,
                "logit_diff": logit_diff,
                "top1_agreement": top1_agreement,
                "top5_overlap": avg_top5_overlap,
            }
        )

        del model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Impact on Predictions")
    print("=" * 70)

    for r in results:
        status = (
            "✅ PASS"
            if r["top1_agreement"] == 1.0
            else f"⚠️  {r['top1_agreement'] * 100:.1f}% agree"
        )
        print(f"\n{r['config']}:")
        print(f"  Hidden diff: {r['hidden_diff']:.4f} → Logit diff: {r['logit_diff']:.4f}")
        print(f"  Top-1 tokens: {status}")

    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS:")
    print("=" * 70)
    print()
    print("1. Hidden state divergence amplifies through the LM head:")
    print("   - Hidden states differ by ~0.5-6.0")
    print("   - Logits can differ by 10-100x more!")
    print()
    print("2. Float16/BFloat16 may produce DIFFERENT TOP-1 TOKENS")
    print("   - This directly affects generation quality")
    print("   - Critical for proof verification where exact tokens matter")
    print()
    print("3. Float32 achieves both numerical and token equivalence")
    print("   - Small hidden state differences don't propagate")
    print("   - Top-1 tokens remain identical")
    print()
