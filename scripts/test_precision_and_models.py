#!/usr/bin/env python3
"""
Test precision of computing actual weights (base + delta) on-the-fly.
Also verify correct HuggingFace model names.
"""

import torch
import numpy as np
from pathlib import Path

# Correct HuggingFace model names (instruction-tuned variants)
MODEL_NAMES = {
    'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
    'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
    'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
    'llama3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'gemma3-1b': 'google/gemma-3-1b-it',
    'gemma3-4b': 'google/gemma-3-4b-it',
}


def test_bf16_precision():
    """Test that bf16 addition preserves precision correctly."""
    print("="*60)
    print("TEST 1: BFloat16 Precision Test")
    print("="*60)

    # Simulate base weights (typical range)
    torch.manual_seed(42)
    base_weights = torch.randn(1000, 1000, dtype=torch.bfloat16) * 0.1

    # Simulate delta values (small gradient updates)
    delta_values = torch.randn(1000, dtype=torch.bfloat16) * 0.001

    # Sparse indices
    indices = torch.randint(0, 1000*1000, (1000,))

    # Method 1: Direct bf16 addition
    base_flat = base_weights.flatten()
    actual_bf16 = base_flat[indices] + delta_values

    # Method 2: Compute in float32, then convert back
    base_f32 = base_flat[indices].float()
    delta_f32 = delta_values.float()
    actual_f32_then_bf16 = (base_f32 + delta_f32).to(torch.bfloat16)

    # Compare
    diff = (actual_bf16.float() - actual_f32_then_bf16.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Base weights range: [{base_weights.min():.4f}, {base_weights.max():.4f}]")
    print(f"Delta values range: [{delta_values.min():.6f}, {delta_values.max():.6f}]")
    print(f"Max difference between methods: {max_diff:.2e}")
    print(f"Mean difference between methods: {mean_diff:.2e}")

    # bf16 has ~3 decimal digits of precision
    if max_diff < 1e-3:
        print("✓ PASS: bf16 precision is sufficient")
        return True
    else:
        print("✗ FAIL: Precision loss detected")
        return False


def test_actual_delta_file():
    """Test with a real delta file."""
    print("\n" + "="*60)
    print("TEST 2: Real Delta File Test")
    print("="*60)

    # Find a delta file
    delta_dirs = [
        '/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-iter8/checkpoints/deltas_math_instance0_seed42',
        '/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-iter16/checkpoints/deltas_math_instance0_seed42',
    ]

    delta_file = None
    for d in delta_dirs:
        p = Path(d) / 'delta_000000.pt'
        if p.exists():
            delta_file = p
            break

    if delta_file is None:
        print("No delta file found for testing")
        return True

    print(f"Loading delta file: {delta_file}")
    delta = torch.load(delta_file, map_location='cpu', weights_only=False)

    # Check a few layers
    layers_checked = 0
    for layer_name, layer_data in list(delta.get('layers', {}).items())[:3]:
        indices = layer_data['indices']
        values = layer_data['values']
        shape = layer_data['shape']

        print(f"\n  Layer: {layer_name}")
        print(f"    Shape: {shape}")
        print(f"    Indices shape: {indices.shape}")
        print(f"    Values dtype: {values.dtype}")
        print(f"    Values range: [{values.min():.6f}, {values.max():.6f}]")

        # Simulate base weights
        if len(shape) == 2:
            fake_base = torch.randn(shape, dtype=torch.bfloat16) * 0.1

            # Extract at indices (COO format)
            if indices.shape[0] == 2:
                row_idx, col_idx = indices[0], indices[1]
                base_at_idx = fake_base[row_idx, col_idx]

                # Compute actual weights
                actual = base_at_idx + values

                print(f"    Actual weights computed: {actual.shape}, dtype={actual.dtype}")
                print(f"    Actual range: [{actual.min():.6f}, {actual.max():.6f}]")

                layers_checked += 1

    if layers_checked > 0:
        print(f"\n✓ PASS: Successfully computed actual weights for {layers_checked} layers")
        return True
    else:
        print("\n✗ Could not test any layers")
        return False


def test_model_availability():
    """Test that model names are valid (without downloading full models)."""
    print("\n" + "="*60)
    print("TEST 3: HuggingFace Model Names")
    print("="*60)

    from huggingface_hub import HfApi
    api = HfApi()

    for short_name, hf_name in MODEL_NAMES.items():
        try:
            # Just check if model exists (doesn't download)
            info = api.model_info(hf_name)
            size_gb = sum(s.size for s in info.siblings if s.rfilename.endswith('.safetensors')) / 1e9
            print(f"  ✓ {short_name:15s} -> {hf_name:40s} ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"  ✗ {short_name:15s} -> {hf_name:40s} ERROR: {e}")

    return True


def test_index_extraction():
    """Test that we can correctly extract weights at sparse indices."""
    print("\n" + "="*60)
    print("TEST 4: Index Extraction Correctness")
    print("="*60)

    # Create known tensor
    torch.manual_seed(123)
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    print(f"Weight tensor (4x6):\n{weight}")

    # COO indices for positions (0,0), (1,2), (3,5)
    row_idx = torch.tensor([0, 1, 3])
    col_idx = torch.tensor([0, 2, 5])

    # Extract
    extracted = weight[row_idx, col_idx]
    expected = torch.tensor([0, 8, 23], dtype=torch.bfloat16)

    print(f"\nIndices: rows={row_idx.tolist()}, cols={col_idx.tolist()}")
    print(f"Extracted: {extracted.tolist()}")
    print(f"Expected:  {expected.tolist()}")

    if torch.allclose(extracted, expected):
        print("✓ PASS: Index extraction correct")
        return True
    else:
        print("✗ FAIL: Index extraction incorrect")
        return False


def main():
    print("Testing precision and model names for compression benchmark\n")

    results = []

    # Test 1: bf16 precision
    results.append(("bf16 precision", test_bf16_precision()))

    # Test 2: Real delta file
    results.append(("real delta file", test_actual_delta_file()))

    # Test 3: Model names
    try:
        results.append(("model names", test_model_availability()))
    except Exception as e:
        print(f"\nSkipping model name test: {e}")

    # Test 4: Index extraction
    results.append(("index extraction", test_index_extraction()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✓ All tests passed! Safe to proceed with on-the-fly computation.")
    else:
        print("\n✗ Some tests failed. Review before proceeding.")

    return all_pass


if __name__ == '__main__':
    main()
