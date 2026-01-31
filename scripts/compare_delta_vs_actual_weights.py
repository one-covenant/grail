#!/usr/bin/env python3
"""
Quick comparison: Compression of delta values vs actual weight values.

Tests whether compressing (base + delta) differs from compressing (delta).
"""

import os
import sys
import time
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

import torch
from transformers import AutoModelForCausalLM

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("ERROR: zstandard not installed")
    sys.exit(1)


@dataclass
class ComparisonResult:
    step: int
    file_name: str
    original_size_mb: float

    # Delta values compression
    delta_compressed_mb: float
    delta_ratio: float

    # Actual weights compression
    actual_compressed_mb: float
    actual_ratio: float

    # Difference
    ratio_diff_pct: float


def load_delta_file(path: Path) -> Dict:
    """Load a delta .pt file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def extract_step_from_filename(filename: str) -> int:
    """Extract step number from filename."""
    import re
    match = re.search(r'delta_(\d+)\.pt', filename)
    return int(match.group(1)) if match else -1


def get_base_weights_at_indices(base_model: Dict[str, torch.Tensor],
                                 layer_name: str,
                                 indices: torch.Tensor) -> torch.Tensor:
    """Extract base model weights at sparse indices."""
    if layer_name not in base_model:
        # Try common name mappings
        return None

    base_weight = base_model[layer_name]

    # indices shape: (ndim, nnz) for COO format
    if indices.dim() == 2 and indices.shape[0] == 2:
        # 2D weight matrix
        row_idx, col_idx = indices[0], indices[1]
        return base_weight[row_idx, col_idx]
    elif indices.dim() == 1:
        return base_weight.flatten()[indices]
    else:
        # Handle other cases
        return None


def compress_data(data: bytes, level: int = 9) -> Tuple[bytes, float]:
    """Compress with zstd and return (compressed, time)."""
    cctx = zstd.ZstdCompressor(level=level)
    start = time.perf_counter()
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def serialize_sparse_data(indices: torch.Tensor, values: torch.Tensor) -> bytes:
    """Serialize sparse indices and values to bytes."""
    buffer = io.BytesIO()
    # Save as dict to mirror actual storage
    torch.save({
        'indices': indices,
        'values': values,
    }, buffer)
    return buffer.getvalue()


def compare_compression(delta_file: Dict, base_model: Dict[str, torch.Tensor]) -> Tuple[float, float, float, float]:
    """
    Compare compression of delta values vs actual weights.
    Returns: (delta_orig_size, delta_comp_size, actual_orig_size, actual_comp_size)
    """
    delta_data_list = []
    actual_data_list = []

    for layer_name, layer_data in delta_file.get('layers', {}).items():
        indices = layer_data['indices']
        delta_values = layer_data['values']

        # Get base weights at these indices
        base_values = get_base_weights_at_indices(base_model, layer_name, indices)

        if base_values is not None:
            # Compute actual weights: base + delta
            # Ensure same dtype
            if base_values.dtype != delta_values.dtype:
                if delta_values.dtype == torch.bfloat16:
                    base_values = base_values.to(torch.bfloat16)
                else:
                    base_values = base_values.to(delta_values.dtype)

            actual_values = base_values + delta_values
        else:
            # Fallback: use delta values as actual (for layers not in base)
            actual_values = delta_values

        # Serialize both
        delta_bytes = serialize_sparse_data(indices, delta_values)
        actual_bytes = serialize_sparse_data(indices, actual_values)

        delta_data_list.append(delta_bytes)
        actual_data_list.append(actual_bytes)

    # Combine all layers
    all_delta_bytes = b''.join(delta_data_list)
    all_actual_bytes = b''.join(actual_data_list)

    # Compress both
    delta_compressed, _ = compress_data(all_delta_bytes, level=9)
    actual_compressed, _ = compress_data(all_actual_bytes, level=9)

    return (
        len(all_delta_bytes),
        len(delta_compressed),
        len(all_actual_bytes),
        len(actual_compressed)
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B',
                       help='HuggingFace model to use as base')
    parser.add_argument('--delta-dir', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-iter1/checkpoints/deltas_math_instance0_seed42',
                       help='Directory containing delta files')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of delta files to sample')

    args = parser.parse_args()

    delta_dir = Path(args.delta_dir)

    # Check if delta directory exists, try alternatives
    if not delta_dir.exists():
        # Try other qwen 1.5b experiments
        alternatives = [
            '/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-iter8/checkpoints/deltas_math_instance0_seed42',
            '/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-iter16/checkpoints/deltas_math_instance0_seed42',
            '/root/grail/research/sparsity_analysis/experiments/qwen2.5-1.5b-lr5e-6/checkpoints/deltas_math_instance0_seed42',
        ]
        for alt in alternatives:
            if Path(alt).exists():
                delta_dir = Path(alt)
                print(f"Using alternative delta dir: {delta_dir}")
                break

    if not delta_dir.exists():
        print(f"ERROR: Delta directory not found: {delta_dir}")
        sys.exit(1)

    # Find all delta files
    delta_files = sorted(delta_dir.glob('delta_*.pt'))
    print(f"Found {len(delta_files)} delta files in {delta_dir}")

    if len(delta_files) == 0:
        print("ERROR: No delta files found")
        sys.exit(1)

    # Sample uniformly across steps
    steps = [extract_step_from_filename(f.name) for f in delta_files]
    min_step, max_step = min(steps), max(steps)
    target_steps = np.linspace(min_step, max_step, args.num_samples, dtype=int)

    # Find closest files to target steps
    step_to_file = {extract_step_from_filename(f.name): f for f in delta_files}
    available_steps = np.array(list(step_to_file.keys()))

    selected_files = []
    for target in target_steps:
        idx = np.argmin(np.abs(available_steps - target))
        closest_step = available_steps[idx]
        if step_to_file[closest_step] not in selected_files:
            selected_files.append(step_to_file[closest_step])

    print(f"\nSelected {len(selected_files)} files at steps: {[extract_step_from_filename(f.name) for f in selected_files]}")

    # Load base model
    print(f"\nLoading base model: {args.model}")
    print("(This may take a minute...)")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base_weights = {name: param.data.clone() for name, param in model.named_parameters()}
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"Loaded {len(base_weights)} weight tensors")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Proceeding with comparison using only delta values (no base weights)")
        base_weights = {}

    # Run comparison
    print("\n" + "="*80)
    print("COMPRESSION COMPARISON: Delta Values vs Actual Weights")
    print("="*80)
    print(f"{'Step':<8} {'File':<20} {'Delta Ratio':<14} {'Actual Ratio':<14} {'Diff %':<10}")
    print("-"*80)

    results = []

    for file_path in selected_files:
        step = extract_step_from_filename(file_path.name)

        try:
            delta = load_delta_file(file_path)

            delta_orig, delta_comp, actual_orig, actual_comp = compare_compression(delta, base_weights)

            delta_ratio = delta_orig / delta_comp if delta_comp > 0 else 0
            actual_ratio = actual_orig / actual_comp if actual_comp > 0 else 0
            diff_pct = ((actual_ratio - delta_ratio) / delta_ratio) * 100 if delta_ratio > 0 else 0

            result = ComparisonResult(
                step=step,
                file_name=file_path.name,
                original_size_mb=delta_orig / 1024 / 1024,
                delta_compressed_mb=delta_comp / 1024 / 1024,
                delta_ratio=delta_ratio,
                actual_compressed_mb=actual_comp / 1024 / 1024,
                actual_ratio=actual_ratio,
                ratio_diff_pct=diff_pct
            )
            results.append(result)

            print(f"{step:<8} {file_path.name:<20} {delta_ratio:<14.3f} {actual_ratio:<14.3f} {diff_pct:>+8.2f}%")

        except Exception as e:
            print(f"{step:<8} {file_path.name:<20} ERROR: {e}")

    # Summary
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        avg_delta_ratio = np.mean([r.delta_ratio for r in results])
        avg_actual_ratio = np.mean([r.actual_ratio for r in results])
        avg_diff_pct = np.mean([r.ratio_diff_pct for r in results])

        print(f"Average delta compression ratio:  {avg_delta_ratio:.3f}x")
        print(f"Average actual compression ratio: {avg_actual_ratio:.3f}x")
        print(f"Average difference: {avg_diff_pct:+.2f}%")
        print()

        if abs(avg_diff_pct) < 5:
            print("CONCLUSION: Difference is NEGLIGIBLE (<5%)")
            print("The current benchmark on delta values is representative.")
        elif abs(avg_diff_pct) < 15:
            print("CONCLUSION: Difference is MODERATE (5-15%)")
            print("Consider noting this in the paper, but results are still valid.")
        else:
            print("CONCLUSION: Difference is SIGNIFICANT (>15%)")
            print("Should re-run benchmark with actual weight values.")


if __name__ == '__main__':
    main()
