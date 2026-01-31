#!/usr/bin/env python3
"""
Compare compression performance: 2D COO vs Flat indices.

Tests whether converting to flat indices (like production) improves compression.
"""

import io
import time
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("Warning: zstandard not installed")


def delta_encode_indices_1d(indices: np.ndarray) -> np.ndarray:
    """Delta encode 1D sorted indices."""
    sorted_idx = np.sort(indices)
    return np.diff(sorted_idx, prepend=0)


def delta_encode_indices_2d(indices: np.ndarray) -> np.ndarray:
    """Delta encode 2D COO indices (each dimension separately)."""
    encoded = np.zeros_like(indices)
    for dim in range(indices.shape[0]):
        row = indices[dim]
        sort_order = np.argsort(row)
        sorted_row = row[sort_order]
        deltas = np.diff(sorted_row, prepend=0)
        encoded[dim] = deltas[np.argsort(sort_order)]
    return encoded


def compress_and_measure(data: bytes, level: int = 9) -> tuple[int, float]:
    """Compress data and return (compressed_size, time)."""
    compressor = zstd.ZstdCompressor(level=level)
    start = time.perf_counter()
    compressed = compressor.compress(data)
    elapsed = time.perf_counter() - start
    return len(compressed), elapsed


def benchmark_single_layer(indices_2d: torch.Tensor, values: torch.Tensor, shape: tuple) -> dict:
    """Benchmark a single layer in both formats."""
    results = {}

    # Original 2D COO format
    indices_2d_np = indices_2d.numpy().astype(np.int32)
    # BFloat16 can't be directly converted to numpy, view as int16
    if values.dtype == torch.bfloat16:
        values_np = values.view(torch.int16).numpy()
    else:
        values_np = values.numpy()

    # Convert to flat indices
    flat_indices = (indices_2d_np[0].astype(np.int64) * shape[1] + indices_2d_np[1]).astype(np.int32)

    nnz = len(values_np)

    # =========== RAW SIZE ===========
    raw_2d_indices = indices_2d_np.nbytes  # 2 * nnz * 4 bytes
    raw_flat_indices = flat_indices.nbytes  # nnz * 4 bytes
    raw_values = values_np.nbytes  # nnz * 2 bytes (bf16)

    results['nnz'] = nnz
    results['raw_2d_total'] = raw_2d_indices + raw_values
    results['raw_flat_total'] = raw_flat_indices + raw_values
    results['raw_reduction'] = 1 - results['raw_flat_total'] / results['raw_2d_total']

    # =========== ZSTD ONLY ===========
    # 2D COO
    data_2d = indices_2d_np.tobytes() + values_np.tobytes()
    comp_2d, _ = compress_and_measure(data_2d)
    results['zstd_2d'] = comp_2d
    results['zstd_2d_ratio'] = len(data_2d) / comp_2d

    # Flat
    data_flat = flat_indices.tobytes() + values_np.tobytes()
    comp_flat, _ = compress_and_measure(data_flat)
    results['zstd_flat'] = comp_flat
    results['zstd_flat_ratio'] = len(data_flat) / comp_flat

    # =========== DELTA ENCODE + ZSTD ===========
    # 2D COO delta encoded
    delta_2d = delta_encode_indices_2d(indices_2d_np)
    data_2d_delta = delta_2d.tobytes() + values_np.tobytes()
    comp_2d_delta, _ = compress_and_measure(data_2d_delta)
    results['delta_zstd_2d'] = comp_2d_delta
    results['delta_zstd_2d_ratio'] = len(data_2d) / comp_2d_delta  # ratio vs original raw

    # Flat delta encoded
    delta_flat = delta_encode_indices_1d(flat_indices)
    data_flat_delta = delta_flat.tobytes() + values_np.tobytes()
    comp_flat_delta, _ = compress_and_measure(data_flat_delta)
    results['delta_zstd_flat'] = comp_flat_delta
    results['delta_zstd_flat_ratio'] = len(data_flat) / comp_flat_delta  # ratio vs flat raw

    # Overall compression ratio (vs original 2D raw size)
    results['overall_2d_ratio'] = results['raw_2d_total'] / comp_2d_delta
    results['overall_flat_ratio'] = results['raw_2d_total'] / comp_flat_delta  # vs same baseline

    return results


def benchmark_delta_file(delta_path: Path) -> dict:
    """Benchmark entire delta file."""
    delta = torch.load(delta_path, map_location='cpu', weights_only=False)

    total_nnz = 0
    total_raw_2d = 0
    total_raw_flat = 0
    total_zstd_2d = 0
    total_zstd_flat = 0
    total_delta_zstd_2d = 0
    total_delta_zstd_flat = 0

    layers = delta.get('layers', {})

    for name, layer_data in layers.items():
        indices = layer_data['indices']
        values = layer_data['values']
        shape = layer_data['shape']

        # Only process 2D layers (most common)
        if indices.ndim == 2 and indices.shape[0] == 2 and len(shape) == 2:
            r = benchmark_single_layer(indices, values, shape)
            total_nnz += r['nnz']
            total_raw_2d += r['raw_2d_total']
            total_raw_flat += r['raw_flat_total']
            total_zstd_2d += r['zstd_2d']
            total_zstd_flat += r['zstd_flat']
            total_delta_zstd_2d += r['delta_zstd_2d']
            total_delta_zstd_flat += r['delta_zstd_flat']

    return {
        'file': delta_path.name,
        'total_nnz': total_nnz,
        'raw_2d_mb': total_raw_2d / 1e6,
        'raw_flat_mb': total_raw_flat / 1e6,
        'raw_reduction_pct': (1 - total_raw_flat / total_raw_2d) * 100,
        'zstd_2d_mb': total_zstd_2d / 1e6,
        'zstd_flat_mb': total_zstd_flat / 1e6,
        'zstd_2d_ratio': total_raw_2d / total_zstd_2d,
        'zstd_flat_ratio': total_raw_flat / total_zstd_flat,
        'delta_zstd_2d_mb': total_delta_zstd_2d / 1e6,
        'delta_zstd_flat_mb': total_delta_zstd_flat / 1e6,
        'delta_zstd_2d_ratio': total_raw_2d / total_delta_zstd_2d,
        'delta_zstd_flat_ratio': total_raw_flat / total_delta_zstd_flat,
        # Key metric: overall compression vs original 2D raw
        'overall_2d_ratio': total_raw_2d / total_delta_zstd_2d,
        'overall_flat_ratio': total_raw_2d / total_delta_zstd_flat,
    }


def main():
    import sys

    # Find sample delta files
    base_path = Path('/root/grail/research/sparsity_analysis/experiments')

    # Sample from different models
    sample_patterns = [
        'qwen2.5-1.5b-iter1/checkpoints/deltas_math_instance0_seed42/delta_000100.pt',
        'qwen2.5-7b-grpo-math-lr3e-06/seed42/deltas/delta_000100.pt',
        'llama3.2-3b-iter1/checkpoints/deltas_math_instance0_seed42/delta_000100.pt',
        'gemma3-4b-iter1/checkpoints/deltas_math_instance0_seed42/delta_000100.pt',
    ]

    print("=" * 80)
    print("FLAT vs 2D COO Compression Benchmark")
    print("=" * 80)
    print()

    all_results = []

    for pattern in sample_patterns:
        path = base_path / pattern
        if not path.exists():
            # Try alternative paths
            alt_path = list(base_path.glob(f"**/{path.name}"))
            if alt_path:
                path = alt_path[0]
            else:
                print(f"Skipping (not found): {pattern}")
                continue

        print(f"Processing: {path.parent.parent.parent.name}")
        r = benchmark_delta_file(path)
        all_results.append(r)

        print(f"  NNZ: {r['total_nnz']:,}")
        print(f"  Raw size:     2D={r['raw_2d_mb']:.2f} MB, Flat={r['raw_flat_mb']:.2f} MB ({r['raw_reduction_pct']:.1f}% smaller)")
        print(f"  zstd-9:       2D={r['zstd_2d_ratio']:.2f}x, Flat={r['zstd_flat_ratio']:.2f}x")
        print(f"  delta+zstd-9: 2D={r['delta_zstd_2d_ratio']:.2f}x, Flat={r['delta_zstd_flat_ratio']:.2f}x")
        print(f"  OVERALL (vs 2D raw): 2D={r['overall_2d_ratio']:.2f}x, Flat={r['overall_flat_ratio']:.2f}x")
        print()

    if all_results:
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        avg_2d = sum(r['overall_2d_ratio'] for r in all_results) / len(all_results)
        avg_flat = sum(r['overall_flat_ratio'] for r in all_results) / len(all_results)
        print(f"Average overall compression (vs 2D raw):")
        print(f"  2D COO + delta-encode + zstd-9:   {avg_2d:.2f}x")
        print(f"  Flat + delta-encode + zstd-9:     {avg_flat:.2f}x")
        print(f"  Flat advantage: {(avg_flat/avg_2d - 1)*100:.1f}% better" if avg_flat > avg_2d else f"  2D COO advantage: {(avg_2d/avg_flat - 1)*100:.1f}% better")


if __name__ == '__main__':
    main()
