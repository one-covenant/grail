#!/usr/bin/env python3
"""
Light version of delta compression benchmark.
Tests only the most promising methods for faster results.
"""

import os
import sys
import time
import zlib
import io
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("Warning: zstandard not installed")

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


@dataclass
class CompressionResult:
    method: str
    original_size: int
    compressed_size: int
    compression_time: float

    @property
    def ratio(self) -> float:
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 0

    @property
    def speed_mb_s(self) -> float:
        return (self.original_size / 1024 / 1024) / self.compression_time if self.compression_time > 0 else 0


def load_delta_file(path: Path) -> Dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def get_raw_bytes(delta: Dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(delta, buffer)
    return buffer.getvalue()


def compress_zstd(data: bytes, level: int = 3) -> Tuple[bytes, float]:
    if not HAS_ZSTD:
        return data, 0.0
    cctx = zstd.ZstdCompressor(level=level)
    start = time.perf_counter()
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_brotli(data: bytes, quality: int = 6) -> Tuple[bytes, float]:
    if not HAS_BROTLI:
        return data, 0.0
    start = time.perf_counter()
    compressed = brotli.compress(data, quality=quality)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_gzip(data: bytes, level: int = 6) -> Tuple[bytes, float]:
    start = time.perf_counter()
    compressed = zlib.compress(data, level=level)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def downcast_indices(delta: Dict) -> Dict:
    """Downcast indices to smallest possible integer type per layer."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']
        shape = layer_data['shape']
        max_dim = max(shape) if shape else 0

        if max_dim <= 255:
            new_dtype = torch.uint8
        elif max_dim <= 65535:
            new_dtype = torch.int16
        else:
            new_dtype = torch.int32

        new_delta['layers'][layer_name] = {
            'indices': indices.to(new_dtype),
            'values': layer_data['values'],
            'shape': shape,
            'nnz': layer_data['nnz'],
        }

    return new_delta


def separate_streams(delta: Dict) -> Tuple[bytes, bytes, bytes]:
    """Separate indices, values, and metadata into different byte streams."""
    indices_buffer = io.BytesIO()
    values_buffer = io.BytesIO()
    metadata_buffer = io.BytesIO()

    metadata = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layer_info': {}
    }

    indices_list = []
    values_list = []

    for layer_name, layer_data in delta.get('layers', {}).items():
        metadata['layer_info'][layer_name] = {
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
            'indices_dtype': str(layer_data['indices'].dtype),
            'values_dtype': str(layer_data['values'].dtype),
        }
        indices_list.append(layer_data['indices'].numpy().tobytes())
        values_list.append(layer_data['values'].numpy().tobytes())

    pickle.dump(metadata, metadata_buffer)

    for idx_bytes in indices_list:
        indices_buffer.write(idx_bytes)

    for val_bytes in values_list:
        values_buffer.write(val_bytes)

    return indices_buffer.getvalue(), values_buffer.getvalue(), metadata_buffer.getvalue()


def delta_encode_indices(indices: np.ndarray) -> np.ndarray:
    """Delta encode sorted indices for better compression."""
    if len(indices) == 0:
        return indices

    if indices.ndim == 2:
        encoded = np.zeros_like(indices)
        for dim in range(indices.shape[0]):
            row = indices[dim]
            sorted_idx = np.argsort(row)
            sorted_row = row[sorted_idx]
            deltas = np.diff(sorted_row, prepend=0)
            encoded[dim] = deltas
        return encoded
    else:
        sorted_indices = np.sort(indices)
        return np.diff(sorted_indices, prepend=0)


def apply_delta_encoding_to_delta(delta: Dict) -> Dict:
    """Apply delta encoding to all indices in a delta file."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices'].numpy()
        encoded_indices = delta_encode_indices(indices)

        new_delta['layers'][layer_name] = {
            'indices': torch.from_numpy(encoded_indices),
            'values': layer_data['values'],
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
        }

    return new_delta


def benchmark_single_file(file_path: Path) -> Dict[str, CompressionResult]:
    """Benchmark key compression methods on a single file."""
    results = {}

    delta = load_delta_file(file_path)
    raw_bytes = get_raw_bytes(delta)
    original_size = len(raw_bytes)

    # 1. Baseline: zstd-3 (standard)
    if HAS_ZSTD:
        compressed, elapsed = compress_zstd(raw_bytes, level=3)
        results['zstd-3'] = CompressionResult('zstd-3', original_size, len(compressed), elapsed)

        # zstd-19 (high compression)
        compressed, elapsed = compress_zstd(raw_bytes, level=19)
        results['zstd-19'] = CompressionResult('zstd-19', original_size, len(compressed), elapsed)

    # 2. brotli-9
    if HAS_BROTLI:
        compressed, elapsed = compress_brotli(raw_bytes, quality=9)
        results['brotli-9'] = CompressionResult('brotli-9', original_size, len(compressed), elapsed)

    # 3. gzip-9
    compressed, elapsed = compress_gzip(raw_bytes, level=9)
    results['gzip-9'] = CompressionResult('gzip-9', original_size, len(compressed), elapsed)

    # 4. Downcast indices + zstd
    start = time.perf_counter()
    downcast_delta = downcast_indices(delta)
    downcast_bytes = get_raw_bytes(downcast_delta)
    downcast_time = time.perf_counter() - start

    if HAS_ZSTD:
        compressed, compress_time = compress_zstd(downcast_bytes, level=3)
        results['downcast+zstd-3'] = CompressionResult(
            'downcast+zstd-3', original_size, len(compressed), downcast_time + compress_time
        )

        compressed, compress_time = compress_zstd(downcast_bytes, level=19)
        results['downcast+zstd-19'] = CompressionResult(
            'downcast+zstd-19', original_size, len(compressed), downcast_time + compress_time
        )

    # 5. Separate streams + zstd
    start = time.perf_counter()
    idx_stream, val_stream, meta_stream = separate_streams(delta)
    separate_time = time.perf_counter() - start

    if HAS_ZSTD:
        idx_comp, t1 = compress_zstd(idx_stream, level=3)
        val_comp, t2 = compress_zstd(val_stream, level=3)
        meta_comp, t3 = compress_zstd(meta_stream, level=3)
        total_compressed = len(idx_comp) + len(val_comp) + len(meta_comp)
        results['separate+zstd-3'] = CompressionResult(
            'separate+zstd-3', original_size, total_compressed, separate_time + t1 + t2 + t3
        )

        idx_comp, t1 = compress_zstd(idx_stream, level=19)
        val_comp, t2 = compress_zstd(val_stream, level=19)
        meta_comp, t3 = compress_zstd(meta_stream, level=19)
        total_compressed = len(idx_comp) + len(val_comp) + len(meta_comp)
        results['separate+zstd-19'] = CompressionResult(
            'separate+zstd-19', original_size, total_compressed, separate_time + t1 + t2 + t3
        )

    # 6. Delta encoding + zstd
    start = time.perf_counter()
    delta_encoded = apply_delta_encoding_to_delta(delta)
    delta_enc_bytes = get_raw_bytes(delta_encoded)
    delta_enc_time = time.perf_counter() - start

    if HAS_ZSTD:
        compressed, compress_time = compress_zstd(delta_enc_bytes, level=3)
        results['delta-enc+zstd-3'] = CompressionResult(
            'delta-enc+zstd-3', original_size, len(compressed), delta_enc_time + compress_time
        )

    # 7. Combined: downcast + delta encoding + separate streams + zstd-19
    start = time.perf_counter()
    combined_delta = downcast_indices(delta)
    combined_delta = apply_delta_encoding_to_delta(combined_delta)
    idx_stream, val_stream, meta_stream = separate_streams(combined_delta)
    combined_time = time.perf_counter() - start

    if HAS_ZSTD:
        idx_comp, t1 = compress_zstd(idx_stream, level=19)
        val_comp, t2 = compress_zstd(val_stream, level=19)
        meta_comp, t3 = compress_zstd(meta_stream, level=19)
        total_compressed = len(idx_comp) + len(val_comp) + len(meta_comp)
        results['combined+zstd-19'] = CompressionResult(
            'combined+zstd-19', original_size, total_compressed, combined_time + t1 + t2 + t3
        )

    return results


def discover_experiments(base_path: Path) -> Dict[str, List[Path]]:
    """Discover experiments grouped by model."""
    experiments = defaultdict(list)

    for root, dirs, files in os.walk(base_path):
        delta_files = sorted([f for f in files if f.startswith('delta_') and f.endswith('.pt')])
        if delta_files:
            rel_path = str(Path(root).relative_to(base_path)).lower()

            # Identify model
            if 'qwen' in rel_path:
                if '7b' in rel_path:
                    model = 'qwen2.5-7b'
                elif '0.5b' in rel_path:
                    model = 'qwen2.5-0.5b'
                else:
                    model = 'qwen2.5-1.5b'
            elif 'llama' in rel_path:
                model = 'llama3.2-3b'
            elif 'gemma' in rel_path:
                if '4b' in rel_path:
                    model = 'gemma3-4b'
                else:
                    model = 'gemma3-1b'
            else:
                model = 'unknown'

            for f in delta_files:
                experiments[model].append(Path(root) / f)

    return experiments


def run_benchmark(base_path: Path, samples_per_model: int = 3):
    """Run benchmark across all models."""

    print(f"Discovering experiments in {base_path}...")
    experiments = discover_experiments(base_path)

    print(f"\nFound {len(experiments)} model groups:")
    for model, files in sorted(experiments.items()):
        print(f"  {model}: {len(files)} files")

    all_results = {}

    for model, files in sorted(experiments.items()):
        print(f"\n{'='*70}")
        print(f"Benchmarking {model} ({len(files)} files, sampling {min(samples_per_model, len(files))})")
        print('='*70)

        # Sample files
        if len(files) <= samples_per_model:
            sample_files = files
        else:
            step = len(files) // samples_per_model
            sample_files = [files[i * step] for i in range(samples_per_model)]

        model_results = defaultdict(list)

        for i, file_path in enumerate(sample_files):
            print(f"\n  [{i+1}/{len(sample_files)}] {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")

            try:
                results = benchmark_single_file(file_path)

                for method, result in results.items():
                    model_results[method].append(result)

                # Print quick summary
                if 'zstd-3' in results:
                    baseline = results['zstd-3']
                    print(f"      zstd-3 (baseline): {baseline.ratio:.2f}x @ {baseline.speed_mb_s:.1f} MB/s")

                best = max(results.values(), key=lambda x: x.ratio)
                print(f"      Best: {best.method} = {best.ratio:.2f}x @ {best.speed_mb_s:.1f} MB/s")

            except Exception as e:
                print(f"      ERROR: {e}")

        all_results[model] = model_results

    return all_results


def print_results_table(results: Dict[str, Dict[str, List[CompressionResult]]]):
    """Print formatted results tables."""

    all_methods = set()
    for model_results in results.values():
        all_methods.update(model_results.keys())
    all_methods = sorted(all_methods)

    models = sorted(results.keys())

    # Table 1: Compression Ratio
    print("\n" + "="*100)
    print("TABLE 1: COMPRESSION RATIO BY MODEL (higher is better)")
    print("="*100)

    header = f"{'Method':<22}"
    for model in models:
        header += f" {model[:11]:>11}"
    header += f" {'AVG':>8}"
    print(header)
    print("-"*100)

    summary_data = []
    for method in all_methods:
        row = f"{method:<22}"
        ratios = []
        for model in models:
            if method in results[model]:
                avg_ratio = np.mean([r.ratio for r in results[model][method]])
                row += f" {avg_ratio:>11.3f}"
                ratios.append(avg_ratio)
            else:
                row += f" {'N/A':>11}"

        if ratios:
            avg = np.mean(ratios)
            row += f" {avg:>8.3f}"
            summary_data.append((method, avg, ratios))
        print(row)

    # Table 2: Compression Speed
    print("\n" + "="*100)
    print("TABLE 2: COMPRESSION SPEED BY MODEL (MB/s, higher is better)")
    print("="*100)

    header = f"{'Method':<22}"
    for model in models:
        header += f" {model[:11]:>11}"
    header += f" {'AVG':>8}"
    print(header)
    print("-"*100)

    speed_data = []
    for method in all_methods:
        row = f"{method:<22}"
        speeds = []
        for model in models:
            if method in results[model]:
                avg_speed = np.mean([r.speed_mb_s for r in results[model][method]])
                row += f" {avg_speed:>11.1f}"
                speeds.append(avg_speed)
            else:
                row += f" {'N/A':>11}"

        if speeds:
            avg = np.mean(speeds)
            row += f" {avg:>8.1f}"
            speed_data.append((method, avg))
        print(row)

    # Table 3: Aggregated Summary
    print("\n" + "="*80)
    print("TABLE 3: AGGREGATED SUMMARY (sorted by compression ratio)")
    print("="*80)
    print(f"{'Method':<22} {'Avg Ratio':>12} {'Avg Speed':>12} {'Score':>10}")
    print(f"{'':22} {'(higher)':>12} {'(MB/s)':>12} {'(ratio×√speed)':>10}")
    print("-"*80)

    # Combine ratio and speed data
    speed_dict = {m: s for m, s in speed_data}
    combined = []
    for method, ratio, _ in summary_data:
        speed = speed_dict.get(method, 0)
        score = ratio * np.sqrt(speed) if speed > 0 else 0
        combined.append((method, ratio, speed, score))

    combined.sort(key=lambda x: x[1], reverse=True)

    for method, ratio, speed, score in combined:
        print(f"{method:<22} {ratio:>12.3f} {speed:>12.1f} {score:>10.2f}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_ratio = max(combined, key=lambda x: x[1])
    best_speed = max(combined, key=lambda x: x[2])
    best_score = max(combined, key=lambda x: x[3])

    print(f"• Best compression ratio: {best_ratio[0]} ({best_ratio[1]:.3f}x)")
    print(f"• Fastest compression:    {best_speed[0]} ({best_speed[2]:.1f} MB/s)")
    print(f"• Best balance:           {best_score[0]} (score: {best_score[3]:.2f})")

    # Improvement over baseline
    baseline_ratio = next((r for m, r, _, _ in combined if m == 'zstd-3'), None)
    if baseline_ratio and best_ratio[1] > baseline_ratio:
        improvement = (best_ratio[1] / baseline_ratio - 1) * 100
        print(f"\n• Best method achieves {improvement:.1f}% better compression than zstd-3 baseline")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Light delta compression benchmark')
    parser.add_argument('--base-path', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments',
                       help='Base path to experiments')
    parser.add_argument('--samples', type=int, default=3,
                       help='Number of files to sample per model')
    parser.add_argument('--nice', type=int, default=10,
                       help='Nice value (0-19)')
    parser.add_argument('--max-threads', type=int, default=2,
                       help='Max threads')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON')

    args = parser.parse_args()

    # Set low priority
    if args.nice > 0:
        try:
            os.nice(args.nice)
            print(f"Set nice value to {args.nice}")
        except:
            pass

    # Limit threads
    os.environ['OMP_NUM_THREADS'] = str(args.max_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.max_threads)
    torch.set_num_threads(args.max_threads)
    print(f"Limited to {args.max_threads} threads\n")

    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    results = run_benchmark(base_path, samples_per_model=args.samples)
    print_results_table(results)

    if args.output:
        import json
        output_data = {}
        for model, model_results in results.items():
            output_data[model] = {}
            for method, method_results in model_results.items():
                output_data[model][method] = {
                    'avg_ratio': float(np.mean([r.ratio for r in method_results])),
                    'avg_speed_mb_s': float(np.mean([r.speed_mb_s for r in method_results])),
                }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
