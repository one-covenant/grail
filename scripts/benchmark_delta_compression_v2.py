#!/usr/bin/env python3
"""
Delta compression benchmark v2.
- Handles BFloat16 properly
- Saves per-file metrics to CSV
- Tests key lossless compression methods
"""

import os
import sys
import time
import zlib
import csv
import io
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime

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
    print("Warning: brotli not installed")


@dataclass
class CompressionMetrics:
    """Comprehensive metrics for a single compression test."""
    # File info
    model: str
    experiment: str
    file_name: str
    file_path: str
    step: int

    # Original data characteristics
    original_size_bytes: int
    num_layers: int
    total_nnz: int
    avg_sparsity: float

    # Compression method
    method: str

    # Results
    compressed_size_bytes: int
    compression_ratio: float
    compression_time_sec: float
    speed_mb_per_sec: float

    # Stream breakdown (if applicable)
    indices_size_bytes: int = 0
    values_size_bytes: int = 0
    metadata_size_bytes: int = 0
    indices_compressed_bytes: int = 0
    values_compressed_bytes: int = 0
    metadata_compressed_bytes: int = 0


def load_delta_file(path: Path) -> Dict:
    """Load a delta .pt file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def get_raw_bytes(delta: Dict) -> bytes:
    """Serialize delta to bytes, handling BFloat16."""
    # Convert BFloat16 tensors to Float32 for numpy compatibility
    converted_delta = convert_bf16_to_f32(delta)
    buffer = io.BytesIO()
    torch.save(converted_delta, buffer)
    return buffer.getvalue()


def convert_bf16_to_f32(delta: Dict) -> Dict:
    """Convert BFloat16 tensors to Float32 for numpy compatibility."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']
        values = layer_data['values']

        # Convert BFloat16 values to Float32
        if values.dtype == torch.bfloat16:
            values = values.to(torch.float32)

        new_delta['layers'][layer_name] = {
            'indices': indices,
            'values': values,
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
        }

    return new_delta


def get_delta_stats(delta: Dict) -> Tuple[int, int, float]:
    """Get statistics about a delta file."""
    num_layers = len(delta.get('layers', {}))
    total_nnz = 0
    sparsities = []

    for layer_name, layer_data in delta.get('layers', {}).items():
        nnz = layer_data.get('nnz', 0)
        total_nnz += nnz
        shape = layer_data.get('shape', ())
        total_params = 1
        for s in shape:
            total_params *= s
        if total_params > 0:
            sparsities.append(1.0 - (nnz / total_params))

    avg_sparsity = np.mean(sparsities) if sparsities else 0.0
    return num_layers, total_nnz, avg_sparsity


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
    """Downcast indices to smallest integer type."""
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
    """Separate indices, values, metadata into streams."""
    indices_buffer = io.BytesIO()
    values_buffer = io.BytesIO()
    metadata_buffer = io.BytesIO()

    metadata = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layer_info': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']
        values = layer_data['values']

        # Convert to numpy-compatible dtype
        if indices.dtype == torch.int32:
            indices_np = indices.numpy()
        else:
            indices_np = indices.to(torch.int32).numpy()

        if values.dtype == torch.bfloat16:
            values_np = values.to(torch.float32).numpy()
        else:
            values_np = values.numpy()

        metadata['layer_info'][layer_name] = {
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
            'indices_dtype': str(indices.dtype),
            'values_dtype': str(values.dtype),
        }

        indices_buffer.write(indices_np.tobytes())
        values_buffer.write(values_np.tobytes())

    pickle.dump(metadata, metadata_buffer)

    return indices_buffer.getvalue(), values_buffer.getvalue(), metadata_buffer.getvalue()


def delta_encode_indices(indices: np.ndarray) -> np.ndarray:
    """Delta encode indices for better compression."""
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


def apply_delta_encoding(delta: Dict) -> Dict:
    """Apply delta encoding to indices."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']

        # Convert to numpy, handling dtype
        if indices.dtype in (torch.int32, torch.int64):
            indices_np = indices.numpy()
        else:
            indices_np = indices.to(torch.int32).numpy()

        encoded = delta_encode_indices(indices_np)

        new_delta['layers'][layer_name] = {
            'indices': torch.from_numpy(encoded),
            'values': layer_data['values'],
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
        }

    return new_delta


def benchmark_single_file(
    file_path: Path,
    model: str,
    experiment: str
) -> List[CompressionMetrics]:
    """Benchmark all methods on a single file and return metrics."""

    metrics_list = []

    # Load delta
    delta = load_delta_file(file_path)
    step = delta.get('step', 0)

    # Get original bytes (with bf16->f32 conversion for fair comparison)
    raw_bytes = get_raw_bytes(delta)
    original_size = len(raw_bytes)

    # Get stats
    num_layers, total_nnz, avg_sparsity = get_delta_stats(delta)

    def make_metric(method: str, compressed_size: int, comp_time: float,
                   idx_size=0, val_size=0, meta_size=0,
                   idx_comp=0, val_comp=0, meta_comp=0) -> CompressionMetrics:
        return CompressionMetrics(
            model=model,
            experiment=experiment,
            file_name=file_path.name,
            file_path=str(file_path),
            step=step,
            original_size_bytes=original_size,
            num_layers=num_layers,
            total_nnz=total_nnz,
            avg_sparsity=avg_sparsity,
            method=method,
            compressed_size_bytes=compressed_size,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
            compression_time_sec=comp_time,
            speed_mb_per_sec=(original_size / 1024 / 1024) / comp_time if comp_time > 0 else 0,
            indices_size_bytes=idx_size,
            values_size_bytes=val_size,
            metadata_size_bytes=meta_size,
            indices_compressed_bytes=idx_comp,
            values_compressed_bytes=val_comp,
            metadata_compressed_bytes=meta_comp,
        )

    # === Test methods ===

    # 1. zstd at various levels
    if HAS_ZSTD:
        for level in [1, 3, 9, 19]:
            compressed, elapsed = compress_zstd(raw_bytes, level=level)
            metrics_list.append(make_metric(f'zstd-{level}', len(compressed), elapsed))

    # 2. brotli
    if HAS_BROTLI:
        for quality in [6, 9, 11]:
            compressed, elapsed = compress_brotli(raw_bytes, quality=quality)
            metrics_list.append(make_metric(f'brotli-{quality}', len(compressed), elapsed))

    # 3. gzip
    for level in [6, 9]:
        compressed, elapsed = compress_gzip(raw_bytes, level=level)
        metrics_list.append(make_metric(f'gzip-{level}', len(compressed), elapsed))

    # 4. Downcast indices + zstd
    if HAS_ZSTD:
        start = time.perf_counter()
        downcast_delta = downcast_indices(delta)
        downcast_bytes = get_raw_bytes(downcast_delta)
        prep_time = time.perf_counter() - start

        compressed, comp_time = compress_zstd(downcast_bytes, level=3)
        metrics_list.append(make_metric('downcast+zstd-3', len(compressed), prep_time + comp_time))

        compressed, comp_time = compress_zstd(downcast_bytes, level=19)
        metrics_list.append(make_metric('downcast+zstd-19', len(compressed), prep_time + comp_time))

    # 5. Separate streams + zstd
    start = time.perf_counter()
    idx_stream, val_stream, meta_stream = separate_streams(delta)
    sep_time = time.perf_counter() - start

    idx_size, val_size, meta_size = len(idx_stream), len(val_stream), len(meta_stream)

    if HAS_ZSTD:
        idx_comp, t1 = compress_zstd(idx_stream, level=3)
        val_comp, t2 = compress_zstd(val_stream, level=3)
        meta_comp, t3 = compress_zstd(meta_stream, level=3)
        total_comp = len(idx_comp) + len(val_comp) + len(meta_comp)
        metrics_list.append(make_metric(
            'separate+zstd-3', total_comp, sep_time + t1 + t2 + t3,
            idx_size, val_size, meta_size,
            len(idx_comp), len(val_comp), len(meta_comp)
        ))

        idx_comp, t1 = compress_zstd(idx_stream, level=19)
        val_comp, t2 = compress_zstd(val_stream, level=19)
        meta_comp, t3 = compress_zstd(meta_stream, level=19)
        total_comp = len(idx_comp) + len(val_comp) + len(meta_comp)
        metrics_list.append(make_metric(
            'separate+zstd-19', total_comp, sep_time + t1 + t2 + t3,
            idx_size, val_size, meta_size,
            len(idx_comp), len(val_comp), len(meta_comp)
        ))

    # 6. Delta encoding + zstd
    if HAS_ZSTD:
        start = time.perf_counter()
        delta_enc = apply_delta_encoding(delta)
        delta_enc_bytes = get_raw_bytes(delta_enc)
        enc_time = time.perf_counter() - start

        compressed, comp_time = compress_zstd(delta_enc_bytes, level=3)
        metrics_list.append(make_metric('delta-enc+zstd-3', len(compressed), enc_time + comp_time))

        compressed, comp_time = compress_zstd(delta_enc_bytes, level=19)
        metrics_list.append(make_metric('delta-enc+zstd-19', len(compressed), enc_time + comp_time))

    # 7. Combined: downcast + delta-enc + separate + zstd
    if HAS_ZSTD:
        start = time.perf_counter()
        combined = downcast_indices(delta)
        combined = apply_delta_encoding(combined)
        idx_s, val_s, meta_s = separate_streams(combined)
        comb_time = time.perf_counter() - start

        idx_comp, t1 = compress_zstd(idx_s, level=19)
        val_comp, t2 = compress_zstd(val_s, level=19)
        meta_comp, t3 = compress_zstd(meta_s, level=19)
        total_comp = len(idx_comp) + len(val_comp) + len(meta_comp)
        metrics_list.append(make_metric(
            'combined+zstd-19', total_comp, comb_time + t1 + t2 + t3,
            len(idx_s), len(val_s), len(meta_s),
            len(idx_comp), len(val_comp), len(meta_comp)
        ))

    return metrics_list


def discover_experiments(base_path: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Discover experiments grouped by model and experiment name."""
    experiments = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(base_path):
        delta_files = sorted([f for f in files if f.startswith('delta_') and f.endswith('.pt')])
        if delta_files:
            rel_path = Path(root).relative_to(base_path)
            rel_str = str(rel_path).lower()

            # Identify model
            if 'qwen' in rel_str:
                if '7b' in rel_str:
                    model = 'qwen2.5-7b'
                elif '0.5b' in rel_str:
                    model = 'qwen2.5-0.5b'
                else:
                    model = 'qwen2.5-1.5b'
            elif 'llama' in rel_str:
                model = 'llama3.2-3b'
            elif 'gemma' in rel_str:
                if '4b' in rel_str:
                    model = 'gemma3-4b'
                else:
                    model = 'gemma3-1b'
            else:
                model = 'unknown'

            # Get experiment name (first directory component)
            exp_name = rel_path.parts[0] if rel_path.parts else 'unknown'

            for f in delta_files:
                experiments[model][exp_name].append(Path(root) / f)

    return experiments


def run_benchmark(
    base_path: Path,
    output_csv: Path,
    samples_per_experiment: int = 3
):
    """Run benchmark and save results to CSV."""

    print(f"Discovering experiments in {base_path}...")
    experiments = discover_experiments(base_path)

    # Count totals
    total_models = len(experiments)
    total_experiments = sum(len(exps) for exps in experiments.values())
    total_files = sum(
        len(files)
        for model_exps in experiments.values()
        for files in model_exps.values()
    )

    print(f"\nFound {total_models} models, {total_experiments} experiments, {total_files} total files")
    for model, exps in sorted(experiments.items()):
        print(f"  {model}:")
        for exp, files in sorted(exps.items()):
            print(f"    {exp}: {len(files)} files")

    # Calculate estimated time
    total_samples = sum(
        min(samples_per_experiment, len(files))
        for model_exps in experiments.values()
        for files in model_exps.values()
    )
    print(f"\nWill process {total_samples} files × ~15 methods = {total_samples * 15} compression tests")
    print(f"Estimated time: ~{total_samples * 2} minutes (varies by file size)\n")

    # Open CSV for writing
    all_metrics = []

    csv_fields = [
        'model', 'experiment', 'file_name', 'file_path', 'step',
        'original_size_bytes', 'num_layers', 'total_nnz', 'avg_sparsity',
        'method', 'compressed_size_bytes', 'compression_ratio',
        'compression_time_sec', 'speed_mb_per_sec',
        'indices_size_bytes', 'values_size_bytes', 'metadata_size_bytes',
        'indices_compressed_bytes', 'values_compressed_bytes', 'metadata_compressed_bytes'
    ]

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for model, model_exps in sorted(experiments.items()):
            print(f"\n{'='*70}")
            print(f"Model: {model}")
            print('='*70)

            for exp_name, files in sorted(model_exps.items()):
                print(f"\n  Experiment: {exp_name} ({len(files)} files)")

                # Sample files
                if len(files) <= samples_per_experiment:
                    sample_files = files
                else:
                    step = len(files) // samples_per_experiment
                    sample_files = [files[i * step] for i in range(samples_per_experiment)]

                for i, file_path in enumerate(sample_files):
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    print(f"\n    [{i+1}/{len(sample_files)}] {file_path.name} ({size_mb:.1f} MB)")

                    try:
                        start_time = time.perf_counter()
                        metrics = benchmark_single_file(file_path, model, exp_name)
                        elapsed = time.perf_counter() - start_time

                        # Write to CSV immediately
                        for m in metrics:
                            writer.writerow(asdict(m))
                        f.flush()

                        all_metrics.extend(metrics)

                        # Print summary
                        baseline = next((m for m in metrics if m.method == 'zstd-3'), None)
                        best = max(metrics, key=lambda m: m.compression_ratio)

                        if baseline:
                            print(f"      Baseline (zstd-3): {baseline.compression_ratio:.3f}x @ {baseline.speed_mb_per_sec:.1f} MB/s")
                        print(f"      Best: {best.method} = {best.compression_ratio:.3f}x @ {best.speed_mb_per_sec:.1f} MB/s")
                        print(f"      Time for all methods: {elapsed:.1f}s")

                    except Exception as e:
                        print(f"      ERROR: {e}")
                        import traceback
                        traceback.print_exc()

    print(f"\n\nResults saved to: {output_csv}")
    print(f"Total metrics recorded: {len(all_metrics)}")

    return all_metrics


def print_summary_tables(csv_path: Path):
    """Print summary tables from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    print("\n" + "="*100)
    print("SUMMARY TABLE 1: AVERAGE COMPRESSION RATIO BY MODEL AND METHOD")
    print("="*100)

    pivot = df.pivot_table(
        values='compression_ratio',
        index='method',
        columns='model',
        aggfunc='mean'
    ).round(3)
    pivot['AVG'] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values('AVG', ascending=False)
    print(pivot.to_string())

    print("\n" + "="*100)
    print("SUMMARY TABLE 2: AVERAGE SPEED (MB/s) BY MODEL AND METHOD")
    print("="*100)

    pivot_speed = df.pivot_table(
        values='speed_mb_per_sec',
        index='method',
        columns='model',
        aggfunc='mean'
    ).round(1)
    pivot_speed['AVG'] = pivot_speed.mean(axis=1).round(1)
    # Keep same order as ratio table
    pivot_speed = pivot_speed.reindex(pivot.index)
    print(pivot_speed.to_string())

    print("\n" + "="*80)
    print("SUMMARY TABLE 3: OVERALL RANKINGS")
    print("="*80)

    summary = df.groupby('method').agg({
        'compression_ratio': 'mean',
        'speed_mb_per_sec': 'mean'
    }).round(3)
    summary['efficiency'] = (summary['compression_ratio'] * np.sqrt(summary['speed_mb_per_sec'])).round(2)
    summary = summary.sort_values('compression_ratio', ascending=False)
    summary.columns = ['Avg Ratio', 'Avg Speed (MB/s)', 'Efficiency Score']
    print(summary.to_string())

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_ratio = summary['Avg Ratio'].idxmax()
    best_speed = summary['Avg Speed (MB/s)'].idxmax()
    best_efficiency = summary['Efficiency Score'].idxmax()

    print(f"• Best compression:  {best_ratio} ({summary.loc[best_ratio, 'Avg Ratio']:.3f}x)")
    print(f"• Fastest:           {best_speed} ({summary.loc[best_speed, 'Avg Speed (MB/s)']:.1f} MB/s)")
    print(f"• Best balance:      {best_efficiency} (efficiency: {summary.loc[best_efficiency, 'Efficiency Score']:.2f})")

    baseline = summary.loc['zstd-3', 'Avg Ratio'] if 'zstd-3' in summary.index else None
    if baseline:
        improvement = (summary.loc[best_ratio, 'Avg Ratio'] / baseline - 1) * 100
        print(f"\n• Best method is {improvement:.1f}% better than zstd-3 baseline")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Delta compression benchmark v2')
    parser.add_argument('--base-path', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments',
                       help='Base path to experiments')
    parser.add_argument('--samples', type=int, default=3,
                       help='Samples per experiment')
    parser.add_argument('--output', type=str,
                       default='/root/grail/scripts/compression_metrics.csv',
                       help='Output CSV path')
    parser.add_argument('--nice', type=int, default=10)
    parser.add_argument('--max-threads', type=int, default=2)
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary from existing CSV')

    args = parser.parse_args()

    if args.summary_only:
        print_summary_tables(Path(args.output))
        return

    # Set priority
    if args.nice > 0:
        try:
            os.nice(args.nice)
            print(f"Nice value: {args.nice}")
        except:
            pass

    os.environ['OMP_NUM_THREADS'] = str(args.max_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.max_threads)
    torch.set_num_threads(args.max_threads)
    print(f"Max threads: {args.max_threads}")
    print(f"Output CSV: {args.output}")
    print(f"Started at: {datetime.now().isoformat()}\n")

    base_path = Path(args.base_path)
    output_csv = Path(args.output)

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    run_benchmark(base_path, output_csv, samples_per_experiment=args.samples)

    # Print summary
    try:
        print_summary_tables(output_csv)
    except Exception as e:
        print(f"Could not print summary: {e}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
