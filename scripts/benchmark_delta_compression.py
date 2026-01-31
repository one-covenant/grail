#!/usr/bin/env python3
"""
Benchmark various lossless compression approaches for delta files.

Tests:
1. Generic compressors: zstd, brotli, lzma, gzip at various levels
2. Index optimizations: downcast, delta encoding, separate streams
3. Format conversions: COO to CSR
4. Dictionary compression: layer-specific zstd dictionaries
5. Cross-step compression: temporal delta encoding
"""

import os
import sys
import time
import zlib
import lzma
import struct
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import io
import pickle

import torch
import numpy as np

# Optional imports with fallbacks
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False
    print("Warning: brotli not installed. Run: pip install brotli")

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("Warning: zstandard not installed. Run: pip install zstandard")


@dataclass
class CompressionResult:
    """Result of a compression benchmark."""
    method: str
    original_size: int
    compressed_size: int
    compression_time: float
    decompression_time: float = 0.0

    @property
    def ratio(self) -> float:
        return self.original_size / self.compressed_size if self.compressed_size > 0 else 0

    @property
    def speed_mb_s(self) -> float:
        return (self.original_size / 1024 / 1024) / self.compression_time if self.compression_time > 0 else 0


@dataclass
class ModelBenchmarkResults:
    """Results for a single model."""
    model_name: str
    num_files: int
    total_original_size: int
    results: Dict[str, List[CompressionResult]] = field(default_factory=dict)


def load_delta_file(path: Path) -> Dict:
    """Load a delta .pt file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def get_raw_bytes(delta: Dict) -> bytes:
    """Serialize delta to raw bytes for compression."""
    buffer = io.BytesIO()
    torch.save(delta, buffer)
    return buffer.getvalue()


def extract_tensors_only(delta: Dict) -> bytes:
    """Extract just the tensor data (indices + values) as raw bytes."""
    buffer = io.BytesIO()
    tensor_data = {}
    for layer_name, layer_data in delta.get('layers', {}).items():
        tensor_data[layer_name] = {
            'indices': layer_data['indices'].numpy().tobytes(),
            'values': layer_data['values'].numpy().tobytes(),
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
        }
    pickle.dump(tensor_data, buffer)
    return buffer.getvalue()


# =============================================================================
# Generic Compressors
# =============================================================================

def compress_zstd(data: bytes, level: int = 3) -> Tuple[bytes, float]:
    """Compress with zstandard."""
    if not HAS_ZSTD:
        return data, 0.0
    cctx = zstd.ZstdCompressor(level=level)
    start = time.perf_counter()
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_zstd_dict(data: bytes, dictionary: bytes, level: int = 3) -> Tuple[bytes, float]:
    """Compress with zstandard using a pre-trained dictionary."""
    if not HAS_ZSTD:
        return data, 0.0
    dict_data = zstd.ZstdCompressionDict(dictionary)
    cctx = zstd.ZstdCompressor(level=level, dict_data=dict_data)
    start = time.perf_counter()
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_brotli(data: bytes, quality: int = 6) -> Tuple[bytes, float]:
    """Compress with brotli."""
    if not HAS_BROTLI:
        return data, 0.0
    start = time.perf_counter()
    compressed = brotli.compress(data, quality=quality)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_lzma(data: bytes, preset: int = 6) -> Tuple[bytes, float]:
    """Compress with LZMA."""
    start = time.perf_counter()
    compressed = lzma.compress(data, preset=preset)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def compress_gzip(data: bytes, level: int = 6) -> Tuple[bytes, float]:
    """Compress with gzip/zlib."""
    start = time.perf_counter()
    compressed = zlib.compress(data, level=level)
    elapsed = time.perf_counter() - start
    return compressed, elapsed


# =============================================================================
# Index Optimization Techniques
# =============================================================================

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

        # Find max value needed for each dimension
        max_dim = max(shape) if shape else 0

        # Choose smallest dtype that fits
        if max_dim <= 255:
            new_dtype = torch.uint8
        elif max_dim <= 65535:
            new_dtype = torch.int16
        else:
            new_dtype = torch.int32  # Keep as-is

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

    # Metadata stream
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

    # Flatten multi-dim indices to 1D for delta encoding
    if indices.ndim == 2:
        # COO format: (ndim, nnz) - encode each dimension separately
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
            'delta_encoded': True,
        }

    return new_delta


# =============================================================================
# Cross-Step Compression
# =============================================================================

def compute_index_diff(prev_delta: Dict, curr_delta: Dict) -> Dict:
    """Compute difference in sparsity patterns between consecutive deltas."""
    diff_delta = {
        'step': curr_delta.get('step'),
        'timestamp': curr_delta.get('timestamp'),
        'metadata': curr_delta.get('metadata'),
        'layers': {},
        'is_diff': True,
    }

    for layer_name, curr_layer in curr_delta.get('layers', {}).items():
        prev_layer = prev_delta.get('layers', {}).get(layer_name)

        if prev_layer is None:
            # No previous layer, store full
            diff_delta['layers'][layer_name] = curr_layer
            continue

        curr_indices = curr_layer['indices']
        prev_indices = prev_layer['indices']

        # Convert to sets of tuples for comparison
        curr_set = set(map(tuple, curr_indices.T.tolist())) if curr_indices.numel() > 0 else set()
        prev_set = set(map(tuple, prev_indices.T.tolist())) if prev_indices.numel() > 0 else set()

        # Find added and removed indices
        added = curr_set - prev_set
        removed = prev_set - curr_set
        unchanged = curr_set & prev_set

        # Store compact diff representation
        diff_delta['layers'][layer_name] = {
            'added_indices': list(added) if added else [],
            'removed_indices': list(removed) if removed else [],
            'unchanged_count': len(unchanged),
            'values': curr_layer['values'],  # Still need all values
            'shape': curr_layer['shape'],
            'nnz': curr_layer['nnz'],
        }

    return diff_delta


# =============================================================================
# COO to CSR Conversion
# =============================================================================

def coo_to_csr(delta: Dict) -> Dict:
    """Convert COO format to CSR format for potentially better compression."""
    from scipy import sparse

    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {},
        'format': 'csr',
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices'].numpy()
        values = layer_data['values'].numpy()
        shape = layer_data['shape']

        if len(shape) != 2 or indices.shape[0] != 2:
            # Not a 2D layer, keep as COO
            new_delta['layers'][layer_name] = layer_data
            continue

        # Create COO matrix and convert to CSR
        coo = sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        csr = coo.tocsr()

        new_delta['layers'][layer_name] = {
            'indptr': torch.from_numpy(csr.indptr.astype(np.int32)),
            'indices': torch.from_numpy(csr.indices.astype(np.int32)),
            'data': torch.from_numpy(csr.data),
            'shape': shape,
            'nnz': layer_data['nnz'],
        }

    return new_delta


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_single_file(file_path: Path, prev_delta: Optional[Dict] = None) -> Dict[str, CompressionResult]:
    """Benchmark all compression methods on a single file."""
    results = {}

    # Load delta
    delta = load_delta_file(file_path)
    raw_bytes = get_raw_bytes(delta)
    original_size = len(raw_bytes)

    # --- Generic compressors on raw bytes ---

    # zstd at various levels
    if HAS_ZSTD:
        for level in [1, 3, 9, 19, 22]:
            compressed, elapsed = compress_zstd(raw_bytes, level=level)
            results[f'zstd-{level}'] = CompressionResult(
                method=f'zstd-{level}',
                original_size=original_size,
                compressed_size=len(compressed),
                compression_time=elapsed
            )

    # brotli at various levels
    if HAS_BROTLI:
        for quality in [4, 6, 9, 11]:
            compressed, elapsed = compress_brotli(raw_bytes, quality=quality)
            results[f'brotli-{quality}'] = CompressionResult(
                method=f'brotli-{quality}',
                original_size=original_size,
                compressed_size=len(compressed),
                compression_time=elapsed
            )

    # lzma at various presets
    for preset in [3, 6, 9]:
        compressed, elapsed = compress_lzma(raw_bytes, preset=preset)
        results[f'lzma-{preset}'] = CompressionResult(
            method=f'lzma-{preset}',
            original_size=original_size,
            compressed_size=len(compressed),
            compression_time=elapsed
        )

    # gzip
    for level in [6, 9]:
        compressed, elapsed = compress_gzip(raw_bytes, level=level)
        results[f'gzip-{level}'] = CompressionResult(
            method=f'gzip-{level}',
            original_size=original_size,
            compressed_size=len(compressed),
            compression_time=elapsed
        )

    # --- Index optimizations ---

    # Downcast indices + zstd
    start = time.perf_counter()
    downcast_delta = downcast_indices(delta)
    downcast_bytes = get_raw_bytes(downcast_delta)
    downcast_time = time.perf_counter() - start

    if HAS_ZSTD:
        compressed, compress_time = compress_zstd(downcast_bytes, level=3)
        results['downcast+zstd-3'] = CompressionResult(
            method='downcast+zstd-3',
            original_size=original_size,
            compressed_size=len(compressed),
            compression_time=downcast_time + compress_time
        )

    # Separate streams + zstd
    start = time.perf_counter()
    idx_stream, val_stream, meta_stream = separate_streams(delta)
    separate_time = time.perf_counter() - start

    if HAS_ZSTD:
        idx_comp, t1 = compress_zstd(idx_stream, level=3)
        val_comp, t2 = compress_zstd(val_stream, level=3)
        meta_comp, t3 = compress_zstd(meta_stream, level=3)
        total_compressed = len(idx_comp) + len(val_comp) + len(meta_comp)
        results['separate+zstd-3'] = CompressionResult(
            method='separate+zstd-3',
            original_size=original_size,
            compressed_size=total_compressed,
            compression_time=separate_time + t1 + t2 + t3
        )

        # Higher compression on separated streams
        idx_comp, t1 = compress_zstd(idx_stream, level=19)
        val_comp, t2 = compress_zstd(val_stream, level=19)
        meta_comp, t3 = compress_zstd(meta_stream, level=19)
        total_compressed = len(idx_comp) + len(val_comp) + len(meta_comp)
        results['separate+zstd-19'] = CompressionResult(
            method='separate+zstd-19',
            original_size=original_size,
            compressed_size=total_compressed,
            compression_time=separate_time + t1 + t2 + t3
        )

    # Delta encoding on indices + zstd
    start = time.perf_counter()
    delta_encoded = apply_delta_encoding_to_delta(delta)
    delta_enc_bytes = get_raw_bytes(delta_encoded)
    delta_enc_time = time.perf_counter() - start

    if HAS_ZSTD:
        compressed, compress_time = compress_zstd(delta_enc_bytes, level=3)
        results['delta-enc+zstd-3'] = CompressionResult(
            method='delta-enc+zstd-3',
            original_size=original_size,
            compressed_size=len(compressed),
            compression_time=delta_enc_time + compress_time
        )

    # Downcast + delta encoding + separate streams + zstd (combined best)
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
            method='combined+zstd-19',
            original_size=original_size,
            compressed_size=total_compressed,
            compression_time=combined_time + t1 + t2 + t3
        )

    # --- COO to CSR ---
    try:
        start = time.perf_counter()
        csr_delta = coo_to_csr(delta)
        csr_bytes = get_raw_bytes(csr_delta)
        csr_time = time.perf_counter() - start

        if HAS_ZSTD:
            compressed, compress_time = compress_zstd(csr_bytes, level=3)
            results['csr+zstd-3'] = CompressionResult(
                method='csr+zstd-3',
                original_size=original_size,
                compressed_size=len(compressed),
                compression_time=csr_time + compress_time
            )
    except Exception as e:
        print(f"  CSR conversion failed: {e}")

    # --- Cross-step diff (if previous delta available) ---
    if prev_delta is not None:
        start = time.perf_counter()
        diff_delta = compute_index_diff(prev_delta, delta)
        diff_bytes = get_raw_bytes(diff_delta)
        diff_time = time.perf_counter() - start

        if HAS_ZSTD:
            compressed, compress_time = compress_zstd(diff_bytes, level=3)
            results['cross-step+zstd-3'] = CompressionResult(
                method='cross-step+zstd-3',
                original_size=original_size,
                compressed_size=len(compressed),
                compression_time=diff_time + compress_time
            )

    return results, delta


def discover_experiments(base_path: Path) -> Dict[str, List[Path]]:
    """Discover all experiments and their delta files."""
    experiments = defaultdict(list)

    # Walk through directory structure
    for root, dirs, files in os.walk(base_path):
        delta_files = sorted([f for f in files if f.startswith('delta_') and f.endswith('.pt')])
        if delta_files:
            # Extract experiment name from path
            rel_path = Path(root).relative_to(base_path)
            parts = rel_path.parts

            # Try to identify model name
            if 'qwen' in str(rel_path).lower():
                if '7b' in str(rel_path).lower():
                    model = 'qwen2.5-7b'
                elif '1.5b' in str(rel_path).lower() or '1_5b' in str(rel_path).lower():
                    model = 'qwen2.5-1.5b'
                elif '0.5b' in str(rel_path).lower():
                    model = 'qwen2.5-0.5b'
                else:
                    model = 'qwen2.5'
            elif 'llama' in str(rel_path).lower():
                model = 'llama3.2-3b'
            elif 'gemma' in str(rel_path).lower():
                if '4b' in str(rel_path).lower():
                    model = 'gemma3-4b'
                elif '1b' in str(rel_path).lower():
                    model = 'gemma3-1b'
                else:
                    model = 'gemma3'
            else:
                model = parts[0] if parts else 'unknown'

            for f in delta_files:
                experiments[model].append(Path(root) / f)

    return experiments


def run_benchmark(
    base_path: Path,
    samples_per_model: int = 5,
    include_cross_step: bool = True
) -> Dict[str, ModelBenchmarkResults]:
    """Run full benchmark across all models."""

    print(f"Discovering experiments in {base_path}...")
    experiments = discover_experiments(base_path)

    print(f"Found {len(experiments)} models:")
    for model, files in experiments.items():
        print(f"  {model}: {len(files)} files")

    all_results = {}

    for model, files in experiments.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking {model} ({len(files)} files, sampling {samples_per_model})")
        print('='*60)

        # Sample files evenly across the range
        if len(files) <= samples_per_model:
            sample_files = files
        else:
            step = len(files) // samples_per_model
            sample_files = [files[i * step] for i in range(samples_per_model)]

        model_results = ModelBenchmarkResults(
            model_name=model,
            num_files=len(files),
            total_original_size=0,
            results=defaultdict(list)
        )

        prev_delta = None
        for i, file_path in enumerate(sample_files):
            print(f"\n  [{i+1}/{len(sample_files)}] {file_path.name}")

            try:
                results, delta = benchmark_single_file(
                    file_path,
                    prev_delta if include_cross_step else None
                )

                for method, result in results.items():
                    model_results.results[method].append(result)
                    if i == 0:
                        model_results.total_original_size += result.original_size

                # Print quick summary
                if 'zstd-3' in results:
                    baseline = results['zstd-3']
                    print(f"    Baseline (zstd-3): {baseline.ratio:.2f}x @ {baseline.speed_mb_s:.1f} MB/s")

                    best_ratio = max(results.values(), key=lambda x: x.ratio)
                    print(f"    Best ratio: {best_ratio.method} = {best_ratio.ratio:.2f}x")

                if include_cross_step:
                    prev_delta = delta

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        all_results[model] = model_results

    return all_results


def print_results_table(results: Dict[str, ModelBenchmarkResults]):
    """Print formatted results tables."""

    # Collect all methods
    all_methods = set()
    for model_results in results.values():
        all_methods.update(model_results.results.keys())
    all_methods = sorted(all_methods)

    # Table 1: Compression Ratio per Model
    print("\n" + "="*100)
    print("COMPRESSION RATIO BY MODEL (higher is better)")
    print("="*100)

    # Header
    header = f"{'Method':<25}"
    for model in results.keys():
        header += f" {model[:12]:>12}"
    header += f" {'Avg':>10}"
    print(header)
    print("-"*100)

    method_avgs = {}
    for method in all_methods:
        row = f"{method:<25}"
        ratios = []
        for model, model_results in results.items():
            if method in model_results.results:
                avg_ratio = np.mean([r.ratio for r in model_results.results[method]])
                row += f" {avg_ratio:>12.2f}"
                ratios.append(avg_ratio)
            else:
                row += f" {'N/A':>12}"

        if ratios:
            avg = np.mean(ratios)
            method_avgs[method] = avg
            row += f" {avg:>10.2f}"
        print(row)

    # Table 2: Compression Speed per Model
    print("\n" + "="*100)
    print("COMPRESSION SPEED BY MODEL (MB/s, higher is better)")
    print("="*100)

    header = f"{'Method':<25}"
    for model in results.keys():
        header += f" {model[:12]:>12}"
    header += f" {'Avg':>10}"
    print(header)
    print("-"*100)

    for method in all_methods:
        row = f"{method:<25}"
        speeds = []
        for model, model_results in results.items():
            if method in model_results.results:
                avg_speed = np.mean([r.speed_mb_s for r in model_results.results[method]])
                row += f" {avg_speed:>12.1f}"
                speeds.append(avg_speed)
            else:
                row += f" {'N/A':>12}"

        if speeds:
            row += f" {np.mean(speeds):>10.1f}"
        print(row)

    # Table 3: Aggregated Summary
    print("\n" + "="*80)
    print("AGGREGATED SUMMARY (sorted by compression ratio)")
    print("="*80)
    print(f"{'Method':<25} {'Avg Ratio':>12} {'Avg Speed':>12} {'Efficiency':>12}")
    print(f"{'':25} {'(higher=better)':>12} {'(MB/s)':>12} {'(ratio*speed)':>12}")
    print("-"*80)

    summary = []
    for method in all_methods:
        ratios = []
        speeds = []
        for model_results in results.values():
            if method in model_results.results:
                ratios.extend([r.ratio for r in model_results.results[method]])
                speeds.extend([r.speed_mb_s for r in model_results.results[method]])

        if ratios and speeds:
            avg_ratio = np.mean(ratios)
            avg_speed = np.mean(speeds)
            efficiency = avg_ratio * np.log10(avg_speed + 1)  # Balance ratio and speed
            summary.append((method, avg_ratio, avg_speed, efficiency))

    # Sort by ratio
    summary.sort(key=lambda x: x[1], reverse=True)

    for method, ratio, speed, efficiency in summary:
        print(f"{method:<25} {ratio:>12.2f} {speed:>12.1f} {efficiency:>12.2f}")

    # Best practices recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Best ratio
    best_ratio = max(summary, key=lambda x: x[1])
    print(f"Best compression ratio: {best_ratio[0]} ({best_ratio[1]:.2f}x)")

    # Best speed
    best_speed = max(summary, key=lambda x: x[2])
    print(f"Fastest compression: {best_speed[0]} ({best_speed[2]:.1f} MB/s)")

    # Best balance
    best_efficiency = max(summary, key=lambda x: x[3])
    print(f"Best balance (ratio*log(speed)): {best_efficiency[0]}")

    # Practical recommendations
    print("\nPractical recommendations:")
    print("  - For archival (max compression): Use the best ratio method")
    print("  - For frequent access (balance): Use the best efficiency method")
    print("  - For real-time (max speed): Use the fastest method")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark delta compression methods')
    parser.add_argument('--base-path', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments',
                       help='Base path to experiments')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of files to sample per model')
    parser.add_argument('--no-cross-step', action='store_true',
                       help='Disable cross-step compression benchmarks')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--nice', type=int, default=10,
                       help='Nice value for low priority (0-19, higher=nicer)')
    parser.add_argument('--max-threads', type=int, default=2,
                       help='Max threads to use (limits CPU interference)')

    args = parser.parse_args()

    # Set low priority to avoid interfering with other experiments
    if args.nice > 0:
        try:
            os.nice(args.nice)
            print(f"Set process nice value to {args.nice} (low priority)")
        except Exception as e:
            print(f"Warning: Could not set nice value: {e}")

    # Limit threads for numpy/torch operations
    os.environ['OMP_NUM_THREADS'] = str(args.max_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.max_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.max_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.max_threads)
    torch.set_num_threads(args.max_threads)
    print(f"Limited to {args.max_threads} threads")

    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    results = run_benchmark(
        base_path,
        samples_per_model=args.samples,
        include_cross_step=not args.no_cross_step
    )

    print_results_table(results)

    if args.output:
        import json

        # Convert to serializable format
        output_data = {}
        for model, model_results in results.items():
            output_data[model] = {
                'num_files': model_results.num_files,
                'methods': {}
            }
            for method, method_results in model_results.results.items():
                output_data[model]['methods'][method] = {
                    'avg_ratio': np.mean([r.ratio for r in method_results]),
                    'avg_speed_mb_s': np.mean([r.speed_mb_s for r in method_results]),
                    'samples': len(method_results)
                }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
