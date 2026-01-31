#!/usr/bin/env python3
"""
Benchmark delta encoding with ALL compression algorithms.

This fills the gap in our benchmarks by testing:
- Delta-encoded indices with all algorithms (lz4, snappy, zstd-1, zstd-3, zstd-9, gzip-6, brotli-1)
- Both sparse_coo (2D) and sparse_flat (1D) representations
- Comparison with raw (non-delta-encoded) for validation

Output: CSV file that can be merged with existing benchmark results.
"""

import os
import sys
import time
import csv
import gzip
import io
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import numpy as np

# Compression libraries
import lz4.frame as lz4f
import zstandard as zstd
import snappy
import brotli


def setup_logging(log_file: str = None):
    """Setup logging with immediate flush."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, force=True)
    sys.stdout.reconfigure(line_buffering=True)
    return logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    experiment: str
    model: str
    model_size: str
    source_file: str
    data_type: str  # sparse_delta
    raw_size_bytes: int
    num_elements: int
    sparsity: float
    representation: str  # sparse_coo, sparse_flat
    encoding: str  # raw, delta_encoded
    algorithm: str
    threads: int
    compressed_size_bytes: int
    compression_time_ms: float
    decompression_time_ms: float
    compression_ratio: float
    throughput_compress_mb_s: float
    throughput_decompress_mb_s: float


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================

def compress_data(data: bytes, algorithm: str, threads: int = 0) -> bytes:
    """Compress data with specified algorithm."""
    if algorithm == "lz4":
        return lz4f.compress(data)
    elif algorithm == "snappy":
        return snappy.compress(data)
    elif algorithm.startswith("zstd-"):
        level = int(algorithm.split("-")[1])
        cctx = zstd.ZstdCompressor(level=level, threads=threads if threads > 0 else -1)
        return cctx.compress(data)
    elif algorithm.startswith("gzip-"):
        level = int(algorithm.split("-")[1])
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=level) as f:
            f.write(data)
        return buf.getvalue()
    elif algorithm.startswith("brotli-"):
        quality = int(algorithm.split("-")[1])
        return brotli.compress(data, quality=quality)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def decompress_data(data: bytes, algorithm: str) -> bytes:
    """Decompress data with specified algorithm."""
    if algorithm == "lz4":
        return lz4f.decompress(data)
    elif algorithm == "snappy":
        return snappy.decompress(data)
    elif algorithm.startswith("zstd-"):
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    elif algorithm.startswith("gzip-"):
        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode='rb') as f:
            return f.read()
    elif algorithm.startswith("brotli-"):
        return brotli.decompress(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# =============================================================================
# DELTA ENCODING
# =============================================================================

def delta_encode_1d(indices: np.ndarray) -> np.ndarray:
    """Delta encode 1D indices (sorted, then diff)."""
    sorted_idx = np.sort(indices.flatten())
    return np.diff(sorted_idx, prepend=0).astype(np.int32)


def delta_encode_2d(indices: np.ndarray) -> np.ndarray:
    """Delta encode 2D indices (each dimension separately)."""
    if indices.ndim != 2:
        raise ValueError(f"Expected 2D array, got {indices.ndim}D")

    encoded = np.zeros_like(indices, dtype=np.int32)
    for dim in range(indices.shape[0]):
        row = indices[dim].astype(np.int64)
        sorted_row = np.sort(row)
        encoded[dim] = np.diff(sorted_row, prepend=0).astype(np.int32)
    return encoded


# =============================================================================
# SPARSE REPRESENTATIONS
# =============================================================================

def tensor_to_numpy_values(values: torch.Tensor) -> np.ndarray:
    """Convert tensor values to numpy, handling bfloat16."""
    if values.dtype == torch.bfloat16:
        # View as int16 to preserve exact bytes
        return values.view(torch.int16).numpy()
    elif values.dtype == torch.float16:
        return values.numpy()
    elif values.dtype == torch.float32:
        return values.numpy()
    else:
        return values.float().numpy()


def to_sparse_coo(delta: Dict) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """Convert delta to sparse COO format (2D indices)."""
    all_indices = []
    all_values = []
    shapes = []

    for name, layer in delta.get('layers', {}).items():
        indices = layer['indices']
        values = layer['values']
        shape = layer['shape']

        # Convert indices to numpy
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # Convert values to numpy (handle bfloat16)
        if isinstance(values, torch.Tensor):
            values = tensor_to_numpy_values(values)

        # Ensure 2D indices (row, col)
        if indices.ndim == 1:
            # Convert flat to 2D
            rows = indices // shape[1]
            cols = indices % shape[1]
            indices = np.stack([rows, cols], axis=0)

        all_indices.append(indices.astype(np.int32))
        all_values.append(values)
        shapes.append(shape)

    # Concatenate
    combined_indices = np.concatenate(all_indices, axis=1)
    combined_values = np.concatenate(all_values)

    return combined_indices, combined_values, shapes


def to_sparse_flat(delta: Dict) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """Convert delta to sparse flat format (1D indices)."""
    all_indices = []
    all_values = []
    shapes = []
    offset = 0

    for name, layer in delta.get('layers', {}).items():
        indices = layer['indices']
        values = layer['values']
        shape = layer['shape']

        # Convert indices to numpy
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        # Convert values to numpy (handle bfloat16)
        if isinstance(values, torch.Tensor):
            values = tensor_to_numpy_values(values)

        # Convert to flat indices
        if indices.ndim == 2:
            flat_indices = indices[0] * shape[1] + indices[1]
        else:
            flat_indices = indices

        # Add offset for layer concatenation
        all_indices.append((flat_indices + offset).astype(np.int64))
        all_values.append(values)
        shapes.append(shape)
        offset += np.prod(shape)

    combined_indices = np.concatenate(all_indices)
    combined_values = np.concatenate(all_values)

    return combined_indices, combined_values, shapes


def serialize_sparse(indices: np.ndarray, values: np.ndarray,
                     delta_encode: bool = False) -> bytes:
    """Serialize sparse data to bytes."""
    if delta_encode:
        if indices.ndim == 2:
            indices = delta_encode_2d(indices)
        else:
            indices = delta_encode_1d(indices)

    # Ensure consistent dtypes
    indices = indices.astype(np.int32)

    # Values are already in numpy format (int16 for bfloat16, or float32)
    # Just ensure contiguous array
    values = np.ascontiguousarray(values)

    # Concatenate indices and values
    return indices.tobytes() + values.tobytes()


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def get_algorithm_configs() -> List[Tuple[str, int]]:
    """Get all algorithm configurations to test."""
    configs = []

    # Fast algorithms (single-threaded only)
    configs.append(("lz4", 0))
    configs.append(("snappy", 0))

    # Zstandard - key levels, single-threaded only for fair comparison
    for level in [1, 3, 9]:
        configs.append((f"zstd-{level}", 0))

    # Gzip baseline
    configs.append(("gzip-6", 0))

    # Brotli fast
    configs.append(("brotli-1", 0))

    return configs


def benchmark_single_config(
    data: bytes,
    algorithm: str,
    threads: int,
    num_iterations: int = 3
) -> Tuple[int, float, float]:
    """Benchmark a single compression configuration."""
    # Compression
    compress_times = []
    compressed = None
    for _ in range(num_iterations):
        start = time.perf_counter()
        compressed = compress_data(data, algorithm, threads)
        compress_times.append(time.perf_counter() - start)

    # Decompression
    decompress_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        decompressed = decompress_data(compressed, algorithm)
        decompress_times.append(time.perf_counter() - start)

    # Verify lossless
    assert decompressed == data, f"Decompression mismatch for {algorithm}"

    return (
        len(compressed),
        np.median(compress_times) * 1000,  # ms
        np.median(decompress_times) * 1000  # ms
    )


def benchmark_file(
    delta_path: Path,
    experiment: str,
    model: str,
    model_size: str,
    algorithm_configs: List[Tuple[str, int]],
) -> List[Dict]:
    """Benchmark all configurations for a single delta file."""
    results = []

    # Load delta
    delta = torch.load(delta_path, map_location='cpu', weights_only=False)

    # Calculate sparsity
    total_params = 0
    total_nnz = 0
    for layer in delta.get('layers', {}).values():
        total_params += np.prod(layer['shape'])
        total_nnz += layer['nnz']
    sparsity = 1.0 - (total_nnz / total_params) if total_params > 0 else 0

    # Test configurations
    representations = [
        ('sparse_coo', to_sparse_coo),
        ('sparse_flat', to_sparse_flat),
    ]

    encodings = [
        ('raw', False),
        ('delta_encoded', True),
    ]

    for rep_name, rep_func in representations:
        try:
            indices, values, shapes = rep_func(delta)
        except Exception as e:
            logging.warning(f"Failed to convert to {rep_name}: {e}")
            continue

        for enc_name, do_delta_encode in encodings:
            try:
                data = serialize_sparse(indices.copy(), values.copy(), delta_encode=do_delta_encode)
            except Exception as e:
                logging.warning(f"Failed to serialize {rep_name}/{enc_name}: {e}")
                continue

            raw_size = len(data)

            for algorithm, threads in algorithm_configs:
                try:
                    comp_size, comp_time_ms, decomp_time_ms = benchmark_single_config(
                        data, algorithm, threads
                    )

                    compression_ratio = raw_size / comp_size if comp_size > 0 else 0
                    throughput_compress = (raw_size / 1024 / 1024) / (comp_time_ms / 1000) if comp_time_ms > 0 else 0
                    throughput_decompress = (raw_size / 1024 / 1024) / (decomp_time_ms / 1000) if decomp_time_ms > 0 else 0

                    results.append({
                        'experiment': experiment,
                        'model': model,
                        'model_size': model_size,
                        'source_file': delta_path.name,
                        'data_type': 'sparse_delta',
                        'raw_size_bytes': raw_size,
                        'num_elements': total_nnz,
                        'sparsity': sparsity,
                        'representation': rep_name,
                        'encoding': enc_name,
                        'algorithm': algorithm,
                        'threads': threads,
                        'compressed_size_bytes': comp_size,
                        'compression_time_ms': comp_time_ms,
                        'decompression_time_ms': decomp_time_ms,
                        'compression_ratio': compression_ratio,
                        'throughput_compress_mb_s': throughput_compress,
                        'throughput_decompress_mb_s': throughput_decompress,
                    })

                except Exception as e:
                    logging.warning(f"Failed {rep_name}/{enc_name}/{algorithm}: {e}")

    return results


def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    delta_path, experiment, model, model_size, algorithm_configs = args
    try:
        return benchmark_file(delta_path, experiment, model, model_size, algorithm_configs)
    except Exception as e:
        logging.error(f"Error processing {delta_path}: {e}")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark delta encoding with all algorithms')
    parser.add_argument('--base-path', type=str,
                        default='/root/grail/research/sparsity_analysis/experiments',
                        help='Base path to experiments')
    parser.add_argument('--output', type=str,
                        default='/root/grail/data/delta_encoding_all_algos.csv',
                        help='Output CSV file')
    parser.add_argument('--samples', type=int, default=20,
                        help='Number of samples per experiment')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--log', type=str,
                        default='/root/grail/logs/delta_encoding_all_algos.log',
                        help='Log file')
    args = parser.parse_args()

    # Setup logging
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log)

    logger.info("=" * 70)
    logger.info("Delta Encoding Benchmark - All Algorithms")
    logger.info("=" * 70)

    # Get algorithm configs
    algorithm_configs = get_algorithm_configs()
    logger.info(f"Algorithms: {[a[0] for a in algorithm_configs]}")
    logger.info(f"Representations: sparse_coo, sparse_flat")
    logger.info(f"Encodings: raw, delta_encoded")
    logger.info(f"Total configs per file: {len(algorithm_configs) * 2 * 2}")

    # Discover experiments
    base_path = Path(args.base_path)
    experiments = []

    # Structure: base_path/{experiment}/checkpoints/deltas_*/delta_*.pt
    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        experiment = exp_dir.name

        # Find delta files in checkpoints/deltas_*/
        delta_files = sorted(exp_dir.glob("**/delta_*.pt"))
        if not delta_files:
            continue

        # Sample files
        if len(delta_files) > args.samples:
            indices = np.linspace(0, len(delta_files) - 1, args.samples, dtype=int)
            delta_files = [delta_files[i] for i in indices]

        # Determine model and size from experiment name
        exp_lower = experiment.lower()
        if 'qwen' in exp_lower:
            model = 'qwen2.5'
        elif 'llama' in exp_lower:
            model = 'llama3.2'
        elif 'gemma' in exp_lower:
            model = 'gemma3'
        else:
            model = 'unknown'

        if '7b' in exp_lower:
            model_size = '7B'
        elif '4b' in exp_lower:
            model_size = '4B'
        elif '3b' in exp_lower:
            model_size = '3B'
        elif '1.5b' in exp_lower or '1_5b' in exp_lower:
            model_size = '1.5B'
        elif '0.5b' in exp_lower or '0_5b' in exp_lower:
            model_size = '0.5B'
        elif '1b' in exp_lower:
            model_size = '1B'
        else:
            model_size = 'Unknown'

        logger.info(f"  {experiment}: {len(delta_files)} files, model={model}, size={model_size}")

        for df in delta_files:
            experiments.append((df, experiment, model, model_size, algorithm_configs))

    logger.info(f"Found {len(experiments)} files to process")

    # Process files
    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_wrapper, exp): exp for exp in experiments}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            exp = futures[future]
            try:
                results = future.result()
                all_results.extend(results)

                if completed % 10 == 0 or completed == len(experiments):
                    logger.info(f"Progress: {completed}/{len(experiments)} files, {len(all_results)} results")

            except Exception as e:
                logger.error(f"Error: {e}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logger.info(f"Saved {len(all_results)} results to {output_path}")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        import pandas as pd
        df = pd.DataFrame(all_results)

        summary = df.groupby(['representation', 'encoding', 'algorithm']).agg({
            'compression_ratio': 'mean',
            'throughput_compress_mb_s': 'mean',
        }).round(2)

        logger.info("\nCompression ratios (mean):")
        for (rep, enc, algo), row in summary.iterrows():
            logger.info(f"  {rep:12s} {enc:14s} {algo:10s}: {row['compression_ratio']:6.2f}x @ {row['throughput_compress_mb_s']:7.1f} MB/s")
    else:
        logger.error("No results generated!")

    logger.info("\nBenchmark complete!")


if __name__ == '__main__':
    main()
