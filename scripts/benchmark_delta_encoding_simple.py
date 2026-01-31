#!/usr/bin/env python3
"""
Simple delta encoding benchmark for all algorithms.
Matches v3 benchmark approach: only 2D layers, delta-encoded.
Tests both raw (no delta encoding) and delta-encoded indices.
"""

import os
import sys
import time
import csv
import gzip
import io
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np

# Compression libraries
import lz4.frame as lz4f
import zstandard as zstd
import snappy
import brotli


def setup_logging(log_file: str = None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='w'))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers, force=True)
    sys.stdout.reconfigure(line_buffering=True)
    return logging.getLogger(__name__)


# =============================================================================
# COMPRESSION
# =============================================================================

def compress_data(data: bytes, algorithm: str) -> bytes:
    if algorithm == "lz4":
        return lz4f.compress(data)
    elif algorithm == "snappy":
        return snappy.compress(data)
    elif algorithm.startswith("zstd-"):
        level = int(algorithm.split("-")[1])
        cctx = zstd.ZstdCompressor(level=level)
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
    raise ValueError(f"Unknown algorithm: {algorithm}")


def decompress_data(data: bytes, algorithm: str) -> bytes:
    if algorithm == "lz4":
        return lz4f.decompress(data)
    elif algorithm == "snappy":
        return snappy.decompress(data)
    elif algorithm.startswith("zstd-"):
        return zstd.ZstdDecompressor().decompress(data)
    elif algorithm.startswith("gzip-"):
        return gzip.decompress(data)
    elif algorithm.startswith("brotli-"):
        return brotli.decompress(data)
    raise ValueError(f"Unknown algorithm: {algorithm}")


# =============================================================================
# DELTA ENCODING
# =============================================================================

def delta_encode_2d(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delta encode 2D COO indices (lexicographically sorted)."""
    row_indices = indices[0]
    col_indices = indices[1]
    sort_order = np.lexsort((col_indices, row_indices))

    sorted_rows = row_indices[sort_order]
    sorted_cols = col_indices[sort_order]

    delta_rows = np.diff(sorted_rows, prepend=0)
    delta_cols = np.diff(sorted_cols, prepend=0)

    return np.stack([delta_rows, delta_cols], axis=0).astype(np.int32), sort_order


def delta_encode_1d(indices: np.ndarray) -> np.ndarray:
    """Delta encode sorted 1D indices."""
    sorted_idx = np.sort(indices)
    return np.diff(sorted_idx, prepend=0).astype(np.int32)


# =============================================================================
# BENCHMARK
# =============================================================================

ALGORITHMS = ["lz4", "snappy", "zstd-1", "zstd-3", "zstd-9", "gzip-6", "brotli-1"]


def benchmark_file(delta_path: Path, experiment: str) -> List[Dict]:
    """Benchmark a single delta file."""
    results = []

    # Load delta
    delta = torch.load(delta_path, map_location='cpu', weights_only=False)
    layers = delta.get('layers', {})

    # Collect 2D layers only (same as v3)
    all_indices_2d = []
    all_values = []
    all_shapes = []
    total_params = 0
    total_nnz = 0

    for name, layer_data in layers.items():
        indices = layer_data['indices']
        values = layer_data['values']
        shape = layer_data['shape']

        if len(shape) == 2:
            all_indices_2d.append(indices.numpy().astype(np.int32))
            if values.dtype == torch.bfloat16:
                all_values.append(values.view(torch.int16).numpy())
            else:
                all_values.append(values.numpy())
            all_shapes.append(shape)
            total_params += shape[0] * shape[1]
            total_nnz += len(values)

    if total_nnz == 0:
        return results

    sparsity = 1.0 - (total_nnz / total_params)

    # Parse model info
    exp_lower = experiment.lower()
    if "qwen" in exp_lower:
        model = "qwen2.5"
        model_size = "7B" if "7b" in exp_lower else "1.5B" if "1.5b" in exp_lower else "0.5B"
    elif "llama" in exp_lower:
        model = "llama3.2"
        model_size = "3B"
    elif "gemma" in exp_lower:
        model = "gemma3"
        model_size = "4B" if "4b" in exp_lower else "1B"
    else:
        model, model_size = "unknown", "unknown"

    # Concatenate
    indices_2d = np.concatenate(all_indices_2d, axis=1)
    values = np.concatenate(all_values)

    # ==========================================================================
    # SPARSE_COO (2D)
    # ==========================================================================

    # RAW (no delta encoding)
    raw_2d_data = indices_2d.astype(np.int32).tobytes() + values.tobytes()
    raw_2d_size = len(raw_2d_data)

    # DELTA ENCODED
    delta_2d, sort_order_2d = delta_encode_2d(indices_2d)
    values_sorted_2d = values[sort_order_2d]
    delta_2d_data = delta_2d.tobytes() + values_sorted_2d.tobytes()
    delta_2d_size = len(delta_2d_data)

    # ==========================================================================
    # SPARSE_FLAT (1D)
    # ==========================================================================

    # Convert to flat indices
    flat_indices_list = []
    for idx_2d, shape in zip(all_indices_2d, all_shapes):
        flat = (idx_2d[0].astype(np.int64) * shape[1] + idx_2d[1]).astype(np.int32)
        flat_indices_list.append(flat)
    flat_indices = np.concatenate(flat_indices_list)

    # RAW flat
    raw_flat_data = flat_indices.tobytes() + values.tobytes()
    raw_flat_size = len(raw_flat_data)

    # DELTA ENCODED flat
    delta_flat = delta_encode_1d(flat_indices)
    flat_sort_order = np.argsort(flat_indices)
    values_sorted_flat = values[flat_sort_order]
    delta_flat_data = delta_flat.tobytes() + values_sorted_flat.tobytes()
    delta_flat_size = len(delta_flat_data)

    # ==========================================================================
    # BENCHMARK ALL COMBINATIONS
    # ==========================================================================

    configs = [
        ('sparse_coo', 'raw', raw_2d_data, raw_2d_size),
        ('sparse_coo', 'delta_encoded', delta_2d_data, delta_2d_size),
        ('sparse_flat', 'raw', raw_flat_data, raw_flat_size),
        ('sparse_flat', 'delta_encoded', delta_flat_data, delta_flat_size),
    ]

    for representation, encoding, data, raw_size in configs:
        for algorithm in ALGORITHMS:
            try:
                # Compress
                start = time.perf_counter()
                compressed = compress_data(data, algorithm)
                compress_time = (time.perf_counter() - start) * 1000

                # Decompress
                start = time.perf_counter()
                decompressed = decompress_data(compressed, algorithm)
                decompress_time = (time.perf_counter() - start) * 1000

                # Verify
                assert decompressed == data, "Decompression mismatch"

                comp_size = len(compressed)
                ratio = raw_size / comp_size if comp_size > 0 else 0
                compress_speed = (raw_size / 1024 / 1024) / (compress_time / 1000) if compress_time > 0 else 0
                decompress_speed = (raw_size / 1024 / 1024) / (decompress_time / 1000) if decompress_time > 0 else 0

                results.append({
                    'experiment': experiment,
                    'model': model,
                    'model_size': model_size,
                    'source_file': delta_path.name,
                    'data_type': 'sparse_delta',
                    'raw_size_bytes': raw_size,
                    'num_elements': total_nnz,
                    'sparsity': sparsity,
                    'representation': representation,
                    'encoding': encoding,
                    'algorithm': algorithm,
                    'compressed_size_bytes': comp_size,
                    'compression_time_ms': compress_time,
                    'decompression_time_ms': decompress_time,
                    'compression_ratio': ratio,
                    'throughput_compress_mb_s': compress_speed,
                    'throughput_decompress_mb_s': decompress_speed,
                })
            except Exception as e:
                logging.warning(f"Failed {representation}/{encoding}/{algorithm}: {e}")

    return results


def process_wrapper(args):
    delta_path, experiment = args
    try:
        return benchmark_file(delta_path, experiment)
    except Exception as e:
        logging.error(f"Error processing {delta_path}: {e}")
        return []


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str,
                        default='/root/grail/research/sparsity_analysis/experiments')
    parser.add_argument('--output', type=str,
                        default='/root/grail/data/delta_encoding_all_algos.csv')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--log', type=str,
                        default='/root/grail/logs/delta_encoding_all_algos.log')
    args = parser.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log)

    logger.info("=" * 70)
    logger.info("Delta Encoding Benchmark - All Algorithms")
    logger.info("=" * 70)
    logger.info(f"Algorithms: {ALGORITHMS}")
    logger.info(f"Representations: sparse_coo, sparse_flat")
    logger.info(f"Encodings: raw, delta_encoded")
    logger.info(f"Total configs per file: {len(ALGORITHMS) * 2 * 2}")

    # Discover files
    base_path = Path(args.base_path)
    experiments = []

    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        delta_files = sorted(exp_dir.glob("**/delta_*.pt"))
        if not delta_files:
            continue

        if len(delta_files) > args.samples:
            indices = np.linspace(0, len(delta_files) - 1, args.samples, dtype=int)
            delta_files = [delta_files[i] for i in indices]

        logger.info(f"  {exp_dir.name}: {len(delta_files)} files")

        for df in delta_files:
            experiments.append((df, exp_dir.name))

    logger.info(f"Total files: {len(experiments)}")

    # Process
    all_results = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_wrapper, exp): exp for exp in experiments}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            results = future.result()
            all_results.extend(results)

            if completed % 20 == 0 or completed == len(experiments):
                logger.info(f"Progress: {completed}/{len(experiments)}, {len(all_results)} results")

    # Save
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        logger.info(f"Saved {len(all_results)} results to {output_path}")

        # Summary
        import pandas as pd
        df = pd.DataFrame(all_results)

        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)

        for rep in ['sparse_coo', 'sparse_flat']:
            logger.info(f"\n{rep}:")
            for enc in ['raw', 'delta_encoded']:
                logger.info(f"  {enc}:")
                subset = df[(df['representation'] == rep) & (df['encoding'] == enc)]
                for algo in ALGORITHMS:
                    algo_data = subset[subset['algorithm'] == algo]
                    if len(algo_data) > 0:
                        ratio = algo_data['compression_ratio'].mean()
                        speed = algo_data['throughput_compress_mb_s'].mean()
                        logger.info(f"    {algo:10s}: {ratio:6.2f}x @ {speed:7.1f} MB/s")

    logger.info("\nBenchmark complete!")


if __name__ == '__main__':
    main()
