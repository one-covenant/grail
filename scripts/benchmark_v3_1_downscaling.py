#!/usr/bin/env python3
"""Benchmark V3.1 sparse codec (index downscaling) and append to existing CSV."""

import csv
import time
from pathlib import Path

import torch

from grail.infrastructure.sparse_codec import (
    encode_sparse_delta_v3_1,
    decode_sparse_delta_v3_1,
)


# Experiments directory
EXPERIMENTS_DIR = Path("/root/grail/research/sparsity_analysis/experiments")
OUTPUT_CSV = Path("/root/grail/data/delta_encoding_all_algos.csv")

# Model info mapping
MODEL_INFO = {
    "gemma3-4b-iter1": ("gemma3", "4B"),
    "llama3.2-3b-iter1": ("llama3.2", "3B"),
    "qwen2.5-1.5b-iter16": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-iter32": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-iter8": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-lr1e-6": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-lr5e-6": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-lr5e-7": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-sft-math-lr2e-05": ("qwen2.5", "1.5B"),
    "qwen2.5-1.5b-sft-math-lr3e-06": ("qwen2.5", "1.5B"),
    "qwen2.5-7b-grpo-math-lr3e-06": ("qwen2.5", "7B"),
}


def load_delta_file(path: Path) -> tuple[dict, dict, int, int, float]:
    """Load delta file and return sparse tensors, shapes, and stats."""
    delta = torch.load(path, map_location="cpu", weights_only=True)
    layers = delta.get("layers", {})

    sparse_tensors = {}
    shapes = {}
    total_elements = 0
    total_params = 0

    for name, layer_data in layers.items():
        indices = layer_data["indices"]
        values = layer_data["values"]
        shape = list(layer_data["shape"])

        sparse_tensors[f"{name}.indices"] = indices
        sparse_tensors[f"{name}.values"] = values
        shapes[name] = shape

        total_elements += values.numel()
        # Calculate total params from shape
        params = 1
        for dim in shape:
            params *= dim
        total_params += params

    sparsity = 1.0 - (total_elements / total_params) if total_params > 0 else 0.0

    return sparse_tensors, shapes, total_elements, total_params, sparsity


def compute_raw_size(sparse_tensors: dict) -> int:
    """Compute raw size of sparse tensors (COO format with int32 indices)."""
    total = 0
    for key, tensor in sparse_tensors.items():
        if key.endswith(".indices"):
            # 2D COO indices: shape [2, nnz], int32 = 4 bytes
            total += tensor.numel() * 4
        elif key.endswith(".values"):
            total += tensor.numel() * tensor.element_size()
    return total


def benchmark_file(path: Path, experiment: str) -> dict:
    """Benchmark a single delta file with V3.1 codec."""
    # Load data
    sparse_tensors, shapes, num_elements, total_params, sparsity = load_delta_file(path)

    if not sparse_tensors:
        return None

    raw_size = compute_raw_size(sparse_tensors)

    # Warm up
    _ = encode_sparse_delta_v3_1(sparse_tensors, shapes)

    # Benchmark encode (3 iterations)
    encode_times = []
    compressed = None
    for _ in range(3):
        start = time.perf_counter()
        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        encode_times.append((time.perf_counter() - start) * 1000)  # ms

    # Benchmark decode (3 iterations)
    decode_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = decode_sparse_delta_v3_1(compressed)
        decode_times.append((time.perf_counter() - start) * 1000)  # ms

    compress_time_ms = sum(encode_times) / len(encode_times)
    decompress_time_ms = sum(decode_times) / len(decode_times)
    compressed_size = len(compressed)

    compression_ratio = raw_size / compressed_size if compressed_size > 0 else 1.0
    throughput_compress = (raw_size / 1e6) / (compress_time_ms / 1000) if compress_time_ms > 0 else 0
    throughput_decompress = (raw_size / 1e6) / (decompress_time_ms / 1000) if decompress_time_ms > 0 else 0

    model, model_size = MODEL_INFO.get(experiment, ("unknown", "unknown"))

    return {
        "experiment": experiment,
        "model": model,
        "model_size": model_size,
        "source_file": path.name,
        "data_type": "sparse_delta",
        "raw_size_bytes": raw_size,
        "num_elements": num_elements,
        "sparsity": sparsity,
        "representation": "sparse_coo",
        "encoding": "delta_encoded_downscaled",
        "algorithm": "zstd-1",
        "compressed_size_bytes": compressed_size,
        "compression_time_ms": compress_time_ms,
        "decompression_time_ms": decompress_time_ms,
        "compression_ratio": compression_ratio,
        "throughput_compress_mb_s": throughput_compress,
        "throughput_decompress_mb_s": throughput_decompress,
    }


def main():
    print("=" * 70)
    print("V3.1 Benchmark - Index Downscaling (uint8 rows + uint16 cols)")
    print("=" * 70)
    print()

    # Find all experiment directories
    if not EXPERIMENTS_DIR.exists():
        print(f"Experiments directory not found: {EXPERIMENTS_DIR}")
        return

    experiments = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(experiments)} experiments")

    # Collect all delta files
    all_files = []
    for exp_dir in experiments:
        delta_dir = exp_dir / "seed9999" / "deltas"
        if not delta_dir.exists():
            delta_dir = exp_dir / "seed42" / "deltas"
        if not delta_dir.exists():
            # Try direct deltas subdirectory
            delta_dir = exp_dir / "deltas"

        if delta_dir.exists():
            files = sorted(delta_dir.glob("delta_*.pt"))[:20]  # Max 20 per experiment
            for f in files:
                all_files.append((exp_dir.name, f))

    print(f"Total files to benchmark: {len(all_files)}")
    print()

    # Run benchmarks
    results = []
    for i, (experiment, file_path) in enumerate(all_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"Progress: {i + 1}/{len(all_files)}")

        try:
            result = benchmark_file(file_path, experiment)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    print()
    print(f"Completed: {len(results)} results")

    # Append to existing CSV
    if results:
        fieldnames = list(results[0].keys())

        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Don't write header since we're appending
            writer.writerows(results)

        print(f"Appended {len(results)} rows to {OUTPUT_CSV}")

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        avg_ratio = sum(r["compression_ratio"] for r in results) / len(results)
        avg_compress_tp = sum(r["throughput_compress_mb_s"] for r in results) / len(results)
        avg_decompress_tp = sum(r["throughput_decompress_mb_s"] for r in results) / len(results)

        print(f"  Average compression ratio: {avg_ratio:.2f}x")
        print(f"  Average compress throughput: {avg_compress_tp:.1f} MB/s")
        print(f"  Average decompress throughput: {avg_decompress_tp:.1f} MB/s")

        # Group by experiment
        print()
        print("Per-experiment compression ratios:")
        by_exp = {}
        for r in results:
            exp = r["experiment"]
            if exp not in by_exp:
                by_exp[exp] = []
            by_exp[exp].append(r["compression_ratio"])

        for exp in sorted(by_exp.keys()):
            ratios = by_exp[exp]
            avg = sum(ratios) / len(ratios)
            print(f"  {exp}: {avg:.2f}x ({len(ratios)} files)")


if __name__ == "__main__":
    main()
