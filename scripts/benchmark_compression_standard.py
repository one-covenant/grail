#!/usr/bin/env python3
"""
Compression Benchmark - Standard (Publication Quality)

A rigorous benchmark for sparse delta checkpoint compression.

Design Decisions:
=================

1. BASELINE: Original .pt file size
   - This is what we're replacing, so it's the meaningful comparison point
   - All compression ratios are: original_pt_size / compressed_size

2. REPRESENTATIONS TESTED:
   - raw_coo: Unsorted COO indices (int32) + values, no preprocessing
   - raw_flat: Unsorted flat indices (int32) + values, no preprocessing
   - delta_coo_int32: Sorted + delta-encoded COO (int32), per-layer
   - delta_flat_int32: Sorted + delta-encoded flat (int32), per-layer
   - delta_coo_downscaled: Sorted + delta-encoded COO (uint8/uint16), per-layer

   Note: delta_flat_int32 + zstd-1 ≈ V2 codec performance
         delta_coo_int32 + zstd-1 ≈ V3 codec performance
         delta_coo_downscaled + zstd-1 ≈ V3.1 codec performance

3. ALGORITHMS:
   Industry-standard compression libraries:
   - lz4: Facebook, fastest, used in Linux kernel
   - snappy: Google, used in BigTable/Kafka/Cassandra
   - zstd-{1,3,9,19}: Facebook, modern standard, multiple compression levels
   - gzip-6: Universal baseline (zlib)

4. TIMING METHODOLOGY:
   - 2 warmup iterations (for JIT compilation effects)
   - 5 measurement iterations
   - Report median (robust to outliers)
   - GC before each measurement batch (not inside loop)
   - Verify correctness: decompress(compress(x)) == x

5. THROUGHPUT CALCULATION:
   - All throughputs use uncompressed data size as numerator
   - throughput_encode = uncompressed_bytes / encode_time
   - throughput_decode = uncompressed_bytes / decode_time

6. PER-LAYER PROCESSING:
   - All representations process layers independently
   - Data is reconstructable (unlike v3 benchmark's cross-layer concat)

7. REPRODUCIBILITY:
   - Log library versions
   - Log system info
   - Fixed random seed (N/A - no randomness)

Author: Grail Research
"""

import csv
import gc
import gzip
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

# Add grail to path before importing
sys.path.insert(0, "/root/grail")

# Compression libraries with version tracking
LIBRARY_VERSIONS = {}

try:
    import lz4
    import lz4.frame as lz4f
    LIBRARY_VERSIONS["lz4"] = lz4.__version__
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import snappy
    LIBRARY_VERSIONS["snappy"] = "installed"  # snappy doesn't expose version
    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False

try:
    import zstandard as zstd
    LIBRARY_VERSIONS["zstandard"] = zstd.__version__
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli
    LIBRARY_VERSIONS["brotli"] = "installed"  # brotli doesn't expose version easily
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


class FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit."""
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[FlushingStreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark measurement with full metadata."""
    # Experiment info
    experiment: str
    model: str
    model_size: str
    source_file: str

    # Data characteristics
    num_layers: int
    num_elements: int  # total nnz
    total_params: int
    sparsity: float
    original_pt_size_bytes: int

    # Compression config
    representation: str
    algorithm: str

    # Size results
    uncompressed_size_bytes: int  # Size before compression algorithm
    compressed_size_bytes: int
    compression_ratio: float  # original_pt_size / compressed_size

    # Timing results (median of N iterations, in milliseconds)
    encode_time_ms: float
    decode_time_ms: float

    # Throughput (MB/s, based on uncompressed size)
    throughput_encode_mb_s: float
    throughput_decode_mb_s: float

    # Verification
    verified_correct: bool


# =============================================================================
# COMPRESSION ALGORITHMS
# =============================================================================

def get_available_algorithms() -> list[str]:
    """Return list of available compression algorithms.

    Must-have (production-relevant):
    - lz4: Fastest option
    - snappy: Industry standard (Kafka, BigTable)
    - zstd-1: Best speed/ratio tradeoff, used in production codec

    Nice-to-have:
    - zstd-3: Shows diminishing returns
    - gzip-6: Universal baseline

    Deferred (too slow for production, run separately):
    - lz4hc, zstd-9, zstd-19, brotli
    """
    algos = []

    if HAS_LZ4:
        algos.append("lz4")

    if HAS_SNAPPY:
        algos.append("snappy")

    if HAS_ZSTD:
        algos.extend(["zstd-1", "zstd-3"])

    # gzip is always available (stdlib) - commented out, too slow
    # algos.append("gzip-6")

    return algos


def compress(data: bytes, algorithm: str) -> bytes:
    """Compress data with specified algorithm."""
    if algorithm == "lz4":
        return lz4f.compress(data)
    elif algorithm == "lz4hc":
        return lz4f.compress(data, compression_level=lz4f.COMPRESSIONLEVEL_MAX)
    elif algorithm == "snappy":
        return snappy.compress(data)
    elif algorithm.startswith("zstd-"):
        level = int(algorithm.split("-")[1])
        return zstd.ZstdCompressor(level=level).compress(data)
    elif algorithm.startswith("gzip-"):
        level = int(algorithm.split("-")[1])
        return gzip.compress(data, compresslevel=level)
    elif algorithm.startswith("brotli-"):
        level = int(algorithm.split("-")[1])
        return brotli.compress(data, quality=level)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def decompress(data: bytes, algorithm: str) -> bytes:
    """Decompress data with specified algorithm."""
    if algorithm in ("lz4", "lz4hc"):
        return lz4f.decompress(data)
    elif algorithm == "snappy":
        return snappy.decompress(data)
    elif algorithm.startswith("zstd-"):
        return zstd.ZstdDecompressor().decompress(data)
    elif algorithm.startswith("gzip-"):
        return gzip.decompress(data)
    elif algorithm.startswith("brotli-"):
        return brotli.decompress(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def benchmark_compression(
    data: bytes,
    algorithm: str,
    warmup_iterations: int = 1,
    measure_iterations: int = 1,
) -> tuple[bytes, float, float, bool]:
    """
    Benchmark compression with proper methodology.

    Returns: (compressed_data, median_encode_ms, median_decode_ms, verified_correct)
    """
    # Warmup (outside timing)
    for _ in range(warmup_iterations):
        compressed = compress(data, algorithm)
        _ = decompress(compressed, algorithm)

    # Measure encode
    gc.collect()  # GC before measurement batch, not inside loop
    encode_times = []
    for _ in range(measure_iterations):
        start = time.perf_counter()
        compressed = compress(data, algorithm)
        encode_times.append((time.perf_counter() - start) * 1000)

    # Measure decode
    gc.collect()
    decode_times = []
    for _ in range(measure_iterations):
        start = time.perf_counter()
        decompressed = decompress(compressed, algorithm)
        decode_times.append((time.perf_counter() - start) * 1000)

    # Verify correctness
    verified = (decompressed == data)

    # Return median times
    encode_times.sort()
    decode_times.sort()
    median_encode = encode_times[measure_iterations // 2]
    median_decode = decode_times[measure_iterations // 2]

    return compressed, median_encode, median_decode, verified


# =============================================================================
# DELTA ENCODING (FIXED - maintains int32, no promotion)
# =============================================================================

def delta_encode_coo_int32(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Delta encode 2D COO indices with int32 output.

    Algorithm:
    1. Lexicographic sort by (row, col)
    2. Row deltas: simple diff
    3. Col deltas: diff within same row, absolute at row boundary

    Returns: (delta_rows, delta_cols, sort_order)
    """
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    if len(rows) == 0:
        return rows, cols, np.array([], dtype=np.int64)

    # Lexicographic sort
    sort_order = np.lexsort((cols, rows))
    sorted_rows = rows[sort_order]
    sorted_cols = cols[sort_order]

    # Delta encode rows (vectorized, no int64 promotion)
    delta_rows = np.empty_like(sorted_rows, dtype=np.int32)
    delta_rows[0] = sorted_rows[0]
    delta_rows[1:] = sorted_rows[1:] - sorted_rows[:-1]

    # Delta encode cols (vectorized)
    # At row boundaries: use absolute col
    # Within same row: use col diff
    row_changed = np.ones(len(sorted_rows), dtype=bool)
    row_changed[1:] = delta_rows[1:] != 0

    col_diff = np.empty_like(sorted_cols, dtype=np.int32)
    col_diff[0] = sorted_cols[0]
    col_diff[1:] = sorted_cols[1:] - sorted_cols[:-1]

    delta_cols = np.where(row_changed, sorted_cols, col_diff).astype(np.int32)

    return delta_rows, delta_cols, sort_order


def delta_encode_flat_int32(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Delta encode 1D flat indices with int32 output.

    Returns: (delta_indices, sort_order)
    """
    indices = np.asarray(indices, dtype=np.int32)

    if len(indices) == 0:
        return indices, np.array([], dtype=np.int64)

    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]

    delta = np.empty_like(sorted_indices, dtype=np.int32)
    delta[0] = sorted_indices[0]
    delta[1:] = sorted_indices[1:] - sorted_indices[:-1]

    return delta, sort_order


# =============================================================================
# REPRESENTATION BUILDERS
# =============================================================================

def values_to_bytes(values: torch.Tensor) -> bytes:
    """Convert values tensor to bytes, preserving bfloat16 as int16."""
    if values.dtype == torch.bfloat16:
        return values.view(torch.int16).numpy().tobytes()
    else:
        return values.numpy().tobytes()


def build_raw_coo(layers: dict) -> bytes:
    """
    Raw COO: unsorted indices (int32) + values, per-layer concatenated.

    Format per layer: [rows_int32][cols_int32][values]
    No sorting, no delta encoding.
    """
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]

        if indices.ndim == 2 and indices.shape[0] == 2:
            chunks.append(indices[0].astype(np.int32).tobytes())
            chunks.append(indices[1].astype(np.int32).tobytes())
        else:
            chunks.append(indices.astype(np.int32).tobytes())
        chunks.append(values_to_bytes(values))

    return b"".join(chunks)


def build_raw_flat(layers: dict) -> bytes:
    """
    Raw flat: unsorted flat indices (int32) + values, per-layer concatenated.

    Format per layer: [flat_indices_int32][values]
    No sorting, no delta encoding.
    """
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]
        shape = layer_data["shape"]

        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
            flat = (rows.astype(np.int64) * shape[1] + cols).astype(np.int32)
        else:
            flat = indices.astype(np.int32)

        chunks.append(flat.tobytes())
        chunks.append(values_to_bytes(values))

    return b"".join(chunks)


def build_delta_coo_int32(layers: dict) -> bytes:
    """
    Delta COO: sorted + delta-encoded COO indices (int32), per-layer.

    Format per layer: [delta_rows_int32][delta_cols_int32][sorted_values]
    """
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]
        values_np = values.view(torch.int16).numpy() if values.dtype == torch.bfloat16 else values.numpy()

        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
            delta_rows, delta_cols, sort_order = delta_encode_coo_int32(rows, cols)
            values_sorted = values_np[sort_order]

            chunks.append(delta_rows.tobytes())
            chunks.append(delta_cols.tobytes())
            chunks.append(values_sorted.tobytes())
        else:
            # 1D layer: flatten indices from (1, nnz) to (nnz,)
            flat_indices = indices.flatten().astype(np.int32)
            delta, sort_order = delta_encode_flat_int32(flat_indices)
            values_sorted = values_np[sort_order]
            chunks.append(delta.tobytes())
            chunks.append(values_sorted.tobytes())

    return b"".join(chunks)


def build_delta_flat_int32(layers: dict) -> bytes:
    """
    Delta flat: sorted + delta-encoded flat indices (int32), per-layer.

    Format per layer: [delta_flat_int32][sorted_values]
    """
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]
        shape = layer_data["shape"]
        values_np = values.view(torch.int16).numpy() if values.dtype == torch.bfloat16 else values.numpy()

        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
            flat = (rows.astype(np.int64) * shape[1] + cols).astype(np.int32)
        else:
            # 1D layer: flatten indices from (1, nnz) to (nnz,)
            flat = indices.flatten().astype(np.int32)

        delta, sort_order = delta_encode_flat_int32(flat)
        values_sorted = values_np[sort_order]

        chunks.append(delta.tobytes())
        chunks.append(values_sorted.tobytes())

    return b"".join(chunks)


def build_delta_coo_downscaled(layers: dict) -> bytes:
    """
    Delta COO with index downscaling (uint8 rows, uint16 cols), per-layer.

    This captures V3.1's optimization: delta-encoded row indices typically fit
    in uint8 (max 255), and col indices fit in uint16 (max 65535).

    Format per layer: [delta_rows_uint8][delta_cols_uint16][sorted_values]
    Falls back to int32 if values exceed dtype range.
    """
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]
        values_np = values.view(torch.int16).numpy() if values.dtype == torch.bfloat16 else values.numpy()

        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
            delta_rows, delta_cols, sort_order = delta_encode_coo_int32(rows, cols)
            values_sorted = values_np[sort_order]

            # Downscale rows to uint8 if possible (delta rows are non-negative after sort)
            if delta_rows.size > 0 and delta_rows.min() >= 0 and delta_rows.max() <= 255:
                chunks.append(delta_rows.astype(np.uint8).tobytes())
            else:
                chunks.append(delta_rows.astype(np.int32).tobytes())

            # Downscale cols to uint16 if possible
            if delta_cols.size > 0 and delta_cols.min() >= 0 and delta_cols.max() <= 65535:
                chunks.append(delta_cols.astype(np.uint16).tobytes())
            else:
                chunks.append(delta_cols.astype(np.int32).tobytes())

            chunks.append(values_sorted.tobytes())
        else:
            # 1D layer: flatten indices from (1, nnz) to (nnz,), use int32
            flat_indices = indices.flatten().astype(np.int32)
            delta, sort_order = delta_encode_flat_int32(flat_indices)
            values_sorted = values_np[sort_order]
            chunks.append(delta.tobytes())
            chunks.append(values_sorted.tobytes())

    return b"".join(chunks)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def parse_model_info(experiment_name: str) -> tuple[str, str]:
    """Parse model family and size from experiment name."""
    exp_lower = experiment_name.lower()

    if "qwen" in exp_lower:
        model = "qwen2.5"
        if "7b" in exp_lower: return model, "7B"
        if "1.5b" in exp_lower: return model, "1.5B"
        if "0.5b" in exp_lower: return model, "0.5B"
    elif "llama" in exp_lower:
        model = "llama3.2"
        if "3b" in exp_lower: return model, "3B"
    elif "gemma" in exp_lower:
        model = "gemma3"
        if "4b" in exp_lower: return model, "4B"
        if "1b" in exp_lower: return model, "1B"

    return "unknown", "unknown"


def benchmark_delta_file(
    delta_path: Path,
    experiment_name: str,
    algorithms: list[str],
) -> list[dict]:
    """Benchmark all representations and algorithms on a single delta file."""
    results = []

    # Load delta file
    delta = torch.load(delta_path, map_location="cpu", weights_only=True)
    layers = delta.get("layers", {})

    if not layers:
        return results

    # Compute statistics
    num_layers = len(layers)
    total_nnz = sum(ld["values"].numel() for ld in layers.values())
    total_params = sum(
        np.prod(ld["shape"]) for ld in layers.values()
    )
    sparsity = 1.0 - (total_nnz / total_params) if total_params > 0 else 0.0
    original_pt_size = delta_path.stat().st_size
    model, model_size = parse_model_info(experiment_name)

    base_info = {
        "experiment": experiment_name,
        "model": model,
        "model_size": model_size,
        "source_file": delta_path.name,
        "num_layers": num_layers,
        "num_elements": total_nnz,
        "total_params": int(total_params),
        "sparsity": sparsity,
        "original_pt_size_bytes": original_pt_size,
    }

    # =========================================================================
    # Part 1: Raw representations + compression algorithms
    # =========================================================================
    representations = [
        ("raw_coo", build_raw_coo),
        ("raw_flat", build_raw_flat),
        ("delta_coo_int32", build_delta_coo_int32),
        ("delta_flat_int32", build_delta_flat_int32),
        ("delta_coo_downscaled", build_delta_coo_downscaled),
    ]

    for rep_name, builder in representations:
        try:
            data = builder(layers)
            uncompressed_size = len(data)

            for algo in algorithms:
                try:
                    compressed, enc_ms, dec_ms, verified = benchmark_compression(data, algo)

                    ratio = original_pt_size / len(compressed) if compressed else 0
                    enc_tp = (uncompressed_size / 1e6) / (enc_ms / 1000) if enc_ms > 0 else 0
                    dec_tp = (uncompressed_size / 1e6) / (dec_ms / 1000) if dec_ms > 0 else 0

                    results.append(asdict(BenchmarkResult(
                        **base_info,
                        representation=rep_name,
                        algorithm=algo,
                        uncompressed_size_bytes=uncompressed_size,
                        compressed_size_bytes=len(compressed),
                        compression_ratio=ratio,
                        encode_time_ms=enc_ms,
                        decode_time_ms=dec_ms,
                        throughput_encode_mb_s=enc_tp,
                        throughput_decode_mb_s=dec_tp,
                        verified_correct=verified,
                    )))
                except Exception as e:
                    logger.warning(f"Error {rep_name}+{algo}: {e}")
        except Exception as e:
            logger.warning(f"Error building {rep_name}: {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compression Benchmark - Standard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", type=str, default="/root/grail/data/compression_benchmark_v4.csv")
    parser.add_argument("--samples", type=int, default=10, help="Max samples per experiment")
    parser.add_argument("--experiments", type=str, nargs="*", help="Specific experiments to run")
    args = parser.parse_args()

    # Log system info for reproducibility
    logger.info("=" * 70)
    logger.info("Compression Benchmark - Standard")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Library versions: {LIBRARY_VERSIONS}")
    logger.info("")
    logger.info(f"Output: {args.output}")
    logger.info(f"Samples per experiment: {args.samples}")

    algorithms = get_available_algorithms()
    logger.info(f"Algorithms: {algorithms}")
    logger.info("Representations: raw_coo, raw_flat, delta_coo_int32, delta_flat_int32, delta_coo_downscaled")

    # Find experiments
    base_path = Path("/root/grail/research/sparsity_analysis/experiments")
    if args.experiments:
        exp_dirs = [base_path / exp for exp in args.experiments if (base_path / exp).exists()]
    else:
        exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.endswith("-partial")]
        exp_dirs = [d for d in exp_dirs if list(d.glob("**/delta_*.pt"))]

    # Sort by model size
    def size_key(d):
        n = d.name.lower()
        if "0.5b" in n: return 0
        if "1.5b" in n or "1b" in n: return 1
        if "3b" in n: return 2
        if "4b" in n: return 3
        if "7b" in n: return 4
        return 5

    exp_dirs = sorted(exp_dirs, key=size_key)
    logger.info(f"Found {len(exp_dirs)} experiments")

    all_results = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # Find seed directories (patterns: seed*, deltas_*_seed*)
        seed_dirs = []
        for pattern in ["seed*", "**/deltas_*_seed*", "**/seed*"]:
            seed_dirs.extend([d for d in exp_dir.glob(pattern) if d.is_dir()])

        # Deduplicate and sort
        seed_dirs = sorted(set(seed_dirs), key=lambda x: str(x))

        # If no seed dirs found, treat the whole experiment as one "seed"
        if not seed_dirs:
            seed_dirs = [exp_dir]

        # Sample evenly from each seed
        delta_files = []
        for seed_dir in seed_dirs:
            seed_files = sorted(seed_dir.glob("**/delta_*.pt"))
            if not seed_files:
                continue

            # Sample args.samples files evenly from this seed
            if len(seed_files) > args.samples:
                step = len(seed_files) / args.samples
                seed_files = [seed_files[int(i * step)] for i in range(args.samples)]

            delta_files.extend(seed_files)

        if not delta_files:
            logger.warning(f"No delta files found for {exp_name}")
            continue

        logger.info(f"\nBenchmarking {exp_name} ({len(delta_files)} files, {len(seed_dirs)} seeds)...")

        for i, delta_path in enumerate(delta_files):
            try:
                logger.info(f"  Processing {delta_path.name}...")
                results = benchmark_delta_file(delta_path, exp_name, algorithms)
                all_results.extend(results)

                # Log progress (show delta_coo_downscaled + zstd-1 as production-relevant)
                best = next((r for r in results if r["representation"] == "delta_coo_downscaled" and r["algorithm"] == "zstd-1"), None)
                if best:
                    logger.info(
                        f"  [{i+1}/{len(delta_files)}] {delta_path.name}: "
                        f"ratio={best['compression_ratio']:.2f}x, "
                        f"{best['compressed_size_bytes']/1e6:.1f}MB, "
                        f"enc={best['encode_time_ms']:.0f}ms, "
                        f"dec={best['decode_time_ms']:.0f}ms"
                    )
            except Exception as e:
                logger.error(f"Error processing {delta_path}: {e}")

    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(all_results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logger.info(f"\nSaved {len(all_results)} results to {output_path}")

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY (zstd-1 results)")
        logger.info("=" * 70)

        for rep in ["raw_coo", "raw_flat", "delta_coo_int32", "delta_flat_int32", "delta_coo_downscaled"]:
            rep_results = [r for r in all_results if r["representation"] == rep and r["algorithm"] == "zstd-1"]
            if rep_results:
                avg_ratio = sum(r["compression_ratio"] for r in rep_results) / len(rep_results)
                avg_enc = sum(r["encode_time_ms"] for r in rep_results) / len(rep_results)
                avg_dec = sum(r["decode_time_ms"] for r in rep_results) / len(rep_results)
                logger.info(f"{rep:25s}: {avg_ratio:.2f}x ratio, {avg_enc:.0f}ms enc, {avg_dec:.0f}ms dec")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
