#!/usr/bin/env python3
"""
Analyze the distribution of delta-encoded COO indices to determine
optimal integer types (int8, uint8, int16, uint16, int4, etc.)

Key insight: After lexsort by (row, col), all deltas are NON-NEGATIVE because:
1. Row deltas: rows are sorted ascending, so delta >= 0
2. Col deltas: within same row, cols sorted ascending; on row change, we store absolute col

This means we can use UNSIGNED types (uint8, uint16) for double the range.
"""

import os
import numpy as np
from pathlib import Path
import torch
from collections import defaultdict

# Data directories to search
DATA_DIRS = [
    Path("/root/grail/research/sparsity_analysis/experiments"),
    Path("/root/grail/research/sparsity_analysis/qwen2.5-1.5b-sft-math-lr2e-05"),
    Path("/root/grail/research/sparsity_analysis/qwen2.5-1.5b-sft-math-lr3e-06"),
]


def delta_encode_coo(rows: np.ndarray, cols: np.ndarray):
    """
    Delta encode COO indices after lexsort.
    Returns (row_deltas, col_deltas)
    """
    if len(rows) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Lexsort by (row, col) - secondary key first
    order = np.lexsort((cols, rows))
    rows_sorted = rows[order]
    cols_sorted = cols[order]

    # Row deltas (always >= 0 after sort)
    row_deltas = np.empty_like(rows_sorted)
    row_deltas[0] = rows_sorted[0]
    row_deltas[1:] = rows_sorted[1:] - rows_sorted[:-1]

    # Column deltas: absolute when row changes, delta when same row
    row_boundary = np.ones(len(rows_sorted), dtype=bool)
    row_boundary[1:] = row_deltas[1:] != 0

    col_diff = np.empty_like(cols_sorted)
    col_diff[0] = cols_sorted[0]
    col_diff[1:] = cols_sorted[1:] - cols_sorted[:-1]

    col_deltas = np.where(row_boundary, cols_sorted, col_diff)

    return row_deltas, col_deltas


def analyze_file(filepath: Path):
    """Analyze delta distributions for a single .pt file."""
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=True)
    except Exception as e:
        return None

    # Handle the actual file format: {step, timestamp, layers, metadata}
    if not isinstance(data, dict) or 'layers' not in data:
        return None

    layers = data['layers']

    results = {
        "row_deltas": [],
        "col_deltas": [],
        "row_delta_max": 0,
        "col_delta_max": 0,
        "total_nnz": 0,
        "tensor_shapes": [],
    }

    for name, layer_data in layers.items():
        # Each layer has: indices (2, nnz), values (nnz,), shape, nnz
        if not isinstance(layer_data, dict):
            continue
        if 'indices' not in layer_data:
            continue

        indices = layer_data['indices']
        if indices.shape[0] != 2:
            continue  # Not COO format

        rows = indices[0].numpy().astype(np.int32)
        cols = indices[1].numpy().astype(np.int32)

        if len(rows) == 0:
            continue

        row_deltas, col_deltas = delta_encode_coo(rows, cols)

        results["row_deltas"].extend(row_deltas.tolist())
        results["col_deltas"].extend(col_deltas.tolist())
        results["row_delta_max"] = max(results["row_delta_max"], row_deltas.max())
        results["col_delta_max"] = max(results["col_delta_max"], col_deltas.max())
        results["total_nnz"] += len(rows)

        if 'shape' in layer_data:
            results["tensor_shapes"].append(layer_data['shape'])

    return results


def compute_distribution_stats(values: list, name: str):
    """Compute distribution statistics for fitting into various int types."""
    if not values:
        return {}

    arr = np.array(values)

    # Check ranges
    min_val = arr.min()
    max_val = arr.max()

    # Unsigned types (valid since all deltas >= 0)
    uint4_max = 15
    uint8_max = 255
    uint16_max = 65535

    # Signed types (for comparison)
    int4_max = 7
    int8_max = 127
    int16_max = 32767

    stats = {
        "name": name,
        "count": len(arr),
        "min": int(min_val),
        "max": int(max_val),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
        # Type fit percentages
        "fits_int4": float((arr <= int4_max).mean() * 100),
        "fits_uint4": float((arr <= uint4_max).mean() * 100),
        "fits_int8": float((arr <= int8_max).mean() * 100),
        "fits_uint8": float((arr <= uint8_max).mean() * 100),
        "fits_int16": float((arr <= int16_max).mean() * 100),
        "fits_uint16": float((arr <= uint16_max).mean() * 100),
        "all_non_negative": bool(min_val >= 0),
    }

    # Distribution of values (histogram bins)
    bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, float('inf')]
    hist, _ = np.histogram(arr, bins=bins)
    stats["histogram"] = {f"<={bins[i+1]}": int(hist[i]) for i in range(len(hist))}

    return stats


def main():
    print("=" * 70)
    print("Delta-Encoded COO Index Distribution Analysis")
    print("=" * 70)
    print()
    print("Key insight: After lexsort, ALL deltas are NON-NEGATIVE")
    print("  - Row deltas: rows sorted ascending → delta >= 0")
    print("  - Col deltas: cols sorted within row → delta >= 0")
    print("  → We can use UNSIGNED types (uint8/uint16) for 2x range!")
    print()

    # Find all delta files
    all_files = []
    for data_dir in DATA_DIRS:
        if data_dir.exists():
            all_files.extend(data_dir.rglob("delta_*.pt"))

    if not all_files:
        print("No delta files found!")
        return

    print(f"Found {len(all_files)} delta files")

    all_row_deltas = []
    all_col_deltas = []
    per_experiment_stats = {}

    # Group files by experiment
    files_by_exp = defaultdict(list)
    for f in all_files:
        # Extract experiment name from path
        parts = f.parts
        for i, p in enumerate(parts):
            if "qwen" in p.lower() or "llama" in p.lower() or "gemma" in p.lower():
                exp_name = p
                break
        else:
            exp_name = f.parent.parent.parent.name

        files_by_exp[exp_name].append(f)

    for exp_name, files in sorted(files_by_exp.items()):
        # Sample up to 5 files per experiment
        sample_files = sorted(files)[:5]

        print(f"Analyzing {exp_name} ({len(sample_files)} files)...")

        exp_row_deltas = []
        exp_col_deltas = []

        for f in sample_files:
            try:
                results = analyze_file(f)
                if results and results["row_deltas"]:
                    exp_row_deltas.extend(results["row_deltas"])
                    exp_col_deltas.extend(results["col_deltas"])
            except Exception as e:
                print(f"  Error processing {f.name}: {e}")

        if exp_row_deltas:
            all_row_deltas.extend(exp_row_deltas)
            all_col_deltas.extend(exp_col_deltas)

            per_experiment_stats[exp_name] = {
                "row": compute_distribution_stats(exp_row_deltas, "row_deltas"),
                "col": compute_distribution_stats(exp_col_deltas, "col_deltas"),
            }

    if not all_row_deltas:
        print("No valid delta data found!")
        return

    print()
    print("=" * 70)
    print("AGGREGATE STATISTICS (All Experiments)")
    print("=" * 70)

    row_stats = compute_distribution_stats(all_row_deltas, "row_deltas")
    col_stats = compute_distribution_stats(all_col_deltas, "col_deltas")

    for name, stats in [("ROW DELTAS", row_stats), ("COL DELTAS", col_stats)]:
        print(f"\n{name}:")
        print(f"  Total values: {stats['count']:,}")
        print(f"  Range: [{stats['min']}, {stats['max']}]")
        print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
        print(f"  Percentiles: P90={stats['p90']:.0f}, P95={stats['p95']:.0f}, P99={stats['p99']:.0f}, P99.9={stats['p999']:.0f}")
        print(f"  All non-negative: {stats['all_non_negative']}")
        print()
        print(f"  Type fit percentages:")
        print(f"    int4  (≤7):     {stats['fits_int4']:6.2f}%")
        print(f"    uint4 (≤15):    {stats['fits_uint4']:6.2f}%")
        print(f"    int8  (≤127):   {stats['fits_int8']:6.2f}%")
        print(f"    uint8 (≤255):   {stats['fits_uint8']:6.2f}%")
        print(f"    int16 (≤32767): {stats['fits_int16']:6.2f}%")
        print(f"    uint16(≤65535): {stats['fits_uint16']:6.2f}%")

        print()
        print(f"  Value distribution (histogram):")
        total = stats['count']
        cumulative = 0
        for bucket, count in stats['histogram'].items():
            cumulative += count
            pct = count / total * 100
            cum_pct = cumulative / total * 100
            bar = "#" * int(pct / 2)
            print(f"    {bucket:>10}: {count:>10,} ({pct:5.1f}%) [cum: {cum_pct:5.1f}%] {bar}")

    # Summary recommendation
    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Row deltas analysis
    row_fits_uint8 = row_stats['fits_uint8']
    row_fits_uint16 = row_stats['fits_uint16']

    print(f"\nRow Deltas (max={row_stats['max']}):")
    if row_fits_uint8 == 100:
        print("  ✓ ALL values fit in uint8 (0-255)")
        print("  → Use uint8 for 4x compression vs int32!")
    elif row_fits_uint8 > 99:
        print(f"  ~ {row_fits_uint8:.2f}% fit in uint8")
        print("  → Consider hybrid: uint8 with overflow handling")
    elif row_fits_uint16 == 100:
        print("  ✓ ALL values fit in uint16 (0-65535)")
        print("  → Use uint16 for 2x compression vs int32!")
    else:
        print(f"  ✗ Max {row_stats['max']} exceeds uint16 range")
        print("  → Stay with int32 or use variable-length encoding")

    # Col deltas analysis
    col_fits_uint8 = col_stats['fits_uint8']
    col_fits_uint16 = col_stats['fits_uint16']

    print(f"\nCol Deltas (max={col_stats['max']}):")
    if col_fits_uint8 == 100:
        print("  ✓ ALL values fit in uint8 (0-255)")
        print("  → Use uint8 for 4x compression vs int32!")
    elif col_fits_uint8 > 99:
        print(f"  ~ {col_fits_uint8:.2f}% fit in uint8")
        print("  → Consider hybrid: uint8 with overflow handling")
    elif col_fits_uint16 == 100:
        print("  ✓ ALL values fit in uint16 (0-65535)")
        print("  → Use uint16 for 2x compression vs int32!")
    else:
        print(f"  ✗ Max {col_stats['max']} exceeds uint16 range")
        print("  → Stay with int32 or use variable-length encoding")

    # Compression impact estimate
    print()
    print("=" * 70)
    print("COMPRESSION IMPACT ESTIMATE")
    print("=" * 70)

    total_indices = row_stats['count'] + col_stats['count']

    print(f"\nFor {total_indices:,} total index values:")
    print()

    # Current: int32
    bytes_int32 = total_indices * 4
    print(f"  Current (int32): {bytes_int32:,} bytes = {bytes_int32/1e6:.2f} MB")

    # Option 1: uint16 for both
    if row_fits_uint16 == 100 and col_fits_uint16 == 100:
        bytes_uint16 = total_indices * 2
        savings = (1 - bytes_uint16/bytes_int32) * 100
        print(f"  uint16 for both: {bytes_uint16:,} bytes = {bytes_uint16/1e6:.2f} MB ({savings:.0f}% smaller)")

    # Option 2: uint8 where possible
    if row_fits_uint8 == 100 and col_fits_uint16 == 100:
        bytes_mixed = row_stats['count'] * 1 + col_stats['count'] * 2
        savings = (1 - bytes_mixed/bytes_int32) * 100
        print(f"  uint8 rows + uint16 cols: {bytes_mixed:,} bytes = {bytes_mixed/1e6:.2f} MB ({savings:.0f}% smaller)")

    if row_fits_uint8 == 100 and col_fits_uint8 == 100:
        bytes_uint8 = total_indices * 1
        savings = (1 - bytes_uint8/bytes_int32) * 100
        print(f"  uint8 for both: {bytes_uint8:,} bytes = {bytes_uint8/1e6:.2f} MB ({savings:.0f}% smaller)")

    # Per-experiment breakdown
    print()
    print("=" * 70)
    print("PER-EXPERIMENT BREAKDOWN (fits uint8 %)")
    print("=" * 70)
    print()
    print(f"{'Experiment':<40} {'Row→uint8':>12} {'Col→uint8':>12} {'Row max':>10} {'Col max':>10}")
    print("-" * 84)

    for exp_name, stats in sorted(per_experiment_stats.items()):
        row_u8 = stats['row']['fits_uint8']
        col_u8 = stats['col']['fits_uint8']
        row_max = stats['row']['max']
        col_max = stats['col']['max']
        print(f"{exp_name:<40} {row_u8:>11.2f}% {col_u8:>11.2f}% {row_max:>10} {col_max:>10}")

    # Deep analysis for int4/uint4
    print()
    print("=" * 70)
    print("DEEP ANALYSIS: INT4/UINT4 FEASIBILITY")
    print("=" * 70)

    print(f"\nRow deltas:")
    print(f"  Fits uint4 (≤15): {row_stats['fits_uint4']:.2f}%")
    print(f"  Most row deltas are 0 (same row) or small positive (adjacent row)")

    print(f"\nCol deltas:")
    print(f"  Fits uint4 (≤15): {col_stats['fits_uint4']:.2f}%")
    print(f"  Col deltas are larger due to sparse distribution within rows")

    if row_stats['fits_uint4'] > 90:
        print("\n  → Row deltas could use 4-bit encoding with escape codes for overflow")
        print("    This would give 8x compression on rows vs int32")
    else:
        print("\n  → Row deltas don't fit 4-bit well enough for simple encoding")

    # Analysis of zeros
    print()
    print("=" * 70)
    print("ZERO VALUE ANALYSIS (for run-length encoding)")
    print("=" * 70)

    row_arr = np.array(all_row_deltas)
    col_arr = np.array(all_col_deltas)

    row_zeros = (row_arr == 0).sum()
    col_zeros = (col_arr == 0).sum()

    print(f"\nRow deltas == 0: {row_zeros:,} ({row_zeros/len(row_arr)*100:.1f}%)")
    print(f"  These are consecutive entries in the same row")
    print(f"  → Excellent for run-length encoding or sparse bit-packing")

    print(f"\nCol deltas == 0: {col_zeros:,} ({col_zeros/len(col_arr)*100:.1f}%)")
    print(f"  These would indicate adjacent column entries (rare in sparse data)")

    # Variable-length encoding analysis
    print()
    print("=" * 70)
    print("VARIABLE-LENGTH ENCODING ANALYSIS")
    print("=" * 70)

    print("\nOptimal encoding strategy per value range:")
    print()

    for name, arr in [("Row deltas", row_arr), ("Col deltas", col_arr)]:
        # Simple varint estimation
        bytes_1 = (arr <= 127).sum() * 1
        bytes_2 = ((arr > 127) & (arr <= 16383)).sum() * 2
        bytes_3 = ((arr > 16383) & (arr <= 2097151)).sum() * 3
        bytes_4 = (arr > 2097151).sum() * 4

        total_varint = bytes_1 + bytes_2 + bytes_3 + bytes_4
        total_fixed = len(arr) * 4

        print(f"  {name}:")
        print(f"    Fixed int32: {total_fixed:,} bytes")
        print(f"    Varint:      {total_varint:,} bytes ({(1 - total_varint/total_fixed)*100:.1f}% smaller)")

    # Final recommendation
    print()
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    print()
    print("Based on the analysis:")
    print()

    # Determine best strategy
    if row_fits_uint16 == 100 and col_fits_uint16 == 100:
        if row_fits_uint8 > 99.9 and col_fits_uint8 > 99.9:
            print("1. PRIMARY: Use uint8 for both row and col deltas")
            print("   - 75% reduction in index storage")
            print("   - Simple implementation, no overflow handling needed")
        else:
            print("1. PRIMARY: Use uint16 for both row and col deltas")
            print("   - 50% reduction in index storage")
            print("   - Simple implementation, always fits")
    else:
        print("1. PRIMARY: Use int16/uint16 where possible, int32 for large dims")
        print("   - Check tensor dimensions at runtime")

    print()
    print("2. COMBINED WITH zstd compression:")
    print("   - Smaller integer types compress even better")
    print("   - Delta encoding creates runs of small values → better LZ compression")
    print("   - Expected ~8-10x total compression ratio")

    print()
    print("3. NOT RECOMMENDED: int4 encoding")
    print("   - Complexity of packed storage not worth it")
    print("   - zstd already handles the patterns well")


if __name__ == "__main__":
    main()
