#!/usr/bin/env python3
"""Analyze COO vs Flat compression with fair comparison (COO raw size as baseline)."""

import pandas as pd
from pathlib import Path

CSV_PATH = Path("/root/grail/data/compression_benchmark_v3.csv")

def main():
    df = pd.read_csv(CSV_PATH)

    # Filter to sparse_delta only
    sparse = df[df['data_type'] == 'sparse_delta'].copy()

    print("=" * 70)
    print("FAIR COMPARISON: COO vs Flat (using COO raw size as baseline)")
    print("=" * 70)
    print()

    # For each (experiment, source_file, algorithm, threads) pair,
    # compare COO and flat compressed sizes

    coo = sparse[sparse['representation'] == 'sparse_coo']
    flat = sparse[sparse['representation'] == 'sparse_flat']

    # Merge on common keys
    merged = coo.merge(
        flat[['experiment', 'source_file', 'algorithm', 'threads',
              'compressed_size_bytes']],
        on=['experiment', 'source_file', 'algorithm', 'threads'],
        suffixes=('_coo', '_flat')
    )

    print(f"Total matched pairs: {len(merged)}")
    print()

    # Add fair compression ratios (both relative to COO raw size)
    # raw_size_bytes is COO's raw size (from the left side of merge)
    merged['ratio_coo_fair'] = merged['raw_size_bytes'] / merged['compressed_size_bytes_coo']
    merged['ratio_flat_fair'] = merged['raw_size_bytes'] / merged['compressed_size_bytes_flat']
    merged['flat_vs_coo_pct'] = (merged['compressed_size_bytes_flat'] / merged['compressed_size_bytes_coo'] - 1) * 100

    print("=" * 70)
    print("PER-ALGORITHM COMPARISON (averaged across all files)")
    print("=" * 70)
    print()
    print(f"{'Algorithm':<15} {'Threads':>7} {'COO Ratio':>10} {'Flat Ratio':>11} {'Flat vs COO':>12}")
    print(f"{'':15} {'':>7} {'(fair)':>10} {'(fair)':>11} {'(% smaller)':>12}")
    print("-" * 70)

    # Group by algorithm and threads
    algo_stats = merged.groupby(['algorithm', 'threads']).agg({
        'ratio_coo_fair': 'mean',
        'ratio_flat_fair': 'mean',
        'compressed_size_bytes_coo': 'sum',
        'compressed_size_bytes_flat': 'sum',
    }).reset_index()

    algo_stats['flat_smaller_pct'] = (1 - algo_stats['compressed_size_bytes_flat'] / algo_stats['compressed_size_bytes_coo']) * 100

    # Sort by COO ratio descending
    algo_stats = algo_stats.sort_values('ratio_coo_fair', ascending=False)

    for _, row in algo_stats.iterrows():
        print(f"{row['algorithm']:<15} {int(row['threads']):>7} {row['ratio_coo_fair']:>10.2f}x {row['ratio_flat_fair']:>10.2f}x {row['flat_smaller_pct']:>+11.1f}%")

    print()
    print("=" * 70)
    print("BEST ALGORITHM: zstd-1 (single-threaded) DETAILED ANALYSIS")
    print("=" * 70)
    print()

    zstd1 = merged[(merged['algorithm'] == 'zstd-1') & (merged['threads'] == 0)]

    print(f"{'Experiment':<35} {'COO (KB)':>10} {'Flat (KB)':>10} {'Flat Better':>12}")
    print("-" * 70)

    exp_stats = zstd1.groupby('experiment').agg({
        'compressed_size_bytes_coo': 'mean',
        'compressed_size_bytes_flat': 'mean',
        'raw_size_bytes': 'mean',
    }).reset_index()

    for _, row in exp_stats.iterrows():
        coo_kb = row['compressed_size_bytes_coo'] / 1024
        flat_kb = row['compressed_size_bytes_flat'] / 1024
        diff_pct = (1 - flat_kb / coo_kb) * 100
        print(f"{row['experiment']:<35} {coo_kb:>10.1f} {flat_kb:>10.1f} {diff_pct:>+11.1f}%")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Overall stats for zstd-1
    zstd1_total_coo = zstd1['compressed_size_bytes_coo'].sum()
    zstd1_total_flat = zstd1['compressed_size_bytes_flat'].sum()
    zstd1_total_raw = zstd1['raw_size_bytes'].sum()

    print(f"\nUsing zstd-1 (best speed/ratio tradeoff):")
    print(f"  Total raw COO size:        {zstd1_total_raw / 1e6:>10.2f} MB")
    print(f"  Total COO compressed:      {zstd1_total_coo / 1e6:>10.2f} MB  ({zstd1_total_raw/zstd1_total_coo:.2f}x ratio)")
    print(f"  Total Flat compressed:     {zstd1_total_flat / 1e6:>10.2f} MB  ({zstd1_total_raw/zstd1_total_flat:.2f}x ratio)")
    print(f"  Flat is {(1 - zstd1_total_flat/zstd1_total_coo)*100:.1f}% smaller than COO")

    # Show what the ORIGINAL (unfair) ratios were reporting
    print()
    print("=" * 70)
    print("ORIGINAL (UNFAIR) RATIOS vs FAIR RATIOS")
    print("=" * 70)
    print()
    print("The original benchmark used different baselines:")
    print("  - COO ratio = COO_raw / COO_compressed")
    print("  - Flat ratio = Flat_raw / Flat_compressed  (Flat_raw is 40% smaller!)")
    print()

    # Get original ratios from the CSV
    orig_coo = coo[coo['algorithm'] == 'zstd-1'][coo['threads'] == 0]['compression_ratio'].mean()
    orig_flat = flat[flat['algorithm'] == 'zstd-1'][flat['threads'] == 0]['compression_ratio'].mean()
    fair_coo = zstd1['ratio_coo_fair'].mean()
    fair_flat = zstd1['ratio_flat_fair'].mean()

    print(f"zstd-1 compression ratios:")
    print(f"  {'':20} {'Original':>12} {'Fair':>12}")
    print(f"  {'COO:':20} {orig_coo:>12.2f}x {fair_coo:>12.2f}x")
    print(f"  {'Flat:':20} {orig_flat:>12.2f}x {fair_flat:>12.2f}x")
    print()
    print(f"Original showed COO {orig_coo/orig_flat:.1f}x better than Flat")
    print(f"Fair comparison shows Flat is actually {(1 - zstd1_total_flat/zstd1_total_coo)*100:.1f}% smaller!")


if __name__ == "__main__":
    main()
