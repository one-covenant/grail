#!/usr/bin/env python3
"""
Generate Figure 8: Total synchronization time vs. network bandwidth.

This plot helps practitioners select the optimal compression algorithm
based on their deployment's network bandwidth constraints.

For the paper: "Understanding and Exploiting Weight Update Sparsity
for Communication-Efficient Distributed RL"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


# =============================================================================
# BENCHMARK DATA
# =============================================================================

# Benchmark results for 7B model sparse delta (~420 MB uncompressed after sparse encoding)
# These values should be updated with actual benchmark results from:
#   scripts/benchmark_delta_encoding_all_algos.py

ALGORITHMS = {
    # Algorithm: (compressed_size_MB, compression_throughput_MB_s, color, linestyle)
    'LZ4': (420, 850, '#2ecc71', '-'),           # Green, solid
    'Snappy': (440, 820, '#27ae60', '--'),       # Dark green, dashed
    'Zstd-1': (380, 580, '#3498db', '-'),        # Blue, solid
    'Zstd-3': (280, 420, '#e74c3c', '-'),        # Red, solid (selected)
    'Zstd-9': (200, 85, '#9b59b6', '-'),         # Purple, solid
    'Gzip-6': (240, 65, '#f39c12', '--'),        # Orange, dashed
    'Brotli-1': (290, 95, '#1abc9c', '--'),      # Teal, dashed
}

# Raw size before compression (sparse COO format for 7B model at ~98% sparsity)
RAW_SIZE_MB = 420  # ~70M changed params × 6 bytes (4 idx + 2 val)

# Full checkpoint size for cumulative ratio calculation
FULL_CHECKPOINT_MB = 14000  # 14 GB for 7B model in BF16

# Bandwidth tiers (in bits per second)
BANDWIDTH_TIERS = {
    'Home\n(100 Mbps)': 100e6,
    'Office\n(1 Gbps)': 1e9,
    'Datacenter\n(10 Gbps)': 10e9,
    'HPC\n(100 Gbps)': 100e9,
}


# =============================================================================
# COMPUTATION
# =============================================================================

def compute_total_time(
    compressed_size_mb: float,
    compression_throughput_mb_s: float,
    bandwidth_bps: np.ndarray,
    raw_size_mb: float = RAW_SIZE_MB
) -> np.ndarray:
    """
    Compute total synchronization time = compression time + transfer time.

    Args:
        compressed_size_mb: Size after compression (MB)
        compression_throughput_mb_s: Compression speed (MB/s)
        bandwidth_bps: Network bandwidth array (bits per second)
        raw_size_mb: Size before compression (MB)

    Returns:
        Total time in seconds for each bandwidth value
    """
    # Compression time (fixed, independent of bandwidth)
    compression_time_s = raw_size_mb / compression_throughput_mb_s

    # Transfer time (depends on bandwidth)
    # Convert: MB to bits, then divide by bits/second
    compressed_size_bits = compressed_size_mb * 8 * 1e6  # MB -> bits
    transfer_time_s = compressed_size_bits / bandwidth_bps

    return compression_time_s + transfer_time_s


def find_crossover_points(
    algorithms: Dict,
    bandwidth_range: np.ndarray
) -> List[Tuple[str, str, float]]:
    """
    Find bandwidth values where algorithm optimality changes.

    Returns:
        List of (algo1, algo2, crossover_bandwidth_bps)
    """
    # Compute times for all algorithms
    times = {}
    for name, (comp_size, comp_throughput, _, _) in algorithms.items():
        times[name] = compute_total_time(comp_size, comp_throughput, bandwidth_range)

    # Find which algorithm is optimal at each bandwidth
    algo_names = list(algorithms.keys())
    time_matrix = np.array([times[name] for name in algo_names])
    optimal_idx = np.argmin(time_matrix, axis=0)

    # Detect changes in optimal algorithm
    crossovers = []
    for i in range(1, len(optimal_idx)):
        if optimal_idx[i] != optimal_idx[i-1]:
            algo1 = algo_names[optimal_idx[i-1]]
            algo2 = algo_names[optimal_idx[i]]
            bw = bandwidth_range[i]
            crossovers.append((algo1, algo2, bw))

    return crossovers


def find_optimal_regions(
    algorithms: Dict,
    bandwidth_range: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """
    Find the bandwidth range where each algorithm is optimal.

    Returns:
        Dict mapping algorithm name to (min_bw, max_bw) where it's optimal
    """
    # Compute times for all algorithms
    times = {}
    for name, (comp_size, comp_throughput, _, _) in algorithms.items():
        times[name] = compute_total_time(comp_size, comp_throughput, bandwidth_range)

    # Find which algorithm is optimal at each bandwidth
    algo_names = list(algorithms.keys())
    time_matrix = np.array([times[name] for name in algo_names])
    optimal_idx = np.argmin(time_matrix, axis=0)

    # Find regions
    regions = {}
    for idx, name in enumerate(algo_names):
        mask = optimal_idx == idx
        if np.any(mask):
            indices = np.where(mask)[0]
            regions[name] = (bandwidth_range[indices[0]], bandwidth_range[indices[-1]])

    return regions


# =============================================================================
# PLOTTING
# =============================================================================

def create_figure(
    algorithms: Dict = ALGORITHMS,
    raw_size_mb: float = RAW_SIZE_MB,
    bandwidth_tiers: Dict = BANDWIDTH_TIERS,
    figsize: Tuple[float, float] = (10, 6),
    output_path: Optional[str] = None,
    show_crossovers: bool = True,
    show_optimal_regions: bool = True,
    highlight_algorithm: str = 'Zstd-3',
) -> plt.Figure:
    """
    Create the bandwidth vs. total sync time plot.

    Args:
        algorithms: Dict of algorithm data
        raw_size_mb: Uncompressed size in MB
        bandwidth_tiers: Dict of labeled bandwidth markers
        figsize: Figure size (width, height)
        output_path: Path to save figure (optional)
        show_crossovers: Whether to annotate crossover points
        show_optimal_regions: Whether to shade optimal regions
        highlight_algorithm: Algorithm to highlight as "selected"

    Returns:
        matplotlib Figure object
    """
    # Set up figure with publication-quality settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Bandwidth range: 50 Mbps to 200 Gbps
    bandwidth_bps = np.logspace(np.log10(50e6), np.log10(200e9), 500)
    bandwidth_gbps = bandwidth_bps / 1e9

    # Find optimal regions for shading
    if show_optimal_regions:
        regions = find_optimal_regions(algorithms, bandwidth_bps)

        # Color map for regions (subtle background shading)
        region_colors = {
            'Zstd-9': '#f5eef8',    # Light purple
            'Zstd-3': '#fdedec',    # Light red
            'Zstd-1': '#eaf2f8',    # Light blue
            'LZ4': '#e9f7ef',       # Light green
        }

        for algo, (bw_min, bw_max) in regions.items():
            if algo in region_colors:
                ax.axvspan(
                    bw_min / 1e9, bw_max / 1e9,
                    alpha=0.4,
                    color=region_colors[algo],
                    label=f'{algo} optimal' if algo == highlight_algorithm else None
                )

    # Plot each algorithm
    for name, (comp_size, comp_throughput, color, linestyle) in algorithms.items():
        total_time = compute_total_time(comp_size, comp_throughput, bandwidth_bps, raw_size_mb)

        linewidth = 2.5 if name == highlight_algorithm else 1.5
        alpha = 1.0 if name == highlight_algorithm else 0.8

        ax.plot(
            bandwidth_gbps,
            total_time,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f'{name} ({comp_size} MB)',
            zorder=10 if name == highlight_algorithm else 5
        )

        # Add marker for highlighted algorithm
        if name == highlight_algorithm:
            # Add a star marker at a representative point
            idx = len(bandwidth_gbps) // 3
            ax.scatter(
                [bandwidth_gbps[idx]], [total_time[idx]],
                marker='*', s=200, color=color, zorder=15,
                edgecolors='black', linewidths=0.5
            )

    # Add bandwidth tier markers
    for label, bw_bps in bandwidth_tiers.items():
        bw_gbps = bw_bps / 1e9
        ax.axvline(
            x=bw_gbps,
            color='gray',
            linestyle=':',
            alpha=0.7,
            linewidth=1
        )
        # Add label at top
        ax.text(
            bw_gbps, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100,
            label,
            ha='center',
            va='bottom',
            fontsize=8,
            color='gray',
            rotation=0
        )

    # Find and annotate crossover points
    if show_crossovers:
        crossovers = find_crossover_points(algorithms, bandwidth_bps)
        for algo1, algo2, bw in crossovers:
            bw_gbps = bw / 1e9
            # Only annotate significant crossovers
            if algo1 in ['Zstd-9', 'Zstd-3', 'LZ4'] and algo2 in ['Zstd-9', 'Zstd-3', 'LZ4']:
                time_at_crossover = compute_total_time(
                    algorithms[algo1][0], algorithms[algo1][1], np.array([bw]), raw_size_mb
                )[0]
                ax.scatter([bw_gbps], [time_at_crossover], marker='o', s=50,
                          color='black', zorder=20)
                ax.annotate(
                    f'{algo1}→{algo2}\n({bw_gbps:.1f} Gbps)',
                    xy=(bw_gbps, time_at_crossover),
                    xytext=(bw_gbps * 1.5, time_at_crossover * 1.5),
                    fontsize=8,
                    ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                )

    # Add horizontal line for "acceptable latency" threshold
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(bandwidth_gbps[-1], 60, ' 60s threshold', va='bottom', ha='right',
            fontsize=8, color='red', alpha=0.5)

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Network Bandwidth (Gbps)')
    ax.set_ylabel('Total Synchronization Time (seconds)')
    ax.set_title('Compression Algorithm Selection by Network Bandwidth\n(7B Model Sparse Delta)')

    # Set axis limits
    ax.set_xlim(0.05, 200)
    ax.set_ylim(0.1, 200)

    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15)

    # Legend
    ax.legend(
        loc='upper right',
        framealpha=0.95,
        ncol=2,
        fontsize=8
    )

    # Add regime annotations
    ax.text(0.08, 0.15, 'Bandwidth-\nlimited', fontsize=9, style='italic',
            color='#9b59b6', alpha=0.7, ha='center')
    ax.text(1.5, 0.15, 'Balanced', fontsize=9, style='italic',
            color='#e74c3c', alpha=0.7, ha='center')
    ax.text(50, 0.15, 'Compute-\nlimited', fontsize=9, style='italic',
            color='#2ecc71', alpha=0.7, ha='center')

    plt.tight_layout()

    # Save figure
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save in multiple formats
        for fmt in ['pdf', 'png']:
            fig.savefig(
                output_path.with_suffix(f'.{fmt}'),
                format=fmt,
                dpi=300 if fmt == 'png' else None,
                bbox_inches='tight'
            )
        print(f"Saved figure to {output_path.with_suffix('.pdf')} and .png")

    return fig


def create_table_data(algorithms: Dict = ALGORITHMS) -> str:
    """Generate LaTeX table with algorithm comparison data."""

    latex = r"""
\begin{table}[t]
\centering
\caption{Compression algorithm characteristics for sparse delta checkpoints (7B model).
Cumulative ratio is relative to the full 14\,GB checkpoint.}
\label{tab:compression-algorithms}
\begin{tabular}{lrrrrrr}
\toprule
Algorithm & Size & Cumul. & Throughput & Comp. & Transfer & Best For \\
          & (MB) & Ratio  & (MB/s)     & (s)   & @1Gbps (s) & \\
\midrule
"""

    for name, (comp_size, throughput, _, _) in algorithms.items():
        cumul_ratio = FULL_CHECKPOINT_MB / comp_size
        comp_time = RAW_SIZE_MB / throughput
        transfer_time_1gbps = (comp_size * 8) / 1000  # MB to Gb, then divide by 1 Gbps

        # Determine best use case based on crossover analysis
        if throughput > 500:
            best_for = "$>$2\\,Gbps"
        elif throughput > 200:
            best_for = "0.5--2\\,Gbps"
        else:
            best_for = "$<$0.5\\,Gbps"

        highlight = r'\textbf{' if name == 'Zstd-3' else ''
        end_highlight = '}' if name == 'Zstd-3' else ''

        latex += f"{highlight}{name}{end_highlight} & {comp_size:.0f} & {cumul_ratio:.0f}$\\times$ & {throughput:.0f} & {comp_time:.1f} & {transfer_time_1gbps:.1f} & {best_for} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def load_benchmark_data(csv_path: str) -> Dict:
    """
    Load actual benchmark data from CSV file.

    Expected CSV columns:
    - algorithm: Algorithm name
    - compressed_size_bytes: Compressed size
    - compression_time_ms: Compression time in milliseconds
    - raw_size_bytes: Original size

    Returns:
        Dict in same format as ALGORITHMS
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Aggregate by algorithm (median across all files)
    agg = df.groupby('algorithm').agg({
        'compressed_size_bytes': 'median',
        'compression_time_ms': 'median',
        'raw_size_bytes': 'median',
    }).reset_index()

    # Define colors for algorithms
    colors = {
        'lz4': ('#2ecc71', '-'),
        'snappy': ('#27ae60', '--'),
        'zstd-1': ('#3498db', '-'),
        'zstd-3': ('#e74c3c', '-'),
        'zstd-9': ('#9b59b6', '-'),
        'gzip-6': ('#f39c12', '--'),
        'brotli-1': ('#1abc9c', '--'),
    }

    algorithms = {}
    for _, row in agg.iterrows():
        algo = row['algorithm']
        comp_size_mb = row['compressed_size_bytes'] / 1e6
        raw_size_mb = row['raw_size_bytes'] / 1e6
        comp_time_s = row['compression_time_ms'] / 1000
        throughput_mb_s = raw_size_mb / comp_time_s

        color, ls = colors.get(algo.lower(), ('#333333', '-'))

        # Normalize algorithm name for display
        display_name = algo.replace('zstd-', 'Zstd-').replace('gzip-', 'Gzip-').replace('brotli-', 'Brotli-')
        display_name = display_name.replace('lz4', 'LZ4').replace('snappy', 'Snappy')

        algorithms[display_name] = (comp_size_mb, throughput_mb_s, color, ls)

    return algorithms


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate compression bandwidth trade-off figure for paper'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/figures/compression_bandwidth_tradeoff',
        help='Output path (without extension)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to benchmark CSV (optional, uses hardcoded data if not provided)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot interactively'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Print LaTeX table to stdout'
    )
    args = parser.parse_args()

    # Load data
    if args.csv and Path(args.csv).exists():
        print(f"Loading benchmark data from {args.csv}")
        algorithms = load_benchmark_data(args.csv)
    else:
        print("Using hardcoded benchmark data")
        algorithms = ALGORITHMS

    # Print summary
    print("\nAlgorithm Summary:")
    print("-" * 85)
    print(f"{'Algorithm':<12} {'Compressed':<12} {'Comp Ratio':<12} {'Cumul Ratio':<12} {'Throughput':<12} {'Comp Time':<10}")
    print(f"{'':12} {'(MB)':<12} {'(sparse)':<12} {'(full ckpt)':<12} {'(MB/s)':<12} {'(s)':<10}")
    print("-" * 85)

    for name, (comp_size, throughput, _, _) in algorithms.items():
        comp_ratio = RAW_SIZE_MB / comp_size
        cumul_ratio = FULL_CHECKPOINT_MB / comp_size
        comp_time = RAW_SIZE_MB / throughput
        print(f"{name:<12} {comp_size:<12.0f} {comp_ratio:.1f}x{'':<9} {cumul_ratio:.0f}x{'':<10} {throughput:<12.0f} {comp_time:<10.2f}")

    # Create figure
    fig = create_figure(
        algorithms=algorithms,
        output_path=args.output,
        show_crossovers=True,
        show_optimal_regions=True,
    )

    # Print LaTeX table
    if args.latex:
        print("\n" + "=" * 70)
        print("LaTeX Table:")
        print("=" * 70)
        print(create_table_data(algorithms))

    # Show plot
    if args.show:
        plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
