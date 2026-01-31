#!/usr/bin/env python3
"""
Generate publication-quality visualizations for v3 compression benchmark.
Includes threading comparison and algorithm analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# =============================================================================
# COVENANT LABS COLOR PALETTE
# =============================================================================
COVENANT_RED = '#FF3A3A'
COVENANT_RED_LIGHT = '#FF6B6B'
COVENANT_RED_DARK = '#CC2020'
COVENANT_BLACK_1000 = '#101010'
COVENANT_BLACK_800 = '#333333'
COVENANT_BLACK_600 = '#555555'
COVENANT_BLACK_500 = '#828282'
COVENANT_BLACK_400 = '#A0A0A0'
COVENANT_BLACK_300 = '#BBBBBB'
COVENANT_WHITE_100 = '#DDDDDD'

# Algorithm colors
ALGO_COLORS = {
    'zstd-1': COVENANT_RED,
    'zstd-3': '#CC4444',
    'zstd-9': '#994444',
    'lz4': COVENANT_BLACK_500,
    'lz4hc': COVENANT_BLACK_500,
    'snappy': COVENANT_BLACK_400,
    'gzip-6': COVENANT_BLACK_400,
    'brotli-1': COVENANT_BLACK_300,
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.2,
    'axes.edgecolor': COVENANT_BLACK_500,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'grid.color': COVENANT_WHITE_100,
    'grid.linestyle': '--',
})


def setup_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COVENANT_BLACK_500)
    ax.spines['bottom'].set_color(COVENANT_BLACK_500)
    ax.tick_params(colors=COVENANT_BLACK_800, width=1.2, length=6)
    if title:
        ax.set_title(title, color=COVENANT_BLACK_1000, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, color=COVENANT_BLACK_1000)
    if ylabel:
        ax.set_ylabel(ylabel, color=COVENANT_BLACK_1000)
    ax.yaxis.grid(True, alpha=0.7)
    ax.xaxis.grid(False)


def save_figure(fig, output_dir: Path, name: str):
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {name}.png, {name}.pdf')


def plot_threading_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare single-threaded vs multi-threaded zstd."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Filter to sparse_coo (best representation)
    coo_df = df[df['representation'] == 'sparse_coo']

    for ax, algo in zip(axes, ['zstd-1', 'zstd-3', 'zstd-9']):
        algo_df = coo_df[coo_df['algorithm'] == algo]

        # Group by threads
        t0 = algo_df[algo_df['threads'] == 0]['throughput_compress_mb_s']
        t4 = algo_df[algo_df['threads'] == 4]['throughput_compress_mb_s']

        x = [0, 1]
        heights = [t0.mean(), t4.mean()]
        errors = [t0.std(), t4.std()]
        colors = [COVENANT_BLACK_500, COVENANT_RED]

        bars = ax.bar(x, heights, yerr=errors, capsize=5, color=colors,
                      edgecolor=COVENANT_BLACK_800, linewidth=1)

        # Add value labels
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Speedup annotation
        speedup = heights[1] / heights[0] if heights[0] > 0 else 0
        color = COVENANT_RED if speedup > 1 else COVENANT_BLACK_500
        ax.text(0.5, max(heights) * 0.5, f'{speedup:.2f}x', ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(['1 Thread', '4 Threads'])
        setup_axes(ax, title=algo, ylabel='Throughput (MB/s)' if algo == 'zstd-1' else None)

    fig.suptitle('Multi-Threading Impact on zstd Compression Speed (sparse_coo)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, output_dir, '11_threading_comparison')


def plot_algorithm_pareto(df: pd.DataFrame, output_dir: Path):
    """Pareto frontier of all algorithms."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to sparse_coo, single-threaded
    coo_df = df[(df['representation'] == 'sparse_coo') & (df['threads'] == 0)]

    agg = coo_df.groupby('algorithm').agg({
        'compression_ratio': 'mean',
        'throughput_compress_mb_s': 'mean',
    }).reset_index()

    for _, row in agg.iterrows():
        algo = row['algorithm']
        color = ALGO_COLORS.get(algo, COVENANT_BLACK_400)
        is_zstd = 'zstd' in algo

        ax.scatter(row['throughput_compress_mb_s'], row['compression_ratio'],
                   c=color, s=200 if is_zstd else 120,
                   marker='s' if is_zstd else 'o',
                   edgecolors=COVENANT_BLACK_1000 if is_zstd else 'none',
                   linewidths=2 if is_zstd else 0, zorder=3 if is_zstd else 2,
                   alpha=1.0 if is_zstd else 0.7)

        # Label
        offset = (15, 10) if row['throughput_compress_mb_s'] < 1000 else (-60, 10)
        ax.annotate(algo, (row['throughput_compress_mb_s'], row['compression_ratio']),
                    textcoords='offset points', xytext=offset, fontsize=10,
                    color=color, fontweight='bold' if is_zstd else 'normal')

    setup_axes(ax,
               title='Algorithm Comparison: Compression Ratio vs Speed (sparse_coo)',
               xlabel='Compression Throughput (MB/s)',
               ylabel='Compression Ratio (×)')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COVENANT_RED,
               markersize=12, markeredgecolor=COVENANT_BLACK_1000, markeredgewidth=2,
               label='zstd (recommended)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_400,
               markersize=10, label='Other algorithms'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    save_figure(fig, output_dir, '12_algorithm_pareto')


def plot_representation_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare sparse_coo vs sparse_flat representations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to zstd-3, single-threaded (best tradeoff)
    zstd3_df = df[(df['algorithm'] == 'zstd-3') & (df['threads'] == 0)]

    # Left: Compression ratio
    agg_ratio = zstd3_df.groupby('representation')['compression_ratio'].agg(['mean', 'std'])
    colors = [COVENANT_RED if r == 'sparse_coo' else COVENANT_BLACK_400
              for r in agg_ratio.index]

    bars1 = ax1.bar(agg_ratio.index, agg_ratio['mean'], yerr=agg_ratio['std'],
                    capsize=5, color=colors, edgecolor=COVENANT_BLACK_800)

    for bar, val in zip(bars1, agg_ratio['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}×', ha='center', va='bottom', fontsize=14, fontweight='bold')

    setup_axes(ax1, title='Compression Ratio', ylabel='Compression Ratio (×)')

    # Right: Speed
    agg_speed = zstd3_df.groupby('representation')['throughput_compress_mb_s'].agg(['mean', 'std'])
    colors = [COVENANT_RED if r == 'sparse_coo' else COVENANT_BLACK_400
              for r in agg_speed.index]

    bars2 = ax2.bar(agg_speed.index, agg_speed['mean'], yerr=agg_speed['std'],
                    capsize=5, color=colors, edgecolor=COVENANT_BLACK_800)

    for bar, val in zip(bars2, agg_speed['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    setup_axes(ax2, title='Compression Speed', ylabel='Throughput (MB/s)')

    fig.suptitle('Sparse Representation Comparison (zstd-3)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, output_dir, '13_representation_comparison')


def plot_full_algorithm_bars(df: pd.DataFrame, output_dir: Path):
    """Full algorithm comparison with all metrics."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter to sparse_coo, single-threaded
    coo_df = df[(df['representation'] == 'sparse_coo') & (df['threads'] == 0)]

    agg = coo_df.groupby('algorithm').agg({
        'compression_ratio': 'mean',
        'throughput_compress_mb_s': 'mean',
    }).reset_index()

    agg = agg.sort_values('compression_ratio', ascending=True)

    y_pos = np.arange(len(agg))
    colors = [ALGO_COLORS.get(a, COVENANT_BLACK_400) for a in agg['algorithm']]

    bars = ax.barh(y_pos, agg['compression_ratio'], color=colors,
                   edgecolor=COVENANT_BLACK_800, linewidth=0.5)

    for i, (bar, ratio, speed, algo) in enumerate(zip(bars, agg['compression_ratio'],
                                                       agg['throughput_compress_mb_s'],
                                                       agg['algorithm'])):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{ratio:.1f}× @ {speed:.0f} MB/s',
                va='center', ha='left', fontsize=10, color=COVENANT_BLACK_800)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['algorithm'])
    ax.set_xlim(0, max(agg['compression_ratio']) * 1.4)

    setup_axes(ax,
               title='Algorithm Comparison: sparse_coo Representation',
               xlabel='Compression Ratio (×)')

    # Highlight zstd
    legend_elements = [
        mpatches.Patch(facecolor=COVENANT_RED, edgecolor=COVENANT_BLACK_800,
                       label='zstd (recommended)'),
        mpatches.Patch(facecolor=COVENANT_BLACK_400, edgecolor=COVENANT_BLACK_800,
                       label='Other algorithms'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    save_figure(fig, output_dir, '14_algorithm_bars')


def plot_sparse_vs_dense(df: pd.DataFrame, output_dir: Path):
    """Compare sparse representations vs dense."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter to zstd-3, single-threaded
    zstd3_df = df[(df['algorithm'] == 'zstd-3') & (df['threads'] == 0)]

    agg = zstd3_df.groupby('representation')['compression_ratio'].mean().sort_values(ascending=False)

    colors = [COVENANT_RED if 'coo' in r else
              COVENANT_RED_LIGHT if 'flat' in r else
              COVENANT_BLACK_400 for r in agg.index]

    bars = ax.bar(agg.index, agg.values, color=colors, edgecolor=COVENANT_BLACK_800, linewidth=1)

    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add gain annotations
    if 'sparse_coo' in agg.index and 'dense' in agg.index:
        gain = agg['sparse_coo'] / agg['dense']
        ax.annotate(f'{gain:.0f}× better\nthan dense!',
                    xy=(0, agg['sparse_coo']), xytext=(0.5, agg['sparse_coo'] - 3),
                    fontsize=12, fontweight='bold', color=COVENANT_RED,
                    arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=2))

    setup_axes(ax,
               title='Sparsity Benefit: Compression Ratio by Representation (zstd-3)',
               ylabel='Compression Ratio (×)')

    ax.set_xticklabels(['sparse_coo\n(2D indices)', 'sparse_flat\n(1D indices)', 'dense\n(full tensor)'],
                       rotation=0)

    save_figure(fig, output_dir, '15_sparse_vs_dense')


def plot_summary_heatmap(df: pd.DataFrame, output_dir: Path):
    """Summary heatmap of all configurations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to single-threaded
    t0_df = df[df['threads'] == 0]

    pivot = t0_df.pivot_table(
        values='compression_ratio',
        index='algorithm',
        columns='representation',
        aggfunc='mean'
    ).round(1)

    # Sort by sparse_coo ratio
    if 'sparse_coo' in pivot.columns:
        pivot = pivot.sort_values('sparse_coo', ascending=False)

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', ax=ax,
                cbar_kws={'label': 'Compression Ratio (×)'},
                linewidths=0.5, linecolor=COVENANT_WHITE_100,
                annot_kws={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_title('Compression Ratio: Algorithm × Representation',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Representation', fontweight='bold')
    ax.set_ylabel('Algorithm', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_dir, '16_summary_heatmap')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/root/grail/data/compression_benchmark_v3.csv')
    parser.add_argument('--output-dir', type=str, default='/root/grail/research/figures/compression_v2')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {args.input}...')
    df = pd.read_csv(args.input)
    print(f'Loaded {len(df)} rows')

    print(f'\nGenerating v3 figures to {output_dir}...\n')

    plot_threading_comparison(df, output_dir)
    plot_algorithm_pareto(df, output_dir)
    plot_representation_comparison(df, output_dir)
    plot_full_algorithm_bars(df, output_dir)
    plot_sparse_vs_dense(df, output_dir)
    plot_summary_heatmap(df, output_dir)

    print(f'\nDone! Generated 6 additional figures')

    # Print key findings
    print('\n' + '=' * 70)
    print('V3 KEY FINDINGS')
    print('=' * 70)

    coo_t0 = df[(df['representation'] == 'sparse_coo') & (df['threads'] == 0)]
    best = coo_t0.groupby('algorithm')['compression_ratio'].mean().sort_values(ascending=False)
    print(f'\nBest algorithm (sparse_coo): {best.index[0]} ({best.iloc[0]:.1f}×)')

    # Threading impact
    for algo in ['zstd-1', 'zstd-3', 'zstd-9']:
        t0 = df[(df['algorithm'] == algo) & (df['threads'] == 0) &
                (df['representation'] == 'sparse_coo')]['throughput_compress_mb_s'].mean()
        t4 = df[(df['algorithm'] == algo) & (df['threads'] == 4) &
                (df['representation'] == 'sparse_coo')]['throughput_compress_mb_s'].mean()
        print(f'{algo} threading speedup: {t4/t0:.2f}x')


if __name__ == '__main__':
    main()
