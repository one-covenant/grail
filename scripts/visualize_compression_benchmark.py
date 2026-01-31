#!/usr/bin/env python3
"""
Generate publication-quality visualizations for compression benchmark results.
Using Covenant Labs brand color palette.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
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
COVENANT_WHITE_50 = '#F5F5F5'

# Algorithm colors - gradient from red to black
ALGO_COLORS = {
    'zstd-1': COVENANT_RED,           # Best compression - highlight red
    'zstd-3': '#CC4444',              # Medium red
    'zstd-9': '#994444',              # Dark red
    'lz4': COVENANT_BLACK_600,        # Fast - dark gray
    'none': COVENANT_BLACK_400,       # Baseline - light gray
}

# Representation colors
REP_COLORS = {
    '2d_coo': COVENANT_BLACK_600,
    'flat': COVENANT_RED,
}

# Model colors
MODEL_COLORS = {
    'qwen2.5': COVENANT_RED,
    'llama3.2': COVENANT_BLACK_600,
    'gemma3': COVENANT_BLACK_400,
}

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================
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
    'axes.facecolor': 'white',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'grid.color': COVENANT_WHITE_100,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
})


def setup_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to axes."""
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


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess benchmark data."""
    df = pd.read_csv(csv_path)
    df['config'] = df['representation'] + '+' + df['stage'] + '+' + df['algorithm']
    return df


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Pareto frontier of compression ratio vs speed."""
    fig, ax = plt.subplots(figsize=(10, 7))
    setup_axes(ax,
               title='Compression-Speed Tradeoff',
               xlabel='Compression Throughput (MB/s)',
               ylabel='Compression Ratio (×)')

    # Aggregate by config
    agg = df.groupby(['representation', 'stage', 'algorithm']).agg({
        'compression_ratio': 'mean',
        'throughput_compress_mb_s': 'mean',
    }).reset_index()

    # Filter out 'none' algorithm
    agg = agg[agg['algorithm'] != 'none']

    # Plot points
    for _, row in agg.iterrows():
        is_delta = row['stage'] == 'delta_encode'
        color = ALGO_COLORS.get(row['algorithm'], COVENANT_BLACK_500)

        # Marker style based on representation
        marker = 's' if row['representation'] == 'flat' else 'o'

        # Size and alpha based on delta encoding
        size = 180 if is_delta else 60
        alpha = 1.0 if is_delta else 0.3

        ax.scatter(row['throughput_compress_mb_s'], row['compression_ratio'],
                   c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors=COVENANT_BLACK_1000 if is_delta else 'none',
                   linewidths=1.5 if is_delta else 0, zorder=3 if is_delta else 2)

    # Highlight Pareto optimal points with annotations
    pareto_configs = [
        ('flat', 'delta_encode', 'zstd-1', 'zstd-1 (8.7×)\nBest ratio', (15, 15)),
        ('2d_coo', 'delta_encode', 'lz4', 'lz4 (4.0×)\nFastest', (15, -25)),
    ]

    for rep, stage, algo, label, offset in pareto_configs:
        row = agg[(agg['representation'] == rep) & (agg['stage'] == stage) & (agg['algorithm'] == algo)]
        if len(row) > 0:
            x, y = row['throughput_compress_mb_s'].values[0], row['compression_ratio'].values[0]
            ax.annotate(label, (x, y),
                       textcoords="offset points", xytext=offset,
                       fontsize=10, fontweight='bold', color=COVENANT_BLACK_1000,
                       ha='left',
                       arrowprops=dict(arrowstyle='->', color=COVENANT_BLACK_600, lw=1.2),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COVENANT_BLACK_300))

    # Draw Pareto frontier line
    pareto_points = []
    for rep, stage, algo, _, _ in pareto_configs:
        row = agg[(agg['representation'] == rep) & (agg['stage'] == stage) & (agg['algorithm'] == algo)]
        if len(row) > 0:
            pareto_points.append((row['throughput_compress_mb_s'].values[0], row['compression_ratio'].values[0]))

    if len(pareto_points) >= 2:
        pareto_points.sort()
        px, py = zip(*pareto_points)
        ax.plot(px, py, '--', color=COVENANT_RED, alpha=0.5, linewidth=2, label='Pareto frontier')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_RED, markersize=10, label='zstd-1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC4444', markersize=10, label='zstd-3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#994444', markersize=10, label='zstd-9'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_600, markersize=10, label='lz4'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_500, markersize=12,
               markeredgecolor=COVENANT_BLACK_1000, markeredgewidth=1.5, label='With delta enc.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_500, markersize=8, alpha=0.4, label='Without delta'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COVENANT_BLACK_500, markersize=10, label='Flat repr.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_500, markersize=10, label='2D COO repr.'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              edgecolor=COVENANT_BLACK_300, fancybox=False)

    ax.set_xlim(0, 450)
    ax.set_ylim(1, 10)

    plt.tight_layout()
    plt.savefig(output_dir / '01_pareto_frontier.png')
    plt.savefig(output_dir / '01_pareto_frontier.pdf')
    plt.close()
    print('Created: 01_pareto_frontier.png')


def plot_attribution_bars(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Bar chart showing compression attribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_axes(ax,
               title='Compression Pipeline: Cumulative Gains',
               xlabel='',
               ylabel='Compression Ratio (×)')

    # Calculate values
    baseline = 1.0
    zstd9_only = df[(df['representation'] == '2d_coo') & (df['stage'] == 'raw') & (df['algorithm'] == 'zstd-9')]['compression_ratio'].mean()
    delta_zstd9 = df[(df['representation'] == '2d_coo') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-9')]['compression_ratio'].mean()
    flat_only = df[(df['representation'] == 'flat') & (df['stage'] == 'raw') & (df['algorithm'] == 'none')]['compression_ratio'].mean()
    flat_delta_zstd1 = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]['compression_ratio'].mean()

    # Data
    labels = ['Baseline\n(2D COO)', '+zstd-9', '+Delta\nEncoding', 'Flat\nRepr.', 'Flat+Delta\n+zstd-1']
    values = [baseline, zstd9_only, delta_zstd9, flat_only, flat_delta_zstd1]
    colors = [COVENANT_BLACK_400, COVENANT_BLACK_600, '#994444', COVENANT_BLACK_500, COVENANT_RED]

    # Create bars
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=COVENANT_BLACK_1000, linewidth=1.2, width=0.65)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color=COVENANT_BLACK_1000)

    # Add improvement arrows
    improvements = [
        (0, 1, f'+{zstd9_only - baseline:.1f}×'),
        (1, 2, f'+{delta_zstd9 - zstd9_only:.1f}×'),
    ]

    for start, end, label in improvements:
        y_start = values[start] + 0.3
        y_end = values[end] - 0.1
        x_mid = (start + end) / 2

        ax.annotate('', xy=(end, y_end), xytext=(start, y_start),
                    arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=2))
        ax.text(x_mid, (y_start + y_end) / 2 + 0.3, label, ha='center', fontsize=10,
                color=COVENANT_RED, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 11)
    ax.axhline(y=1, color=COVENANT_BLACK_300, linestyle='--', linewidth=1, alpha=0.7)

    # Highlight best
    bars[-1].set_edgecolor(COVENANT_RED_DARK)
    bars[-1].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(output_dir / '02_attribution_bars.png')
    plt.savefig(output_dir / '02_attribution_bars.pdf')
    plt.close()
    print('Created: 02_attribution_bars.png')


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot 3: Heatmap of all algorithm × representation × stage combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Custom colormap: white -> red
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ['#FFFFFF', '#FFEEEE', '#FFCCCC', '#FF9999', '#FF6666', COVENANT_RED]
    cmap = LinearSegmentedColormap.from_list('covenant', colors_cmap)

    for idx, rep in enumerate(['2d_coo', 'flat']):
        ax = axes[idx]

        # Pivot data
        subset = df[df['representation'] == rep]
        pivot = subset.pivot_table(values='compression_ratio',
                                    index='stage',
                                    columns='algorithm',
                                    aggfunc='mean')

        # Reorder
        stage_order = ['raw', 'downcast', 'delta_encode']
        algo_order = ['none', 'lz4', 'zstd-1', 'zstd-3', 'zstd-9']
        pivot = pivot.reindex(index=stage_order, columns=[a for a in algo_order if a in pivot.columns])

        # Rename for display
        pivot.index = ['Raw', 'Downcast', 'Delta Encode']

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                    vmin=1, vmax=10, ax=ax,
                    cbar_kws={'label': 'Compression Ratio (×)', 'shrink': 0.8},
                    linewidths=2, linecolor='white',
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})

        title = '2D COO Representation' if rep == '2d_coo' else 'Flat Representation'
        ax.set_title(title, fontsize=14, fontweight='bold', color=COVENANT_BLACK_1000, pad=10)
        ax.set_xlabel('Algorithm', fontsize=12, color=COVENANT_BLACK_1000)
        ax.set_ylabel('Pipeline Stage', fontsize=12, color=COVENANT_BLACK_1000)
        ax.tick_params(colors=COVENANT_BLACK_800)

        # Highlight best cell
        best_val = pivot.max().max()
        for i, stage in enumerate(pivot.index):
            for j, algo in enumerate(pivot.columns):
                if abs(pivot.loc[stage, algo] - best_val) < 0.01:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor=COVENANT_BLACK_1000, lw=3))

    plt.suptitle('Compression Ratio: All Configurations', fontsize=16, fontweight='bold',
                 color=COVENANT_BLACK_1000, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '03_heatmap.png')
    plt.savefig(output_dir / '03_heatmap.pdf')
    plt.close()
    print('Created: 03_heatmap.png')


def plot_bandwidth_crossover(df: pd.DataFrame, output_dir: Path):
    """Plot 4: Line plot showing optimal method at different bandwidths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_axes(ax,
               title='Bandwidth-Aware Algorithm Selection',
               xlabel='Network Bandwidth (Mbps)',
               ylabel='Total Transfer Time for 100MB (seconds)')

    # Get key configs
    configs = [
        ('flat', 'delta_encode', 'zstd-1', 'flat+delta+zstd-1', COVENANT_RED, '-', 2.5),
        ('flat', 'delta_encode', 'zstd-9', 'flat+delta+zstd-9', '#994444', '--', 2),
        ('2d_coo', 'delta_encode', 'lz4', '2d_coo+delta+lz4', COVENANT_BLACK_600, '-', 2.5),
        ('flat', 'raw', 'none', 'No compression', COVENANT_BLACK_400, ':', 2),
    ]

    bandwidths = np.logspace(np.log10(5), np.log10(2000), 100)

    for rep, stage, algo, label, color, linestyle, lw in configs:
        subset = df[(df['representation'] == rep) & (df['stage'] == stage) & (df['algorithm'] == algo)]
        if len(subset) == 0:
            continue

        ratio = subset['compression_ratio'].mean()
        comp_speed = subset['throughput_compress_mb_s'].mean() if algo != 'none' else float('inf')
        decomp_speed = subset['throughput_decompress_mb_s'].mean() if algo != 'none' else float('inf')

        times = []
        for bw_mbps in bandwidths:
            bw_MBps = bw_mbps / 8
            raw_size = 100

            if comp_speed == float('inf'):
                comp_time = 0
                decomp_time = 0
                compressed_size = raw_size / ratio
            else:
                compressed_size = raw_size / ratio
                comp_time = raw_size / comp_speed
                decomp_time = compressed_size / decomp_speed

            transfer_time = compressed_size / bw_MBps
            total_time = comp_time + transfer_time + decomp_time
            times.append(total_time)

        ax.plot(bandwidths, times, color=color, linestyle=linestyle, linewidth=lw, label=label)

    # Mark crossover regions with subtle shading
    ax.axvspan(5, 347, alpha=0.08, color=COVENANT_RED, zorder=0)
    ax.axvspan(347, 605, alpha=0.08, color=COVENANT_BLACK_600, zorder=0)
    ax.axvspan(605, 2000, alpha=0.08, color=COVENANT_BLACK_400, zorder=0)

    # Crossover lines
    for xval, label in [(347, '347 Mbps'), (605, '605 Mbps')]:
        ax.axvline(x=xval, color=COVENANT_BLACK_300, linestyle='--', linewidth=1, alpha=0.8)
        ax.text(xval, 48, label, fontsize=9, ha='center', color=COVENANT_BLACK_500,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Region labels
    ax.text(50, 2, 'zstd-1\noptimal', fontsize=11, ha='center', color=COVENANT_RED, fontweight='bold')
    ax.text(460, 2, 'lz4\noptimal', fontsize=11, ha='center', color=COVENANT_BLACK_600, fontweight='bold')
    ax.text(1100, 2, 'No compression\noptimal', fontsize=11, ha='center', color=COVENANT_BLACK_500, fontweight='bold')

    ax.set_xscale('log')
    ax.legend(loc='upper right', frameon=True, edgecolor=COVENANT_BLACK_300)
    ax.set_xlim(5, 2000)
    ax.set_ylim(0, 55)

    plt.tight_layout()
    plt.savefig(output_dir / '04_bandwidth_crossover.png')
    plt.savefig(output_dir / '04_bandwidth_crossover.pdf')
    plt.close()
    print('Created: 04_bandwidth_crossover.png')


def plot_variance_violin(df: pd.DataFrame, output_dir: Path):
    """Plot 5: Violin plot showing variance by experiment."""
    fig, ax = plt.subplots(figsize=(14, 6))
    setup_axes(ax,
               title='Compression Ratio Distribution by Experiment',
               xlabel='Experiment',
               ylabel='Compression Ratio (×)')

    config = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]

    # Sort by mean
    exp_order = config.groupby('experiment')['compression_ratio'].mean().sort_values(ascending=False).index.tolist()

    # Shorten names
    name_map = {
        'qwen2.5-1.5b-sft-math-lr2e-05': 'Qwen-1.5B\nSFT-2e-5',
        'qwen2.5-1.5b-sft-math-lr3e-06': 'Qwen-1.5B\nSFT-3e-6',
        'gemma3-4b-iter1': 'Gemma-4B',
        'qwen2.5-1.5b-iter32': 'Qwen-1.5B\niter32',
        'qwen2.5-1.5b-iter16': 'Qwen-1.5B\niter16',
        'qwen2.5-1.5b-iter8': 'Qwen-1.5B\niter8',
        'llama3.2-3b-iter1': 'Llama-3B',
        'qwen2.5-1.5b-lr5e-6': 'Qwen-1.5B\nlr5e-6',
        'qwen2.5-7b-grpo-math-lr3e-06': 'Qwen-7B',
        'qwen2.5-1.5b-lr1e-6': 'Qwen-1.5B\nlr1e-6',
        'qwen2.5-1.5b-lr5e-7': 'Qwen-1.5B\nlr5e-7',
    }

    config = config.copy()
    config['exp_short'] = config['experiment'].map(name_map)
    exp_order_short = [name_map[e] for e in exp_order]

    # Create color palette based on model family
    palette = []
    for exp in exp_order:
        if 'gemma' in exp:
            palette.append(COVENANT_BLACK_400)
        elif 'llama' in exp:
            palette.append(COVENANT_BLACK_600)
        else:
            palette.append(COVENANT_RED)

    # Violin plot
    parts = ax.violinplot([config[config['exp_short'] == e]['compression_ratio'].values for e in exp_order_short],
                          positions=range(len(exp_order_short)), showmeans=True, showmedians=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor(COVENANT_BLACK_1000)
        pc.set_linewidth(1)

    parts['cmeans'].set_color(COVENANT_BLACK_1000)
    parts['cmeans'].set_linewidth(2)
    parts['cbars'].set_color(COVENANT_BLACK_600)
    parts['cmins'].set_color(COVENANT_BLACK_600)
    parts['cmaxes'].set_color(COVENANT_BLACK_600)

    # Add individual points
    for i, exp in enumerate(exp_order_short):
        y = config[config['exp_short'] == exp]['compression_ratio'].values
        x = np.random.normal(i, 0.05, size=len(y))
        ax.scatter(x, y, c=COVENANT_BLACK_1000, alpha=0.3, s=20, zorder=3)

    ax.set_xticks(range(len(exp_order_short)))
    ax.set_xticklabels(exp_order_short, fontsize=9, rotation=45, ha='right')

    # Overall mean line
    overall_mean = config['compression_ratio'].mean()
    ax.axhline(y=overall_mean, color=COVENANT_RED, linestyle='--', linewidth=2, alpha=0.7,
               label=f'Overall mean: {overall_mean:.1f}×')
    ax.legend(loc='upper right', frameon=True, edgecolor=COVENANT_BLACK_300)

    ax.set_ylim(5, 15)

    plt.tight_layout()
    plt.savefig(output_dir / '05_variance_violin.png')
    plt.savefig(output_dir / '05_variance_violin.pdf')
    plt.close()
    print('Created: 05_variance_violin.png')


def plot_sparsity_scatter(df: pd.DataFrame, output_dir: Path):
    """Plot 6: Scatter plot of sparsity vs compression with regression."""
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_axes(ax,
               title='Sparsity vs Compression: Counter-intuitive Finding',
               xlabel='Sparsity (%)',
               ylabel='Compression Ratio (×)')

    config = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]

    # Plot by model
    for model in ['qwen2.5', 'llama3.2', 'gemma3']:
        subset = config[config['model'] == model]
        color = MODEL_COLORS.get(model, COVENANT_BLACK_500)
        label = model.replace('2.5', ' 2.5').replace('3.2', ' 3.2').replace('3', ' 3').title()
        ax.scatter(subset['sparsity'] * 100, subset['compression_ratio'],
                   c=color, alpha=0.6, s=60, label=label,
                   edgecolors='white', linewidths=0.5)

    # Regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        config['sparsity'] * 100, config['compression_ratio'])

    x_line = np.linspace(config['sparsity'].min() * 100, config['sparsity'].max() * 100, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color=COVENANT_BLACK_1000, linewidth=2,
            label=f'Trend line (r = {r_value:.2f})')

    # Annotation box
    textstr = 'Higher sparsity → Lower compression\n(fewer patterns for algorithm to exploit)'
    props = dict(boxstyle='round,pad=0.5', facecolor=COVENANT_WHITE_50,
                 edgecolor=COVENANT_BLACK_300, alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props,
            color=COVENANT_BLACK_1000)

    ax.legend(loc='upper right', frameon=True, edgecolor=COVENANT_BLACK_300)
    ax.set_xlim(95.5, 100)
    ax.set_ylim(5, 14)

    plt.tight_layout()
    plt.savefig(output_dir / '06_sparsity_scatter.png')
    plt.savefig(output_dir / '06_sparsity_scatter.pdf')
    plt.close()
    print('Created: 06_sparsity_scatter.png')


def plot_waterfall(df: pd.DataFrame, output_dir: Path):
    """Plot 7: Waterfall chart showing size reduction pipeline."""
    fig, ax = plt.subplots(figsize=(11, 6))
    setup_axes(ax,
               title='Size Reduction Pipeline: 100MB → 11.5MB (8.7× compression)',
               xlabel='',
               ylabel='Size (MB)')

    # Calculate sizes
    flat_ratio = df[(df['representation'] == 'flat') & (df['stage'] == 'raw') & (df['algorithm'] == 'none')]['compression_ratio'].mean()
    final_ratio = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]['compression_ratio'].mean()

    coo_raw = 100
    flat_raw = coo_raw / flat_ratio
    final_size = coo_raw / final_ratio

    # Waterfall data
    stages = ['2D COO\nBaseline', 'Convert to\nFlat Indices', 'Apply Delta\nEncoding', 'Apply\nzstd-1', 'Final\nSize']

    # Starting values, changes
    starts = [0, coo_raw, flat_raw, flat_raw, 0]
    heights = [coo_raw, -(coo_raw - flat_raw), 0, -(flat_raw - final_size), final_size]

    colors = [COVENANT_BLACK_500, COVENANT_RED_LIGHT, COVENANT_BLACK_400, COVENANT_RED, COVENANT_RED_DARK]

    x = np.arange(len(stages))

    # Draw bars
    for i, (start, height, color) in enumerate(zip(starts, heights, colors)):
        if i == 0:  # First bar - full height from 0
            ax.bar(i, coo_raw, bottom=0, color=color, edgecolor=COVENANT_BLACK_1000, linewidth=1.2, width=0.6)
        elif i == len(stages) - 1:  # Last bar - final size
            ax.bar(i, final_size, bottom=0, color=color, edgecolor=COVENANT_BLACK_1000, linewidth=1.2, width=0.6)
        elif height < 0:  # Reduction
            ax.bar(i, -height, bottom=start + height, color=color, edgecolor=COVENANT_BLACK_1000, linewidth=1.2, width=0.6)
            # Show the reduction amount
            ax.text(i, start + height/2, f'{-height:.0f}MB\n({-height/coo_raw*100:.0f}%)',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        else:  # No change
            ax.bar(i, 3, bottom=flat_raw - 1.5, color=color, edgecolor=COVENANT_BLACK_1000, linewidth=1.2, width=0.6, alpha=0.5)
            ax.text(i, flat_raw, 'Enables\ncompression', ha='center', va='center', fontsize=9, color=COVENANT_BLACK_600)

    # Value labels on top
    size_labels = [coo_raw, flat_raw, flat_raw, final_size, final_size]
    for i, (stage, size) in enumerate(zip(stages, size_labels)):
        if i != 2:  # Skip delta encode (no size change)
            y_pos = size + 3 if i < 3 else size + 3
            ax.text(i, y_pos, f'{size:.1f} MB', ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=COVENANT_BLACK_1000)

    # Connecting lines
    for i in range(len(stages) - 1):
        if i == 0:
            y = flat_raw
        elif i == 1 or i == 2:
            y = flat_raw
        else:
            y = final_size
        ax.plot([i + 0.35, i + 0.65], [y, y], color=COVENANT_BLACK_600, linewidth=1.5, linestyle='--')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.5, len(stages) - 0.5)

    plt.tight_layout()
    plt.savefig(output_dir / '07_waterfall.png')
    plt.savefig(output_dir / '07_waterfall.pdf')
    plt.close()
    print('Created: 07_waterfall.png')


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot 8: Multi-panel comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    config = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]

    models = [
        ('qwen2.5', 'Qwen 2.5', COVENANT_RED),
        ('llama3.2', 'Llama 3.2', COVENANT_BLACK_600),
        ('gemma3', 'Gemma 3', COVENANT_BLACK_400),
    ]

    for ax, (model, label, color) in zip(axes, models):
        subset = config[config['model'] == model]

        # Box plot
        bp = ax.boxplot([subset['compression_ratio']], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][0].set_edgecolor(COVENANT_BLACK_1000)
        bp['boxes'][0].set_linewidth(1.5)
        bp['medians'][0].set_color(COVENANT_BLACK_1000)
        bp['medians'][0].set_linewidth(2)
        for whisker in bp['whiskers']:
            whisker.set_color(COVENANT_BLACK_600)
            whisker.set_linewidth(1.5)
        for cap in bp['caps']:
            cap.set_color(COVENANT_BLACK_600)
            cap.set_linewidth(1.5)

        # Scatter points
        x = np.random.normal(1, 0.06, size=len(subset))
        ax.scatter(x, subset['compression_ratio'], c=color, alpha=0.4, s=40, zorder=3,
                   edgecolors='white', linewidths=0.3)

        mean = subset['compression_ratio'].mean()
        std = subset['compression_ratio'].std()

        setup_axes(ax, title=f'{label}\n{mean:.1f}× ± {std:.1f}')
        ax.set_ylabel('Compression Ratio (×)' if ax == axes[0] else '')
        ax.set_xticks([])
        ax.set_ylim(5, 14)
        ax.text(1, 5.3, f'n={len(subset)}', ha='center', fontsize=9, color=COVENANT_BLACK_500)

    plt.suptitle('Compression Consistency Across Model Architectures', fontsize=16,
                 fontweight='bold', color=COVENANT_BLACK_1000, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '08_model_comparison.png')
    plt.savefig(output_dir / '08_model_comparison.pdf')
    plt.close()
    print('Created: 08_model_comparison.png')


def plot_decision_flowchart(df: pd.DataFrame, output_dir: Path):
    """Plot 9: Decision flowchart for algorithm selection."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    def draw_box(x, y, w, h, text, facecolor='white', edgecolor=COVENANT_BLACK_600, fontsize=11, fontcolor=COVENANT_BLACK_1000):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=fontcolor, linespacing=1.3)

    # Title
    ax.text(6, 9.5, 'Algorithm Selection Decision Guide', fontsize=18, fontweight='bold',
            ha='center', color=COVENANT_BLACK_1000)

    # Decision node
    draw_box(6, 8, 3.5, 1.2, 'What is your\nnetwork bandwidth?',
             facecolor=COVENANT_WHITE_50, edgecolor=COVENANT_BLACK_600)

    # Branch labels
    ax.text(2.5, 6.8, '< 350 Mbps', fontsize=11, ha='center', fontweight='bold', color=COVENANT_BLACK_600)
    ax.text(6, 6.8, '350 - 600 Mbps', fontsize=11, ha='center', fontweight='bold', color=COVENANT_BLACK_600)
    ax.text(9.5, 6.8, '> 600 Mbps', fontsize=11, ha='center', fontweight='bold', color=COVENANT_BLACK_600)

    # Result boxes
    draw_box(2.5, 5.5, 2.8, 1.5, 'zstd-1\n8.7× compression',
             facecolor=COVENANT_RED, edgecolor=COVENANT_RED_DARK, fontcolor='white')
    draw_box(6, 5.5, 2.8, 1.5, 'lz4\n4.0× compression',
             facecolor=COVENANT_BLACK_600, edgecolor=COVENANT_BLACK_800, fontcolor='white')
    draw_box(9.5, 5.5, 2.8, 1.5, 'None\n1.7× (flat only)',
             facecolor=COVENANT_BLACK_400, edgecolor=COVENANT_BLACK_600, fontcolor='white')

    # Use case boxes
    draw_box(2.5, 3.5, 2.8, 1.3, 'Best for:\nWAN, Mobile,\nSlow connections',
             facecolor='#FFEEEE', edgecolor=COVENANT_RED_LIGHT, fontsize=10)
    draw_box(6, 3.5, 2.8, 1.3, 'Best for:\nLAN, Datacenter\nconnections',
             facecolor='#F0F0F0', edgecolor=COVENANT_BLACK_500, fontsize=10)
    draw_box(9.5, 3.5, 2.8, 1.3, 'Best for:\n10G+ networks,\nLocal transfers',
             facecolor='#F5F5F5', edgecolor=COVENANT_BLACK_400, fontsize=10)

    # Arrows from decision to results
    arrow_style = dict(arrowstyle='->', color=COVENANT_BLACK_600, lw=2)
    ax.annotate('', xy=(2.5, 6.3), xytext=(5, 7.4), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 6.3), xytext=(6, 7.4), arrowprops=arrow_style)
    ax.annotate('', xy=(9.5, 6.3), xytext=(7, 7.4), arrowprops=arrow_style)

    # Arrows from results to use cases
    arrow_style_light = dict(arrowstyle='->', color=COVENANT_BLACK_400, lw=1.5)
    for x in [2.5, 6, 9.5]:
        ax.annotate('', xy=(x, 4.2), xytext=(x, 4.7), arrowprops=arrow_style_light)

    # Note at bottom
    note_box = FancyBboxPatch((2, 1.5), 8, 1, boxstyle="round,pad=0.1",
                               facecolor=COVENANT_WHITE_50, edgecolor=COVENANT_BLACK_300, linewidth=1.5)
    ax.add_patch(note_box)
    ax.text(6, 2, 'Always use: Flat representation + Delta encoding as preprocessing steps',
            fontsize=11, ha='center', va='center', style='italic', color=COVENANT_BLACK_600)

    plt.tight_layout()
    plt.savefig(output_dir / '09_decision_flowchart.png')
    plt.savefig(output_dir / '09_decision_flowchart.pdf')
    plt.close()
    print('Created: 09_decision_flowchart.png')


def plot_step_timeline(df: pd.DataFrame, output_dir: Path):
    """Plot 10: Compression ratio over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    setup_axes(ax,
               title='Compression Ratio vs Training Step',
               xlabel='Training Step',
               ylabel='Compression Ratio (×)')

    config = df[(df['representation'] == 'flat') & (df['stage'] == 'delta_encode') & (df['algorithm'] == 'zstd-1')]

    # Group by step
    step_stats = config.groupby('step').agg({
        'compression_ratio': ['mean', 'std', 'count']
    }).reset_index()
    step_stats.columns = ['step', 'mean', 'std', 'count']

    # Fill between for confidence band
    ax.fill_between(step_stats['step'],
                    step_stats['mean'] - step_stats['std'],
                    step_stats['mean'] + step_stats['std'],
                    alpha=0.2, color=COVENANT_RED, linewidth=0)

    # Main line with markers
    ax.plot(step_stats['step'], step_stats['mean'], 'o-',
            color=COVENANT_RED, linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor=COVENANT_RED,
            label='Mean ± Std')

    # Trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(step_stats['step'], step_stats['mean'])
    x_line = np.array([step_stats['step'].min(), step_stats['step'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color=COVENANT_BLACK_600, linewidth=2,
            label=f'Trend (r = {r_value:.2f})')

    # Annotation
    ax.annotate('Early steps: more structured\nupdates → better compression',
                xy=(30, step_stats[step_stats['step'] <= 50]['mean'].mean()),
                xytext=(100, 12),
                fontsize=10, color=COVENANT_BLACK_600,
                arrowprops=dict(arrowstyle='->', color=COVENANT_BLACK_500, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COVENANT_BLACK_300))

    ax.legend(loc='upper right', frameon=True, edgecolor=COVENANT_BLACK_300)
    ax.set_ylim(6, 14)

    plt.tight_layout()
    plt.savefig(output_dir / '10_step_timeline.png')
    plt.savefig(output_dir / '10_step_timeline.pdf')
    plt.close()
    print('Created: 10_step_timeline.png')


def plot_speed_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot 11: Bar chart comparing compression and decompression speeds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ('flat', 'delta_encode', 'zstd-1', 'zstd-1', COVENANT_RED),
        ('flat', 'delta_encode', 'zstd-3', 'zstd-3', '#CC4444'),
        ('flat', 'delta_encode', 'zstd-9', 'zstd-9', '#994444'),
        ('2d_coo', 'delta_encode', 'lz4', 'lz4 (2D)', COVENANT_BLACK_600),
        ('flat', 'delta_encode', 'lz4', 'lz4 (flat)', COVENANT_BLACK_500),
    ]

    labels = [c[3] for c in configs]
    colors = [c[4] for c in configs]

    compress_speeds = []
    decompress_speeds = []

    for rep, stage, algo, _, _ in configs:
        subset = df[(df['representation'] == rep) & (df['stage'] == stage) & (df['algorithm'] == algo)]
        compress_speeds.append(subset['throughput_compress_mb_s'].mean())
        decompress_speeds.append(subset['throughput_decompress_mb_s'].mean())

    x = np.arange(len(labels))
    width = 0.6

    # Compression speed
    ax = axes[0]
    setup_axes(ax, title='Compression Speed', xlabel='', ylabel='Throughput (MB/s)')
    bars = ax.bar(x, compress_speeds, width, color=colors, edgecolor=COVENANT_BLACK_1000, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    for bar, speed in zip(bars, compress_speeds):
        ax.annotate(f'{speed:.0f}', xy=(bar.get_x() + bar.get_width()/2, speed),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # Decompression speed
    ax = axes[1]
    setup_axes(ax, title='Decompression Speed', xlabel='', ylabel='Throughput (MB/s)')
    bars = ax.bar(x, decompress_speeds, width, color=colors, edgecolor=COVENANT_BLACK_1000, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    for bar, speed in zip(bars, decompress_speeds):
        ax.annotate(f'{speed:.0f}', xy=(bar.get_x() + bar.get_width()/2, speed),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Compression & Decompression Throughput', fontsize=16, fontweight='bold',
                 color=COVENANT_BLACK_1000, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '11_speed_comparison.png')
    plt.savefig(output_dir / '11_speed_comparison.pdf')
    plt.close()
    print('Created: 11_speed_comparison.png')


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    configs = [
        ('flat', 'delta_encode', 'zstd-1'),
        ('flat', 'delta_encode', 'zstd-3'),
        ('flat', 'delta_encode', 'zstd-9'),
        ('2d_coo', 'delta_encode', 'zstd-9'),
        ('2d_coo', 'delta_encode', 'lz4'),
        ('flat', 'delta_encode', 'lz4'),
        ('2d_coo', 'raw', 'zstd-9'),
        ('flat', 'raw', 'none'),
    ]

    table_data = []
    for rep, stage, algo in configs:
        subset = df[(df['representation'] == rep) & (df['stage'] == stage) & (df['algorithm'] == algo)]
        if len(subset) == 0:
            continue

        config_name = f'{rep}+{stage[:5]}+{algo}'
        row = [
            config_name,
            f'{subset["compression_ratio"].mean():.2f}×',
            f'±{subset["compression_ratio"].std():.2f}',
            f'{subset["compression_ratio"].min():.1f}–{subset["compression_ratio"].max():.1f}×',
            f'{subset["throughput_compress_mb_s"].mean():.0f}',
            f'{subset["throughput_decompress_mb_s"].mean():.0f}',
        ]
        table_data.append(row)

    columns = ['Configuration', 'Mean\nRatio', 'Std', 'Range', 'Compress\n(MB/s)', 'Decompress\n(MB/s)']

    # Create table
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=[COVENANT_WHITE_100]*6)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor(COVENANT_BLACK_600)
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best row
    for i in range(len(columns)):
        table[(1, i)].set_facecolor('#FFEEEE')
        table[(1, i)].set_edgecolor(COVENANT_RED)

    # Style all cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(COVENANT_BLACK_300)
        cell.set_linewidth(1)

    ax.set_title('Compression Benchmark Summary', fontsize=18, fontweight='bold',
                 color=COVENANT_BLACK_1000, pad=20, y=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / '00_summary_table.png')
    plt.savefig(output_dir / '00_summary_table.pdf')
    plt.close()
    print('Created: 00_summary_table.png')


def main():
    # Setup
    csv_path = '/root/grail/data/compression_benchmark_full.csv'
    output_dir = Path('/root/grail/research/figures/compression')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {csv_path}...')
    df = load_data(csv_path)
    print(f'Loaded {len(df)} rows')
    print(f'Output directory: {output_dir}')
    print()

    print('Generating visualizations with Covenant Labs palette...')
    print('-' * 50)

    create_summary_table(df, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_attribution_bars(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_bandwidth_crossover(df, output_dir)
    plot_variance_violin(df, output_dir)
    plot_sparsity_scatter(df, output_dir)
    plot_waterfall(df, output_dir)
    plot_model_comparison(df, output_dir)
    plot_decision_flowchart(df, output_dir)
    plot_step_timeline(df, output_dir)
    plot_speed_comparison(df, output_dir)

    print('-' * 50)
    print(f'All visualizations saved to {output_dir}')
    print()
    print('Files created:')
    for f in sorted(output_dir.glob('*.png')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
