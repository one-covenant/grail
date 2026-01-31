#!/usr/bin/env python3
"""
Generate publication-quality visualizations for actual weights compression benchmark.
Using Covenant Labs brand color palette.

Key analyses:
1. Method comparison (encoding strategies)
2. Pareto frontier (speed vs compression)
3. Delta encoding benefit
4. Model comparison
5. Algorithm comparison
6. Summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

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

# Method colors - gradient based on compression benefit
METHOD_COLORS = {
    'combined+zstd-9': COVENANT_RED,           # Best - highlight red
    'delta-enc+zstd-9': '#E63333',
    'delta-enc+zstd-3': '#CC4444',
    'downcast+zstd-9': COVENANT_BLACK_500,
    'downcast+zstd-3': COVENANT_BLACK_500,
    'separate+zstd-9': COVENANT_BLACK_500,
    'separate+zstd-3': COVENANT_BLACK_500,
    'brotli-6': COVENANT_BLACK_400,
    'zstd-9': COVENANT_BLACK_400,
    'zstd-3': COVENANT_BLACK_400,
    'zstd-1': COVENANT_BLACK_400,
    'gzip-6': COVENANT_BLACK_300,
}

# Model colors
MODEL_COLORS = {
    'qwen2.5-1.5b': COVENANT_RED,
    'qwen2.5-7b': COVENANT_RED_DARK,
    'llama3.2-3b': COVENANT_BLACK_600,
    'gemma3-4b': COVENANT_BLACK_400,
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

    # Extract encoding type from method
    def get_encoding_type(method):
        if 'delta-enc' in method or 'combined' in method:
            return 'Delta Encoded'
        elif 'downcast' in method:
            return 'Downcast'
        elif 'separate' in method:
            return 'Separate'
        else:
            return 'Raw'

    def get_algorithm(method):
        if '+' in method:
            return method.split('+')[-1]
        return method

    df['encoding_type'] = df['method'].apply(get_encoding_type)
    df['algorithm'] = df['method'].apply(get_algorithm)

    return df


def plot_00_summary_table(df: pd.DataFrame, output_dir: Path):
    """Plot 0: Summary statistics table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Aggregate by method
    summary = df.groupby('method').agg({
        'compression_ratio': ['mean', 'std', 'min', 'max'],
        'compress_speed_mb_s': ['mean', 'std'],
    }).round(2)

    summary.columns = ['Ratio Mean', 'Ratio Std', 'Ratio Min', 'Ratio Max',
                       'Speed Mean', 'Speed Std']
    summary = summary.sort_values('Ratio Mean', ascending=False)

    # Create table
    cell_text = []
    for method in summary.index:
        row = summary.loc[method]
        cell_text.append([
            method,
            f"{row['Ratio Mean']:.2f}×",
            f"±{row['Ratio Std']:.2f}",
            f"{row['Ratio Min']:.2f}×",
            f"{row['Ratio Max']:.2f}×",
            f"{row['Speed Mean']:.1f}",
            f"±{row['Speed Std']:.1f}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Method', 'Mean', 'Std', 'Min', 'Max', 'Speed (MB/s)', 'Std'],
        cellLoc='center',
        loc='center',
        colColours=[COVENANT_BLACK_300] * 7,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Color the best rows
    for i, method in enumerate(summary.index):
        if 'delta-enc' in method or 'combined' in method:
            for j in range(7):
                table[(i + 1, j)].set_facecolor(COVENANT_RED_LIGHT + '40')

    ax.set_title('Compression Benchmark Summary: All Methods',
                 fontsize=18, fontweight='bold', pad=20, color=COVENANT_BLACK_1000)

    save_figure(fig, output_dir, '00_summary_table')


def plot_01_method_comparison_bars(df: pd.DataFrame, output_dir: Path):
    """Plot 1: Bar chart comparing all methods."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Aggregate by method
    agg = df.groupby('method').agg({
        'compression_ratio': 'mean',
        'compress_speed_mb_s': 'mean',
    }).reset_index()
    agg = agg.sort_values('compression_ratio', ascending=True)

    # Create bars
    y_pos = np.arange(len(agg))
    colors = [METHOD_COLORS.get(m, COVENANT_BLACK_400) for m in agg['method']]

    bars = ax.barh(y_pos, agg['compression_ratio'], color=colors, edgecolor=COVENANT_BLACK_800, linewidth=0.5)

    # Add value labels
    for i, (bar, ratio, speed) in enumerate(zip(bars, agg['compression_ratio'], agg['compress_speed_mb_s'])):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.2f}× @ {speed:.0f} MB/s',
                va='center', ha='left', fontsize=10, color=COVENANT_BLACK_800)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['method'])
    ax.set_xlim(0, max(agg['compression_ratio']) * 1.3)

    setup_axes(ax,
               title='Compression Ratio by Method',
               xlabel='Compression Ratio (×)')

    # Add legend for encoding types
    legend_elements = [
        mpatches.Patch(facecolor=COVENANT_RED, edgecolor=COVENANT_BLACK_800,
                       label='Delta Encoded (Best)'),
        mpatches.Patch(facecolor=COVENANT_BLACK_500, edgecolor=COVENANT_BLACK_800,
                       label='Other Encoding'),
        mpatches.Patch(facecolor=COVENANT_BLACK_400, edgecolor=COVENANT_BLACK_800,
                       label='Raw (Baseline)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

    save_figure(fig, output_dir, '01_method_comparison_bars')


def plot_02_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Plot 2: Pareto frontier of compression ratio vs speed."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Aggregate by method
    agg = df.groupby('method').agg({
        'compression_ratio': 'mean',
        'compress_speed_mb_s': 'mean',
    }).reset_index()

    # Plot points
    for _, row in agg.iterrows():
        method = row['method']
        is_delta = 'delta-enc' in method or 'combined' in method
        color = METHOD_COLORS.get(method, COVENANT_BLACK_400)

        size = 250 if is_delta else 100
        alpha = 1.0 if is_delta else 0.6
        marker = 's' if is_delta else 'o'

        ax.scatter(row['compress_speed_mb_s'], row['compression_ratio'],
                   c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors=COVENANT_BLACK_1000 if is_delta else 'none',
                   linewidths=2 if is_delta else 0, zorder=3 if is_delta else 2)

    # Annotate key points
    best_ratio = agg.loc[agg['compression_ratio'].idxmax()]
    best_speed = agg.loc[agg['compress_speed_mb_s'].idxmax()]
    best_tradeoff = agg[agg['method'] == 'delta-enc+zstd-3'].iloc[0] if 'delta-enc+zstd-3' in agg['method'].values else None

    ax.annotate(f"{best_ratio['method']}\n({best_ratio['compression_ratio']:.1f}×)\nBest ratio",
                xy=(best_ratio['compress_speed_mb_s'], best_ratio['compression_ratio']),
                xytext=(best_ratio['compress_speed_mb_s'] - 30, best_ratio['compression_ratio'] + 0.5),
                fontsize=10, fontweight='bold', color=COVENANT_RED,
                arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=1.5))

    if best_tradeoff is not None:
        ax.annotate(f"delta-enc+zstd-3\n({best_tradeoff['compression_ratio']:.1f}×)\nBest tradeoff",
                    xy=(best_tradeoff['compress_speed_mb_s'], best_tradeoff['compression_ratio']),
                    xytext=(best_tradeoff['compress_speed_mb_s'] + 30, best_tradeoff['compression_ratio'] - 0.8),
                    fontsize=10, fontweight='bold', color=COVENANT_RED_DARK,
                    arrowprops=dict(arrowstyle='->', color=COVENANT_RED_DARK, lw=1.5))

    # Draw Pareto frontier
    pareto_points = agg.sort_values('compress_speed_mb_s')
    pareto_x, pareto_y = [pareto_points.iloc[0]['compress_speed_mb_s']], [pareto_points.iloc[0]['compression_ratio']]
    max_ratio = pareto_points.iloc[0]['compression_ratio']

    for _, row in pareto_points.iterrows():
        if row['compression_ratio'] >= max_ratio:
            pareto_x.append(row['compress_speed_mb_s'])
            pareto_y.append(row['compression_ratio'])
            max_ratio = row['compression_ratio']

    ax.plot(pareto_x, pareto_y, '--', color=COVENANT_RED, alpha=0.5, linewidth=2, zorder=1)

    setup_axes(ax,
               title='Compression-Speed Tradeoff (Pareto Frontier)',
               xlabel='Compression Throughput (MB/s)',
               ylabel='Compression Ratio (×)')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COVENANT_RED,
               markersize=12, markeredgecolor=COVENANT_BLACK_1000, markeredgewidth=2,
               label='Delta Encoded'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COVENANT_BLACK_400,
               markersize=10, label='Raw/Other'),
        Line2D([0], [0], linestyle='--', color=COVENANT_RED, alpha=0.5, linewidth=2,
               label='Pareto Frontier'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)

    save_figure(fig, output_dir, '02_pareto_frontier')


def plot_03_delta_encoding_benefit(df: pd.DataFrame, output_dir: Path):
    """Plot 3: Show the benefit of delta encoding (key finding)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compare raw vs delta-encoded
    methods_order = ['zstd-1', 'zstd-3', 'zstd-9', 'delta-enc+zstd-3', 'delta-enc+zstd-9', 'combined+zstd-9']

    agg = df.groupby('method').agg({
        'compression_ratio': 'mean',
    }).reset_index()

    # Filter to methods of interest
    agg = agg[agg['method'].isin(methods_order)]
    agg['order'] = agg['method'].apply(lambda x: methods_order.index(x))
    agg = agg.sort_values('order')

    x_pos = np.arange(len(agg))
    colors = [COVENANT_BLACK_400 if 'delta' not in m and 'combined' not in m else COVENANT_RED
              for m in agg['method']]

    bars = ax.bar(x_pos, agg['compression_ratio'], color=colors,
                  edgecolor=COVENANT_BLACK_800, linewidth=1)

    # Add value labels
    for bar, ratio in zip(bars, agg['compression_ratio']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{ratio:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add arrow showing gain
    raw_zstd3 = agg[agg['method'] == 'zstd-3']['compression_ratio'].values[0]
    delta_zstd3 = agg[agg['method'] == 'delta-enc+zstd-3']['compression_ratio'].values[0]
    gain = delta_zstd3 / raw_zstd3

    ax.annotate('', xy=(3, delta_zstd3 - 0.3), xytext=(1, raw_zstd3 + 0.3),
                arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=3))
    ax.text(2, (raw_zstd3 + delta_zstd3) / 2, f'+{gain:.1f}×\ngain',
            ha='center', va='center', fontsize=14, fontweight='bold', color=COVENANT_RED)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['zstd-1\n(Raw)', 'zstd-3\n(Raw)', 'zstd-9\n(Raw)',
                        'delta-enc\n+zstd-3', 'delta-enc\n+zstd-9', 'combined\n+zstd-9'], fontsize=10)

    setup_axes(ax,
               title='Delta Encoding Doubles Compression Ratio',
               ylabel='Compression Ratio (×)')

    # Add horizontal line at raw baseline
    ax.axhline(y=raw_zstd3, color=COVENANT_BLACK_400, linestyle='--', alpha=0.5, linewidth=1)
    ax.text(5.5, raw_zstd3 + 0.1, 'Raw baseline', ha='right', va='bottom',
            fontsize=9, color=COVENANT_BLACK_500)

    save_figure(fig, output_dir, '03_delta_encoding_benefit')


def plot_04_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot 4: Compare compression across models."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)

    models = ['qwen2.5-1.5b', 'qwen2.5-7b', 'llama3.2-3b', 'gemma3-4b']

    for ax, model in zip(axes, models):
        model_df = df[df['model'] == model]

        # Use best method (delta-enc+zstd-3 or combined+zstd-9)
        best_method = 'combined+zstd-9' if 'combined+zstd-9' in model_df['method'].values else 'delta-enc+zstd-9'
        method_df = model_df[model_df['method'] == best_method]

        if len(method_df) == 0:
            continue

        # Box plot with scatter overlay
        bp = ax.boxplot(method_df['compression_ratio'], widths=0.6,
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor(MODEL_COLORS.get(model, COVENANT_RED_LIGHT))
        bp['boxes'][0].set_alpha(0.7)
        bp['medians'][0].set_color(COVENANT_BLACK_1000)
        bp['medians'][0].set_linewidth(2)

        # Scatter overlay
        x_jitter = np.random.normal(1, 0.08, len(method_df))
        ax.scatter(x_jitter, method_df['compression_ratio'],
                   c=MODEL_COLORS.get(model, COVENANT_RED), alpha=0.4, s=30, zorder=3)

        mean_ratio = method_df['compression_ratio'].mean()
        std_ratio = method_df['compression_ratio'].std()

        ax.set_title(f'{model}\n{mean_ratio:.1f}× ± {std_ratio:.1f}', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.text(1, ax.get_ylim()[0] + 0.3, f'n={len(method_df)}', ha='center',
                fontsize=9, color=COVENANT_BLACK_500)

        setup_axes(ax)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, alpha=0.5)

    axes[0].set_ylabel('Compression Ratio (×)', fontweight='bold')
    fig.suptitle('Compression Consistency Across Model Architectures',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, output_dir, '04_model_comparison')


def plot_05_algorithm_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot 5: Compare compression algorithms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to raw methods only (no encoding)
    raw_methods = ['zstd-1', 'zstd-3', 'zstd-9', 'gzip-6', 'brotli-6']
    raw_df = df[df['method'].isin(raw_methods)]

    agg = raw_df.groupby('method').agg({
        'compression_ratio': 'mean',
        'compress_speed_mb_s': 'mean',
    }).reset_index()

    # Sort by compression ratio
    agg = agg.sort_values('compression_ratio', ascending=False)

    # Left plot: Compression ratio
    colors = [COVENANT_RED if 'zstd' in m else COVENANT_BLACK_400 for m in agg['method']]
    bars1 = ax1.bar(agg['method'], agg['compression_ratio'], color=colors,
                    edgecolor=COVENANT_BLACK_800, linewidth=1)

    for bar, ratio in zip(bars1, agg['compression_ratio']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{ratio:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')

    setup_axes(ax1, title='Compression Ratio', ylabel='Compression Ratio (×)')
    ax1.set_xticklabels(agg['method'], rotation=45, ha='right')

    # Right plot: Speed
    agg_speed = agg.sort_values('compress_speed_mb_s', ascending=False)
    colors_speed = [COVENANT_RED if 'zstd' in m else COVENANT_BLACK_400 for m in agg_speed['method']]
    bars2 = ax2.bar(agg_speed['method'], agg_speed['compress_speed_mb_s'], color=colors_speed,
                    edgecolor=COVENANT_BLACK_800, linewidth=1)

    for bar, speed in zip(bars2, agg_speed['compress_speed_mb_s']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{speed:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    setup_axes(ax2, title='Compression Speed', ylabel='Throughput (MB/s)')
    ax2.set_xticklabels(agg_speed['method'], rotation=45, ha='right')

    fig.suptitle('Algorithm Comparison (Raw Data, No Encoding)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, output_dir, '05_algorithm_comparison')


def plot_06_heatmap(df: pd.DataFrame, output_dir: Path):
    """Plot 6: Heatmap of all methods across models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Pivot table
    pivot = df.pivot_table(
        values='compression_ratio',
        index='method',
        columns='model',
        aggfunc='mean'
    )

    # Sort by mean compression ratio
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False)
    pivot = pivot.drop('mean', axis=1)

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', ax=ax,
                cbar_kws={'label': 'Compression Ratio (×)'},
                linewidths=0.5, linecolor=COVENANT_WHITE_100,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    # Highlight best methods
    for i, method in enumerate(pivot.index):
        if 'delta-enc' in method or 'combined' in method:
            ax.add_patch(plt.Rectangle((0, i), len(pivot.columns), 1,
                         fill=False, edgecolor=COVENANT_RED, linewidth=3))

    ax.set_title('Compression Ratio: All Methods × Models',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Method', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_dir, '06_heatmap')


def plot_07_encoding_waterfall(df: pd.DataFrame, output_dir: Path):
    """Plot 7: Waterfall chart showing cumulative encoding gains."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get mean compression for each method type
    agg = df.groupby('method')['compression_ratio'].mean()

    # Define the waterfall steps
    steps = [
        ('Raw (gzip-6)', agg.get('gzip-6', 0), COVENANT_BLACK_300),
        ('Raw (zstd-3)', agg.get('zstd-3', 0), COVENANT_BLACK_400),
        ('Downcast + zstd-3', agg.get('downcast+zstd-3', 0), COVENANT_BLACK_500),
        ('Separate + zstd-3', agg.get('separate+zstd-3', 0), COVENANT_BLACK_500),
        ('Delta-enc + zstd-3', agg.get('delta-enc+zstd-3', 0), COVENANT_RED_LIGHT),
        ('Combined + zstd-9', agg.get('combined+zstd-9', 0), COVENANT_RED),
    ]

    x_pos = np.arange(len(steps))
    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [s[2] for s in steps]

    bars = ax.bar(x_pos, values, color=colors, edgecolor=COVENANT_BLACK_800, linewidth=1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add gain arrows between consecutive bars
    for i in range(1, len(values)):
        if values[i] > values[i-1]:
            gain = values[i] / values[i-1]
            mid_y = (values[i] + values[i-1]) / 2
            ax.annotate('', xy=(i, values[i] - 0.2), xytext=(i-1, values[i-1] + 0.2),
                        arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=2, alpha=0.7))
            ax.text(i - 0.5, mid_y, f'+{(gain-1)*100:.0f}%', ha='center', va='center',
                    fontsize=9, color=COVENANT_RED, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

    setup_axes(ax,
               title='Compression Pipeline: Cumulative Gains',
               ylabel='Compression Ratio (×)')

    save_figure(fig, output_dir, '07_encoding_waterfall')


def plot_08_sparsity_vs_compression(df: pd.DataFrame, output_dir: Path):
    """Plot 8: Scatter plot of sparsity vs compression ratio."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use best method
    best_method = 'combined+zstd-9'
    method_df = df[df['method'] == best_method]

    # Color by model
    for model in method_df['model'].unique():
        model_data = method_df[method_df['model'] == model]
        color = MODEL_COLORS.get(model, COVENANT_BLACK_400)
        ax.scatter(model_data['sparsity_pct'], model_data['compression_ratio'],
                   c=color, alpha=0.6, s=50, label=model, edgecolors='white', linewidths=0.5)

    # Add trend line
    x = method_df['sparsity_pct']
    y = method_df['compression_ratio']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), '--', color=COVENANT_RED, alpha=0.7, linewidth=2,
            label=f'Trend (slope={z[0]:.2f})')

    setup_axes(ax,
               title=f'Sparsity vs Compression Ratio ({best_method})',
               xlabel='Sparsity (%)',
               ylabel='Compression Ratio (×)')

    ax.legend(loc='lower right', framealpha=0.95)

    # Add correlation coefficient
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', color=COVENANT_BLACK_600)

    save_figure(fig, output_dir, '08_sparsity_vs_compression')


def plot_09_experiment_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot 9: Compare compression across different experiments."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Use best method
    best_method = 'combined+zstd-9'
    method_df = df[df['method'] == best_method]

    # Aggregate by experiment
    agg = method_df.groupby('experiment').agg({
        'compression_ratio': ['mean', 'std'],
    }).reset_index()
    agg.columns = ['experiment', 'mean', 'std']
    agg = agg.sort_values('mean', ascending=False)

    x_pos = np.arange(len(agg))

    # Color by model type
    colors = []
    for exp in agg['experiment']:
        if 'qwen2.5-7b' in exp:
            colors.append(COVENANT_RED_DARK)
        elif 'qwen2.5-1.5b' in exp:
            colors.append(COVENANT_RED)
        elif 'llama' in exp:
            colors.append(COVENANT_BLACK_600)
        else:
            colors.append(COVENANT_BLACK_400)

    bars = ax.bar(x_pos, agg['mean'], yerr=agg['std'], capsize=3, color=colors,
                  edgecolor=COVENANT_BLACK_800, linewidth=1, error_kw={'linewidth': 1.5})

    # Add value labels
    for bar, mean, std in zip(bars, agg['mean'], agg['std']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.2,
                f'{mean:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(agg['experiment'], rotation=45, ha='right', fontsize=9)

    setup_axes(ax,
               title=f'Compression by Experiment ({best_method})',
               ylabel='Compression Ratio (×)')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COVENANT_RED, label='Qwen 2.5 1.5B'),
        mpatches.Patch(facecolor=COVENANT_RED_DARK, label='Qwen 2.5 7B'),
        mpatches.Patch(facecolor=COVENANT_BLACK_600, label='Llama 3.2 3B'),
        mpatches.Patch(facecolor=COVENANT_BLACK_400, label='Gemma 3 4B'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, output_dir, '09_experiment_comparison')


def plot_10_speed_vs_level(df: pd.DataFrame, output_dir: Path):
    """Plot 10: Show speed vs compression level tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compare zstd levels
    zstd_methods = ['zstd-1', 'zstd-3', 'zstd-9']
    delta_methods = ['delta-enc+zstd-3', 'delta-enc+zstd-9']

    agg = df.groupby('method').agg({
        'compression_ratio': 'mean',
        'compress_speed_mb_s': 'mean',
    }).reset_index()

    # Plot raw zstd
    raw_df = agg[agg['method'].isin(zstd_methods)]
    ax.plot(raw_df['compress_speed_mb_s'], raw_df['compression_ratio'], 'o-',
            color=COVENANT_BLACK_500, markersize=12, linewidth=2, label='Raw zstd')

    for _, row in raw_df.iterrows():
        ax.annotate(row['method'], (row['compress_speed_mb_s'], row['compression_ratio']),
                    textcoords='offset points', xytext=(10, 5), fontsize=10)

    # Plot delta-encoded
    delta_df = agg[agg['method'].isin(delta_methods)]
    ax.plot(delta_df['compress_speed_mb_s'], delta_df['compression_ratio'], 's-',
            color=COVENANT_RED, markersize=12, linewidth=2, label='Delta-encoded zstd')

    for _, row in delta_df.iterrows():
        ax.annotate(row['method'].replace('delta-enc+', ''),
                    (row['compress_speed_mb_s'], row['compression_ratio']),
                    textcoords='offset points', xytext=(10, 5), fontsize=10, color=COVENANT_RED)

    setup_axes(ax,
               title='Compression Level vs Speed Tradeoff',
               xlabel='Compression Throughput (MB/s)',
               ylabel='Compression Ratio (×)')

    ax.legend(loc='upper right', framealpha=0.95)

    # Add annotation showing delta encoding benefit
    ax.annotate('Delta encoding\ndoubles ratio!',
                xy=(delta_df['compress_speed_mb_s'].mean(), delta_df['compression_ratio'].mean()),
                xytext=(150, 5),
                fontsize=12, fontweight='bold', color=COVENANT_RED,
                arrowprops=dict(arrowstyle='->', color=COVENANT_RED, lw=2))

    save_figure(fig, output_dir, '10_speed_vs_level')


def save_figure(fig, output_dir: Path, name: str):
    """Save figure in both PNG and PDF formats."""
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {name}.png, {name}.pdf')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize actual weights compression benchmark')
    parser.add_argument('--input', type=str, default='/root/grail/scripts/compression_actual_weights.csv',
                        help='Input CSV file')
    parser.add_argument('--output-dir', type=str, default='/root/grail/research/figures/compression_v2',
                        help='Output directory for figures')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading data from {args.input}...')
    df = load_data(args.input)
    print(f'Loaded {len(df)} rows')

    print(f'\nGenerating figures to {output_dir}...\n')

    # Generate all figures
    plot_00_summary_table(df, output_dir)
    plot_01_method_comparison_bars(df, output_dir)
    plot_02_pareto_frontier(df, output_dir)
    plot_03_delta_encoding_benefit(df, output_dir)
    plot_04_model_comparison(df, output_dir)
    plot_05_algorithm_comparison(df, output_dir)
    plot_06_heatmap(df, output_dir)
    plot_07_encoding_waterfall(df, output_dir)
    plot_08_sparsity_vs_compression(df, output_dir)
    plot_09_experiment_comparison(df, output_dir)
    plot_10_speed_vs_level(df, output_dir)

    print(f'\nDone! Generated 11 figures in {output_dir}')

    # Print key findings
    print('\n' + '=' * 70)
    print('KEY FINDINGS')
    print('=' * 70)

    agg = df.groupby('method')['compression_ratio'].mean().sort_values(ascending=False)
    print(f'\nBest method: {agg.index[0]} ({agg.iloc[0]:.2f}×)')
    print(f'Baseline (zstd-3): {agg.get("zstd-3", 0):.2f}×')
    print(f'Gain from delta encoding: {agg.iloc[0] / agg.get("zstd-3", 1):.1f}×')


if __name__ == '__main__':
    main()
