#!/usr/bin/env python3
"""
Plot learning rate effect on sparsity over training steps.
Following Covenant Labs brand guidelines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Covenant Labs colors
COVENANT_RED = '#FF3A3A'
COVENANT_BLACK_1000 = '#101010'
COVENANT_BLACK_500 = '#828282'
COVENANT_WHITE_100 = '#DDDDDD'

# Elegant luxurious color palette - sophisticated gradient
# From deep royal tones (low LR) to warm gold/coral (high LR)
LR_COLORS = {
    5e-07: '#1A1A2E',              # Deep midnight blue (most conservative)
    1e-06: '#16213E',              # Royal navy
    3e-06: '#0F4C75',              # Elegant teal blue
    5e-06: '#C17817',              # Rich amber gold
    2e-05: '#BE3144',              # Luxe burgundy red (most aggressive)
}

# Solid lines with subtle width variation for hierarchy
LR_LINESTYLES = {
    5e-07: '-',
    1e-06: '-',
    3e-06: '-',
    5e-06: '-',
    2e-05: '-',
}

LR_LINEWIDTHS = {
    5e-07: 2.2,
    1e-06: 2.4,
    3e-06: 2.8,                    # Slightly thicker for reference LR
    5e-06: 2.4,
    2e-05: 2.2,
}

# Configure matplotlib
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = COVENANT_BLACK_500


def load_lr_sparsity_data(
    csv_paths: list[str],
    model_family: str = 'Qwen',
    model_size: str = '1.5B',
    k_value: int = 1,
    iteration_num: int = 1,
) -> pd.DataFrame:
    """
    Load sparsity data for multiple learning rates.

    Returns DataFrame with columns: step, learning_rate, sparsity_mean, sparsity_std
    """
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Filter
    df = df[
        (df['model_family'] == model_family) &
        (df['model_size'] == model_size) &
        (df['k'] == k_value) &
        (df['iteration_num'] == iteration_num)
    ]

    if len(df) == 0:
        return None

    # Exclude very high learning rates (outliers)
    df = df[df['learning_rate'] <= 5e-6]

    # Aggregate across seeds: mean and std per (step, learning_rate)
    agg = df.groupby(['step', 'learning_rate'])['sparsity'].agg(['mean', 'std']).reset_index()
    agg.columns = ['step', 'learning_rate', 'sparsity_mean', 'sparsity_std']

    return agg


def create_lr_sparsity_plot(
    data: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (14, 8),
    dpi: int = 150,
    title: str = None,
    show_bands: bool = True,
    subsample_step: int = 1,
):
    """
    Create a line plot showing sparsity over training steps for different learning rates.

    Args:
        data: DataFrame with step, learning_rate, sparsity_mean, sparsity_std
        output_path: Path to save figure
        figsize: Figure dimensions
        dpi: Resolution
        title: Optional title
        show_bands: Whether to show ±1 std bands
        subsample_step: Plot every Nth step (for cleaner visualization)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Get unique learning rates, sorted
    learning_rates = sorted(data['learning_rate'].unique(), reverse=True)

    for lr in learning_rates:
        subset = data[data['learning_rate'] == lr].sort_values('step')

        # Subsample for cleaner lines
        if subsample_step > 1:
            subset = subset.iloc[::subsample_step]

        steps = subset['step'].values
        means = subset['sparsity_mean'].values
        stds = subset['sparsity_std'].values

        color = LR_COLORS.get(lr, COVENANT_BLACK_500)
        linestyle = LR_LINESTYLES.get(lr, '-')
        linewidth = LR_LINEWIDTHS.get(lr, 2.0)

        # Format learning rate for label
        lr_label = f'lr={lr:.0e}'.replace('e-0', 'e-')

        # Plot confidence band first (behind line)
        if show_bands and stds is not None:
            ax.fill_between(
                steps,
                means - stds,
                means + stds,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        # Plot main line
        ax.plot(
            steps,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=lr_label,
            zorder=3,
        )

    # Configure axes
    ax.set_xlabel('Training Step', fontsize=20, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.set_ylabel('Sparsity (%)', fontsize=20, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)

    # Y-axis: focus on the interesting range
    y_min = data['sparsity_mean'].min() - 0.5
    y_max = min(100, data['sparsity_mean'].max() + 0.5)

    # Round to nice values
    y_min = max(95, np.floor(y_min * 2) / 2)  # Floor to nearest 0.5, min 95
    ax.set_ylim(y_min, 100)

    # Set y-ticks at 0.5% intervals for precision
    y_ticks = np.arange(y_min, 100.5, 0.5)
    if len(y_ticks) > 10:
        y_ticks = np.arange(y_min, 100.5, 1.0)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis='y', labelsize=14, colors=COVENANT_BLACK_1000, width=1.5, length=6)

    # X-axis
    ax.set_xlim(0, data['step'].max())
    ax.tick_params(axis='x', labelsize=14, colors=COVENANT_BLACK_1000, width=1.5, length=6)

    # Grid - subtle horizontal lines only
    ax.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100,
                  linewidth=0.8, zorder=1, alpha=0.8)
    ax.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100,
                  linewidth=0.5, zorder=1, alpha=0.5)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COVENANT_BLACK_500)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(COVENANT_BLACK_500)
    ax.spines['bottom'].set_linewidth(1.2)

    # Legend - positioned in lower left where there's usually space
    legend = ax.legend(
        loc='lower left',
        fontsize=14,
        framealpha=0.95,
        edgecolor=COVENANT_BLACK_500,
        fancybox=False,
        borderpad=0.8,
        handlelength=2.5,
        title='Learning Rate',
        title_fontsize=14,
    )
    legend.get_title().set_fontweight('bold')

    # Title
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold',
                     color=COVENANT_BLACK_1000, pad=20)

    # Add annotation for key insight
    # Find the final sparsity values
    final_step = data['step'].max()
    final_data = data[data['step'] == final_step]

    # Annotate the spread at final step
    min_final = final_data['sparsity_mean'].min()
    max_final = final_data['sparsity_mean'].max()
    spread = max_final - min_final

    # Add text box with key finding
    textstr = f'Final spread: {spread:.2f}%\n(all LRs > {min_final:.1f}%)'
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor=COVENANT_BLACK_500, alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, color=COVENANT_BLACK_1000)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {output_path}')
    else:
        plt.show()

    plt.close()
    return fig


def create_lr_sparsity_plot_dual(
    data: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (16, 6),
    dpi: int = 150,
):
    """
    Create a dual-panel plot:
    - Left: Sparsity over steps (full view)
    - Right: Zoomed view of final 100 steps to show convergence
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')

    learning_rates = sorted(data['learning_rate'].unique(), reverse=True)

    for ax, (start_step, end_step, title, subsample) in [
        (ax1, (0, data['step'].max(), 'Full Training', 2)),
        (ax2, (data['step'].max() - 100, data['step'].max(), 'Final 100 Steps', 1)),
    ]:
        ax.set_facecolor('white')

        for lr in learning_rates:
            subset = data[
                (data['learning_rate'] == lr) &
                (data['step'] >= start_step) &
                (data['step'] <= end_step)
            ].sort_values('step')

            if subsample > 1:
                subset = subset.iloc[::subsample]

            steps = subset['step'].values
            means = subset['sparsity_mean'].values
            stds = subset['sparsity_std'].values

            color = LR_COLORS.get(lr, COVENANT_BLACK_500)
            linestyle = LR_LINESTYLES.get(lr, '-')
            linewidth = LR_LINEWIDTHS.get(lr, 2.0)
            lr_label = f'lr={lr:.0e}'.replace('e-0', 'e-')

            # Confidence band
            ax.fill_between(steps, means - stds, means + stds,
                           color=color, alpha=0.12, linewidth=0)

            # Main line
            ax.plot(steps, means, color=color, linestyle=linestyle,
                   linewidth=linewidth, label=lr_label, zorder=3)

        # Axis formatting
        ax.set_xlabel('Training Step', fontsize=16, fontweight='bold',
                      color=COVENANT_BLACK_1000, labelpad=10)
        ax.set_ylabel('Sparsity (%)', fontsize=16, fontweight='bold',
                      color=COVENANT_BLACK_1000, labelpad=10)
        ax.set_title(title, fontsize=18, fontweight='bold',
                     color=COVENANT_BLACK_1000, pad=15)

        # Y-axis range
        if ax == ax1:
            y_min = max(95, data['sparsity_mean'].min() - 1)
        else:
            subset = data[(data['step'] >= start_step) & (data['step'] <= end_step)]
            y_min = max(95, subset['sparsity_mean'].min() - 0.5)
        ax.set_ylim(y_min, 100)

        ax.tick_params(axis='both', labelsize=12, colors=COVENANT_BLACK_1000)

        # Grid
        ax.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.8, alpha=0.8)
        ax.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.5, alpha=0.5)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COVENANT_BLACK_500)
        ax.spines['bottom'].set_color(COVENANT_BLACK_500)

    # Single legend for both panels
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(learning_rates),
               fontsize=12, framealpha=0.95, edgecolor=COVENANT_BLACK_500,
               fancybox=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {output_path}')
    else:
        plt.show()

    plt.close()
    return fig


def create_lr_sparsity_plot_stacked(
    data: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (14, 10),
    dpi: int = 150,
):
    """
    Create a stacked 2-panel plot:
    - Top: All learning rates (full scale, showing the outlier clearly)
    - Bottom: Zoomed on high-sparsity region (98-100%) for typical LRs

    This design clearly shows that high LR (2e-5) is an outlier while
    all practical LRs maintain >98% sparsity.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 1.2])
    fig.patch.set_facecolor('white')

    learning_rates = sorted(data['learning_rate'].unique(), reverse=True)
    max_step = data['step'].max()

    # Top panel: All LRs, full scale
    ax1.set_facecolor('white')
    for lr in learning_rates:
        subset = data[data['learning_rate'] == lr].sort_values('step').iloc[::2]
        steps = subset['step'].values
        means = subset['sparsity_mean'].values
        stds = subset['sparsity_std'].values

        color = LR_COLORS.get(lr, COVENANT_BLACK_500)
        linestyle = LR_LINESTYLES.get(lr, '-')
        linewidth = LR_LINEWIDTHS.get(lr, 2.0)
        lr_label = f'lr={lr:.0e}'.replace('e-0', 'e-')

        ax1.fill_between(steps, means - stds, means + stds,
                        color=color, alpha=0.12, linewidth=0)
        ax1.plot(steps, means, color=color, linestyle=linestyle,
                linewidth=linewidth, label=lr_label, zorder=3)

    ax1.set_ylabel('Sparsity (%)', fontsize=18, fontweight='bold',
                   color=COVENANT_BLACK_1000, labelpad=12)
    ax1.set_title('All Learning Rates — Full Scale', fontsize=20, fontweight='bold',
                  color=COVENANT_BLACK_1000, pad=15)
    ax1.set_ylim(94, 100.5)
    ax1.set_xlim(0, max_step)
    ax1.tick_params(axis='both', labelsize=12, colors=COVENANT_BLACK_1000)
    ax1.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.8, alpha=0.8)
    ax1.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.5, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COVENANT_BLACK_500)
    ax1.spines['bottom'].set_color(COVENANT_BLACK_500)

    # Add annotation for high LR
    high_lr_data = data[data['learning_rate'] == 2e-05]
    if len(high_lr_data) > 0:
        # Get the final available step for this LR
        final_step_for_lr = high_lr_data['step'].max()
        final_row = high_lr_data[high_lr_data['step'] == final_step_for_lr]
        if len(final_row) > 0:
            final_sparsity = final_row['sparsity_mean'].values[0]
            ax1.annotate(
                f'High LR (2e-5)\n→ {final_sparsity:.1f}% sparsity',
                xy=(max_step * 0.75, final_sparsity + 0.3),
                fontsize=11,
                color=COVENANT_RED,
                fontweight='bold',
                ha='center',
            )

    # Elegant legend in lower right
    ax1.legend(loc='lower right', fontsize=12, framealpha=0.95,
               edgecolor=COVENANT_BLACK_500, fancybox=False,
               borderpad=0.8, handlelength=2.5, title='Learning Rate',
               title_fontsize=12)

    # Bottom panel: Zoomed on high-sparsity LRs (exclude 2e-5)
    ax2.set_facecolor('white')
    typical_lrs = [lr for lr in learning_rates if lr <= 5e-6]  # Exclude very high LRs (>5e-6)

    for lr in typical_lrs:
        subset = data[data['learning_rate'] == lr].sort_values('step').iloc[::2]
        steps = subset['step'].values
        means = subset['sparsity_mean'].values
        stds = subset['sparsity_std'].values

        color = LR_COLORS.get(lr, COVENANT_BLACK_500)
        linestyle = LR_LINESTYLES.get(lr, '-')
        linewidth = LR_LINEWIDTHS.get(lr, 2.0) + 0.5  # Slightly thicker for visibility
        lr_label = f'lr={lr:.0e}'.replace('e-0', 'e-')

        ax2.fill_between(steps, means - stds, means + stds,
                        color=color, alpha=0.15, linewidth=0)
        ax2.plot(steps, means, color=color, linestyle=linestyle,
                linewidth=linewidth, label=lr_label, zorder=3)

    ax2.set_xlabel('Training Step', fontsize=18, fontweight='bold',
                   color=COVENANT_BLACK_1000, labelpad=12)
    ax2.set_ylabel('Sparsity (%)', fontsize=18, fontweight='bold',
                   color=COVENANT_BLACK_1000, labelpad=12)
    ax2.set_title('Typical Learning Rates — Zoomed (98–100%)', fontsize=20, fontweight='bold',
                  color=COVENANT_BLACK_1000, pad=15)
    ax2.set_ylim(98, 100.2)
    ax2.set_xlim(0, max_step)
    ax2.set_yticks([98, 98.5, 99, 99.5, 100])
    ax2.tick_params(axis='both', labelsize=12, colors=COVENANT_BLACK_1000)
    ax2.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.8, alpha=0.8)
    ax2.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.5, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COVENANT_BLACK_500)
    ax2.spines['bottom'].set_color(COVENANT_BLACK_500)

    # Add key insight annotation - get final sparsity for each typical LR
    typical_final_sparsities = []
    for lr in typical_lrs:
        lr_data = data[data['learning_rate'] == lr]
        final_step_for_lr = lr_data['step'].max()
        final_val = lr_data[lr_data['step'] == final_step_for_lr]['sparsity_mean'].values
        if len(final_val) > 0:
            typical_final_sparsities.append(final_val[0])
    min_sparsity = min(typical_final_sparsities) if typical_final_sparsities else 99.0

    textstr = f'All typical LRs maintain\n>{min_sparsity:.1f}% sparsity'
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor=COVENANT_BLACK_500, alpha=0.95)
    ax2.text(0.98, 0.05, textstr, transform=ax2.transAxes, fontsize=13,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props, color=COVENANT_BLACK_1000, fontweight='bold')

    # Legend
    ax2.legend(loc='lower left', fontsize=12, framealpha=0.95,
               edgecolor=COVENANT_BLACK_500, fancybox=False,
               borderpad=0.6, handlelength=2.0)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {output_path}')
    else:
        plt.show()

    plt.close()
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot LR effect on sparsity over training steps')
    parser.add_argument('--csv', nargs='+', default=[
        'data/sparsity_k_step_combined.csv',
    ], help='CSV files with sparsity data')
    parser.add_argument('--output', '-o', default='research/figures/lr_sparsity_over_steps.png',
                        help='Output path')
    parser.add_argument('--model-family', default='Qwen', help='Model family')
    parser.add_argument('--model-size', default='1.5B', help='Model size')
    parser.add_argument('--k', type=int, default=1, help='K value (step gap)')
    parser.add_argument('--iter', type=int, default=1, help='Iteration number')
    parser.add_argument('--dual', action='store_true', help='Create dual-panel plot (side by side)')
    parser.add_argument('--stacked', action='store_true', help='Create stacked 2-panel plot (recommended)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f'Loading data for {args.model_family} {args.model_size}, k={args.k}, iter={args.iter}')
    data = load_lr_sparsity_data(
        args.csv,
        model_family=args.model_family,
        model_size=args.model_size,
        k_value=args.k,
        iteration_num=args.iter,
    )

    if data is None or len(data) == 0:
        print('No data found!')
        exit(1)

    print(f'Loaded {len(data)} data points')
    print(f'Learning rates: {sorted(data["learning_rate"].unique())}')
    print(f'Steps: {data["step"].min()} - {data["step"].max()}')

    # Print summary stats
    print('\nSparsity by learning rate:')
    for lr in sorted(data['learning_rate'].unique(), reverse=True):
        subset = data[data['learning_rate'] == lr]
        print(f'  lr={lr:.0e}: mean={subset["sparsity_mean"].mean():.2f}%, '
              f'final={subset[subset["step"] == subset["step"].max()]["sparsity_mean"].values[0]:.2f}%')

    # Create plot
    if args.stacked:
        create_lr_sparsity_plot_stacked(
            data,
            output_path=str(output_path),
            dpi=args.dpi,
        )
    elif args.dual:
        create_lr_sparsity_plot_dual(
            data,
            output_path=str(output_path),
            dpi=args.dpi,
        )
    else:
        title = f'{args.model_family} {args.model_size} — Weight Update Sparsity by Learning Rate'
        create_lr_sparsity_plot(
            data,
            output_path=str(output_path),
            title=title,
            dpi=args.dpi,
            subsample_step=2,  # Plot every other step for cleaner lines
        )
