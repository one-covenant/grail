#!/usr/bin/env python3
"""
Grouped bar plot for weight update sparsity statistics.
Following Covenant Labs brand guidelines.
Matplotlib version of the TikZ visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Covenant Labs colors
COVENANT_RED = '#FF3A3A'
COVENANT_BLACK_1000 = '#101010'
COVENANT_BLACK_500 = '#828282'
COVENANT_WHITE_100 = '#DDDDDD'

# Configure matplotlib for clean output
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = COVENANT_BLACK_500


def create_sparsity_bar_plot(
    data: dict = None,
    output_path: str = None,
    figsize: tuple = (12, 7),
    dpi: int = 150,
):
    """
    Create a grouped bar plot for sparsity statistics.

    Args:
        data: Dictionary with model families as keys, containing:
              {family: [(size_label, mean, std), ...]}
              If None, uses placeholder data.
        output_path: Path to save the figure. If None, displays interactively.
        figsize: Figure size in inches (width, height).
        dpi: Resolution for saved figure.
    """

    # Default placeholder data (replace with actual computed values)
    if data is None:
        data = {
            'Qwen2.5': [
                ('0.5B', 96.1, 1.2),
                ('1.5B', 97.4, 0.9),
                ('7B', 98.2, 0.6),
            ],
            'Llama-3.2': [
                ('3B', 97.1, 1.0),
            ],
            'Gemma-3': [
                ('1B', 96.3, 1.1),
                ('4B', 97.5, 0.8),
            ],
        }

    # Style settings for each family: color, hatch, alpha, edgecolor, error_bar_color
    family_styles = {
        'Qwen2.5': {
            'color': COVENANT_RED,
            'hatch': '///',
            'alpha': 0.85,
            'edgecolor': '#CC2020',
            'linewidth': 1.5,
            'error_color': COVENANT_BLACK_1000,  # Dark error bars on light background
        },
        'Llama-3.2': {
            'color': '#404040',  # Dark charcoal - distinguishable but professional
            'hatch': '\\\\\\',
            'alpha': 0.9,
            'edgecolor': '#202020',
            'linewidth': 1.5,
            'error_color': COVENANT_BLACK_1000,  # Dark error bars now visible
        },
        'Gemma-3': {
            'color': COVENANT_BLACK_500,
            'hatch': 'xxx',
            'alpha': 0.7,
            'edgecolor': '#505050',
            'linewidth': 1.5,
            'error_color': COVENANT_BLACK_1000,  # Dark error bars
        },
    }

    # Calculate x positions - tighter within groups, gap between groups
    x_positions = []
    x_labels = []
    styles = []
    values = []
    errors = []
    family_ranges = {}

    current_x = 0
    bar_spacing = 0.7  # Tighter spacing within groups
    group_gap = 1.4     # Gap between family groups

    for family in ['Qwen2.5', 'Llama-3.2', 'Gemma-3']:
        if family not in data:
            continue

        family_start = current_x
        style = family_styles[family]

        for i, (size_label, mean, std) in enumerate(data[family]):
            x_positions.append(current_x)
            x_labels.append(size_label)
            styles.append(style)
            values.append(mean)
            errors.append(std)
            current_x += bar_spacing

        family_ranges[family] = (family_start, current_x - bar_spacing)
        current_x += group_gap - bar_spacing  # Add gap for next group

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Bar width
    bar_width = 0.55

    # Draw bars with hatching and styling
    for x, val, style, err in zip(x_positions, values, styles, errors):
        # Main bar
        bar = ax.bar(x, val - 95, bottom=95, width=bar_width,
                     color=style['color'], alpha=style['alpha'],
                     edgecolor=style['edgecolor'], linewidth=style['linewidth'],
                     hatch=style['hatch'], zorder=3)

        # Add subtle shadow/depth effect
        shadow_offset = 0.03
        ax.bar(x + shadow_offset, val - 95, bottom=95, width=bar_width,
               color='#000000', alpha=0.1, zorder=2)

    # Draw error bars
    cap_width = 0.15
    for x, val, err, style in zip(x_positions, values, errors, styles):
        error_color = style['error_color']
        # Vertical line
        ax.plot([x, x], [val - err, val + err], color=error_color,
                linewidth=2.0, zorder=4, solid_capstyle='round')
        # Top cap
        ax.plot([x - cap_width, x + cap_width], [val + err, val + err],
                color=error_color, linewidth=2.0, zorder=4, solid_capstyle='round')
        # Bottom cap
        ax.plot([x - cap_width, x + cap_width], [val - err, val - err],
                color=error_color, linewidth=2.0, zorder=4, solid_capstyle='round')

    # Value labels above error bars
    for x, val, err, style in zip(x_positions, values, errors, styles):
        label_y = val + err + 0.3
        ax.text(x, label_y, f'{val:.1f}', ha='center', va='bottom',
                fontsize=16, fontweight='bold', color=COVENANT_BLACK_1000, zorder=5)

    # Family labels with lines above plot - closer to bars
    max_y = 100
    family_line_y = max_y + 0.35
    family_text_y = max_y + 0.45

    for family, (start, end) in family_ranges.items():
        style = family_styles[family]
        mid = (start + end) / 2

        # Horizontal line with matching color
        line_extend = 0.25
        ax.plot([start - line_extend, end + line_extend], [family_line_y, family_line_y],
                color=style['color'], linewidth=3.0, zorder=5, clip_on=False,
                solid_capstyle='round')

        # Family name - LARGE FONT
        ax.text(mid, family_text_y, family, ha='center', va='bottom',
                fontsize=20, fontweight='bold', color=style['color'], zorder=5, clip_on=False)

    # Configure axes - LARGE FONTS for readability
    ax.set_ylim(95, 100)
    ax.set_yticks([95, 96, 97, 98, 99, 100])
    ax.set_ylabel('Sparsity (%)', fontsize=22, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.tick_params(axis='y', labelsize=18, colors=COVENANT_BLACK_1000, width=1.5, length=8)

    ax.set_xlim(min(x_positions) - 0.6, max(x_positions) + 0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=18, fontweight='bold', color=COVENANT_BLACK_1000)
    ax.tick_params(axis='x', colors=COVENANT_BLACK_500, width=1.5, length=8, pad=10)

    # Grid - subtle horizontal lines
    ax.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100,
                  linewidth=0.8, zorder=1, alpha=0.8)
    ax.xaxis.grid(False)

    # Spines styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COVENANT_BLACK_500)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(COVENANT_BLACK_500)
    ax.spines['bottom'].set_linewidth(1.2)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def load_sparsity_data_from_csv(
    csv_paths: list[str],
    k_value: int = 1,
    learning_rate: float = 3e-6,
    iteration_num: int = 1,
) -> dict:
    """
    Load sparsity data from CSV files and compute statistics.

    Args:
        csv_paths: List of paths to sparsity CSV files.
        k_value: Which k value to use for the plot.
        learning_rate: Filter to this learning rate (default 3e-6).
        iteration_num: Filter to this iteration number (default 1).

    Returns:
        Dictionary formatted for create_sparsity_bar_plot.
    """
    import pandas as pd

    # Load and concatenate all CSVs
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {path}")

    if not dfs:
        print("No CSV files found!")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Parse learning_rate column if it's a string
    if df['learning_rate'].dtype == object:
        df['learning_rate'] = df['learning_rate'].apply(lambda x: float(x))

    # Filter by k value
    df = df[df['k'] == k_value]
    print(f"After k={k_value} filter: {len(df)} rows")

    # Filter by learning rate (with tolerance for floating point)
    lr_tolerance = learning_rate * 0.1  # 10% tolerance
    df = df[(df['learning_rate'] >= learning_rate - lr_tolerance) &
            (df['learning_rate'] <= learning_rate + lr_tolerance)]
    print(f"After lr={learning_rate:.0e} filter: {len(df)} rows")

    # Filter by iteration number
    df = df[df['iteration_num'] == iteration_num]
    print(f"After iter={iteration_num} filter: {len(df)} rows")

    if len(df) == 0:
        print("No data after filtering!")
        return None

    # Show unique combinations
    print("\nUnique model combinations:")
    for (fam, size), group in df.groupby(['model_family', 'model_size']):
        seeds = group['seed'].unique()
        print(f"  {fam} {size}: seeds {sorted(seeds)}")

    # Group by model family and size, compute mean and SD across steps
    # First, get mean sparsity per (family, size, step) across seeds
    step_means = df.groupby(['model_family', 'model_size', 'step'])['sparsity'].mean().reset_index()

    # Then compute mean and SD across steps (shows per-step variability)
    stats = step_means.groupby(['model_family', 'model_size'])['sparsity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['model_family', 'model_size', 'mean', 'sd']

    # Organize into the expected format
    data = {}
    family_mapping = {
        'Qwen': 'Qwen2.5',
        'Llama': 'Llama-3.2',
        'Gemma': 'Gemma-3',
    }

    # Define size order for each family
    size_order = {
        'Qwen2.5': ['0.5B', '1.5B', '7B'],
        'Llama-3.2': ['3B'],
        'Gemma-3': ['1B', '4B'],
    }

    for _, row in stats.iterrows():
        family = family_mapping.get(row['model_family'], row['model_family'])
        size = row['model_size']
        mean = row['mean']
        sd_val = row['sd'] if not pd.isna(row['sd']) else 0.0

        if family not in data:
            data[family] = []
        data[family].append((size, mean, sd_val))

    # Sort by size order
    for family in data:
        if family in size_order:
            order = {s: i for i, s in enumerate(size_order[family])}
            data[family].sort(key=lambda x: order.get(x[0], 999))

    return data


def create_sparsity_line_plot(
    data: dict = None,
    output_path: str = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
):
    """
    Create a line plot showing sparsity vs k value for each model.

    Args:
        data: Dictionary with model names as keys, containing:
              {model_name: [(k, mean, se), ...]}
              If None, uses placeholder data.
        output_path: Path to save the figure. If None, displays interactively.
        figsize: Figure size in inches (width, height).
        dpi: Resolution for saved figure.
    """

    # Default placeholder data
    if data is None:
        data = {
            'Qwen2.5 0.5B': [(1, 99.3, 0.1), (2, 98.5, 0.2), (4, 97.2, 0.3), (8, 95.1, 0.4)],
            'Qwen2.5 1.5B': [(1, 99.4, 0.1), (2, 98.7, 0.2), (4, 97.5, 0.3), (8, 95.8, 0.4)],
            'Llama-3.2 3B': [(1, 99.1, 0.1), (2, 98.3, 0.2), (4, 97.0, 0.3), (8, 94.9, 0.4)],
        }

    # Style settings for each model family
    family_styles = {
        'Qwen2.5': {
            'color': COVENANT_RED,
            'marker': 'o',
        },
        'Llama-3.2': {
            'color': '#404040',
            'marker': 's',
        },
        'Gemma-3': {
            'color': COVENANT_BLACK_500,
            'marker': '^',
        },
    }

    # Line styles for different sizes within a family
    size_linestyles = {
        '0.5B': '-',
        '1B': '-',
        '1.5B': '--',
        '3B': '-',
        '4B': '--',
        '7B': ':',
    }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot each model
    for model_name, points in data.items():
        # Parse family and size from model name
        parts = model_name.split()
        family = parts[0]
        size = parts[1] if len(parts) > 1 else ''

        style = family_styles.get(family, {'color': COVENANT_BLACK_500, 'marker': 'o'})
        linestyle = size_linestyles.get(size, '-')

        k_vals = np.array([p[0] for p in points])
        means = np.array([p[1] for p in points])
        ses = np.array([p[2] for p in points])

        # Plot shaded confidence band
        ax.fill_between(k_vals, means - ses, means + ses,
                        color=style['color'], alpha=0.15, linewidth=0, zorder=1)

        # Plot line with markers
        line, = ax.plot(k_vals, means, color=style['color'], linestyle=linestyle,
                        linewidth=2.5, marker=style['marker'], markersize=10,
                        markerfacecolor='white', markeredgewidth=2.5,
                        markeredgecolor=style['color'], label=model_name, zorder=3)

    # Configure axes
    k_values_all = sorted(set(k for points in data.values() for k, _, _ in points))
    ax.set_xticks(k_values_all)
    ax.set_xticklabels([str(k) for k in k_values_all], fontsize=18, fontweight='bold')
    ax.set_xlabel('Step Gap (k)', fontsize=22, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.tick_params(axis='x', colors=COVENANT_BLACK_500, width=1.5, length=8, pad=10)

    # Y-axis
    ax.set_ylim(95, 100)
    ax.set_yticks([95, 96, 97, 98, 99, 100])
    ax.set_ylabel('Sparsity (%)', fontsize=22, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.tick_params(axis='y', labelsize=18, colors=COVENANT_BLACK_1000, width=1.5, length=8)

    # Grid
    ax.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100,
                  linewidth=0.8, zorder=1, alpha=0.8)
    ax.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100,
                  linewidth=0.8, zorder=1, alpha=0.8)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COVENANT_BLACK_500)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(COVENANT_BLACK_500)
    ax.spines['bottom'].set_linewidth(1.2)

    # Legend
    ax.legend(loc='lower left', fontsize=14, framealpha=0.95,
              edgecolor=COVENANT_BLACK_500, fancybox=False,
              borderpad=0.8, handlelength=2.5)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def load_sparsity_vs_k_data(
    csv_paths: list[str],
    learning_rate: float = 3e-6,
    iteration_num: int = 1,
) -> dict:
    """
    Load sparsity data for line plot (sparsity vs k).

    Args:
        csv_paths: List of paths to sparsity CSV files.
        learning_rate: Filter to this learning rate (default 3e-6).
        iteration_num: Filter to this iteration number (default 1).

    Returns:
        Dictionary formatted for create_sparsity_line_plot.
    """
    import pandas as pd

    # Load and concatenate all CSVs
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {path}")

    if not dfs:
        print("No CSV files found!")
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # Parse learning_rate column if it's a string
    if df['learning_rate'].dtype == object:
        df['learning_rate'] = df['learning_rate'].apply(lambda x: float(x))

    # Filter by learning rate
    lr_tolerance = learning_rate * 0.1
    df = df[(df['learning_rate'] >= learning_rate - lr_tolerance) &
            (df['learning_rate'] <= learning_rate + lr_tolerance)]
    print(f"After lr={learning_rate:.0e} filter: {len(df)} rows")

    # Filter by iteration number
    df = df[df['iteration_num'] == iteration_num]
    print(f"After iter={iteration_num} filter: {len(df)} rows")

    if len(df) == 0:
        print("No data after filtering!")
        return None

    # Get unique k values
    k_values = sorted(df['k'].unique())
    print(f"K values: {k_values}")

    # Family name mapping
    family_mapping = {
        'Qwen': 'Qwen2.5',
        'Llama': 'Llama-3.2',
        'Gemma': 'Gemma-3',
    }

    # For each model (family + size), compute mean and SD for each k
    # First, get mean sparsity per (family, size, step, k) across seeds
    step_means = df.groupby(['model_family', 'model_size', 'step', 'k'])['sparsity'].mean().reset_index()

    # Then compute mean and SD across steps for each (family, size, k)
    # This shows per-step variability
    stats = step_means.groupby(['model_family', 'model_size', 'k'])['sparsity'].agg(['mean', 'std']).reset_index()
    stats.columns = ['model_family', 'model_size', 'k', 'mean', 'sd']

    # Organize into expected format
    data = {}
    for (family, size), group in stats.groupby(['model_family', 'model_size']):
        family_name = family_mapping.get(family, family)
        model_name = f"{family_name} {size}"

        points = []
        for _, row in group.sort_values('k').iterrows():
            points.append((int(row['k']), row['mean'], row['sd'] if not pd.isna(row['sd']) else 0.0))

        data[model_name] = points

    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create sparsity plots')
    parser.add_argument('--csv', nargs='+', help='CSV files with sparsity data')
    parser.add_argument('--output', '-o', default='research/figures/sparsity_plot.png',
                        help='Output path for the plot')
    parser.add_argument('--type', choices=['bar', 'line'], default='bar',
                        help='Plot type: bar (sparsity by model) or line (sparsity vs k)')
    parser.add_argument('--k', type=int, default=1, help='K value for bar plot (default: 1)')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate filter (default: 3e-6)')
    parser.add_argument('--iter', type=int, default=1, help='Iteration number filter (default: 1)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.type == 'bar':
        if args.csv:
            data = load_sparsity_data_from_csv(
                args.csv,
                k_value=args.k,
                learning_rate=args.lr,
                iteration_num=args.iter,
            )
            if data:
                print("\nData to plot:")
                for family, items in data.items():
                    for size, mean, se in items:
                        print(f"  {family} {size}: {mean:.2f} ± {se:.2f}")
        else:
            print("Using placeholder data...")
            data = None

        create_sparsity_bar_plot(data=data, output_path=str(output_path), dpi=args.dpi)

    elif args.type == 'line':
        if args.csv:
            data = load_sparsity_vs_k_data(
                args.csv,
                learning_rate=args.lr,
                iteration_num=args.iter,
            )
            if data:
                print("\nData to plot:")
                for model, points in data.items():
                    print(f"  {model}:")
                    for k, mean, se in points:
                        print(f"    k={k}: {mean:.2f} ± {se:.2f}")
        else:
            print("Using placeholder data...")
            data = None

        create_sparsity_line_plot(data=data, output_path=str(output_path), dpi=args.dpi)
