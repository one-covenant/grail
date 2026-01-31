#!/usr/bin/env python3
"""
Plot SFT vs GRPO sparsity over training steps.
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

# Elegant colors for comparison
GRPO_COLOR = '#0F4C75'       # Elegant teal blue
SFT_3E6_COLOR = '#BE3144'    # Luxe burgundy red
SFT_2E5_COLOR = '#C17817'    # Rich amber gold

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = COVENANT_BLACK_500


def load_data():
    """Load and prepare GRPO and SFT data."""
    # Load GRPO data
    grpo = pd.read_csv('data/sparsity_k_step_combined.csv')
    grpo = grpo[(grpo['model_family'] == 'Qwen') &
                (grpo['model_size'] == '1.5B') &
                (grpo['learning_rate'] == 3e-6) &
                (grpo['k'] == 1) &
                (grpo['iteration_num'] == 1)]

    # Load SFT data
    sft = pd.read_csv('data/sparsity_sft.csv')
    sft = sft[sft['k'] == 1]

    # Split SFT by learning rate
    sft_3e6 = sft[sft['learning_rate'] == 3e-6]
    sft_2e5 = sft[sft['learning_rate'] == 2e-5]

    # Aggregate by step
    grpo_agg = grpo.groupby('step')['sparsity'].agg(['mean', 'std']).reset_index()
    grpo_agg.columns = ['step', 'mean', 'std']

    sft_3e6_agg = sft_3e6.groupby('step')['sparsity'].agg(['mean', 'std']).reset_index()
    sft_3e6_agg.columns = ['step', 'mean', 'std']

    sft_2e5_agg = sft_2e5.groupby('step')['sparsity'].agg(['mean', 'std']).reset_index()
    sft_2e5_agg.columns = ['step', 'mean', 'std']

    return grpo_agg, sft_3e6_agg, sft_2e5_agg


def create_comparison_plot(
    grpo_data: pd.DataFrame,
    sft_3e6_data: pd.DataFrame,
    sft_2e5_data: pd.DataFrame,
    output_path: str = None,
    figsize: tuple = (12, 7),
    dpi: int = 200,
):
    """Create line plot comparing SFT vs GRPO sparsity."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot GRPO (lr=3e-6)
    grpo_steps = grpo_data['step'].values
    grpo_means = grpo_data['mean'].values
    grpo_stds = grpo_data['std'].values

    ax.fill_between(grpo_steps, grpo_means - grpo_stds, grpo_means + grpo_stds,
                    color=GRPO_COLOR, alpha=0.15, linewidth=0)
    ax.plot(grpo_steps, grpo_means, color=GRPO_COLOR, linewidth=2.5,
            label='GRPO (lr=3e-6)', zorder=3)

    # Plot SFT lr=3e-6
    sft_3e6_steps = sft_3e6_data['step'].values
    sft_3e6_means = sft_3e6_data['mean'].values
    sft_3e6_stds = sft_3e6_data['std'].values

    ax.fill_between(sft_3e6_steps, sft_3e6_means - sft_3e6_stds, sft_3e6_means + sft_3e6_stds,
                    color=SFT_3E6_COLOR, alpha=0.15, linewidth=0)
    ax.plot(sft_3e6_steps, sft_3e6_means, color=SFT_3E6_COLOR, linewidth=2.5,
            label='SFT (lr=3e-6)', zorder=3)

    # Plot SFT lr=2e-5
    sft_2e5_steps = sft_2e5_data['step'].values
    sft_2e5_means = sft_2e5_data['mean'].values
    sft_2e5_stds = sft_2e5_data['std'].values

    ax.fill_between(sft_2e5_steps, sft_2e5_means - sft_2e5_stds, sft_2e5_means + sft_2e5_stds,
                    color=SFT_2E5_COLOR, alpha=0.15, linewidth=0)
    ax.plot(sft_2e5_steps, sft_2e5_means, color=SFT_2E5_COLOR, linewidth=2.5,
            label='SFT (lr=2e-5)', zorder=3)

    # Configure axes
    ax.set_xlabel('Training Step', fontsize=18, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.set_ylabel('Sparsity (%)', fontsize=18, fontweight='bold',
                  color=COVENANT_BLACK_1000, labelpad=12)
    ax.set_title('Weight Update Sparsity: GRPO vs SFT (Qwen 1.5B)', fontsize=20, fontweight='bold',
                 color=COVENANT_BLACK_1000, pad=15)

    # Y-axis range
    y_min = min(sft_2e5_means.min(), sft_3e6_means.min(), grpo_means.min()) - 1
    y_min = max(92, np.floor(y_min))
    ax.set_ylim(y_min, 100.5)

    ax.tick_params(axis='both', labelsize=12, colors=COVENANT_BLACK_1000)

    # Grid
    ax.yaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.8, alpha=0.8)
    ax.xaxis.grid(True, linestyle='--', color=COVENANT_WHITE_100, linewidth=0.5, alpha=0.5)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COVENANT_BLACK_500)
    ax.spines['bottom'].set_color(COVENANT_BLACK_500)

    # Legend
    ax.legend(loc='lower left', fontsize=13, framealpha=0.95,
              edgecolor=COVENANT_BLACK_500, fancybox=False, borderpad=0.8)

    # Add summary box
    grpo_mean = grpo_means.mean()
    sft_3e6_mean = sft_3e6_means.mean()
    sft_2e5_mean = sft_2e5_means.mean()

    textstr = (f'Mean sparsity:\n'
               f'  GRPO (lr=3e-6): {grpo_mean:.1f}%\n'
               f'  SFT (lr=3e-6): {sft_3e6_mean:.1f}%\n'
               f'  SFT (lr=2e-5): {sft_2e5_mean:.1f}%')
    props = dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor=COVENANT_BLACK_500, alpha=0.95)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, color=COVENANT_BLACK_1000, family='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved: {output_path}')
    else:
        plt.show()

    plt.close()


def main():
    output_dir = Path('research/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_data, sft_3e6_data, sft_2e5_data = load_data()

    print(f'GRPO (lr=3e-6): {len(grpo_data)} steps, mean={grpo_data["mean"].mean():.2f}%')
    print(f'SFT (lr=3e-6): {len(sft_3e6_data)} steps, mean={sft_3e6_data["mean"].mean():.2f}%')
    print(f'SFT (lr=2e-5): {len(sft_2e5_data)} steps, mean={sft_2e5_data["mean"].mean():.2f}%')

    create_comparison_plot(
        grpo_data, sft_3e6_data, sft_2e5_data,
        output_path=str(output_dir / 'sft_vs_grpo_sparsity.png'),
        dpi=200,
    )


if __name__ == '__main__':
    main()
