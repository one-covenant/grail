#!/usr/bin/env python3
"""
Generate comparison plots (GRAIL vs TRL, bucketing analysis).

Outputs:
- fig6_grail_vs_trl.png
- fig7_bucketing_comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

from .style import COLORS, setup_style

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# GRAIL vs TRL Data
# ─────────────────────────────────────────────────────────────────────────────
DATASETS = ["GSM8K", "MATH", "AMC 2023"]
TRL_DATA = {"values": [66.2, 38.4, 15.0], "stderr": [1.30, 0.65, 5.72]}
GRAIL_DATA = {"values": [72.2, 47.6, 25.0], "stderr": [1.23, 0.66, 6.9]}


def create_grail_vs_trl_plot() -> plt.Figure:
    """Create GRAIL vs TRL comparison bar plot."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(DATASETS))
    bar_width = 0.35
    
    error_kw_trl = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#555555"}
    error_kw_grail = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#AA2020"}
    
    bars_trl = ax.bar(
        x - bar_width/2, TRL_DATA["values"], bar_width,
        label="TRL", color=COLORS["black"],
        yerr=TRL_DATA["stderr"], capsize=4, error_kw=error_kw_trl,
    )
    bars_grail = ax.bar(
        x + bar_width/2, GRAIL_DATA["values"], bar_width,
        label="grail-v0", color=COLORS["red"],
        yerr=GRAIL_DATA["stderr"], capsize=4, error_kw=error_kw_grail,
    )
    
    # Add value labels
    for bars, data in [(bars_trl, TRL_DATA), (bars_grail, GRAIL_DATA)]:
        for bar, val, err in zip(bars, data["values"], data["stderr"]):
            height = bar.get_height()
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height + err),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=16, fontweight="bold", color=COLORS["black"],
            )
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_title("grail-v0 vs TRL: Test Set Performance", fontweight="bold", pad=22, fontsize=26)
    
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 88)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    legend = ax.legend(loc="upper right", frameon=True, fancybox=False,
                       borderpad=0.7, handlelength=1.8, fontsize=18)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Bucketing Comparison (uses different style - dark theme)
# ─────────────────────────────────────────────────────────────────────────────
DARK_COLORS = {
    'bg': '#0d1117',
    'grid': '#21262d',
    'text': '#c9d1d9',
    'accent1': '#58a6ff',
    'accent2': '#f78166',
    'accent3': '#7ee787',
    'accent4': '#d2a8ff',
    'warning': '#d29922',
}

B = 16  # Number of buckets


def log_magnitude_bucket(value: float, num_buckets: int = B) -> int:
    """GRAIL logarithmic bucketing implementation."""
    abs_val = abs(value)
    if abs_val < 1e-6:
        return 0
    log_val = math.log2(abs_val + 1.0)
    scale_factor = num_buckets / 10.0
    bucket = int(log_val * scale_factor)
    bucket = max(0, min(num_buckets - 1, bucket))
    return bucket if value >= 0 else -bucket


def linear_bucket(value: float, max_val: float = 100.0, num_buckets: int = B) -> int:
    """Simple linear bucketing for comparison."""
    abs_val = min(abs(value), max_val)
    bucket = int((abs_val / max_val) * (num_buckets - 1))
    bucket = max(0, min(num_buckets - 1, bucket))
    return bucket if value >= 0 else -bucket


def create_bucketing_comparison() -> plt.Figure:
    """Create bucketing comparison visualization."""
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(16, 14), facecolor=DARK_COLORS['bg'])
    
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], 
                          hspace=0.35, wspace=0.25,
                          left=0.08, right=0.95, top=0.92, bottom=0.06)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor(DARK_COLORS['bg'])
        ax.tick_params(colors=DARK_COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(DARK_COLORS['grid'])
    
    # Plot 1: Bucket widths
    log_boundaries = [2 ** (b / 1.6) - 1 for b in range(B + 1)]
    log_widths = [log_boundaries[i+1] - log_boundaries[i] for i in range(B)]
    linear_widths = [100.0 / B] * B
    
    x = np.arange(B)
    width = 0.35
    ax1.bar(x - width/2, log_widths, width, label='Logarithmic', 
            color=DARK_COLORS['accent1'], alpha=0.8)
    ax1.bar(x + width/2, linear_widths, width, label='Linear', 
            color=DARK_COLORS['accent2'], alpha=0.8)
    ax1.set_xlabel('Bucket Index', color=DARK_COLORS['text'], fontsize=12)
    ax1.set_ylabel('Bucket Width', color=DARK_COLORS['text'], fontsize=12)
    ax1.set_title('Bucket Width Distribution', color=DARK_COLORS['text'], fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')
    
    # Plot 2: Value-to-bucket mapping
    values = np.logspace(-2, 2, 200)
    log_buckets = [log_magnitude_bucket(v) for v in values]
    linear_buckets = [linear_bucket(v) for v in values]
    
    ax2.plot(values, log_buckets, color=DARK_COLORS['accent1'], linewidth=2, label='Logarithmic')
    ax2.plot(values, linear_buckets, color=DARK_COLORS['accent2'], linewidth=2, label='Linear')
    ax2.set_xscale('log')
    ax2.set_xlabel('Value', color=DARK_COLORS['text'], fontsize=12)
    ax2.set_ylabel('Bucket Index', color=DARK_COLORS['text'], fontsize=12)
    ax2.set_title('Value-to-Bucket Mapping', color=DARK_COLORS['text'], fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Plot 3 & 4: FP error sensitivity
    for ax, bucket_func, name, color in [
        (ax3, log_magnitude_bucket, 'Logarithmic', DARK_COLORS['accent1']),
        (ax4, linear_bucket, 'Linear', DARK_COLORS['accent2'])
    ]:
        base_values = np.logspace(-1, 2, 50)
        errors = np.logspace(-10, -3, 50)
        X, Y = np.meshgrid(base_values, errors)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                base = X[i, j]
                err = Y[i, j]
                b1 = bucket_func(base)
                b2 = bucket_func(base + err)
                Z[i, j] = abs(b1 - b2)
        
        im = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Base Value', color=DARK_COLORS['text'], fontsize=12)
        ax.set_ylabel('Error Magnitude', color=DARK_COLORS['text'], fontsize=12)
        ax.set_title(f'{name}: Bucket Change from FP Error', color=DARK_COLORS['text'], fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Bucket Difference')
    
    # Plot 5: Summary
    ax5.text(0.5, 0.8, 'Logarithmic Bucketing Advantages', 
             transform=ax5.transAxes, fontsize=18, fontweight='bold',
             color=DARK_COLORS['text'], ha='center')
    
    advantages = [
        '✓ Smaller buckets for small values → higher precision where needed',
        '✓ Stable under floating-point errors (relative error tolerance)',
        '✓ Handles wide dynamic range (10⁻⁶ to 10²) efficiently',
        '✓ Cross-platform reproducibility (GPU/CPU differences absorbed)',
    ]
    
    for i, adv in enumerate(advantages):
        ax5.text(0.1, 0.6 - i*0.15, adv, transform=ax5.transAxes, fontsize=14,
                color=DARK_COLORS['accent3'], ha='left')
    
    ax5.axis('off')
    
    fig.suptitle('GRAIL: Logarithmic vs Linear Bucketing Analysis', 
                 fontsize=20, fontweight='bold', color=DARK_COLORS['text'], y=0.98)
    
    return fig


def generate_all() -> None:
    """Generate all comparison figures."""
    # GRAIL vs TRL (uses standard white style)
    setup_style()
    fig = create_grail_vs_trl_plot()
    output_path = FIGURES_DIR / "fig6_grail_vs_trl.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ {output_path}")
    plt.close(fig)
    
    # Bucketing comparison (uses dark style)
    fig = create_bucketing_comparison()
    output_path = FIGURES_DIR / "fig7_bucketing_comparison.png"
    fig.savefig(output_path, dpi=200, facecolor=DARK_COLORS['bg'], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ {output_path}")
    plt.close(fig)
    
    # Reset to default style
    plt.style.use('default')


if __name__ == "__main__":
    generate_all()

