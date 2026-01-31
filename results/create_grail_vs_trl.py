#!/usr/bin/env python3
"""
Create bar plot comparing GRAIL-v0 vs TRL performance on GSM8K, MATH, and AMC 2023.

Color palette:
- White: #FFFFFF
- Red: #FF3A3A (GRAIL)
- Black: #101010 (TRL)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white": "#FFFFFF",
    "red": "#FF3A3A",
    "black": "#101010",
    "light_gray": "#888888",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
datasets = ["GSM8K", "MATH", "AMC 2023"]

# TRL results
trl = {"values": [66.2, 38.4, 15.0], "stderr": [1.30, 0.65, 5.72]}

# GRAIL-v0 results (our trained model)
grail = {"values": [72.2, 47.6, 25.0], "stderr": [1.23, 0.66, 6.9]}


def setup_style() -> None:
    """Configure matplotlib for elegant, modern styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "font.weight": "normal",
        "figure.facecolor": COLORS["white"],
        "figure.edgecolor": COLORS["white"],
        "figure.dpi": 150,
        "axes.facecolor": COLORS["white"],
        "axes.edgecolor": COLORS["light_gray"],
        "axes.labelcolor": COLORS["black"],
        "axes.titlecolor": COLORS["black"],
        "axes.labelsize": 18,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.grid": False,
        "lines.linewidth": 2.5,
        "lines.antialiased": True,
    })


def create_comparison_plot() -> None:
    """Create GRAIL vs TRL comparison bar plot."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(datasets))
    bar_width = 0.35
    
    # Error bar styling
    error_kw_trl = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#555555"}
    error_kw_grail = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#AA2020"}
    
    # Create bars
    bars_trl = ax.bar(
        x - bar_width/2, trl["values"], bar_width,
        label="TRL", color=COLORS["black"],
        yerr=trl["stderr"], capsize=4, error_kw=error_kw_trl,
    )
    bars_grail = ax.bar(
        x + bar_width/2, grail["values"], bar_width,
        label="grail-v0", color=COLORS["red"],
        yerr=grail["stderr"], capsize=4, error_kw=error_kw_grail,
    )
    
    # Add value labels
    for bars, data in [(bars_trl, trl), (bars_grail, grail)]:
        for bar, val, err in zip(bars, data["values"], data["stderr"]):
            height = bar.get_height()
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height + err),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=16, fontweight="bold",
                color=COLORS["black"],
            )
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_title("grail-v0 vs TRL: Test Set Performance", fontweight="bold", pad=22, fontsize=26)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 88)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    # Legend
    legend = ax.legend(
        loc="upper right", frameon=True, fancybox=False,
        borderpad=0.7, handlelength=1.8, fontsize=18,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    
    output_path = Path(__file__).parent / "grail_vs_trl.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    create_comparison_plot()

