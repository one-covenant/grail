#!/usr/bin/env python3
"""
Create a bar plot comparing base model vs trained model on AMC2023 dataset.

Color palette:
- White: #FFFFFF
- Red: #FF3A3A
- Black: #101010
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
# AMC2023 dataset (40 problems)
# Base model: standard evaluation (non-reasoning)
# Trained model: reasoning mode (as trained)
# Standard error provided
data = {
    "labels": ["Base\n(0-shot)", "Base\n(4-shot)", "grail-v0\n(0-shot)"],
    "values": [7.5, 7.5, 25.0],
    "stderr": [4.22, 4.22, 6.9],
    "colors": [COLORS["black"], COLORS["black"], COLORS["red"]],
}


def setup_style() -> None:
    """Configure matplotlib for elegant, modern styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "font.weight": "normal",
        "figure.facecolor": COLORS["white"],
        "figure.edgecolor": COLORS["white"],
        "figure.dpi": 150,
        "axes.facecolor": COLORS["white"],
        "axes.edgecolor": COLORS["light_gray"],
        "axes.labelcolor": COLORS["black"],
        "axes.titlecolor": COLORS["black"],
        "axes.labelsize": 20,
        "axes.titlesize": 26,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.grid": False,
        "lines.linewidth": 2.5,
        "lines.antialiased": True,
    })


def create_bar_plot() -> None:
    """Create the bar plot."""
    setup_style()
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(data["labels"]))
    bar_width = 0.6
    
    # Create bars
    bars = ax.bar(
        x,
        data["values"],
        bar_width,
        color=data["colors"],
        edgecolor="none",
        yerr=data["stderr"],
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]},
    )
    
    # Add value labels on top of bars
    for bar, val, err in zip(bars, data["values"], data["stderr"]):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height + err),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color=COLORS["black"],
        )
    
    # Styling
    ax.set_xlabel("Model Configuration", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title("AMC 2023 Performance", fontweight="bold", pad=20, fontsize=26)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data["labels"], fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    
    # Y-axis from 0 to 40
    ax.set_ylim(0, 40)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    # Clean up
    ax.grid(False)
    ax.spines["left"].set_color(COLORS["light_gray"])
    ax.spines["bottom"].set_color(COLORS["light_gray"])
    
    fig.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "amc2023_performance.png"
    fig.savefig(
        output_path,
        dpi=200,
        facecolor=COLORS["white"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    print(f"✓ Saved plot to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    create_bar_plot()

