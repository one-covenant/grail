#!/usr/bin/env python3
"""
Create combined bar plots comparing model performance across GSM8K, MATH, and AMC 2023.

Three designs:
A) Classic grouped bar chart (datasets on x-axis)
B) Simplified two-bar comparison with improvement annotations
C) Horizontal layout with annotations

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
    "dark_gray": "#404040",
    "light_gray": "#888888",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
datasets = ["GSM8K", "MATH", "AMC 2023"]

base_0shot = {"values": [6.0, 1.9, 7.5], "stderr": [0.65, 0.19, 4.22]}
base_4shot = {"values": [57.9, 12.7, 7.5], "stderr": [1.36, 0.47, 4.22]}
trained = {"values": [72.2, 47.6, 25.0], "stderr": [1.23, 0.66, 6.9]}


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


def design_a_grouped_vertical() -> None:
    """
    Design A: Classic grouped bar chart with datasets on x-axis.
    Three bars per dataset: Base 0-shot, Base 4-shot, Trained.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS["white"])
    
    x = np.arange(len(datasets))
    bar_width = 0.26
    
    # Error bar styling - subtle, professional
    error_kw_light = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#555555"}
    error_kw_dark = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#666666"}
    error_kw_red = {"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#AA2020"}
    
    # Create bars
    bars1 = ax.bar(
        x - bar_width, base_0shot["values"], bar_width,
        label="Base (0-shot)", color=COLORS["light_gray"],
        yerr=base_0shot["stderr"], capsize=3, error_kw=error_kw_light,
    )
    bars2 = ax.bar(
        x, base_4shot["values"], bar_width,
        label="Base (4-shot)", color=COLORS["black"],
        yerr=base_4shot["stderr"], capsize=3, error_kw=error_kw_dark,
    )
    bars3 = ax.bar(
        x + bar_width, trained["values"], bar_width,
        label="grail-v0 (0-shot)", color=COLORS["red"],
        yerr=trained["stderr"], capsize=3, error_kw=error_kw_red,
    )
    
    # Add value labels - LARGER fonts
    for bars, err_vals in [(bars1, base_0shot["stderr"]), 
                            (bars2, base_4shot["stderr"]), 
                            (bars3, trained["stderr"])]:
        for bar, err in zip(bars, err_vals):
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height + err),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=16, fontweight="bold",
                color=COLORS["black"],
            )
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=22)
    ax.set_title("Model Performance Across Benchmarks", fontweight="bold", pad=22, fontsize=28)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 88)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    # Legend - larger
    legend = ax.legend(
        loc="upper right", frameon=True, fancybox=False,
        borderpad=0.7, handlelength=1.8, fontsize=16,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    
    output_path = Path(__file__).parent / "combined_design_a.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def design_b_simplified() -> None:
    """
    Design B: Simplified comparison - Base 0-shot vs Trained only.
    Cleaner, with improvement multipliers annotated.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(datasets))
    bar_width = 0.35
    
    # Calculate improvement multipliers
    improvements = [t / b for t, b in zip(trained["values"], base_0shot["values"])]
    
    # Create bars
    bars1 = ax.bar(
        x - bar_width/2, base_0shot["values"], bar_width,
        label="Base (0-shot)", color=COLORS["black"],
        yerr=base_0shot["stderr"], capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]},
    )
    bars2 = ax.bar(
        x + bar_width/2, trained["values"], bar_width,
        label="grail-v0 (0-shot)", color=COLORS["red"],
        yerr=trained["stderr"], capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]},
    )
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["black"],
        )
    
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["black"],
        )
        # Add improvement annotation
        ax.annotate(
            f"{imp:.0f}×",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 24),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=COLORS["red"],
        )
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_title("Training Impact: Base vs Trained Model", fontweight="bold", pad=20, fontsize=24)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=16)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    # Legend
    legend = ax.legend(
        loc="upper right", frameon=True, fancybox=False,
        borderpad=0.6, handlelength=1.5, fontsize=14,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    
    output_path = Path(__file__).parent / "combined_design_b.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


def design_c_horizontal() -> None:
    """
    Design C: Horizontal bar chart - better for labels, modern look.
    Shows all three configurations with clear hierarchy.
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS["white"])
    
    y = np.arange(len(datasets))
    bar_height = 0.25
    
    # Create horizontal bars (note: reversed order for top-to-bottom reading)
    bars1 = ax.barh(
        y + bar_height, base_0shot["values"], bar_height,
        label="Base (0-shot)", color=COLORS["light_gray"],
        xerr=base_0shot["stderr"], capsize=4,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]},
    )
    bars2 = ax.barh(
        y, base_4shot["values"], bar_height,
        label="Base (4-shot)", color=COLORS["black"],
        xerr=base_4shot["stderr"], capsize=4,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]},
    )
    bars3 = ax.barh(
        y - bar_height, trained["values"], bar_height,
        label="grail-v0 (0-shot)", color=COLORS["red"],
        xerr=trained["stderr"], capsize=4,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]},
    )
    
    # Add value labels at end of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.1f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(8, 0),
                textcoords="offset points",
                ha="left", va="center",
                fontsize=13, fontweight="bold", color=COLORS["black"],
            )
    
    ax.set_ylabel("Benchmark", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_xlabel("Accuracy (%)", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_title("Model Performance Across Benchmarks", fontweight="bold", pad=20, fontsize=24)
    
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=16)
    ax.set_xlim(0, 90)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    # Remove top/right spines, keep left/bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend at bottom
    legend = ax.legend(
        loc="lower right", frameon=True, fancybox=False,
        borderpad=0.6, handlelength=1.5, fontsize=14,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    
    output_path = Path(__file__).parent / "combined_design_c.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    design_a_grouped_vertical()
    design_b_simplified()
    design_c_horizontal()
    print("\n✓ All three designs generated!")

