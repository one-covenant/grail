#!/usr/bin/env python3
"""
Generate benchmark bar plots comparing base model vs trained model.

Outputs:
- fig2_gsm8k_benchmark.png
- fig3_math_benchmark.png
- fig4_amc2023_benchmark.png
- fig5_combined_benchmarks_*.png (3 design variants)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .style import COLORS, setup_style

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_DATA = {
    "gsm8k": {
        "title": "GSM8K Test Set Performance",
        "labels": ["Base\n(0-shot)", "Base\n(4-shot)", "Trained\n(0-shot)"],
        "values": [6.0, 57.9, 72.2],
        "stderr": [0.65, 1.36, 1.23],
        "ylim": 85,
    },
    "math": {
        "title": "MATH Test Set Performance",
        "labels": ["Base\n(0-shot)", "Base\n(4-shot)", "Trained\n(0-shot)"],
        "values": [1.9, 12.7, 47.6],
        "stderr": [0.19, 0.47, 0.66],
        "ylim": 60,
    },
    "amc2023": {
        "title": "AMC 2023 Performance",
        "labels": ["Base\n(0-shot)", "Base\n(4-shot)", "Trained\n(0-shot)"],
        "values": [7.5, 7.5, 25.0],
        "stderr": [4.22, 4.22, 6.9],
        "ylim": 40,
    },
}

# Combined data for multi-benchmark plots
DATASETS = ["GSM8K", "MATH", "AMC 2023"]
BASE_0SHOT = {"values": [6.0, 1.9, 7.5], "stderr": [0.65, 0.19, 4.22]}
BASE_4SHOT = {"values": [57.9, 12.7, 7.5], "stderr": [1.36, 0.47, 4.22]}
TRAINED = {"values": [72.2, 47.6, 25.0], "stderr": [1.23, 0.66, 6.9]}


def create_single_benchmark_plot(benchmark: str) -> None:
    """Create a single benchmark bar plot."""
    data = BENCHMARK_DATA[benchmark]
    
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(data["labels"]))
    bar_width = 0.6
    colors = [COLORS["black"], COLORS["black"], COLORS["red"]]
    
    bars = ax.bar(
        x, data["values"], bar_width,
        color=colors, edgecolor="none",
        yerr=data["stderr"], capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]},
    )
    
    # Add value labels
    for bar, val, err in zip(bars, data["values"], data["stderr"]):
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height + err),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=18, fontweight="bold", color=COLORS["black"],
        )
    
    ax.set_xlabel("Model Configuration", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title(data["title"], fontweight="bold", pad=20, fontsize=26)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data["labels"], fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_ylim(0, data["ylim"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    ax.grid(False)
    ax.spines["left"].set_color(COLORS["light_gray"])
    ax.spines["bottom"].set_color(COLORS["light_gray"])
    
    fig.tight_layout()
    return fig


def create_combined_design_a() -> plt.Figure:
    """Design A: Classic grouped bar chart with datasets on x-axis."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(DATASETS))
    bar_width = 0.25
    
    bars1 = ax.bar(x - bar_width, BASE_0SHOT["values"], bar_width,
                   label="Base (0-shot)", color=COLORS["light_gray"],
                   yerr=BASE_0SHOT["stderr"], capsize=4,
                   error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    bars2 = ax.bar(x, BASE_4SHOT["values"], bar_width,
                   label="Base (4-shot)", color=COLORS["black"],
                   yerr=BASE_4SHOT["stderr"], capsize=4,
                   error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    bars3 = ax.bar(x + bar_width, TRAINED["values"], bar_width,
                   label="Trained (0-shot)", color=COLORS["red"],
                   yerr=TRAINED["stderr"], capsize=4,
                   error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_title("Model Performance Across Benchmarks", fontweight="bold", pad=20, fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=16)
    ax.set_ylim(0, 85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    legend = ax.legend(loc="upper right", frameon=True, fontsize=14)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    return fig


def create_combined_design_b() -> plt.Figure:
    """Design B: Simplified comparison with improvement multipliers."""
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=COLORS["white"])
    
    x = np.arange(len(DATASETS))
    bar_width = 0.35
    improvements = [t / b for t, b in zip(TRAINED["values"], BASE_0SHOT["values"])]
    
    bars1 = ax.bar(x - bar_width/2, BASE_0SHOT["values"], bar_width,
                   label="Base (0-shot)", color=COLORS["black"],
                   yerr=BASE_0SHOT["stderr"], capsize=5,
                   error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]})
    bars2 = ax.bar(x + bar_width/2, TRAINED["values"], bar_width,
                   label="Trained (0-shot)", color=COLORS["red"],
                   yerr=TRAINED["stderr"], capsize=5,
                   error_kw={"elinewidth": 2, "capthick": 2, "ecolor": COLORS["light_gray"]})
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.annotate(f"{imp:.0f}×", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 24), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color=COLORS["red"])
    
    ax.set_xlabel("Benchmark", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_title("Training Impact: Base vs Trained Model", fontweight="bold", pad=20, fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=16)
    ax.set_ylim(0, 90)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    legend = ax.legend(loc="upper right", frameon=True, fontsize=14)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    return fig


def create_combined_design_c() -> plt.Figure:
    """Design C: Horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=COLORS["white"])
    
    y = np.arange(len(DATASETS))
    bar_height = 0.25
    
    ax.barh(y + bar_height, BASE_0SHOT["values"], bar_height,
            label="Base (0-shot)", color=COLORS["light_gray"],
            xerr=BASE_0SHOT["stderr"], capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    ax.barh(y, BASE_4SHOT["values"], bar_height,
            label="Base (4-shot)", color=COLORS["black"],
            xerr=BASE_4SHOT["stderr"], capsize=4,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    bars3 = ax.barh(y - bar_height, TRAINED["values"], bar_height,
                    label="Trained (0-shot)", color=COLORS["red"],
                    xerr=TRAINED["stderr"], capsize=4,
                    error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": COLORS["dark_gray"]})
    
    for bar in bars3:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}%", xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=13, fontweight="bold")
    
    ax.set_ylabel("Benchmark", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_xlabel("Accuracy (%)", fontweight="medium", labelpad=12, fontsize=18)
    ax.set_title("Model Performance Across Benchmarks", fontweight="bold", pad=20, fontsize=24)
    ax.set_yticks(y)
    ax.set_yticklabels(DATASETS, fontsize=16)
    ax.set_xlim(0, 90)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    
    legend = ax.legend(loc="lower right", frameon=True, fontsize=14)
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor(COLORS["light_gray"])
    
    fig.tight_layout()
    return fig


def generate_all() -> None:
    """Generate all benchmark figures."""
    setup_style()
    
    # Individual benchmark plots
    for benchmark, fignum in [("gsm8k", 2), ("math", 3), ("amc2023", 4)]:
        fig = create_single_benchmark_plot(benchmark)
        output_path = FIGURES_DIR / f"fig{fignum}_{benchmark}_benchmark.png"
        fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
        print(f"✓ {output_path}")
        plt.close(fig)
    
    # Combined plots (3 designs)
    for design_func, suffix in [
        (create_combined_design_a, "full"),
        (create_combined_design_b, "simplified"),
        (create_combined_design_c, "horizontal"),
    ]:
        fig = design_func()
        output_path = FIGURES_DIR / f"fig5_combined_benchmarks_{suffix}.png"
        fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
        print(f"✓ {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    generate_all()

