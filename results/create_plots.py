#!/usr/bin/env python3
"""
Create stylish line plots for pass@1 and pass@5 metrics on evaluation and training sets.

Color palette:
- White: #EFEFEF
- Red: #FF3A3A
- Black: #101010
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white": "#FFFFFF",  # Pure white background
    "red": "#FF3A3A",
    "black": "#101010",
    "dark_gray": "#1A1A1A",
    "mid_gray": "#2A2A2A",
    "light_gray": "#888888",
}

# Derived colors for the two lines
PASS_1_COLOR = COLORS["black"]  # Strong contrast on white
PASS_5_COLOR = COLORS["red"]    # Bold accent

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Processing
# ─────────────────────────────────────────────────────────────────────────────
def load_and_process_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and merge into eval and training dataframes."""
    
    # Load evaluation data (from grail_training subdirectory)
    data_dir = results_dir / "grail_training"
    eval_pass1_df = pd.read_csv(data_dir / "wandb_export_2025-11-30T16_44_37.800-05_00.csv")
    eval_pass5_df = pd.read_csv(data_dir / "wandb_export_2025-11-30T17_01_56.500-05_00.csv")
    
    # Load training data
    train_pass5_df = pd.read_csv(data_dir / "wandb_export_2025-11-30T17_02_29.789-05_00.csv")
    train_pass1_df = pd.read_csv(data_dir / "wandb_export_2025-11-30T17_02_47.878-05_00.csv")
    
    # Rename columns for clarity
    eval_pass1_df = eval_pass1_df.rename(columns={
        "trainer_grail - eval/pass@1": "pass@1"
    })[["block_number", "pass@1"]]
    
    eval_pass5_df = eval_pass5_df.rename(columns={
        "trainer_grail - eval/pass@5": "pass@5"
    })[["block_number", "pass@5"]]
    
    train_pass1_df = train_pass1_df.rename(columns={
        "trainer_grail - training/prefilter/pass@1": "pass@1"
    })[["block_number", "pass@1"]]
    
    train_pass5_df = train_pass5_df.rename(columns={
        "trainer_grail - training/prefilter/pass@5": "pass@5"
    })[["block_number", "pass@5"]]
    
    # Merge eval data
    eval_df = eval_pass1_df.merge(eval_pass5_df, on="block_number", how="outer").sort_values("block_number")
    
    # Merge training data
    train_df = train_pass1_df.merge(train_pass5_df, on="block_number", how="outer").sort_values("block_number")
    
    # Filter to block_number range: 6991700 to 6994900
    block_min, block_max = 6991700, 6994900
    eval_df = eval_df[(eval_df["block_number"] >= block_min) & (eval_df["block_number"] <= block_max)].copy()
    train_df = train_df[(train_df["block_number"] >= block_min) & (train_df["block_number"] <= block_max)].copy()
    
    # Create update_step: map block_number range to 0-320
    def map_to_update_step(df: pd.DataFrame) -> pd.DataFrame:
        # Linear projection from [block_min, block_max] to [0, 320]
        df["update_step"] = ((df["block_number"] - block_min) / (block_max - block_min)) * 320
        return df
    
    eval_df = map_to_update_step(eval_df)
    train_df = map_to_update_step(train_df)
    
    return eval_df, train_df


def setup_style() -> None:
    """Configure matplotlib for elegant, modern styling."""
    plt.rcParams.update({
        # Font settings - using a modern sans-serif, much larger base size
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "font.weight": "normal",
        
        # Figure - white background from palette
        "figure.facecolor": COLORS["white"],
        "figure.edgecolor": COLORS["white"],
        "figure.dpi": 150,
        
        # Axes - white background, dark text for contrast
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
        
        # Ticks - dark and large for readability
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        
        # No grid - clean minimalist look
        "axes.grid": False,
        
        # Legend - white background, larger font
        "legend.facecolor": COLORS["white"],
        "legend.edgecolor": COLORS["light_gray"],
        "legend.fontsize": 18,
        "legend.framealpha": 0.95,
        
        # Lines
        "lines.linewidth": 2.5,
        "lines.antialiased": True,
    })


def create_plot(
    df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
) -> None:
    """Create a single plot with pass@1 and pass@5 lines."""
    
    # Plot pass@1 with subtle styling
    mask_pass1 = df["pass@1"].notna()
    ax.plot(
        df.loc[mask_pass1, "update_step"],
        df.loc[mask_pass1, "pass@1"],
        color=PASS_1_COLOR,
        linewidth=2.2,
        label="pass@1",
        alpha=0.9,
        zorder=2,
    )
    
    # Plot pass@5 with bold accent color
    mask_pass5 = df["pass@5"].notna()
    ax.plot(
        df.loc[mask_pass5, "update_step"],
        df.loc[mask_pass5, "pass@5"],
        color=PASS_5_COLOR,
        linewidth=2.5,
        label="pass@5",
        alpha=0.95,
        zorder=3,
    )
    
    # Add subtle glow effect for pass@5 line
    ax.plot(
        df.loc[mask_pass5, "update_step"],
        df.loc[mask_pass5, "pass@5"],
        color=PASS_5_COLOR,
        linewidth=7,
        alpha=0.12,
        zorder=1,
    )
    
    # Styling - much larger fonts for readability
    ax.set_xlabel("Update Step", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title(title, fontweight="bold", pad=20, fontsize=26)
    
    # Set x-axis limits to 0-320
    ax.set_xlim(0, 320)
    
    # Set y-axis with some padding
    y_min = min(df["pass@1"].min(), df["pass@5"].min()) if df["pass@1"].notna().any() else 0
    y_max = max(df["pass@1"].max(), df["pass@5"].max()) if df["pass@5"].notna().any() else 1
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - y_padding), min(1.0, y_max + y_padding))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    
    # Much larger tick labels
    ax.tick_params(axis='both', labelsize=18)
    
    # Legend with custom styling - larger font, compact box
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        borderpad=0.5,
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.3,
        fontsize=18,
    )
    legend.get_frame().set_linewidth(0.5)
    for text in legend.get_texts():
        text.set_color(COLORS["black"])
    
    # No grid - clean minimalist look
    ax.grid(False)


def main() -> None:
    """Main function to create and save the plots."""
    results_dir = Path(__file__).parent
    
    # Load data
    eval_df, train_df = load_and_process_data(results_dir)
    
    # Setup styling
    setup_style()
    
    # Create figure with two subplots - white background
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(15, 6),
        facecolor=COLORS["white"],
    )
    
    # Add some padding - more top space since no main title
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.14, wspace=0.22)
    
    # Create both plots - "Validation Set" instead of "Evaluation Set"
    create_plot(eval_df, "Validation Set", ax1)
    create_plot(train_df, "Training Set", ax2)
    
    # No main title - minimalistic approach
    
    # Save the figure with white background
    output_path = results_dir / "pass_at_k_metrics.png"
    fig.savefig(
        output_path,
        dpi=200,
        facecolor=COLORS["white"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    print(f"✓ Saved plot to: {output_path}")
    
    # Also save individual plots
    for df, name, title in [
        (eval_df, "validation", "Validation Set"),
        (train_df, "training", "Training Set"),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(9, 6), facecolor=COLORS["white"])
        create_plot(df, title, ax_single)
        fig_single.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.14)
        single_path = results_dir / f"pass_at_k_{name}.png"
        fig_single.savefig(
            single_path,
            dpi=200,
            facecolor=COLORS["white"],
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        print(f"✓ Saved plot to: {single_path}")
        plt.close(fig_single)
    
    plt.close(fig)
    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()

