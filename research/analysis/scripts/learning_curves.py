#!/usr/bin/env python3
"""
Generate learning curve plots (pass@1 and pass@5 metrics).

Outputs:
- fig1_learning_curves.png (combined validation + training)
- fig1a_validation_curves.png
- fig1b_training_curves.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .style import COLORS, PASS_1_COLOR, PASS_5_COLOR, setup_style

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "grail_training"
FIGURES_DIR = SCRIPT_DIR.parent.parent / "figures"


def load_and_process_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and merge into eval and training dataframes."""
    
    # Load evaluation data
    eval_pass1_df = pd.read_csv(DATA_DIR / "wandb_export_2025-11-30T16_44_37.800-05_00.csv")
    eval_pass5_df = pd.read_csv(DATA_DIR / "wandb_export_2025-11-30T17_01_56.500-05_00.csv")
    
    # Load training data
    train_pass5_df = pd.read_csv(DATA_DIR / "wandb_export_2025-11-30T17_02_29.789-05_00.csv")
    train_pass1_df = pd.read_csv(DATA_DIR / "wandb_export_2025-11-30T17_02_47.878-05_00.csv")
    
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
        df["update_step"] = ((df["block_number"] - block_min) / (block_max - block_min)) * 320
        return df
    
    eval_df = map_to_update_step(eval_df)
    train_df = map_to_update_step(train_df)
    
    return eval_df, train_df


def create_plot(df: pd.DataFrame, title: str, ax: plt.Axes) -> None:
    """Create a single plot with pass@1 and pass@5 lines."""
    
    # Plot pass@1
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
    
    # Styling
    ax.set_xlabel("Update Step", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title(title, fontweight="bold", pad=20, fontsize=26)
    
    ax.set_xlim(0, 320)
    
    # Set y-axis with some padding
    y_min = min(df["pass@1"].min(), df["pass@5"].min()) if df["pass@1"].notna().any() else 0
    y_max = max(df["pass@1"].max(), df["pass@5"].max()) if df["pass@5"].notna().any() else 1
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - y_padding), min(1.0, y_max + y_padding))
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis='both', labelsize=18)
    
    # Legend
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
    
    ax.grid(False)


def generate_all() -> None:
    """Generate all learning curve figures."""
    setup_style()
    eval_df, train_df = load_and_process_data()
    
    # Combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=COLORS["white"])
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.14, wspace=0.22)
    
    create_plot(eval_df, "Validation Set", ax1)
    create_plot(train_df, "Training Set", ax2)
    
    output_path = FIGURES_DIR / "fig1_learning_curves.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.3)
    print(f"✓ {output_path}")
    plt.close(fig)
    
    # Individual figures
    for df, name, title in [
        (eval_df, "fig1a_validation_curves.png", "Validation Set"),
        (train_df, "fig1b_training_curves.png", "Training Set"),
    ]:
        fig_single, ax_single = plt.subplots(figsize=(9, 6), facecolor=COLORS["white"])
        create_plot(df, title, ax_single)
        fig_single.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.14)
        single_path = FIGURES_DIR / name
        fig_single.savefig(single_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.2)
        print(f"✓ {single_path}")
        plt.close(fig_single)


if __name__ == "__main__":
    generate_all()

