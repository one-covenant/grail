#!/usr/bin/env python3
"""
Create comparison plot of grail vs TRL training curves.

Shows that decentralized rollouts in grail produce learning dynamics
comparable to a centralized RL pipeline (TRL).

Color palette:
- White background: #FFFFFF
- grail: #101010 (black)
- TRL: #FF3A3A (red)
- Solid lines for pass@5, dashed lines for pass@1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white": "#FFFFFF",
    "red": "#FF3A3A",
    "black": "#101010",
    "light_gray": "#888888",
}

GRAIL_COLOR = COLORS["black"]
TRL_COLOR = COLORS["red"]

# Line styles: solid for pass@5, dashed for pass@1
PASS5_STYLE = "-"
PASS1_STYLE = "--"


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Processing
# ─────────────────────────────────────────────────────────────────────────────
def load_trl_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TRL training data (already in update step format)."""
    trl_dir = results_dir / "trl_training"
    
    # Load eval data
    eval_pass1_df = pd.read_csv(trl_dir / "wandb_export_2025-12-01T22_57_12.739-05_00.csv")
    eval_pass5_df = pd.read_csv(trl_dir / "wandb_export_2025-12-01T22_56_11.017-05_00.csv")
    
    # Load train data  
    train_pass1_df = pd.read_csv(trl_dir / "wandb_export_2025-12-01T22_57_52.582-05_00.csv")
    train_pass5_df = pd.read_csv(trl_dir / "wandb_export_2025-12-01T22_57_34.447-05_00.csv")
    
    # Process eval data
    eval_df = pd.DataFrame({
        "update_step": eval_pass1_df["eval_step"].astype(float),
        "pass@1": eval_pass1_df["trl_math_grpo_qwen15b_grail_matched_final - eval/pass@1"].astype(float),
    })
    eval_pass5 = eval_pass5_df["trl_math_grpo_qwen15b_grail_matched_final - eval/pass@5"].astype(float)
    eval_df["pass@5"] = eval_pass5.values
    
    # Process train data
    train_df = pd.DataFrame({
        "update_step": train_pass1_df["train/global_step"].astype(float),
        "pass@1": train_pass1_df["trl_math_grpo_qwen15b_grail_matched_final - train/pass@1"].astype(float),
    })
    train_pass5 = train_pass5_df["trl_math_grpo_qwen15b_grail_matched_final - train/pass@5"].astype(float)
    train_df["pass@5"] = train_pass5.values
    
    return eval_df, train_df


def load_grail_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load grail training data and map block_number to update steps."""
    grail_dir = results_dir / "grail_training"
    
    # Load eval data
    eval_pass1_df = pd.read_csv(grail_dir / "wandb_export_2025-11-30T16_44_37.800-05_00.csv")
    eval_pass5_df = pd.read_csv(grail_dir / "wandb_export_2025-11-30T17_01_56.500-05_00.csv")
    
    # Load training data
    train_pass1_df = pd.read_csv(grail_dir / "wandb_export_2025-11-30T17_02_47.878-05_00.csv")
    train_pass5_df = pd.read_csv(grail_dir / "wandb_export_2025-11-30T17_02_29.789-05_00.csv")
    
    # Map block_number to update_step (0-320)
    # Use the internal step column for better alignment
    def map_step_to_update(df: pd.DataFrame, step_col: str) -> pd.DataFrame:
        """Map internal step to update step 0-320."""
        # The step goes from ~0 to ~20000, we map to 0-320
        # Based on TRL: 320 eval steps correspond to ~125000 internal steps
        # So ~391 internal steps per update step
        # For grail, step goes from ~0 to ~20000 for 320 updates
        # So ~62.5 internal steps per update step
        step_max = df[step_col].max()
        df["update_step"] = (df[step_col] / step_max) * 320
        return df
    
    # Process eval data - use _step column for mapping
    eval_pass1_df = eval_pass1_df.rename(columns={
        "trainer_grail - eval/pass@1": "pass@1"
    })
    eval_pass5_df = eval_pass5_df.rename(columns={
        "trainer_grail - eval/pass@5": "pass@5"
    })
    
    # Merge eval data on block_number
    eval_df = eval_pass1_df[["block_number", "trainer_grail - _step", "pass@1"]].merge(
        eval_pass5_df[["block_number", "pass@5"]], 
        on="block_number", 
        how="outer"
    ).sort_values("block_number")
    
    # Map step to update_step
    eval_df["update_step"] = (eval_df["trainer_grail - _step"] / eval_df["trainer_grail - _step"].max()) * 320
    
    # Process training data
    train_pass1_df = train_pass1_df.rename(columns={
        "trainer_grail - training/prefilter/pass@1": "pass@1"
    })
    train_pass5_df = train_pass5_df.rename(columns={
        "trainer_grail - training/prefilter/pass@5": "pass@5"
    })
    
    # Merge training data on block_number
    train_df = train_pass1_df[["block_number", "trainer_grail - _step", "pass@1"]].merge(
        train_pass5_df[["block_number", "pass@5"]], 
        on="block_number", 
        how="outer"
    ).sort_values("block_number")
    
    # Map step to update_step
    train_df["update_step"] = (train_df["trainer_grail - _step"] / train_df["trainer_grail - _step"].max()) * 320
    
    return eval_df, train_df


def smooth_data(y: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing."""
    return uniform_filter1d(y, size=window_size, mode='nearest')


def setup_style() -> None:
    """Configure matplotlib for elegant, modern styling."""
    plt.rcParams.update({
        # Font settings
        "font.family": "sans-serif",
        "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "font.weight": "normal",
        
        # Figure
        "figure.facecolor": COLORS["white"],
        "figure.edgecolor": COLORS["white"],
        "figure.dpi": 150,
        
        # Axes
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
        
        # Ticks
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        
        # No grid
        "axes.grid": False,
        
        # Legend
        "legend.facecolor": COLORS["white"],
        "legend.edgecolor": COLORS["light_gray"],
        "legend.fontsize": 18,
        "legend.framealpha": 0.95,
        
        # Lines
        "lines.linewidth": 2.5,
        "lines.antialiased": True,
    })


def create_comparison_plot(
    grail_df: pd.DataFrame,
    trl_df: pd.DataFrame,
    title: str,
    ax: plt.Axes,
    smooth: bool = True,
    smooth_window: int = 10,
) -> None:
    """Create a comparison plot with grail and TRL lines for both pass@1 and pass@5."""
    
    # Sort data by update_step
    grail_df = grail_df.sort_values("update_step").copy()
    trl_df = trl_df.sort_values("update_step").copy()
    
    # Get x values
    grail_x = grail_df["update_step"].values
    trl_x = trl_df["update_step"].values
    
    # Get pass@1 and pass@5 data
    grail_pass1 = grail_df["pass@1"].values
    grail_pass5 = grail_df["pass@5"].values
    trl_pass1 = trl_df["pass@1"].values
    trl_pass5 = trl_df["pass@5"].values
    
    # Apply smoothing to training data (which is noisy)
    if smooth and len(grail_pass1) > smooth_window:
        grail_pass1_smooth = smooth_data(grail_pass1, smooth_window)
        grail_pass5_smooth = smooth_data(grail_pass5, smooth_window)
    else:
        grail_pass1_smooth = grail_pass1
        grail_pass5_smooth = grail_pass5
        
    if smooth and len(trl_pass1) > smooth_window:
        trl_pass1_smooth = smooth_data(trl_pass1, smooth_window)
        trl_pass5_smooth = smooth_data(trl_pass5, smooth_window)
    else:
        trl_pass1_smooth = trl_pass1
        trl_pass5_smooth = trl_pass5
    
    # Plot grail-v0 pass@5 (solid line)
    ax.plot(
        grail_x,
        grail_pass5_smooth,
        color=GRAIL_COLOR,
        linewidth=2.5,
        linestyle=PASS5_STYLE,
        label="grail-v0 pass@5",
        alpha=0.9,
        zorder=4,
    )
    
    # Plot grail-v0 pass@1 (dashed line)
    ax.plot(
        grail_x,
        grail_pass1_smooth,
        color=GRAIL_COLOR,
        linewidth=2.2,
        linestyle=PASS1_STYLE,
        label="grail-v0 pass@1",
        alpha=0.85,
        zorder=3,
    )
    
    # Plot TRL pass@5 (solid line)
    ax.plot(
        trl_x,
        trl_pass5_smooth,
        color=TRL_COLOR,
        linewidth=2.5,
        linestyle=PASS5_STYLE,
        label="TRL pass@5",
        alpha=0.9,
        zorder=2,
    )
    
    # Plot TRL pass@1 (dashed line)
    ax.plot(
        trl_x,
        trl_pass1_smooth,
        color=TRL_COLOR,
        linewidth=2.2,
        linestyle=PASS1_STYLE,
        label="TRL pass@1",
        alpha=0.85,
        zorder=1,
    )
    
    # Styling
    ax.set_xlabel("Update Step", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title(title, fontweight="bold", pad=20, fontsize=26)
    
    # Set x-axis limits to 0-320
    ax.set_xlim(0, 320)
    
    # Set y-axis with consistent range across all lines
    all_y = np.concatenate([grail_pass1_smooth, grail_pass5_smooth, 
                           trl_pass1_smooth, trl_pass5_smooth])
    y_min = all_y.min()
    y_max = all_y.max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - y_padding), min(1.0, y_max + y_padding))
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    
    # Tick labels
    ax.tick_params(axis='both', labelsize=18)
    
    # Legend with custom styling - arranged in 2 columns
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        borderpad=0.5,
        handlelength=2.0,
        handletextpad=0.5,
        labelspacing=0.4,
        columnspacing=1.0,
        fontsize=14,
        ncol=2,
    )
    legend.get_frame().set_linewidth(0.5)
    for text in legend.get_texts():
        text.set_color(COLORS["black"])
    
    ax.grid(False)


def main() -> None:
    """Main function to create and save the comparison plot."""
    results_dir = Path(__file__).parent
    
    # Load data
    print("Loading TRL data...")
    trl_eval_df, trl_train_df = load_trl_data(results_dir)
    
    print("Loading grail data...")
    grail_eval_df, grail_train_df = load_grail_data(results_dir)
    
    print(f"TRL eval points: {len(trl_eval_df)}, train points: {len(trl_train_df)}")
    print(f"grail eval points: {len(grail_eval_df)}, train points: {len(grail_train_df)}")
    
    # Setup styling
    setup_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(15, 6),
        facecolor=COLORS["white"],
    )
    
    # Add padding
    fig.subplots_adjust(left=0.07, right=0.96, top=0.90, bottom=0.14, wspace=0.22)
    
    # Create both plots
    # Eval data has fewer points so less smoothing needed
    create_comparison_plot(grail_eval_df, trl_eval_df, "Validation Set", ax1, smooth=False)
    
    # Training data is noisy, apply smoothing
    create_comparison_plot(grail_train_df, trl_train_df, "Training Set", ax2, smooth=True, smooth_window=15)
    
    # Save the figure
    output_path = results_dir / "grail_vs_trl_comparison.png"
    fig.savefig(
        output_path,
        dpi=200,
        facecolor=COLORS["white"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    print(f"\n✓ Saved comparison plot to: {output_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
