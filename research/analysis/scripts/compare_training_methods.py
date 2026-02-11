#!/usr/bin/env python3
"""
Compare full training vs delta training methods.

Creates a line plot with:
- X-axis: Optimizer step (0 to 200)
- Y-axis: pass@1
- Two lines: full training and delta training
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from .style import COLORS, setup_style

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_LENGTH = 30  # blocks per window
EVAL_WINDOWS = [0, 20, 40, 60, 80, 100]  # Evaluation windows (6 total)
OPTIMIZER_STEPS = [0, 40, 80, 120, 160, 200]  # Corresponding optimizer steps
TOTAL_WINDOWS = 100  # Only use 100 windows of data

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
FULL_TRAINING_DIR = SCRIPT_DIR.parent / "data" / "full_training"
DELTA_TRAINING_DIR = SCRIPT_DIR.parent / "data" / "delta_training"
FIGURES_DIR = SCRIPT_DIR.parent.parent / "figures"


def load_data(file_path: Path) -> List[Tuple[int, float]]:
    """Load CSV data and extract block_number and pass@1."""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_number = int(row['block_number'])
            pass1_str = row.get('training_grail - eval/pass@1', '')
            try:
                pass1 = float(pass1_str) if pass1_str else None
                if pass1 is not None:
                    data.append((block_number, pass1))
            except ValueError:
                continue
    return data


def process_data(data: List[Tuple[int, float]], dataset_name: str) -> List[Tuple[int, float]]:
    """
    Process data to convert block numbers to optimizer steps.
    
    Only keeps data from the first 100 windows (6 evaluation points).
    Evaluations happen at windows 0, 20, 40, 60, 80, 100.
    These map to optimizer steps 0, 40, 80, 120, 160, 200.
    """
    if not data:
        return []
    
    # Find the starting block (first evaluation at window 0)
    starting_block = data[0][0]
    
    # Calculate window numbers for each block
    window_data = []
    for block_number, pass1 in data:
        window_number = (block_number - starting_block) / WINDOW_LENGTH
        window_data.append((window_number, block_number, pass1))
    
    # Filter to only evaluation windows (0, 20, 40, 60, 80, 100)
    # Allow tolerance of 0.5 windows for rounding
    eval_data = []
    for eval_window, opt_step in zip(EVAL_WINDOWS, OPTIMIZER_STEPS):
        # Find the entry closest to this evaluation window
        closest_entry = None
        min_diff = float('inf')
        
        for window_num, block_num, pass1_val in window_data:
            window_diff = abs(window_num - eval_window)
            if window_diff < min_diff:
                min_diff = window_diff
                closest_entry = (window_num, block_num, pass1_val)
        
        # Check if it's close enough (within 0.5 windows)
        if closest_entry and min_diff < 0.5:
            eval_data.append((opt_step, closest_entry[2]))
    
    # Sort by optimizer step
    eval_data.sort(key=lambda x: x[0])
    
    return eval_data


def create_comparison_plot(full_data: List[Tuple[int, float]], delta_data: List[Tuple[int, float]]) -> None:
    """Create a line plot comparing full training vs delta training."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["white"])
    
    # Extract x and y values
    full_x = [x for x, _ in full_data]
    full_y = [y for _, y in full_data]
    delta_x = [x for x, _ in delta_data]
    delta_y = [y for _, y in delta_data]
    
    # Plot full training
    ax.plot(
        full_x,
        full_y,
        color=COLORS["black"],
        linewidth=2.5,
        label="Full Training",
        marker="o",
        markersize=8,
        alpha=0.9,
        zorder=2,
    )
    
    # Plot delta training
    ax.plot(
        delta_x,
        delta_y,
        color=COLORS["red"],
        linewidth=2.5,
        label="Delta Training",
        marker="s",
        markersize=8,
        alpha=0.9,
        zorder=2,
    )
    
    # Styling
    ax.set_xlabel("Optimizer Step", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_ylabel("pass@1", fontweight="medium", labelpad=14, fontsize=20)
    ax.set_title("Full Training vs Delta Training", fontweight="bold", pad=20, fontsize=26)
    
    ax.set_xlim(0, 200)
    
    # Set y-axis with some padding
    all_pass1 = full_y + delta_y
    if all_pass1:
        y_min = min(all_pass1)
        y_max = max(all_pass1)
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        ax.set_ylim(max(0, y_min - y_padding), min(1.0, y_max + y_padding))
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.tick_params(axis='both', labelsize=18)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    legend = ax.legend(
        loc="best",
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
    
    fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.14)
    
    output_path = FIGURES_DIR / "training_comparison.png"
    fig.savefig(output_path, dpi=200, facecolor=COLORS["white"], bbox_inches="tight", pad_inches=0.2)
    print(f"✓ {output_path}")
    plt.close(fig)


def generate_plot() -> None:
    """Generate the comparison plot."""
    setup_style()
    
    # Find CSV files
    full_csv_files = list(FULL_TRAINING_DIR.glob("*.csv"))
    delta_csv_files = list(DELTA_TRAINING_DIR.glob("*.csv"))
    
    if not full_csv_files:
        raise FileNotFoundError(f"No CSV files found in {FULL_TRAINING_DIR}")
    if not delta_csv_files:
        raise FileNotFoundError(f"No CSV files found in {DELTA_TRAINING_DIR}")
    
    # Load data (use first CSV file found)
    full_data = load_data(full_csv_files[0])
    delta_data = load_data(delta_csv_files[0])
    
    # Process data
    full_processed = process_data(full_data, "full")
    delta_processed = process_data(delta_data, "delta")
    
    print(f"Full training: {len(full_processed)} evaluation points")
    print(f"Delta training: {len(delta_processed)} evaluation points")
    
    # Create plot
    create_comparison_plot(full_processed, delta_processed)


if __name__ == "__main__":
    generate_plot()
