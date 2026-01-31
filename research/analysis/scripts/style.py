"""
Shared style configuration for all research figures.

Color palette:
- White: #FFFFFF
- Red: #FF3A3A
- Black: #101010
"""

import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "white": "#FFFFFF",
    "red": "#FF3A3A",
    "black": "#101010",
    "dark_gray": "#404040",
    "mid_gray": "#2A2A2A",
    "light_gray": "#888888",
}

# Line colors
PASS_1_COLOR = COLORS["black"]
PASS_5_COLOR = COLORS["red"]


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
        
        # Grid
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

