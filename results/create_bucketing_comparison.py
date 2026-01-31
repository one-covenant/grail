#!/usr/bin/env python3
"""
Visualization comparing logarithmic vs linear bucketing for GRAIL proof system.

Shows why logarithmic bucketing is superior for floating-point arithmetic
and numerical stability in hidden state verification.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import math

# Set up the style
plt.style.use('dark_background')

# Custom color palette
COLORS = {
    'bg': '#0d1117',
    'grid': '#21262d',
    'text': '#c9d1d9',
    'accent1': '#58a6ff',  # Blue
    'accent2': '#f78166',  # Orange/red
    'accent3': '#7ee787',  # Green
    'accent4': '#d2a8ff',  # Purple
    'warning': '#d29922',  # Yellow
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


def create_figure():
    fig = plt.figure(figsize=(16, 14), facecolor=COLORS['bg'])
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], 
                          hspace=0.35, wspace=0.25,
                          left=0.08, right=0.95, top=0.92, bottom=0.06)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Bucket width comparison
    ax2 = fig.add_subplot(gs[0, 1])  # Value-to-bucket mapping
    ax3 = fig.add_subplot(gs[1, 0])  # FP error sensitivity - log
    ax4 = fig.add_subplot(gs[1, 1])  # FP error sensitivity - linear
    ax5 = fig.add_subplot(gs[2, :])  # Bucket stability analysis
    
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_facecolor(COLORS['bg'])
        ax.tick_params(colors=COLORS['text'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
    
    return fig, (ax1, ax2, ax3, ax4, ax5)


def plot_bucket_widths(ax):
    """Plot 1: Compare bucket widths for log vs linear bucketing."""
    
    # Calculate bucket boundaries for logarithmic
    log_boundaries = []
    for b in range(B + 1):
        # Inverse of: bucket = floor(1.6 * log2(1 + |h|))
        # b = 1.6 * log2(1 + h) => h = 2^(b/1.6) - 1
        h = 2 ** (b / 1.6) - 1
        log_boundaries.append(h)
    
    log_widths = [log_boundaries[i+1] - log_boundaries[i] for i in range(B)]
    
    # Linear boundaries (assuming max value = 100)
    max_val = 100.0
    linear_widths = [max_val / B] * B
    
    x = np.arange(B)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, log_widths, width, label='Logarithmic', 
                   color=COLORS['accent1'], alpha=0.8, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, linear_widths, width, label='Linear', 
                   color=COLORS['accent2'], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Bucket Index', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Bucket Width (|h| range)', color=COLORS['text'], fontsize=11)
    ax.set_title('Bucket Width Distribution', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_yscale('log')
    ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    
    # Add annotation
    ax.annotate('Log: Fine granularity\nfor small values', 
                xy=(2, log_widths[2]), xytext=(4, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent3']),
                fontsize=9, color=COLORS['accent3'])
    ax.annotate('Log: Coarse for\nlarge values', 
                xy=(14, log_widths[14]), xytext=(11, 150),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent3']),
                fontsize=9, color=COLORS['accent3'])


def plot_value_mapping(ax):
    """Plot 2: Show how values map to buckets."""
    
    values = np.logspace(-2, 3, 500)  # 0.01 to 1000
    
    log_buckets = [log_magnitude_bucket(v) for v in values]
    linear_buckets = [linear_bucket(v, max_val=100) for v in values]
    
    ax.semilogx(values, log_buckets, color=COLORS['accent1'], linewidth=2, 
                label='Logarithmic', alpha=0.9)
    ax.semilogx(values, linear_buckets, color=COLORS['accent2'], linewidth=2, 
                label='Linear (max=100)', alpha=0.9, linestyle='--')
    
    ax.set_xlabel('Hidden State Value |h|', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Bucket Index', color=COLORS['text'], fontsize=11)
    ax.set_title('Value → Bucket Mapping', color=COLORS['text'], fontsize=13, fontweight='bold')
    ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.set_yticks(range(0, B+1, 2))
    
    # Highlight regions
    ax.axvspan(0.01, 1, alpha=0.15, color=COLORS['accent3'], label='Typical FP precision zone')
    ax.axvspan(1, 10, alpha=0.1, color=COLORS['accent4'])
    
    ax.text(0.15, 12, 'High FP\nprecision', fontsize=8, color=COLORS['accent3'], ha='center')
    ax.text(3, 12, 'Medium\nprecision', fontsize=8, color=COLORS['accent4'], ha='center')


def plot_fp_sensitivity_log(ax):
    """Plot 3: Floating-point error sensitivity for log bucketing."""
    
    # Test values across different magnitudes
    test_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    
    # Relative FP errors (typically ~1e-7 for float32, proportional to magnitude)
    relative_error = 1e-5  # Exaggerated for visibility
    
    bucket_changes = []
    for v in test_values:
        original_bucket = log_magnitude_bucket(v)
        # Apply relative error (FP error is proportional to magnitude)
        perturbed_values = [v * (1 + relative_error), v * (1 - relative_error),
                           v + v * relative_error * 10, v - v * relative_error * 10]
        changed = sum(1 for pv in perturbed_values if log_magnitude_bucket(pv) != original_bucket)
        bucket_changes.append(changed / len(perturbed_values) * 100)
    
    bars = ax.bar(range(len(test_values)), bucket_changes, color=COLORS['accent1'], 
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Test Value |h|', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Bucket Change Rate (%)', color=COLORS['text'], fontsize=11)
    ax.set_title('Log Bucketing: FP Error Sensitivity', color=COLORS['text'], 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(test_values)))
    ax.set_xticklabels([f'{v:.1f}' for v in test_values], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    ax.set_ylim(0, 100)
    
    # Add stability indicator
    avg_change = np.mean(bucket_changes)
    ax.axhline(y=avg_change, color=COLORS['accent3'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(test_values)-1, avg_change + 5, f'Avg: {avg_change:.1f}%', 
            color=COLORS['accent3'], fontsize=10, ha='right')


def plot_fp_sensitivity_linear(ax):
    """Plot 4: Floating-point error sensitivity for linear bucketing."""
    
    test_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    relative_error = 1e-5
    
    bucket_changes = []
    for v in test_values:
        original_bucket = linear_bucket(v)
        perturbed_values = [v * (1 + relative_error), v * (1 - relative_error),
                           v + v * relative_error * 10, v - v * relative_error * 10]
        changed = sum(1 for pv in perturbed_values if linear_bucket(pv) != original_bucket)
        bucket_changes.append(changed / len(perturbed_values) * 100)
    
    bars = ax.bar(range(len(test_values)), bucket_changes, color=COLORS['accent2'], 
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Test Value |h|', color=COLORS['text'], fontsize=11)
    ax.set_ylabel('Bucket Change Rate (%)', color=COLORS['text'], fontsize=11)
    ax.set_title('Linear Bucketing: FP Error Sensitivity', color=COLORS['text'], 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(test_values)))
    ax.set_xticklabels([f'{v:.1f}' for v in test_values], fontsize=9)
    ax.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    ax.set_ylim(0, 100)
    
    avg_change = np.mean(bucket_changes)
    ax.axhline(y=avg_change, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(test_values)-1, avg_change + 5, f'Avg: {avg_change:.1f}%', 
            color=COLORS['warning'], fontsize=10, ha='right')


def plot_stability_analysis(ax):
    """Plot 5: Comprehensive stability analysis showing bucket boundaries vs FP precision."""
    
    # Create a range of values
    values = np.logspace(-1, 2.5, 1000)  # 0.1 to ~300
    
    # Calculate floating-point relative precision (machine epsilon * magnitude)
    # For float32, epsilon ≈ 1.19e-7
    fp_epsilon = 1.19e-7
    fp_absolute_error = values * fp_epsilon * 1000  # Scale up for visibility
    
    # Log bucket boundaries
    log_boundaries = []
    for b in range(B + 1):
        h = 2 ** (b / 1.6) - 1
        if h <= 350:
            log_boundaries.append(h)
    
    # Linear bucket boundaries
    linear_boundaries = np.linspace(0, 100, B + 1)
    
    # Plot FP error magnitude
    ax.fill_between(values, 0, fp_absolute_error, alpha=0.3, color=COLORS['warning'],
                    label='Floating-point error zone (scaled)')
    
    # Plot log bucket boundaries
    for i, b in enumerate(log_boundaries[1:-1]):
        ax.axvline(x=b, color=COLORS['accent1'], linestyle='-', linewidth=1.5, alpha=0.7)
    ax.axvline(x=log_boundaries[1], color=COLORS['accent1'], linestyle='-', 
               linewidth=1.5, alpha=0.7, label='Log bucket boundaries')
    
    # Plot linear bucket boundaries
    for i, b in enumerate(linear_boundaries[1:-1]):
        ax.axvline(x=b, color=COLORS['accent2'], linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=linear_boundaries[1], color=COLORS['accent2'], linestyle='--', 
               linewidth=1, alpha=0.5, label='Linear bucket boundaries')
    
    ax.set_xlabel('Hidden State Value |h|', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Magnitude', color=COLORS['text'], fontsize=12)
    ax.set_title('Bucket Boundaries vs Floating-Point Error Growth', 
                 color=COLORS['text'], fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(facecolor=COLORS['bg'], edgecolor=COLORS['grid'], labelcolor=COLORS['text'],
              loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.set_xlim(0.1, 350)
    ax.set_ylim(1e-8, 1)
    
    # Add explanatory annotations
    ax.annotate('Log buckets are WIDER here\n→ tolerates larger FP errors', 
                xy=(100, 1e-5), xytext=(150, 1e-3),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent3'], lw=1.5),
                fontsize=10, color=COLORS['accent3'], ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], 
                         edgecolor=COLORS['accent3'], alpha=0.8))
    
    ax.annotate('Log buckets are NARROWER here\n→ FP errors are small anyway', 
                xy=(1, 1e-7), xytext=(0.3, 1e-4),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent3'], lw=1.5),
                fontsize=10, color=COLORS['accent3'], ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], 
                         edgecolor=COLORS['accent3'], alpha=0.8))
    
    ax.annotate('Linear buckets: UNIFORM width\n→ small values cross boundaries easily', 
                xy=(6.25, 0.01), xytext=(20, 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent2'], lw=1.5),
                fontsize=10, color=COLORS['accent2'], ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], 
                         edgecolor=COLORS['accent2'], alpha=0.8))


def add_summary_box(fig):
    """Add a summary text box."""
    summary_text = (
        "Why Logarithmic Bucketing?\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• FP errors scale with magnitude: ε·|h|\n"
        "• Log buckets scale similarly: width ∝ |h|\n"
        "• Result: Constant relative error tolerance\n"
        "• Linear buckets: Fixed width → unstable for small values"
    )
    
    fig.text(0.5, 0.97, summary_text, transform=fig.transFigure,
             fontsize=11, color=COLORS['text'], ha='center', va='top',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['bg'], 
                      edgecolor=COLORS['accent1'], alpha=0.9, linewidth=2))


def main():
    fig, (ax1, ax2, ax3, ax4, ax5) = create_figure()
    
    plot_bucket_widths(ax1)
    plot_value_mapping(ax2)
    plot_fp_sensitivity_log(ax3)
    plot_fp_sensitivity_linear(ax4)
    plot_stability_analysis(ax5)
    
    # Main title
    fig.suptitle('GRAIL Proof: Logarithmic vs Linear Bucketing (B=16)', 
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.99)
    
    plt.savefig('results/bucketing_comparison.png', dpi=150, 
                facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    print("Saved: results/bucketing_comparison.png")
    
    plt.show()


if __name__ == '__main__':
    main()

