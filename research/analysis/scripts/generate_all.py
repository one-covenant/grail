#!/usr/bin/env python3
"""
Generate all research figures.

Usage:
    python -m scripts.generate_all          # From analysis/ directory
    python scripts/generate_all.py          # Direct execution

This will regenerate all figures in ../figures/
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from scripts.learning_curves import generate_all as generate_learning_curves
from scripts.benchmark_bars import generate_all as generate_benchmark_bars
from scripts.comparisons import generate_all as generate_comparisons


def main() -> None:
    """Generate all figures."""
    print("=" * 60)
    print("Generating all research figures...")
    print("=" * 60)
    
    print("\n📈 Learning Curves (Fig 1)")
    print("-" * 40)
    generate_learning_curves()
    
    print("\n📊 Benchmark Bar Plots (Fig 2-5)")
    print("-" * 40)
    generate_benchmark_bars()
    
    print("\n🔬 Comparison Plots (Fig 6-7)")
    print("-" * 40)
    generate_comparisons()
    
    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print("=" * 60)
    
    # List generated figures
    figures_dir = SCRIPT_DIR.parent.parent / "figures"
    print(f"\nOutput directory: {figures_dir}")
    print("\nGenerated files:")
    for f in sorted(figures_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

