# Research Analysis

This directory contains scripts and data for generating research figures.

## Directory Structure

```
analysis/
├── data/                    # Raw data (WandB exports)
│   ├── grail_training/      # GRAIL training metrics
│   └── trl_training/        # TRL baseline metrics
├── scripts/                 # Figure generation scripts
│   ├── __init__.py
│   ├── style.py             # Shared style configuration
│   ├── learning_curves.py   # Fig 1: Training/validation curves
│   ├── benchmark_bars.py    # Fig 2-5: Benchmark comparisons
│   ├── comparisons.py       # Fig 6-7: GRAIL vs TRL, bucketing
│   └── generate_all.py      # Regenerate all figures
└── Makefile                 # Convenience commands
```

## Usage

### Generate All Figures

```bash
cd research/analysis
make figures
```

Or using Python directly:

```bash
cd research/analysis
python -m scripts.generate_all
```

### Generate Specific Figures

```bash
make learning      # Fig 1: Learning curves
make benchmarks    # Fig 2-5: Benchmark bars
make comparisons   # Fig 6-7: Comparisons
```

### Clean Figures

```bash
make clean
```

## Output

All figures are saved to `research/figures/` with paper-ready naming:

| File | Description |
|------|-------------|
| `fig1_learning_curves.png` | Combined validation + training curves |
| `fig1a_validation_curves.png` | Validation set only |
| `fig1b_training_curves.png` | Training set only |
| `fig2_gsm8k_benchmark.png` | GSM8K test set performance |
| `fig3_math_benchmark.png` | MATH test set performance |
| `fig4_amc2023_benchmark.png` | AMC 2023 performance |
| `fig5_combined_benchmarks_*.png` | Combined benchmark comparison (3 variants) |
| `fig6_grail_vs_trl.png` | GRAIL vs TRL comparison |
| `fig7_bucketing_comparison.png` | Logarithmic vs linear bucketing analysis |

## Style

All figures use a consistent style defined in `scripts/style.py`:

- **Color palette**: White (#FFFFFF), Red (#FF3A3A), Black (#101010)
- **Font**: SF Pro Display / Helvetica Neue / Arial
- **No grid**: Clean, minimalist aesthetic
- **Large fonts**: Optimized for readability in papers/blogs

## Dependencies

```bash
pip install matplotlib numpy pandas
```

Or use the project's existing environment.

