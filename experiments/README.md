# GRAIL Security Experiments

This directory contains comprehensive experiments validating the security of the GRAIL protocol against theoretical concerns about hyperplane non-uniqueness.

## Overview

The experiments demonstrate that while mathematical non-uniqueness exists for single constraints, GRAIL's multi-position verification (k=16) creates an exponentially constrained optimization problem that is computationally infeasible to solve.

## Experiments

1. **Experiment 1: Hyperplane Collision Analysis**
   - Tests collision probability in high-dimensional spaces
   - Validates theoretical bounds on random collisions

2. **Experiment 2: Multi-Position Constraint Analysis**
   - Shows exponential scaling of difficulty with k constraints
   - Demonstrates cascade effects in autoregressive generation

3. **Experiment 3: Model Perturbation Robustness**
   - Tests effects of weight perturbations on sketch preservation
   - Shows difficulty of maintaining both token accuracy and sketch values

4. **Experiment 4: Attack Simulation**
   - Gradient-based attack attempts
   - Computational complexity analysis

5. **Experiment 5: Single Constraint Analysis**
   - Exhaustive search for k=1 collisions
   - Birthday paradox validation
   - Multi-constraint scaling from k=1 to k=16

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster execution)
- 8GB+ GPU memory for model experiments

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd grail/experiments
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install torch torchvision transformers numpy matplotlib tqdm
```

## Running Experiments

### Option 1: Run All Experiments

```bash
python run.py --all
```

This will run all 5 experiments sequentially and generate results in the `results/` directory.

### Option 2: Run Specific Experiments

```bash
# Run a single experiment
python run.py --exp 1

# Run multiple experiments
python run.py --exp 1 3 5

# Run with verbose output
python run.py --exp 2 --verbose
```

### Option 3: Run Individual Experiment Scripts

```bash
# Run experiment 1
python experiment_1_hyperplane_collision.py

# Run experiment 2
python experiment_2_multi_position_constraints.py

# Run experiment 3
python experiment_3_model_perturbation.py

# Run experiment 4
python experiment_4_attack_simulation.py

# Run experiment 5 (multiple versions available)
python experiment_5_single_constraint_analysis.py
python experiment_5_exhaustive_search.py
```

### Generate Summary Report

After running experiments:
```bash
python run.py --summary
```

## Output

All experiments generate:
- JSON results in `results/` directory
- PNG plots in `results/` directory
- Console output with real-time progress

### Result Files

- `experiment_1_hyperplane_collision.json` - Collision probability data
- `experiment_2_multi_position_constraints.json` - Constraint satisfaction rates
- `experiment_3_model_perturbation.json` - Perturbation effects
- `experiment_4_attack_simulation.json` - Attack success rates
- `experiment_5_*.json` - Various single constraint analyses
- `experiment_summary.json` - Overall summary (if using run.py)

### Plots

Each experiment generates visualizations:
- Collision probability curves
- Constraint scaling graphs
- Perturbation analysis charts
- Attack success summaries
- Multi-constraint scaling plots

## GPU Acceleration

The experiments automatically use GPU if available. To force CPU usage:
```bash
CUDA_VISIBLE_DEVICES="" python run.py --all
```

## Experiment Duration

Approximate run times on NVIDIA A100:
- Experiment 1: ~30 seconds
- Experiment 2: ~2 minutes
- Experiment 3: ~1 minute
- Experiment 4: ~1 minute
- Experiment 5: ~2-5 minutes (depending on variant)

Total time for all experiments: ~10 minutes

## Troubleshooting

### CUDA Out of Memory
If you encounter GPU memory issues:
1. Reduce batch sizes in the experiment scripts
2. Use a smaller model (edit model_name in scripts)
3. Run on CPU instead

### Missing Dependencies
If imports fail:
```bash
pip install --upgrade torch transformers numpy matplotlib
```

### Permission Errors
Ensure write permissions for the `results/` directory:
```bash
chmod -R 755 results/
```

## Key Findings

The experiments conclusively demonstrate:

1. **Exponential Scaling**: Success rate drops from 1.40% (k=1) to 0.00% (k≥4)
2. **Attack Infeasibility**: All gradient-based attacks achieve 0% success
3. **Computational Complexity**: For k=16, success probability ≈ 10^-135.8
4. **Perturbation Sensitivity**: Even 0.1 scale perturbations destroy sketch preservation

These results validate GRAIL's security against the theoretical hyperplane non-uniqueness concern.

## Citation

If you use these experiments in your research, please cite the GRAIL technical report.

## License

[License information]