#!/usr/bin/env python3
"""Compute K-step sparsity for newly downloaded experiments.

These experiments have a different directory structure:
- experiments/qwen2.5-7b-grpo-math-lr3e-06/seed42/deltas/
- experiments/qwen2.5-1.5b-sft-math-lr2e-05/seed42/deltas/

Imports shared functions from compute_k_step_sparsity.py (DRY principle).
"""

import csv
import gc
import logging
import re
import sys
from pathlib import Path

# Import shared functions from main script
from compute_k_step_sparsity import (
    ExperimentConfig,
    K_VALUES,
    process_experiment,
    logger,
)

# Specific experiments to process (different structure from main script)
NEW_EXPERIMENTS = [
    "qwen2.5-7b-grpo-math-lr3e-06",
    "qwen2.5-1.5b-sft-math-lr2e-05",
    "qwen2.5-1.5b-sft-math-lr3e-06",
]


def parse_experiment_name(name: str) -> tuple[str, str, bool, int, float] | None:
    """Parse experiment folder name to extract model info."""
    name_lower = name.lower()

    if "qwen" in name_lower:
        family = "Qwen"
    elif "llama" in name_lower:
        family = "Llama"
    elif "gemma" in name_lower:
        family = "Gemma"
    else:
        return None

    size_match = re.search(r"(\d+\.?\d*)b", name_lower)
    if not size_match:
        return None
    size = size_match.group(1) + "B"

    is_sft = "sft" in name_lower

    # Extract iteration number
    iter_match = re.search(r"iter(\d+)", name_lower)
    iteration_num = int(iter_match.group(1)) if iter_match else 1

    # Extract learning rate
    lr_match = re.search(r"lr(\d+\.?\d*)e-?(\d+)", name_lower)
    if lr_match:
        mantissa = float(lr_match.group(1))
        exponent = int(lr_match.group(2))
        learning_rate = mantissa * (10 ** -exponent)
    else:
        learning_rate = 3e-6

    return family, size, is_sft, iteration_num, learning_rate


def detect_delta_indexing(delta_dir: Path) -> tuple[int, int, list[int]]:
    """Detect delta file indexing scheme."""
    delta_files = sorted(delta_dir.glob("delta_*.pt"))
    if not delta_files:
        return 0, 0, []

    indices = []
    for f in delta_files:
        match = re.search(r"delta_(\d+)\.pt", f.name)
        if match:
            indices.append(int(match.group(1)))

    if not indices:
        return 0, 0, []

    indices = sorted(indices)
    start_index = indices[0]
    end_index = indices[-1]

    expected = set(range(start_index, end_index + 1))
    actual = set(indices)
    missing = sorted(expected - actual)

    return start_index, len(indices), missing


def discover_new_experiments(base_dir: Path) -> list[ExperimentConfig]:
    """Discover experiments with seed-based directory structure."""
    experiments = []
    experiments_dir = base_dir / "experiments"

    for exp_name in NEW_EXPERIMENTS:
        exp_dir = experiments_dir / exp_name
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        parsed = parse_experiment_name(exp_name)
        if parsed is None:
            continue

        family, size, is_sft, iteration_num, learning_rate = parsed

        # Look for seed directories
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue

            seed_match = re.search(r"seed(\d+)", seed_dir.name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))

            delta_dir = seed_dir / "deltas"
            if not delta_dir.exists():
                continue

            start_index, num_deltas, missing = detect_delta_indexing(delta_dir)

            if missing:
                logger.warning(f"Skipping {exp_name}/{seed_dir.name}: Missing {len(missing)} files")
                continue

            # Validate delta count
            if is_sft:
                if num_deltas < 99:
                    logger.warning(f"Skipping {exp_name}/{seed_dir.name}: SFT needs >=99, found {num_deltas}")
                    continue
            else:
                if num_deltas != 399:
                    logger.warning(f"Skipping {exp_name}/{seed_dir.name}: GRPO needs 399, found {num_deltas}")
                    continue

            experiments.append(
                ExperimentConfig(
                    model_family=family,
                    model_size=size,
                    seed=seed,
                    delta_dir=delta_dir,
                    num_deltas=num_deltas,
                    start_index=start_index,
                    is_sft=is_sft,
                    iteration_num=iteration_num,
                    learning_rate=learning_rate,
                )
            )
            logger.info(
                f"Found: {family} {size} seed={seed} "
                f"iter={iteration_num} lr={learning_rate:.0e} ({num_deltas} deltas)"
            )

    return experiments


def main():
    data_dir = Path("/root/grail/research/sparsity_analysis")
    output_file = Path("/root/grail/data/sparsity_k_step_new_downloads.csv")

    logger.info(f"Discovering new experiments in {data_dir}")
    experiments = discover_new_experiments(data_dir)

    if not experiments:
        logger.error("No valid experiments found!")
        sys.exit(1)

    logger.info(f"Found {len(experiments)} experiment/seed combinations")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_file}")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_family", "model_size", "seed", "step", "k", "sparsity", "iteration_num", "learning_rate"])

        for config in experiments:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"{config.model_family} {config.model_size} seed={config.seed}")
            logger.info(f"{'=' * 60}")

            try:
                results = process_experiment(config)

                for step, k, sparsity in results:
                    writer.writerow([
                        config.model_family,
                        config.model_size,
                        config.seed,
                        step,
                        k,
                        f"{sparsity:.4f}",
                        config.iteration_num,
                        f"{config.learning_rate:.0e}",
                    ])

                f.flush()
                logger.info(f"  Wrote {len(results)} rows")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback
                traceback.print_exc()

            gc.collect()

    logger.info(f"\nDone! Results: {output_file}")


if __name__ == "__main__":
    main()
