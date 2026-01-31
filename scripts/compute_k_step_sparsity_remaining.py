#!/usr/bin/env python3
"""Compute K-step sparsity for remaining unfinished experiments.

Imports shared functions from compute_k_step_sparsity.py (DRY principle).
"""

import csv
import gc
import re
import sys
from pathlib import Path

from compute_k_step_sparsity import (
    ExperimentConfig,
    K_VALUES,
    process_experiment,
    logger,
    detect_delta_indexing,
    parse_iteration_num,
    parse_learning_rate,
)

# Explicitly define unfinished experiments
UNFINISHED_EXPERIMENTS = [
    # Format: (exp_dir_name, delta_subdir_pattern, seeds, is_sft)
    # Original batch - unfinished
    ("llama3.2-3b-iter1", "checkpoints/deltas_math_instance{}_seed{}", [(3, 9999)], False),
    ("qwen2.5-1.5b-lr5e-6", "checkpoints/deltas_math_instance{}_seed{}", [(2, 2024), (3, 9999)], False),
    ("qwen2.5-1.5b-lr5e-7", "checkpoints/deltas_math_instance{}_seed{}", [(0, 42), (1, 1337), (2, 2024), (3, 9999)], False),

    # New downloads batch - unfinished
    ("qwen2.5-7b-grpo-math-lr3e-06", "seed{1}/deltas", [(None, 42), (None, 9999)], False),
    ("qwen2.5-1.5b-sft-math-lr2e-05", "seed{1}/deltas", [(None, 42), (None, 1337), (None, 2024), (None, 9999)], True),
    ("qwen2.5-1.5b-sft-math-lr3e-06", "seed{1}/deltas", [(None, 42), (None, 1337), (None, 2024), (None, 9999)], True),

    # Gemma 4B - all seeds
    ("gemma3-4b-iter1", "checkpoints/deltas_math_instance{}_seed{}", [(0, 42), (1, 1337), (2, 2024), (3, 9999)], False),
]


def parse_model_info(name: str) -> tuple[str, str] | None:
    """Parse model family and size from experiment name."""
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

    return family, size


def discover_remaining_experiments(base_dir: Path) -> list[ExperimentConfig]:
    """Discover remaining unfinished experiments."""
    experiments = []
    experiments_dir = base_dir / "experiments"

    for exp_name, pattern, seeds, is_sft in UNFINISHED_EXPERIMENTS:
        exp_dir = experiments_dir / exp_name
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue

        model_info = parse_model_info(exp_name)
        if model_info is None:
            logger.warning(f"Could not parse model info from: {exp_name}")
            continue

        family, size = model_info
        iteration_num = parse_iteration_num(exp_name)
        learning_rate = parse_learning_rate(exp_name)

        for instance, seed in seeds:
            # Build delta directory path
            if instance is not None:
                delta_subdir = pattern.format(instance, seed)
            else:
                delta_subdir = pattern.format(instance, seed).replace("None", "")

            delta_dir = exp_dir / delta_subdir

            if not delta_dir.exists():
                logger.warning(f"Delta dir not found: {delta_dir}")
                continue

            start_index, num_deltas, missing = detect_delta_indexing(delta_dir)

            if missing:
                logger.warning(f"Skipping {exp_name} seed{seed}: Missing {len(missing)} files")
                continue

            # Validate
            if is_sft:
                if num_deltas < 99:
                    logger.warning(f"Skipping {exp_name} seed{seed}: SFT needs >=99, found {num_deltas}")
                    continue
            else:
                if num_deltas != 399:
                    logger.warning(f"Skipping {exp_name} seed{seed}: GRPO needs 399, found {num_deltas}")
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
                f"iter={iteration_num} lr={learning_rate:.0e} sft={is_sft} ({num_deltas} deltas)"
            )

    return experiments


def main():
    data_dir = Path("/root/grail/research/sparsity_analysis")
    output_file = Path("/root/grail/data/sparsity_k_step_remaining.csv")

    logger.info("Discovering remaining unfinished experiments...")
    experiments = discover_remaining_experiments(data_dir)

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
