#!/usr/bin/env python3
"""Validate optimized k-step sparsity on real Qwen 0.5B deltas.

Runs the original (dense) and optimized (sparse sliding window) algorithms on a
small slice of the existing Qwen 0.5B checkpoints and asserts exact match.
Defaults keep runtime under 10 minutes.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import compute_k_step_sparsity as ck

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def parse_k_values(value: str) -> list[int]:
    """Parse comma-separated k values."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("k-values must be a non-empty list")
    k_values: list[int] = []
    for part in parts:
        k = int(part)
        if k <= 0:
            raise ValueError(f"Invalid k value: {k}")
        k_values.append(k)
    return sorted(set(k_values))


def select_qwen_0_5b(experiments: list[ck.ExperimentConfig]) -> ck.ExperimentConfig:
    """Return the first Qwen 0.5B experiment."""
    for exp in experiments:
        if exp.model_family == "Qwen" and exp.model_size == "0.5B":
            return exp
    raise RuntimeError("Qwen 0.5B experiment not found in data directory")


def compare_results(
    original: list[tuple[int, int, float]],
    optimized: list[tuple[int, int, float]],
) -> tuple[int, float]:
    """Compare results and return (mismatches, max_diff)."""
    orig_dict = {(t, k): s for t, k, s in original}
    opt_dict = {(t, k): s for t, k, s in optimized}
    mismatches = 0
    max_diff = 0.0

    for key, orig_val in orig_dict.items():
        if key not in opt_dict:
            logger.error("Missing key in optimized results: %s", key)
            mismatches += 1
            continue
        diff = abs(orig_val - opt_dict[key])
        if diff > 1e-6:
            if mismatches < 5:
                logger.error(
                    "Mismatch at %s: original=%.6f optimized=%.6f diff=%.2e",
                    key,
                    orig_val,
                    opt_dict[key],
                    diff,
                )
            mismatches += 1
            max_diff = max(max_diff, diff)

    return mismatches, max_diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate optimized k-step sparsity on Qwen 0.5B checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/grail/research/sparsity_analysis"),
    )
    parser.add_argument("--min-deltas", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Number of steps to validate (default: 10)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,4",
        help="Comma-separated k values (default: 1,2,4)",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=600,
        help="Fail if runtime exceeds this limit (default: 600s)",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    logger.info("Using k values: %s", k_values)

    logger.info("Discovering experiments in %s", args.data_dir)
    experiments = ck.discover_experiments(args.data_dir, args.min_deltas)
    if not experiments:
        logger.error("No valid experiments found")
        return 1

    try:
        config = select_qwen_0_5b(experiments)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Selected: %s %s seed=%s", config.model_family, config.model_size, config.seed)

    ck.K_VALUES = k_values
    max_k = max(k_values)
    limited_num_deltas = min(config.num_deltas, args.max_steps + max_k)
    limited_config = ck.ExperimentConfig(
        model_family=config.model_family,
        model_size=config.model_size,
        seed=config.seed,
        delta_dir=config.delta_dir,
        num_deltas=limited_num_deltas,
    )

    start_time = time.time()

    logger.info("Running original algorithm (dense)")
    orig_start = time.time()
    original_results = ck.process_experiment_original(
        limited_config, max_steps=args.max_steps
    )
    orig_time = time.time() - orig_start
    logger.info("Original: %d results in %.1fs", len(original_results), orig_time)

    logger.info("Running optimized algorithm (sparse)")
    opt_start = time.time()
    optimized_results = ck.process_experiment(limited_config)
    opt_time = time.time() - opt_start
    optimized_results = [
        (t, k, s) for t, k, s in optimized_results if t < args.max_steps
    ]
    logger.info("Optimized: %d results in %.1fs", len(optimized_results), opt_time)

    mismatches, max_diff = compare_results(original_results, optimized_results)
    total_time = time.time() - start_time

    if total_time > args.max_runtime_seconds:
        logger.error(
            "Runtime exceeded limit: %.1fs > %ss", total_time, args.max_runtime_seconds
        )
        return 1

    if mismatches == 0:
        logger.info(
            "VALIDATION PASSED: %d values match exactly (speedup %.1fx)",
            len(original_results),
            orig_time / max(opt_time, 1e-9),
        )
        logger.info("Total runtime: %.1fs", total_time)
        return 0

    logger.error(
        "VALIDATION FAILED: %d mismatches, max diff=%.2e", mismatches, max_diff
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Validate optimized k-step sparsity on real Qwen 0.5B deltas.

Runs the original (dense) and optimized (sparse sliding window) algorithms on a
small slice of the existing Qwen 0.5B checkpoints and asserts exact match.
Defaults keep runtime under 10 minutes.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import compute_k_step_sparsity as ck

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def parse_k_values(value: str) -> list[int]:
    """Parse comma-separated k values."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("k-values must be a non-empty list")
    k_values: list[int] = []
    for part in parts:
        k = int(part)
        if k <= 0:
            raise ValueError(f"Invalid k value: {k}")
        k_values.append(k)
    return sorted(set(k_values))


def select_qwen_0_5b(experiments: list[ck.ExperimentConfig]) -> ck.ExperimentConfig:
    """Return the first Qwen 0.5B experiment."""
    for exp in experiments:
        if exp.model_family == "Qwen" and exp.model_size == "0.5B":
            return exp
    raise RuntimeError("Qwen 0.5B experiment not found in data directory")


def compare_results(
    original: list[tuple[int, int, float]],
    optimized: list[tuple[int, int, float]],
) -> tuple[int, float]:
    """Compare results and return (mismatches, max_diff)."""
    orig_dict = {(t, k): s for t, k, s in original}
    opt_dict = {(t, k): s for t, k, s in optimized}
    mismatches = 0
    max_diff = 0.0

    for key, orig_val in orig_dict.items():
        if key not in opt_dict:
            logger.error("Missing key in optimized results: %s", key)
            mismatches += 1
            continue
        diff = abs(orig_val - opt_dict[key])
        if diff > 1e-6:
            if mismatches < 5:
                logger.error(
                    "Mismatch at %s: original=%.6f optimized=%.6f diff=%.2e",
                    key,
                    orig_val,
                    opt_dict[key],
                    diff,
                )
            mismatches += 1
            max_diff = max(max_diff, diff)

    return mismatches, max_diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate optimized k-step sparsity on Qwen 0.5B checkpoints"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/grail/research/sparsity_analysis"),
    )
    parser.add_argument("--min-deltas", type=int, default=100)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Number of steps to validate (default: 10)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,4",
        help="Comma-separated k values (default: 1,2,4)",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=600,
        help="Fail if runtime exceeds this limit (default: 600s)",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    logger.info("Using k values: %s", k_values)

    logger.info("Discovering experiments in %s", args.data_dir)
    experiments = ck.discover_experiments(args.data_dir, args.min_deltas)
    if not experiments:
        logger.error("No valid experiments found")
        return 1

    try:
        config = select_qwen_0_5b(experiments)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Selected: %s %s seed=%s", config.model_family, config.model_size, config.seed)

    # Use smaller k values for fast validation
    ck.K_VALUES = k_values
    max_k = max(k_values)
    limited_num_deltas = min(config.num_deltas, args.max_steps + max_k)
    limited_config = ck.ExperimentConfig(
        model_family=config.model_family,
        model_size=config.model_size,
        seed=config.seed,
        delta_dir=config.delta_dir,
        num_deltas=limited_num_deltas,
    )

    start_time = time.time()

    logger.info("Running original algorithm (dense)")
    orig_start = time.time()
    original_results = ck.process_experiment_original(
        limited_config, max_steps=args.max_steps
    )
    orig_time = time.time() - orig_start
    logger.info("Original: %d results in %.1fs", len(original_results), orig_time)

    logger.info("Running optimized algorithm (sparse)")
    opt_start = time.time()
    optimized_results = ck.process_experiment(limited_config)
    opt_time = time.time() - opt_start
    optimized_results = [
        (t, k, s) for t, k, s in optimized_results if t < args.max_steps
    ]
    logger.info("Optimized: %d results in %.1fs", len(optimized_results), opt_time)

    mismatches, max_diff = compare_results(original_results, optimized_results)
    total_time = time.time() - start_time

    if total_time > args.max_runtime_seconds:
        logger.error(
            "Runtime exceeded limit: %.1fs > %ss", total_time, args.max_runtime_seconds
        )
        return 1

    if mismatches == 0:
        logger.info(
            "VALIDATION PASSED: %d values match exactly (speedup %.1fx)",
            len(original_results),
            orig_time / max(opt_time, 1e-9),
        )
        logger.info("Total runtime: %.1fs", total_time)
        return 0

    logger.error(
        "VALIDATION FAILED: %d mismatches, max diff=%.2e", mismatches, max_diff
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
