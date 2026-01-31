#!/usr/bin/env python3
"""Compute K-step sparsity using sparse sliding window algorithm.

Optimized version:
- Loads all deltas once as sparse (indices, values) tuples
- Uses sliding window with incremental zero-count tracking
- Memory: ~16 GB (sparse deltas + one window_sum tensor)
- Speed: ~20-50x faster than checkpoint-based approach

Precision: fp32 for all accumulation, exact zero comparison.
"""

import argparse
import csv
import gc
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

K_VALUES = [1, 2, 4, 8, 16, 32]


@dataclass
class ExperimentConfig:
    model_family: str
    model_size: str
    seed: int
    delta_dir: Path
    num_deltas: int
    start_index: int = 0  # Starting index of delta files (0 or 1)
    is_sft: bool = False  # Whether this is an SFT experiment
    iteration_num: int = 1  # Iteration number (default 1, e.g., iter16 -> 16)
    learning_rate: float = 3e-6  # Learning rate (default 3e-6)


def parse_experiment_path(path: Path) -> tuple[str, str] | None:
    name = path.name.lower()
    if "gemma" in name:
        family = "Gemma"
    elif "qwen" in name:
        family = "Qwen"
    elif "llama" in name:
        family = "Llama"
    else:
        return None

    size_match = re.search(r"(\d+\.?\d*)b", name)
    if not size_match:
        return None
    size = size_match.group(1) + "B"
    return family, size


def parse_iteration_num(name: str) -> int:
    """Extract iteration number from experiment name. Default is 1.

    Examples:
        "qwen2.5-1.5b-iter16" -> 16
        "qwen2.5-1.5b-iter1" -> 1
        "gemma3-1b-iter1" -> 1
        "qwen2.5-1.5b" -> 1 (default)
    """
    # Match patterns like iter16, iter8, iter32, iter1
    match = re.search(r"iter(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return 1  # Default


def parse_learning_rate(name: str) -> float:
    """Extract learning rate from experiment name. Default is 3e-6.

    Examples:
        "qwen2.5-1.5b-lr5e-7" -> 5e-7
        "qwen2.5-1.5b-sft-math-lr2e-05" -> 2e-5
        "qwen2.5-7b-grpo-math-lr3e-06" -> 3e-6
        "gemma3-1b-iter1" -> 3e-6 (default)
    """
    # Match patterns like lr5e-7, lr2e-05, lr3e-06
    match = re.search(r"lr(\d+\.?\d*)e-?(\d+)", name.lower())
    if match:
        mantissa = float(match.group(1))
        exponent = int(match.group(2))
        return mantissa * (10 ** -exponent)
    return 3e-6  # Default


def detect_delta_indexing(delta_dir: Path) -> tuple[int, int, list[int]]:
    """Detect delta file indexing scheme and validate.

    Returns:
        (start_index, num_deltas, missing_indices)
        - start_index: 0 or 1 depending on file naming
        - num_deltas: number of consecutive delta files found
        - missing_indices: list of missing indices (should be empty for valid data)
    """
    delta_files = sorted(delta_dir.glob("delta_*.pt"))
    if not delta_files:
        return 0, 0, []

    # Extract indices from filenames
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

    # Check for missing indices
    expected = set(range(start_index, end_index + 1))
    actual = set(indices)
    missing = sorted(expected - actual)

    num_deltas = len(indices)
    return start_index, num_deltas, missing


def validate_experiment(delta_dir: Path, is_sft: bool) -> tuple[bool, int, int, str]:
    """Validate experiment has correct number of delta files.

    Returns:
        (is_valid, start_index, num_deltas, error_message)
    """
    start_index, num_deltas, missing = detect_delta_indexing(delta_dir)

    if missing:
        return False, start_index, num_deltas, f"Missing {len(missing)} delta files: {missing[:5]}..."

    if is_sft:
        # SFT experiments need at least 99 deltas
        if num_deltas < 99:
            return False, start_index, num_deltas, f"SFT needs >=99 deltas, found {num_deltas}"
    else:
        # GRPO experiments need exactly 399 deltas
        if num_deltas != 399:
            return False, start_index, num_deltas, f"GRPO needs 399 deltas, found {num_deltas}"

    return True, start_index, num_deltas, ""


def discover_experiments(base_dir: Path, min_deltas: int = 100) -> list[ExperimentConfig]:
    experiments = []
    experiments_dir = base_dir / "experiments"

    if not experiments_dir.exists():
        return experiments

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        name = exp_dir.name.lower()
        # Skip SFT experiments (handled separately)
        if "sft" in name:
            continue
        # Skip partial/incomplete experiments
        if "partial" in name:
            continue

        model_info = parse_experiment_path(exp_dir)
        if model_info is None:
            continue

        model_family, model_size = model_info
        is_sft = "sft" in name
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        for delta_dir in sorted(checkpoints_dir.iterdir()):
            if not delta_dir.is_dir() or not delta_dir.name.startswith("deltas_"):
                continue

            seed_match = re.search(r"seed(\d+)", delta_dir.name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))

            # Validate and detect indexing
            is_valid, start_index, num_deltas, error_msg = validate_experiment(
                delta_dir, is_sft
            )

            if not is_valid:
                logger.warning(f"Skipping {delta_dir}: {error_msg}")
                continue

            if num_deltas < min_deltas:
                continue

            # Extract iteration number and learning rate from experiment folder name
            iteration_num = parse_iteration_num(exp_dir.name)
            learning_rate = parse_learning_rate(exp_dir.name)

            experiments.append(
                ExperimentConfig(
                    model_family=model_family,
                    model_size=model_size,
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
                f"Found: {model_family} {model_size} seed={seed} "
                f"iter={iteration_num} lr={learning_rate:.0e} ({num_deltas} deltas)"
            )

    # Deduplicate by (model_family, model_size, seed, iteration_num, learning_rate)
    # Keep first occurrence (prefer 0-indexed if available)
    seen = set()
    unique_experiments = []
    for exp in experiments:
        key = (exp.model_family, exp.model_size, exp.seed, exp.iteration_num, exp.learning_rate)
        if key not in seen:
            seen.add(key)
            unique_experiments.append(exp)
        else:
            logger.warning(f"Skipping duplicate: {exp.model_family} {exp.model_size} seed={exp.seed} iter={exp.iteration_num}")

    if len(unique_experiments) < len(experiments):
        logger.info(f"Deduplicated: {len(experiments)} -> {len(unique_experiments)} experiments")

    return unique_experiments


def get_layer_info(delta_path: Path) -> tuple[list, int]:
    """Get layer info and total params."""
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    layer_info = []
    offset = 0
    for name, data in delta["layers"].items():
        shape = tuple(data["shape"])
        numel = 1
        for s in shape:
            numel *= s
        layer_info.append((name, shape, offset, numel))
        offset += numel
    return layer_info, offset


def load_delta_sparse(
    path: Path, layer_info: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load delta and return flat indices and values as sparse representation."""
    delta = torch.load(path, map_location="cpu", weights_only=False)
    layers = delta["layers"]

    all_indices = []
    all_values = []

    for name, shape, offset, _numel in layer_info:
        if name not in layers:
            continue

        sparse_data = layers[name]
        indices = sparse_data["indices"]
        values = sparse_data["values"].float()  # bf16 -> fp32

        if indices.numel() == 0:
            continue

        # Convert ND indices to flat indices
        if len(shape) == 1:
            flat_idx = indices[0].long()
        elif len(shape) == 2:
            flat_idx = indices[0].long() * shape[1] + indices[1].long()
        else:
            flat_idx = torch.zeros(indices.size(1), dtype=torch.long)
            stride = 1
            for dim in range(len(shape) - 1, -1, -1):
                flat_idx += indices[dim].long() * stride
                stride *= shape[dim]

        all_indices.append(offset + flat_idx)
        all_values.append(values)

    del delta, layers

    if not all_indices:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

    return torch.cat(all_indices), torch.cat(all_values)


def load_delta_to_dense(path: Path, layer_info: list, total_params: int) -> torch.Tensor:
    """Load delta file and convert to flat dense fp32 tensor (for validation)."""
    indices, values = load_delta_sparse(path, layer_info)
    flat = torch.zeros(total_params, dtype=torch.float32)
    if indices.numel() > 0:
        flat[indices] = values
    return flat


def load_all_deltas_sparse(
    delta_dir: Path, num_deltas: int, layer_info: list, start_index: int = 0
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load all deltas as sparse (indices, values) tuples.

    Args:
        delta_dir: Directory containing delta files
        num_deltas: Number of delta files to load
        layer_info: Layer information from get_layer_info
        start_index: Starting index of delta files (0 or 1)
    """
    logger.info(f"  Loading {num_deltas} deltas as sparse tensors (start_index={start_index})...")
    all_deltas = []
    total_nnz = 0

    for i in range(num_deltas):
        # Use actual file index (accounting for start_index)
        file_idx = start_index + i
        delta_path = delta_dir / f"delta_{file_idx:06d}.pt"
        indices, values = load_delta_sparse(delta_path, layer_info)
        all_deltas.append((indices, values))
        total_nnz += indices.numel()

        if (i + 1) % 100 == 0:
            logger.info(f"    Loaded {i + 1}/{num_deltas} deltas")

    avg_nnz = total_nnz / num_deltas if num_deltas > 0 else 0
    mem_gb = total_nnz * 12 / 1e9  # 8 bytes for int64 + 4 bytes for float32
    logger.info(f"  Loaded {num_deltas} deltas: avg nnz={avg_nnz:,.0f}, total mem={mem_gb:.2f} GB")
    return all_deltas


def sparse_update_add(
    window_sum: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    zero_count: int,
) -> int:
    """Add sparse delta to window_sum and update zero_count incrementally."""
    if indices.numel() == 0:
        return zero_count

    old_vals = window_sum[indices]
    window_sum[indices] += values
    new_vals = window_sum[indices]

    was_zero = old_vals == 0.0
    is_zero = new_vals == 0.0

    # gained_zeros: was non-zero, now zero
    # lost_zeros: was zero, now non-zero
    gained_zeros = (~was_zero & is_zero).sum().item()
    lost_zeros = (was_zero & ~is_zero).sum().item()

    return zero_count + gained_zeros - lost_zeros


def sparse_update_sub(
    window_sum: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    zero_count: int,
) -> int:
    """Subtract sparse delta from window_sum and update zero_count incrementally."""
    if indices.numel() == 0:
        return zero_count

    old_vals = window_sum[indices]
    window_sum[indices] -= values
    new_vals = window_sum[indices]

    was_zero = old_vals == 0.0
    is_zero = new_vals == 0.0

    gained_zeros = (~was_zero & is_zero).sum().item()
    lost_zeros = (was_zero & ~is_zero).sum().item()

    return zero_count + gained_zeros - lost_zeros


def process_experiment(config: ExperimentConfig) -> list[tuple[int, int, float]]:
    """Process experiment using sparse sliding window algorithm.

    Note: Step indices in results are always 0-based regardless of file indexing.
    """
    logger.info(f"Processing {config.model_family} {config.model_size} seed={config.seed}")
    start_time = time.time()

    # Get layer info (use actual first file based on start_index)
    first_delta = config.delta_dir / f"delta_{config.start_index:06d}.pt"
    layer_info, total_params = get_layer_info(first_delta)
    logger.info(f"  Total params: {total_params:,}")

    # Load all deltas as sparse (handles start_index internally)
    all_deltas = load_all_deltas_sparse(
        config.delta_dir, config.num_deltas, layer_info, config.start_index
    )

    results = []

    # For each k value, compute sparsity using sliding window
    for k in K_VALUES:
        logger.info(f"  Computing k={k} sparsity...")
        max_t = config.num_deltas - k  # t ranges from 0 to max_t inclusive

        if max_t < 0:
            continue

        k_start = time.time()

        # Initialize window_sum and zero_count
        window_sum = torch.zeros(total_params, dtype=torch.float32)
        zero_count = total_params  # All zeros initially

        # Build initial window: sum of deltas 0 to k-1
        for i in range(k):
            indices, values = all_deltas[i]
            zero_count = sparse_update_add(window_sum, indices, values, zero_count)

        # Sliding window loop
        for t in range(max_t + 1):
            # Output sparsity for step t
            sparsity = zero_count / total_params * 100
            results.append((t, k, sparsity))

            # Update for next iteration
            if t < max_t:
                # Remove delta_t from window
                indices_out, values_out = all_deltas[t]
                zero_count = sparse_update_sub(window_sum, indices_out, values_out, zero_count)

                # Add delta_{t+k} to window
                indices_in, values_in = all_deltas[t + k]
                zero_count = sparse_update_add(window_sum, indices_in, values_in, zero_count)

            if (t + 1) % 100 == 0:
                logger.info(f"    k={k}: {t + 1}/{max_t + 1} steps")

        del window_sum
        k_elapsed = time.time() - k_start
        logger.info(f"    k={k}: done ({max_t + 1} results) in {k_elapsed:.1f}s")

    # Cleanup
    del all_deltas
    gc.collect()

    elapsed = time.time() - start_time
    logger.info(f"  Generated {len(results)} results in {elapsed:.1f}s")
    return results


# ============================================================================
# Validation: Original checkpoint-based algorithm for correctness comparison
# ============================================================================

CHECKPOINT_INTERVAL = 32


def build_checkpoints_original(
    delta_dir: Path, num_deltas: int, layer_info: list, total_params: int,
    start_index: int = 0
) -> list[torch.Tensor]:
    """Build cumsum checkpoints (original algorithm for validation)."""
    logger.info(f"    Building checkpoints for {num_deltas} deltas (start_index={start_index})...")
    checkpoints = [torch.zeros(total_params, dtype=torch.float32)]
    cumsum = torch.zeros(total_params, dtype=torch.float32)

    for i in range(num_deltas):
        file_idx = start_index + i
        delta_path = delta_dir / f"delta_{file_idx:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        cumsum = cumsum + delta
        del delta

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoints.append(cumsum.clone())
            logger.info(f"      Checkpoint {len(checkpoints) - 1} at step {i + 1}")

        if (i + 1) % 20 == 0:
            logger.info(f"      Processed {i + 1}/{num_deltas} deltas")

    checkpoints.append(cumsum.clone())
    logger.info(f"    Built {len(checkpoints)} checkpoints")
    return checkpoints


def compute_cumsum_original(
    target_idx: int,
    checkpoints: list[torch.Tensor],
    delta_dir: Path,
    layer_info: list,
    total_params: int,
    num_deltas: int,
    start_index: int = 0,
) -> torch.Tensor:
    """Compute cumsum[target_idx] from nearest checkpoint (original algorithm)."""
    if target_idx == 0:
        return torch.zeros(total_params, dtype=torch.float32)

    cp_idx = target_idx // CHECKPOINT_INTERVAL
    cp_cumsum_idx = cp_idx * CHECKPOINT_INTERVAL

    if cp_idx >= len(checkpoints) - 1:
        cp_idx = len(checkpoints) - 2
        cp_cumsum_idx = cp_idx * CHECKPOINT_INTERVAL

    result = checkpoints[cp_idx].clone()

    for i in range(cp_cumsum_idx, target_idx):
        if i >= num_deltas:
            break
        file_idx = start_index + i
        delta_path = delta_dir / f"delta_{file_idx:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        result = result + delta
        del delta

    return result


def process_experiment_original(
    config: ExperimentConfig, max_steps: Optional[int] = None
) -> list[tuple[int, int, float]]:
    """Original checkpoint-based algorithm for validation.

    Note: Step indices in results are always 0-based regardless of file indexing.
    """
    start_index = config.start_index
    first_delta = config.delta_dir / f"delta_{start_index:06d}.pt"
    layer_info, total_params = get_layer_info(first_delta)

    num_deltas = config.num_deltas
    if max_steps is not None:
        num_deltas = min(num_deltas, max_steps + max(K_VALUES))

    checkpoints = build_checkpoints_original(
        config.delta_dir, num_deltas, layer_info, total_params, start_index
    )

    results = []

    for k in K_VALUES:
        max_t = num_deltas - k
        if max_steps is not None:
            max_t = min(max_t, max_steps - 1)
        if max_t < 0:
            continue

        cumsum_t = torch.zeros(total_params, dtype=torch.float32)
        cumsum_t_plus_k = compute_cumsum_original(
            k, checkpoints, config.delta_dir, layer_info, total_params, num_deltas,
            start_index
        )

        for t in range(max_t + 1):
            k_step_delta = cumsum_t_plus_k - cumsum_t
            zeros = (k_step_delta == 0.0).sum().item()
            sparsity = zeros / total_params * 100
            results.append((t, k, sparsity))  # t is 0-based output index

            if t < max_t:
                # File index = start_index + logical index
                file_idx_t = start_index + t
                delta_t_path = config.delta_dir / f"delta_{file_idx_t:06d}.pt"
                delta_t = load_delta_to_dense(delta_t_path, layer_info, total_params)
                cumsum_t = cumsum_t + delta_t
                del delta_t

                delta_tk_idx = t + k
                if delta_tk_idx < num_deltas:
                    file_idx_tk = start_index + delta_tk_idx
                    delta_tk_path = config.delta_dir / f"delta_{file_idx_tk:06d}.pt"
                    delta_tk = load_delta_to_dense(delta_tk_path, layer_info, total_params)
                    cumsum_t_plus_k = cumsum_t_plus_k + delta_tk
                    del delta_tk

        del cumsum_t, cumsum_t_plus_k

    del checkpoints
    gc.collect()
    return results


def validate_against_original(config: ExperimentConfig, max_steps: int = 50) -> bool:
    """Validate optimized algorithm against original on first max_steps."""
    logger.info(f"Validating optimized vs original on first {max_steps} steps...")

    # Run original
    logger.info("  Running original algorithm...")
    start = time.time()
    original_results = process_experiment_original(config, max_steps=max_steps)
    orig_time = time.time() - start
    logger.info(f"  Original: {len(original_results)} results in {orig_time:.1f}s")

    # Run optimized (need to limit deltas loaded)
    logger.info("  Running optimized algorithm...")
    # Create a modified config with fewer deltas for fair comparison
    limited_config = ExperimentConfig(
        model_family=config.model_family,
        model_size=config.model_size,
        seed=config.seed,
        delta_dir=config.delta_dir,
        num_deltas=min(config.num_deltas, max_steps + max(K_VALUES)),
        start_index=config.start_index,
        is_sft=config.is_sft,
        iteration_num=config.iteration_num,
        learning_rate=config.learning_rate,
    )
    start = time.time()
    optimized_results = process_experiment(limited_config)
    opt_time = time.time() - start
    # Filter to only first max_steps
    optimized_results = [(t, k, s) for t, k, s in optimized_results if t < max_steps]
    logger.info(f"  Optimized: {len(optimized_results)} results in {opt_time:.1f}s")

    # Compare
    orig_dict = {(t, k): s for t, k, s in original_results}
    opt_dict = {(t, k): s for t, k, s in optimized_results}

    mismatches = 0
    max_diff = 0.0
    for key in orig_dict:
        if key not in opt_dict:
            logger.error(f"  Missing key in optimized: {key}")
            mismatches += 1
            continue
        diff = abs(orig_dict[key] - opt_dict[key])
        if diff > 1e-6:
            if mismatches < 5:
                logger.error(
                    f"  Mismatch at {key}: original={orig_dict[key]:.6f}, "
                    f"optimized={opt_dict[key]:.6f}, diff={diff:.2e}"
                )
            mismatches += 1
            max_diff = max(max_diff, diff)

    if mismatches == 0:
        logger.info(f"  VALIDATION PASSED: All {len(orig_dict)} values match exactly!")
        logger.info(f"  Speedup: {orig_time / opt_time:.1f}x")
        return True
    else:
        logger.error(f"  VALIDATION FAILED: {mismatches} mismatches, max diff={max_diff:.2e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Compute K-step sparsity")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/root/grail/research/sparsity_analysis")
    )
    parser.add_argument("--output", type=Path, default=Path("/root/grail/data/sparsity_k_step.csv"))
    parser.add_argument("--min-deltas", type=int, default=100)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate optimized algorithm against original on first experiment",
    )
    parser.add_argument(
        "--validate-steps",
        type=int,
        default=50,
        help="Number of steps to validate (default: 50)",
    )
    parser.add_argument(
        "--validate-model",
        type=str,
        default=None,
        help="Model size to validate on (e.g., '0.5B', '1B'). Uses first found if not specified.",
    )
    args = parser.parse_args()

    logger.info(f"Discovering experiments in {args.data_dir}")
    experiments = discover_experiments(args.data_dir, args.min_deltas)

    if not experiments:
        logger.error("No valid experiments found!")
        sys.exit(1)

    logger.info(f"Found {len(experiments)} experiment/seed combinations")

    # Validation mode: compare optimized vs original on first experiment
    if args.validate:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION MODE")
        logger.info("=" * 60)

        # Select experiment based on model size if specified
        config = experiments[0]
        if args.validate_model:
            target_size = args.validate_model.upper()
            if not target_size.endswith("B"):
                target_size += "B"
            for exp in experiments:
                if exp.model_size == target_size:
                    config = exp
                    break
            else:
                logger.warning(f"Model size {target_size} not found, using {config.model_size}")

        logger.info(f"Testing on: {config.model_family} {config.model_size} seed={config.seed}")
        success = validate_against_original(config, max_steps=args.validate_steps)
        sys.exit(0 if success else 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {args.output}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_family", "model_size", "seed", "step", "k", "sparsity", "iteration_num", "learning_rate"])

        for config in experiments:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"{config.model_family} {config.model_size} seed={config.seed}")
            logger.info(f"{'=' * 60}")

            try:
                results = process_experiment(config)

                for step, k, sparsity in results:
                    writer.writerow(
                        [
                            config.model_family,
                            config.model_size,
                            config.seed,
                            step,
                            k,
                            f"{sparsity:.4f}",
                            config.iteration_num,
                            f"{config.learning_rate:.0e}",
                        ]
                    )

                f.flush()
                logger.info(f"  Wrote {len(results)} rows")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback

                traceback.print_exc()

            gc.collect()

    logger.info(f"\nDone! Results: {args.output}")


if __name__ == "__main__":
    main()
