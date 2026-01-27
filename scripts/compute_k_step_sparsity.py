#!/usr/bin/env python3
"""Compute K-step sparsity using checkpointed cumulative sums.

Precision: fp32 for all accumulation, exact zero comparison.
Memory: ~65GB (13 checkpoints × 4GB + working memory)
"""

import argparse
import csv
import gc
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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
CHECKPOINT_INTERVAL = 32


@dataclass
class ExperimentConfig:
    model_family: str
    model_size: str
    seed: int
    delta_dir: Path
    num_deltas: int


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


def discover_experiments(base_dir: Path, min_deltas: int = 100) -> list[ExperimentConfig]:
    experiments = []
    experiments_dir = base_dir / "experiments"

    if not experiments_dir.exists():
        return experiments

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        name = exp_dir.name.lower()
        if "iter1" not in name or "sft" in name:
            continue

        model_info = parse_experiment_path(exp_dir)
        if model_info is None:
            continue

        model_family, model_size = model_info
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

            delta_files = list(delta_dir.glob("delta_*.pt"))
            num_deltas = len(delta_files)

            if num_deltas < min_deltas:
                continue

            experiments.append(
                ExperimentConfig(
                    model_family=model_family,
                    model_size=model_size,
                    seed=seed,
                    delta_dir=delta_dir,
                    num_deltas=num_deltas,
                )
            )
            logger.info(f"Found: {model_family} {model_size} seed={seed} ({num_deltas} deltas)")

    return experiments


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


def load_delta_to_dense(path: Path, layer_info: list, total_params: int) -> torch.Tensor:
    """Load delta file and convert to flat dense fp32 tensor."""
    delta = torch.load(path, map_location="cpu", weights_only=False)
    layers = delta["layers"]

    flat = torch.zeros(total_params, dtype=torch.float32)

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

        flat[offset + flat_idx] = values

    del delta, layers
    return flat


def build_checkpoints(
    delta_dir: Path, num_deltas: int, layer_info: list, total_params: int
) -> list[torch.Tensor]:
    """Build cumsum checkpoints at every CHECKPOINT_INTERVAL steps.

    checkpoint[j] = cumsum[j * CHECKPOINT_INTERVAL] = sum of delta_0 to delta_{j*CHECKPOINT_INTERVAL - 1}
    checkpoint[0] = cumsum[0] = zeros (no deltas)
    """
    logger.info(f"  Building checkpoints (interval={CHECKPOINT_INTERVAL})...")

    checkpoints = [torch.zeros(total_params, dtype=torch.float32)]  # checkpoint[0] = cumsum[0] = 0
    cumsum = torch.zeros(total_params, dtype=torch.float32)

    for i in range(num_deltas):
        # Load and add delta_i
        delta_path = delta_dir / f"delta_{i:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        cumsum = cumsum + delta  # cumsum now = sum of delta_0 to delta_i = cumsum[i+1]
        del delta

        # After adding delta_i, cumsum = cumsum[i+1]
        # Store checkpoint if (i+1) is multiple of CHECKPOINT_INTERVAL
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoints.append(cumsum.clone())
            logger.info(f"    Checkpoint {len(checkpoints) - 1}: cumsum[{i + 1}]")

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{num_deltas} deltas")

    # Store final cumsum for convenience
    # cumsum now = cumsum[num_deltas]
    final_cumsum_idx = num_deltas
    checkpoints.append(cumsum.clone())  # This might not be at a checkpoint boundary
    logger.info(f"    Final cumsum[{final_cumsum_idx}] stored")

    logger.info(f"  Built {len(checkpoints)} checkpoints")
    return checkpoints


def compute_cumsum_from_checkpoints(
    target_idx: int,
    checkpoints: list[torch.Tensor],
    delta_dir: Path,
    layer_info: list,
    total_params: int,
    num_deltas: int,
) -> torch.Tensor:
    """Compute cumsum[target_idx] from nearest checkpoint.

    cumsum[i] = sum of delta_0 to delta_{i-1}
    """
    if target_idx == 0:
        return torch.zeros(total_params, dtype=torch.float32)

    # Find nearest checkpoint <= target_idx
    cp_idx = target_idx // CHECKPOINT_INTERVAL
    cp_cumsum_idx = cp_idx * CHECKPOINT_INTERVAL

    # Handle edge case: if target is beyond last checkpoint
    if cp_idx >= len(checkpoints) - 1:
        # Use the checkpoint just before final
        cp_idx = len(checkpoints) - 2
        cp_cumsum_idx = cp_idx * CHECKPOINT_INTERVAL

    # Start from checkpoint
    result = checkpoints[cp_idx].clone()

    # Add deltas from cp_cumsum_idx to target_idx - 1
    # result = cumsum[cp_cumsum_idx], we need cumsum[target_idx]
    # cumsum[target_idx] = cumsum[cp_cumsum_idx] + delta_{cp_cumsum_idx} + ... + delta_{target_idx - 1}
    for i in range(cp_cumsum_idx, target_idx):
        if i >= num_deltas:
            break
        delta_path = delta_dir / f"delta_{i:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        result = result + delta
        del delta

    return result


def process_experiment(config: ExperimentConfig) -> list[tuple[int, int, float]]:
    """Process experiment using checkpointed cumsum."""
    logger.info(f"Processing {config.model_family} {config.model_size} seed={config.seed}")

    # Get layer info
    first_delta = config.delta_dir / "delta_000000.pt"
    layer_info, total_params = get_layer_info(first_delta)
    logger.info(f"  Total params: {total_params:,}")

    # Build checkpoints
    checkpoints = build_checkpoints(config.delta_dir, config.num_deltas, layer_info, total_params)

    results = []

    # For each k value, compute sparsity using sliding window
    for k in K_VALUES:
        logger.info(f"  Computing k={k} sparsity...")
        max_t = config.num_deltas - k  # t ranges from 0 to max_t inclusive

        if max_t < 0:
            continue

        # Initialize: cumsum_t = cumsum[0], cumsum_t_plus_k = cumsum[k]
        cumsum_t = torch.zeros(total_params, dtype=torch.float32)
        cumsum_t_plus_k = compute_cumsum_from_checkpoints(
            k, checkpoints, config.delta_dir, layer_info, total_params, config.num_deltas
        )

        for t in range(max_t + 1):
            # Compute k-step delta and sparsity
            k_step_delta = cumsum_t_plus_k - cumsum_t
            zeros = (k_step_delta == 0.0).sum().item()
            sparsity = zeros / total_params * 100
            results.append((t, k, sparsity))

            # Update for next iteration: t -> t+1
            # cumsum_t becomes cumsum[t+1] = cumsum[t] + delta_t
            # cumsum_t_plus_k becomes cumsum[t+k+1] = cumsum[t+k] + delta_{t+k}

            if t < max_t:  # Don't load on last iteration
                # Load delta_t
                delta_t_path = config.delta_dir / f"delta_{t:06d}.pt"
                delta_t = load_delta_to_dense(delta_t_path, layer_info, total_params)
                cumsum_t = cumsum_t + delta_t
                del delta_t

                # Load delta_{t+k}
                delta_tk_idx = t + k
                if delta_tk_idx < config.num_deltas:
                    delta_tk_path = config.delta_dir / f"delta_{delta_tk_idx:06d}.pt"
                    delta_tk = load_delta_to_dense(delta_tk_path, layer_info, total_params)
                    cumsum_t_plus_k = cumsum_t_plus_k + delta_tk
                    del delta_tk

            if (t + 1) % 100 == 0:
                logger.info(f"    k={k}: {t + 1}/{max_t + 1} steps")

        del cumsum_t, cumsum_t_plus_k
        gc.collect()
        logger.info(f"    k={k}: done ({max_t + 1} results)")

    # Cleanup
    del checkpoints
    gc.collect()

    logger.info(f"  Generated {len(results)} total results")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute K-step sparsity")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/root/grail/research/sparsity_analysis")
    )
    parser.add_argument("--output", type=Path, default=Path("/root/grail/data/sparsity_k_step.csv"))
    parser.add_argument("--min-deltas", type=int, default=100)
    args = parser.parse_args()

    logger.info(f"Discovering experiments in {args.data_dir}")
    experiments = discover_experiments(args.data_dir, args.min_deltas)

    if not experiments:
        logger.error("No valid experiments found!")
        sys.exit(1)

    logger.info(f"Found {len(experiments)} experiment/seed combinations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {args.output}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_family", "model_size", "seed", "step", "k", "sparsity"])

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
