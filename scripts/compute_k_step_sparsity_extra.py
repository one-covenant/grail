#!/usr/bin/env python3
"""Compute K-step sparsity for non-iter1 experiments and SFT.

Processes: iter8, iter16, iter32, and SFT experiments.
SFT experiments are capped at 100 deltas.
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
SFT_MAX_DELTAS = 100


@dataclass
class ExperimentConfig:
    model_family: str
    model_size: str
    seed: int
    delta_dir: Path
    num_deltas: int
    experiment_type: str  # 'iter8', 'iter16', 'iter32', 'sft'
    start_index: int = 0  # First delta file index (some experiments start at 1)


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


def get_experiment_type(name: str) -> str | None:
    """Determine experiment type from directory name."""
    name = name.lower()
    if "sft" in name:
        return "sft"
    elif "iter32" in name:
        return "iter32"
    elif "iter16" in name:
        return "iter16"
    elif "iter8" in name:
        return "iter8"
    elif "iter1" in name:
        return "iter1"  # Will be filtered out
    return None


def discover_experiments(base_dir: Path, min_deltas: int = 33) -> list[ExperimentConfig]:
    """Discover non-iter1 experiments and SFT experiments."""
    experiments = []

    # 1. Search in experiments/ directory (iter8, iter16, iter32)
    experiments_dir = base_dir / "experiments"
    if experiments_dir.exists():
        for exp_dir in sorted(experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            name = exp_dir.name.lower()
            exp_type = get_experiment_type(name)

            if exp_type is None:
                logger.info(f"Skipping {exp_dir.name}: unknown experiment type")
                continue

            # Skip iter1 non-sft (already being processed by main script)
            if exp_type == "iter1" and "sft" not in name:
                logger.info(f"Skipping {exp_dir.name}: iter1 non-sft (processed by main script)")
                continue

            model_info = parse_experiment_path(exp_dir)
            if model_info is None:
                logger.info(f"Skipping {exp_dir.name}: could not parse model info")
                continue

            model_family, model_size = model_info
            checkpoints_dir = exp_dir / "checkpoints"
            if not checkpoints_dir.exists():
                logger.info(f"Skipping {exp_dir.name}: no checkpoints dir")
                continue

            for delta_dir in sorted(checkpoints_dir.iterdir()):
                if not delta_dir.is_dir() or not delta_dir.name.startswith("deltas_"):
                    continue

                seed_match = re.search(r"seed(\d+)", delta_dir.name)
                if not seed_match:
                    continue
                seed = int(seed_match.group(1))

                delta_files = sorted(delta_dir.glob("delta_*.pt"))
                num_deltas = len(delta_files)

                if num_deltas < min_deltas:
                    logger.info(
                        f"Skipping {exp_dir.name}/{delta_dir.name}: only {num_deltas} deltas (< {min_deltas})"
                    )
                    continue

                # Detect starting index from first file
                first_file = delta_files[0].name  # e.g., delta_000001.pt
                start_index = int(re.search(r"delta_(\d+)\.pt", first_file).group(1))

                experiments.append(
                    ExperimentConfig(
                        model_family=model_family,
                        model_size=model_size,
                        seed=seed,
                        delta_dir=delta_dir,
                        num_deltas=num_deltas,
                        experiment_type=exp_type,
                        start_index=start_index,
                    )
                )
                logger.info(
                    f"Found: {model_family} {model_size} {exp_type} seed={seed} ({num_deltas} deltas, start_idx={start_index})"
                )

    # 2. Search for SFT experiments at top level (different structure)
    # Structure: base_dir/qwen2.5-1.5b-sft-xxx/seed42/deltas/delta_*.pt
    for sft_dir in sorted(base_dir.iterdir()):
        if not sft_dir.is_dir():
            continue

        name = sft_dir.name.lower()
        if "sft" not in name:
            continue

        model_info = parse_experiment_path(sft_dir)
        if model_info is None:
            logger.info(f"Skipping SFT {sft_dir.name}: could not parse model info")
            continue

        model_family, model_size = model_info

        # Extract learning rate or other identifier from name
        lr_match = re.search(r"lr(\d+e-?\d+)", name)
        exp_type = f"sft-{lr_match.group(1)}" if lr_match else "sft"

        # Look for seed directories
        for seed_dir in sorted(sft_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            seed_match = re.search(r"seed(\d+)", seed_dir.name)
            if not seed_match:
                continue
            seed = int(seed_match.group(1))

            delta_dir = seed_dir / "deltas"
            if not delta_dir.exists():
                logger.info(f"Skipping SFT {sft_dir.name}/{seed_dir.name}: no deltas dir")
                continue

            delta_files = sorted(delta_dir.glob("delta_*.pt"))
            total_deltas = len(delta_files)

            # Cap SFT at SFT_MAX_DELTAS
            num_deltas = min(total_deltas, SFT_MAX_DELTAS)

            if num_deltas < min_deltas:
                logger.info(
                    f"Skipping SFT {sft_dir.name}/{seed_dir.name}: only {num_deltas} deltas (< {min_deltas})"
                )
                continue

            # Detect starting index from first file
            first_file = delta_files[0].name
            start_index = int(re.search(r"delta_(\d+)\.pt", first_file).group(1))

            experiments.append(
                ExperimentConfig(
                    model_family=model_family,
                    model_size=model_size,
                    seed=seed,
                    delta_dir=delta_dir,
                    num_deltas=num_deltas,
                    experiment_type=exp_type,
                    start_index=start_index,
                )
            )
            logger.info(
                f"Found: {model_family} {model_size} {exp_type} seed={seed} ({num_deltas} deltas, start_idx={start_index})"
            )

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
    delta_dir: Path, num_deltas: int, layer_info: list, total_params: int, start_index: int = 0
) -> list[torch.Tensor]:
    """Build cumsum checkpoints at every CHECKPOINT_INTERVAL steps."""
    logger.info(
        f"  Building checkpoints (interval={CHECKPOINT_INTERVAL}, start_index={start_index})..."
    )

    checkpoints = [torch.zeros(total_params, dtype=torch.float32)]
    cumsum = torch.zeros(total_params, dtype=torch.float32)

    for i in range(num_deltas):
        file_idx = i + start_index  # Actual file index
        delta_path = delta_dir / f"delta_{file_idx:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        cumsum = cumsum + delta
        del delta

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoints.append(cumsum.clone())
            logger.info(f"    Checkpoint {len(checkpoints) - 1}: cumsum[{i + 1}]")

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{num_deltas} deltas")

    final_cumsum_idx = num_deltas
    checkpoints.append(cumsum.clone())
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
    start_index: int = 0,
) -> torch.Tensor:
    """Compute cumsum[target_idx] from nearest checkpoint."""
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
        file_idx = i + start_index  # Actual file index
        delta_path = delta_dir / f"delta_{file_idx:06d}.pt"
        delta = load_delta_to_dense(delta_path, layer_info, total_params)
        result = result + delta
        del delta

    return result


def process_experiment(config: ExperimentConfig) -> list[tuple[int, int, float]]:
    """Process experiment using checkpointed cumsum."""
    logger.info(
        f"Processing {config.model_family} {config.model_size} {config.experiment_type} seed={config.seed}"
    )

    # Use start_index to find first delta file
    first_delta = config.delta_dir / f"delta_{config.start_index:06d}.pt"
    layer_info, total_params = get_layer_info(first_delta)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Start index: {config.start_index}")

    checkpoints = build_checkpoints(
        config.delta_dir, config.num_deltas, layer_info, total_params, config.start_index
    )

    results = []

    for k in K_VALUES:
        logger.info(f"  Computing k={k} sparsity...")
        max_t = config.num_deltas - k

        if max_t < 0:
            continue

        cumsum_t = torch.zeros(total_params, dtype=torch.float32)
        cumsum_t_plus_k = compute_cumsum_from_checkpoints(
            k,
            checkpoints,
            config.delta_dir,
            layer_info,
            total_params,
            config.num_deltas,
            config.start_index,
        )

        for t in range(max_t + 1):
            k_step_delta = cumsum_t_plus_k - cumsum_t
            zeros = (k_step_delta == 0.0).sum().item()
            sparsity = zeros / total_params * 100
            results.append((t, k, sparsity))

            if t < max_t:
                file_idx_t = t + config.start_index
                delta_t_path = config.delta_dir / f"delta_{file_idx_t:06d}.pt"
                delta_t = load_delta_to_dense(delta_t_path, layer_info, total_params)
                cumsum_t = cumsum_t + delta_t
                del delta_t

                delta_tk_idx = t + k
                if delta_tk_idx < config.num_deltas:
                    file_idx_tk = delta_tk_idx + config.start_index
                    delta_tk_path = config.delta_dir / f"delta_{file_idx_tk:06d}.pt"
                    delta_tk = load_delta_to_dense(delta_tk_path, layer_info, total_params)
                    cumsum_t_plus_k = cumsum_t_plus_k + delta_tk
                    del delta_tk

            if (t + 1) % 100 == 0:
                logger.info(f"    k={k}: {t + 1}/{max_t + 1} steps")

        del cumsum_t, cumsum_t_plus_k
        gc.collect()
        logger.info(f"    k={k}: done ({max_t + 1} results)")

    del checkpoints
    gc.collect()

    logger.info(f"  Generated {len(results)} total results")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute K-step sparsity for extra experiments")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/root/grail/research/sparsity_analysis")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/root/grail/data/sparsity_k_step_extra.csv")
    )
    parser.add_argument(
        "--min-deltas", type=int, default=33
    )  # Lower threshold for smaller experiments
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
        writer.writerow(
            ["model_family", "model_size", "experiment_type", "seed", "step", "k", "sparsity"]
        )

        for config in experiments:
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"{config.model_family} {config.model_size} {config.experiment_type} seed={config.seed}"
            )
            logger.info(f"{'=' * 60}")

            try:
                results = process_experiment(config)

                for step, k, sparsity in results:
                    writer.writerow(
                        [
                            config.model_family,
                            config.model_size,
                            config.experiment_type,
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
