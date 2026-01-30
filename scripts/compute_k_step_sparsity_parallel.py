#!/usr/bin/env python3
"""Compute K-step sparsity for a single model/seed combination.

Usage: python compute_k_step_sparsity_parallel.py --model "Gemma 1B" --seed 42 --output output.csv
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


def find_experiment(
    base_dir: Path, model_family: str, model_size: str, seed: int
) -> ExperimentConfig | None:
    """Find a specific experiment by model and seed."""
    experiments_dir = base_dir / "experiments"

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        name = exp_dir.name.lower()
        # Use regex to match exactly 'iter1' (not iter16, iter10, etc.)
        if not re.search(r"-iter1\b", name) or "sft" in name:
            continue

        # Check model family
        if model_family.lower() == "gemma" and "gemma" not in name:
            continue
        if model_family.lower() == "qwen" and "qwen" not in name:
            continue
        if model_family.lower() == "llama" and "llama" not in name:
            continue

        # Check model size
        size_match = re.search(r"(\d+\.?\d*)b", name)
        if not size_match:
            continue
        size = size_match.group(1) + "B"
        if size != model_size:
            continue

        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            continue

        for delta_dir in checkpoints_dir.iterdir():
            if not delta_dir.is_dir() or not delta_dir.name.startswith("deltas_"):
                continue

            seed_match = re.search(r"seed(\d+)", delta_dir.name)
            if not seed_match or int(seed_match.group(1)) != seed:
                continue

            delta_files = list(delta_dir.glob("delta_*.pt"))
            num_deltas = len(delta_files)

            if num_deltas < 100:
                continue

            return ExperimentConfig(
                model_family=model_family,
                model_size=model_size,
                seed=seed,
                delta_dir=delta_dir,
                num_deltas=num_deltas,
            )

    return None


def get_layer_info(delta_path: Path) -> tuple[list, int]:
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


def load_delta_to_dense(
    path: Path, layer_info: list, total_params: int, flat: torch.Tensor = None
) -> torch.Tensor:
    """Load delta file and convert to flat dense fp32 tensor. Reuses tensor if provided."""
    delta = torch.load(path, map_location="cpu", weights_only=False)
    layers = delta["layers"]

    if flat is None:
        flat = torch.zeros(total_params, dtype=torch.float32)
    else:
        flat.zero_()

    for name, shape, offset, _numel in layer_info:
        if name not in layers:
            continue

        sparse_data = layers[name]
        indices = sparse_data["indices"]
        values = sparse_data["values"].float()

        if indices.numel() == 0:
            continue

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
    logger.info(f"  Building checkpoints (interval={CHECKPOINT_INTERVAL})...")

    checkpoints = [torch.zeros(total_params, dtype=torch.float32)]
    cumsum = torch.zeros(total_params, dtype=torch.float32)
    delta_buffer = torch.zeros(total_params, dtype=torch.float32)  # Reusable buffer

    for i in range(num_deltas):
        delta_path = delta_dir / f"delta_{i:06d}.pt"
        load_delta_to_dense(delta_path, layer_info, total_params, delta_buffer)
        cumsum.add_(delta_buffer)  # In-place addition

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoints.append(cumsum.clone())
            logger.info(f"    Checkpoint {len(checkpoints) - 1}: cumsum[{i + 1}]")

        if (i + 1) % 50 == 0:
            logger.info(f"    Processed {i + 1}/{num_deltas} deltas")

    checkpoints.append(cumsum.clone())
    logger.info(f"    Final cumsum[{num_deltas}] stored")
    logger.info(f"  Built {len(checkpoints)} checkpoints")

    del delta_buffer
    return checkpoints


def compute_cumsum_from_checkpoints(
    target_idx: int,
    checkpoints: list[torch.Tensor],
    delta_dir: Path,
    layer_info: list,
    total_params: int,
    num_deltas: int,
    delta_buffer: torch.Tensor,
) -> torch.Tensor:
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
        delta_path = delta_dir / f"delta_{i:06d}.pt"
        load_delta_to_dense(delta_path, layer_info, total_params, delta_buffer)
        result.add_(delta_buffer)

    return result


def process_experiment(config: ExperimentConfig) -> list[tuple[int, int, float]]:
    logger.info(f"Processing {config.model_family} {config.model_size} seed={config.seed}")

    first_delta = config.delta_dir / "delta_000000.pt"
    layer_info, total_params = get_layer_info(first_delta)
    logger.info(f"  Total params: {total_params:,}")

    checkpoints = build_checkpoints(config.delta_dir, config.num_deltas, layer_info, total_params)

    results = []

    # Reusable buffers
    delta_buffer = torch.zeros(total_params, dtype=torch.float32)
    delta_buffer2 = torch.zeros(total_params, dtype=torch.float32)

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
            delta_buffer,
        )

        for t in range(max_t + 1):
            k_step_delta = cumsum_t_plus_k - cumsum_t
            zeros = (k_step_delta == 0.0).sum().item()
            sparsity = zeros / total_params * 100
            results.append((t, k, sparsity))

            if t < max_t:
                delta_t_path = config.delta_dir / f"delta_{t:06d}.pt"
                load_delta_to_dense(delta_t_path, layer_info, total_params, delta_buffer)
                cumsum_t.add_(delta_buffer)

                delta_tk_idx = t + k
                if delta_tk_idx < config.num_deltas:
                    delta_tk_path = config.delta_dir / f"delta_{delta_tk_idx:06d}.pt"
                    load_delta_to_dense(delta_tk_path, layer_info, total_params, delta_buffer2)
                    cumsum_t_plus_k.add_(delta_buffer2)

            if (t + 1) % 100 == 0:
                logger.info(f"    k={k}: {t + 1}/{max_t + 1} steps")

        del cumsum_t, cumsum_t_plus_k
        gc.collect()
        logger.info(f"    k={k}: done ({max_t + 1} results)")

    del checkpoints, delta_buffer, delta_buffer2
    gc.collect()

    logger.info(f"  Generated {len(results)} total results")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute K-step sparsity for a single model/seed")
    parser.add_argument("--model", type=str, required=True, help='Model name, e.g., "Gemma 1B"')
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/root/grail/research/sparsity_analysis")
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    args = parser.parse_args()

    # Parse model name
    parts = args.model.split()
    if len(parts) != 2:
        logger.error(
            f"Invalid model format: {args.model}. Expected 'Family Size', e.g., 'Gemma 1B'"
        )
        sys.exit(1)
    model_family, model_size = parts

    logger.info(f"Looking for {model_family} {model_size} seed={args.seed}")

    config = find_experiment(args.data_dir, model_family, model_size, args.seed)

    if config is None:
        logger.error(f"Experiment not found: {model_family} {model_size} seed={args.seed}")
        sys.exit(1)

    logger.info(f"Found: {config.delta_dir} ({config.num_deltas} deltas)")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = process_experiment(config)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_family", "model_size", "seed", "step", "k", "sparsity"])
        for step, k, sparsity in results:
            writer.writerow(
                [config.model_family, config.model_size, config.seed, step, k, f"{sparsity:.4f}"]
            )

    logger.info(f"Done! Results: {args.output}")


if __name__ == "__main__":
    main()
