#!/usr/bin/env python3
"""
Compression Benchmark - Production Accurate

Benchmarks compression on ACTUAL weight values (like production), not delta values.

Key difference from benchmark_compression_standard.py:
- Standard benchmark: compresses W_new - W_old (delta values, ~1e-6 magnitude)
- This benchmark: compresses W_new[changed_positions] (actual weights, ~1e-4 magnitude)

This matches production behavior in grail/infrastructure/delta_checkpoint.py:
    values = current_2d[row_idx, col_idx].contiguous()  # Actual weights

For research delta files, we reconstruct actual weights by:
    W_new = W_base + delta_values
Then extract values at changed positions (where delta != 0).

Author: Grail Research
"""

import argparse
import csv
import gc
import json
import logging
import platform
import struct
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, "/root/grail")

LIBRARY_VERSIONS: dict[str, str] = {}

try:
    import lz4.frame as lz4f
    import lz4
    LIBRARY_VERSIONS["lz4"] = lz4.__version__
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import snappy
    LIBRARY_VERSIONS["snappy"] = "installed"
    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False

try:
    import zstandard as zstd
    LIBRARY_VERSIONS["zstandard"] = zstd.__version__
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    experiment: str
    source_type: str
    num_layers: int
    num_elements: int
    total_params: int
    sparsity: float
    representation: str
    algorithm: str
    coo_baseline_bytes: int
    uncompressed_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    encode_time_ms: float
    decode_time_ms: float
    throughput_encode_mb_s: float
    throughput_decode_mb_s: float
    verified_correct: bool


def get_available_algorithms() -> list[str]:
    algos = []
    if HAS_LZ4:
        algos.append("lz4")
    if HAS_SNAPPY:
        algos.append("snappy")
    if HAS_ZSTD:
        algos.extend(["zstd-1", "zstd-3"])
    return algos


def compress(data: bytes, algorithm: str) -> bytes:
    if algorithm == "lz4":
        return lz4f.compress(data)
    elif algorithm == "snappy":
        return snappy.compress(data)
    elif algorithm.startswith("zstd-"):
        level = int(algorithm.split("-")[1])
        return zstd.ZstdCompressor(level=level).compress(data)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def decompress(data: bytes, algorithm: str) -> bytes:
    if algorithm in ("lz4", "lz4hc"):
        return lz4f.decompress(data)
    elif algorithm == "snappy":
        return snappy.decompress(data)
    elif algorithm.startswith("zstd-"):
        return zstd.ZstdDecompressor().decompress(data)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def benchmark_compression(data: bytes, algorithm: str) -> tuple[bytes, float, float, bool]:
    compressed = compress(data, algorithm)
    _ = decompress(compressed, algorithm)
    
    gc.collect()
    start = time.perf_counter()
    compressed = compress(data, algorithm)
    enc_ms = (time.perf_counter() - start) * 1000
    
    gc.collect()
    start = time.perf_counter()
    decompressed = decompress(compressed, algorithm)
    dec_ms = (time.perf_counter() - start) * 1000
    
    return compressed, enc_ms, dec_ms, (decompressed == data)


def delta_encode_coo_int32(rows: np.ndarray, cols: np.ndarray):
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    if len(rows) == 0:
        return rows, cols, np.array([], dtype=np.int64)
    
    sort_order = np.lexsort((cols, rows))
    sorted_rows = rows[sort_order]
    sorted_cols = cols[sort_order]
    
    delta_rows = np.empty_like(sorted_rows, dtype=np.int32)
    delta_rows[0] = sorted_rows[0]
    delta_rows[1:] = sorted_rows[1:] - sorted_rows[:-1]
    
    row_changed = np.ones(len(sorted_rows), dtype=bool)
    row_changed[1:] = delta_rows[1:] != 0
    
    col_diff = np.empty_like(sorted_cols, dtype=np.int32)
    col_diff[0] = sorted_cols[0]
    col_diff[1:] = sorted_cols[1:] - sorted_cols[:-1]
    
    delta_cols = np.where(row_changed, sorted_cols, col_diff).astype(np.int32)
    return delta_rows, delta_cols, sort_order


def delta_encode_flat_int32(indices: np.ndarray):
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) == 0:
        return indices, np.array([], dtype=np.int64)
    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]
    delta = np.empty_like(sorted_indices, dtype=np.int32)
    delta[0] = sorted_indices[0]
    delta[1:] = sorted_indices[1:] - sorted_indices[:-1]
    return delta, sort_order


def build_delta_coo_downscaled(layers: dict) -> bytes:
    chunks = []
    for layer_data in layers.values():
        indices = layer_data["indices"].numpy()
        values = layer_data["values"]
        values_np = values.view(torch.int16).numpy() if values.dtype == torch.bfloat16 else values.numpy()
        
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0], indices[1]
            delta_rows, delta_cols, sort_order = delta_encode_coo_int32(rows, cols)
            values_sorted = values_np[sort_order]
            
            if delta_rows.size > 0 and delta_rows.min() >= 0 and delta_rows.max() <= 255:
                chunks.append(delta_rows.astype(np.uint8).tobytes())
            else:
                chunks.append(delta_rows.astype(np.int32).tobytes())
            
            if delta_cols.size > 0 and delta_cols.min() >= 0 and delta_cols.max() <= 65535:
                chunks.append(delta_cols.astype(np.uint16).tobytes())
            else:
                chunks.append(delta_cols.astype(np.int32).tobytes())
            
            chunks.append(values_sorted.tobytes())
        else:
            flat = indices.flatten().astype(np.int32)
            delta, sort_order = delta_encode_flat_int32(flat)
            values_sorted = values_np[sort_order]
            chunks.append(delta.tobytes())
            chunks.append(values_sorted.tobytes())
    return b"".join(chunks)


def load_production_delta_from_r2(checkpoint_window: int) -> dict | None:
    """Load production delta from R2 (actual weight values)."""
    import boto3
    
    R2_ACCOUNT_ID = "91561e574629960f78e985efa5a37e59"
    R2_READ_ACCESS_KEY_ID = "2a4e12f622668457a871d2e80de3439e"
    R2_READ_SECRET_ACCESS_KEY = "eea24b0551188a57a90bbe3b83de32880030b1108955022f3d78d7f33895c058"
    BUCKET_NAME = "91561e574629960f78e985efa5a37e59"
    
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_READ_ACCESS_KEY_ID,
        aws_secret_access_key=R2_READ_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    
    delta_key = f"grail/checkpoints/checkpoint-{checkpoint_window}/DELTA/delta_sparse.bin.zst"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=delta_key)
        compressed_bytes = response["Body"].read()
    except Exception as e:
        logger.warning(f"Failed to download: {e}")
        return None
    
    decompressor = zstd.ZstdDecompressor()
    uncompressed = decompressor.decompress(compressed_bytes)
    
    header_len = struct.unpack("<I", uncompressed[:4])[0]
    header = json.loads(uncompressed[4:4+header_len].decode("utf-8"))
    data_start = 4 + header_len
    
    layers = {}
    for t in header["tensors"]:
        name = t["name"]
        shape = t["shape"]
        value_dtype = t["value_dtype"]
        indices_offset = t["indices_offset"]
        indices_size = t["indices_size"]
        values_offset = t["values_offset"]
        values_size = t["values_size"]
        
        indices_bytes = uncompressed[data_start + indices_offset:data_start + indices_offset + indices_size]
        values_bytes = uncompressed[data_start + values_offset:data_start + values_offset + values_size]
        
        delta_indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
        sorted_indices = np.cumsum(delta_indices)
        
        if value_dtype == "bfloat16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.bfloat16)
        else:
            values_np = np.frombuffer(values_bytes, dtype=np.float32).copy()
            values = torch.from_numpy(values_np)
        
        cols_per_row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        rows = (sorted_indices // cols_per_row).astype(np.int32)
        cols = (sorted_indices % cols_per_row).astype(np.int32)
        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        
        layers[name] = {"indices": indices, "values": values, "shape": shape, "nnz": values.numel()}
    
    return {"layers": layers, "window": checkpoint_window}


def list_production_windows(limit: int = 10) -> list[int]:
    import boto3
    
    R2_ACCOUNT_ID = "91561e574629960f78e985efa5a37e59"
    R2_READ_ACCESS_KEY_ID = "2a4e12f622668457a871d2e80de3439e"
    R2_READ_SECRET_ACCESS_KEY = "eea24b0551188a57a90bbe3b83de32880030b1108955022f3d78d7f33895c058"
    BUCKET_NAME = "91561e574629960f78e985efa5a37e59"
    
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_READ_ACCESS_KEY_ID,
        aws_secret_access_key=R2_READ_SECRET_ACCESS_KEY,
        region_name="auto",
    )
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="grail/checkpoints/", Delimiter="/")
    
    windows = []
    for prefix_obj in response.get("CommonPrefixes", []):
        prefix_str = prefix_obj["Prefix"]
        if "checkpoint-" in prefix_str:
            try:
                window = int(prefix_str.split("checkpoint-")[1].split("/")[0])
                windows.append(window)
            except (IndexError, ValueError):
                continue
    
    windows.sort(reverse=True)
    return windows[:limit]


def reconstruct_actual_weights_from_delta(
    delta_layers: dict[str, dict[str, Any]],
    base_model_path: str,
) -> dict[str, dict[str, Any]]:
    """
    Reconstruct actual weight values from base model + delta.
    
    This simulates what production does:
    - Production stores: values = current_weights[changed_positions]
    - We compute: actual_weights = base_weights + delta_values
    - Then extract: values = actual_weights[changed_positions]
    
    Args:
        delta_layers: Dict of layer_name -> {indices, values (deltas), shape}
        base_model_path: Path to base model (HuggingFace format)
    
    Returns:
        layers dict with actual weight values instead of deltas
    """
    from transformers import AutoModelForCausalLM
    
    logger.info(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_state_dict = base_model.state_dict()
    del base_model
    gc.collect()
    
    reconstructed_layers = {}
    
    for layer_name, layer_data in delta_layers.items():
        indices = layer_data["indices"]
        delta_values = layer_data["values"]  # These are W_new - W_old
        shape = layer_data["shape"]
        
        # Find matching base weight
        base_weight = None
        for base_name, base_tensor in base_state_dict.items():
            if base_name == layer_name or base_name.replace("model.", "") == layer_name:
                base_weight = base_tensor
                break
        
        if base_weight is None:
            # Try partial match
            layer_suffix = layer_name.split(".")[-2:]  # e.g., ["q_proj", "weight"]
            for base_name, base_tensor in base_state_dict.items():
                if all(s in base_name for s in layer_suffix):
                    base_weight = base_tensor
                    break
        
        if base_weight is None:
            logger.warning(f"  Could not find base weight for {layer_name}, using deltas as-is")
            reconstructed_layers[layer_name] = layer_data
            continue
        
        # Flatten base weight to match delta indexing
        base_flat = base_weight.view(-1)
        
        # Convert COO indices to flat indices
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows, cols = indices[0].numpy(), indices[1].numpy()
            cols_per_row = shape[1] if len(shape) > 1 else 1
            flat_indices = rows.astype(np.int64) * cols_per_row + cols
        else:
            flat_indices = indices.numpy().flatten()
        
        # Get base values at changed positions
        flat_indices_tensor = torch.from_numpy(flat_indices).long()
        base_values_at_positions = base_flat[flat_indices_tensor]
        
        # Compute actual weights: W_new = W_base + delta
        actual_values = base_values_at_positions + delta_values.to(base_values_at_positions.dtype)
        
        reconstructed_layers[layer_name] = {
            "indices": indices,
            "values": actual_values.to(torch.bfloat16),
            "shape": shape,
            "nnz": actual_values.numel(),
        }
    
    return reconstructed_layers


def compute_coo_baseline(layers: dict) -> int:
    total = 0
    for ld in layers.values():
        values = ld["values"]
        nnz = values.numel()
        val_bytes = 2 if values.dtype in [torch.bfloat16, torch.float16] else 4
        total += nnz * (2 * 4 + val_bytes)
    return total


def benchmark_layers(layers: dict, experiment: str, source_type: str, algorithms: list[str]) -> list[dict]:
    results = []
    
    num_layers = len(layers)
    total_nnz = sum(ld["values"].numel() for ld in layers.values())
    total_params = sum(int(np.prod(ld["shape"])) for ld in layers.values())
    sparsity = 1.0 - (total_nnz / total_params) if total_params > 0 else 0.0
    coo_baseline = compute_coo_baseline(layers)
    
    base = {
        "experiment": experiment,
        "source_type": source_type,
        "num_layers": num_layers,
        "num_elements": total_nnz,
        "total_params": total_params,
        "sparsity": sparsity,
        "coo_baseline_bytes": coo_baseline,
    }
    
    try:
        data = build_delta_coo_downscaled(layers)
        uncomp_size = len(data)
        
        for algo in algorithms:
            try:
                comp, enc_ms, dec_ms, verified = benchmark_compression(data, algo)
                ratio = coo_baseline / len(comp) if comp else 0
                enc_tp = (uncomp_size / 1e6) / (enc_ms / 1000) if enc_ms > 0 else 0
                dec_tp = (uncomp_size / 1e6) / (dec_ms / 1000) if dec_ms > 0 else 0
                
                results.append(asdict(BenchmarkResult(
                    **base,
                    representation="delta_coo_downscaled",
                    algorithm=algo,
                    uncompressed_size_bytes=uncomp_size,
                    compressed_size_bytes=len(comp),
                    compression_ratio=ratio,
                    encode_time_ms=enc_ms,
                    decode_time_ms=dec_ms,
                    throughput_encode_mb_s=enc_tp,
                    throughput_decode_mb_s=dec_tp,
                    verified_correct=verified,
                )))
            except Exception as e:
                logger.warning(f"Error compressing with {algo}: {e}")
    except Exception as e:
        logger.warning(f"Error building representation: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Production-accurate compression benchmark")
    parser.add_argument("--output", type=str, default="/root/grail/data/compression_benchmark_production.csv")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--include-research", action="store_true", help="Include research delta files")
    parser.add_argument("--reconstruct-weights", action="store_true", 
                       help="Reconstruct actual weights from base model + deltas for research files")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B",
                       help="Base model for weight reconstruction")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Compression Benchmark - Production Accurate")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Libraries: {LIBRARY_VERSIONS}")
    if args.reconstruct_weights:
        logger.info(f"Reconstructing weights from base model: {args.base_model}")
    
    algorithms = get_available_algorithms()
    logger.info(f"Algorithms: {algorithms}")
    
    all_results = []
    
    # Production checkpoints (already contain actual weights)
    logger.info("\n--- PRODUCTION CHECKPOINTS (actual weight values) ---")
    windows = list_production_windows(args.samples * 2)
    logger.info(f"Found {len(windows)} checkpoints")
    
    count = 0
    for window in windows:
        if count >= args.samples:
            break
        
        logger.info(f"Loading checkpoint-{window}...")
        data = load_production_delta_from_r2(window)
        if data is None:
            continue
        
        layers = data["layers"]
        if not layers:
            continue
        
        results = benchmark_layers(layers, f"production-{window}", "production", algorithms)
        all_results.extend(results)
        
        best = next((r for r in results if r["algorithm"] == "zstd-1"), None)
        if best:
            logger.info(f"  ✓ ratio={best['compression_ratio']:.2f}x, nnz={best['num_elements']:,}")
        count += 1
    
    # Research delta files
    if args.include_research:
        research_path = Path("/root/grail/research/sparsity_analysis/experiments")
        if research_path.exists():
            exp_dirs = [d for d in research_path.iterdir() if d.is_dir() and "7b" in d.name.lower()]
            
            if args.reconstruct_weights:
                logger.info("\n--- RESEARCH FILES (reconstructed actual weights) ---")
            else:
                logger.info("\n--- RESEARCH DELTA FILES (raw deltas, for comparison) ---")
            
            for exp_dir in exp_dirs[:1]:
                delta_files = sorted(exp_dir.glob("**/delta_*.pt"))[:3]
                
                for delta_path in delta_files:
                    try:
                        delta = torch.load(delta_path, map_location="cpu", weights_only=True)
                        layers = delta.get("layers", {})
                        if not layers:
                            continue
                        
                        source_type = "research_delta"
                        
                        if args.reconstruct_weights:
                            # Reconstruct actual weights from base + delta
                            layers = reconstruct_actual_weights_from_delta(layers, args.base_model)
                            source_type = "research_reconstructed"
                        
                        results = benchmark_layers(layers, f"research-{exp_dir.name}", source_type, algorithms)
                        all_results.extend(results)
                        
                        best = next((r for r in results if r["algorithm"] == "zstd-1"), None)
                        if best:
                            logger.info(f"  ✓ {delta_path.name}: ratio={best['compression_ratio']:.2f}x")
                    except Exception as e:
                        logger.warning(f"Error processing {delta_path.name}: {e}")
    
    # Save results
    if all_results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"\nSaved {len(all_results)} results to {output_path}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY (delta_coo_downscaled + zstd-1)")
        logger.info("=" * 80)
        
        for source in ["production", "research_delta", "research_reconstructed"]:
            src_results = [r for r in all_results 
                          if r["source_type"] == source 
                          and r["algorithm"] == "zstd-1"]
            if src_results:
                avg_ratio = sum(r["compression_ratio"] for r in src_results) / len(src_results)
                logger.info(f"  {source:25s}: {avg_ratio:.2f}x ratio ({len(src_results)} samples)")


if __name__ == "__main__":
    main()
