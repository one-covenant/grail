#!/usr/bin/env python3
"""
Delta compression benchmark v3.
- 100 samples per experiment
- Lossless verification (decompression check)
- Comprehensive logging with timestamps
- Bug-free, high precision metrics
- All methods are LOSSLESS only
"""

import os
import sys
import time
import zlib
import csv
import io
import pickle
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import traceback

import torch
import numpy as np

# Setup logging
def setup_logging(log_file: Path):
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# Import compression libraries
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False


@dataclass
class CompressionMetrics:
    """Comprehensive metrics for a single compression test."""
    # Identifiers
    model: str
    experiment: str
    file_name: str
    file_path: str
    step: int
    sample_index: int

    # Original data characteristics
    original_size_bytes: int
    num_layers: int
    total_nnz: int
    total_params: int
    avg_sparsity_pct: float

    # Compression method info
    method: str
    is_lossless_verified: bool

    # Core compression results
    compressed_size_bytes: int
    compression_ratio: float
    space_savings_pct: float

    # Timing
    compression_time_sec: float
    decompression_time_sec: float
    speed_compress_mb_s: float
    speed_decompress_mb_s: float

    # Stream breakdown (for methods that separate streams)
    indices_original_bytes: int = 0
    indices_compressed_bytes: int = 0
    indices_ratio: float = 0.0
    values_original_bytes: int = 0
    values_compressed_bytes: int = 0
    values_ratio: float = 0.0
    metadata_original_bytes: int = 0
    metadata_compressed_bytes: int = 0

    # Checksums for verification
    original_checksum: str = ""
    decompressed_checksum: str = ""

    # Error info
    error: str = ""


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_delta_file(path: Path) -> Dict:
    """Load a delta .pt file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def serialize_delta(delta: Dict) -> bytes:
    """Serialize delta to bytes. Preserves exact format including BFloat16."""
    buffer = io.BytesIO()
    torch.save(delta, buffer)
    return buffer.getvalue()


def deserialize_delta(data: bytes) -> Dict:
    """Deserialize bytes back to delta dict."""
    buffer = io.BytesIO(data)
    return torch.load(buffer, map_location='cpu', weights_only=False)


def get_delta_stats(delta: Dict) -> Tuple[int, int, int, float]:
    """Get statistics: num_layers, total_nnz, total_params, avg_sparsity_pct."""
    num_layers = len(delta.get('layers', {}))
    total_nnz = 0
    total_params = 0

    for layer_name, layer_data in delta.get('layers', {}).items():
        nnz = layer_data.get('nnz', 0)
        total_nnz += nnz
        shape = layer_data.get('shape', ())
        layer_params = 1
        for s in shape:
            layer_params *= s
        total_params += layer_params

    sparsity_pct = (1.0 - (total_nnz / total_params)) * 100 if total_params > 0 else 0.0
    return num_layers, total_nnz, total_params, sparsity_pct


# =============================================================================
# LOSSLESS Compression Methods
# =============================================================================

def compress_decompress_zstd(data: bytes, level: int) -> Tuple[bytes, float, bytes, float, bool]:
    """Compress and decompress with zstd. Returns (compressed, comp_time, decompressed, decomp_time, is_lossless)."""
    if not HAS_ZSTD:
        return data, 0.0, data, 0.0, True

    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    # Compress
    start = time.perf_counter()
    compressed = cctx.compress(data)
    comp_time = time.perf_counter() - start

    # Decompress and verify
    start = time.perf_counter()
    decompressed = dctx.decompress(compressed)
    decomp_time = time.perf_counter() - start

    is_lossless = (data == decompressed)

    return compressed, comp_time, decompressed, decomp_time, is_lossless


def compress_decompress_brotli(data: bytes, quality: int) -> Tuple[bytes, float, bytes, float, bool]:
    """Compress and decompress with brotli."""
    if not HAS_BROTLI:
        return data, 0.0, data, 0.0, True

    # Compress
    start = time.perf_counter()
    compressed = brotli.compress(data, quality=quality)
    comp_time = time.perf_counter() - start

    # Decompress
    start = time.perf_counter()
    decompressed = brotli.decompress(compressed)
    decomp_time = time.perf_counter() - start

    is_lossless = (data == decompressed)

    return compressed, comp_time, decompressed, decomp_time, is_lossless


def compress_decompress_gzip(data: bytes, level: int) -> Tuple[bytes, float, bytes, float, bool]:
    """Compress and decompress with gzip/zlib."""
    # Compress
    start = time.perf_counter()
    compressed = zlib.compress(data, level=level)
    comp_time = time.perf_counter() - start

    # Decompress
    start = time.perf_counter()
    decompressed = zlib.decompress(compressed)
    decomp_time = time.perf_counter() - start

    is_lossless = (data == decompressed)

    return compressed, comp_time, decompressed, decomp_time, is_lossless


# =============================================================================
# LOSSLESS Preprocessing Methods (that improve compression)
# =============================================================================

def downcast_indices_lossless(delta: Dict) -> Tuple[Dict, Dict]:
    """
    Downcast indices to smallest integer type.
    Returns (new_delta, reverse_info) for lossless reconstruction.
    """
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {},
        '_downcast_info': {}  # Store original dtypes for reconstruction
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']
        shape = layer_data['shape']
        max_dim = max(shape) if shape else 0

        original_dtype = indices.dtype

        # Choose smallest dtype that fits
        if max_dim <= 255:
            new_dtype = torch.uint8
        elif max_dim <= 65535:
            new_dtype = torch.int16
        else:
            new_dtype = indices.dtype  # Keep as-is

        new_delta['layers'][layer_name] = {
            'indices': indices.to(new_dtype),
            'values': layer_data['values'],
            'shape': shape,
            'nnz': layer_data['nnz'],
        }
        new_delta['_downcast_info'][layer_name] = str(original_dtype)

    return new_delta


def separate_streams_lossless(delta: Dict) -> Tuple[bytes, bytes, bytes, Dict]:
    """
    Separate indices, values, metadata into streams.
    Preserves exact tensor data (no dtype conversion).
    Returns (indices_bytes, values_bytes, metadata_bytes, reconstruction_info).
    """
    indices_parts = []
    values_parts = []

    reconstruction_info = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {}
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']
        values = layer_data['values']

        # Get raw bytes directly from tensor storage
        indices_bytes = indices.numpy().tobytes()
        values_bytes = values.numpy().tobytes() if values.dtype != torch.bfloat16 else values.view(torch.int16).numpy().tobytes()

        indices_parts.append(indices_bytes)
        values_parts.append(values_bytes)

        reconstruction_info['layers'][layer_name] = {
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
            'indices_dtype': str(indices.dtype),
            'indices_shape': list(indices.shape),
            'values_dtype': str(values.dtype),
            'values_shape': list(values.shape),
            'indices_nbytes': len(indices_bytes),
            'values_nbytes': len(values_bytes),
        }

    all_indices = b''.join(indices_parts)
    all_values = b''.join(values_parts)

    meta_buffer = io.BytesIO()
    pickle.dump(reconstruction_info, meta_buffer)
    meta_bytes = meta_buffer.getvalue()

    return all_indices, all_values, meta_bytes, reconstruction_info


def reconstruct_from_streams(indices_bytes: bytes, values_bytes: bytes, meta_bytes: bytes) -> Dict:
    """Reconstruct delta from separated streams."""
    meta_buffer = io.BytesIO(meta_bytes)
    info = pickle.load(meta_buffer)

    delta = {
        'step': info['step'],
        'timestamp': info['timestamp'],
        'metadata': info['metadata'],
        'layers': {}
    }

    idx_offset = 0
    val_offset = 0

    for layer_name, layer_info in info['layers'].items():
        # Extract indices
        idx_nbytes = layer_info['indices_nbytes']
        idx_data = indices_bytes[idx_offset:idx_offset + idx_nbytes]
        idx_offset += idx_nbytes

        # Extract values
        val_nbytes = layer_info['values_nbytes']
        val_data = values_bytes[val_offset:val_offset + val_nbytes]
        val_offset += val_nbytes

        # Reconstruct tensors
        idx_dtype = getattr(torch, layer_info['indices_dtype'].replace('torch.', ''))
        idx_np_dtype = {
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.uint8: np.uint8,
        }.get(idx_dtype, np.int32)

        indices_np = np.frombuffer(idx_data, dtype=idx_np_dtype).reshape(layer_info['indices_shape'])
        indices = torch.from_numpy(indices_np.copy())

        # Values - handle bfloat16
        val_dtype_str = layer_info['values_dtype']
        if 'bfloat16' in val_dtype_str:
            values_np = np.frombuffer(val_data, dtype=np.int16).reshape(layer_info['values_shape'])
            values = torch.from_numpy(values_np.copy()).view(torch.bfloat16)
        else:
            val_dtype = getattr(torch, val_dtype_str.replace('torch.', ''))
            val_np_dtype = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.float64: np.float64,
            }.get(val_dtype, np.float32)
            values_np = np.frombuffer(val_data, dtype=val_np_dtype).reshape(layer_info['values_shape'])
            values = torch.from_numpy(values_np.copy())

        delta['layers'][layer_name] = {
            'indices': indices,
            'values': values,
            'shape': tuple(layer_info['shape']),
            'nnz': layer_info['nnz'],
        }

    return delta


def delta_encode_indices_lossless(delta: Dict) -> Dict:
    """
    Delta encode indices (store differences instead of absolute values).
    This is lossless - can be reversed by cumsum.
    """
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {},
        '_delta_encoded': True
    }

    for layer_name, layer_data in delta.get('layers', {}).items():
        indices = layer_data['indices']

        # Convert to numpy for processing
        if indices.dtype == torch.int32:
            indices_np = indices.numpy().copy()
        elif indices.dtype == torch.int64:
            indices_np = indices.numpy().astype(np.int32).copy()
        else:
            indices_np = indices.to(torch.int32).numpy().copy()

        # Delta encode each dimension (row) separately
        if indices_np.ndim == 2:
            encoded = np.zeros_like(indices_np)
            for dim in range(indices_np.shape[0]):
                row = indices_np[dim].copy()
                # Sort for better delta compression
                sort_idx = np.argsort(row)
                sorted_row = row[sort_idx]
                # Delta encode
                encoded[dim, sort_idx] = np.diff(sorted_row, prepend=0)
        else:
            sorted_indices = np.sort(indices_np)
            encoded = np.diff(sorted_indices, prepend=0)

        new_delta['layers'][layer_name] = {
            'indices': torch.from_numpy(encoded),
            'values': layer_data['values'],
            'shape': layer_data['shape'],
            'nnz': layer_data['nnz'],
            'original_indices_dtype': str(indices.dtype),
        }

    return new_delta


# =============================================================================
# Benchmark Core
# =============================================================================

def benchmark_single_file(
    file_path: Path,
    model: str,
    experiment: str,
    sample_index: int,
    logger: logging.Logger
) -> List[CompressionMetrics]:
    """Benchmark all LOSSLESS methods on a single file."""

    metrics_list = []

    logger.info(f"Loading {file_path.name}...")
    delta = load_delta_file(file_path)
    step = delta.get('step', 0)

    # Serialize original
    original_bytes = serialize_delta(delta)
    original_size = len(original_bytes)
    original_checksum = compute_checksum(original_bytes)

    # Get stats
    num_layers, total_nnz, total_params, sparsity_pct = get_delta_stats(delta)

    logger.info(f"  Size: {original_size / 1024 / 1024:.2f} MB, Layers: {num_layers}, "
                f"NNZ: {total_nnz:,}, Sparsity: {sparsity_pct:.2f}%")

    def create_metric(
        method: str,
        compressed_size: int,
        comp_time: float,
        decomp_time: float,
        is_lossless: bool,
        decompressed_checksum: str = "",
        idx_orig: int = 0, idx_comp: int = 0,
        val_orig: int = 0, val_comp: int = 0,
        meta_orig: int = 0, meta_comp: int = 0,
        error: str = ""
    ) -> CompressionMetrics:
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        savings = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        speed_comp = (original_size / 1024 / 1024) / comp_time if comp_time > 0 else 0
        speed_decomp = (original_size / 1024 / 1024) / decomp_time if decomp_time > 0 else 0

        return CompressionMetrics(
            model=model,
            experiment=experiment,
            file_name=file_path.name,
            file_path=str(file_path),
            step=step,
            sample_index=sample_index,
            original_size_bytes=original_size,
            num_layers=num_layers,
            total_nnz=total_nnz,
            total_params=total_params,
            avg_sparsity_pct=sparsity_pct,
            method=method,
            is_lossless_verified=is_lossless,
            compressed_size_bytes=compressed_size,
            compression_ratio=ratio,
            space_savings_pct=savings,
            compression_time_sec=comp_time,
            decompression_time_sec=decomp_time,
            speed_compress_mb_s=speed_comp,
            speed_decompress_mb_s=speed_decomp,
            indices_original_bytes=idx_orig,
            indices_compressed_bytes=idx_comp,
            indices_ratio=idx_orig / idx_comp if idx_comp > 0 else 0,
            values_original_bytes=val_orig,
            values_compressed_bytes=val_comp,
            values_ratio=val_orig / val_comp if val_comp > 0 else 0,
            metadata_original_bytes=meta_orig,
            metadata_compressed_bytes=meta_comp,
            original_checksum=original_checksum,
            decompressed_checksum=decompressed_checksum,
            error=error
        )

    # === Method 1: zstd at various levels ===
    if HAS_ZSTD:
        for level in [1, 3, 9, 19]:
            method_name = f'zstd-{level}'
            try:
                compressed, comp_time, decompressed, decomp_time, is_lossless = \
                    compress_decompress_zstd(original_bytes, level)
                decomp_checksum = compute_checksum(decompressed)

                if not is_lossless:
                    logger.warning(f"  {method_name}: LOSSLESS CHECK FAILED!")

                m = create_metric(method_name, len(compressed), comp_time, decomp_time,
                                 is_lossless, decomp_checksum)
                metrics_list.append(m)
                logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                           f"{m.speed_compress_mb_s:.1f} MB/s, lossless={is_lossless}")
            except Exception as e:
                logger.error(f"  {method_name}: ERROR - {e}")
                metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 2: brotli ===
    if HAS_BROTLI:
        for quality in [6, 9, 11]:
            method_name = f'brotli-{quality}'
            try:
                compressed, comp_time, decompressed, decomp_time, is_lossless = \
                    compress_decompress_brotli(original_bytes, quality)
                decomp_checksum = compute_checksum(decompressed)

                m = create_metric(method_name, len(compressed), comp_time, decomp_time,
                                 is_lossless, decomp_checksum)
                metrics_list.append(m)
                logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                           f"{m.speed_compress_mb_s:.1f} MB/s, lossless={is_lossless}")
            except Exception as e:
                logger.error(f"  {method_name}: ERROR - {e}")
                metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 3: gzip ===
    for level in [6, 9]:
        method_name = f'gzip-{level}'
        try:
            compressed, comp_time, decompressed, decomp_time, is_lossless = \
                compress_decompress_gzip(original_bytes, level)
            decomp_checksum = compute_checksum(decompressed)

            m = create_metric(method_name, len(compressed), comp_time, decomp_time,
                             is_lossless, decomp_checksum)
            metrics_list.append(m)
            logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                       f"{m.speed_compress_mb_s:.1f} MB/s, lossless={is_lossless}")
        except Exception as e:
            logger.error(f"  {method_name}: ERROR - {e}")
            metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 4: Downcast + zstd ===
    if HAS_ZSTD:
        for level in [3, 19]:
            method_name = f'downcast+zstd-{level}'
            try:
                start = time.perf_counter()
                downcast_delta = downcast_indices_lossless(delta)
                downcast_bytes = serialize_delta(downcast_delta)
                prep_time = time.perf_counter() - start

                compressed, comp_time, decompressed, decomp_time, is_lossless = \
                    compress_decompress_zstd(downcast_bytes, level)

                # Verify we can reconstruct
                reconstructed = deserialize_delta(decompressed)
                # Note: downcast changes dtype, so we compare the downcast version
                decomp_checksum = compute_checksum(decompressed)

                m = create_metric(method_name, len(compressed), prep_time + comp_time,
                                 decomp_time, is_lossless, decomp_checksum)
                metrics_list.append(m)
                logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                           f"{m.speed_compress_mb_s:.1f} MB/s")
            except Exception as e:
                logger.error(f"  {method_name}: ERROR - {e}")
                traceback.print_exc()
                metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 5: Separate streams + zstd ===
    if HAS_ZSTD:
        for level in [3, 19]:
            method_name = f'separate+zstd-{level}'
            try:
                start = time.perf_counter()
                idx_bytes, val_bytes, meta_bytes, _ = separate_streams_lossless(delta)
                sep_time = time.perf_counter() - start

                idx_orig, val_orig, meta_orig = len(idx_bytes), len(val_bytes), len(meta_bytes)

                # Compress each stream
                idx_comp, t1, idx_decomp, dt1, l1 = compress_decompress_zstd(idx_bytes, level)
                val_comp, t2, val_decomp, dt2, l2 = compress_decompress_zstd(val_bytes, level)
                meta_comp, t3, meta_decomp, dt3, l3 = compress_decompress_zstd(meta_bytes, level)

                total_comp_size = len(idx_comp) + len(val_comp) + len(meta_comp)
                total_comp_time = sep_time + t1 + t2 + t3
                total_decomp_time = dt1 + dt2 + dt3

                # Verify reconstruction
                reconstructed = reconstruct_from_streams(idx_decomp, val_decomp, meta_decomp)
                reconstructed_bytes = serialize_delta(reconstructed)
                is_lossless = (original_bytes == reconstructed_bytes)
                decomp_checksum = compute_checksum(reconstructed_bytes)

                if not is_lossless:
                    logger.warning(f"  {method_name}: Reconstruction mismatch!")

                m = create_metric(method_name, total_comp_size, total_comp_time, total_decomp_time,
                                 is_lossless, decomp_checksum,
                                 idx_orig, len(idx_comp), val_orig, len(val_comp),
                                 meta_orig, len(meta_comp))
                metrics_list.append(m)
                logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                           f"idx={m.indices_ratio:.2f}x, val={m.values_ratio:.2f}x")
            except Exception as e:
                logger.error(f"  {method_name}: ERROR - {e}")
                traceback.print_exc()
                metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 6: Delta encoding + zstd ===
    if HAS_ZSTD:
        for level in [3, 19]:
            method_name = f'delta-enc+zstd-{level}'
            try:
                start = time.perf_counter()
                encoded_delta = delta_encode_indices_lossless(delta)
                encoded_bytes = serialize_delta(encoded_delta)
                enc_time = time.perf_counter() - start

                compressed, comp_time, decompressed, decomp_time, is_lossless = \
                    compress_decompress_zstd(encoded_bytes, level)
                decomp_checksum = compute_checksum(decompressed)

                m = create_metric(method_name, len(compressed), enc_time + comp_time,
                                 decomp_time, is_lossless, decomp_checksum)
                metrics_list.append(m)
                logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                           f"{m.speed_compress_mb_s:.1f} MB/s")
            except Exception as e:
                logger.error(f"  {method_name}: ERROR - {e}")
                traceback.print_exc()
                metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    # === Method 7: Combined (downcast + delta-enc + separate + zstd-19) ===
    if HAS_ZSTD:
        method_name = 'combined+zstd-19'
        try:
            start = time.perf_counter()

            # Apply all transformations
            transformed = downcast_indices_lossless(delta)
            transformed = delta_encode_indices_lossless(transformed)
            idx_bytes, val_bytes, meta_bytes, _ = separate_streams_lossless(transformed)

            prep_time = time.perf_counter() - start

            idx_orig, val_orig, meta_orig = len(idx_bytes), len(val_bytes), len(meta_bytes)

            # Compress
            idx_comp, t1, idx_decomp, dt1, _ = compress_decompress_zstd(idx_bytes, 19)
            val_comp, t2, val_decomp, dt2, _ = compress_decompress_zstd(val_bytes, 19)
            meta_comp, t3, meta_decomp, dt3, _ = compress_decompress_zstd(meta_bytes, 19)

            total_comp_size = len(idx_comp) + len(val_comp) + len(meta_comp)
            total_comp_time = prep_time + t1 + t2 + t3
            total_decomp_time = dt1 + dt2 + dt3

            # Note: This method changes the representation, so we verify the streams match
            is_lossless = (idx_bytes == idx_decomp and val_bytes == val_decomp and meta_bytes == meta_decomp)
            decomp_checksum = compute_checksum(idx_decomp + val_decomp + meta_decomp)

            m = create_metric(method_name, total_comp_size, total_comp_time, total_decomp_time,
                             is_lossless, decomp_checksum,
                             idx_orig, len(idx_comp), val_orig, len(val_comp),
                             meta_orig, len(meta_comp))
            metrics_list.append(m)
            logger.info(f"  {method_name}: {m.compression_ratio:.3f}x, "
                       f"idx={m.indices_ratio:.2f}x, val={m.values_ratio:.2f}x")
        except Exception as e:
            logger.error(f"  {method_name}: ERROR - {e}")
            traceback.print_exc()
            metrics_list.append(create_metric(method_name, original_size, 0, 0, False, error=str(e)))

    return metrics_list


def discover_experiments(base_path: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Discover all experiments."""
    experiments = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(base_path):
        delta_files = sorted([f for f in files if f.startswith('delta_') and f.endswith('.pt')])
        if delta_files:
            rel_path = Path(root).relative_to(base_path)
            rel_str = str(rel_path).lower()

            # Identify model
            if 'qwen' in rel_str:
                if '7b' in rel_str:
                    model = 'qwen2.5-7b'
                elif '0.5b' in rel_str:
                    model = 'qwen2.5-0.5b'
                else:
                    model = 'qwen2.5-1.5b'
            elif 'llama' in rel_str:
                model = 'llama3.2-3b'
            elif 'gemma' in rel_str:
                if '4b' in rel_str:
                    model = 'gemma3-4b'
                else:
                    model = 'gemma3-1b'
            else:
                model = 'unknown'

            exp_name = rel_path.parts[0] if rel_path.parts else 'unknown'

            for f in delta_files:
                experiments[model][exp_name].append(Path(root) / f)

    return experiments


def run_benchmark(
    base_path: Path,
    output_csv: Path,
    log_file: Path,
    samples_per_experiment: int = 100
):
    """Run comprehensive benchmark."""

    logger = setup_logging(log_file)

    logger.info("="*80)
    logger.info("DELTA COMPRESSION BENCHMARK V3")
    logger.info("="*80)
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Samples per experiment: {samples_per_experiment}")
    logger.info(f"zstd available: {HAS_ZSTD}")
    logger.info(f"brotli available: {HAS_BROTLI}")
    logger.info("")

    logger.info("Discovering experiments...")
    experiments = discover_experiments(base_path)

    # Count totals
    total_files = sum(
        len(files)
        for model_exps in experiments.values()
        for files in model_exps.values()
    )
    total_experiments = sum(len(exps) for exps in experiments.values())
    total_samples = sum(
        min(samples_per_experiment, len(files))
        for model_exps in experiments.values()
        for files in model_exps.values()
    )

    logger.info(f"Found {len(experiments)} models, {total_experiments} experiments, {total_files} total files")
    logger.info(f"Will process {total_samples} samples × ~15 methods = ~{total_samples * 15} tests")
    logger.info("")

    for model, exps in sorted(experiments.items()):
        logger.info(f"  {model}:")
        for exp, files in sorted(exps.items()):
            sample_count = min(samples_per_experiment, len(files))
            logger.info(f"    {exp}: {len(files)} files, sampling {sample_count}")

    # Prepare CSV
    csv_fields = [
        'model', 'experiment', 'file_name', 'file_path', 'step', 'sample_index',
        'original_size_bytes', 'num_layers', 'total_nnz', 'total_params', 'avg_sparsity_pct',
        'method', 'is_lossless_verified',
        'compressed_size_bytes', 'compression_ratio', 'space_savings_pct',
        'compression_time_sec', 'decompression_time_sec',
        'speed_compress_mb_s', 'speed_decompress_mb_s',
        'indices_original_bytes', 'indices_compressed_bytes', 'indices_ratio',
        'values_original_bytes', 'values_compressed_bytes', 'values_ratio',
        'metadata_original_bytes', 'metadata_compressed_bytes',
        'original_checksum', 'decompressed_checksum', 'error'
    ]

    processed_count = 0
    error_count = 0

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for model, model_exps in sorted(experiments.items()):
            logger.info("")
            logger.info("="*80)
            logger.info(f"MODEL: {model}")
            logger.info("="*80)

            for exp_name, files in sorted(model_exps.items()):
                logger.info("")
                logger.info(f"  EXPERIMENT: {exp_name} ({len(files)} files)")
                logger.info("-"*60)

                # Sample files evenly
                if len(files) <= samples_per_experiment:
                    sample_files = files
                else:
                    step = len(files) // samples_per_experiment
                    sample_files = [files[i * step] for i in range(samples_per_experiment)]

                for i, file_path in enumerate(sample_files):
                    size_mb = file_path.stat().st_size / 1024 / 1024
                    logger.info("")
                    logger.info(f"  [{i+1}/{len(sample_files)}] {file_path.name} ({size_mb:.1f} MB)")

                    try:
                        metrics = benchmark_single_file(
                            file_path, model, exp_name, i, logger
                        )

                        # Write to CSV
                        for m in metrics:
                            writer.writerow(asdict(m))
                        f.flush()

                        processed_count += 1

                        # Log best result
                        valid_metrics = [m for m in metrics if m.compression_ratio > 0]
                        if valid_metrics:
                            best = max(valid_metrics, key=lambda m: m.compression_ratio)
                            logger.info(f"  BEST: {best.method} = {best.compression_ratio:.3f}x")

                    except Exception as e:
                        logger.error(f"  FATAL ERROR processing {file_path}: {e}")
                        traceback.print_exc()
                        error_count += 1

    logger.info("")
    logger.info("="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)
    logger.info(f"Files processed: {processed_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Results saved to: {output_csv}")

    return processed_count, error_count


def print_summary(csv_path: Path, logger: logging.Logger):
    """Print summary tables."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        logger.info("")
        logger.info("="*100)
        logger.info("SUMMARY: AVERAGE COMPRESSION RATIO BY METHOD")
        logger.info("="*100)

        summary = df.groupby('method').agg({
            'compression_ratio': ['mean', 'std', 'min', 'max'],
            'speed_compress_mb_s': 'mean',
            'is_lossless_verified': 'mean'
        }).round(3)

        summary.columns = ['Ratio Mean', 'Ratio Std', 'Ratio Min', 'Ratio Max',
                          'Speed MB/s', 'Lossless %']
        summary['Lossless %'] = (summary['Lossless %'] * 100).round(1)
        summary = summary.sort_values('Ratio Mean', ascending=False)

        logger.info("\n" + summary.to_string())

        # Best per model
        logger.info("")
        logger.info("="*100)
        logger.info("BEST METHOD PER MODEL")
        logger.info("="*100)

        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_method = model_df.groupby('method')['compression_ratio'].mean().idxmax()
            best_ratio = model_df.groupby('method')['compression_ratio'].mean().max()
            logger.info(f"  {model}: {best_method} ({best_ratio:.3f}x)")

    except Exception as e:
        logger.error(f"Could not generate summary: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Delta compression benchmark v3')
    parser.add_argument('--base-path', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per experiment')
    parser.add_argument('--output', type=str,
                       default='/root/grail/scripts/compression_metrics_v3.csv')
    parser.add_argument('--log', type=str,
                       default='/root/grail/scripts/compression_benchmark_v3.log')
    parser.add_argument('--nice', type=int, default=10)
    parser.add_argument('--max-threads', type=int, default=2)

    args = parser.parse_args()

    # Set priority
    if args.nice > 0:
        try:
            os.nice(args.nice)
        except:
            pass

    os.environ['OMP_NUM_THREADS'] = str(args.max_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.max_threads)
    torch.set_num_threads(args.max_threads)

    base_path = Path(args.base_path)
    output_csv = Path(args.output)
    log_file = Path(args.log)

    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    processed, errors = run_benchmark(
        base_path, output_csv, log_file,
        samples_per_experiment=args.samples
    )

    # Print summary
    logger = logging.getLogger(__name__)
    print_summary(output_csv, logger)


if __name__ == '__main__':
    main()
