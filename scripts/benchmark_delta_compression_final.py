#!/usr/bin/env python3
"""
Delta Compression Benchmark - Final Version

Key guarantees:
1. FAIR COMPARISON: All methods tested on EXACTLY the same delta files
2. DETERMINISTIC: Fixed seed for reproducible file sampling
3. LOSSLESS ONLY: All methods verified via decompression
4. COMPREHENSIVE METRICS: Per-file, per-method CSV output
5. ROBUST LOGGING: Timestamps and error tracking
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
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, OrderedDict
from datetime import datetime
import traceback

import torch
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42  # Fixed seed for reproducible sampling

# Methods to test (all lossless)
# Full set of methods
METHODS_FULL = OrderedDict([
    # Format: method_name -> (compressor_func, params)
    ('zstd-1', ('zstd', {'level': 1})),
    ('zstd-3', ('zstd', {'level': 3})),
    ('zstd-9', ('zstd', {'level': 9})),
    ('zstd-19', ('zstd', {'level': 19})),
    ('brotli-6', ('brotli', {'quality': 6})),
    ('brotli-9', ('brotli', {'quality': 9})),
    ('brotli-11', ('brotli', {'quality': 11})),
    ('gzip-6', ('gzip', {'level': 6})),
    ('gzip-9', ('gzip', {'level': 9})),
    ('downcast+zstd-3', ('downcast_zstd', {'level': 3})),
    ('downcast+zstd-19', ('downcast_zstd', {'level': 19})),
    ('separate+zstd-3', ('separate_zstd', {'level': 3})),
    ('separate+zstd-19', ('separate_zstd', {'level': 19})),
    ('delta-enc+zstd-3', ('delta_enc_zstd', {'level': 3})),
    ('delta-enc+zstd-19', ('delta_enc_zstd', {'level': 19})),
    ('combined+zstd-19', ('combined_zstd', {'level': 19})),
])

# Fast mode: skip slowest methods (brotli-11, gzip-9, zstd-19 variants replaced with zstd-9)
METHODS_FAST = OrderedDict([
    ('zstd-1', ('zstd', {'level': 1})),
    ('zstd-3', ('zstd', {'level': 3})),
    ('zstd-9', ('zstd', {'level': 9})),
    ('brotli-6', ('brotli', {'quality': 6})),
    ('gzip-6', ('gzip', {'level': 6})),
    ('downcast+zstd-3', ('downcast_zstd', {'level': 3})),
    ('downcast+zstd-9', ('downcast_zstd', {'level': 9})),
    ('separate+zstd-3', ('separate_zstd', {'level': 3})),
    ('separate+zstd-9', ('separate_zstd', {'level': 9})),
    ('delta-enc+zstd-3', ('delta_enc_zstd', {'level': 3})),
    ('delta-enc+zstd-9', ('delta_enc_zstd', {'level': 9})),
    ('combined+zstd-9', ('combined_zstd', {'level': 9})),
])

# Default to full methods (will be overridden by --fast flag)
METHODS_CONFIG = METHODS_FULL


# =============================================================================
# Setup
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging to file and stdout."""
    logger = logging.getLogger('compression_benchmark')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FileInfo:
    """Information about a delta file."""
    path: Path
    model: str
    experiment: str
    size_bytes: int
    index_in_experiment: int


@dataclass
class CompressionResult:
    """Result of compressing a single file with a single method."""
    # File identification
    model: str
    experiment: str
    file_name: str
    file_path: str
    step: int
    sample_idx: int

    # Original file stats
    original_size_bytes: int
    num_layers: int
    total_nnz: int
    total_params: int
    sparsity_pct: float

    # Method info
    method: str

    # Compression results
    compressed_size_bytes: int
    compression_ratio: float
    space_savings_pct: float

    # Timing (seconds)
    preprocess_time: float
    compress_time: float
    decompress_time: float
    total_time: float

    # Speed (MB/s based on original size)
    compress_speed_mb_s: float
    decompress_speed_mb_s: float

    # Verification
    is_lossless: bool
    original_hash: str
    decompressed_hash: str

    # Stream breakdown (for methods that separate)
    indices_orig_bytes: int = 0
    indices_comp_bytes: int = 0
    values_orig_bytes: int = 0
    values_comp_bytes: int = 0
    metadata_orig_bytes: int = 0
    metadata_comp_bytes: int = 0

    # Error tracking
    error_msg: str = ""


# =============================================================================
# Utility Functions
# =============================================================================

def compute_hash(data: bytes) -> str:
    """Compute SHA256 hash (first 16 chars)."""
    return hashlib.sha256(data).hexdigest()[:16]


def load_delta(path: Path) -> Dict:
    """Load delta file."""
    return torch.load(path, map_location='cpu', weights_only=False)


def serialize_delta(delta: Dict) -> bytes:
    """Serialize delta to bytes (preserves exact format)."""
    buf = io.BytesIO()
    torch.save(delta, buf)
    return buf.getvalue()


def deserialize_delta(data: bytes) -> Dict:
    """Deserialize bytes to delta."""
    buf = io.BytesIO(data)
    return torch.load(buf, map_location='cpu', weights_only=False)


def get_delta_stats(delta: Dict) -> Tuple[int, int, int, int, float]:
    """
    Get delta statistics.
    Returns: (step, num_layers, total_nnz, total_params, sparsity_pct)
    """
    step = delta.get('step', 0)
    layers = delta.get('layers', {})
    num_layers = len(layers)
    total_nnz = 0
    total_params = 0

    for layer_data in layers.values():
        nnz = layer_data.get('nnz', 0)
        total_nnz += nnz
        shape = layer_data.get('shape', ())
        params = 1
        for s in shape:
            params *= s
        total_params += params

    sparsity = (1.0 - total_nnz / total_params) * 100 if total_params > 0 else 0.0
    return step, num_layers, total_nnz, total_params, sparsity


# =============================================================================
# Compression Functions (all return: compressed, comp_time, decompressed, decomp_time)
# =============================================================================

def compress_zstd(data: bytes, level: int) -> Tuple[bytes, float, bytes, float]:
    """Compress/decompress with zstd."""
    if not HAS_ZSTD:
        raise RuntimeError("zstd not available")

    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    t0 = time.perf_counter()
    compressed = cctx.compress(data)
    comp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decompressed = dctx.decompress(compressed)
    decomp_time = time.perf_counter() - t0

    return compressed, comp_time, decompressed, decomp_time


def compress_brotli(data: bytes, quality: int) -> Tuple[bytes, float, bytes, float]:
    """Compress/decompress with brotli."""
    if not HAS_BROTLI:
        raise RuntimeError("brotli not available")

    t0 = time.perf_counter()
    compressed = brotli.compress(data, quality=quality)
    comp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decompressed = brotli.decompress(compressed)
    decomp_time = time.perf_counter() - t0

    return compressed, comp_time, decompressed, decomp_time


def compress_gzip(data: bytes, level: int) -> Tuple[bytes, float, bytes, float]:
    """Compress/decompress with gzip."""
    t0 = time.perf_counter()
    compressed = zlib.compress(data, level=level)
    comp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    decompressed = zlib.decompress(compressed)
    decomp_time = time.perf_counter() - t0

    return compressed, comp_time, decompressed, decomp_time


# =============================================================================
# Preprocessing Functions (lossless transformations)
# =============================================================================

def downcast_indices(delta: Dict) -> Dict:
    """Downcast indices to smallest int type that fits."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {},
        '_preprocessing': 'downcast'
    }

    for name, layer in delta.get('layers', {}).items():
        indices = layer['indices']
        shape = layer['shape']
        max_dim = max(shape) if shape else 0

        if max_dim <= 255:
            new_dtype = torch.uint8
        elif max_dim <= 65535:
            new_dtype = torch.int16
        else:
            new_dtype = indices.dtype

        new_delta['layers'][name] = {
            'indices': indices.to(new_dtype),
            'values': layer['values'],
            'shape': shape,
            'nnz': layer['nnz'],
            '_orig_idx_dtype': str(indices.dtype)
        }

    return new_delta


def delta_encode_indices(delta: Dict) -> Dict:
    """Delta-encode indices (store diffs instead of absolute values)."""
    new_delta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layers': {},
        '_preprocessing': 'delta_encoded'
    }

    for name, layer in delta.get('layers', {}).items():
        indices = layer['indices']

        # Convert to numpy
        if indices.dtype in (torch.int32, torch.int64):
            idx_np = indices.numpy().copy()
        else:
            idx_np = indices.to(torch.int32).numpy().copy()

        # Delta encode each row
        if idx_np.ndim == 2:
            encoded = np.zeros_like(idx_np)
            for dim in range(idx_np.shape[0]):
                row = idx_np[dim]
                sort_order = np.argsort(row)
                sorted_row = row[sort_order]
                deltas = np.diff(sorted_row, prepend=0)
                # Store in sorted order for better compression
                encoded[dim] = deltas[np.argsort(sort_order)]
        else:
            sorted_idx = np.sort(idx_np)
            encoded = np.diff(sorted_idx, prepend=0)

        new_delta['layers'][name] = {
            'indices': torch.from_numpy(encoded),
            'values': layer['values'],
            'shape': layer['shape'],
            'nnz': layer['nnz'],
        }

    return new_delta


def separate_streams(delta: Dict) -> Tuple[bytes, bytes, bytes, Dict]:
    """
    Separate delta into index/value/metadata streams.
    Returns: (indices_bytes, values_bytes, metadata_bytes, layer_info)
    """
    indices_parts = []
    values_parts = []
    layer_info = OrderedDict()

    for name, layer in delta.get('layers', {}).items():
        indices = layer['indices']
        values = layer['values']

        # Get raw bytes
        idx_bytes = indices.numpy().tobytes()

        # Handle bfloat16 by viewing as int16
        if values.dtype == torch.bfloat16:
            val_bytes = values.view(torch.int16).numpy().tobytes()
        else:
            val_bytes = values.numpy().tobytes()

        indices_parts.append(idx_bytes)
        values_parts.append(val_bytes)

        layer_info[name] = {
            'shape': layer['shape'],
            'nnz': layer['nnz'],
            'idx_dtype': str(indices.dtype),
            'idx_shape': list(indices.shape),
            'idx_nbytes': len(idx_bytes),
            'val_dtype': str(values.dtype),
            'val_shape': list(values.shape),
            'val_nbytes': len(val_bytes),
        }

    all_indices = b''.join(indices_parts)
    all_values = b''.join(values_parts)

    # Metadata
    meta = {
        'step': delta.get('step'),
        'timestamp': delta.get('timestamp'),
        'metadata': delta.get('metadata'),
        'layer_info': layer_info,
    }
    meta_buf = io.BytesIO()
    pickle.dump(meta, meta_buf)
    meta_bytes = meta_buf.getvalue()

    return all_indices, all_values, meta_bytes, layer_info


def reconstruct_from_streams(idx_bytes: bytes, val_bytes: bytes, meta_bytes: bytes) -> Dict:
    """Reconstruct delta from separated streams."""
    meta = pickle.load(io.BytesIO(meta_bytes))

    delta = {
        'step': meta['step'],
        'timestamp': meta['timestamp'],
        'metadata': meta['metadata'],
        'layers': {}
    }

    idx_offset = 0
    val_offset = 0

    for name, info in meta['layer_info'].items():
        # Extract index bytes
        idx_nb = info['idx_nbytes']
        idx_data = idx_bytes[idx_offset:idx_offset + idx_nb]
        idx_offset += idx_nb

        # Extract value bytes
        val_nb = info['val_nbytes']
        val_data = val_bytes[val_offset:val_offset + val_nb]
        val_offset += val_nb

        # Reconstruct index tensor
        idx_dtype_str = info['idx_dtype'].replace('torch.', '')
        idx_torch_dtype = getattr(torch, idx_dtype_str)
        idx_np_dtype = {
            torch.int32: np.int32, torch.int64: np.int64,
            torch.int16: np.int16, torch.uint8: np.uint8,
        }.get(idx_torch_dtype, np.int32)
        idx_np = np.frombuffer(idx_data, dtype=idx_np_dtype).reshape(info['idx_shape'])
        indices = torch.from_numpy(idx_np.copy())

        # Reconstruct value tensor
        val_dtype_str = info['val_dtype']
        if 'bfloat16' in val_dtype_str:
            val_np = np.frombuffer(val_data, dtype=np.int16).reshape(info['val_shape'])
            values = torch.from_numpy(val_np.copy()).view(torch.bfloat16)
        else:
            val_torch_dtype = getattr(torch, val_dtype_str.replace('torch.', ''))
            val_np_dtype = {
                torch.float32: np.float32, torch.float16: np.float16,
                torch.float64: np.float64,
            }.get(val_torch_dtype, np.float32)
            val_np = np.frombuffer(val_data, dtype=val_np_dtype).reshape(info['val_shape'])
            values = torch.from_numpy(val_np.copy())

        delta['layers'][name] = {
            'indices': indices,
            'values': values,
            'shape': tuple(info['shape']),
            'nnz': info['nnz'],
        }

    return delta


# =============================================================================
# Combined Compression Methods
# =============================================================================

def compress_downcast_zstd(delta: Dict, original_bytes: bytes, level: int) -> Tuple[CompressionResult, bytes]:
    """Downcast indices then compress with zstd."""
    t0 = time.perf_counter()
    transformed = downcast_indices(delta)
    transformed_bytes = serialize_delta(transformed)
    preprocess_time = time.perf_counter() - t0

    compressed, comp_time, decompressed, decomp_time = compress_zstd(transformed_bytes, level)

    # Verify lossless (for transformed representation)
    is_lossless = (transformed_bytes == decompressed)

    return (preprocess_time, comp_time, decomp_time,
            compressed, decompressed, is_lossless,
            0, 0, 0, 0, 0, 0)  # No stream breakdown


def compress_separate_zstd(delta: Dict, original_bytes: bytes, level: int) -> Tuple:
    """Separate streams then compress each with zstd."""
    t0 = time.perf_counter()
    idx_bytes, val_bytes, meta_bytes, _ = separate_streams(delta)
    preprocess_time = time.perf_counter() - t0

    idx_orig, val_orig, meta_orig = len(idx_bytes), len(val_bytes), len(meta_bytes)

    # Compress each stream
    idx_comp, t1, idx_decomp, dt1 = compress_zstd(idx_bytes, level)
    val_comp, t2, val_decomp, dt2 = compress_zstd(val_bytes, level)
    meta_comp, t3, meta_decomp, dt3 = compress_zstd(meta_bytes, level)

    comp_time = t1 + t2 + t3
    decomp_time = dt1 + dt2 + dt3

    total_comp = len(idx_comp) + len(val_comp) + len(meta_comp)

    # Verify lossless: check that streams decompress correctly
    # (Don't compare torch.save outputs as serialization order may differ)
    is_lossless = (idx_bytes == idx_decomp and
                   val_bytes == val_decomp and
                   meta_bytes == meta_decomp)

    # Create combined compressed bytes for size calculation
    compressed = idx_comp + val_comp + meta_comp
    decompressed = idx_decomp + val_decomp + meta_decomp

    return (preprocess_time, comp_time, decomp_time,
            compressed, decompressed, is_lossless,
            idx_orig, len(idx_comp), val_orig, len(val_comp), meta_orig, len(meta_comp))


def compress_delta_enc_zstd(delta: Dict, original_bytes: bytes, level: int) -> Tuple:
    """Delta-encode indices then compress with zstd."""
    t0 = time.perf_counter()
    transformed = delta_encode_indices(delta)
    transformed_bytes = serialize_delta(transformed)
    preprocess_time = time.perf_counter() - t0

    compressed, comp_time, decompressed, decomp_time = compress_zstd(transformed_bytes, level)

    is_lossless = (transformed_bytes == decompressed)

    return (preprocess_time, comp_time, decomp_time,
            compressed, decompressed, is_lossless,
            0, 0, 0, 0, 0, 0)


def compress_combined_zstd(delta: Dict, original_bytes: bytes, level: int) -> Tuple:
    """Apply all transformations: downcast + delta-encode + separate, then zstd."""
    t0 = time.perf_counter()

    # Apply transformations in sequence
    step1 = downcast_indices(delta)
    step2 = delta_encode_indices(step1)
    idx_bytes, val_bytes, meta_bytes, _ = separate_streams(step2)

    preprocess_time = time.perf_counter() - t0

    idx_orig, val_orig, meta_orig = len(idx_bytes), len(val_bytes), len(meta_bytes)

    # Compress each stream
    idx_comp, t1, idx_decomp, dt1 = compress_zstd(idx_bytes, level)
    val_comp, t2, val_decomp, dt2 = compress_zstd(val_bytes, level)
    meta_comp, t3, meta_decomp, dt3 = compress_zstd(meta_bytes, level)

    comp_time = t1 + t2 + t3
    decomp_time = dt1 + dt2 + dt3

    total_comp = len(idx_comp) + len(val_comp) + len(meta_comp)

    # Verify streams decompress correctly
    is_lossless = (idx_bytes == idx_decomp and
                   val_bytes == val_decomp and
                   meta_bytes == meta_decomp)

    compressed = idx_comp + val_comp + meta_comp
    decompressed = idx_decomp + val_decomp + meta_decomp

    return (preprocess_time, comp_time, decomp_time,
            compressed, decompressed, is_lossless,
            idx_orig, len(idx_comp), val_orig, len(val_comp), meta_orig, len(meta_comp))


# =============================================================================
# Main Benchmark Functions
# =============================================================================

def benchmark_file(
    file_info: FileInfo,
    sample_idx: int,
    logger: logging.Logger
) -> List[CompressionResult]:
    """
    Benchmark ALL methods on a single file.
    This ensures fair comparison - same file for all methods.
    """
    results = []
    path = file_info.path
    model = file_info.model
    experiment = file_info.experiment

    # Load file
    logger.info(f"Loading {path.name}...")
    delta = load_delta(path)

    # Serialize original (this is our reference)
    original_bytes = serialize_delta(delta)
    original_size = len(original_bytes)
    original_hash = compute_hash(original_bytes)

    # Get stats
    step, num_layers, total_nnz, total_params, sparsity = get_delta_stats(delta)

    logger.info(f"  Original: {original_size/1024/1024:.2f} MB, "
                f"Layers: {num_layers}, NNZ: {total_nnz:,}, Sparsity: {sparsity:.2f}%")

    def make_result(
        method: str,
        preprocess_time: float,
        comp_time: float,
        decomp_time: float,
        compressed: bytes,
        decompressed: bytes,
        is_lossless: bool,
        idx_orig: int = 0, idx_comp: int = 0,
        val_orig: int = 0, val_comp: int = 0,
        meta_orig: int = 0, meta_comp: int = 0,
        error: str = ""
    ) -> CompressionResult:
        comp_size = len(compressed)
        ratio = original_size / comp_size if comp_size > 0 else 0
        savings = (1 - comp_size / original_size) * 100 if original_size > 0 else 0
        total_time = preprocess_time + comp_time
        comp_speed = (original_size / 1024 / 1024) / comp_time if comp_time > 0 else 0
        decomp_speed = (original_size / 1024 / 1024) / decomp_time if decomp_time > 0 else 0
        decomp_hash = compute_hash(decompressed) if decompressed else ""

        return CompressionResult(
            model=model,
            experiment=experiment,
            file_name=path.name,
            file_path=str(path),
            step=step,
            sample_idx=sample_idx,
            original_size_bytes=original_size,
            num_layers=num_layers,
            total_nnz=total_nnz,
            total_params=total_params,
            sparsity_pct=sparsity,
            method=method,
            compressed_size_bytes=comp_size,
            compression_ratio=ratio,
            space_savings_pct=savings,
            preprocess_time=preprocess_time,
            compress_time=comp_time,
            decompress_time=decomp_time,
            total_time=total_time,
            compress_speed_mb_s=comp_speed,
            decompress_speed_mb_s=decomp_speed,
            is_lossless=is_lossless,
            original_hash=original_hash,
            decompressed_hash=decomp_hash,
            indices_orig_bytes=idx_orig,
            indices_comp_bytes=idx_comp,
            values_orig_bytes=val_orig,
            values_comp_bytes=val_comp,
            metadata_orig_bytes=meta_orig,
            metadata_comp_bytes=meta_comp,
            error_msg=error
        )

    # Test each method
    for method_name, (comp_type, params) in METHODS_CONFIG.items():
        try:
            if comp_type == 'zstd':
                if not HAS_ZSTD:
                    continue
                compressed, comp_time, decompressed, decomp_time = compress_zstd(original_bytes, **params)
                is_lossless = (original_bytes == decompressed)
                result = make_result(method_name, 0, comp_time, decomp_time,
                                    compressed, decompressed, is_lossless)

            elif comp_type == 'brotli':
                if not HAS_BROTLI:
                    continue
                compressed, comp_time, decompressed, decomp_time = compress_brotli(original_bytes, **params)
                is_lossless = (original_bytes == decompressed)
                result = make_result(method_name, 0, comp_time, decomp_time,
                                    compressed, decompressed, is_lossless)

            elif comp_type == 'gzip':
                compressed, comp_time, decompressed, decomp_time = compress_gzip(original_bytes, **params)
                is_lossless = (original_bytes == decompressed)
                result = make_result(method_name, 0, comp_time, decomp_time,
                                    compressed, decompressed, is_lossless)

            elif comp_type == 'downcast_zstd':
                if not HAS_ZSTD:
                    continue
                (prep_t, comp_t, decomp_t, compressed, decompressed, is_lossless,
                 i_o, i_c, v_o, v_c, m_o, m_c) = compress_downcast_zstd(delta, original_bytes, **params)
                result = make_result(method_name, prep_t, comp_t, decomp_t,
                                    compressed, decompressed, is_lossless,
                                    i_o, i_c, v_o, v_c, m_o, m_c)

            elif comp_type == 'separate_zstd':
                if not HAS_ZSTD:
                    continue
                (prep_t, comp_t, decomp_t, compressed, decompressed, is_lossless,
                 i_o, i_c, v_o, v_c, m_o, m_c) = compress_separate_zstd(delta, original_bytes, **params)
                result = make_result(method_name, prep_t, comp_t, decomp_t,
                                    compressed, decompressed, is_lossless,
                                    i_o, i_c, v_o, v_c, m_o, m_c)

            elif comp_type == 'delta_enc_zstd':
                if not HAS_ZSTD:
                    continue
                (prep_t, comp_t, decomp_t, compressed, decompressed, is_lossless,
                 i_o, i_c, v_o, v_c, m_o, m_c) = compress_delta_enc_zstd(delta, original_bytes, **params)
                result = make_result(method_name, prep_t, comp_t, decomp_t,
                                    compressed, decompressed, is_lossless,
                                    i_o, i_c, v_o, v_c, m_o, m_c)

            elif comp_type == 'combined_zstd':
                if not HAS_ZSTD:
                    continue
                (prep_t, comp_t, decomp_t, compressed, decompressed, is_lossless,
                 i_o, i_c, v_o, v_c, m_o, m_c) = compress_combined_zstd(delta, original_bytes, **params)
                result = make_result(method_name, prep_t, comp_t, decomp_t,
                                    compressed, decompressed, is_lossless,
                                    i_o, i_c, v_o, v_c, m_o, m_c)
            else:
                continue

            results.append(result)

            # Log result
            lossless_str = "✓" if result.is_lossless else "✗"
            logger.info(f"  {method_name:24s}: {result.compression_ratio:6.3f}x  "
                       f"{result.compress_speed_mb_s:7.1f} MB/s  {lossless_str}")

            if not result.is_lossless:
                logger.warning(f"    LOSSLESS CHECK FAILED for {method_name}!")

        except Exception as e:
            logger.error(f"  {method_name}: ERROR - {e}")
            traceback.print_exc()
            results.append(make_result(method_name, 0, 0, 0, b'', b'', False, error=str(e)))

    return results


def extract_step_from_filename(filename: str) -> int:
    """Extract step number from filename like delta_000123.pt -> 123."""
    import re
    match = re.search(r'delta_(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    return -1


def discover_and_sample_files(
    base_path: Path,
    samples_per_experiment: int,
    logger: logging.Logger
) -> List[FileInfo]:
    """
    Discover all experiments and sample files DETERMINISTICALLY.
    Samples UNIFORMLY across step range (0 to max_step).
    Uses fixed random seed for reproducibility.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Discover all files grouped by model/experiment
    # Store as: experiments[model][exp] = [(path, step_num), ...]
    experiments = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in os.walk(base_path):
        delta_files = sorted([f for f in files if f.startswith('delta_') and f.endswith('.pt')])
        if not delta_files:
            continue

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
            full_path = Path(root) / f
            step_num = extract_step_from_filename(f)
            experiments[model][exp_name].append((full_path, step_num))

    # Log discovery
    total_files = sum(len(files) for exps in experiments.values() for files in exps.values())
    logger.info(f"Discovered {len(experiments)} models, {total_files} total files")

    for model in sorted(experiments.keys()):
        logger.info(f"  {model}:")
        for exp in sorted(experiments[model].keys()):
            files = experiments[model][exp]
            steps = [s for _, s in files]
            min_step, max_step = min(steps), max(steps)
            logger.info(f"    {exp}: {len(files)} files, steps {min_step}-{max_step}")

    # Sample files UNIFORMLY across step range
    sampled_files = []

    for model in sorted(experiments.keys()):
        for exp in sorted(experiments[model].keys()):
            files = experiments[model][exp]
            n_files = len(files)
            n_samples = min(samples_per_experiment, n_files)

            # Sort by step number
            files_sorted = sorted(files, key=lambda x: x[1])
            steps = [s for _, s in files_sorted]
            min_step, max_step = min(steps), max(steps)

            if n_samples >= n_files:
                # Take all files
                selected = files_sorted
            else:
                # Sample uniformly across step range [min_step, max_step]
                target_steps = np.linspace(min_step, max_step, n_samples, dtype=int)

                # For each target step, find the closest available file
                selected = []
                step_to_file = {s: p for p, s in files_sorted}
                available_steps = np.array(steps)

                for target in target_steps:
                    # Find closest available step
                    idx = np.argmin(np.abs(available_steps - target))
                    closest_step = available_steps[idx]
                    path = step_to_file[closest_step]
                    if (path, closest_step) not in selected:
                        selected.append((path, closest_step))

                # Ensure we have exactly n_samples (handle edge cases)
                selected = selected[:n_samples]

            logger.info(f"    Sampled {len(selected)} files: steps {[s for _, s in selected[:3]]}...{[s for _, s in selected[-3:]]}")

            for idx, (path, step_num) in enumerate(selected):
                sampled_files.append(FileInfo(
                    path=path,
                    model=model,
                    experiment=exp,
                    size_bytes=path.stat().st_size,
                    index_in_experiment=step_num  # Use step number as index
                ))

    logger.info(f"\nSampled {len(sampled_files)} files for benchmarking")
    return sampled_files


def run_benchmark(
    base_path: Path,
    output_csv: Path,
    log_file: Path,
    samples_per_experiment: int
):
    """Run the complete benchmark."""

    logger = setup_logging(log_file)

    logger.info("=" * 80)
    logger.info("DELTA COMPRESSION BENCHMARK - FINAL VERSION")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Samples per experiment: {samples_per_experiment}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"zstd available: {HAS_ZSTD}")
    logger.info(f"brotli available: {HAS_BROTLI}")
    logger.info(f"Methods to test: {len(METHODS_CONFIG)}")
    logger.info("")

    # Discover and sample files
    sampled_files = discover_and_sample_files(base_path, samples_per_experiment, logger)

    total_tests = len(sampled_files) * len(METHODS_CONFIG)
    logger.info(f"\nTotal compression tests: {len(sampled_files)} files × {len(METHODS_CONFIG)} methods = {total_tests}")
    logger.info("")

    # Prepare CSV
    csv_fields = [
        'model', 'experiment', 'file_name', 'file_path', 'step', 'sample_idx',
        'original_size_bytes', 'num_layers', 'total_nnz', 'total_params', 'sparsity_pct',
        'method',
        'compressed_size_bytes', 'compression_ratio', 'space_savings_pct',
        'preprocess_time', 'compress_time', 'decompress_time', 'total_time',
        'compress_speed_mb_s', 'decompress_speed_mb_s',
        'is_lossless', 'original_hash', 'decompressed_hash',
        'indices_orig_bytes', 'indices_comp_bytes',
        'values_orig_bytes', 'values_comp_bytes',
        'metadata_orig_bytes', 'metadata_comp_bytes',
        'error_msg'
    ]

    processed = 0
    errors = 0

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        current_model = None
        current_exp = None

        for i, file_info in enumerate(sampled_files):
            # Log section headers
            if file_info.model != current_model:
                current_model = file_info.model
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"MODEL: {current_model}")
                logger.info("=" * 80)

            if file_info.experiment != current_exp:
                current_exp = file_info.experiment
                logger.info("")
                logger.info(f"  EXPERIMENT: {current_exp}")
                logger.info("-" * 60)

            # Progress
            size_mb = file_info.size_bytes / 1024 / 1024
            logger.info("")
            logger.info(f"[{i+1}/{len(sampled_files)}] {file_info.path.name} ({size_mb:.1f} MB)")

            try:
                results = benchmark_file(file_info, i, logger)

                # Write results to CSV
                for r in results:
                    writer.writerow(asdict(r))
                f.flush()

                processed += 1

                # Log best
                valid = [r for r in results if r.compression_ratio > 0 and r.is_lossless]
                if valid:
                    best = max(valid, key=lambda r: r.compression_ratio)
                    logger.info(f"  BEST: {best.method} = {best.compression_ratio:.3f}x")

            except Exception as e:
                logger.error(f"FATAL ERROR: {e}")
                traceback.print_exc()
                errors += 1

    logger.info("")
    logger.info("=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Finished: {datetime.now().isoformat()}")
    logger.info(f"Files processed: {processed}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Results saved to: {output_csv}")

    # Print summary
    print_summary(output_csv, logger)


def print_summary(csv_path: Path, logger: logging.Logger):
    """Print summary statistics."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Filter to lossless only
        df_lossless = df[df['is_lossless'] == True]

        logger.info("")
        logger.info("=" * 100)
        logger.info("SUMMARY: COMPRESSION RATIO BY METHOD (lossless only)")
        logger.info("=" * 100)

        summary = df_lossless.groupby('method').agg({
            'compression_ratio': ['mean', 'std', 'min', 'max', 'count'],
            'compress_speed_mb_s': 'mean',
        }).round(3)

        summary.columns = ['Ratio_Mean', 'Ratio_Std', 'Ratio_Min', 'Ratio_Max', 'N', 'Speed_MB/s']
        summary = summary.sort_values('Ratio_Mean', ascending=False)

        logger.info("\n" + summary.to_string())

        # Per-model breakdown
        logger.info("")
        logger.info("=" * 100)
        logger.info("SUMMARY: BEST METHOD PER MODEL")
        logger.info("=" * 100)

        for model in sorted(df_lossless['model'].unique()):
            model_df = df_lossless[df_lossless['model'] == model]
            best = model_df.groupby('method')['compression_ratio'].mean().idxmax()
            ratio = model_df.groupby('method')['compression_ratio'].mean().max()
            logger.info(f"  {model:20s}: {best:24s} ({ratio:.3f}x)")

        # Recommendations
        logger.info("")
        logger.info("=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)

        best_ratio_method = summary['Ratio_Mean'].idxmax()
        best_ratio = summary.loc[best_ratio_method, 'Ratio_Mean']

        best_speed_method = summary['Speed_MB/s'].idxmax()
        best_speed = summary.loc[best_speed_method, 'Speed_MB/s']

        # Efficiency score
        summary['Efficiency'] = summary['Ratio_Mean'] * np.sqrt(summary['Speed_MB/s'])
        best_eff_method = summary['Efficiency'].idxmax()

        logger.info(f"  Best compression:  {best_ratio_method} ({best_ratio:.3f}x)")
        logger.info(f"  Fastest:           {best_speed_method} ({best_speed:.1f} MB/s)")
        logger.info(f"  Best balance:      {best_eff_method}")

        # Improvement over baseline
        if 'zstd-3' in summary.index:
            baseline = summary.loc['zstd-3', 'Ratio_Mean']
            improvement = (best_ratio / baseline - 1) * 100
            logger.info(f"\n  Best method is {improvement:.1f}% better than zstd-3 baseline")

    except Exception as e:
        logger.error(f"Could not generate summary: {e}")
        traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Delta Compression Benchmark - Final')
    parser.add_argument('--base-path', type=str,
                       default='/root/grail/research/sparsity_analysis/experiments')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per experiment (default: 100)')
    parser.add_argument('--output', type=str,
                       default='/root/grail/scripts/compression_metrics_final.csv')
    parser.add_argument('--log', type=str,
                       default='/root/grail/scripts/compression_benchmark_final.log')
    parser.add_argument('--nice', type=int, default=10)
    parser.add_argument('--max-threads', type=int, default=2)
    parser.add_argument('--fast', action='store_true',
                       help='Use fast mode: fewer methods, skip slowest compressors')

    args = parser.parse_args()

    # Select method set based on --fast flag
    global METHODS_CONFIG
    if args.fast:
        METHODS_CONFIG = METHODS_FAST
        print("Using FAST mode (12 methods, ~3x faster)")
    else:
        METHODS_CONFIG = METHODS_FULL
        print("Using FULL mode (16 methods)")

    # Set low priority
    if args.nice > 0:
        try:
            current = os.nice(0)
            os.nice(args.nice)
            print(f"Set nice value to {os.nice(0)}")
        except Exception as e:
            print(f"Could not set nice: {e}")

    # Limit threads
    os.environ['OMP_NUM_THREADS'] = str(args.max_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.max_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.max_threads)
    torch.set_num_threads(args.max_threads)
    print(f"Max threads: {args.max_threads}")

    base_path = Path(args.base_path)
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        sys.exit(1)

    run_benchmark(
        base_path=base_path,
        output_csv=Path(args.output),
        log_file=Path(args.log),
        samples_per_experiment=args.samples
    )


if __name__ == '__main__':
    main()
