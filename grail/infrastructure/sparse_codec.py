"""Optimized sparse encoding for maximum compression.

This module provides delta-encoded sparse serialization that achieves 8-18x compression
ratios compared to the ~1.3x achieved by naive safetensors + zstd.

Key optimizations:
1. COO indices (2D) with per-row delta encoding for better structure
2. Delta encoding on sorted indices - small integers compress extremely well
3. Direct byte packing - no safetensors header overhead
4. zstd level 1 - best speed/ratio tradeoff
5. (v3.1) Integer downscaling: uint8 for row deltas, uint16 for col deltas

Format (v3):
    Uncompressed structure:
    - header_len (4 bytes, little-endian uint32)
    - header (JSON): version, tensors list with metadata
    - for each tensor: delta_rows (int32), delta_cols (int32), values (original dtype)

Format (v3.1 - with index downscaling):
    Same structure as v3, but indices use smaller dtypes:
    - row_dtype: "uint8" or "int32" (per-tensor, based on max value)
    - col_dtype: "uint16" or "int32" (per-tensor, based on max value)
    Delta-encoded COO indices are always non-negative after lexsort, enabling unsigned types.

Per-tensor metadata:
    - name: tensor name (e.g., "model.layers.0.weight")
    - shape: original tensor shape
    - num_nonzero: number of non-zero elements
    - value_dtype: "bfloat16", "float16", or "float32"
    - row_dtype: "uint8" or "int32" (v3.1 only)
    - col_dtype: "uint16" or "int32" (v3.1 only)
    - indices_offset, indices_size: byte offsets for rows+cols
    - rows_size, cols_size: byte lengths for row/col deltas
    - values_offset, values_size: byte offsets for values
"""

from __future__ import annotations

import json
import logging
import math
import struct
from typing import Any

import numpy as np
import torch
import zstandard as zstd

logger = logging.getLogger(__name__)

# Format versions for compatibility checking
FORMAT_VERSION_V2 = 2
FORMAT_VERSION_V3 = 3
FORMAT_VERSION_V3_1 = 31  # v3.1 with index downscaling

# COO baseline sizing (flattening is part of compression)
COO_INDEX_COMPONENTS = 2
COO_INDEX_BYTES = np.dtype(np.int32).itemsize

# Index dtype constants for v3.1
INDEX_DTYPE_INT32 = "int32"
INDEX_DTYPE_UINT16 = "uint16"
INDEX_DTYPE_UINT8 = "uint8"

# Numpy dtype mapping for index types
_INDEX_NP_DTYPES: dict[str, np.dtype[np.generic]] = {
    INDEX_DTYPE_INT32: np.dtype(np.int32),
    INDEX_DTYPE_UINT16: np.dtype(np.uint16),
    INDEX_DTYPE_UINT8: np.dtype(np.uint8),
}

# Compression settings (level 1 is optimal speed/ratio tradeoff from benchmarks)
ZSTD_LEVEL = 1
ZSTD_THREADS = 0  # Auto-detect


def _compute_cols_per_row(shape: list[int]) -> int:
    if not shape:
        return 1
    if len(shape) == 1:
        return 1
    cols = int(math.prod(shape[1:]))
    return max(1, cols)


def _coo_rows_cols_from_indices(
    indices: np.ndarray,
    shape: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if indices.ndim == 1:
        # 1D flat indices: convert to row/col using shape
        cols_per_row = _compute_cols_per_row(shape)
        rows = (indices // cols_per_row).astype(np.int32)
        cols = (indices % cols_per_row).astype(np.int32)
        return rows, cols
    if indices.ndim == 2 and indices.shape[0] == 2:
        # Standard 2D COO format: [rows, cols]
        rows = indices[0].astype(np.int32)
        cols = indices[1].astype(np.int32)
        return rows, cols
    if indices.ndim == 2 and indices.shape[0] == 1:
        # 1D tensor stored as (1, nnz): indices are row positions, col=0
        rows = indices[0].astype(np.int32)
        cols = np.zeros_like(rows, dtype=np.int32)
        return rows, cols
    raise ValueError(f"Invalid indices shape {indices.shape} for COO conversion")


def _coo_to_flat_indices(indices: np.ndarray, shape: list[int]) -> np.ndarray:
    rows, cols = _coo_rows_cols_from_indices(indices, shape)
    cols_per_row = _compute_cols_per_row(shape)
    flat = rows.astype(np.int64) * int(cols_per_row) + cols.astype(np.int64)
    if flat.size == 0:
        return flat.astype(np.int32)
    if flat.max() > np.iinfo(np.int32).max:
        raise ValueError("Flat indices exceed int32 range for v2 encoding")
    return flat.astype(np.int32)


def _sort_coo(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], values[order]


def _delta_encode_rows_cols(
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Delta encode sorted COO indices (vectorized).

    For rows: simple diff encoding.
    For cols: delta within same row, absolute value when row changes.

    This avoids negative deltas on column resets, improving compression.
    """
    if rows.size == 0:
        return rows.astype(np.int32), cols.astype(np.int32)

    rows_delta = np.diff(rows, prepend=0).astype(np.int32)

    # Vectorized column delta encoding:
    # - When row changes (rows_delta != 0): store absolute column
    # - When row is same (rows_delta == 0): store column delta
    row_boundary = np.ones(len(rows), dtype=bool)
    row_boundary[1:] = rows_delta[1:] != 0

    # Column differences (for within-row deltas)
    col_diff = np.empty_like(cols, dtype=np.int32)
    col_diff[0] = cols[0]
    col_diff[1:] = cols[1:] - cols[:-1]

    # Use absolute column at row boundaries, delta otherwise
    cols_delta = np.where(row_boundary, cols, col_diff).astype(np.int32)

    return rows_delta, cols_delta


def _delta_decode_rows_cols(
    rows_delta: np.ndarray,
    cols_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode delta-encoded COO indices (vectorized).

    Inverse of _delta_encode_rows_cols. Uses segment-wise cumsum for columns.
    """
    if rows_delta.size == 0:
        return rows_delta.astype(np.int32), cols_delta.astype(np.int32)

    rows = np.cumsum(rows_delta).astype(np.int32)

    # Find row boundaries (where we need to reset cumsum)
    row_boundary = np.ones(len(rows_delta), dtype=bool)
    row_boundary[1:] = rows_delta[1:] != 0

    # Segment IDs: each contiguous run of same-row gets an ID
    segment_ids = np.cumsum(row_boundary) - 1

    # Segment-wise cumsum for columns
    # Use np.add.reduceat for efficient segment operations
    n_segments = segment_ids[-1] + 1
    segment_starts = np.zeros(n_segments + 1, dtype=np.int64)
    segment_starts[1:] = np.searchsorted(segment_ids, np.arange(n_segments), side="right")

    cols = np.empty_like(cols_delta, dtype=np.int32)
    for seg_id in range(n_segments):
        start = segment_starts[seg_id]
        end = segment_starts[seg_id + 1]
        cols[start:end] = np.cumsum(cols_delta[start:end])

    return rows, cols


def _values_to_numpy(values: torch.Tensor) -> tuple[np.ndarray, str]:
    values_cpu = values.detach().cpu().contiguous()
    if values_cpu.dtype == torch.bfloat16:
        values_np = values_cpu.view(torch.int16).numpy()
        return values_np, "bfloat16"
    if values_cpu.dtype == torch.float16:
        values_np = values_cpu.view(torch.int16).numpy()
        return values_np, "float16"
    if values_cpu.dtype == torch.float32:
        return values_cpu.numpy(), "float32"
    return values_cpu.float().numpy(), "float32"


def _select_index_dtype(
    delta_values: np.ndarray,
    prefer_uint8: bool = True,
) -> tuple[str, np.dtype[np.generic]]:
    """Select the smallest safe dtype for delta-encoded indices.

    Delta-encoded COO indices are always non-negative after lexsort:
    - Row deltas: rows sorted ascending, so delta >= 0
    - Col deltas: within-row sorted ascending or absolute (both >= 0)

    This enables unsigned types for 2x the range vs signed types.

    Args:
        delta_values: Delta-encoded index values (must be non-negative)
        prefer_uint8: If True, try uint8 first for row deltas

    Returns:
        Tuple of (dtype_name, numpy_dtype) for serialization
    """
    if delta_values.size == 0:
        # Empty array - use smallest type
        return INDEX_DTYPE_UINT8, np.dtype(np.uint8)

    max_val = delta_values.max()
    min_val = delta_values.min()

    # Safety check: delta-encoded values should be non-negative
    if min_val < 0:
        logger.warning(
            "Unexpected negative delta value: min=%d. Falling back to int32.",
            min_val,
        )
        return INDEX_DTYPE_INT32, np.dtype(np.int32)

    # Select smallest fitting unsigned type
    if prefer_uint8 and max_val <= np.iinfo(np.uint8).max:
        return INDEX_DTYPE_UINT8, np.dtype(np.uint8)
    if max_val <= np.iinfo(np.uint16).max:
        return INDEX_DTYPE_UINT16, np.dtype(np.uint16)

    # Fallback to int32 for very large values
    return INDEX_DTYPE_INT32, np.dtype(np.int32)


def encode_sparse_delta_v2(
    sparse_tensors: dict[str, torch.Tensor],
    shapes: dict[str, list[int]],
) -> bytes:
    """Encode sparse delta with optimal compression.

    Takes sparse tensors in {name}.indices / {name}.values format and produces
    highly compressed bytes using delta encoding + zstd. For 2D indices
    (COO), indices are flattened to 1D for v2 compatibility.

    Args:
        sparse_tensors: Dict with {name}.indices (int32) and {name}.values (dtype)
        shapes: Dict mapping parameter names to original shapes

    Returns:
        Compressed bytes ready for upload

    Raises:
        ValueError: If input tensors are malformed
    """
    # Handle empty delta
    if not sparse_tensors:
        header = {"version": FORMAT_VERSION_V2, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    # Extract unique tensor names from {name}.indices keys
    tensor_names = sorted({k.rsplit(".", 1)[0] for k in sparse_tensors if k.endswith(".indices")})

    tensors_meta: list[dict[str, Any]] = []
    all_data_chunks: list[bytes] = []
    current_offset = 0
    coo_raw_bytes = 0

    for name in tensor_names:
        indices_key = f"{name}.indices"
        values_key = f"{name}.values"

        # Skip if missing indices or values
        if indices_key not in sparse_tensors or values_key not in sparse_tensors:
            continue

        indices = sparse_tensors[indices_key]
        values = sparse_tensors[values_key]
        shape = shapes.get(name)

        # Shape is required for reconstruction
        if not shape:
            raise ValueError(f"Shape missing for tensor {name}")

        # COO baseline size: indices are 2x int32 + values (flattening is compression)
        nnz = values.numel()
        coo_raw_bytes += nnz * (COO_INDEX_COMPONENTS * COO_INDEX_BYTES + values.element_size())

        # Convert indices to flat numpy array for v2 compatibility
        indices_np = indices.detach().cpu().numpy()
        if indices_np.ndim == 2:
            flat_indices = _coo_to_flat_indices(indices_np, shape)
        else:
            flat_indices = indices_np.astype(np.int32)

        if flat_indices.ndim != 1:
            raise ValueError(f"Invalid indices shape for {name}: {indices_np.shape}")

        # Delta encode: [1024, 1025, 1027, 2048] -> [1024, 1, 2, 1021]
        sort_order = np.argsort(flat_indices)
        sorted_indices = flat_indices[sort_order]
        delta_indices = np.diff(sorted_indices, prepend=0).astype(np.int32)

        # Convert values to numpy, preserving dtype
        values_np, value_dtype = _values_to_numpy(values)
        if values_np.shape[0] != flat_indices.shape[0]:
            raise ValueError(
                f"Index/value mismatch for {name}: "
                f"{flat_indices.shape[0]} indices vs {values_np.shape[0]} values"
            )
        sorted_values = values_np[sort_order]

        indices_bytes = delta_indices.tobytes()
        values_bytes = sorted_values.tobytes()

        tensors_meta.append(
            {
                "name": name,
                "shape": shape,
                "num_nonzero": int(flat_indices.size),
                "value_dtype": value_dtype,
                "indices_offset": current_offset,
                "indices_size": len(indices_bytes),
                "values_offset": current_offset + len(indices_bytes),
                "values_size": len(values_bytes),
            }
        )

        all_data_chunks.append(indices_bytes)
        all_data_chunks.append(values_bytes)
        current_offset += len(indices_bytes) + len(values_bytes)

    # Handle case where all tensors were skipped
    if not tensors_meta:
        header = {"version": FORMAT_VERSION_V2, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    # Build final payload
    header = {"version": FORMAT_VERSION_V2, "tensors": tensors_meta}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    uncompressed = struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(all_data_chunks)

    # Compress with zstd
    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=ZSTD_THREADS)
    compressed = compressor.compress(uncompressed)

    comp_size = len(compressed)
    ratio = coo_raw_bytes / comp_size if comp_size > 0 else 1.0

    logger.debug(
        "Sparse delta encoded (v2, COO baseline): %d tensors | %.2f MB -> %.2f MB (%.1fx ratio)",
        len(tensors_meta),
        coo_raw_bytes / (1024 * 1024),
        comp_size / (1024 * 1024),
        ratio,
    )

    return compressed


def decode_sparse_delta_v2(
    compressed: bytes,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Decode compressed sparse delta back to tensors.

    Args:
        compressed: Compressed bytes from encode_sparse_delta_v2

    Returns:
        Tuple of (sparse_tensors, shapes) where:
        - sparse_tensors: Dict with {name}.indices and {name}.values
        - shapes: Dict mapping parameter names to original shapes

    Raises:
        ValueError: If format version is unsupported or data is malformed
        zstd.ZstdError: If decompression fails
    """
    if not compressed:
        return {}, {}

    # Decompress
    decompressor = zstd.ZstdDecompressor()
    uncompressed = decompressor.decompress(compressed)

    if len(uncompressed) < 4:
        raise ValueError("Compressed data too short to contain header length")

    # Parse header
    header_len = struct.unpack("<I", uncompressed[:4])[0]
    if len(uncompressed) < 4 + header_len:
        raise ValueError("Compressed data too short to contain full header")

    header = json.loads(uncompressed[4 : 4 + header_len].decode("utf-8"))

    version = header.get("version", 0)
    if version != FORMAT_VERSION_V2:
        raise ValueError(f"Unsupported format version: {version} (expected {FORMAT_VERSION_V2})")

    tensors_meta = header.get("tensors", [])
    if not tensors_meta:
        return {}, {}

    data_start = 4 + header_len
    sparse_tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}

    for meta in tensors_meta:
        name = meta["name"]
        shape = meta["shape"]
        value_dtype = meta["value_dtype"]
        indices_offset = meta["indices_offset"]
        indices_size = meta["indices_size"]
        values_offset = meta["values_offset"]
        values_size = meta["values_size"]

        # Extract bytes
        indices_bytes = uncompressed[
            data_start + indices_offset : data_start + indices_offset + indices_size
        ]
        values_bytes = uncompressed[
            data_start + values_offset : data_start + values_offset + values_size
        ]

        # Decode delta-encoded indices: cumsum reverses the diff
        delta_indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
        sorted_indices = np.cumsum(delta_indices)

        # Decode values based on dtype
        if value_dtype == "bfloat16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.bfloat16)
        elif value_dtype == "float16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.float16)
        else:
            values_np = np.frombuffer(values_bytes, dtype=np.float32).copy()
            values = torch.from_numpy(values_np)

        # Note: indices and values are both in sorted order, which is fine for
        # apply_sparse_delta since it just assigns values[i] to position indices[i]
        indices = torch.from_numpy(sorted_indices.astype(np.int32))

        sparse_tensors[f"{name}.indices"] = indices
        sparse_tensors[f"{name}.values"] = values
        shapes[name] = shape

    logger.debug("Sparse delta decoded: %d tensors", len(shapes))

    return sparse_tensors, shapes


# New v3 format: 2D COO indices with per-row delta encoding
def encode_sparse_delta_v3(
    sparse_tensors: dict[str, torch.Tensor],
    shapes: dict[str, list[int]],
) -> bytes:
    """Encode sparse delta with 2D COO indices and optimal compression."""
    if not sparse_tensors:
        header = {"version": FORMAT_VERSION_V3, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    tensor_names = sorted({k.rsplit(".", 1)[0] for k in sparse_tensors if k.endswith(".indices")})

    tensors_meta: list[dict[str, Any]] = []
    all_data_chunks: list[bytes] = []
    current_offset = 0
    coo_raw_bytes = 0

    for name in tensor_names:
        indices_key = f"{name}.indices"
        values_key = f"{name}.values"
        if indices_key not in sparse_tensors or values_key not in sparse_tensors:
            continue

        shape = shapes.get(name)
        if not shape:
            raise ValueError(f"Shape missing for tensor {name}")

        indices_tensor = sparse_tensors[indices_key]
        values_tensor = sparse_tensors[values_key]
        if values_tensor.numel() == 0:
            continue

        # COO baseline size: indices are 2x int32 + values (flattening is compression)
        nnz = values_tensor.numel()
        coo_raw_bytes += nnz * (
            COO_INDEX_COMPONENTS * COO_INDEX_BYTES + values_tensor.element_size()
        )

        indices_np = indices_tensor.detach().cpu().numpy()
        rows, cols = _coo_rows_cols_from_indices(indices_np, shape)
        values_np, value_dtype = _values_to_numpy(values_tensor)

        if rows.size != values_np.shape[0]:
            raise ValueError(
                f"Index/value mismatch for {name}: "
                f"{rows.size} indices vs {values_np.shape[0]} values"
            )

        rows_sorted, cols_sorted, values_sorted = _sort_coo(rows, cols, values_np)
        rows_delta, cols_delta = _delta_encode_rows_cols(rows_sorted, cols_sorted)

        rows_bytes = rows_delta.astype(np.int32).tobytes()
        cols_bytes = cols_delta.astype(np.int32).tobytes()
        values_bytes = values_sorted.tobytes()

        indices_size = len(rows_bytes) + len(cols_bytes)
        tensors_meta.append(
            {
                "name": name,
                "shape": shape,
                "num_nonzero": int(rows.size),
                "value_dtype": value_dtype,
                "indices_offset": current_offset,
                "indices_size": indices_size,
                "rows_size": len(rows_bytes),
                "cols_size": len(cols_bytes),
                "values_offset": current_offset + indices_size,
                "values_size": len(values_bytes),
            }
        )

        all_data_chunks.extend([rows_bytes, cols_bytes, values_bytes])
        current_offset += indices_size + len(values_bytes)

    if not tensors_meta:
        header = {"version": FORMAT_VERSION_V3, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    header = {"version": FORMAT_VERSION_V3, "tensors": tensors_meta}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    uncompressed = struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(all_data_chunks)

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=ZSTD_THREADS)
    compressed = compressor.compress(uncompressed)

    comp_size = len(compressed)
    ratio = coo_raw_bytes / comp_size if comp_size > 0 else 1.0

    logger.debug(
        "Sparse delta encoded (v3, COO baseline): %d tensors | %.2f MB -> %.2f MB (%.1fx ratio)",
        len(tensors_meta),
        coo_raw_bytes / (1024 * 1024),
        comp_size / (1024 * 1024),
        ratio,
    )

    return compressed


def decode_sparse_delta_v3(
    compressed: bytes,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Decode v3 sparse delta with 2D COO indices."""
    if not compressed:
        return {}, {}

    decompressor = zstd.ZstdDecompressor()
    uncompressed = decompressor.decompress(compressed)
    if len(uncompressed) < 4:
        raise ValueError("Compressed data too short to contain header length")

    header_len = struct.unpack("<I", uncompressed[:4])[0]
    if len(uncompressed) < 4 + header_len:
        raise ValueError("Compressed data too short to contain full header")

    header = json.loads(uncompressed[4 : 4 + header_len].decode("utf-8"))
    version = header.get("version", 0)
    if version != FORMAT_VERSION_V3:
        raise ValueError(f"Unsupported format version: {version} (expected {FORMAT_VERSION_V3})")

    tensors_meta = header.get("tensors", [])
    if not tensors_meta:
        return {}, {}

    data_start = 4 + header_len
    sparse_tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}

    for meta in tensors_meta:
        name = meta["name"]
        shape = meta["shape"]
        value_dtype = meta["value_dtype"]
        indices_offset = meta["indices_offset"]
        rows_size = meta["rows_size"]
        cols_size = meta["cols_size"]
        values_offset = meta["values_offset"]
        values_size = meta["values_size"]

        rows_bytes = uncompressed[
            data_start + indices_offset : data_start + indices_offset + rows_size
        ]
        cols_bytes = uncompressed[
            data_start + indices_offset + rows_size : data_start
            + indices_offset
            + rows_size
            + cols_size
        ]
        values_bytes = uncompressed[
            data_start + values_offset : data_start + values_offset + values_size
        ]

        rows_delta = np.frombuffer(rows_bytes, dtype=np.int32).copy()
        cols_delta = np.frombuffer(cols_bytes, dtype=np.int32).copy()
        rows, cols = _delta_decode_rows_cols(rows_delta, cols_delta)

        if value_dtype == "bfloat16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.bfloat16)
        elif value_dtype == "float16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.float16)
        else:
            values_np = np.frombuffer(values_bytes, dtype=np.float32).copy()
            values = torch.from_numpy(values_np)

        indices = torch.stack(
            [
                torch.from_numpy(rows.astype(np.int32)),
                torch.from_numpy(cols.astype(np.int32)),
            ],
            dim=0,
        )

        sparse_tensors[f"{name}.indices"] = indices
        sparse_tensors[f"{name}.values"] = values
        shapes[name] = shape

    logger.debug("Sparse delta decoded (v3): %d tensors", len(shapes))
    return sparse_tensors, shapes


# v3.1 format: v3 with index downscaling (uint8 rows, uint16 cols)
def encode_sparse_delta_v3_1(
    sparse_tensors: dict[str, torch.Tensor],
    shapes: dict[str, list[int]],
) -> bytes:
    """Encode sparse delta with 2D COO indices and index downscaling.

    Same as v3 but with automatic dtype selection for indices:
    - Row deltas: uint8 if max <= 255, else int32
    - Col deltas: uint16 if max <= 65535, else int32

    This typically reduces index storage by 62% (uint8+uint16 vs int32+int32).
    """
    if not sparse_tensors:
        header = {"version": FORMAT_VERSION_V3_1, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    tensor_names = sorted({k.rsplit(".", 1)[0] for k in sparse_tensors if k.endswith(".indices")})

    tensors_meta: list[dict[str, Any]] = []
    all_data_chunks: list[bytes] = []
    current_offset = 0
    coo_raw_bytes = 0

    for name in tensor_names:
        indices_key = f"{name}.indices"
        values_key = f"{name}.values"
        if indices_key not in sparse_tensors or values_key not in sparse_tensors:
            continue

        shape = shapes.get(name)
        if not shape:
            raise ValueError(f"Shape missing for tensor {name}")

        indices_tensor = sparse_tensors[indices_key]
        values_tensor = sparse_tensors[values_key]
        if values_tensor.numel() == 0:
            continue

        # COO baseline size: indices are 2x int32 + values (flattening is compression)
        nnz = values_tensor.numel()
        coo_raw_bytes += nnz * (
            COO_INDEX_COMPONENTS * COO_INDEX_BYTES + values_tensor.element_size()
        )

        indices_np = indices_tensor.detach().cpu().numpy()
        rows, cols = _coo_rows_cols_from_indices(indices_np, shape)
        values_np, value_dtype = _values_to_numpy(values_tensor)

        if rows.size != values_np.shape[0]:
            raise ValueError(
                f"Index/value mismatch for {name}: "
                f"{rows.size} indices vs {values_np.shape[0]} values"
            )

        rows_sorted, cols_sorted, values_sorted = _sort_coo(rows, cols, values_np)
        rows_delta, cols_delta = _delta_encode_rows_cols(rows_sorted, cols_sorted)

        # Select optimal dtypes for indices
        row_dtype_name, row_np_dtype = _select_index_dtype(rows_delta, prefer_uint8=True)
        col_dtype_name, col_np_dtype = _select_index_dtype(cols_delta, prefer_uint8=False)

        # Convert to selected dtypes
        rows_bytes = rows_delta.astype(row_np_dtype).tobytes()
        cols_bytes = cols_delta.astype(col_np_dtype).tobytes()
        values_bytes = values_sorted.tobytes()

        indices_size = len(rows_bytes) + len(cols_bytes)
        tensors_meta.append(
            {
                "name": name,
                "shape": shape,
                "num_nonzero": int(rows.size),
                "value_dtype": value_dtype,
                "row_dtype": row_dtype_name,
                "col_dtype": col_dtype_name,
                "indices_offset": current_offset,
                "indices_size": indices_size,
                "rows_size": len(rows_bytes),
                "cols_size": len(cols_bytes),
                "values_offset": current_offset + indices_size,
                "values_size": len(values_bytes),
            }
        )

        all_data_chunks.extend([rows_bytes, cols_bytes, values_bytes])
        current_offset += indices_size + len(values_bytes)

    if not tensors_meta:
        header = {"version": FORMAT_VERSION_V3_1, "tensors": []}
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_len = struct.pack("<I", len(header_bytes))
        compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
        return compressor.compress(header_len + header_bytes)

    header = {"version": FORMAT_VERSION_V3_1, "tensors": tensors_meta}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    uncompressed = struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(all_data_chunks)

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL, threads=ZSTD_THREADS)
    compressed = compressor.compress(uncompressed)

    comp_size = len(compressed)
    ratio = coo_raw_bytes / comp_size if comp_size > 0 else 1.0

    logger.debug(
        "Sparse delta encoded (v3.1, COO baseline): %d tensors | %.2f MB -> %.2f MB (%.1fx ratio)",
        len(tensors_meta),
        coo_raw_bytes / (1024 * 1024),
        comp_size / (1024 * 1024),
        ratio,
    )

    return compressed


def decode_sparse_delta_v3_1(
    compressed: bytes,
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Decode v3.1 sparse delta with downscaled indices."""
    if not compressed:
        return {}, {}

    decompressor = zstd.ZstdDecompressor()
    uncompressed = decompressor.decompress(compressed)
    if len(uncompressed) < 4:
        raise ValueError("Compressed data too short to contain header length")

    header_len = struct.unpack("<I", uncompressed[:4])[0]
    if len(uncompressed) < 4 + header_len:
        raise ValueError("Compressed data too short to contain full header")

    header = json.loads(uncompressed[4 : 4 + header_len].decode("utf-8"))
    version = header.get("version", 0)
    if version != FORMAT_VERSION_V3_1:
        raise ValueError(f"Unsupported format version: {version} (expected {FORMAT_VERSION_V3_1})")

    tensors_meta = header.get("tensors", [])
    if not tensors_meta:
        return {}, {}

    data_start = 4 + header_len
    sparse_tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}

    for meta in tensors_meta:
        name = meta["name"]
        shape = meta["shape"]
        value_dtype = meta["value_dtype"]
        row_dtype_name = meta.get("row_dtype", INDEX_DTYPE_INT32)
        col_dtype_name = meta.get("col_dtype", INDEX_DTYPE_INT32)
        indices_offset = meta["indices_offset"]
        rows_size = meta["rows_size"]
        cols_size = meta["cols_size"]
        values_offset = meta["values_offset"]
        values_size = meta["values_size"]

        # Get numpy dtypes for indices
        row_np_dtype = _INDEX_NP_DTYPES.get(row_dtype_name, np.dtype(np.int32))
        col_np_dtype = _INDEX_NP_DTYPES.get(col_dtype_name, np.dtype(np.int32))

        rows_bytes = uncompressed[
            data_start + indices_offset : data_start + indices_offset + rows_size
        ]
        cols_bytes = uncompressed[
            data_start + indices_offset + rows_size : data_start
            + indices_offset
            + rows_size
            + cols_size
        ]
        values_bytes = uncompressed[
            data_start + values_offset : data_start + values_offset + values_size
        ]

        # Decode with correct dtypes, then convert to int32 for delta decoding
        rows_delta = np.frombuffer(rows_bytes, dtype=row_np_dtype).astype(np.int32).copy()
        cols_delta = np.frombuffer(cols_bytes, dtype=col_np_dtype).astype(np.int32).copy()
        rows, cols = _delta_decode_rows_cols(rows_delta, cols_delta)

        if value_dtype == "bfloat16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.bfloat16)
        elif value_dtype == "float16":
            values_np = np.frombuffer(values_bytes, dtype=np.int16).copy()
            values = torch.from_numpy(values_np).view(torch.float16)
        else:
            values_np = np.frombuffer(values_bytes, dtype=np.float32).copy()
            values = torch.from_numpy(values_np)

        indices = torch.stack(
            [
                torch.from_numpy(rows.astype(np.int32)),
                torch.from_numpy(cols.astype(np.int32)),
            ],
            dim=0,
        )

        sparse_tensors[f"{name}.indices"] = indices
        sparse_tensors[f"{name}.values"] = values
        shapes[name] = shape

    logger.debug("Sparse delta decoded (v3.1): %d tensors", len(shapes))
    return sparse_tensors, shapes


# Public API aliases - v3.1 is now the optimal encoder
encode_optimal = encode_sparse_delta_v3_1
decode_optimal = decode_sparse_delta_v3_1
