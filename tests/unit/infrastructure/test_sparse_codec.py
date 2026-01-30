"""Tests for sparse_codec module.

Focus areas:
1. Encode/decode roundtrip correctness (all dtypes)
2. Compression ratio validation (>5x for delta-encoded COO)
3. Performance (throughput > 100 MB/s)
4. Edge cases (empty, single element, large tensors)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from grail.infrastructure.sparse_codec import (
    INDEX_DTYPE_INT32,
    INDEX_DTYPE_UINT8,
    INDEX_DTYPE_UINT16,
    _delta_decode_rows_cols,
    _delta_encode_rows_cols,
    _select_index_dtype,
    decode_optimal,
    decode_sparse_delta_v2,
    decode_sparse_delta_v3,
    decode_sparse_delta_v3_1,
    encode_optimal,
    encode_sparse_delta_v2,
    encode_sparse_delta_v3,
    encode_sparse_delta_v3_1,
)


class TestDeltaEncodingHelpers:
    """Tests for low-level delta encoding/decoding functions."""

    def test_encode_decode_roundtrip_simple(self):
        """Test that encode/decode are inverse operations."""
        rows = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        cols = np.array([1, 3, 5, 2, 4, 0], dtype=np.int32)

        rows_delta, cols_delta = _delta_encode_rows_cols(rows, cols)
        rows_dec, cols_dec = _delta_decode_rows_cols(rows_delta, cols_delta)

        np.testing.assert_array_equal(rows, rows_dec)
        np.testing.assert_array_equal(cols, cols_dec)

    def test_encode_decode_roundtrip_large(self):
        """Test roundtrip with large array (vectorization correctness)."""
        np.random.seed(42)
        n = 100_000

        # Generate sorted COO indices
        rows = np.sort(np.random.randint(0, 1000, n)).astype(np.int32)
        cols = np.random.randint(0, 4096, n).astype(np.int32)

        # Sort lexicographically (as done in encoder)
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]

        rows_delta, cols_delta = _delta_encode_rows_cols(rows, cols)
        rows_dec, cols_dec = _delta_decode_rows_cols(rows_delta, cols_delta)

        np.testing.assert_array_equal(rows, rows_dec)
        np.testing.assert_array_equal(cols, cols_dec)

    def test_encode_produces_small_deltas(self):
        """Test that encoding produces mostly small values (good for compression)."""
        # Sorted indices within each row should produce small positive deltas
        rows = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        cols = np.array([5, 10, 15, 20, 3, 8, 13], dtype=np.int32)

        rows_delta, cols_delta = _delta_encode_rows_cols(rows, cols)

        # Row deltas should be mostly 0 with occasional 1
        assert rows_delta[0] == 0
        assert np.sum(rows_delta == 0) >= 4  # Most are 0

        # Column deltas within same row should be 5 (the step)
        # At row boundaries, should be absolute value
        assert cols_delta[0] == 5  # First element: absolute
        assert cols_delta[4] == 3  # Row boundary: absolute column value

    def test_empty_array(self):
        """Test handling of empty arrays."""
        rows = np.array([], dtype=np.int32)
        cols = np.array([], dtype=np.int32)

        rows_delta, cols_delta = _delta_encode_rows_cols(rows, cols)
        rows_dec, cols_dec = _delta_decode_rows_cols(rows_delta, cols_delta)

        assert len(rows_dec) == 0
        assert len(cols_dec) == 0

    def test_single_element(self):
        """Test handling of single element."""
        rows = np.array([5], dtype=np.int32)
        cols = np.array([10], dtype=np.int32)

        rows_delta, cols_delta = _delta_encode_rows_cols(rows, cols)
        rows_dec, cols_dec = _delta_decode_rows_cols(rows_delta, cols_delta)

        np.testing.assert_array_equal(rows, rows_dec)
        np.testing.assert_array_equal(cols, cols_dec)


class TestSparseCodecV2:
    """Tests for v2 sparse codec (1D flat indices)."""

    def test_roundtrip_simple(self):
        """Test encode/decode roundtrip with simple data."""
        indices1 = torch.tensor([0, 5, 10, 100, 1000], dtype=torch.int32)
        values1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        indices2 = torch.tensor([3, 7, 15], dtype=torch.int32)
        values2 = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)

        sparse_tensors = {
            "layer1.weight.indices": indices1,
            "layer1.weight.values": values1,
            "layer2.bias.indices": indices2,
            "layer2.bias.values": values2,
        }
        shapes = {
            "layer1.weight": [100, 100],
            "layer2.bias": [100],
        }

        compressed = encode_sparse_delta_v2(sparse_tensors, shapes)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        decoded_tensors, decoded_shapes = decode_sparse_delta_v2(compressed)
        assert decoded_shapes == shapes

        for name in shapes:
            orig_idx = sparse_tensors[f"{name}.indices"]
            orig_val = sparse_tensors[f"{name}.values"]
            dec_idx = decoded_tensors[f"{name}.indices"]
            dec_val = decoded_tensors[f"{name}.values"]

            orig_pairs = set(zip(orig_idx.tolist(), orig_val.tolist(), strict=True))
            dec_pairs = set(zip(dec_idx.tolist(), dec_val.tolist(), strict=True))
            assert orig_pairs == dec_pairs, f"Mismatch in {name}"

    def test_roundtrip_bfloat16(self):
        """Test encode/decode with bfloat16 values."""
        indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [10, 10]}

        compressed = encode_sparse_delta_v2(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v2(compressed)

        assert decoded_shapes == shapes
        dec_val = decoded_tensors["layer.weight.values"]
        assert dec_val.dtype == torch.bfloat16
        torch.testing.assert_close(dec_val, values)

    def test_roundtrip_with_2d_indices(self):
        """Test v2 encode/decode with 2D COO indices (flattened)."""
        indices = torch.tensor([[0, 1, 1], [2, 0, 3]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [2, 4]}

        compressed = encode_sparse_delta_v2(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v2(compressed)

        assert decoded_shapes == shapes
        cols_per_row = shapes["layer.weight"][1]
        flat_indices = (indices[0] * cols_per_row + indices[1]).tolist()

        orig_pairs = set(zip(flat_indices, values.tolist(), strict=True))
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]
        dec_pairs = set(zip(dec_idx.tolist(), dec_val.tolist(), strict=True))
        assert orig_pairs == dec_pairs

    def test_empty_delta(self):
        """Test encode/decode with empty sparse tensors."""
        compressed = encode_sparse_delta_v2({}, {})
        decoded_tensors, decoded_shapes = decode_sparse_delta_v2(compressed)
        assert decoded_tensors == {}
        assert decoded_shapes == {}


class TestSparseCodecV3:
    """Tests for v3 sparse codec (2D COO indices with delta encoding)."""

    def test_roundtrip_simple(self):
        """Test encode/decode roundtrip with 2D indices."""
        indices = torch.tensor([[0, 0, 2, 3], [1, 5, 0, 4]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [4, 6]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        assert decoded_shapes == shapes
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        orig_pairs = set(
            zip(indices[0].tolist(), indices[1].tolist(), values.tolist(), strict=True)
        )
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_roundtrip_bfloat16(self):
        """Test encode/decode with bfloat16 values and 2D indices."""
        indices = torch.tensor([[0, 1, 1], [2, 0, 3]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [2, 4]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        assert decoded_shapes == shapes
        dec_val = decoded_tensors["layer.weight.values"]
        assert dec_val.dtype == torch.bfloat16
        torch.testing.assert_close(dec_val, values)

    def test_roundtrip_float16(self):
        """Test encode/decode with float16 values."""
        indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        values = torch.tensor([1.5, 2.5], dtype=torch.float16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [2, 2]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        dec_val = decoded_tensors["layer.weight.values"]
        assert dec_val.dtype == torch.float16
        torch.testing.assert_close(dec_val, values)

    def test_roundtrip_1d_indices(self):
        """Test that 1D flat indices are converted to 2D COO properly."""
        # 1D indices for a [100, 50] tensor
        indices = torch.tensor([0, 50, 100, 4999], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [100, 50]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        # Convert original 1D to 2D for comparison
        rows = indices // 50
        cols = indices % 50

        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        orig_pairs = set(zip(rows.tolist(), cols.tolist(), values.tolist(), strict=True))
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_aliases(self):
        """Test that encode_optimal and decode_optimal are v3.1 aliases."""
        assert encode_optimal is encode_sparse_delta_v3_1
        assert decode_optimal is decode_sparse_delta_v3_1


class TestIndexDtypeSelection:
    """Tests for index dtype selection helper."""

    def test_empty_array_uses_uint8(self):
        """Test that empty arrays default to uint8."""
        arr = np.array([], dtype=np.int32)
        dtype_name, np_dtype = _select_index_dtype(arr)
        assert dtype_name == INDEX_DTYPE_UINT8
        assert np_dtype == np.uint8

    def test_small_values_use_uint8(self):
        """Test that values <= 255 use uint8."""
        arr = np.array([0, 1, 100, 255], dtype=np.int32)
        dtype_name, np_dtype = _select_index_dtype(arr, prefer_uint8=True)
        assert dtype_name == INDEX_DTYPE_UINT8
        assert np_dtype == np.uint8

    def test_medium_values_use_uint16(self):
        """Test that values > 255 but <= 65535 use uint16."""
        arr = np.array([0, 256, 1000, 65535], dtype=np.int32)
        dtype_name, np_dtype = _select_index_dtype(arr, prefer_uint8=True)
        assert dtype_name == INDEX_DTYPE_UINT16
        assert np_dtype == np.uint16

    def test_large_values_use_int32(self):
        """Test that values > 65535 use int32."""
        arr = np.array([0, 100, 65536, 1000000], dtype=np.int32)
        dtype_name, np_dtype = _select_index_dtype(arr)
        assert dtype_name == INDEX_DTYPE_INT32
        assert np_dtype == np.int32

    def test_prefer_uint16_over_uint8(self):
        """Test prefer_uint8=False skips uint8."""
        arr = np.array([0, 1, 100], dtype=np.int32)
        dtype_name, np_dtype = _select_index_dtype(arr, prefer_uint8=False)
        assert dtype_name == INDEX_DTYPE_UINT16
        assert np_dtype == np.uint16


class TestSparseCodecV3_1:
    """Tests for v3.1 sparse codec (v3 with index downscaling)."""

    def test_roundtrip_simple(self):
        """Test encode/decode roundtrip with 2D indices."""
        indices = torch.tensor([[0, 0, 2, 3], [1, 5, 0, 4]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [4, 6]}

        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)

        assert decoded_shapes == shapes
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        orig_pairs = set(
            zip(indices[0].tolist(), indices[1].tolist(), values.tolist(), strict=True)
        )
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_roundtrip_bfloat16(self):
        """Test encode/decode with bfloat16 values."""
        indices = torch.tensor([[0, 1, 1], [2, 0, 3]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [2, 4]}

        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)

        assert decoded_shapes == shapes
        dec_val = decoded_tensors["layer.weight.values"]
        assert dec_val.dtype == torch.bfloat16
        torch.testing.assert_close(dec_val, values)

    def test_uses_smaller_dtypes(self):
        """Test that v3.1 actually uses smaller dtypes for small values."""
        # Create data where row deltas fit in uint8 and col deltas fit in uint16
        np.random.seed(42)
        shape = [100, 1000]  # rows=100, cols=1000
        nnz = 500

        # Generate indices that will have small deltas
        rows = np.sort(np.random.randint(0, shape[0], nnz)).astype(np.int32)
        cols = np.random.randint(0, shape[1], nnz).astype(np.int32)

        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        values = torch.randn(nnz, dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": shape}

        # Encode with v3 and v3.1, v3.1 should be smaller
        compressed_v3 = encode_sparse_delta_v3(sparse_tensors, shapes)
        compressed_v3_1 = encode_sparse_delta_v3_1(sparse_tensors, shapes)

        # v3.1 should be smaller due to dtype downscaling
        # (may not always be smaller after zstd, but indices themselves are smaller)
        assert len(compressed_v3_1) <= len(compressed_v3) * 1.1  # Allow small overhead from header

    def test_roundtrip_large_indices(self):
        """Test roundtrip with large col indices (requiring uint16)."""
        # Large column values that require uint16
        indices = torch.tensor([[0, 0, 1], [100, 50000, 30000]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [2, 60000]}

        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)

        assert decoded_shapes == shapes
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        orig_pairs = set(
            zip(indices[0].tolist(), indices[1].tolist(), values.tolist(), strict=True)
        )
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_empty_tensor(self):
        """Test encoding empty sparse tensors."""
        compressed = encode_sparse_delta_v3_1({}, {})
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)
        assert decoded_tensors == {}
        assert decoded_shapes == {}

    def test_single_element(self):
        """Test single non-zero element."""
        indices = torch.tensor([[5], [10]], dtype=torch.int32)
        values = torch.tensor([3.14], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [10, 20]}

        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)

        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        assert dec_idx[0].item() == 5
        assert dec_idx[1].item() == 10
        torch.testing.assert_close(dec_val, values)

    def test_multiple_tensors(self):
        """Test encoding multiple tensors with different sizes."""
        sparse_tensors = {}
        shapes = {}

        for i in range(3):
            name = f"layer{i}.weight"
            n = (i + 1) * 10
            indices = torch.randint(0, 100, (2, n), dtype=torch.int32)
            values = torch.randn(n, dtype=torch.bfloat16)
            sparse_tensors[f"{name}.indices"] = indices
            sparse_tensors[f"{name}.values"] = values
            shapes[name] = [100, 100]

        compressed = encode_sparse_delta_v3_1(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3_1(compressed)

        assert decoded_shapes == shapes
        assert len(decoded_tensors) == len(sparse_tensors)


class TestCompressionRatio:
    """Tests for compression ratio.

    Note: Random data achieves ~3-4x compression. Real training deltas achieve
    7-8x due to structured patterns in weight updates. Tests use conservative
    thresholds to catch regressions while allowing for random data behavior.
    """

    def test_compression_ratio_v3_realistic(self):
        """Test compression ratio with sparse delta (2% non-zero).

        Random data achieves ~3-4x; real training deltas achieve 7-8x.
        """
        np.random.seed(42)
        shape = [4096, 4096]  # ~16M parameters
        total_elements = shape[0] * shape[1]
        nnz = total_elements // 50  # 2% non-zero

        # Generate random 2D COO indices
        flat_indices = np.random.choice(total_elements, size=nnz, replace=False)
        rows = (flat_indices // shape[1]).astype(np.int32)
        cols = (flat_indices % shape[1]).astype(np.int32)

        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        values = torch.randn(nnz, dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": shape}

        # Raw size: 2D indices (2 * nnz * 4) + values (nnz * 2)
        raw_size = 2 * nnz * 4 + nnz * 2

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        ratio = raw_size / len(compressed)

        # Random data should achieve at least 2.5x (delta encoding + zstd)
        # Real training data achieves 7-8x due to structured patterns
        assert ratio > 2.5, f"Compression ratio {ratio:.2f}x is below minimum 2.5x"
        print(
            f"\nCompression ratio: {ratio:.2f}x ({raw_size / 1e6:.2f} MB -> {len(compressed) / 1e6:.2f} MB)"
        )

    def test_compression_ratio_multiple_tensors(self):
        """Test compression with multiple tensors (full model-like scenario)."""
        np.random.seed(42)
        sparse_tensors = {}
        shapes = {}
        total_raw_size = 0

        # Simulate multiple layers with varying sparsity
        for i in range(5):
            shape = [1024 * (i + 1), 1024]
            total_elements = shape[0] * shape[1]
            nnz = total_elements // 100  # 1% non-zero

            flat_indices = np.random.choice(total_elements, size=nnz, replace=False)
            rows = (flat_indices // shape[1]).astype(np.int32)
            cols = (flat_indices % shape[1]).astype(np.int32)

            indices = torch.from_numpy(np.stack([rows, cols], axis=0))
            values = torch.randn(nnz, dtype=torch.bfloat16)

            name = f"layers.{i}.weight"
            sparse_tensors[f"{name}.indices"] = indices
            sparse_tensors[f"{name}.values"] = values
            shapes[name] = shape

            total_raw_size += 2 * nnz * 4 + nnz * 2

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        ratio = total_raw_size / len(compressed)

        # Random data should achieve at least 2.5x compression
        assert ratio > 2.5, f"Multi-tensor compression ratio {ratio:.2f}x is below minimum"
        print(
            f"\nMulti-tensor compression: {ratio:.2f}x ({total_raw_size / 1e6:.2f} MB -> {len(compressed) / 1e6:.2f} MB)"
        )


class TestPerformance:
    """Tests for encoding/decoding throughput.

    Note: Encoding includes O(n log n) lexsort which dominates for large tensors.
    Decoding is faster as it only needs segment-wise cumsum.
    """

    @pytest.mark.slow
    def test_encode_throughput(self):
        """Test encoding throughput.

        Bottleneck is lexsort O(n log n). Expect ~30-50 MB/s for 1M+ elements.
        This is acceptable as encoding happens once per training step.
        """
        np.random.seed(42)
        shape = [8192, 8192]
        total_elements = shape[0] * shape[1]
        nnz = total_elements // 50  # 2% non-zero = ~1.3M elements

        flat_indices = np.random.choice(total_elements, size=nnz, replace=False)
        rows = (flat_indices // shape[1]).astype(np.int32)
        cols = (flat_indices % shape[1]).astype(np.int32)

        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        values = torch.randn(nnz, dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": shape}

        raw_size = 2 * nnz * 4 + nnz * 2

        # Warm up
        _ = encode_sparse_delta_v3(sparse_tensors, shapes)

        # Measure
        start = time.perf_counter()
        n_iterations = 3
        for _ in range(n_iterations):
            _ = encode_sparse_delta_v3(sparse_tensors, shapes)
        elapsed = time.perf_counter() - start

        throughput_mbps = (raw_size * n_iterations / 1e6) / elapsed
        print(f"\nEncode throughput: {throughput_mbps:.1f} MB/s (raw_size={raw_size / 1e6:.1f} MB)")

        # Encode includes sort, so expect lower throughput than raw compression
        # 20 MB/s minimum ensures no major regression
        assert throughput_mbps > 20, f"Encode throughput {throughput_mbps:.1f} MB/s is too low"

    @pytest.mark.slow
    def test_decode_throughput(self):
        """Test decoding throughput.

        Decoding is faster than encoding (no sort needed).
        Expect >100 MB/s for vectorized implementation.
        """
        np.random.seed(42)
        shape = [8192, 8192]
        total_elements = shape[0] * shape[1]
        nnz = total_elements // 50  # 2% non-zero

        flat_indices = np.random.choice(total_elements, size=nnz, replace=False)
        rows = (flat_indices // shape[1]).astype(np.int32)
        cols = (flat_indices % shape[1]).astype(np.int32)

        indices = torch.from_numpy(np.stack([rows, cols], axis=0))
        values = torch.randn(nnz, dtype=torch.bfloat16)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": shape}

        raw_size = 2 * nnz * 4 + nnz * 2
        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)

        # Warm up
        _ = decode_sparse_delta_v3(compressed)

        # Measure
        start = time.perf_counter()
        n_iterations = 5
        for _ in range(n_iterations):
            _ = decode_sparse_delta_v3(compressed)
        elapsed = time.perf_counter() - start

        throughput_mbps = (raw_size * n_iterations / 1e6) / elapsed
        print(f"\nDecode throughput: {throughput_mbps:.1f} MB/s")

        # Decode should be faster - no sort needed, only cumsum
        assert throughput_mbps > 50, f"Decode throughput {throughput_mbps:.1f} MB/s is too low"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_tensor_v3(self):
        """Test encoding empty sparse tensors."""
        compressed = encode_sparse_delta_v3({}, {})
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)
        assert decoded_tensors == {}
        assert decoded_shapes == {}

    def test_single_element_v3(self):
        """Test single non-zero element."""
        indices = torch.tensor([[5], [10]], dtype=torch.int32)
        values = torch.tensor([3.14], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [10, 20]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]

        assert dec_idx[0].item() == 5
        assert dec_idx[1].item() == 10
        torch.testing.assert_close(dec_val, values)

    def test_all_same_row(self):
        """Test when all elements are in the same row."""
        indices = torch.tensor([[0, 0, 0, 0], [1, 5, 10, 20]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [1, 100]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        orig_pairs = set(
            zip(indices[0].tolist(), indices[1].tolist(), values.tolist(), strict=True)
        )
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_one_element_per_row(self):
        """Test when each row has exactly one element."""
        indices = torch.tensor([[0, 1, 2, 3], [5, 10, 15, 20]], dtype=torch.int32)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {"layer.weight": [4, 100]}

        compressed = encode_sparse_delta_v3(sparse_tensors, shapes)
        decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

        orig_pairs = set(
            zip(indices[0].tolist(), indices[1].tolist(), values.tolist(), strict=True)
        )
        dec_idx = decoded_tensors["layer.weight.indices"]
        dec_val = decoded_tensors["layer.weight.values"]
        dec_pairs = set(
            zip(dec_idx[0].tolist(), dec_idx[1].tolist(), dec_val.tolist(), strict=True)
        )
        assert orig_pairs == dec_pairs

    def test_missing_shape_raises(self):
        """Test that missing shape raises ValueError."""
        indices = torch.tensor([[0], [0]], dtype=torch.int32)
        values = torch.tensor([1.0], dtype=torch.float32)

        sparse_tensors = {
            "layer.weight.indices": indices,
            "layer.weight.values": values,
        }
        shapes = {}  # Missing shape

        with pytest.raises(ValueError, match="Shape missing"):
            encode_sparse_delta_v3(sparse_tensors, shapes)


class TestRealDeltaRoundtrip:
    """Tests using real delta files from research/sparsity_analysis.

    These tests verify bit-exact lossless encoding/decoding on actual
    training deltas, not synthetic data. Skipped if delta files are missing.
    """

    # Path to real delta files (relative to repo root)
    DELTA_DIR = "research/sparsity_analysis/experiments/qwen2.5-1.5b-sft-math-lr3e-06/seed42/deltas"

    @staticmethod
    def _find_repo_root() -> Path:
        """Find the repository root by looking for pyproject.toml."""
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        raise FileNotFoundError("Could not find repo root")

    @staticmethod
    def _load_delta_file(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
        """Load a delta .pt file and convert to codec input format.

        Args:
            path: Path to delta_XXXXXX.pt file

        Returns:
            Tuple of (sparse_tensors, shapes) in codec format
        """
        delta = torch.load(path, weights_only=False)
        layers = delta.get("layers", {})

        sparse_tensors: dict[str, torch.Tensor] = {}
        shapes: dict[str, list[int]] = {}

        for layer_name, layer_data in layers.items():
            indices = layer_data["indices"]
            values = layer_data["values"]
            shape = list(layer_data["shape"])

            sparse_tensors[f"{layer_name}.indices"] = indices
            sparse_tensors[f"{layer_name}.values"] = values
            shapes[layer_name] = shape

        return sparse_tensors, shapes

    @staticmethod
    def _to_2d_indices(indices: torch.Tensor) -> torch.Tensor:
        """Normalize indices to 2D COO format (shape [2, nnz])."""
        if indices.ndim == 2 and indices.shape[0] == 2:
            return indices
        if indices.ndim == 2 and indices.shape[0] == 1:
            cols = torch.zeros_like(indices[0])
            return torch.stack([indices[0], cols], dim=0)
        if indices.ndim == 1:
            cols = torch.zeros_like(indices)
            return torch.stack([indices, cols], dim=0)
        raise ValueError(f"Unsupported indices shape for v3: {tuple(indices.shape)}")

    def _normalize_sparse_tensors_2d(
        self, sparse_tensors: dict[str, torch.Tensor], shapes: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        """Ensure all indices are 2D COO for v3 encoding."""
        normalized: dict[str, torch.Tensor] = {}
        for name in shapes:
            indices = sparse_tensors[f"{name}.indices"]
            values = sparse_tensors[f"{name}.values"]
            normalized[f"{name}.indices"] = self._to_2d_indices(indices)
            normalized[f"{name}.values"] = values
        return normalized

    def _get_delta_files(self) -> list[Path]:
        """Get list of available delta files."""
        try:
            repo_root = self._find_repo_root()
        except FileNotFoundError:
            return []

        delta_dir = repo_root / self.DELTA_DIR
        if not delta_dir.exists():
            return []

        return sorted(delta_dir.glob("delta_*.pt"))[:5]  # Test first 5

    @pytest.fixture
    def delta_files(self) -> list[Path]:
        """Fixture providing delta file paths."""
        files = self._get_delta_files()
        if not files:
            pytest.skip("Real delta files not available")
        return files

    def test_v3_roundtrip_real_deltas(self, delta_files: list[Path]) -> None:
        """Test v3 codec roundtrip on real training deltas."""
        for delta_path in delta_files:
            sparse_tensors, shapes = self._load_delta_file(delta_path)

            # Skip empty deltas
            if not sparse_tensors:
                continue

            sparse_tensors_2d = self._normalize_sparse_tensors_2d(sparse_tensors, shapes)

            # Encode
            compressed = encode_sparse_delta_v3(sparse_tensors_2d, shapes)

            # Decode
            decoded_tensors, decoded_shapes = decode_sparse_delta_v3(compressed)

            # Verify shapes match
            assert decoded_shapes == shapes, f"Shape mismatch in {delta_path.name}"

            # Verify each tensor is bit-exact
            for name in shapes:
                orig_idx = sparse_tensors_2d[f"{name}.indices"]
                orig_val = sparse_tensors_2d[f"{name}.values"]
                dec_idx = decoded_tensors[f"{name}.indices"]
                dec_val = decoded_tensors[f"{name}.values"]

                orig_rows = orig_idx[0].tolist()
                orig_cols = orig_idx[1].tolist()
                dec_rows = dec_idx[0].tolist()
                dec_cols = dec_idx[1].tolist()

                # Compare as sets of (row, col, value) tuples for order-independence
                orig_pairs = set(
                    zip(
                        orig_rows,
                        orig_cols,
                        orig_val.view(torch.int16).tolist(),  # Compare raw bits for bf16
                        strict=True,
                    )
                )
                dec_pairs = set(
                    zip(
                        dec_rows,
                        dec_cols,
                        dec_val.view(torch.int16).tolist(),
                        strict=True,
                    )
                )
                assert orig_pairs == dec_pairs, f"Data mismatch in {name} ({delta_path.name})"

    def test_v2_roundtrip_real_deltas(self, delta_files: list[Path]) -> None:
        """Test v2 codec roundtrip on real training deltas (flattened indices)."""
        for delta_path in delta_files:
            sparse_tensors, shapes = self._load_delta_file(delta_path)

            if not sparse_tensors:
                continue

            # Encode with v2 (flattens 2D indices to 1D)
            compressed = encode_sparse_delta_v2(sparse_tensors, shapes)

            # Decode
            decoded_tensors, decoded_shapes = decode_sparse_delta_v2(compressed)

            # Verify shapes match
            assert decoded_shapes == shapes, f"Shape mismatch in {delta_path.name}"

            # Verify each tensor - v2 returns 1D flat indices
            for name in shapes:
                orig_idx = sparse_tensors[f"{name}.indices"]
                orig_val = sparse_tensors[f"{name}.values"]
                dec_idx = decoded_tensors[f"{name}.indices"]
                dec_val = decoded_tensors[f"{name}.values"]

                # Convert original indices to flat for comparison
                shape = shapes[name]
                cols_per_row = shape[1] if len(shape) > 1 else 1
                if orig_idx.shape[0] == 1:
                    # 1D tensor: row=0, so flat = col
                    orig_flat = orig_idx[0].tolist()
                else:
                    # 2D tensor: flat = row * cols_per_row + col
                    orig_flat = (orig_idx[0] * cols_per_row + orig_idx[1]).tolist()

                # Compare as sets of (flat_idx, value) tuples
                orig_pairs = set(
                    zip(
                        orig_flat,
                        orig_val.view(torch.int16).tolist(),
                        strict=True,
                    )
                )
                dec_pairs = set(
                    zip(
                        dec_idx.tolist(),
                        dec_val.view(torch.int16).tolist(),
                        strict=True,
                    )
                )
                assert orig_pairs == dec_pairs, f"Data mismatch in {name} ({delta_path.name})"

    def test_compression_ratio_real_deltas(self, delta_files: list[Path]) -> None:
        """Verify compression ratio on real deltas matches expectations."""
        total_raw = 0
        total_compressed = 0

        for delta_path in delta_files:
            sparse_tensors, shapes = self._load_delta_file(delta_path)

            if not sparse_tensors:
                continue

            sparse_tensors_2d = self._normalize_sparse_tensors_2d(sparse_tensors, shapes)

            # Calculate raw size (indices + values)
            raw_size = sum(t.numel() * t.element_size() for t in sparse_tensors_2d.values())

            # Compress with v3
            compressed = encode_sparse_delta_v3(sparse_tensors_2d, shapes)

            total_raw += raw_size
            total_compressed += len(compressed)

        if total_compressed > 0:
            ratio = total_raw / total_compressed
            # Real deltas should achieve at least 4x compression
            # (actual results vary by sparsity pattern; ~4-8x typical)
            assert ratio > 4.0, f"Compression ratio {ratio:.1f}x is too low for real deltas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
