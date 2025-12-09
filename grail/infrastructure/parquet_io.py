"""Parquet serialization for GRAIL rollout data.

Provides efficient binary serialization of window data using Apache Arrow/Parquet
with nested types to preserve the rollout structure while achieving better
compression and faster I/O than JSON.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Minimum valid Parquet file size (magic bytes + minimal footer)
_MIN_PARQUET_SIZE = 12


class ParquetError(ValueError):
    """Raised when Parquet data is invalid or corrupt."""

    pass


# --------------------------------------------------------------------------- #
#                           Schema Definitions                                #
# --------------------------------------------------------------------------- #

# Model metadata nested structure
MODEL_SCHEMA = pa.struct(
    [
        ("name", pa.string()),
        ("layer_index", pa.int32()),
    ]
)

# Rollout data nested structure
ROLLOUT_SCHEMA = pa.struct(
    [
        ("trajectory", pa.string()),  # JSON-encoded for complex nested lists
        ("total_reward", pa.float64()),
        ("advantage", pa.float64()),
        ("success", pa.bool_()),
        ("token_logprobs", pa.list_(pa.float64())),
        ("prompt_length", pa.int32()),
        ("completion_length", pa.int32()),
        ("satisfied_clauses", pa.int32()),
        ("assignment", pa.list_(pa.bool_())),
    ]
)

# Commit structure (contains model and rollout)
COMMIT_SCHEMA = pa.struct(
    [
        ("tokens", pa.list_(pa.int32())),
        ("commitments", pa.string()),  # JSON-encoded for variable structure
        ("proof_version", pa.string()),
        ("model", MODEL_SCHEMA),
        ("signature", pa.string()),
        ("beacon", pa.string()),  # JSON-encoded for variable structure
        ("rollout", ROLLOUT_SCHEMA),
    ]
)

# Single inference/rollout record schema
INFERENCE_SCHEMA = pa.schema(
    [
        ("window_start", pa.int64()),
        ("block", pa.int64()),
        ("nonce", pa.int64()),
        ("block_hash", pa.string()),
        ("randomness", pa.string()),
        ("use_drand", pa.bool_()),
        ("rollout_group", pa.int64()),
        ("rollout_index", pa.int32()),
        ("total_in_group", pa.int32()),
        ("checkpoint_window", pa.int64()),
        ("commit", COMMIT_SCHEMA),
        ("timestamp", pa.float64()),
        ("challenge", pa.string()),
        ("hotkey", pa.string()),
        ("signature", pa.string()),
    ]
)


# --------------------------------------------------------------------------- #
#                        Serialization Helpers                                #
# --------------------------------------------------------------------------- #


def _convert_inference_to_row(inference: dict[str, Any]) -> dict[str, Any]:
    """Convert a single inference dict to a Parquet-compatible row.

    Handles nested structures by either preserving them (for simple types)
    or JSON-encoding them (for complex/variable structures).

    Args:
        inference: Raw inference dictionary from miner

    Returns:
        Dictionary with Parquet-compatible types
    """
    commit = inference.get("commit", {})
    rollout_data = commit.get("rollout", {})
    model_data = commit.get("model", {})

    # Handle token_logprobs - ensure it's a list of floats
    token_logprobs = rollout_data.get("token_logprobs")
    if token_logprobs is None:
        token_logprobs = []
    elif not isinstance(token_logprobs, list):
        token_logprobs = list(token_logprobs) if hasattr(token_logprobs, "__iter__") else []

    # Handle assignment - ensure it's a list of bools
    assignment = rollout_data.get("assignment")
    if assignment is None:
        assignment = []
    elif not isinstance(assignment, list):
        assignment = list(assignment) if hasattr(assignment, "__iter__") else []

    # Handle tokens - ensure it's a list of ints
    tokens = commit.get("tokens")
    if tokens is None:
        tokens = []
    elif not isinstance(tokens, list):
        tokens = list(tokens) if hasattr(tokens, "__iter__") else []

    # Build the row with proper nesting
    row = {
        "window_start": int(inference.get("window_start", 0)),
        "block": int(inference.get("block", 0)),
        "nonce": int(inference.get("nonce", 0)),
        "block_hash": str(inference.get("block_hash", "")),
        "randomness": str(inference.get("randomness", "")),
        "use_drand": bool(inference.get("use_drand", False)),
        "rollout_group": int(inference.get("rollout_group", 0)),
        "rollout_index": int(inference.get("rollout_index", 0)),
        "total_in_group": int(inference.get("total_in_group", 0)),
        "checkpoint_window": int(inference.get("checkpoint_window", 0)),
        "commit": {
            "tokens": [int(t) for t in tokens],
            "commitments": json.dumps(commit.get("commitments", [])),
            "proof_version": str(commit.get("proof_version", "")),
            "model": {
                "name": str(model_data.get("name", "")),
                "layer_index": int(model_data.get("layer_index", 0)),
            },
            "signature": str(commit.get("signature", "")),
            "beacon": json.dumps(commit.get("beacon", {})),
            "rollout": {
                "trajectory": json.dumps(rollout_data.get("trajectory", [])),
                "total_reward": float(rollout_data.get("total_reward", 0.0)),
                "advantage": float(rollout_data.get("advantage", 0.0)),
                "success": bool(rollout_data.get("success", False)),
                "token_logprobs": [float(lp) for lp in token_logprobs],
                "prompt_length": int(rollout_data.get("prompt_length", 0)),
                "completion_length": int(rollout_data.get("completion_length", 0)),
                "satisfied_clauses": int(rollout_data.get("satisfied_clauses", 0)),
                "assignment": [bool(a) for a in assignment],
            },
        },
        "timestamp": float(inference.get("timestamp", 0.0)),
        "challenge": str(inference.get("challenge", "")),
        "hotkey": str(inference.get("hotkey", "")),
        "signature": str(inference.get("signature", "")),
    }
    return row


def _convert_row_to_inference(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a Parquet row back to the original inference dict format.

    Reverses the JSON encoding of complex fields.

    Args:
        row: Row dictionary from Parquet table

    Returns:
        Original inference dictionary format
    """
    commit = row.get("commit", {})
    rollout_data = commit.get("rollout", {})
    model_data = commit.get("model", {})

    # Decode JSON-encoded fields
    commitments = commit.get("commitments", "[]")
    if isinstance(commitments, str):
        commitments = json.loads(commitments)

    beacon = commit.get("beacon", "{}")
    if isinstance(beacon, str):
        beacon = json.loads(beacon)

    trajectory = rollout_data.get("trajectory", "[]")
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)

    return {
        "window_start": row.get("window_start"),
        "block": row.get("block"),
        "nonce": row.get("nonce"),
        "block_hash": row.get("block_hash"),
        "randomness": row.get("randomness"),
        "use_drand": row.get("use_drand"),
        "rollout_group": row.get("rollout_group"),
        "rollout_index": row.get("rollout_index"),
        "total_in_group": row.get("total_in_group"),
        "checkpoint_window": row.get("checkpoint_window"),
        "commit": {
            "tokens": list(commit.get("tokens", [])),
            "commitments": commitments,
            "proof_version": commit.get("proof_version"),
            "model": {
                "name": model_data.get("name"),
                "layer_index": model_data.get("layer_index"),
            },
            "signature": commit.get("signature"),
            "beacon": beacon,
            "rollout": {
                "trajectory": trajectory,
                "total_reward": rollout_data.get("total_reward"),
                "advantage": rollout_data.get("advantage"),
                "success": rollout_data.get("success"),
                "token_logprobs": list(rollout_data.get("token_logprobs", [])),
                "prompt_length": rollout_data.get("prompt_length"),
                "completion_length": rollout_data.get("completion_length"),
                "satisfied_clauses": rollout_data.get("satisfied_clauses"),
                "assignment": list(rollout_data.get("assignment", [])),
            },
        },
        "timestamp": row.get("timestamp"),
        "challenge": row.get("challenge"),
        "hotkey": row.get("hotkey"),
        "signature": row.get("signature"),
    }


# --------------------------------------------------------------------------- #
#                        Public API Functions                                 #
# --------------------------------------------------------------------------- #


def serialize_window_to_parquet(window_data: dict[str, Any]) -> bytes:
    """Serialize window data to Parquet format.

    Converts the window dictionary (with inferences list) to a Parquet file
    in memory and returns the bytes.

    Args:
        window_data: Window dictionary with 'inferences' list and metadata

    Returns:
        Parquet file contents as bytes
    """
    inferences = window_data.get("inferences", [])

    # Convert inferences to Parquet-compatible rows
    rows = [_convert_inference_to_row(inf) for inf in inferences]

    # Create Arrow table from rows
    if rows:
        table = pa.Table.from_pylist(rows, schema=INFERENCE_SCHEMA)
    else:
        # Empty table with schema
        table = pa.Table.from_pylist([], schema=INFERENCE_SCHEMA)

    # Add window-level metadata to the schema
    metadata = {
        b"wallet": str(window_data.get("wallet", "")).encode(),
        b"window_start": str(window_data.get("window_start", 0)).encode(),
        b"window_length": str(window_data.get("window_length", 0)).encode(),
        b"inference_count": str(len(inferences)).encode(),
        b"timestamp": str(window_data.get("timestamp", 0.0)).encode(),
    }
    table = table.replace_schema_metadata(metadata)

    # Serialize to Parquet bytes with snappy compression
    buffer = io.BytesIO()
    pq.write_table(
        table,
        buffer,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    return buffer.getvalue()


def deserialize_parquet_to_window(data: bytes) -> dict[str, Any]:
    """Deserialize Parquet bytes back to window data dictionary.

    Reads the Parquet file from bytes and reconstructs the original
    window dictionary format with inferences list.

    Args:
        data: Parquet file contents as bytes

    Returns:
        Window dictionary with 'inferences' list and metadata

    Raises:
        ParquetError: If data is empty, too small, or corrupt
    """
    if not data or len(data) < _MIN_PARQUET_SIZE:
        raise ParquetError(f"Invalid Parquet data: {len(data) if data else 0} bytes")

    try:
        buffer = io.BytesIO(data)
        table = pq.read_table(buffer)
    except pa.ArrowException as e:
        raise ParquetError(f"Corrupt Parquet file: {e}") from e

    # Extract window-level metadata
    metadata = table.schema.metadata or {}
    wallet = metadata.get(b"wallet", b"").decode()
    window_start = int(metadata.get(b"window_start", b"0").decode())
    window_length = int(metadata.get(b"window_length", b"0").decode())
    inference_count = int(metadata.get(b"inference_count", b"0").decode())
    timestamp = float(metadata.get(b"timestamp", b"0.0").decode())

    # Convert table to list of dicts
    rows = table.to_pylist()

    # Convert rows back to inference format
    inferences = [_convert_row_to_inference(row) for row in rows]

    return {
        "wallet": wallet,
        "window_start": window_start,
        "window_length": window_length,
        "inference_count": inference_count,
        "inferences": inferences,
        "timestamp": timestamp,
    }
