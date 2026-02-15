"""Tests for null-group Parquet inflation vulnerability and its fix.

Part 1 (TestNullGroupVulnerabilityProof): Documents the exploit mechanics —
  how null rollout_group rows inflate scores via the counting/sampling mismatch.
  These tests use raw Arrow tables to demonstrate the bug without deserialization.

Part 2 (TestNullGroupRejection): Verifies the fix —
  deserialize_parquet_to_window rejects files with null rollout_group at the
  Arrow level (O(1) check) before any expensive Python row conversion.
  _validate_file_structure provides defense-in-depth.
"""

from __future__ import annotations

import io
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from grail.infrastructure.parquet_io import (
    INFERENCE_SCHEMA,
    ParquetError,
    deserialize_parquet_to_window,
)
from grail.validation.miner_validator import (
    MAX_SAMPLES_PER_MINER_THRESHOLD,
    MinerValidator,
)


def _make_real_row(
    group_id: int,
    rollout_index: int,
    nonce: int,
    window_start: int = 100,
    block_hash: str = "abc123",
    hotkey: str = "5FakeHotkey",
) -> dict:
    """Create a realistic-looking inference row with valid structure."""
    return {
        "window_start": window_start,
        "block": window_start + 1,
        "nonce": nonce,
        "block_hash": block_hash,
        "randomness": "rand",
        "use_drand": False,
        "rollout_group": group_id,
        "rollout_index": rollout_index,
        "total_in_group": 16,
        "checkpoint_window": 50,
        "commit": {
            "tokens": [1, 2, 3, 4, 5] * 20,
            "commitments": "[]",
            "proof_version": "v1",
            "model": {"name": "test-model", "layer_index": 0},
            "signature": "deadbeef",
            "beacon": "{}",
            "rollout": {
                "trajectory": "[]",
                "total_reward": 1.0,
                "advantage": 0.0,
                "success": True,
                "token_logprobs": [-0.5] * 10,
                "prompt_length": 10,
                "completion_length": 90,
                "satisfied_clauses": 0,
                "assignment": [],
            },
        },
        "timestamp": 1000.0,
        "challenge": f"seed|{block_hash}|{nonce}",
        "hotkey": hotkey,
        "signature": "aabbccdd",
    }


def _make_dummy_null_row(nonce: int, window_start: int = 100) -> dict:
    """Create a dummy row with rollout_group=None (null in Parquet)."""
    return {
        "window_start": window_start,
        "block": window_start + 1,
        "nonce": nonce,
        "block_hash": "abc123",
        "randomness": "rand",
        "use_drand": False,
        "rollout_group": None,  # <-- THE EXPLOIT: null group
        "rollout_index": 0,
        "total_in_group": 0,
        "checkpoint_window": 50,
        "commit": {
            "tokens": [],
            "commitments": "[]",
            "proof_version": "",
            "model": {"name": "", "layer_index": 0},
            "signature": "",
            "beacon": "{}",
            "rollout": {
                "trajectory": "[]",
                "total_reward": 0.0,
                "advantage": 0.0,
                "success": False,
                "token_logprobs": [],
                "prompt_length": 0,
                "completion_length": 0,
                "satisfied_clauses": 0,
                "assignment": [],
            },
        },
        "timestamp": 1000.0,
        "challenge": "",
        "hotkey": "",
        "signature": "",
    }


def _build_exploit_parquet(
    num_real_groups: int = 2,
    rollouts_per_group: int = 16,
    num_dummy_rows: int = 50_000,
) -> bytes:
    """Build a Parquet file with real rollouts + null-group dummy rows."""
    rows = []
    nonce = 0

    for group_id in range(num_real_groups):
        for rollout_idx in range(rollouts_per_group):
            rows.append(_make_real_row(group_id, rollout_idx, nonce))
            nonce += 1

    for _ in range(num_dummy_rows):
        rows.append(_make_dummy_null_row(nonce))
        nonce += 1

    table = pa.Table.from_pylist(rows, schema=INFERENCE_SCHEMA)

    metadata = {
        b"wallet": b"5FakeHotkey",
        b"window_start": b"100",
        b"window_length": b"30",
        b"inference_count": str(len(rows)).encode(),
        b"timestamp": b"1000.0",
    }
    table = table.replace_schema_metadata(metadata)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


def _build_exploit_arrow_table(
    num_real_groups: int = 2,
    rollouts_per_group: int = 16,
    num_dummy_rows: int = 50_000,
) -> pa.Table:
    """Build an Arrow table directly (for vulnerability proof tests that
    need to inspect rows without going through deserialization)."""
    rows = []
    nonce = 0

    for group_id in range(num_real_groups):
        for rollout_idx in range(rollouts_per_group):
            rows.append(_make_real_row(group_id, rollout_idx, nonce))
            nonce += 1

    for _ in range(num_dummy_rows):
        rows.append(_make_dummy_null_row(nonce))
        nonce += 1

    return pa.Table.from_pylist(rows, schema=INFERENCE_SCHEMA)


def _table_to_inferences(table: pa.Table) -> list[dict]:
    """Convert Arrow table rows to inference-like dicts (minimal, for testing
    the counting/sampling logic only)."""
    result = []
    for row in table.to_pylist():
        result.append(
            {
                "rollout_group": row.get("rollout_group"),
                "commit": row.get("commit", {}),
            }
        )
    return result


class TestNullGroupVulnerabilityProof:
    """Documents the exploit mechanics using raw Arrow tables.

    These tests demonstrate the pre-fix bug in the counting/sampling logic.
    They use Arrow tables directly to bypass the deserialization-level fix.
    """

    def test_arrow_table_preserves_null_rollout_group(self):
        """PyArrow preserves null values in rollout_group column."""
        table = _build_exploit_arrow_table(
            num_real_groups=1, rollouts_per_group=2, num_dummy_rows=5
        )
        rg_col = table.column("rollout_group")

        # First 2 rows: real
        assert rg_col[0].as_py() == 0
        assert rg_col[1].as_py() == 0

        # Dummy rows: null
        for i in range(2, 7):
            assert rg_col[i].as_py() is None

    def test_total_vs_groups_mismatch(self):
        """Demonstrate the counting/sampling mismatch that enables the exploit."""
        table = _build_exploit_arrow_table(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=50_000
        )
        inferences = _table_to_inferences(table)

        # total_inferences counts ALL rows (the bug)
        total_inferences = len(inferences)
        assert total_inferences == 50_032

        # groups_map only includes non-null rows
        groups_map = defaultdict(list)
        for idx, inf in enumerate(inferences):
            raw_gid = inf.get("rollout_group")
            if raw_gid is not None:
                groups_map[str(raw_gid)].append(idx)

        total_in_groups = sum(len(v) for v in groups_map.values())
        assert total_in_groups == 32  # Only real rows
        assert len(groups_map) == 2  # Only real groups

        # Sampling mode forced (total > 20)
        assert total_inferences > MAX_SAMPLES_PER_MINER_THRESHOLD

        # With SAMPLE_RATE=0.10 and 2 groups:
        # groups_to_check = max(1, int(2 * 0.10)) = max(1, 0) = 1
        # Only 16 rollouts checked, all real -> 100% pass rate
        # Extrapolation: estimated_valid = 50,032 * 1.0 = 50,032
        # Inflation factor: 50,032 / 32 = 1,564x

    def test_extrapolation_inflates_scores(self):
        """The extrapolation formula inflates scores using total (inflated) count."""
        total = 2 * 16 + 50_000  # 50,032
        checked = 16
        valid = 16

        sample_pass_rate = valid / checked  # 1.0
        estimated_valid = int(total * sample_pass_rate)  # 50,032

        assert estimated_valid == total
        assert estimated_valid / 32 > 1500  # 1,564x inflation

    def test_null_group_guard_structurally_unreachable(self):
        """The null-group guard in _validate_rollouts is unreachable for dummy rows
        because sampling draws from groups_map which already excluded them."""
        table = _build_exploit_arrow_table(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=100
        )
        inferences = _table_to_inferences(table)
        total_inferences = len(inferences)

        validator = MinerValidator.__new__(MinerValidator)
        validator._hard_check_keys = []
        validator._soft_check_key = None

        class MockWallet:
            class hotkey:
                ss58_address = "5ValidatorKey"

        indices_to_check, _, _ = validator._determine_rollouts_to_check(
            inferences, "5FakeHotkey", "rand", MockWallet(), total_inferences
        )

        null_indices = {i for i, inf in enumerate(inferences) if inf.get("rollout_group") is None}
        assert len(null_indices) == 100
        assert len(set(indices_to_check) & null_indices) == 0

    def test_digest_skips_empty_tokens(self):
        """compute_completion_digest returns None for empty tokens."""
        from grail.shared.digest import compute_completion_digest

        assert compute_completion_digest({"tokens": []}, {"prompt_length": 0}) is None
        assert compute_completion_digest({"tokens": [1, 2, 3]}, {"prompt_length": 1}) is not None


class TestNullGroupRejection:
    """Verifies the fix: exploit files are rejected at the Arrow level
    before any expensive Python deserialization occurs."""

    def test_deserialize_rejects_null_rollout_groups(self):
        """deserialize_parquet_to_window raises ParquetError for null groups."""
        parquet_bytes = _build_exploit_parquet(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=50_000
        )
        with pytest.raises(ParquetError, match="null rollout_group"):
            deserialize_parquet_to_window(parquet_bytes)

    def test_deserialize_rejects_single_null_row(self):
        """Even a single null-group row causes rejection."""
        parquet_bytes = _build_exploit_parquet(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=1
        )
        with pytest.raises(ParquetError, match="null rollout_group"):
            deserialize_parquet_to_window(parquet_bytes)

    def test_deserialize_accepts_clean_file(self):
        """A file with no null rollout_groups deserializes normally."""
        parquet_bytes = _build_exploit_parquet(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=0
        )
        window_data = deserialize_parquet_to_window(parquet_bytes)
        assert len(window_data["inferences"]) == 32

    def test_rejection_is_fast(self):
        """Rejection happens at Arrow level (O(1)), not after row conversion."""
        import time

        # Build a large exploit file
        parquet_bytes = _build_exploit_parquet(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=50_000
        )

        start = time.monotonic()
        with pytest.raises(ParquetError):
            deserialize_parquet_to_window(parquet_bytes)
        elapsed = time.monotonic() - start

        # Should reject in well under 1 second — the O(1) null_count check
        # fires before the expensive to_pylist() + _convert_row_to_inference loop
        assert elapsed < 1.0, f"Rejection took {elapsed:.2f}s — too slow"

    def test_validate_file_structure_defense_in_depth(self):
        """_validate_file_structure also catches null groups as defense-in-depth.

        This tests the secondary check in miner_validator.py that would catch
        null groups if they somehow bypassed the deserialization-level check.
        """
        # Build inferences list manually (as if deserialization somehow passed)
        inferences = [
            {
                "rollout_group": 0,
                "window_start": 100,
                "block_hash": "abc123",
                "nonce": 0,
                "commit": {},
            },
            {
                "rollout_group": None,
                "window_start": 100,
                "block_hash": "abc123",
                "nonce": 1,
                "commit": {},
            },  # null group
        ]
        file_data = {
            "wallet": "5FakeHotkey",
            "window_start": 100,
            "inferences": inferences,
        }

        validator = MinerValidator.__new__(MinerValidator)
        validator._hard_check_keys = []
        validator._soft_check_key = None

        result = validator._validate_file_structure(
            file_data=file_data,
            miner_hotkey="5FakeHotkey",
            window=100,
            window_hash="abc123",
        )
        assert result["valid"] is False
        assert result["reason"] == "null_rollout_group"

    def test_clean_file_passes_both_checks(self):
        """Clean file passes both deserialization and _validate_file_structure."""
        parquet_bytes = _build_exploit_parquet(
            num_real_groups=2, rollouts_per_group=16, num_dummy_rows=0
        )
        window_data = deserialize_parquet_to_window(parquet_bytes)

        validator = MinerValidator.__new__(MinerValidator)
        validator._hard_check_keys = []
        validator._soft_check_key = None

        result = validator._validate_file_structure(
            file_data=window_data,
            miner_hotkey="5FakeHotkey",
            window=100,
            window_hash="abc123",
        )
        assert result["valid"] is True
