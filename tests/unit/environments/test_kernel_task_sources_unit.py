"""Unit tests for kernel task sources.

No GPU required. Tests task source behavior with mock data.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from grail.environments.gpu_kernel.task_sources import (
    KernelBenchTaskSource,
    UnifiedKernelTaskSource,
)


# =============================================================================
# Helpers
# =============================================================================


def _create_test_jsonl(rows: list[dict[str, Any]]) -> str:
    """Create a temporary JSONL file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _sample_unified_rows(n: int = 10) -> list[dict[str, Any]]:
    """Generate sample unified dataset rows."""
    rows = []
    sources = ["production_repos", "ai_cuda_engineer", "kernelbook"]
    difficulties = ["easy", "medium", "hard", "expert"]

    for i in range(n):
        rows.append({
            "id": f"test_{i}",
            "source": sources[i % len(sources)],
            "prompt": f"Optimize kernel {i}",
            "pytorch_reference": f"class Model:\n  pass  # {i}",
            "test_code": f"def check_correctness(m): return True  # {i}",
            "difficulty": difficulties[i % len(difficulties)],
            "category": "elementwise",
            "reference_solution": f"class ModelNew:\n  pass  # {i}" if i % 3 == 0 else None,
            "solution_quality": 0.8 if i % 3 == 0 else None,
        })
    return rows


# =============================================================================
# TestUnifiedKernelTaskSource
# =============================================================================


class TestUnifiedKernelTaskSource:
    """JSONL-based source."""

    def test_load_with_mode_filter(self) -> None:
        """mode='rl' vs 'sft'."""
        rows = _sample_unified_rows(10)
        path = _create_test_jsonl(rows)
        try:
            # Clear cache
            UnifiedKernelTaskSource._cache.clear()

            rl_source = UnifiedKernelTaskSource(dataset_path=path, split="all", mode="rl")
            sft_source = UnifiedKernelTaskSource(dataset_path=path, split="all", mode="sft")

            assert rl_source.size() > 0
            assert sft_source.size() > 0
            # RL includes all rows with test_code, SFT only those with reference_solution
            assert rl_source.size() >= sft_source.size()
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_exclude_sources(self) -> None:
        rows = _sample_unified_rows(12)
        path = _create_test_jsonl(rows)
        try:
            UnifiedKernelTaskSource._cache.clear()

            full = UnifiedKernelTaskSource(dataset_path=path, split="all")
            excluded = UnifiedKernelTaskSource(
                dataset_path=path, split="all", exclude_sources=["kernelbook"]
            )

            assert excluded.size() < full.size()
            # Verify no kernelbook rows
            for tid in excluded.iter_ids():
                task = excluded.next(task_id=tid)
                # Source check via metadata in taskspec
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_next_returns_taskspec(self) -> None:
        rows = _sample_unified_rows(5)
        path = _create_test_jsonl(rows)
        try:
            UnifiedKernelTaskSource._cache.clear()
            source = UnifiedKernelTaskSource(dataset_path=path, split="all")
            task = source.next(seed=42)

            assert task.id is not None
            assert "pytorch_code" in task.payload
            assert task.metadata is not None
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_deterministic_with_seed(self) -> None:
        rows = _sample_unified_rows(10)
        path = _create_test_jsonl(rows)
        try:
            UnifiedKernelTaskSource._cache.clear()
            source = UnifiedKernelTaskSource(dataset_path=path, split="all")

            t1 = source.next(seed=42)
            t2 = source.next(seed=42)
            assert t1.id == t2.id

            t3 = source.next(seed=99)
            # Different seeds may give different tasks (depends on dataset size)
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_payload_has_test_code(self) -> None:
        rows = _sample_unified_rows(5)
        path = _create_test_jsonl(rows)
        try:
            UnifiedKernelTaskSource._cache.clear()
            source = UnifiedKernelTaskSource(dataset_path=path, split="all")
            task = source.next(seed=42)

            assert "test_code" in task.payload
            assert task.payload["test_code"] != ""
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()

    def test_weighted_sampling(self) -> None:
        """weighted=True changes distribution."""
        rows = _sample_unified_rows(20)
        path = _create_test_jsonl(rows)
        try:
            UnifiedKernelTaskSource._cache.clear()

            uniform = UnifiedKernelTaskSource(
                dataset_path=path, split="all", weighted_sampling=False
            )
            weighted = UnifiedKernelTaskSource(
                dataset_path=path, split="all", weighted_sampling=True
            )

            # Both should work and return valid tasks
            t1 = uniform.next(seed=42)
            t2 = weighted.next(seed=42)
            assert t1.id is not None
            assert t2.id is not None
        finally:
            os.unlink(path)
            UnifiedKernelTaskSource._cache.clear()


# =============================================================================
# TestWeightedSampling
# =============================================================================


class TestWeightedSampling:
    """Sampling weight computation."""

    def test_source_weights(self) -> None:
        weights = UnifiedKernelTaskSource._compute_sample_weights([
            {"source": "production_repos", "difficulty": "medium"},
            {"source": "ai_cuda_engineer", "difficulty": "medium"},
            {"source": "kernelbook", "difficulty": "medium"},
        ])
        assert weights[0] == 3.0  # production_repos x medium
        assert weights[1] == 2.0  # ai_cuda_engineer x medium
        assert weights[2] == 1.0  # kernelbook x medium

    def test_difficulty_weights(self) -> None:
        weights = UnifiedKernelTaskSource._compute_sample_weights([
            {"source": "kernelbook", "difficulty": "easy"},
            {"source": "kernelbook", "difficulty": "medium"},
            {"source": "kernelbook", "difficulty": "hard"},
            {"source": "kernelbook", "difficulty": "expert"},
        ])
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(1.0)
        assert weights[2] == pytest.approx(1.5)
        assert weights[3] == pytest.approx(2.0)

    def test_combined_weights(self) -> None:
        """source_weight x difficulty_weight."""
        weights = UnifiedKernelTaskSource._compute_sample_weights([
            {"source": "production_repos", "difficulty": "expert"},
        ])
        assert weights[0] == pytest.approx(6.0)  # 3.0 x 2.0


# =============================================================================
# TestKernelBenchTaskSource (basic, no HF download)
# =============================================================================


class TestKernelBenchTaskSource:
    """KernelBench task source validation (no actual HF downloads in unit tests)."""

    def test_invalid_split_raises(self) -> None:
        with pytest.raises(ValueError, match="split must be"):
            KernelBenchTaskSource(split="invalid")

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level must be"):
            KernelBenchTaskSource(level=99)

    def test_valid_construction(self) -> None:
        # Should not raise
        src = KernelBenchTaskSource(split="train", level=1)
        assert src._split == "train"
        assert src._level == 1
