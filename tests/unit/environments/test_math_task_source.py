"""Unit tests for MATHTaskSource with stratified train/val split."""

from __future__ import annotations

import pytest

from grail.environments.providers import MATHTaskSource, _extract_boxed_answer


class TestExtractBoxedAnswer:
    """Test boxed answer extraction from solutions."""

    def test_simple_numeric(self) -> None:
        solution = r"The answer is $\boxed{42}$."
        assert _extract_boxed_answer(solution) == "42"

    def test_fraction(self) -> None:
        solution = r"Therefore $\boxed{\frac{7}{20}}$."
        assert _extract_boxed_answer(solution) == r"\frac{7}{20}"

    def test_nested_braces(self) -> None:
        solution = r"$\boxed{\frac{2\sqrt{149}}{3}}$"
        assert _extract_boxed_answer(solution) == r"\frac{2\sqrt{149}}{3}"

    def test_matrix(self) -> None:
        solution = r"$\boxed{\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}}$"
        assert _extract_boxed_answer(solution) == r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"

    def test_no_boxed(self) -> None:
        solution = "The answer is 42."
        assert _extract_boxed_answer(solution) == ""

    def test_empty_boxed(self) -> None:
        solution = r"$\boxed{}$"
        assert _extract_boxed_answer(solution) == ""


@pytest.mark.integration
class TestMATHTaskSourceSplits:
    """Test train/val/test split functionality."""

    def test_split_sizes(self) -> None:
        """Verify correct split sizes: train=7000, val=500, test=5000."""
        train = MATHTaskSource(split="train")
        val = MATHTaskSource(split="val")
        test = MATHTaskSource(split="test")

        assert train.size() == 7000
        assert val.size() == 500
        assert test.size() == 5000

    def test_train_val_complete(self) -> None:
        """Verify train + val = original HF train (7500)."""
        train = MATHTaskSource(split="train")
        val = MATHTaskSource(split="val")

        assert train.size() + val.size() == 7500

    def test_no_train_val_overlap(self) -> None:
        """Verify no overlap between train and val sets."""
        train = MATHTaskSource(split="train")
        val = MATHTaskSource(split="val")

        train._ensure_dataset()
        val._ensure_dataset()

        train_problems = {s["problem"] for s in train._data}
        val_problems = {s["problem"] for s in val._data}

        assert len(train_problems & val_problems) == 0

    def test_val_stratified(self) -> None:
        """Verify val set has samples from all 7 subjects."""
        val = MATHTaskSource(split="val")
        val._ensure_dataset()

        subjects = {s["subject"] for s in val._data}
        expected_subjects = {
            "Algebra",
            "Counting & Probability",
            "Geometry",
            "Intermediate Algebra",
            "Number Theory",
            "Prealgebra",
            "Precalculus",
        }

        assert subjects == expected_subjects

    def test_val_deterministic(self) -> None:
        """Verify val set is deterministic across instantiations."""
        # Clear cache to force reload
        MATHTaskSource._cache.clear()

        val1 = MATHTaskSource(split="val")
        val1._ensure_dataset()
        problems1 = [s["problem"] for s in val1._data]

        # Clear and reload
        MATHTaskSource._cache.clear()

        val2 = MATHTaskSource(split="val")
        val2._ensure_dataset()
        problems2 = [s["problem"] for s in val2._data]

        assert problems1 == problems2

    def test_invalid_split_raises(self) -> None:
        """Verify invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            MATHTaskSource(split="invalid")


@pytest.mark.integration
class TestMATHTaskSourceSampling:
    """Test task sampling functionality."""

    def test_next_returns_valid_task(self) -> None:
        """Verify next() returns properly structured TaskSpec."""
        source = MATHTaskSource(split="train")
        task = source.next(seed=12345)

        assert task.id.startswith("train_")
        assert "question" in task.payload
        assert "solution" in task.payload
        assert "answer" in task.payload
        assert task.payload["answer"]  # Non-empty answer
        assert "subject" in task.metadata
        assert "level" in task.metadata

    def test_seed_deterministic(self) -> None:
        """Verify same seed returns same task."""
        source = MATHTaskSource(split="train")

        task1 = source.next(seed=42)
        task2 = source.next(seed=42)

        assert task1.id == task2.id
        assert task1.payload["question"] == task2.payload["question"]

    def test_different_seeds_different_tasks(self) -> None:
        """Verify different seeds return different tasks."""
        source = MATHTaskSource(split="train")

        # Use large spread seeds to ensure different index mapping
        task1 = source.next(seed=1_000_000_000)
        task2 = source.next(seed=3_000_000_000)

        assert task1.id != task2.id

    def test_iter_ids_count(self) -> None:
        """Verify iter_ids returns correct number of IDs."""
        source = MATHTaskSource(split="val")

        ids = source.iter_ids()

        assert len(ids) == 500
        assert all(id.startswith("val_") for id in ids)

    def test_filter_by_level(self) -> None:
        """Verify filtering by level works."""
        source = MATHTaskSource(split="train")

        level_5_size = source.size(level=5)
        all_size = source.size()

        assert 0 < level_5_size < all_size

    def test_filter_by_subject(self) -> None:
        """Verify filtering by subject works."""
        source = MATHTaskSource(split="train")

        algebra_size = source.size(subject="Algebra")
        all_size = source.size()

        assert 0 < algebra_size < all_size
