"""Unit tests for GSM8KTaskSource with train/val split."""

from __future__ import annotations

import pytest

from grail.environments.providers import GSM8KTaskSource


@pytest.mark.integration
class TestGSM8KTaskSourceSplits:
    """Test train/val/test split functionality."""

    def test_split_sizes(self) -> None:
        """Verify correct split sizes: train=7223, val=250, test=1319."""
        train = GSM8KTaskSource(split="train")
        val = GSM8KTaskSource(split="val")
        test = GSM8KTaskSource(split="test")

        # GSM8K has 7473 train, 1319 test
        # We take 250 for val from train
        assert train.size() == 7473 - 250  # 7223
        assert val.size() == 250
        assert test.size() == 1319

    def test_train_val_complete(self) -> None:
        """Verify train + val = original HF train (7473)."""
        train = GSM8KTaskSource(split="train")
        val = GSM8KTaskSource(split="val")

        assert train.size() + val.size() == 7473

    def test_no_train_val_overlap(self) -> None:
        """Verify no overlap between train and val sets."""
        train = GSM8KTaskSource(split="train")
        val = GSM8KTaskSource(split="val")

        train._ensure_dataset()
        val._ensure_dataset()

        train_questions = {s["question"] for s in train._data}
        val_questions = {s["question"] for s in val._data}

        assert len(train_questions & val_questions) == 0

    def test_val_deterministic(self) -> None:
        """Verify val set is deterministic across instantiations."""
        # Clear cache to force reload
        GSM8KTaskSource._cache.clear()

        val1 = GSM8KTaskSource(split="val")
        val1._ensure_dataset()
        questions1 = [s["question"] for s in val1._data]

        # Clear and reload
        GSM8KTaskSource._cache.clear()

        val2 = GSM8KTaskSource(split="val")
        val2._ensure_dataset()
        questions2 = [s["question"] for s in val2._data]

        assert questions1 == questions2

    def test_invalid_split_raises(self) -> None:
        """Verify invalid split raises ValueError."""
        with pytest.raises(ValueError, match="split must be"):
            GSM8KTaskSource(split="invalid")


@pytest.mark.integration
class TestGSM8KTaskSourceSampling:
    """Test task sampling functionality."""

    def test_next_returns_valid_task(self) -> None:
        """Verify next() returns properly structured TaskSpec."""
        source = GSM8KTaskSource(split="train")
        task = source.next(seed=12345)

        assert task.id.startswith("train_")
        assert "question" in task.payload
        assert "answer" in task.payload
        assert task.payload["question"]  # Non-empty question
        assert task.payload["answer"]  # Non-empty answer
        assert task.metadata["split"] == "train"
        assert "index" in task.metadata

    def test_seed_deterministic(self) -> None:
        """Verify same seed returns same task."""
        source = GSM8KTaskSource(split="train")

        task1 = source.next(seed=42)
        task2 = source.next(seed=42)

        assert task1.id == task2.id
        assert task1.payload["question"] == task2.payload["question"]

    def test_different_seeds_different_tasks(self) -> None:
        """Verify different seeds return different tasks."""
        source = GSM8KTaskSource(split="train")

        # Use large spread seeds to ensure different index mapping
        task1 = source.next(seed=1_000_000_000)
        task2 = source.next(seed=3_000_000_000)

        assert task1.id != task2.id

    def test_iter_ids_count(self) -> None:
        """Verify iter_ids returns correct number of IDs."""
        source = GSM8KTaskSource(split="val")

        ids = source.iter_ids()

        assert len(ids) == 250
        assert all(id.startswith("val_") for id in ids)

    def test_val_task_retrieval(self) -> None:
        """Verify can retrieve task from val split."""
        source = GSM8KTaskSource(split="val")
        task = source.next(seed=999)

        assert task.id.startswith("val_")
        assert task.metadata["split"] == "val"

    def test_test_task_retrieval(self) -> None:
        """Verify can retrieve task from test split."""
        source = GSM8KTaskSource(split="test")
        task = source.next(seed=999)

        assert task.id.startswith("test_")
        assert task.metadata["split"] == "test"

    def test_cache_reuse(self) -> None:
        """Verify cache is reused across instances."""
        GSM8KTaskSource._cache.clear()

        # First instance loads data
        source1 = GSM8KTaskSource(split="train")
        source1._ensure_dataset()

        # Second instance should use cache
        source2 = GSM8KTaskSource(split="train")
        source2._ensure_dataset()

        # Both should reference same cached data
        assert source1._data is source2._data
