"""Task providers for environments (dataset-backed and procedural)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import TaskSource


@dataclass(frozen=True)
class TaskSpec:
    """Generic container for a task instance."""

    id: str
    payload: Any
    metadata: dict[str, Any]


class SATTaskSource(TaskSource):
    """Procedural SAT task provider wrapping the existing generator.

    Difficulty is derived deterministically from the seed to ensure
    miner and validator generate identical problems.
    """

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        if task_id is not None:
            seed_material = task_id
        elif seed is not None:
            seed_material = str(seed)
        else:
            # Fallback deterministic material if not provided (not random here)
            seed_material = "sat-default-seed"

        # Derive difficulty deterministically from seed
        # Use modulo to map seed to difficulty range [0.3, 0.7]
        seed_int = int(seed) if seed is not None else 12345
        difficulty = 0.3 + (seed_int % 100) / 250.0  # Maps to [0.3, 0.7]

        # Lazy import to avoid circular dependency with sat_env
        from .sat_env import generate_sat_problem

        problem = generate_sat_problem(seed_material, difficulty)
        return TaskSpec(
            id=str(seed_material),
            payload=problem,
            metadata={"difficulty": difficulty},
        )


# Fixed validation set size for GSM8K (sampled from train split)
_GSM8K_VAL_SIZE = 250
# Seed for deterministic sampling of validation set
_GSM8K_VAL_SEED = 42


class GSM8KTaskSource(TaskSource):
    """HF datasets-backed GSM8K provider with train/val split.

    Lazily loads the dataset on first call to avoid import overhead on
    module import.

    Split behavior:
    - "train": Training set (HF train split minus 250 val holdout)
    - "val": Validation set (250 samples from HF train split)
    - "test": Original test set
    """

    # Class-level cache for loaded data (shared across instances)
    _cache: dict[str, Any] = {}

    def __init__(self, *, split: str = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self._split = split
        self._data: list[dict[str, Any]] | None = None

    def _ensure_dataset(self) -> None:
        if self._data is not None:
            return

        # Check class-level cache first
        cache_key = f"gsm8k_{self._split}"
        if cache_key in GSM8KTaskSource._cache:
            self._data = GSM8KTaskSource._cache[cache_key]
            return

        from datasets import load_dataset

        if self._split == "test":
            # Load original test set directly
            ds = load_dataset("openai/gsm8k", "main", split="test")
            self._data = [
                {
                    "question": sample.get("question", ""),  # type: ignore[union-attr]
                    "answer": sample.get("answer", ""),  # type: ignore[union-attr]
                }
                for sample in ds
            ]
        else:
            # Load train split and create train/val split
            self._load_train_val_split()

        # Cache for reuse
        GSM8KTaskSource._cache[cache_key] = self._data

    def _load_train_val_split(self) -> None:
        """Load HF train split and create train/val split."""
        import random

        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        all_samples = [
            {
                "question": sample.get("question", ""),  # type: ignore[union-attr]
                "answer": sample.get("answer", ""),  # type: ignore[union-attr]
            }
            for sample in ds
        ]

        # Deterministic shuffle for val sampling
        rng = random.Random(_GSM8K_VAL_SEED)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)

        # First _GSM8K_VAL_SIZE indices go to val, rest to train
        val_indices = set(indices[:_GSM8K_VAL_SIZE])

        train_samples: list[dict[str, Any]] = []
        val_samples: list[dict[str, Any]] = []

        for i, sample in enumerate(all_samples):
            if i in val_indices:
                val_samples.append(sample)
            else:
                train_samples.append(sample)

        # Guarantee: val size is exactly _GSM8K_VAL_SIZE (250)
        assert len(val_samples) == _GSM8K_VAL_SIZE, (
            f"Expected {_GSM8K_VAL_SIZE} val samples, got {len(val_samples)}"
        )

        # Cache both splits
        GSM8KTaskSource._cache["gsm8k_train"] = train_samples
        GSM8KTaskSource._cache["gsm8k_val"] = val_samples

        self._data = train_samples if self._split == "train" else val_samples

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        self._ensure_dataset()
        assert self._data is not None

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except Exception:
                idx = 0
        elif seed is not None:
            # Uniform sampling using floating-point scaling to avoid modular bias
            # Convert seed to [0,1) then scale to dataset size
            seed_normalized = (int(seed) % (2**32)) / (2**32)
            idx = int(seed_normalized * len(self._data))
        else:
            idx = 0

        sample = self._data[int(idx) % len(self._data)]
        # GSM8K fields: {question, answer}
        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={"question": sample["question"], "answer": sample["answer"]},
            metadata={"split": self._split, "index": int(idx)},
        )

    # --- Evaluation helpers ---
    def size(self) -> int:
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self) -> list[str]:
        self._ensure_dataset()
        assert self._data is not None
        return [f"{self._split}_{i}" for i in range(len(self._data))]


def _extract_boxed_answer(solution: str) -> str:
    """Extract answer from \\boxed{...} in solution, handling nested braces."""
    import re

    match = re.search(r"\\boxed\{", solution)
    if not match:
        return ""

    start = match.end()
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1

    return solution[start : i - 1] if depth == 0 else ""


# Subsets in EleutherAI/hendrycks_math dataset
_MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

# Fixed validation set size (stratified across problem types)
_MATH_VAL_SIZE = 500
# Seed for deterministic stratified sampling of validation set
_MATH_VAL_SEED = 42


class MATHTaskSource(TaskSource):
    """HF datasets-backed Hendrycks MATH provider with stratified train/val split.

    Uses EleutherAI/hendrycks_math dataset (7 subsets, 7500 train + 5000 test).

    Split behavior:
    - "train": Training set (7000 samples = 7500 - 500 val holdout)
    - "val": Validation set (500 samples, stratified by problem type)
    - "test": Original test set (5000 samples)

    The validation set is deterministically sampled from the HF train split
    using stratified sampling to ensure each problem type is represented.

    Supports filtering by:
    - level: int (1-5) - difficulty level
    - subject: str - mathematical domain (Algebra, Geometry, etc.)
    """

    # Class-level cache for loaded data (shared across instances)
    _cache: dict[str, Any] = {}

    def __init__(self, *, split: str = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        self._split = split
        self._data: list[dict[str, Any]] | None = None
        self._filtered_indices: dict[str, list[int]] = {}

    def _ensure_dataset(self) -> None:
        if self._data is not None:
            return

        # Check class-level cache first
        cache_key = f"math_{self._split}"
        if cache_key in MATHTaskSource._cache:
            self._data = MATHTaskSource._cache[cache_key]
            return

        from datasets import load_dataset

        if self._split == "test":
            # Load original test set directly
            all_samples: list[dict[str, Any]] = []
            for subset in _MATH_SUBSETS:
                ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
                for sample in ds:
                    answer = _extract_boxed_answer(
                        sample.get("solution", "")  # type: ignore[union-attr]
                    )
                    all_samples.append(
                        {
                            "problem": sample.get("problem", ""),  # type: ignore[union-attr]
                            "solution": sample.get("solution", ""),  # type: ignore[union-attr]
                            "answer": answer,
                            "subject": sample.get("type", ""),  # type: ignore[union-attr]
                            "level": self._parse_level(
                                sample.get("level", "")  # type: ignore[union-attr]
                            ),
                        }
                    )
            self._data = all_samples
        else:
            # Load train split and create stratified train/val split
            self._load_train_val_split()

        # Cache for reuse
        MATHTaskSource._cache[cache_key] = self._data

    def _parse_level(self, level_str: str) -> int:
        """Parse level string like 'Level 3' to int 3."""
        import re

        match = re.search(r"\d+", str(level_str))
        return int(match.group()) if match else 0

    def _load_train_val_split(self) -> None:
        """Load HF train split and create stratified train/val split."""
        from datasets import load_dataset

        # Load all training samples grouped by subject
        samples_by_subject: dict[str, list[dict[str, Any]]] = {s: [] for s in _MATH_SUBSETS}

        for subset in _MATH_SUBSETS:
            ds = load_dataset("EleutherAI/hendrycks_math", subset, split="train")
            for sample in ds:
                answer = _extract_boxed_answer(
                    sample.get("solution", "")  # type: ignore[union-attr]
                )
                samples_by_subject[subset].append(
                    {
                        "problem": sample.get("problem", ""),  # type: ignore[union-attr]
                        "solution": sample.get("solution", ""),  # type: ignore[union-attr]
                        "answer": answer,
                        "subject": sample.get("type", ""),  # type: ignore[union-attr]
                        "level": self._parse_level(
                            sample.get("level", "")  # type: ignore[union-attr]
                        ),
                    }
                )

        # Stratified sampling: proportional to each subject's size
        import random

        rng = random.Random(_MATH_VAL_SEED)

        total_train = sum(len(samples) for samples in samples_by_subject.values())
        val_indices_by_subject: dict[str, set[int]] = {}

        # Calculate proportional val samples per subject
        remaining_val = _MATH_VAL_SIZE
        subjects = list(_MATH_SUBSETS)
        for i, subset in enumerate(subjects):
            subject_samples = samples_by_subject[subset]
            if i == len(subjects) - 1:
                # Last subject gets remaining to ensure exact total
                n_val = remaining_val
            else:
                # Proportional allocation
                proportion = len(subject_samples) / total_train
                n_val = int(round(_MATH_VAL_SIZE * proportion))
                n_val = min(n_val, len(subject_samples), remaining_val)

            # Randomly sample indices for validation
            all_indices = list(range(len(subject_samples)))
            rng.shuffle(all_indices)
            val_indices_by_subject[subset] = set(all_indices[:n_val])
            remaining_val -= n_val

        # Build train and val lists
        train_samples: list[dict[str, Any]] = []
        val_samples: list[dict[str, Any]] = []

        for subset in _MATH_SUBSETS:
            val_indices = val_indices_by_subject[subset]
            for i, sample in enumerate(samples_by_subject[subset]):
                if i in val_indices:
                    val_samples.append(sample)
                else:
                    train_samples.append(sample)

        # Guarantee: val size is exactly _MATH_VAL_SIZE (500)
        assert len(val_samples) == _MATH_VAL_SIZE, (
            f"Expected {_MATH_VAL_SIZE} val samples, got {len(val_samples)}"
        )

        # Cache both splits
        MATHTaskSource._cache["math_train"] = train_samples
        MATHTaskSource._cache["math_val"] = val_samples

        self._data = train_samples if self._split == "train" else val_samples

    def _get_filtered_indices(
        self, level: int | None = None, subject: str | None = None
    ) -> list[int]:
        """Get dataset indices matching filter criteria."""
        self._ensure_dataset()
        assert self._data is not None

        cache_key = f"level={level},subject={subject}"
        if cache_key in self._filtered_indices:
            return self._filtered_indices[cache_key]

        indices = []
        for i, sample in enumerate(self._data):
            if level is not None and sample["level"] != level:
                continue
            if subject is not None and sample["subject"] != subject:
                continue
            indices.append(i)

        self._filtered_indices[cache_key] = indices
        return indices

    def next(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
        level: int | None = None,
        subject: str | None = None,
    ) -> TaskSpec:
        """Sample task with optional filtering."""
        self._ensure_dataset()
        assert self._data is not None

        if level is not None or subject is not None:
            valid_indices = self._get_filtered_indices(level, subject)
            if not valid_indices:
                raise ValueError(f"No samples match filters: level={level}, subject={subject}")
        else:
            valid_indices = list(range(len(self._data)))

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except Exception:
                idx = valid_indices[0]
        elif seed is not None:
            seed_normalized = (int(seed) % (2**32)) / (2**32)
            idx = valid_indices[int(seed_normalized * len(valid_indices))]
        else:
            idx = valid_indices[0]

        sample = self._data[int(idx) % len(self._data)]

        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={
                "question": sample["problem"],
                "solution": sample["solution"],
                "answer": sample["answer"],
            },
            metadata={
                "split": self._split,
                "index": int(idx),
                "level": sample["level"],
                "subject": sample["subject"],
            },
        )

    def size(self, level: int | None = None, subject: str | None = None) -> int:
        """Get dataset size with optional filtering."""
        if level is not None or subject is not None:
            return len(self._get_filtered_indices(level, subject))
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self, level: int | None = None, subject: str | None = None) -> list[str]:
        """Get all task IDs with optional filtering."""
        if level is not None or subject is not None:
            indices = self._get_filtered_indices(level, subject)
            return [f"{self._split}_{i}" for i in indices]
        self._ensure_dataset()
        assert self._data is not None
        return [f"{self._split}_{i}" for i in range(len(self._data))]


# Python code generation dataset sources


# Fraction of original test set to add to train (rest goes to validation)
_MBPP_TEST_TO_TRAIN_FRACTION = 0.8
# Seed for deterministic shuffling of test set redistribution
_MBPP_REDISTRIBUTION_SEED = 42


class MBPPTaskSource(TaskSource):
    """HF datasets-backed MBPP (Mostly Basic Python Problems) provider.

    Uses mbpp dataset (full configuration) with redistributed splits.
    The original test set (500 examples) is redistributed to train/validation
    to maximize training data while reserving a validation set for evaluation.

    Dataset structure:
        - task_id: Unique identifier
        - text: Natural language problem description
        - code: Reference solution
        - test_list: List of assertion strings
        - test_setup_code: Setup code for tests (usually empty)
        - test_imports: Required imports for tests

    Split behavior (after redistribution):
        - "train": Original train (374) + 80% of original test (400) = 774 examples
        - "validation": Original validation (90) + 20% of original test (100) = 190 examples
        - "test": NOT AVAILABLE (raises AssertionError)

    Note: Dataset uses 'validation' not 'val' for the validation split.
    """

    # Class-level cache for loaded data
    _cache: dict[str, Any] = {}

    def __init__(self, *, split: str = "train") -> None:
        """Initialize MBPP task source.

        Args:
            split: Dataset split to use ('train' or 'validation')

        Raises:
            AssertionError: If split is 'test' (redistributed to train/validation)
            ValueError: If split is not recognized
        """
        if split == "test":
            raise AssertionError(
                "MBPP 'test' split does not exist. "
                "The original test set (500 examples) has been redistributed to "
                "train (80%) and validation (20%). Use 'train' or 'validation' instead."
            )
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got '{split}'")
        self._split = split
        self._data: list[dict[str, Any]] | None = None

    def _parse_hf_samples(self, ds: Any) -> list[dict[str, Any]]:
        """Parse HuggingFace dataset samples into standard format.

        Args:
            ds: HuggingFace dataset split

        Returns:
            List of parsed sample dictionaries
        """
        samples = []
        for sample in ds:
            # Extract fields from full configuration format
            task_id = sample.get("task_id", 0)
            text = sample.get("text", "")  # Full config uses 'text' field
            code = sample.get("code", "")
            test_list = sample.get("test_list", [])
            test_setup_code = sample.get("test_setup_code", "")
            # Note: full config doesn't have test_imports field
            test_imports = sample.get("test_imports", [])

            samples.append(
                {
                    "task_id": int(task_id) if task_id is not None else 0,
                    "text": str(text),
                    "code": str(code),
                    "test_list": [str(t) for t in test_list],
                    "test_setup_code": str(test_setup_code),
                    "test_imports": [str(imp) for imp in test_imports],
                }
            )
        return samples

    def _ensure_dataset(self) -> None:
        """Lazy load dataset from HuggingFace with test set redistribution."""
        if self._data is not None:
            return

        # Check class-level cache first
        cache_key = f"mbpp_{self._split}"
        if cache_key in MBPPTaskSource._cache:
            self._data = MBPPTaskSource._cache[cache_key]
            return

        # Check if redistribution has already been computed
        if "mbpp_train" in MBPPTaskSource._cache and "mbpp_validation" in MBPPTaskSource._cache:
            self._data = MBPPTaskSource._cache[cache_key]
            return

        # Need to compute redistribution - load all splits
        self._compute_redistributed_splits()
        self._data = MBPPTaskSource._cache[cache_key]

    def _compute_redistributed_splits(self) -> None:
        """Compute train/validation splits with test set redistribution.

        Redistribution strategy:
        1. Load original train, validation, and test splits from HuggingFace
        2. Shuffle test set deterministically
        3. Split test into 80% for train, 20% for validation
        4. Combine: new_train = train + test_train_portion
                    new_validation = validation + test_val_portion
        """
        import random

        from datasets import load_dataset

        # Load all three original splits
        ds_train = load_dataset("mbpp", "full", split="train")
        ds_validation = load_dataset("mbpp", "full", split="validation")
        ds_test = load_dataset("mbpp", "full", split="test")

        # Parse into standard format
        train_samples = self._parse_hf_samples(ds_train)
        validation_samples = self._parse_hf_samples(ds_validation)
        test_samples = self._parse_hf_samples(ds_test)

        # Deterministically shuffle test samples for redistribution
        rng = random.Random(_MBPP_REDISTRIBUTION_SEED)
        test_indices = list(range(len(test_samples)))
        rng.shuffle(test_indices)

        # Split test set: 80% to train, 20% to validation
        n_test_to_train = int(len(test_samples) * _MBPP_TEST_TO_TRAIN_FRACTION)
        test_train_indices = test_indices[:n_test_to_train]
        test_val_indices = test_indices[n_test_to_train:]

        # Combine splits
        new_train = train_samples + [test_samples[i] for i in test_train_indices]
        new_validation = validation_samples + [test_samples[i] for i in test_val_indices]

        # Cache the redistributed splits
        MBPPTaskSource._cache["mbpp_train"] = new_train
        MBPPTaskSource._cache["mbpp_validation"] = new_validation

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        """Sample task from dataset.

        Args:
            seed: Random seed for deterministic sampling
            task_id: Specific task ID to load (format: "{split}_{idx}")

        Returns:
            TaskSpec with problem description, tests, and metadata
        """
        self._ensure_dataset()
        assert self._data is not None

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except (ValueError, IndexError):
                idx = 0
        elif seed is not None:
            # Deterministic sampling - use modulo directly for uniform distribution
            idx = int(seed) % len(self._data)
        else:
            idx = 0

        sample = self._data[int(idx) % len(self._data)]

        return TaskSpec(
            id=f"{self._split}_{idx}",
            payload={
                "question": sample["text"],
                "test_list": sample["test_list"],
                "test_setup_code": sample["test_setup_code"],
                "test_imports": sample["test_imports"],
                "reference_solution": sample["code"],
            },
            metadata={
                "split": self._split,
                "index": int(idx),
                "task_id": sample["task_id"],
                "num_tests": len(sample["test_list"]),
            },
        )

    def size(self) -> int:
        """Get total number of examples in split."""
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self) -> list[str]:
        """Get all task IDs in split."""
        self._ensure_dataset()
        assert self._data is not None
        return [f"{self._split}_{i}" for i in range(len(self._data))]


class HumanEvalTaskSource(TaskSource):
    """HF datasets-backed HumanEval provider.

    Uses openai/openai_humaneval dataset (164 problems, test split only).

    Dataset structure:
        - task_id: Unique identifier (e.g., "HumanEval/0")
        - prompt: Function signature + docstring
        - canonical_solution: Reference implementation
        - test: Test function with assertions
        - entry_point: Function name to test

    Note: HumanEval only has a test split. For RL training, use MBPP instead.
    This source is primarily for evaluation/benchmarking.
    """

    # Class-level cache for loaded data
    _cache: list[dict[str, Any]] | None = None

    def __init__(self) -> None:
        """Initialize HumanEval task source.

        Note: HumanEval only has test split, no train/val.
        """
        self._data: list[dict[str, Any]] | None = None

    def _ensure_dataset(self) -> None:
        """Lazy load dataset from HuggingFace."""
        if self._data is not None:
            return

        # Check class-level cache first
        if HumanEvalTaskSource._cache is not None:
            self._data = HumanEvalTaskSource._cache
            return

        from datasets import load_dataset

        # HumanEval only has 'test' split
        ds = load_dataset("openai_humaneval", split="test")

        samples = []
        for sample in ds:
            samples.append(
                {
                    "task_id": str(sample.get("task_id", "")),  # type: ignore[union-attr]
                    "prompt": str(sample.get("prompt", "")),  # type: ignore[union-attr]
                    "canonical_solution": str(
                        sample.get("canonical_solution", "")  # type: ignore[union-attr]
                    ),
                    "test": str(sample.get("test", "")),  # type: ignore[union-attr]
                    "entry_point": str(sample.get("entry_point", "")),  # type: ignore[union-attr]
                }
            )

        self._data = samples
        HumanEvalTaskSource._cache = self._data

    def next(self, *, seed: int | None = None, task_id: str | None = None) -> TaskSpec:
        """Sample task from dataset.

        Args:
            seed: Random seed for deterministic sampling
            task_id: Specific task ID to load (format: "humaneval_{idx}")

        Returns:
            TaskSpec with problem prompt, test function, and metadata
        """
        self._ensure_dataset()
        assert self._data is not None

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except (ValueError, IndexError):
                idx = 0
        elif seed is not None:
            # Deterministic sampling - use modulo directly for uniform distribution
            idx = int(seed) % len(self._data)
        else:
            idx = 0

        sample = self._data[int(idx) % len(self._data)]

        return TaskSpec(
            id=f"humaneval_{idx}",
            payload={
                "question": sample["prompt"],
                "test": sample["test"],
                "entry_point": sample["entry_point"],
                "reference_solution": sample["canonical_solution"],
            },
            metadata={
                "split": "test",
                "index": int(idx),
                "original_task_id": sample["task_id"],
            },
        )

    def size(self) -> int:
        """Get total number of examples (always 164)."""
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self) -> list[str]:
        """Get all task IDs."""
        self._ensure_dataset()
        assert self._data is not None
        return [f"humaneval_{i}" for i in range(len(self._data))]
