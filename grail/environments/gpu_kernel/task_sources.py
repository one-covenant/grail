"""Task sources for GPU kernel generation environments.

Provides dataset-backed task providers for Triton kernel generation:
- KernelBenchTaskSource: Loads problems from ScalingIntelligence/KernelBench (HuggingFace)
  250 tasks across 4 difficulty levels, each containing a PyTorch Model class
  that the model must rewrite using custom Triton kernels.
- UnifiedKernelTaskSource: Loads from the unified GPU kernel dataset (JSONL/Parquet)
  built by ``research/datasets/build.py``.  Supports filtering by source, difficulty,
  category, quality threshold, and training mode (SFT vs RL).
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from ..core import TaskSource
from ..providers import TaskSpec

logger = logging.getLogger(__name__)

# KernelBench problem levels
KERNELBENCH_LEVELS = (1, 2, 3, 4)

# Split configuration: deterministic validation holdout per level
_VAL_FRACTION = 0.2
_VAL_SEED = 42

# ---------------------------------------------------------------------------
# Kernel evaluation constants
# ---------------------------------------------------------------------------
# Used by _STANDARD_CHECK_CORRECTNESS (embedded as literals in the exec'd
# string) and by the subprocess_backend.py fallback path (imported directly).
# Keep both locations in sync when changing these values.

KERNEL_EVAL_SEED = 42
"""RNG seed for deterministic model weight init and input generation.

Matches KernelBench's default ``seed_num=42``.  Called via
``torch.manual_seed`` / ``torch.cuda.manual_seed`` before constructing
each model so that ``nn.Linear``, ``nn.Conv2d``, etc. produce identical
random weights in both ``Model`` and ``ModelNew``."""

KERNEL_EVAL_TOLERANCE = 1e-2
"""Absolute tolerance for output comparison (max element-wise diff).

Looser than KernelBench's fp32 default (1e-4) to absorb cross-GPU
numerical differences in the decentralized miner/validator setting.
Triton autotuning, CUDNN algorithm selection, and floating-point
accumulation order vary across GPU instances."""

KERNEL_EVAL_NUM_TRIALS = 3
"""Number of correctness trials with different random inputs.

KernelBench uses 5; we use 3 to reduce per-eval subprocess overhead
(~2s per trial) while still catching non-deterministic failures."""

# Standard check_correctness() function appended to KernelBench code as test_code.
# KernelBench `code` already contains Model, get_inputs(), get_init_inputs().
# This adds the check_correctness() function needed by the eval backend.
#
# IMPORTANT: literal values below must match KERNEL_EVAL_* constants above.
_STANDARD_CHECK_CORRECTNESS = '''
def check_correctness(model_new_cls):
    """Check if ModelNew produces correct outputs vs Model reference.

    Seeds RNG before each model construction so nn.Linear, nn.Conv2d, etc.
    produce identical random weights (matches KernelBench methodology).
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _SEED = 42       # KERNEL_EVAL_SEED
    _TOLERANCE = 1e-2 # KERNEL_EVAL_TOLERANCE
    _N_TRIALS = 3     # KERNEL_EVAL_NUM_TRIALS

    init_inputs = get_init_inputs() if get_init_inputs else []

    # Seed before constructing EACH model so parameterized layers
    # (nn.Linear, nn.Conv2d, nn.Parameter(torch.randn(...)), etc.)
    # produce identical random weights.
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    ref_model = Model(*init_inputs).to(device).eval()

    torch.manual_seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    new_model = model_new_cls(*init_inputs).to(device).eval()

    max_diff = 0.0

    for trial in range(_N_TRIALS):
        torch.manual_seed(_SEED + trial)
        inputs = get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        with torch.no_grad():
            ref_out = ref_model(*inputs)
            new_out = new_model(*inputs)

        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        if isinstance(new_out, tuple):
            new_out = new_out[0]

        if ref_out.shape != new_out.shape:
            return {
                "correct": False,
                "compiled": True,
                "error": f"shape_mismatch: {ref_out.shape} vs {new_out.shape}",
                "max_diff": None,
            }

        diff = torch.max(torch.abs(ref_out.float() - new_out.float())).item()
        max_diff = max(max_diff, diff)

    correct = max_diff <= _TOLERANCE
    return {
        "correct": correct,
        "compiled": True,
        "error": None if correct else f"max_diff={max_diff:.6f}",
        "max_diff": max_diff,
    }
'''


class KernelBenchTaskSource(TaskSource):
    """HuggingFace datasets-backed KernelBench provider.

    Loads GPU kernel optimization problems from ScalingIntelligence/KernelBench.
    Each problem contains a PyTorch Model class with forward(), get_inputs(),
    and get_init_inputs() that the model must rewrite using Triton kernels.

    Dataset structure (per sample):
        - code: str - Complete PyTorch module source code
        - level: int - Difficulty level (1=single ops, 2=fusions, 3=architectures, 4=HF models)
        - name: str - Problem name/identifier
        - problem_id: int - Unique problem ID within level
        - split: str - Original split name (level_1, level_2, etc.)

    Split behavior:
        - "train": 80% of problems per level (deterministic split)
        - "val": 20% of problems per level (deterministic split)
        - "all": All problems (no split, for evaluation)

    Level filtering:
        - level=1: Single operators (100 problems) - matmul, conv, relu, etc.
        - level=2: Kernel fusion patterns (100 problems) - Conv+Bias+ReLU, etc.
        - level=3: Complete architectures (50 problems) - MobileNet, VGG, etc.
        - level=4: HuggingFace models (20 problems)
        - level=None: All levels combined
    """

    # Class-level cache for loaded data (shared across instances)
    _cache: dict[str, Any] = {}

    def __init__(
        self,
        *,
        split: str = "train",
        level: int | None = None,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got '{split}'")
        if level is not None and level not in KERNELBENCH_LEVELS:
            raise ValueError(f"level must be one of {KERNELBENCH_LEVELS} or None, got {level}")

        self._split = split
        self._level = level
        self._data: list[dict[str, Any]] | None = None

    def _ensure_dataset(self) -> None:
        if self._data is not None:
            return

        cache_key = f"kernelbench_{self._split}_level{self._level}"
        if cache_key in KernelBenchTaskSource._cache:
            self._data = KernelBenchTaskSource._cache[cache_key]
            return

        # Ensure base data is loaded and split
        if "kernelbench_train" not in KernelBenchTaskSource._cache:
            self._load_and_split()

        # Select the right split
        if self._split == "all":
            all_train = KernelBenchTaskSource._cache.get("kernelbench_train", [])
            all_val = KernelBenchTaskSource._cache.get("kernelbench_val", [])
            base_data = all_train + all_val
        else:
            base_data = KernelBenchTaskSource._cache.get(f"kernelbench_{self._split}", [])

        # Apply level filter
        if self._level is not None:
            self._data = [s for s in base_data if s["level"] == self._level]
        else:
            self._data = list(base_data)

        KernelBenchTaskSource._cache[cache_key] = self._data

    def _load_and_split(self) -> None:
        """Load KernelBench from HuggingFace and create train/val split."""
        from datasets import load_dataset

        all_samples: list[dict[str, Any]] = []

        # Load all levels
        for level_num in KERNELBENCH_LEVELS:
            split_name = f"level_{level_num}"
            try:
                ds = load_dataset(
                    "ScalingIntelligence/KernelBench",
                    split=split_name,
                )
            except Exception:
                # Level might not exist (e.g., level_4 was added later)
                continue

            for sample in ds:
                all_samples.append(
                    {
                        "code": str(sample.get("code", "")),
                        "level": int(sample.get("level", level_num)),
                        "name": str(sample.get("name", "")),
                        "problem_id": int(sample.get("problem_id", 0)),
                        "original_split": str(sample.get("split", split_name)),
                    }
                )

        # Deterministic train/val split per level
        rng = random.Random(_VAL_SEED)
        train_samples: list[dict[str, Any]] = []
        val_samples: list[dict[str, Any]] = []

        # Group by level for stratified splitting
        by_level: dict[int, list[dict[str, Any]]] = {}
        for s in all_samples:
            by_level.setdefault(s["level"], []).append(s)

        for level_num in sorted(by_level.keys()):
            level_samples = by_level[level_num]
            indices = list(range(len(level_samples)))
            rng.shuffle(indices)

            n_val = max(1, int(len(level_samples) * _VAL_FRACTION))
            val_indices = set(indices[:n_val])

            for i, sample in enumerate(level_samples):
                if i in val_indices:
                    val_samples.append(sample)
                else:
                    train_samples.append(sample)

        KernelBenchTaskSource._cache["kernelbench_train"] = train_samples
        KernelBenchTaskSource._cache["kernelbench_val"] = val_samples

    def next(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
    ) -> TaskSpec:
        self._ensure_dataset()
        assert self._data is not None
        assert len(self._data) > 0, (
            f"No KernelBench problems found for split={self._split}, level={self._level}"
        )

        if task_id is not None:
            try:
                idx = int(task_id.split("_")[-1])
            except (ValueError, IndexError):
                idx = 0
        elif seed is not None:
            seed_normalized = (int(seed) % (2**32)) / (2**32)
            idx = int(seed_normalized * len(self._data))
        else:
            idx = 0

        sample = self._data[int(idx) % len(self._data)]

        # Synthesize test_code by appending check_correctness() to the KernelBench code.
        # KernelBench `code` already contains Model, get_inputs(), get_init_inputs().
        test_code = sample["code"] + "\n\n" + _STANDARD_CHECK_CORRECTNESS

        return TaskSpec(
            id=f"kernelbench_{self._split}_l{sample['level']}_{idx}",
            payload={
                "pytorch_code": sample["code"],
                "test_code": test_code,
                "problem_name": sample["name"],
            },
            metadata={
                "split": self._split,
                "index": int(idx),
                "level": sample["level"],
                "problem_id": sample["problem_id"],
                "problem_name": sample["name"],
            },
        )

    def size(self) -> int:
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self) -> list[str]:
        self._ensure_dataset()
        assert self._data is not None
        return [f"kernelbench_{self._split}_l{s['level']}_{i}" for i, s in enumerate(self._data)]


# ---------------------------------------------------------------------------
# Unified kernel dataset source
# ---------------------------------------------------------------------------

# Allowed difficulty / category values (must match research/datasets/schema.py)
_VALID_DIFFICULTIES = ("easy", "medium", "hard", "expert")
_VALID_CATEGORIES = (
    "elementwise",
    "reduction",
    "matmul",
    "attention",
    "normalization",
    "fused_op",
    "full_architecture",
    "memory_op",
    "scan",
    "sort",
)


class UnifiedKernelTaskSource(TaskSource):
    """Task source backed by the unified GPU kernel dataset.

    Loads rows from a JSONL file produced by ``research/datasets/build.py``
    and serves them as ``TaskSpec`` instances.  Supports rich filtering so
    you can control what kind of problems the model trains on.

    Each row in the dataset has (at minimum):
        - ``id``: unique identifier
        - ``source``: origin dataset name
        - ``prompt``: KernelBench-style optimisation prompt
        - ``pytorch_reference``: clean PyTorch ``Model`` code
        - ``test_code``: correctness-checking scaffold
        - ``reference_solution``: (optional) high-quality Triton impl for SFT

    Split behavior:
        - ``"train"``: 80% of rows (deterministic split by row index)
        - ``"val"``:   20% of rows (deterministic split by row index)
        - ``"all"``:   every row (for evaluation / analysis)

    Filtering:
        - ``source``: restrict to a specific origin (e.g. ``"kernelbook"``)
        - ``difficulty``: restrict to a difficulty level
        - ``category``: restrict to an operation category
        - ``mode``: ``"sft"`` keeps only rows with a reference solution that
          meets the quality threshold; ``"rl"`` keeps all rows with test_code;
          ``"all"`` (default) keeps everything.
        - ``sft_quality_threshold``: minimum ``solution_quality`` for SFT mode

    Example::

        source = UnifiedKernelTaskSource(
            dataset_path="data/unified_kernel_dataset/full_dataset.jsonl",
            mode="sft",
            difficulty="medium",
        )
        task = source.next(seed=42)
    """

    # Class-level cache for loaded data (shared across instances)
    _cache: dict[str, list[dict[str, Any]]] = {}

    def __init__(
        self,
        *,
        dataset_path: str,
        split: str = "train",
        source_filter: str | None = None,
        exclude_sources: list[str] | None = None,
        difficulty: str | None = None,
        category: str | None = None,
        mode: str = "all",
        sft_quality_threshold: float = 0.60,
        weighted_sampling: bool = False,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be 'train', 'val', or 'all', got '{split}'")
        if mode not in ("sft", "rl", "all"):
            raise ValueError(f"mode must be 'sft', 'rl', or 'all', got '{mode}'")
        if difficulty is not None and difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {_VALID_DIFFICULTIES}, got '{difficulty}'")
        if category is not None and category not in _VALID_CATEGORIES:
            raise ValueError(f"category must be one of {_VALID_CATEGORIES}, got '{category}'")

        self._dataset_path = dataset_path
        self._split = split
        self._source_filter = source_filter
        self._exclude_sources = exclude_sources
        self._difficulty = difficulty
        self._category = category
        self._mode = mode
        self._sft_quality_threshold = sft_quality_threshold
        self._weighted_sampling = weighted_sampling
        self._data: list[dict[str, Any]] | None = None
        self._sample_weights: list[float] | None = None

    def _ensure_dataset(self) -> None:
        """Lazily load and filter the dataset on first access."""
        if self._data is not None:
            return

        # Build a cache key incorporating all filters
        exclude_key = ",".join(sorted(self._exclude_sources)) if self._exclude_sources else ""
        cache_key = (
            f"unified_{self._dataset_path}_{self._split}_{self._source_filter}_"
            f"{exclude_key}_{self._difficulty}_{self._category}_{self._mode}_"
            f"{self._sft_quality_threshold}"
        )
        if cache_key in UnifiedKernelTaskSource._cache:
            self._data = UnifiedKernelTaskSource._cache[cache_key]
            return

        # Load base data from file
        base_data = self._load_base_data()

        # Deterministic train/val split (80/20 by index)
        if self._split != "all":
            rng = random.Random(_VAL_SEED)
            indices = list(range(len(base_data)))
            rng.shuffle(indices)
            n_val = max(1, int(len(base_data) * _VAL_FRACTION))
            val_indices = set(indices[:n_val])

            if self._split == "val":
                base_data = [base_data[i] for i in range(len(base_data)) if i in val_indices]
            else:  # train
                base_data = [base_data[i] for i in range(len(base_data)) if i not in val_indices]

        # Apply filters
        filtered = self._apply_filters(base_data)

        self._data = filtered

        # Compute sample weights if weighted sampling is enabled
        if self._weighted_sampling and self._data:
            self._sample_weights = self._compute_sample_weights(self._data)

        UnifiedKernelTaskSource._cache[cache_key] = self._data
        logger.info(
            "UnifiedKernelTaskSource: loaded %d rows (split=%s, mode=%s, "
            "source=%s, difficulty=%s, category=%s)",
            len(self._data),
            self._split,
            self._mode,
            self._source_filter,
            self._difficulty,
            self._category,
        )

    def _load_base_data(self) -> list[dict[str, Any]]:
        """Load raw rows from JSONL file, with file-level caching."""
        file_cache_key = f"unified_file_{self._dataset_path}"
        if file_cache_key in UnifiedKernelTaskSource._cache:
            return UnifiedKernelTaskSource._cache[file_cache_key]

        rows: list[dict[str, Any]] = []
        try:
            with open(self._dataset_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except FileNotFoundError:
            logger.error(
                "Unified dataset file not found: %s. "
                "Run 'python -m research.datasets.build' to generate it.",
                self._dataset_path,
            )
            return []
        except Exception:
            logger.exception(
                "Failed to load unified dataset from %s",
                self._dataset_path,
            )
            return []

        UnifiedKernelTaskSource._cache[file_cache_key] = rows
        return rows

    def _apply_filters(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply source/difficulty/category/mode/exclude filters."""
        filtered = rows

        # Source filter (include)
        if self._source_filter is not None:
            filtered = [r for r in filtered if r.get("source") == self._source_filter]

        # Source exclusion filter
        if self._exclude_sources:
            excluded = set(self._exclude_sources)
            filtered = [r for r in filtered if r.get("source") not in excluded]

        # Difficulty filter
        if self._difficulty is not None:
            filtered = [r for r in filtered if r.get("difficulty") == self._difficulty]

        # Category filter
        if self._category is not None:
            filtered = [r for r in filtered if r.get("category") == self._category]

        # Mode filter
        if self._mode == "sft":
            filtered = [
                r
                for r in filtered
                if r.get("reference_solution")
                and (
                    r.get("solution_quality") is None
                    or r.get("solution_quality", 0.0) >= self._sft_quality_threshold
                )
            ]
        elif self._mode == "rl":
            filtered = [
                r for r in filtered if r.get("test_code") and str(r.get("test_code", "")).strip()
            ]

        return filtered

    def next(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
    ) -> TaskSpec:
        """Sample a task from the unified dataset."""
        self._ensure_dataset()
        assert self._data is not None
        assert len(self._data) > 0, (
            f"No tasks found in unified dataset (split={self._split}, "
            f"mode={self._mode}, source={self._source_filter}, "
            f"difficulty={self._difficulty}, category={self._category})"
        )

        if task_id is not None:
            # Try to find by exact ID match first
            for i, row in enumerate(self._data):
                if row.get("id") == task_id:
                    return self._row_to_taskspec(row, i)
            # Fallback: parse index from task_id
            try:
                idx = int(task_id.split("_")[-1])
            except (ValueError, IndexError):
                idx = 0
        elif seed is not None:
            if self._weighted_sampling and self._sample_weights:
                # Weighted sampling: use seed to deterministically pick based on weights
                rng = random.Random(seed)
                idx = rng.choices(range(len(self._data)), weights=self._sample_weights, k=1)[0]
            else:
                seed_normalized = (int(seed) % (2**32)) / (2**32)
                idx = int(seed_normalized * len(self._data))
        else:
            idx = 0

        row = self._data[int(idx) % len(self._data)]
        return self._row_to_taskspec(row, int(idx) % len(self._data))

    def _row_to_taskspec(self, row: dict[str, Any], idx: int) -> TaskSpec:
        """Convert a dataset row dict to a TaskSpec."""
        payload: dict[str, Any] = {
            "pytorch_code": row.get("pytorch_reference", ""),
            "test_code": row.get("test_code", ""),
            "problem_name": row.get("id", ""),
        }

        # Include reference solution in payload when available (SFT mode)
        if row.get("reference_solution"):
            payload["reference_solution"] = row["reference_solution"]
        if row.get("cuda_reference"):
            payload["cuda_reference"] = row["cuda_reference"]

        metadata: dict[str, Any] = {
            "split": self._split,
            "index": idx,
            "source": row.get("source", "unknown"),
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
            "solution_quality": row.get("solution_quality"),
            "solution_origin": row.get("solution_origin"),
            "performance_baseline_ms": row.get("performance_baseline_ms"),
        }

        # Merge original metadata if present
        original_meta = row.get("metadata", {})
        if isinstance(original_meta, dict):
            metadata.update(original_meta)

        return TaskSpec(
            id=row.get("id", f"unified_{idx}"),
            payload=payload,
            metadata=metadata,
        )

    @staticmethod
    def _compute_sample_weights(data: list[dict[str, Any]]) -> list[float]:
        """Compute per-row sampling weights based on source and difficulty.

        Weight = source_weight x difficulty_weight:
        - Source: production repos (3x), ai_cuda_engineer (2x), kernelbook (1x)
        - Difficulty: easy (0.5x), medium (1x), hard (1.5x), expert (2x)
        """
        source_weights: dict[str, float] = {
            "production_repos": 3.0,
            "ai_cuda_engineer": 2.0,
            "kernelbook": 1.0,
        }
        difficulty_weights: dict[str, float] = {
            "easy": 0.5,
            "medium": 1.0,
            "hard": 1.5,
            "expert": 2.0,
        }

        weights = []
        for row in data:
            sw = source_weights.get(row.get("source", ""), 1.0)
            dw = difficulty_weights.get(row.get("difficulty", ""), 1.0)
            weights.append(sw * dw)

        return weights

    def size(self) -> int:
        """Return the number of tasks matching current filters."""
        self._ensure_dataset()
        assert self._data is not None
        return len(self._data)

    def iter_ids(self) -> list[str]:
        """Return all task IDs matching current filters."""
        self._ensure_dataset()
        assert self._data is not None
        return [row.get("id", f"unified_{i}") for i, row in enumerate(self._data)]
