#!/usr/bin/env python3
"""TRL GRPO training script with factory pattern for multiple datasets.

Supports four datasets with exact parity to GRAIL environment implementations:
- GSM8K: Grade school math (7,473 train / 1,319 test)
- MATH: Hendrycks MATH benchmark (7,000 train / 500 val / 5,000 test)
- MBPP: Python code generation (374 train / 90 validation / 500 test)
- Triton Kernel: GPU kernel optimization (10K+ unified dataset)

Usage:
    python train_trl_grpo.py --dataset gsm8k
    python train_trl_grpo.py --dataset math
    python train_trl_grpo.py --dataset mbpp
    python train_trl_grpo.py --dataset triton_kernel --kernel-dataset-path /path/to/data.jsonl
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

# Force unbuffered output for better logging in nohup mode
# Use PYTHONUNBUFFERED-style approach that's safer than reopening file descriptors
try:
    # Check if stdout/stderr are valid before attempting to reconfigure
    if hasattr(sys.stdout, "fileno") and sys.stdout.fileno() >= 0:
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "fileno") and sys.stderr.fileno() >= 0:
        sys.stderr.reconfigure(line_buffering=True)
except (OSError, AttributeError, ValueError):
    # File descriptor invalid or reconfigure not available - continue without unbuffering
    pass

# Determine project root dynamically (research/trl/ -> project root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Load environment from .env for WandB
# Use override=False so CLI/deployment env vars take precedence over .env
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)

sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# GRAIL imports - reuse task sources and validation logic (after sys.path modification)
from grail.environments.execution import (  # noqa: E402
    CodeExecutionPool,
    check_code_executes,
    set_global_execution_pool,
)
from grail.environments.math_hendrycks_env import _math_answers_equal  # noqa: E402
from grail.environments.providers import (  # noqa: E402
    GSM8KTaskSource,
    MATHTaskSource,
    MBPPTaskSource,
)
from grail.shared.chat_templates import apply_chat_template, configure_tokenizer  # noqa: E402
from grail.shared.prompt_constants import (  # noqa: E402
    REASONING_END_TOKEN,
    REASONING_START_TOKEN,
    SOLUTION_END_TOKEN,
    SOLUTION_START_TOKEN,
    SYSTEM_PROMPT,
)
from grail.trainer.analysis import (  # noqa: E402
    AdamSignDescentMetrics,
    AnalysisConfig,
    GradientSparsityMetrics,
    ModelAnalysisManager,
)
from grail.trainer.metrics import KMetricsAggregator, TaskReplicateResult  # noqa: E402

# Local imports (research/trl specific) - shared callbacks
from callbacks import (  # noqa: E402
    DeltaCheckpointCallback,
    SparsityCallback,
    get_profiler,
)

# ════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,  # Override any existing configuration
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS (from .env GRAIL config - exactly matching grail/trainer/algorithms/grpo.py)
# ════════════════════════════════════════════════════════════════════════════
def _get_lr_from_env() -> float:
    """Get learning rate from environment, with default fallback.

    This is a factory function for dataclass field to ensure the env var
    is read at instance creation time, not at class definition time.
    """
    return float(os.environ.get("GRAIL_TRAINER_LR", "3e-6"))


@dataclass
class Config:
    # ────────────────────────────────────────────────────────────────────────
    # Model Configuration (from GRAIL_TRAIN_MODEL_ID)
    # ────────────────────────────────────────────────────────────────────────
    model_id: str = "Qwen/Qwen3-8B"
    # Training precision: "bfloat16" (default mixed-precision) or "float32" (full precision)
    dtype: str = "bfloat16"

    # ────────────────────────────────────────────────────────────────────────
    # Training Hyperparameters (from grail/shared/constants.py + env vars)
    # These match GRAIL's GRPOAlgorithm config exactly
    # ────────────────────────────────────────────────────────────────────────
    # Learning rate (GRAIL_TRAINER_LR, default: 3e-6)
    # Uses field(default_factory=...) to read env var at instantiation time
    lr: float = field(default_factory=_get_lr_from_env)
    # Epochs per training iteration (GRAIL_TRAINER_EPOCHS, constants.py default: 1)
    epochs: int = 1
    # Batch size per device (GRAIL_TRAINER_BATCH_SIZE, constants.py default: 16)
    # For 7B models on 80GB GPU with gradient checkpointing: batch_size=2 is safe
    # For 1.5B models: batch_size=4-8 is feasible
    batch_size: int = 2
    # ────────────────────────────────────────────────────────────────────────
    # Memory Optimization
    # ────────────────────────────────────────────────────────────────────────
    # Gradient checkpointing trades compute for memory by recomputing activations
    # during backward pass. Essential for 7B+ models on 80GB GPUs.
    # Memory savings: ~60-80% activation memory reduction
    # Cost: ~20-30% slower training
    gradient_checkpointing: bool = True
    # Gradient accumulation steps (GRAIL_TRAINER_GRAD_ACCUM_STEPS, constants.py default: 8)
    # Effective batch = batch_size × grad_accum_steps = 4 × 128 = 512
    grad_accum_steps: int = 256
    # Max sequence length (GRAIL_TRAINER_MAX_LENGTH, constants.py default: 2048)
    max_length: int = 3072
    # Gradient clipping threshold (GRAIL_TRAINER_GRAD_CLIP, constants.py default: 0.5)
    grad_clip: float = 1.0
    # Warmup steps for LR scheduler (GRAIL_TRAINER_WARMUP_STEPS, constants.py default: 10)
    warmup_steps: int = 20
    # Total training windows (GRAIL_TRAINER_TOTAL_WINDOWS) - controls iteration count
    # Each optimizer step = 32 groups × 16 rollouts = 512 samples
    # total_optimizer_steps calculated below based on total_windows
    total_steps: int = 100

    # ────────────────────────────────────────────────────────────────────────
    # GRPO Loss Configuration (from grail/trainer/algorithms/grpo.py)
    # ────────────────────────────────────────────────────────────────────────
    # KL divergence coefficient (GRAIL_TRAINER_KL_COEF, constants.py default: 0.02)
    kl_coef: float = 0.0
    # Entropy coefficient for exploration (GRAIL_TRAINER_ENTROPY_COEF, constants.py default: 0.001)
    # Note: TRL may not support entropy regularization directly
    entropy_coef: float = 0.0
    # Adam beta2 (second moment decay rate, default: 0.999)
    adam_beta2: float = 0.999
    # PPO clip epsilon lower bound (TRAINER_PPO_CLIP_EPS, constants.py default: 0.2)
    epsilon: float = 0.2
    # PPO clip epsilon upper bound - DAPO-style asymmetric clipping
    # (TRAINER_epsilon_UPPER, constants.py default: 0.28)
    epsilon_high: float = 0.28
    # Importance sampling ratio ceiling (GRAIL_TRAINER_IS_RATIO_MAX, constants.py default: 10.0)
    # Prevents training instability from extreme ratios
    is_ratio_max: float = 2.5
    # GRPO loss variant (GRAIL_GRPO_VARIANT, constants.py default: "dapo")
    # Options: 'grpo', 'bnpo', 'dapo', 'dr_grpo'
    grpo_variant: str = "dapo"
    # Importance sampling level (GRAIL_IMPORTANCE_SAMPLING_LEVEL, constants.py default: "sequence")
    # Options: 'sequence' (one ratio per sequence), 'token' (per-token ratios)
    # Note: TRL uses token-level IS by default when using vLLM
    importance_sampling_level: str = "token"

    # ────────────────────────────────────────────────────────────────────────
    # GRPO Data Configuration (from grail/shared/constants.py)
    # ────────────────────────────────────────────────────────────────────────
    # Groups per optimizer step = effective_batch / rollouts_per_problem = 512 / 16 = 32
    max_groups: int = 32
    # Max completion tokens (GRPO_MAX_COMPLETION_TOKENS, constants.py default: 1024)
    max_new_tokens: int = 2048
    # Rollouts per problem (ROLLOUTS_PER_PROBLEM, constants.py: 16)
    rollouts_per_problem: int = 16

    # ────────────────────────────────────────────────────────────────────────
    # Dataset Sampling
    # ────────────────────────────────────────────────────────────────────────
    num_train_samples: int | None = None  # None = use all training samples
    num_eval_samples: int | None = None  # None = use all test samples

    # ────────────────────────────────────────────────────────────────────────
    # Generation Parameters
    # ────────────────────────────────────────────────────────────────────────
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation Configuration
    # ────────────────────────────────────────────────────────────────────────
    eval_replicates: int = 5
    report_ks: tuple[int, ...] = (1, 5, 10)
    eval_batch_size: int = 128
    eval_num_workers: int = 4

    # ────────────────────────────────────────────────────────────────────────
    # Delta Checkpoint Configuration
    # ────────────────────────────────────────────────────────────────────────
    # Enable delta checkpointing (saves sparse weight updates after each step)
    delta_checkpoint_enabled: bool = True
    # Dtype for storing delta values ("bfloat16" or "float32")
    delta_checkpoint_dtype: str = "bfloat16"


cfg = Config()

# Tags, system prompt, and chat template are imported from grail.shared modules.
# See grail/shared/thinking.py for the single source of truth.


# ════════════════════════════════════════════════════════════════════════════
# DATASET ADAPTER (Abstract Base + Concrete Implementations)
# ════════════════════════════════════════════════════════════════════════════
class DatasetAdapter(abc.ABC):
    """Abstract base class for dataset adapters.

    Provides unified interface for:
    - Loading train/eval datasets
    - Parsing gold answers
    - Computing rewards
    - Determining success threshold
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Dataset name for logging."""
        ...

    @property
    @abc.abstractmethod
    def question_field(self) -> str:
        """Field name for question/problem text."""
        ...

    @property
    @abc.abstractmethod
    def answer_field(self) -> str:
        """Field name for gold answer."""
        ...

    @property
    @abc.abstractmethod
    def correctness_weight(self) -> float:
        """Weight for correctness component in reward."""
        ...

    @property
    @abc.abstractmethod
    def success_threshold(self) -> float:
        """Reward threshold for success (correctness weight)."""
        ...

    @abc.abstractmethod
    def load_train_data(self) -> list[dict[str, Any]]:
        """Load training data as list of dicts."""
        ...

    @abc.abstractmethod
    def load_eval_data(self) -> list[dict[str, Any]]:
        """Load evaluation data as list of dicts."""
        ...

    @abc.abstractmethod
    def parse_gold_answer(self, raw_answer: Any) -> Any:
        """Extract gold answer from dataset format.

        Notes:
            - For GSM8K/MATH this is typically a string.
            - For MBPP this may be structured data (e.g., dict with tests).
        """
        ...

    @abc.abstractmethod
    def validate_answer(self, predicted: str, gold: Any) -> bool:
        """Check if predicted answer matches gold."""
        ...

    @abc.abstractmethod
    def compute_reward(self, completion: str, gold_answer: Any) -> float:
        """Compute total reward for completion."""
        ...

    def get_gold_data(self, sample: dict[str, Any]) -> Any:
        """Get gold answer data from sample for evaluation.

        Override in subclass if answer format differs from answer_field.
        Default: return sample[answer_field]
        """
        return sample.get(self.answer_field, "")


# ────────────────────────────────────────────────────────────────────────────
# GSM8K Adapter
# ────────────────────────────────────────────────────────────────────────────
class GSM8KAdapter(DatasetAdapter):
    """GSM8K dataset adapter using GRAIL's GSM8KTaskSource."""

    # Regex patterns (from gsm8k_env.py)
    _HASH_PATTERN = re.compile(r"####\s*(?P<ans>.+)")
    _NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")
    _NUMERIC_ONLY_PATTERN = re.compile(r"^[-+]?[\d.,]+$")

    def __init__(self) -> None:
        self._train_source = GSM8KTaskSource(split="train")
        self._eval_source = GSM8KTaskSource(split="test")

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def question_field(self) -> str:
        return "question"

    @property
    def answer_field(self) -> str:
        return "answer"

    @property
    def correctness_weight(self) -> float:
        return 0.6  # GSM8K uses 0.6 for correctness

    @property
    def success_threshold(self) -> float:
        return 0.6  # Success if correctness achieved

    def load_train_data(self) -> list[dict[str, Any]]:
        """Load GSM8K training data via task source."""
        self._train_source._ensure_dataset()
        assert self._train_source._ds is not None
        data = []
        for i in range(len(self._train_source._ds)):
            sample = self._train_source._ds[i]
            data.append(
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                }
            )
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        """Load GSM8K test data via task source."""
        self._eval_source._ensure_dataset()
        assert self._eval_source._ds is not None
        data = []
        for i in range(len(self._eval_source._ds)):
            sample = self._eval_source._ds[i]
            data.append(
                {
                    "question": sample["question"],
                    "answer": sample["answer"],
                }
            )
        return data

    def parse_gold_answer(self, raw_answer: str) -> str:
        """Parse GSM8K gold answer from #### format."""
        match = None
        for m in self._HASH_PATTERN.finditer(raw_answer or ""):
            match = m
        if match is not None:
            return match.group("ans").strip()
        nums = list(self._NUMBER_PATTERN.finditer(raw_answer or ""))
        if nums:
            return nums[-1].group(0).replace(",", "").strip()
        return ""

    def validate_answer(self, predicted: str, gold: str) -> bool:
        """Validate GSM8K answer (numeric exact match)."""
        pred_norm = re.sub(r"[\s\.]+$", "", predicted.strip().lower())
        gold_norm = re.sub(r"[\s\.]+$", "", gold.strip().lower())
        return pred_norm == gold_norm

    def _parse_completion(self, text: str) -> dict[str, Any]:
        """Parse completion for thinking/answer tags."""
        flags = re.DOTALL | re.IGNORECASE
        has_thinking = bool(
            re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
        )
        answer_match = re.search(
            rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
        )

        answer_text = ""
        has_answer = bool(answer_match)
        is_numeric_only = False
        trailing = 0

        if answer_match:
            inside = answer_match.group(1).strip()
            num_match = self._NUMBER_PATTERN.search(inside)
            if num_match:
                answer_text = num_match.group(0).replace(",", "").strip()
                is_numeric_only = bool(self._NUMERIC_ONLY_PATTERN.match(inside.replace(" ", "")))
            trailing = len(text) - answer_match.end()

        return {
            "answer_text": answer_text,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "is_numeric_only": is_numeric_only,
            "trailing": trailing,
        }

    def compute_reward(self, completion: str, gold_answer: str) -> float:
        """Compute GSM8K reward (matching GSM8KEnv weights).

        Components:
        - Correctness (0.6): exact match
        - Strict format (0.15): numeric-only + no trailing
        - Thinking (0.1): has thinking block
        - Answer (0.1): has answer block
        - No trailing (0.05): penalty for trailing text
        """
        parsed = self._parse_completion(completion)
        gold_parsed = self.parse_gold_answer(gold_answer)

        # Correctness
        correctness = 0.6 if self.validate_answer(parsed["answer_text"], gold_parsed) else 0.0

        # Strict format
        strict_format = (
            0.15
            if (parsed["has_answer"] and parsed["is_numeric_only"] and parsed["trailing"] == 0)
            else 0.0
        )

        # Thinking format
        thinking = 0.1 if parsed["has_thinking"] else 0.0

        # Answer format
        answer = 0.1 if parsed["has_answer"] else 0.0

        # No trailing
        no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

        return correctness + strict_format + thinking + answer + no_trailing


# ────────────────────────────────────────────────────────────────────────────
# MATH (Hendrycks) Adapter
# ────────────────────────────────────────────────────────────────────────────
class MATHAdapter(DatasetAdapter):
    """MATH dataset adapter using GRAIL's MATHTaskSource.

    Uses exact same validation logic as MATHEnv:
    - Multi-strategy comparison (exact, symbolic via sympy, numeric)
    - LaTeX normalization
    - Stratified train/val split (500 val samples)
    """

    def __init__(self) -> None:
        self._train_source = MATHTaskSource(split="train")
        self._eval_source = MATHTaskSource(split="val")  # Use stratified val split

    @property
    def name(self) -> str:
        return "math"

    @property
    def question_field(self) -> str:
        return "question"  # Normalized to 'question' for consistency

    @property
    def answer_field(self) -> str:
        return "answer"

    @property
    def correctness_weight(self) -> float:
        return 0.7  # MATH uses 0.7 for correctness

    @property
    def success_threshold(self) -> float:
        return 0.7  # Success if correctness achieved

    def load_train_data(self) -> list[dict[str, Any]]:
        """Load MATH training data via task source (7000 samples)."""
        self._train_source._ensure_dataset()
        assert self._train_source._data is not None
        data = []
        for sample in self._train_source._data:
            data.append(
                {
                    "question": sample["problem"],  # Normalize field name
                    "answer": sample["answer"],  # Pre-extracted from \boxed{}
                    "solution": sample["solution"],
                    "level": sample["level"],
                    "subject": sample["subject"],
                }
            )
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        """Load MATH validation data via task source (500 samples, stratified)."""
        self._eval_source._ensure_dataset()
        assert self._eval_source._data is not None
        data = []
        for sample in self._eval_source._data:
            data.append(
                {
                    "question": sample["problem"],
                    "answer": sample["answer"],
                    "solution": sample["solution"],
                    "level": sample["level"],
                    "subject": sample["subject"],
                }
            )
        return data

    def parse_gold_answer(self, raw_answer: str) -> str:
        """For MATH, answer is already extracted from \\boxed{} by TaskSource."""
        return raw_answer

    def validate_answer(self, predicted: str, gold: str) -> bool:
        """Validate MATH answer using multi-strategy comparison.

        Uses GRAIL's _math_answers_equal which tries:
        1. Exact match (after LaTeX normalization)
        2. Symbolic equivalence (via sympy)
        3. Numeric comparison (floats)
        """
        return _math_answers_equal(predicted, gold)

    def _parse_completion(self, text: str) -> dict[str, Any]:
        """Parse completion for thinking/answer tags (MATH-specific)."""
        flags = re.DOTALL | re.IGNORECASE
        has_thinking = bool(
            re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
        )
        answer_match = re.search(
            rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
        )

        answer_text = ""
        has_answer = bool(answer_match)
        trailing = 0

        if answer_match:
            answer_text = answer_match.group(1).strip()
            trailing = len(text) - answer_match.end()

        return {
            "answer_text": answer_text,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "trailing": trailing,
        }

    def compute_reward(self, completion: str, gold_answer: str) -> float:
        """Compute MATH reward (matching MATHEnv weights).

        Components:
        - Correctness (0.7): Multi-strategy validation
        - Answer format (0.15): Has answer + minimal trailing
        - Thinking (0.1): Has thinking block
        - No trailing (0.05): Penalty for excessive trailing
        """
        parsed = self._parse_completion(completion)

        # Correctness (using multi-strategy validation)
        correctness = 0.7 if self.validate_answer(parsed["answer_text"], gold_answer) else 0.0

        # Answer format (has answer + trailing < 50)
        answer_format = 0.15 if (parsed["has_answer"] and parsed["trailing"] < 50) else 0.0

        # Thinking format
        thinking = 0.1 if parsed["has_thinking"] else 0.0

        # No trailing (stricter check)
        no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

        return correctness + answer_format + thinking + no_trailing


# ────────────────────────────────────────────────────────────────────────────
# MBPP (Python Code) Adapter
# ────────────────────────────────────────────────────────────────────────────
class MBPPAdapter(DatasetAdapter):
    """MBPP dataset adapter for Python code generation.

    Uses GRAIL's MBPPTaskSource and execution engine to validate code
    by running test cases in a sandboxed subprocess.
    """

    def __init__(self) -> None:
        self._train_source = MBPPTaskSource(split="train")
        self._eval_source = MBPPTaskSource(split="validation")

    @property
    def name(self) -> str:
        return "mbpp"

    @property
    def question_field(self) -> str:
        return "question"

    @property
    def answer_field(self) -> str:
        # Note: This returns "test_list" for API compatibility but actual
        # gold data is a dict containing test_list, test_setup_code, test_imports.
        # Use get_gold_data() for evaluation to get the full dict.
        return "test_list"

    def get_gold_data(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Get full test data dict for MBPP sample (used for evaluation).

        Unlike answer_field which returns just the field name, this returns
        the complete dict needed for code execution.
        """
        return {
            "test_list": sample.get("test_list", []),
            "test_setup_code": sample.get("test_setup_code", ""),
            "test_imports": sample.get("test_imports", []),
        }

    @property
    def correctness_weight(self) -> float:
        return 0.7  # MBPP uses 0.7 for correctness (test pass rate)

    @property
    def success_threshold(self) -> float:
        return 0.7  # Success if correctness achieved

    def load_train_data(self) -> list[dict[str, Any]]:
        """Load MBPP training data (374 samples)."""
        self._train_source._ensure_dataset()
        assert self._train_source._data is not None
        data = []
        for sample in self._train_source._data:
            data.append({
                "question": sample["text"],
                "test_list": sample["test_list"],
                "test_setup_code": sample["test_setup_code"],
                "test_imports": sample["test_imports"],
                "reference_solution": sample["code"],
            })
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        """Load MBPP validation data (90 samples)."""
        self._eval_source._ensure_dataset()
        assert self._eval_source._data is not None
        data = []
        for sample in self._eval_source._data:
            data.append({
                "question": sample["text"],
                "test_list": sample["test_list"],
                "test_setup_code": sample["test_setup_code"],
                "test_imports": sample["test_imports"],
                "reference_solution": sample["code"],
            })
        return data

    def parse_gold_answer(self, raw_answer: Any) -> Any:
        """For MBPP, return test_list as-is (used for validation)."""
        return raw_answer

    def validate_answer(self, predicted: str, test_data: dict[str, Any]) -> bool:
        """Validate MBPP answer by executing code against test cases.

        Args:
            predicted: Generated Python code
            test_data: Dict with test_list, test_setup_code, test_imports

        Returns:
            True if all tests pass
        """
        if not predicted or not isinstance(test_data, dict):
            return False

        test_list = test_data.get("test_list", [])
        if not test_list:
            return False

        # Prepare test cases with setup code and imports
        test_setup = test_data.get("test_setup_code", "")
        test_imports = test_data.get("test_imports", [])

        setup_code = "\n".join(test_imports) if test_imports else ""
        if test_setup:
            setup_code += f"\n{test_setup}"

        # Prepend setup to each test
        test_cases = []
        for test in test_list:
            if setup_code:
                test_cases.append(f"{setup_code}\n{test}")
            else:
                test_cases.append(test)

        # Execute tests
        result = check_code_executes(predicted, test_cases, timeout=5.0)
        return result["status"] == "all_passed"

    def _parse_completion(self, text: str) -> dict[str, Any]:
        """Parse completion for thinking/solution tags and code extraction."""
        flags = re.DOTALL | re.IGNORECASE
        has_thinking = bool(
            re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, flags)
        )
        solution_match = re.search(
            rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, flags
        )

        code = ""
        has_solution = bool(solution_match)
        syntax_valid = False
        trailing = 0

        if solution_match:
            code = solution_match.group(1).strip()
            trailing = len(text) - solution_match.end()

            # Check syntax validity
            if code:
                try:
                    compile(code, "<string>", "exec")
                    syntax_valid = True
                except SyntaxError:
                    syntax_valid = False

        return {
            "code": code,
            "has_thinking": has_thinking,
            "has_solution": has_solution,
            "syntax_valid": syntax_valid,
            "trailing": trailing,
        }

    def compute_reward(self, completion: str, test_data: Any) -> float:
        """Compute MBPP reward (matching PythonCodeEnv weights).

        Components:
        - Correctness (0.7): Test pass rate
        - Syntax (0.1): Code compiles
        - Format (0.1): Has solution tags + minimal trailing
        - Thinking (0.1): Has thinking block

        Args:
            completion: Model completion
            test_data: Dict with test_list, test_setup_code, test_imports

        Returns:
            Total reward (0.0 to 1.0)
        """
        parsed = self._parse_completion(completion)

        # Correctness: execute code and compute test pass rate
        correctness = 0.0
        if parsed["code"] and isinstance(test_data, dict):
            test_list = test_data.get("test_list", [])
            if test_list:
                test_setup = test_data.get("test_setup_code", "")
                test_imports = test_data.get("test_imports", [])

                setup_code = "\n".join(test_imports) if test_imports else ""
                if test_setup:
                    setup_code += f"\n{test_setup}"

                test_cases = []
                for test in test_list:
                    if setup_code:
                        test_cases.append(f"{setup_code}\n{test}")
                    else:
                        test_cases.append(test)

                # Execute and get pass rate
                result = check_code_executes(parsed["code"], test_cases, timeout=5.0)
                if result["total"] > 0:
                    pass_rate = result["passed"] / result["total"]
                    correctness = 0.7 * pass_rate

        # Syntax validity
        syntax = 0.1 if parsed["syntax_valid"] else 0.0

        # Solution format (has solution + minimal trailing)
        solution_format = 0.1 if (parsed["has_solution"] and parsed["trailing"] < 50) else 0.0

        # Thinking format
        thinking = 0.1 if parsed["has_thinking"] else 0.0

        return correctness + syntax + solution_format + thinking


# ────────────────────────────────────────────────────────────────────────────
# Triton Kernel Adapter
# ────────────────────────────────────────────────────────────────────────────
class TritonKernelAdapter(DatasetAdapter):
    """Triton kernel dataset adapter for GPU kernel optimization training.

    Uses UnifiedKernelTaskSource for training (JSONL-backed, 10K+ rows)
    and KernelBenchTaskSource for evaluation (HF-backed, 250 problems).

    GPU eval is managed via the pluggable eval backend system.
    """

    def __init__(
        self,
        *,
        dataset_path: str = "",
        gpu_eval: bool = False,
        eval_backend_name: str = "subprocess",
        eval_gpu_ids: list[int] | None = None,
        warmup_count: int = 20,
    ) -> None:
        self._dataset_path = dataset_path
        self._gpu_eval = gpu_eval
        self._eval_backend_name = eval_backend_name
        self._eval_gpu_ids = eval_gpu_ids or []
        self._warmup_count = warmup_count
        self._backend = None

    @property
    def name(self) -> str:
        return "triton_kernel"

    @property
    def question_field(self) -> str:
        return "pytorch_code"

    @property
    def answer_field(self) -> str:
        return "test_code"

    @property
    def correctness_weight(self) -> float:
        return 0.50

    @property
    def success_threshold(self) -> float:
        return 0.50

    def _ensure_backend(self) -> None:
        """Set up eval backend if gpu_eval is enabled."""
        if self._backend is not None or not self._gpu_eval:
            return

        from grail.environments.gpu_kernel.eval_backends import (
            create_backend,
            set_global_backend,
            validate_gpu_config,
        )

        validate_gpu_config(self._eval_gpu_ids, self._gpu_eval)

        self._backend = create_backend(
            self._eval_backend_name,
            gpu_ids=self._eval_gpu_ids,
        )
        self._backend.start()
        set_global_backend(self._backend)
        logger.info(
            "Triton kernel eval backend started: %s on GPUs %s",
            self._eval_backend_name,
            self._eval_gpu_ids,
        )

    def load_train_data(self) -> list[dict[str, Any]]:
        """Load training data from unified kernel dataset."""
        from grail.environments.gpu_kernel.task_sources import UnifiedKernelTaskSource

        if not self._dataset_path:
            raise ValueError(
                "TritonKernelAdapter requires --kernel-dataset-path for training. "
                "Run 'python -m research.datasets.build' to generate the dataset."
            )

        source = UnifiedKernelTaskSource(
            dataset_path=self._dataset_path,
            split="train",
            mode="rl",
            exclude_sources=["kernelbench"],
            weighted_sampling=True,
        )

        data = []
        for task_id in source.iter_ids():
            task = source.next(task_id=task_id)
            data.append({
                "question": task.payload.get("pytorch_code", ""),
                "test_code": task.payload.get("test_code", ""),
                "problem_name": task.payload.get("problem_name", ""),
            })

        logger.info("Loaded %d triton kernel training samples", len(data))
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        """Load eval data from KernelBench (val split)."""
        from grail.environments.gpu_kernel.task_sources import KernelBenchTaskSource

        source = KernelBenchTaskSource(split="val")

        data = []
        for task_id in source.iter_ids():
            task = source.next(task_id=task_id)
            data.append({
                "question": task.payload.get("pytorch_code", ""),
                "test_code": task.payload.get("test_code", ""),
                "problem_name": task.payload.get("problem_name", ""),
            })

        logger.info("Loaded %d triton kernel eval samples", len(data))
        return data

    def parse_gold_answer(self, raw_answer: Any) -> Any:
        """For triton kernels, gold data is the test_code."""
        return raw_answer

    def validate_answer(self, predicted: str, gold: Any) -> bool:
        """Validate by running GPU eval if available."""
        if not self._gpu_eval or self._backend is None:
            # Structural validation only
            return self._check_structure(predicted)

        test_code = gold if isinstance(gold, str) else ""
        if not test_code:
            return False

        result = self._backend.evaluate(test_code, predicted)
        return result.correct

    def _check_structure(self, code: str) -> bool:
        """Check if code has valid Triton structure."""
        from grail.environments.gpu_kernel.parser import TritonKernelParser

        parser = TritonKernelParser()
        parsed = parser.parse(f"<SOLUTION>\n{code}\n</SOLUTION>", {})
        return parsed.get("structure_valid", False)

    def _parse_completion(self, text: str) -> dict[str, Any]:
        """Parse completion using TritonKernelParser."""
        from grail.environments.gpu_kernel.parser import TritonKernelParser

        parser = TritonKernelParser()
        return parser.parse(text, {})

    def compute_reward(self, completion: str, gold_answer: Any) -> float:
        """Compute triton kernel reward (matching TritonKernelRubric weights).

        Components:
        - Compilation (0.05): Valid Python syntax
        - Structure (0.10): ModelNew + @triton.jit + imports
        - GPU Compilation (0.15): Code runs on GPU
        - Correctness (0.50): Outputs match reference
        - Format (0.10): <SOLUTION> tags
        - Thinking (0.10): Reasoning block
        """
        parsed = self._parse_completion(completion)
        code = parsed.get("code", "") or ""

        # Structural rewards (always available)
        compilation = 0.05 if parsed.get("syntax_valid", False) else 0.0

        structure_score = 0.0
        for key in ("has_model_new", "has_triton_jit", "has_triton_import", "has_torch_import"):
            if parsed.get(key, False):
                structure_score += 0.25
        structure = 0.10 * structure_score

        solution_format = (
            0.10
            if (parsed.get("has_solution", False) and parsed.get("trailing_after_solution", 100) < 50)
            else 0.0
        )
        thinking = 0.10 if parsed.get("has_thinking", False) else 0.0

        # GPU rewards (only if backend available and structure valid)
        gpu_compilation = 0.0
        correctness = 0.0

        if (
            self._gpu_eval
            and self._backend is not None
            and code
            and parsed.get("structure_valid", False)
        ):
            test_code = gold_answer if isinstance(gold_answer, str) else ""
            if test_code:
                result = self._backend.evaluate(test_code, code)
                if result.compiled:
                    gpu_compilation = 0.15
                if result.correct:
                    correctness = 0.50

        return compilation + structure + gpu_compilation + correctness + solution_format + thinking

    def get_gold_data(self, sample: dict[str, Any]) -> Any:
        """Get test_code from sample for evaluation."""
        return sample.get("test_code", "")

    def shutdown(self) -> None:
        """Clean up eval backend."""
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None


# ════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def get_dataset_adapter(dataset_name: str, **kwargs: Any) -> DatasetAdapter:
    """Factory function to get dataset adapter by name.

    Args:
        dataset_name: 'gsm8k', 'math', 'mbpp', or 'triton_kernel'
        **kwargs: Additional arguments passed to adapter constructor.

    Returns:
        DatasetAdapter instance

    Raises:
        ValueError: If dataset_name is not supported
    """
    adapters: dict[str, type[DatasetAdapter]] = {
        "gsm8k": GSM8KAdapter,
        "math": MATHAdapter,
        "mbpp": MBPPAdapter,
        "triton_kernel": TritonKernelAdapter,
    }

    if dataset_name.lower() not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")

    return adapters[dataset_name.lower()](**kwargs)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING PASS@K TRACKER
# ════════════════════════════════════════════════════════════════════════════
class TrainingPassAtKTracker:
    """Computes and logs pass@k metrics during GRPO training.

    This class wraps the reward computation and tracks pass@k metrics
    by grouping completions by their prompts. Uses the same unbiased pass@k
    formula as evaluation (KMetricsAggregator from grail.trainer.metrics).

    Usage:
        tracker = TrainingPassAtKTracker(adapter, prompt_to_answer)
        trainer = GRPOTrainer(..., reward_funcs=tracker, ...)
    """

    # Required by TRL GRPOTrainer for reward function naming
    __name__ = "reward_with_pass_at_k"

    def __init__(
        self,
        adapter: DatasetAdapter,
        prompt_to_answer: dict[str, Any],
        report_ks: tuple[int, ...] = (1, 5, 10),
    ) -> None:
        """Initialize the tracker.

        Args:
            adapter: Dataset adapter for reward computation and success threshold
            prompt_to_answer: Mapping from prompt text to gold answer data
            report_ks: Tuple of k values for pass@k metrics
        """
        self._adapter = adapter
        self._prompt_to_answer = prompt_to_answer
        self._report_ks = report_ks
        self._step_count = 0

    def __call__(
        self,
        completions: list[str],
        prompts: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """Compute rewards and log pass@k metrics.

        This method is called by GRPOTrainer for each batch of completions.

        Args:
            completions: List of model completions
            prompts: List of corresponding prompts
            **kwargs: Additional arguments (gold_answer, metadatas, etc.)

        Returns:
            List of reward values for each completion
        """
        gold_answers = self._extract_gold_answers(prompts, kwargs)
        rewards = self._compute_rewards(completions, gold_answers)
        metrics = self._compute_pass_at_k_metrics(prompts, rewards)
        self._log_to_wandb(metrics)
        self._step_count += 1
        return rewards

    def _extract_gold_answers(
        self,
        prompts: list[str],
        kwargs: dict[str, Any],
    ) -> list[Any]:
        """Extract gold answers from kwargs or prompt mapping.

        Returns:
            List of gold answers - can be strings (gsm8k/math) or dicts (mbpp)
        """
        if "gold_answer" in kwargs and kwargs["gold_answer"]:
            return kwargs["gold_answer"]
        if "metadatas" in kwargs and kwargs["metadatas"]:
            return [m.get("gold_answer", "") for m in kwargs["metadatas"]]
        return [self._prompt_to_answer.get(p, "") for p in prompts]

    def _compute_rewards(
        self,
        completions: list[str],
        gold_answers: list[Any],
    ) -> list[float]:
        """Compute reward for each completion.

        Args:
            completions: Model completions
            gold_answers: Gold data (strings for math, dicts for mbpp)
        """
        return [
            self._adapter.compute_reward(c, g)
            for c, g in zip(completions, gold_answers, strict=False)
        ]

    def _compute_pass_at_k_metrics(
        self,
        prompts: list[str],
        rewards: list[float],
    ) -> dict[str, float]:
        """Compute all metrics using KMetricsAggregator (unbiased pass@k formula)."""
        from collections import defaultdict

        # Group rewards by prompt
        prompt_groups: dict[str, list[float]] = defaultdict(list)
        for prompt, reward in zip(prompts, rewards, strict=False):
            prompt_groups[prompt].append(reward)

        group_count = len(prompt_groups)
        expected_groups = cfg.max_groups
        step_index = self._step_count + 1
        logger.info(
            "[TrainingPassAtKTracker] "
            f"Step {step_index}: grouped {group_count} prompts "
            f"(max_groups={expected_groups})"
        )
        if group_count != expected_groups:
            logger.warning(
                "[TrainingPassAtKTracker] ⚠️ "
                f"group_count ({group_count}) != max_groups ({expected_groups})"
            )

        # Use KMetricsAggregator for metrics computation
        aggregator = KMetricsAggregator(report_ks=self._report_ks)
        threshold = self._adapter.success_threshold

        for task_id, group_rewards in enumerate(prompt_groups.values()):
            successes = [r >= threshold for r in group_rewards]
            aggregator.add_group(
                task_id=str(task_id),
                rewards=group_rewards,
                successes=successes,
            )

        return aggregator.summarize()

    def _log_to_wandb(self, metrics: dict[str, float]) -> None:
        """Log metrics to WandB."""
        try:
            import wandb

            if wandb.run is not None and metrics:
                wandb_data = {f"train/{k}": v for k, v in metrics.items()}
                wandb.log(wandb_data)
        except Exception:
            pass  # Silently ignore WandB errors


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════
def prepare_train_dataset(adapter: DatasetAdapter, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load and format training dataset for TRL GRPO.

    Args:
        adapter: Dataset adapter instance
        tokenizer: Tokenizer for chat template formatting

    Returns:
        HuggingFace Dataset with 'prompt' and metadata columns
    """
    raw_data = adapter.load_train_data()

    if cfg.num_train_samples is not None:
        raw_data = raw_data[: cfg.num_train_samples]

    formatted = []
    for sample in raw_data:
        question = sample[adapter.question_field]
        prompt = apply_chat_template(
            tokenizer, [{"role": "user", "content": question}]
        )

        # For MBPP, store full test data dict; for others, store answer string
        if adapter.name == "mbpp":
            gold_data = {
                "test_list": sample.get("test_list", []),
                "test_setup_code": sample.get("test_setup_code", ""),
                "test_imports": sample.get("test_imports", []),
            }
        else:
            gold_data = sample[adapter.answer_field]

        formatted.append({
            "prompt": prompt,
            "gold_answer": gold_data,
        })

    logger.info(f"  Training dataset ({adapter.name}): {len(formatted)} samples")
    return Dataset.from_list(formatted)


def prepare_eval_dataset(adapter: DatasetAdapter) -> tuple[Dataset, list[dict[str, Any]]]:
    """Load evaluation dataset.

    Args:
        adapter: Dataset adapter instance

    Returns:
        Tuple of (HuggingFace Dataset, raw data list for reward computation)
    """
    raw_data = adapter.load_eval_data()

    if cfg.num_eval_samples is not None:
        raw_data = raw_data[: cfg.num_eval_samples]

    logger.info(f"  Eval dataset ({adapter.name}): {len(raw_data)} samples")
    return Dataset.from_list(raw_data), raw_data


# ════════════════════════════════════════════════════════════════════════════
# VLLM EVALUATION CALLBACK
# ════════════════════════════════════════════════════════════════════════════
class VLLMEvalCallback(TrainerCallback):
    """Evaluation callback using TRL vLLM server with dataset adapter."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        eval_data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        vllm_base_url: str,
        eval_every_n_steps: int = 40,
    ) -> None:
        self.adapter = adapter
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.eval_every_n = eval_every_n_steps
        self.base_url = vllm_base_url.rstrip("/")
        self._wandb_configured = False

        logger.info(
            f"✓ VLLMEvalCallback initialized: dataset={adapter.name}, "
            f"url={vllm_base_url}, eval_every={eval_every_n_steps}"
        )

    def run_and_log(self, step: int, label: str = "VLLM EVAL") -> dict[str, float]:
        """Run evaluation and log to WandB."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{label}] Step {step}: Starting {self.adapter.name.upper()} evaluation...")
        logger.info(f"{'=' * 80}")

        profiler = get_profiler()
        with profiler.track("evaluation"):
            metrics = asyncio.run(self._run_eval())

        try:
            import wandb

            if wandb.run is not None:
                # Configure step metric for eval on first call
                if not self._wandb_configured:
                    wandb.define_metric("eval_step")
                    wandb.define_metric("eval/*", step_metric="eval_step")
                    self._wandb_configured = True

                # Log eval metrics with 'eval/' prefix and custom step
                wandb_data = {"eval_step": step}
                wandb_data.update({f"eval/{k}": v for k, v in metrics.items()})
                wandb.log(wandb_data)
        except Exception as e:
            logger.warning(f"⚠️  WandB logging failed: {e}")

        logger.info(f"[{label}] Results: {metrics}")
        logger.info(f"{'=' * 80}\n")
        return metrics

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Run evaluation every N steps."""
        if state.global_step >= self.eval_every_n and state.global_step % self.eval_every_n == 0:
            self.run_and_log(state.global_step)

    async def _run_eval(self) -> dict[str, float]:
        """Run evaluation using vLLM chat completions API."""
        import time

        from tqdm import tqdm

        start_time = time.time()
        aggregator = KMetricsAggregator(report_ks=cfg.report_ks)

        total_tasks = len(self.eval_data)
        batch_size = cfg.eval_batch_size

        with tqdm(total=total_tasks, desc=f"Eval ({self.adapter.name})", unit="task") as pbar:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch = self.eval_data[batch_start:batch_end]

                # Get questions using adapter's field name
                batch_questions = [s[self.adapter.question_field] for s in batch]
                # Use get_gold_data() for proper data extraction (esp. MBPP dict)
                batch_golds = [self.adapter.get_gold_data(s) for s in batch]

                # Expand: each question gets N replicates
                tasks_to_generate = []
                task_metadata = []

                for idx, question in enumerate(batch_questions):
                    task_id = f"q{batch_start + idx}"
                    for rep_idx in range(cfg.eval_replicates):
                        tasks_to_generate.append(question)
                        task_metadata.append(
                            {
                                "task_id": task_id,
                                "task_idx": idx,
                                "replicate_idx": rep_idx,
                            }
                        )

                # Generate completions
                completions = await self._generate_batch(tasks_to_generate)

                # Log sample completions
                if batch_start == 0:
                    logger.info("\n  ━━━ Sample Completions ━━━")
                    for i in range(min(3, len(completions))):
                        question = tasks_to_generate[i]
                        completion = completions[i]
                        metadata = task_metadata[i]
                        gold = batch_golds[metadata["task_idx"]]
                        reward = self.adapter.compute_reward(completion, gold)

                        q_display = question[:150] + "..." if len(question) > 150 else question
                        c_display = (
                            completion[:300] + "..." if len(completion) > 300 else completion
                        )
                        # Handle both string (math/gsm8k) and dict (mbpp) gold answers
                        if isinstance(gold, str):
                            gold_display = gold[:50] + "..." if len(gold) > 50 else gold
                        else:
                            # MBPP: show test count instead of raw dict
                            test_count = len(gold.get("test_list", [])) if isinstance(gold, dict) else 0
                            gold_display = f"[{test_count} test cases]"
                        logger.info(f"\n  Sample {i + 1}:")
                        logger.info(f"    Question: {q_display}")
                        logger.info(f"    Completion: {c_display}")
                        logger.info(f"    Reward: {reward:.3f} | Gold: {gold_display}")
                    logger.info("  ━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                # Compute rewards and aggregate
                for completion_text, metadata in zip(completions, task_metadata, strict=False):
                    task_id = metadata["task_id"]
                    task_idx = metadata["task_idx"]
                    replicate_idx = metadata["replicate_idx"]
                    gold = batch_golds[task_idx]

                    reward = self.adapter.compute_reward(completion_text, gold)
                    success = reward >= self.adapter.success_threshold

                    aggregator.add(
                        TaskReplicateResult(
                            task_id=task_id,
                            replicate_idx=replicate_idx,
                            reward=reward,
                            success=success,
                        )
                    )

                pbar.update(len(batch_questions))

        metrics = aggregator.summarize()
        elapsed = time.time() - start_time
        throughput = (total_tasks * cfg.eval_replicates) / elapsed if elapsed > 0 else 0

        logger.info(
            f"  ✓ Evaluated {total_tasks} tasks × {cfg.eval_replicates} reps in {elapsed:.2f}s "
            f"({throughput:.1f} completions/sec)"
        )

        return metrics

    async def _generate_batch(self, questions: list[str]) -> list[str]:
        """Generate completions using TRL /chat/ endpoint with batching."""
        import asyncio

        import aiohttp

        vllm_batch_size = 64
        total = len(questions)
        num_requests = (total + vllm_batch_size - 1) // vllm_batch_size
        logger.info(f"    Generating {total} completions via {num_requests} batched requests")

        async def generate_batch_request(
            session: aiohttp.ClientSession, batch_questions: list[str], start_idx: int
        ) -> tuple[int, list[list[int]]]:
            max_retries = 3
            base_backoff = 1.0

            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                for q in batch_questions
            ]

            payload = {
                "messages": messages,
                "max_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "repetition_penalty": 1.1,
                "n": 1,
            }

            for attempt in range(max_retries):
                try:
                    async with session.post(
                        f"{self.base_url}/chat/",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300.0),
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return (start_idx, data["completion_ids"])
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        backoff = base_backoff * (2**attempt)
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(f"  ⚠️  Batch {start_idx} failed: {type(e).__name__}")
                        return (start_idx, [[] for _ in batch_questions])
            return (start_idx, [[] for _ in batch_questions])

        async with aiohttp.ClientSession() as session:
            tasks = []
            for batch_start in range(0, total, vllm_batch_size):
                batch_end = min(batch_start + vllm_batch_size, total)
                batch_questions = questions[batch_start:batch_end]
                tasks.append(generate_batch_request(session, batch_questions, batch_start))

            results = await asyncio.gather(*tasks, return_exceptions=False)

        all_completion_ids: list[list[int]] = [[] for _ in range(total)]
        for start_idx, completion_ids_batch in results:
            for offset, comp_ids in enumerate(completion_ids_batch):
                all_completion_ids[start_idx + offset] = comp_ids

        completions = []
        for comp_ids in all_completion_ids:
            if comp_ids:
                completion_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions.append(completion_text)
            else:
                completions.append("")

        return completions


# ════════════════════════════════════════════════════════════════════════════
# MEMORY ESTIMATION
# ════════════════════════════════════════════════════════════════════════════
def estimate_memory_requirements(
    model_size_b: float,
    batch_size: int,
    seq_len: int,
    gradient_checkpointing: bool,
    fp32_master_weights: bool = False,
) -> dict[str, float]:
    """Estimate GPU memory requirements for training.

    Args:
        model_size_b: Model size in billions of parameters (e.g., 7.0 for 7B)
        batch_size: Microbatch size per device
        seq_len: Maximum sequence length
        gradient_checkpointing: Whether gradient checkpointing is enabled
        fp32_master_weights: Whether using FP32 master weights (vs BF16)

    Returns:
        Dict with memory estimates in GB
    """
    if fp32_master_weights:
        # FP32 master weights: model in FP32, optimizer states in FP32
        model_gb = model_size_b * 4  # 4 bytes per param in fp32
        optimizer_gb = model_size_b * 4 * 2  # 4 bytes × 2 states (fp32)
        gradients_gb = model_size_b * 4  # Gradients accumulated in fp32
    else:
        # BF16 training: model in BF16, optimizer states in BF16
        model_gb = model_size_b * 2  # 2 bytes per param in bf16
        optimizer_gb = model_size_b * 2 * 2  # 2 bytes × 2 states (bf16)
        gradients_gb = model_size_b * 2  # Gradients in bf16

    # Activation memory estimation (rough heuristic)
    # Scales with batch × seq × model_size
    # Gradient checkpointing reduces by ~5-10x
    base_activation_gb = batch_size * (seq_len / 1024) * (model_size_b / 7) * 4
    if gradient_checkpointing:
        activation_gb = base_activation_gb * 0.15  # ~85% reduction
    else:
        activation_gb = base_activation_gb

    # Overhead (CUDA context, fragmentation, etc.)
    overhead_gb = 2.0

    total_gb = model_gb + optimizer_gb + gradients_gb + activation_gb + overhead_gb

    return {
        "model_gb": model_gb,
        "optimizer_gb": optimizer_gb,
        "gradients_gb": gradients_gb,
        "activation_gb": activation_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
    }


def print_memory_estimate(
    model_id: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
    gradient_checkpointing: bool | None = None,
    fp32_master_weights: bool = False,
) -> None:
    """Print memory estimation and warnings for current config.

    Args:
        model_id: Model identifier used to infer parameter count.
        batch_size: Override batch size used in estimate.
        seq_len: Override sequence length used in estimate.
        gradient_checkpointing: Override gradient checkpointing flag.
        fp32_master_weights: Whether using FP32 master weights.
    """
    # Estimate model size from model_id
    model_id_lower = model_id.lower()
    if "72b" in model_id_lower:
        model_size_b = 72.0
    elif "32b" in model_id_lower:
        model_size_b = 32.0
    elif "14b" in model_id_lower:
        model_size_b = 14.0
    elif "7b" in model_id_lower:
        model_size_b = 7.0
    elif "3b" in model_id_lower:
        model_size_b = 3.0
    elif "1.5b" in model_id_lower or "1b" in model_id_lower:
        model_size_b = 1.5
    elif "0.5b" in model_id_lower or "500m" in model_id_lower:
        model_size_b = 0.5
    else:
        # Default assumption for unknown models
        model_size_b = 7.0

    effective_batch_size = cfg.batch_size if batch_size is None else batch_size
    effective_seq_len = cfg.max_length if seq_len is None else seq_len
    effective_gradient_checkpointing = (
        cfg.gradient_checkpointing if gradient_checkpointing is None else gradient_checkpointing
    )

    mem = estimate_memory_requirements(
        model_size_b=model_size_b,
        batch_size=effective_batch_size,
        seq_len=effective_seq_len,
        gradient_checkpointing=effective_gradient_checkpointing,
        fp32_master_weights=fp32_master_weights,
    )

    # Get available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        gpu_memory_gb = 80.0  # Assume A100

    # Determine dtype labels based on master weight setting
    model_dtype = "fp32" if fp32_master_weights else "bf16"
    optim_dtype = "fp32" if fp32_master_weights else "bf16"
    grad_dtype = "fp32" if fp32_master_weights else "bf16"

    logger.info("\n💾 Memory Estimation:")
    logger.info("─" * 60)
    logger.info(f"  Model ({model_size_b:.1f}B params, {model_dtype}):    {mem['model_gb']:.1f} GB")
    logger.info(f"  Optimizer states ({optim_dtype}):        {mem['optimizer_gb']:.1f} GB")
    logger.info(f"  Gradients ({grad_dtype}):               {mem['gradients_gb']:.1f} GB")
    ckpt_status = "enabled" if cfg.gradient_checkpointing else "disabled"
    logger.info(f"  Activations (checkpointing {ckpt_status}): {mem['activation_gb']:.1f} GB")
    logger.info(f"  Overhead:                       {mem['overhead_gb']:.1f} GB")
    logger.info("─" * 60)
    logger.info(f"  Estimated total:                {mem['total_gb']:.1f} GB")
    logger.info(f"  Available GPU memory:           {gpu_memory_gb:.1f} GB")

    headroom = gpu_memory_gb - mem["total_gb"]
    if headroom < 0:
        logger.warning(f"  ⚠️  WARNING: Estimated {-headroom:.1f} GB OVER capacity!")
        logger.warning("     → Try: --batch-size 1 or reduce --max-length")
    elif headroom < 5:
        logger.warning(f"  ⚠️  TIGHT: Only {headroom:.1f} GB headroom - may OOM on long sequences")
    else:
        logger.info(f"  ✓ Headroom: {headroom:.1f} GB")
    logger.info("─" * 60)


def get_gradient_checkpointing_kwargs(model_id: str) -> dict[str, Any]:
    """Get gradient checkpointing kwargs based on model architecture.

    Different model architectures have different requirements for gradient
    checkpointing. In particular:

    - Gemma 3: Uses sliding window attention (alternating local/global layers)
      which requires use_reentrant=False for correct gradient computation.
    - Gemma 2: Also uses sliding window attention, requires use_reentrant=False.
    - Qwen/Llama/other: Can use either, but use_reentrant=False is recommended
      as it's more robust and becoming the PyTorch default.

    Args:
        model_id: The HuggingFace model identifier

    Returns:
        Dict with gradient_checkpointing_kwargs for the trainer config
    """
    model_id_lower = model_id.lower()

    # Gemma models require use_reentrant=False due to sliding window attention
    if "gemma" in model_id_lower:
        logger.info("  → Gemma model detected: using use_reentrant=False (required for sliding window attention)")
        return {"use_reentrant": False}

    # For all other models, use_reentrant=False is recommended (PyTorch future default)
    # This includes Qwen, Llama, Mistral, etc.
    return {"use_reentrant": False}


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TRL GRPO training with GSM8K, MATH, MBPP, or Triton Kernel dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "mbpp", "triton_kernel"],
        help="Dataset to use for training (default: gsm8k)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=40,
        help="Run evaluation every N steps (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=8000,
        help="VLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--group-port",
        type=int,
        default=51216,
        help="NCCL group coordination port for vLLM weight sync (default: 51216). "
             "Must be unique per parallel instance.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Suffix for run name and output directories (default: empty)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of training updates per batch of rollouts (default: 1)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (overrides WANDB_PROJECT env var)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags (overrides WANDB_TAGS env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use (overrides default Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device to use for training (e.g., 'cuda:0', 'cuda:1'). "
        "If not specified, uses CUDA_VISIBLE_DEVICES default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size per device. For 7B models on 80GB GPU: use 1-2. "
        "For 1.5B models: use 2-4. Default from config.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps. Effective batch = batch_size * grad_accum_steps. "
        "Default from config (256).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (uses more memory but faster training).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: 3e-6 or GRAIL_TRAINER_LR env var).",
    )
    parser.add_argument(
        "--fp32-master-weights",
        action="store_true",
        help="Use FP32 master weights with BF16 training. Model weights and optimizer "
        "states are kept in FP32 for numerical stability, while forward/backward "
        "passes use BF16 via autocast. Increases memory by ~2x for model+optimizer.",
    )
    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=None,
        help="Override Adam beta2 (second moment decay rate, default: 0.999).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float32"],
        help="Training precision dtype (default: bfloat16). Use float32 for full-precision training.",
    )

    # Triton kernel-specific arguments
    parser.add_argument(
        "--kernel-dataset-path",
        type=str,
        default="",
        help="Path to unified kernel JSONL dataset (required for --dataset triton_kernel).",
    )
    parser.add_argument(
        "--kernel-eval-backend",
        type=str,
        default="subprocess",
        choices=["subprocess", "affinetes", "modal"],
        help="Eval backend for Triton kernel GPU evaluation (default: subprocess).",
    )
    parser.add_argument(
        "--kernel-eval-gpu-ids",
        type=str,
        default="",
        help="Comma-separated GPU IDs for kernel eval (e.g., '2,3'). "
             "Uses KERNEL_EVAL_GPU_IDS env var if not specified.",
    )
    parser.add_argument(
        "--kernel-gpu-eval",
        action="store_true",
        help="Enable GPU-based kernel evaluation (default: structural rewards only).",
    )
    parser.add_argument(
        "--kernel-warmup-count",
        type=int,
        default=20,
        help="Number of kernels to compile during JIT warmup (default: 20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config via CLI (takes precedence over defaults)
    if args.model:
        cfg.model_id = args.model
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
    if args.lr is not None:
        cfg.lr = args.lr
    if args.adam_beta2 is not None:
        cfg.adam_beta2 = args.adam_beta2
    if args.dtype is not None:
        cfg.dtype = args.dtype
    # Auto-set delta checkpoint dtype to match training dtype
    if cfg.dtype == "float32":
        cfg.delta_checkpoint_dtype = "float32"

    # Set random seeds for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f"🚀 Starting TRL GRPO training with {args.dataset.upper()} dataset")
    logger.info(f"   Seed: {args.seed} | VLLM Port: {args.vllm_port} | Num Iterations: {args.num_iterations}")
    logger.info(f"   Model: {cfg.model_id}")
    logger.info(f"   Batch size: {cfg.batch_size} | Grad accum: {cfg.grad_accum_steps} | Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    logger.info(f"   Gradient checkpointing: {cfg.gradient_checkpointing}")
    logger.info(f"   FP32 master weights: {args.fp32_master_weights}")
    if args.run_suffix:
        logger.info(f"   Run suffix: {args.run_suffix}")
    logger.info("=" * 80)

    # Initialize fast code execution pool for MBPP dataset
    # This eliminates ~6s spawn overhead per code execution (7000x speedup)
    execution_pool: CodeExecutionPool | None = None
    if args.dataset == "mbpp":
        try:
            execution_pool = CodeExecutionPool(
                num_workers=8,
                max_tasks_per_child=50,
            )
            execution_pool.start()
            set_global_execution_pool(execution_pool)
            logger.info("✅ Fast code execution pool initialized: 8 workers")
        except Exception as e:
            logger.warning(f"⚠️  Failed to init execution pool, using slow path: {e}")
            execution_pool = None

    # Print hyperparameter alignment summary
    logger.info("\n📋 GRAIL Hyperparameter Alignment Summary:")
    logger.info("─" * 80)
    logger.info(f"  {'Parameter':<40} {'Value':<15} {'GRAIL Env Var'}")
    logger.info("─" * 80)
    logger.info(f"  {'Model ID':<40} {cfg.model_id:<15} GRAIL_TRAIN_MODEL_ID")
    logger.info(f"  {'Training Dtype':<40} {cfg.dtype:<15} GRAIL_DTYPE")
    logger.info(f"  {'Learning Rate':<40} {cfg.lr:<15} GRAIL_TRAINER_LR")
    logger.info(f"  {'Epochs (per window)':<40} {cfg.epochs:<15} GRAIL_TRAINER_EPOCHS")
    logger.info(f"  {'Batch Size':<40} {cfg.batch_size:<15} GRAIL_TRAINER_BATCH_SIZE")
    logger.info(
        f"  {'Gradient Accum Steps':<40} {cfg.grad_accum_steps:<15} GRAIL_TRAINER_GRAD_ACCUM_STEPS"
    )
    logger.info(f"  {'Max Length':<40} {cfg.max_length:<15} GRAIL_TRAINER_MAX_LENGTH")
    logger.info(f"  {'Max Completion Tokens':<40} {cfg.max_new_tokens:<15} GRPO_MAX_COMPLETION_TOKENS")
    logger.info(f"  {'Gradient Clip':<40} {cfg.grad_clip:<15} GRAIL_TRAINER_GRAD_CLIP")
    logger.info(f"  {'Warmup Steps':<40} {cfg.warmup_steps:<15} GRAIL_TRAINER_WARMUP_STEPS")
    logger.info(f"  {'Total Steps':<40} {cfg.total_steps:<15} GRAIL_TRAINER_TOTAL_STEPS")
    logger.info(f"  {'KL Coefficient':<40} {cfg.kl_coef:<15} GRAIL_TRAINER_KL_COEF")
    logger.info(f"  {'Entropy Coefficient':<40} {cfg.entropy_coef:<15} GRAIL_TRAINER_ENTROPY_COEF")
    logger.info(f"  {'PPO Clip Epsilon':<40} {cfg.epsilon:<15} TRAINER_PPO_CLIP_EPS")
    logger.info(
        f"  {'PPO Clip Epsilon Upper':<40} {cfg.epsilon_high:<15} TRAINER_PPO_CLIP_EPS_UPPER"
    )
    logger.info(f"  {'IS Ratio Max':<40} {cfg.is_ratio_max:<15} GRAIL_TRAINER_IS_RATIO_MAX")
    logger.info(f"  {'GRPO Variant':<40} {cfg.grpo_variant:<15} GRAIL_GRPO_VARIANT")
    logger.info(f"  {'IS Level':<40} {cfg.importance_sampling_level:<15} GRAIL_IMPORTANCE_SAMPLING_LEVEL")
    logger.info(f"  {'Adam Beta2':<40} {cfg.adam_beta2:<15} GRAIL_ADAM_BETA2")
    logger.info(f"  {'Max Groups':<40} {cfg.max_groups:<15} GRPO_MAX_GROUPS")
    logger.info(f"  {'Rollouts per Problem':<40} {cfg.rollouts_per_problem:<15} ROLLOUTS_PER_PROBLEM")
    logger.info(f"  {'Gradient Checkpointing':<40} {cfg.gradient_checkpointing!s:<15} Memory optimization")
    logger.info(f"  {'FP32 Master Weights':<40} {args.fp32_master_weights!s:<15} Numerical precision")
    logger.info("─" * 80)

    # Print memory estimation
    print_memory_estimate(cfg.model_id, fp32_master_weights=args.fp32_master_weights)

    # Get dataset adapter
    adapter_kwargs: dict[str, Any] = {}
    if args.dataset == "triton_kernel":
        # Parse GPU IDs from CLI or env var
        gpu_ids_str = args.kernel_eval_gpu_ids or os.environ.get("KERNEL_EVAL_GPU_IDS", "")
        eval_gpu_ids = [int(x) for x in gpu_ids_str.split(",") if x.strip()] if gpu_ids_str else []
        adapter_kwargs = {
            "dataset_path": args.kernel_dataset_path,
            "gpu_eval": args.kernel_gpu_eval,
            "eval_backend_name": args.kernel_eval_backend,
            "eval_gpu_ids": eval_gpu_ids,
            "warmup_count": args.kernel_warmup_count,
        }
    adapter = get_dataset_adapter(args.dataset, **adapter_kwargs)
    logger.info("\n📚 Dataset Configuration:")
    logger.info(f"  Dataset: {adapter.name}")
    logger.info(f"  Correctness weight: {adapter.correctness_weight}")
    logger.info(f"  Success threshold: {adapter.success_threshold}")

    # Initialize profiler
    profiler = get_profiler()

    # Load model and tokenizer
    logger.info("\n📦 Loading model and tokenizer...")
    # Determine device for training
    if args.device:
        train_device = torch.device(args.device)
        # Set default CUDA device so model loads to correct GPU
        if train_device.type == "cuda":
            torch.cuda.set_device(train_device)
        logger.info(f"  Using device: {train_device}")
    else:
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_dtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    logger.info(f"  Model dtype: {torch_dtype}")

    with profiler.track("model_loading"):
        if cfg.dtype == "float32":
            # Flash Attention 2 doesn't support fp32; use SDPA directly
            logger.info(f"  Full FP32 mode: using SDPA attention (Flash Attention 2 requires bf16/fp16)")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
                device_map=train_device,
            )
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_id,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    device_map=train_device,
                )
            except (ImportError, RuntimeError) as e:
                logger.warning(f"⚠️  Flash Attention 2 unavailable ({type(e).__name__}), using SDPA")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_id,
                    torch_dtype=torch_dtype,
                    attn_implementation="sdpa",
                    device_map=train_device,
                )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        configure_tokenizer(tokenizer)

    # Prepare datasets
    logger.info("\n📊 Preparing datasets...")
    with profiler.track("dataset_preparation"):
        train_ds = prepare_train_dataset(adapter, tokenizer)
        eval_ds, eval_data = prepare_eval_dataset(adapter)
        prompt_to_answer = {row["prompt"]: row["gold_answer"] for row in train_ds}

    # WandB setup
    logger.info("\n⚙️  Configuring GRPO trainer...")
    import wandb

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        effective_project = args.wandb_project or os.getenv("WANDB_PROJECT", "grail")
        logger.info(f"  ✓ WandB logged in (project: {effective_project})")

    # Calculate training schedule
    # Each optimizer step = generation_batch_size = effective_batch = 512 samples
    # = 32 groups × 16 rollouts
    effective_batch = cfg.batch_size * cfg.grad_accum_steps  # 4 × 128 = 512
    groups_per_step = effective_batch // cfg.rollouts_per_problem  # 512 / 16 = 32
    total_optimizer_steps = cfg.total_steps  # Fixed: maintains original training duration

    logger.info("\n📊 Training Schedule:")
    logger.info(f"  • Effective batch size: {effective_batch} samples")
    logger.info(f"  • Groups per optimizer step: {groups_per_step}")
    logger.info(f"  • Rollouts per group: {cfg.rollouts_per_problem}")
    logger.info(f"  • Total optimizer steps: {total_optimizer_steps}")

    # Build run identifiers with seed and suffix
    run_id = f"seed{args.seed}" if not args.run_suffix else args.run_suffix

    # Use GRAIL_OUTPUT_BASE env var for output directories (default: current dir)
    # Set to /ephemeral on cloud instances with limited home storage
    output_base = os.getenv("GRAIL_OUTPUT_BASE", ".")

    grpo_config = GRPOConfig(
        output_dir=f"{output_base}/outputs/trl_{adapter.name}_{run_id}",
        # ─────────────────────────────────────────────────────────────────────
        # Learning Rate & Schedule (matching GRAIL trainer config)
        # ─────────────────────────────────────────────────────────────────────
        learning_rate=cfg.lr,  # GRAIL_TRAINER_LR
        warmup_steps=cfg.warmup_steps,  # GRAIL_TRAINER_WARMUP_STEPS
        lr_scheduler_type="constant_with_warmup",  # warmup then constant
        # Use max_steps to control iterations (matching GRAIL_TRAINER_TOTAL_WINDOWS)
        # num_train_epochs is ignored when max_steps is set
        num_train_epochs=cfg.epochs,
        max_steps=total_optimizer_steps,  # Calculated from total_windows
        # ─────────────────────────────────────────────────────────────────────
        # Batch Size & Gradient Accumulation (matching GRAIL trainer config)
        # ─────────────────────────────────────────────────────────────────────
        per_device_train_batch_size=cfg.batch_size,  # GRAIL_TRAINER_BATCH_SIZE
        gradient_accumulation_steps=cfg.grad_accum_steps,  # GRAIL_TRAINER_GRAD_ACCUM_STEPS
        max_grad_norm=cfg.grad_clip,  # GRAIL_TRAINER_GRAD_CLIP
        # ─────────────────────────────────────────────────────────────────────
        # Optimizer Configuration (AdamW defaults)
        # ─────────────────────────────────────────────────────────────────────
        optim="adamw_torch",  # Optimizer type (default PyTorch AdamW)
        adam_beta1=0.9,  # Beta1 momentum (default: 0.9)
        adam_beta2=cfg.adam_beta2,  # Beta2 momentum (default: 0.999)
        adam_epsilon=1e-8,  # Numerical stability (default: 1e-8)
        weight_decay=0.0,  # L2 regularization (default: 0.0)
        # ─────────────────────────────────────────────────────────────────────
        # GRPO Loss Configuration (matching grail/trainer/algorithms/grpo.py)
        # ─────────────────────────────────────────────────────────────────────
        num_iterations=args.num_iterations,  # Number of training updates on generated rollouts (μ in GRPO)
        beta=cfg.kl_coef,  # GRAIL_TRAINER_KL_COEF (KL divergence coefficient)
        epsilon=cfg.epsilon,  # TRAINER_PPO_CLIP_EPS (lower clip bound)
        epsilon_high=cfg.epsilon_high,  # TRAINER_PPO_CLIP_EPS_UPPER (DAPO asymmetric)
        loss_type=cfg.grpo_variant,  # GRAIL_GRPO_VARIANT ("dapo")
        # ─────────────────────────────────────────────────────────────────────
        # Sequence Length (matching GRAIL trainer config)
        # ─────────────────────────────────────────────────────────────────────
        max_completion_length=cfg.max_new_tokens,  # GRPO_MAX_COMPLETION_TOKENS
        # ─────────────────────────────────────────────────────────────────────
        # Importance Sampling Level
        # ─────────────────────────────────────────────────────────────────────
        importance_sampling_level=cfg.importance_sampling_level,  # GRAIL_IMPORTANCE_SAMPLING_LEVEL
        # ─────────────────────────────────────────────────────────────────────
        # Generation Parameters
        # ─────────────────────────────────────────────────────────────────────
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=1.1,
        num_generations=cfg.rollouts_per_problem,  # ROLLOUTS_PER_PROBLEM
        # generation_batch_size must equal effective_batch to ensure:
        # - One generation per optimizer step (no stale advantages)
        # - 32 groups × 16 rollouts = 512 samples per optimizer update
        generation_batch_size=cfg.batch_size * cfg.grad_accum_steps,  # 4 × 128 = 512
        # ─────────────────────────────────────────────────────────────────────
        # Logging & Checkpointing
        # ─────────────────────────────────────────────────────────────────────
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        save_strategy="steps",
        save_steps=50,
        bf16=(cfg.dtype == "bfloat16"),
        # ─────────────────────────────────────────────────────────────────────
        # Memory Optimization
        # ─────────────────────────────────────────────────────────────────────
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=get_gradient_checkpointing_kwargs(cfg.model_id),
        report_to=["wandb"],
        eval_strategy="no",
        run_name=f"trl_{adapter.name}_grpo_{cfg.model_id.replace('/', '_')}_{run_id}",
        # ─────────────────────────────────────────────────────────────────────
        # vLLM Configuration
        # ─────────────────────────────────────────────────────────────────────
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=f"http://127.0.0.1:{args.vllm_port}",
        vllm_group_port=args.group_port,  # CRITICAL: unique port per instance for parallel runs
        vllm_importance_sampling_correction=False,
        vllm_importance_sampling_cap=cfg.is_ratio_max,  # GRAIL_TRAINER_IS_RATIO_MAX
    )

    # HuggingFace Trainer will wrap the model in DataParallel when `args.n_gpu > 1`.
    # In our vLLM-server setup we intentionally keep both GPUs visible (for NCCL peer access),
    # but training must remain single-GPU to avoid accidentally using the vLLM GPU.
    _ = grpo_config.device  # trigger device setup + cache it
    if (
        grpo_config.use_vllm
        and grpo_config.vllm_mode == "server"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        grpo_config._n_gpu = 1
        logger.info("  ✓ Forced Trainer to single-GPU (n_gpu=1) to avoid DataParallel on vLLM GPU")

    # Create reward tracker with pass@k logging
    reward_tracker = TrainingPassAtKTracker(
        adapter=adapter,
        prompt_to_answer=prompt_to_answer,
        report_ks=cfg.report_ks,
    )
    logger.info(f"  ✓ TrainingPassAtKTracker initialized (report_ks={cfg.report_ks})")

    logger.info(f"\n🏋️  Training with GRPO on {adapter.name.upper()}...")

    # Initialize sparsity analysis (parameter change tracking + gradient sparsity)
    sparsity_config = AnalysisConfig(
        interval=1,
        param_change_enabled=True,
        param_change_thresholds=[0.0],  # Only track exact zero weight deltas
        sparse_quality_enabled=False,
        snapshot_dtype=cfg.delta_checkpoint_dtype,  # Match training precision
        gradient_enabled=True,  # Enable gradient analysis
    )
    sparsity_analyzer = ModelAnalysisManager.create(sparsity_config)

    # Add gradient sparsity metric
    gradient_sparsity = GradientSparsityMetrics(
        thresholds=[0.0, 10, 1, 1e-4, 1e-8, 1e-16, 1e-20],  # Only track exact zero gradients
        track_per_layer=False,
    )
    sparsity_analyzer.add_metric(gradient_sparsity)

    # Add Adam sign descent metric
    adam_sign_metrics = AdamSignDescentMetrics(
        track_per_component=True,
        histogram_samples=1_000_000,
    )
    sparsity_analyzer.add_metric(adam_sign_metrics)

    sparsity_callback = SparsityCallback(sparsity_analyzer)
    logger.info(f"  ✓ Sparsity analysis enabled (interval={sparsity_config.interval})")

    # Initialize delta checkpoint callback
    delta_checkpoint_callback = DeltaCheckpointCallback(
        output_dir=f"{output_base}/checkpoints/deltas_{adapter.name}_{run_id}",
        enabled=cfg.delta_checkpoint_enabled,
        snapshot_dtype=cfg.delta_checkpoint_dtype,
        profiler=profiler,
    )
    if cfg.delta_checkpoint_enabled:
        logger.info(f"  ✓ Delta checkpointing enabled (threshold=0.0, exact sparsity, dtype={cfg.delta_checkpoint_dtype})")
    else:
        logger.info(f"  ✗ Delta checkpointing disabled (set cfg.delta_checkpoint_enabled=True to enable)")

    # Initialize evaluation callback
    vllm_eval_callback = VLLMEvalCallback(
        adapter=adapter,
        eval_data=eval_data,
        tokenizer=tokenizer,
        vllm_base_url=grpo_config.vllm_server_base_url,
        eval_every_n_steps=args.eval_every,
    )

    logger.info(f"  ✓ Using vllm_group_port={args.group_port} (set via GRPOConfig)")

    # Ensure CUDA device is correctly set before GRPOTrainer initialization
    # GRPOTrainer uses torch.cuda.current_device() for NCCL communicator setup
    if args.device:
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(device_idx)
        logger.info(f"  ✓ CUDA device set to {device_idx} for NCCL communicator")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_tracker,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[vllm_eval_callback, sparsity_callback, delta_checkpoint_callback],
    )

    # Initialize WandB explicitly before baseline eval (GRPOTrainer does it lazily in .train())
    import wandb

    if wandb.run is None and grpo_config.report_to and "wandb" in grpo_config.report_to:
        # CLI args take precedence over env vars
        wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "grail")
        wandb_tags_str = args.wandb_tags or os.getenv("WANDB_TAGS", "")
        wandb_tags = [t.strip() for t in wandb_tags_str.split(",") if t.strip()]

        # Build comprehensive config for wandb (includes both our Config and GRPOConfig)
        wandb_config = {
            # Our hyperparameters (Config class)
            "grail/model_id": cfg.model_id,
            "grail/dtype": cfg.dtype,
            "grail/learning_rate": cfg.lr,
            "grail/batch_size": cfg.batch_size,
            "grail/grad_accum_steps": cfg.grad_accum_steps,
            "grail/effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
            "grail/max_length": cfg.max_length,
            "grail/max_new_tokens": cfg.max_new_tokens,
            "grail/grad_clip": cfg.grad_clip,
            "grail/warmup_steps": cfg.warmup_steps,
            "grail/total_steps": cfg.total_steps,
            "grail/kl_coef": cfg.kl_coef,
            "grail/entropy_coef": cfg.entropy_coef,
            "grail/epsilon": cfg.epsilon,
            "grail/epsilon_high": cfg.epsilon_high,
            "grail/is_ratio_max": cfg.is_ratio_max,
            "grail/grpo_variant": cfg.grpo_variant,
            "grail/importance_sampling_level": cfg.importance_sampling_level,
            "grail/max_groups": cfg.max_groups,
            "grail/rollouts_per_problem": cfg.rollouts_per_problem,
            "grail/gradient_checkpointing": cfg.gradient_checkpointing,
            "grail/temperature": cfg.temperature,
            "grail/top_p": cfg.top_p,
            "grail/top_k": cfg.top_k,
            "grail/adam_beta2": cfg.adam_beta2,
            "grail/num_iterations": args.num_iterations,
            "grail/seed": args.seed,
            "grail/dataset": args.dataset,
            "grail/eval_every": args.eval_every,
            "grail/fp32_master_weights": args.fp32_master_weights,
            # TRL GRPOConfig (for reference)
            **{f"trl/{k}": v for k, v in grpo_config.to_dict().items()},
        }

        wandb.init(
            project=wandb_project,
            name=grpo_config.run_name,
            config=wandb_config,
            tags=wandb_tags if wandb_tags else None,
        )
        logger.info(f"  ✓ WandB initialized (project: {wandb_project}, tags: {wandb_tags})")

    # Baseline evaluation
    vllm_eval_callback.run_and_log(step=0, label="BASELINE EVAL")

    # Train
    with profiler.track("training"):
        trainer.train()

    # Final evaluation
    final_step = trainer.state.global_step if hasattr(trainer, "state") else 9999
    final_metrics = vllm_eval_callback.run_and_log(step=final_step, label="FINAL EVAL")

    # Print profiling summary
    profiler.print_summary()
    profiler.log_to_wandb()

    # Print results summary
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL RESULTS SUMMARY ({adapter.name.upper()})")
    logger.info("=" * 60)
    for k in cfg.report_ks:
        if k > cfg.eval_replicates:
            continue
        logger.info(f"\nMetrics @ k={k}:")
        logger.info(f"  pass@{k}:        {final_metrics[f'pass@{k}']:.3f}")
        logger.info(f"  pass_ordered@{k}: {final_metrics[f'pass_ordered@{k}']:.3f}")
        logger.info(f"  mean@{k}:        {final_metrics[f'mean@{k}']:.3f}")
        logger.info(f"  best@{k}:        {final_metrics[f'best@{k}']:.3f}")
    logger.info("\nGlobal metrics:")
    logger.info(f"  reward_mean_all: {final_metrics['reward_mean_all']:.3f}")
    logger.info(f"  success_rate_all: {final_metrics['success_rate_all']:.3f}")

    # Cleanup execution pool
    if execution_pool is not None:
        try:
            logger.info("\n🧹 Shutting down code execution pool...")
            set_global_execution_pool(None)
            execution_pool.shutdown()
            logger.info("✅ Code execution pool shutdown complete")
        except Exception as e:
            logger.warning(f"⚠️  Error shutting down execution pool: {e}")


if __name__ == "__main__":
    main()
