#!/usr/bin/env python3
"""VeRL GRPO training script with factory pattern for GSM8K, MATH, and MBPP datasets.

This script provides equivalent functionality to train_trl_grpo.py but using the VeRL
framework instead of TRL.

Supports three datasets with exact parity to GRAIL environment implementations:
- GSM8K: Grade school math (7,473 train / 1,319 test)
- MATH: Hendrycks MATH benchmark (7,000 train / 500 val / 5,000 test)
- MBPP: Python code generation (374 train / 90 validation / 500 test)

Usage:
    # Prepare data first (generates parquet files)
    python train_verl_grpo.py --dataset gsm8k --prepare-data-only

    # Run training (uses Ray for distributed execution)
    python -m verl.trainer.main_ppo \\
        data.train_files=~/data/grail_gsm8k/train.parquet \\
        data.val_files=~/data/grail_gsm8k/test.parquet \\
        data.train_batch_size=512 \\
        actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \\
        actor_rollout_ref.rollout.n=16 \\
        algorithm.adv_estimator=grpo \\
        +custom_reward_function.path=/root/grail/research/verl/reward_functions.py \\
        +custom_reward_function.name=compute_score

    # Or use the all-in-one training mode
    python train_verl_grpo.py --dataset gsm8k
"""

from __future__ import annotations

import abc
import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Determine project root dynamically (research/verl/ -> project root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Load environment from .env for WandB
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)

sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# GRAIL imports - reuse task sources and validation logic
from grail.environments.execution import (
    CodeExecutionPool,
    check_code_executes,
    set_global_execution_pool,
)
from grail.environments.math_hendrycks_env import _math_answers_equal
from grail.environments.providers import (
    GSM8KTaskSource,
    MATHTaskSource,
    MBPPTaskSource,
)
# Note: Chat template is not applied here - VeRL applies it internally based on model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS (matching TRL GRPO config for exact parity)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """Training configuration matching GRAIL hyperparameters."""

    # Model Configuration
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Training Hyperparameters
    lr: float = 3e-6
    epochs: int = 1
    batch_size: int = 2
    gradient_checkpointing: bool = True
    grad_accum_steps: int = 256
    max_length: int = 4096
    grad_clip: float = 1.0
    warmup_steps: int = 20
    total_steps: int = 400

    # GRPO Loss Configuration
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    epsilon: float = 0.2  # PPO clip epsilon (lower bound for DAPO)
    epsilon_high: float = 0.28  # DAPO asymmetric upper bound
    grpo_variant: str = "dapo"  # Used in experiment naming

    # GRPO Data Configuration
    max_new_tokens: int = 2048  # Max response length
    rollouts_per_problem: int = 16  # Number of completions per prompt (key for GRPO)

    # Dataset Sampling (None = use full dataset)
    num_train_samples: int | None = None
    num_eval_samples: int | None = None

    # Generation Parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50

    # Data output directory
    data_dir: str = field(default_factory=lambda: os.path.expanduser("~/data"))


cfg = Config()


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & TAGS (shared across datasets)
# ════════════════════════════════════════════════════════════════════════════
REASONING_START_TOKEN = "start_working_out"
REASONING_END_TOKEN = "end_working_out"
SOLUTION_START_TOKEN = "SOLUTION"
SOLUTION_END_TOKEN = "SOLUTION"

REASONING_START = f"<{REASONING_START_TOKEN}>"
REASONING_END = f"</{REASONING_END_TOKEN}>"
SOLUTION_START = f"<{SOLUTION_START_TOKEN}>"
SOLUTION_END = f"</{SOLUTION_END_TOKEN}>"

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}."
)


# ════════════════════════════════════════════════════════════════════════════
# DATASET ADAPTER (Abstract Base + Concrete Implementations)
# ════════════════════════════════════════════════════════════════════════════
class DatasetAdapter(abc.ABC):
    """Abstract base class for dataset adapters."""

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
        """Reward threshold for success."""
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
        """Extract gold answer from dataset format."""
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
        """Get gold answer data from sample for evaluation."""
        return sample.get(self.answer_field, "")


# ────────────────────────────────────────────────────────────────────────────
# GSM8K Adapter
# ────────────────────────────────────────────────────────────────────────────
class GSM8KAdapter(DatasetAdapter):
    """GSM8K dataset adapter using GRAIL's GSM8KTaskSource."""

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
        return 0.6

    @property
    def success_threshold(self) -> float:
        return 0.6

    def load_train_data(self) -> list[dict[str, Any]]:
        self._train_source._ensure_dataset()
        assert self._train_source._ds is not None
        data = []
        for i in range(len(self._train_source._ds)):
            sample = self._train_source._ds[i]
            data.append({
                "question": sample["question"],
                "answer": sample["answer"],
            })
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        self._eval_source._ensure_dataset()
        assert self._eval_source._ds is not None
        data = []
        for i in range(len(self._eval_source._ds)):
            sample = self._eval_source._ds[i]
            data.append({
                "question": sample["question"],
                "answer": sample["answer"],
            })
        return data

    def parse_gold_answer(self, raw_answer: str) -> str:
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
        pred_norm = re.sub(r"[\s\.]+$", "", predicted.strip().lower())
        gold_norm = re.sub(r"[\s\.]+$", "", gold.strip().lower())
        return pred_norm == gold_norm

    def _parse_completion(self, text: str) -> dict[str, Any]:
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
        parsed = self._parse_completion(completion)
        gold_parsed = self.parse_gold_answer(gold_answer)

        correctness = 0.6 if self.validate_answer(parsed["answer_text"], gold_parsed) else 0.0
        strict_format = (
            0.15
            if (parsed["has_answer"] and parsed["is_numeric_only"] and parsed["trailing"] == 0)
            else 0.0
        )
        thinking = 0.1 if parsed["has_thinking"] else 0.0
        answer = 0.1 if parsed["has_answer"] else 0.0
        no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

        return correctness + strict_format + thinking + answer + no_trailing


# ────────────────────────────────────────────────────────────────────────────
# MATH (Hendrycks) Adapter
# ────────────────────────────────────────────────────────────────────────────
class MATHAdapter(DatasetAdapter):
    """MATH dataset adapter using GRAIL's MATHTaskSource."""

    def __init__(self) -> None:
        self._train_source = MATHTaskSource(split="train")
        self._eval_source = MATHTaskSource(split="val")

    @property
    def name(self) -> str:
        return "math"

    @property
    def question_field(self) -> str:
        return "question"

    @property
    def answer_field(self) -> str:
        return "answer"

    @property
    def correctness_weight(self) -> float:
        return 0.7

    @property
    def success_threshold(self) -> float:
        return 0.7

    def load_train_data(self) -> list[dict[str, Any]]:
        self._train_source._ensure_dataset()
        assert self._train_source._data is not None
        data = []
        for sample in self._train_source._data:
            data.append({
                "question": sample["problem"],
                "answer": sample["answer"],
                "solution": sample["solution"],
                "level": sample["level"],
                "subject": sample["subject"],
            })
        return data

    def load_eval_data(self) -> list[dict[str, Any]]:
        self._eval_source._ensure_dataset()
        assert self._eval_source._data is not None
        data = []
        for sample in self._eval_source._data:
            data.append({
                "question": sample["problem"],
                "answer": sample["answer"],
                "solution": sample["solution"],
                "level": sample["level"],
                "subject": sample["subject"],
            })
        return data

    def parse_gold_answer(self, raw_answer: str) -> str:
        return raw_answer

    def validate_answer(self, predicted: str, gold: str) -> bool:
        return _math_answers_equal(predicted, gold)

    def _parse_completion(self, text: str) -> dict[str, Any]:
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
        parsed = self._parse_completion(completion)

        correctness = 0.7 if self.validate_answer(parsed["answer_text"], gold_answer) else 0.0
        answer_format = 0.15 if (parsed["has_answer"] and parsed["trailing"] < 50) else 0.0
        thinking = 0.1 if parsed["has_thinking"] else 0.0
        no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

        return correctness + answer_format + thinking + no_trailing


# ────────────────────────────────────────────────────────────────────────────
# MBPP (Python Code) Adapter
# ────────────────────────────────────────────────────────────────────────────
class MBPPAdapter(DatasetAdapter):
    """MBPP dataset adapter for Python code generation."""

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
        return "test_list"

    def get_gold_data(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "test_list": sample.get("test_list", []),
            "test_setup_code": sample.get("test_setup_code", ""),
            "test_imports": sample.get("test_imports", []),
        }

    @property
    def correctness_weight(self) -> float:
        return 0.7

    @property
    def success_threshold(self) -> float:
        return 0.7

    def load_train_data(self) -> list[dict[str, Any]]:
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
        return raw_answer

    def validate_answer(self, predicted: str, test_data: dict[str, Any]) -> bool:
        if not predicted or not isinstance(test_data, dict):
            return False

        test_list = test_data.get("test_list", [])
        if not test_list:
            return False

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

        result = check_code_executes(predicted, test_cases, timeout=5.0)
        return result["status"] == "all_passed"

    def _parse_completion(self, text: str) -> dict[str, Any]:
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
        parsed = self._parse_completion(completion)

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

                result = check_code_executes(parsed["code"], test_cases, timeout=5.0)
                if result["total"] > 0:
                    pass_rate = result["passed"] / result["total"]
                    correctness = 0.7 * pass_rate

        syntax = 0.1 if parsed["syntax_valid"] else 0.0
        solution_format = 0.1 if (parsed["has_solution"] and parsed["trailing"] < 50) else 0.0
        thinking = 0.1 if parsed["has_thinking"] else 0.0

        return correctness + syntax + solution_format + thinking


# ════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def get_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    """Factory function to get dataset adapter by name."""
    adapters: dict[str, type[DatasetAdapter]] = {
        "gsm8k": GSM8KAdapter,
        "math": MATHAdapter,
        "mbpp": MBPPAdapter,
    }

    if dataset_name.lower() not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")

    return adapters[dataset_name.lower()]()


# ════════════════════════════════════════════════════════════════════════════
# VERL DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════
def prepare_verl_parquet(
    adapter: DatasetAdapter,
    output_dir: str,
) -> tuple[str, str]:
    """Prepare dataset in VeRL's required parquet format.

    VeRL requires parquet files with these fields:
    - data_source: Dataset identifier for reward function indexing
    - prompt: List of message dicts (VeRL applies chat template internally)
    - ability: Task category (e.g., "math", "code")
    - reward_model: Dict containing ground_truth and other metadata
    - extra_info: Additional metadata (split, index, etc.)

    Args:
        adapter: Dataset adapter instance
        output_dir: Directory to save parquet files

    Returns:
        Tuple of (train_path, val_path)
    """
    import json as json_module

    dataset_dir = os.path.join(output_dir, f"grail_{adapter.name}")
    os.makedirs(dataset_dir, exist_ok=True)

    def format_sample(sample: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
        """Format a single sample for VeRL."""
        question = sample[adapter.question_field]

        # VeRL expects prompt as list of message dicts (applies chat template internally)
        # Include system prompt in the messages
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Extract ground truth - handle different formats
        if adapter.name == "mbpp":
            # For MBPP, store test data as JSON string
            ground_truth = json_module.dumps({
                "test_list": sample.get("test_list", []),
                "test_setup_code": sample.get("test_setup_code", ""),
                "test_imports": sample.get("test_imports", []),
            })
        else:
            # For GSM8K/MATH, store answer string
            ground_truth = sample[adapter.answer_field]

        # Determine ability category
        ability = "code" if adapter.name == "mbpp" else "math"

        return {
            "data_source": f"grail_{adapter.name}",
            "prompt": prompt,
            "ability": ability,
            "reward_model": {
                "ground_truth": ground_truth,
                "style": "rule",  # Rule-based reward
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "dataset": adapter.name,
            },
        }

    # Process training data
    logger.info(f"Loading {adapter.name} training data...")
    train_data = adapter.load_train_data()
    if cfg.num_train_samples is not None:
        train_data = train_data[:cfg.num_train_samples]

    train_records = [
        format_sample(sample, "train", idx)
        for idx, sample in enumerate(train_data)
    ]

    # Process evaluation data
    logger.info(f"Loading {adapter.name} evaluation data...")
    eval_data = adapter.load_eval_data()
    if cfg.num_eval_samples is not None:
        eval_data = eval_data[:cfg.num_eval_samples]

    eval_records = [
        format_sample(sample, "eval", idx)
        for idx, sample in enumerate(eval_data)
    ]

    # Convert to parquet
    train_path = os.path.join(dataset_dir, "train.parquet")
    val_path = os.path.join(dataset_dir, "test.parquet")

    train_df = pd.DataFrame(train_records)
    eval_df = pd.DataFrame(eval_records)

    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(val_path, index=False)

    logger.info(f"Saved {len(train_records)} training samples to {train_path}")
    logger.info(f"Saved {len(eval_records)} evaluation samples to {val_path}")

    return train_path, val_path


# ════════════════════════════════════════════════════════════════════════════
# VERL CONFIG GENERATION
# ════════════════════════════════════════════════════════════════════════════
def generate_verl_config(
    adapter: DatasetAdapter,
    train_path: str,
    val_path: str,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate VeRL configuration for GRPO training.

    This creates a configuration dict that can be used with VeRL's main_ppo
    trainer, matching the hyperparameters from the TRL GRPO script.

    Args:
        adapter: Dataset adapter instance
        train_path: Path to training parquet file
        val_path: Path to validation parquet file
        output_path: Optional path to save config as YAML

    Returns:
        Configuration dictionary
    """
    # Calculate derived parameters
    max_prompt_length = cfg.max_length - cfg.max_new_tokens
    effective_batch = cfg.batch_size * cfg.grad_accum_steps

    config = {
        # ─────────────────────────────────────────────────────────────────────
        # Data Configuration
        # ─────────────────────────────────────────────────────────────────────
        "data": {
            "train_files": train_path,
            "val_files": val_path,
            "train_batch_size": effective_batch,  # Global batch size
            "max_prompt_length": max_prompt_length,
            "max_response_length": cfg.max_new_tokens,
            "shuffle": True,
            "seed": 42,
            "filter_overlong_prompts": True,
            "truncation": "left",
        },

        # ─────────────────────────────────────────────────────────────────────
        # Actor-Rollout-Reference Model Configuration
        # ─────────────────────────────────────────────────────────────────────
        "actor_rollout_ref": {
            "model": {
                "path": cfg.model_id,
                "enable_gradient_checkpointing": cfg.gradient_checkpointing,
                "trust_remote_code": True,
            },
            "actor": {
                # PPO/GRPO mini-batch configuration
                "ppo_mini_batch_size": cfg.batch_size * cfg.grad_accum_steps,
                "ppo_micro_batch_size_per_gpu": cfg.batch_size,
                "use_dynamic_bsz": True,  # Dynamic batching for variable-length sequences
                "grad_clip": cfg.grad_clip,
                # DAPO-style asymmetric clipping
                "clip_ratio_low": cfg.epsilon,  # Lower clip bound
                "clip_ratio_high": cfg.epsilon_high,  # Upper clip bound (DAPO)
                "entropy_coeff": cfg.entropy_coef,
                # KL configuration for GRPO
                "use_kl_loss": cfg.kl_coef > 0,
                "kl_loss_coef": cfg.kl_coef,
                "kl_loss_type": "low_var_kl",
                "ppo_epochs": cfg.epochs,
                "shuffle": True,
                # Loss aggregation mode (DAPO uses token-mean)
                "loss_agg_mode": "token-mean",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": cfg.batch_size * 2,
            },
            "rollout": {
                "name": "vllm",  # Use vLLM for generation
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "response_length": cfg.max_new_tokens,
                "gpu_memory_utilization": 0.6,  # Balanced for training + inference
                "tensor_model_parallel_size": 1,
                "do_sample": True,
                "n": cfg.rollouts_per_problem,  # GRPO: multiple completions per prompt
                "free_cache_engine": True,  # Free KV cache after generation (saves memory)
            },
            "optim": {
                "lr": cfg.lr,
                "lr_warmup_steps": cfg.warmup_steps,
                "lr_scheduler_type": "constant",
                "min_lr_ratio": 0.0,
            },
            "fsdp": {
                "wrap_policy": "transformer_auto_wrap_policy",
                "param_offload": False,
                "optimizer_offload": False,
            },
        },

        # ─────────────────────────────────────────────────────────────────────
        # Algorithm Configuration (GRPO)
        # ─────────────────────────────────────────────────────────────────────
        "algorithm": {
            "gamma": 1.0,  # Discount factor
            "lam": 1.0,  # GAE lambda
            "adv_estimator": "grpo",  # Use GRPO advantage estimator
            "use_kl_in_reward": False,
            "kl_ctrl": {
                "type": "fixed",
                "kl_coef": cfg.kl_coef,
            },
        },

        # ─────────────────────────────────────────────────────────────────────
        # Reward Model Configuration
        # ─────────────────────────────────────────────────────────────────────
        "reward_model": {
            "enable": False,  # Use function-based reward instead
        },

        # ─────────────────────────────────────────────────────────────────────
        # Custom Reward Function
        # ─────────────────────────────────────────────────────────────────────
        "custom_reward_function": {
            "path": os.path.join(_SCRIPT_DIR, "reward_functions.py"),
            "name": "compute_score",
        },

        # ─────────────────────────────────────────────────────────────────────
        # Trainer Configuration
        # ─────────────────────────────────────────────────────────────────────
        "trainer": {
            "total_epochs": cfg.total_steps,  # Total training iterations
            "project_name": "grail-verl",
            "experiment_name": f"{adapter.name}_grpo_{cfg.grpo_variant}",
            "logger": ["console", "wandb"],
            "log_val_generations": 5,
            "nnodes": 1,
            "n_gpus_per_node": 1,
            "save_freq": 50,
            "test_freq": 40,
            "val_before_train": True,
        },
    }

    # Save config to YAML if path provided
    if output_path:
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved VeRL config to {output_path}")

    return config


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VeRL GRPO training with GSM8K, MATH, or MBPP dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to use for training (default: gsm8k)",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Only prepare data (generate parquet files) without training",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate VeRL YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to use (overrides default Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory for data files (default: ~/data)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Run VeRL training after data preparation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config via CLI
    if args.model:
        cfg.model_id = args.model
    if args.data_dir:
        cfg.data_dir = args.data_dir

    logger.info(f"VeRL GRPO Training Setup for {args.dataset.upper()}")
    logger.info("=" * 80)

    # Get dataset adapter
    adapter = get_dataset_adapter(args.dataset)
    logger.info(f"Dataset: {adapter.name}")
    logger.info(f"Model: {cfg.model_id}")

    # Initialize code execution pool for MBPP
    execution_pool: CodeExecutionPool | None = None
    if args.dataset == "mbpp":
        try:
            execution_pool = CodeExecutionPool(num_workers=8, max_tasks_per_child=50)
            execution_pool.start()
            set_global_execution_pool(execution_pool)
            logger.info("Code execution pool initialized (8 workers)")
        except Exception as e:
            logger.warning(f"Failed to init execution pool: {e}")

    # Step 1: Prepare data in parquet format
    logger.info("\nStep 1: Preparing data in VeRL parquet format...")
    train_path, val_path = prepare_verl_parquet(
        adapter=adapter,
        output_dir=cfg.data_dir,
    )

    # Step 2: Verify reward functions module exists
    # Note: reward_functions.py is maintained as a standalone file, not generated
    reward_module_path = os.path.join(_SCRIPT_DIR, "reward_functions.py")
    if not os.path.exists(reward_module_path):
        logger.error(f"Reward functions module not found: {reward_module_path}")
        logger.error("Please ensure reward_functions.py exists in the verl directory.")
        raise FileNotFoundError(reward_module_path)
    logger.info(f"\nStep 2: Using reward functions module: {reward_module_path}")

    # Step 3: Generate config if requested
    config_path = os.path.join(_SCRIPT_DIR, f"config_{adapter.name}.yaml")
    if args.generate_config or not args.prepare_data_only:
        logger.info("\nStep 3: Generating VeRL configuration...")
        # Config is written to config_path, return value not needed
        _ = generate_verl_config(
            adapter=adapter,
            train_path=train_path,
            val_path=val_path,
            output_path=config_path,
        )

    if args.prepare_data_only:
        logger.info("\n" + "=" * 80)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nTrain data: {train_path}")
        logger.info(f"Val data: {val_path}")
        logger.info(f"Reward module: {reward_module_path}")
        logger.info("\nTo run training with VeRL, use:")
        logger.info(f"""
python -m verl.trainer.main_ppo \\
    data.train_files={train_path} \\
    data.val_files={val_path} \\
    data.train_batch_size=512 \\
    data.max_prompt_length={cfg.max_length - cfg.max_new_tokens} \\
    data.max_response_length={cfg.max_new_tokens} \\
    actor_rollout_ref.model.path={cfg.model_id} \\
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \\
    actor_rollout_ref.actor.clip_ratio_low={cfg.epsilon} \\
    actor_rollout_ref.actor.clip_ratio_high={cfg.epsilon_high} \\
    actor_rollout_ref.actor.grad_clip={cfg.grad_clip} \\
    actor_rollout_ref.rollout.n={cfg.rollouts_per_problem} \\
    actor_rollout_ref.rollout.temperature={cfg.temperature} \\
    actor_rollout_ref.optim.lr={cfg.lr} \\
    actor_rollout_ref.optim.lr_warmup_steps={cfg.warmup_steps} \\
    algorithm.adv_estimator=grpo \\
    algorithm.kl_ctrl.kl_coef={cfg.kl_coef} \\
    trainer.total_epochs={cfg.total_steps} \\
    trainer.project_name=grail-verl \\
    trainer.experiment_name={adapter.name}_grpo \\
    +custom_reward_function.path={reward_module_path} \\
    +custom_reward_function.name=compute_score
""")
    elif args.run_training:
        logger.info("\nStep 4: Starting VeRL training...")

        # Run VeRL training via subprocess
        import subprocess

        cmd = [
            "python", "-m", "verl.trainer.main_ppo",
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            f"data.train_batch_size={cfg.batch_size * cfg.grad_accum_steps}",
            f"data.max_prompt_length={cfg.max_length - cfg.max_new_tokens}",
            f"data.max_response_length={cfg.max_new_tokens}",
            f"actor_rollout_ref.model.path={cfg.model_id}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.batch_size * cfg.grad_accum_steps}",
            f"actor_rollout_ref.actor.clip_ratio_low={cfg.epsilon}",
            f"actor_rollout_ref.actor.clip_ratio_high={cfg.epsilon_high}",
            f"actor_rollout_ref.actor.grad_clip={cfg.grad_clip}",
            f"actor_rollout_ref.actor.entropy_coeff={cfg.entropy_coef}",
            f"actor_rollout_ref.rollout.n={cfg.rollouts_per_problem}",
            f"actor_rollout_ref.rollout.temperature={cfg.temperature}",
            f"actor_rollout_ref.rollout.top_p={cfg.top_p}",
            f"actor_rollout_ref.rollout.top_k={cfg.top_k}",
            f"actor_rollout_ref.optim.lr={cfg.lr}",
            f"actor_rollout_ref.optim.lr_warmup_steps={cfg.warmup_steps}",
            "algorithm.adv_estimator=grpo",
            f"algorithm.kl_ctrl.kl_coef={cfg.kl_coef}",
            f"trainer.total_epochs={cfg.total_steps}",
            "trainer.project_name=grail-verl",
            f"trainer.experiment_name={adapter.name}_grpo_{cfg.grpo_variant}",
            f"+custom_reward_function.path={reward_module_path}",
            "+custom_reward_function.name=compute_score",
        ]

        logger.info("Running command:")
        logger.info(" ".join(cmd))

        subprocess.run(cmd, check=True)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("SETUP COMPLETE")
        logger.info("=" * 80)
        logger.info("\nData and config files have been generated.")
        logger.info("Use --run-training to start training, or run manually with:")
        logger.info(f"  python -m verl.trainer.main_ppo --config-path {config_path}")

    # Cleanup
    if execution_pool is not None:
        try:
            set_global_execution_pool(None)
            execution_pool.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
