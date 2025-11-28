#!/usr/bin/env python3
"""TRL GRPO training script with factory pattern for GSM8K and MATH datasets.

Supports both datasets with exact parity to GRAIL environment implementations:
- GSM8K: Grade school math (7,473 train / 1,319 test)
- MATH: Hendrycks MATH benchmark (7,000 train / 500 val / 5,000 test)

Usage:
    python train_trl_grpo.py --dataset gsm8k
    python train_trl_grpo.py --dataset math
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
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
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# Load environment from .env for WandB
load_dotenv("/root/grail/.env")

sys.path.append("/root/grail")

# GRAIL imports - reuse task sources and validation logic (after sys.path.append)
from grail.environments.math_hendrycks_env import _math_answers_equal  # noqa: E402
from grail.environments.providers import GSM8KTaskSource, MATHTaskSource  # noqa: E402
from grail.shared.chat_templates import build_qwen_chat_template  # noqa: E402
from grail.trainer.metrics import KMetricsAggregator, TaskReplicateResult  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS (from .env GRAIL config)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # Model (from GRAIL_TRAIN_MODEL_ID)
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Learning rate (from GRAIL_TRAINER_LR)
    lr: float = 3e-6
    # Epochs per window (from GRAIL_TRAINER_EPOCHS)
    epochs: int = 1
    # Batch size (from GRAIL_TRAINER_BATCH_SIZE)
    batch_size: int = 4
    # Gradient accumulation (from GRAIL_TRAINER_GRAD_ACCUM_STEPS)
    grad_accum_steps: int = 128
    # Max sequence length (from GRAIL_TRAINER_MAX_LENGTH)
    max_length: int = 2048
    # Gradient clipping (from GRAIL_TRAINER_GRAD_CLIP)
    grad_clip: float = 1.0
    # Warmup steps (from GRAIL_TRAINER_WARMUP_STEPS)
    warmup_steps: int = 50
    # KL coefficient (from GRAIL_TRAINER_KL_COEF)
    kl_coef: float = 0.0
    # Entropy coefficient (from GRAIL_TRAINER_ENTROPY_COEF)
    entropy_coef: float = 0.0005
    # PPO clip epsilon (standard GRAIL values)
    ppo_clip_eps: float = 0.2
    ppo_clip_eps_upper: float = 0.28
    # Importance sampling ratio max (from GRAIL_TRAINER_IS_RATIO_MAX)
    is_ratio_max: float = 2.5
    # Log-ratio clamp (from GRAIL_TRAINER_LOGRATIO_CLAMP)
    logratio_clamp: float = 0.92
    # Dataset sampling
    num_train_samples: int | None = None  # None = use all training samples
    num_eval_samples: int | None = None  # None = use all test samples
    # Rollouts per problem (matches GRAIL default)
    rollouts_per_problem: int = 16
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    # Max completion tokens (from GRPO_MAX_COMPLETION_TOKENS)
    max_new_tokens: int = 1024
    # Evaluation config
    eval_replicates: int = 5
    report_ks: tuple[int, ...] = (1, 5, 10)
    # Evaluation optimization
    eval_batch_size: int = 128
    eval_num_workers: int = 4
    # Max groups for GRPO (from GRPO_MAX_GROUPS)
    max_groups: int = 128


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

QWEN_CHAT_TEMPLATE = build_qwen_chat_template(
    system_prompt=SYSTEM_PROMPT, reasoning_start=REASONING_START
)


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
    def parse_gold_answer(self, raw_answer: str) -> str:
        """Extract gold answer from dataset format."""
        ...

    @abc.abstractmethod
    def validate_answer(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold."""
        ...

    @abc.abstractmethod
    def compute_reward(self, completion: str, gold_answer: str) -> float:
        """Compute total reward for completion."""
        ...


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


# ════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def get_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    """Factory function to get dataset adapter by name.

    Args:
        dataset_name: 'gsm8k' or 'math'

    Returns:
        DatasetAdapter instance

    Raises:
        ValueError: If dataset_name is not supported
    """
    adapters: dict[str, type[DatasetAdapter]] = {
        "gsm8k": GSM8KAdapter,
        "math": MATHAdapter,
    }

    if dataset_name.lower() not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")

    return adapters[dataset_name.lower()]()


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════
def prepare_train_dataset(adapter: DatasetAdapter, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load and format training dataset for TRL GRPO.

    Args:
        adapter: Dataset adapter instance
        tokenizer: Tokenizer for chat template formatting

    Returns:
        HuggingFace Dataset with 'prompt' and 'gold_answer' columns
    """
    raw_data = adapter.load_train_data()

    if cfg.num_train_samples is not None:
        raw_data = raw_data[: cfg.num_train_samples]

    formatted = []
    for sample in raw_data:
        question = sample[adapter.question_field]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(
            {
                "prompt": prompt,
                "gold_answer": sample[adapter.answer_field],
            }
        )

    print(f"  Training dataset ({adapter.name}): {len(formatted)} samples")
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

    print(f"  Eval dataset ({adapter.name}): {len(raw_data)} samples")
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
        eval_every_n_steps: int = 30,
    ) -> None:
        self.adapter = adapter
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.eval_every_n = eval_every_n_steps
        self.base_url = vllm_base_url.rstrip("/")
        self._metrics_defined = False

        print(
            f"✓ VLLMEvalCallback initialized: dataset={adapter.name}, "
            f"url={vllm_base_url}, eval_every={eval_every_n_steps}"
        )

    def run_and_log(self, step: int, label: str = "VLLM EVAL") -> dict[str, float]:
        """Run evaluation and log to WandB."""
        print(f"\n{'=' * 80}")
        print(f"[{label}] Step {step}: Starting {self.adapter.name.upper()} evaluation...")
        print(f"{'=' * 80}")

        metrics = asyncio.run(self._run_eval())

        try:
            import wandb

            if wandb.run is not None:
                if not self._metrics_defined:
                    wandb.define_metric("eval_step")
                    wandb.define_metric("eval_vllm/*", step_metric="eval_step")
                    self._metrics_defined = True

                wandb_data = {
                    "eval_step": step,
                    "trainer/global_step": step,
                }
                wandb_data.update({f"eval_vllm/{k}": v for k, v in metrics.items()})
                wandb.log(wandb_data)
        except Exception as e:
            print(f"⚠️  WandB logging failed: {e}")

        print(f"[{label}] Results: {metrics}")
        print(f"{'=' * 80}\n")
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
                batch_golds = [s[self.adapter.answer_field] for s in batch]

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
                    print("\n  ━━━ Sample Completions ━━━")
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
                        print(f"\n  Sample {i + 1}:")
                        print(f"    Question: {q_display}")
                        print(f"    Completion: {c_display}")
                        print(f"    Reward: {reward:.3f} | Gold: {gold[:50]}...")
                    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━\n")

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

        print(
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
        print(f"    Generating {total} completions via {num_requests} batched requests")

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
                        print(f"  ⚠️  Batch {start_idx} failed: {type(e).__name__}")
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
# MAIN TRAINING
# ════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TRL GRPO training with GSM8K or MATH dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math"],
        help="Dataset to use for training (default: gsm8k)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=30,
        help="Run evaluation every N steps (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"🚀 Starting TRL GRPO training with {args.dataset.upper()} dataset")
    print("=" * 60)

    # Get dataset adapter
    adapter = get_dataset_adapter(args.dataset)
    print(f"  Dataset: {adapter.name}")
    print(f"  Correctness weight: {adapter.correctness_weight}")
    print(f"  Success threshold: {adapter.success_threshold}")

    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, RuntimeError) as e:
        print(f"⚠️  Flash Attention 2 unavailable ({type(e).__name__}), using default")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    # Prepare datasets
    print("\n📊 Preparing datasets...")
    train_ds = prepare_train_dataset(adapter, tokenizer)
    eval_ds, eval_data = prepare_eval_dataset(adapter)
    prompt_to_answer = {row["prompt"]: row["gold_answer"] for row in train_ds}

    # WandB setup
    print("\n⚙️  Configuring GRPO trainer...")
    import wandb

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print(f"  ✓ WandB logged in (project: {os.getenv('WANDB_PROJECT', 'grail')})")

    # Calculate max_prompt_length
    max_prompt_length = cfg.max_length - cfg.max_new_tokens

    grpo_config = GRPOConfig(
        output_dir=f"./outputs/trl_{adapter.name}",
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_grad_norm=cfg.grad_clip,
        warmup_steps=cfg.warmup_steps,
        beta=cfg.kl_coef,
        epsilon=cfg.ppo_clip_eps,
        epsilon_high=cfg.ppo_clip_eps_upper,
        max_prompt_length=max_prompt_length,
        max_completion_length=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=1.1,
        num_generations=cfg.rollouts_per_problem,
        generation_batch_size=16,
        steps_per_generation=None,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        wandb_log_unique_prompts=True,
        save_strategy="no",
        bf16=True,
        report_to=["wandb"],
        eval_strategy="no",
        run_name=f"trl_{adapter.name}_grpo_qwen15b_env_matched",
        loss_type="dapo",
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://127.0.0.1:8000",
        vllm_importance_sampling_correction=False,
        vllm_importance_sampling_cap=cfg.is_ratio_max,
    )

    # Create reward function using adapter
    def reward_fn(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
        if "gold_answer" in kwargs and kwargs["gold_answer"]:
            golds = kwargs["gold_answer"]
            return [adapter.compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]
        if "metadatas" in kwargs and kwargs["metadatas"]:
            golds = [m.get("gold_answer", "") for m in kwargs["metadatas"]]
            return [adapter.compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]
        golds = [prompt_to_answer.get(p, "") for p in prompts]
        return [adapter.compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]

    print(f"\n🏋️  Training with GRPO on {adapter.name.upper()}...")

    # Initialize evaluation callback
    vllm_eval_callback = VLLMEvalCallback(
        adapter=adapter,
        eval_data=eval_data,
        tokenizer=tokenizer,
        vllm_base_url=grpo_config.vllm_server_base_url,
        eval_every_n_steps=args.eval_every,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[vllm_eval_callback],
    )

    # Baseline evaluation
    vllm_eval_callback.run_and_log(step=0, label="BASELINE EVAL")

    # Train
    trainer.train()

    # Final evaluation
    final_step = trainer.state.global_step if hasattr(trainer, "state") else 9999
    final_metrics = vllm_eval_callback.run_and_log(step=final_step, label="FINAL EVAL")

    # Print summary
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS SUMMARY ({adapter.name.upper()})")
    print("=" * 60)
    for k in cfg.report_ks:
        if k > cfg.eval_replicates:
            continue
        print(f"\nMetrics @ k={k}:")
        print(f"  pass@{k}:        {final_metrics[f'pass@{k}']:.3f}")
        print(f"  pass_ordered@{k}: {final_metrics[f'pass_ordered@{k}']:.3f}")
        print(f"  mean@{k}:        {final_metrics[f'mean@{k}']:.3f}")
        print(f"  best@{k}:        {final_metrics[f'best@{k}']:.3f}")
    print("\nGlobal metrics:")
    print(f"  reward_mean_all: {final_metrics['reward_mean_all']:.3f}")
    print(f"  success_rate_all: {final_metrics['success_rate_all']:.3f}")


if __name__ == "__main__":
    main()
