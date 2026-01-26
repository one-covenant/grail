#!/usr/bin/env python3
"""TRL SFT (Supervised Fine-Tuning) training script.

Supports three datasets with complete conversation formatting:
- GSM8K: Grade school math (formatted with final answer)
- MATH: Hendrycks MATH (formatted with full solution + answer)
- MBPP: Python code generation (formatted with reference solution)

Shares analysis infrastructure with train_trl_grpo.py via callbacks module.

Usage:
    python train_trl_sft.py --dataset math
    python train_trl_sft.py --dataset gsm8k --lr 1e-5
    python train_trl_sft.py --dataset mbpp --max-steps 200
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# Force unbuffered output for better logging in nohup mode
try:
    if hasattr(sys.stdout, "fileno") and sys.stdout.fileno() >= 0:
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "fileno") and sys.stderr.fileno() >= 0:
        sys.stderr.reconfigure(line_buffering=True)
except (OSError, AttributeError, ValueError):
    pass

# Determine project root dynamically (research/trl/ -> project root)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Load environment from .env for WandB
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)

sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# GRAIL imports (after sys.path modification)
from grail.trainer.analysis import (  # noqa: E402
    AnalysisConfig,
    GradientSparsityMetrics,
    ModelAnalysisManager,
)

# Import from train_trl_grpo.py for shared utilities
from train_trl_grpo import (  # noqa: E402
    QWEN_CHAT_TEMPLATE,
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    SYSTEM_PROMPT,
    DatasetAdapter,
    get_dataset_adapter,
    get_gradient_checkpointing_kwargs,
    prepare_eval_dataset,
    print_memory_estimate,
)

# Import shared callbacks
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
    force=True,
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SFT CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
def _get_sft_lr_from_env() -> float:
    """Get learning rate from environment, with SFT default fallback."""
    return float(os.environ.get("GRAIL_SFT_LR", "2e-5"))


@dataclass
class SFTHyperConfig:
    """SFT-specific hyperparameters (different from GRPO)."""

    # Model
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Learning rate (higher than GRPO since no policy gradient variance)
    lr: float = field(default_factory=_get_sft_lr_from_env)

    # Batch size (can be larger than GRPO - no generation overhead)
    batch_size: int = 4
    grad_accum_steps: int = 32  # Effective batch = 128

    # Training schedule
    max_steps: int = 400
    warmup_steps: int = 20  # ~5% of training

    # Sequence length (shorter than GRPO - no long rollouts)
    max_length: int = 2048

    # Evaluation
    num_eval_samples: int = 50  # Limit eval samples for faster baseline eval

    # Gradient settings
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True

    # Delta checkpoint config
    delta_checkpoint_enabled: bool = True
    delta_checkpoint_dtype: str = "bfloat16"


cfg = SFTHyperConfig()


# ════════════════════════════════════════════════════════════════════════════
# DATA FORMATTING (Complete Conversations)
# ════════════════════════════════════════════════════════════════════════════
def format_sft_sample_math(sample: dict[str, Any]) -> str:
    """Format MATH sample as complete conversation with full solution.

    MATH has the richest data - full reasoning in 'solution' field.
    Format: User: {problem} | Assistant: <reasoning>{solution}</reasoning><SOLUTION>{answer}</SOLUTION>
    """
    solution = sample.get("solution", "")
    answer = sample.get("answer", "")

    # Build assistant response with reasoning and answer
    assistant_content = f"{REASONING_START}{solution}{REASONING_END}{SOLUTION_START}{answer}{SOLUTION_END}"

    return assistant_content


def format_sft_sample_gsm8k(sample: dict[str, Any]) -> str:
    """Format GSM8K sample as complete conversation.

    GSM8K has 'answer' field with "#### <final_answer>" format.
    Extract the final numeric answer.
    """
    raw_answer = sample.get("answer", "")

    # Parse the answer - GSM8K format is "reasoning #### final_answer"
    hash_pattern = re.compile(r"####\s*(?P<ans>.+)")
    match = None
    for m in hash_pattern.finditer(raw_answer):
        match = m
    if match:
        final_answer = match.group("ans").strip()
        # The reasoning is everything before ####
        reasoning = raw_answer[:match.start()].strip()
    else:
        final_answer = raw_answer.strip()
        reasoning = f"The answer is {final_answer}."

    assistant_content = f"{REASONING_START}{reasoning}{REASONING_END}{SOLUTION_START}{final_answer}{SOLUTION_END}"

    return assistant_content


def format_sft_sample_mbpp(sample: dict[str, Any]) -> str:
    """Format MBPP sample as complete conversation with reference code.

    MBPP has 'reference_solution' (or 'code') field with the solution.
    """
    code = sample.get("reference_solution", sample.get("code", ""))

    # Brief reasoning + code solution
    reasoning = "Here's the implementation:"
    assistant_content = f"{REASONING_START}{reasoning}{REASONING_END}{SOLUTION_START}{code}{SOLUTION_END}"

    return assistant_content


def get_format_function(dataset_name: str):
    """Get the appropriate formatting function for a dataset."""
    formatters = {
        "math": format_sft_sample_math,
        "gsm8k": format_sft_sample_gsm8k,
        "mbpp": format_sft_sample_mbpp,
    }
    return formatters[dataset_name.lower()]


def prepare_sft_dataset(
    adapter: DatasetAdapter,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """Prepare SFT dataset with complete conversations.

    Args:
        adapter: Dataset adapter instance
        tokenizer: Tokenizer for chat template formatting

    Returns:
        HuggingFace Dataset with 'text' column for SFTTrainer
    """
    raw_data = adapter.load_train_data()
    format_fn = get_format_function(adapter.name)

    formatted = []
    for sample in raw_data:
        question = sample[adapter.question_field]
        assistant_response = format_fn(sample)

        # Format as complete conversation
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        formatted.append({"text": text})

    logger.info(f"  SFT training dataset ({adapter.name}): {len(formatted)} samples")
    return Dataset.from_list(formatted)


# ════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED EVALUATION CALLBACK (Greedy Decoding)
# ════════════════════════════════════════════════════════════════════════════
class SFTEvalCallback(TrainerCallback):
    """Simple evaluation callback using HF generate (no vLLM needed).

    Uses greedy decoding (temperature=0) with single completion per task.
    Reports accuracy metric.
    """

    def __init__(
        self,
        adapter: DatasetAdapter,
        eval_data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        eval_every_n_steps: int = 40,
        max_new_tokens: int = 1024,
    ) -> None:
        self.adapter = adapter
        self.eval_data = eval_data
        self.tokenizer = tokenizer
        self.eval_every_n = eval_every_n_steps
        self.max_new_tokens = max_new_tokens
        self._wandb_configured = False

        logger.info(
            f"SFTEvalCallback initialized: dataset={adapter.name}, "
            f"eval_every={eval_every_n_steps}, max_new_tokens={max_new_tokens}"
        )

    def run_and_log(
        self,
        model: Any,
        step: int,
        label: str = "SFT EVAL",
    ) -> dict[str, float]:
        """Run evaluation and log to WandB."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"[{label}] Step {step}: Starting {self.adapter.name.upper()} evaluation...")
        logger.info(f"{'=' * 80}")

        profiler = get_profiler()
        with profiler.track("evaluation"):
            metrics = self._run_eval(model)

        try:
            if wandb.run is not None:
                if not self._wandb_configured:
                    wandb.define_metric("eval_step")
                    wandb.define_metric("eval/*", step_metric="eval_step")
                    self._wandb_configured = True

                wandb_data = {"eval_step": step}
                wandb_data.update({f"eval/{k}": v for k, v in metrics.items()})
                wandb.log(wandb_data)
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

        logger.info(f"[{label}] Results: {metrics}")
        logger.info(f"{'=' * 80}\n")
        return metrics

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Run evaluation every N steps."""
        if state.global_step >= self.eval_every_n and state.global_step % self.eval_every_n == 0:
            model = kwargs.get("model")
            if model is not None:
                self.run_and_log(model, state.global_step)

    def _run_eval(self, model: Any) -> dict[str, float]:
        """Run evaluation with greedy decoding."""
        from tqdm import tqdm

        model.eval()
        device = next(model.parameters()).device

        correct = 0
        total = 0
        total_reward = 0.0

        with torch.no_grad():
            for sample in tqdm(self.eval_data, desc=f"Eval ({self.adapter.name})", unit="task"):
                question = sample[self.adapter.question_field]
                gold = self.adapter.get_gold_data(sample)

                # Format prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_length - self.max_new_tokens,
                ).to(device)

                # Generate with greedy decoding
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Greedy
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode completion (exclude prompt tokens)
                completion = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                # Compute reward and check success
                reward = self.adapter.compute_reward(completion, gold)
                success = reward >= self.adapter.success_threshold

                total_reward += reward
                if success:
                    correct += 1
                total += 1

                # Log first few samples
                if total <= 3:
                    q_display = question[:100] + "..." if len(question) > 100 else question
                    c_display = completion[:200] + "..." if len(completion) > 200 else completion
                    logger.info(f"\n  Sample {total}:")
                    logger.info(f"    Question: {q_display}")
                    logger.info(f"    Completion: {c_display}")
                    logger.info(f"    Reward: {reward:.3f} | Success: {success}")

        model.train()

        accuracy = correct / total if total > 0 else 0.0
        mean_reward = total_reward / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "mean_reward": mean_reward,
            "correct": correct,
            "total": total,
        }


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TRL SFT training with GSM8K, MATH, or MBPP dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (default: 400)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=40,
        help="Run evaluation every N steps (default: 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per device (default: 4)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Suffix for run name and output directories",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device (e.g., 'cuda:0')",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override config via CLI
    if args.model:
        cfg.model_id = args.model
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f"Starting TRL SFT training with {args.dataset.upper()} dataset")
    logger.info(f"   Seed: {args.seed}")
    logger.info(f"   Model: {cfg.model_id}")
    logger.info(f"   Learning rate: {cfg.lr}")
    logger.info(f"   Batch size: {cfg.batch_size} | Grad accum: {cfg.grad_accum_steps} | Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    logger.info(f"   Max steps: {cfg.max_steps}")
    logger.info(f"   Gradient checkpointing: {cfg.gradient_checkpointing}")
    if args.run_suffix:
        logger.info(f"   Run suffix: {args.run_suffix}")
    logger.info("=" * 80)

    # Print memory estimation (use SFT config instead of GRPO defaults)
    print_memory_estimate(
        cfg.model_id,
        batch_size=cfg.batch_size,
        seq_len=cfg.max_length,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # Get dataset adapter
    adapter = get_dataset_adapter(args.dataset)
    logger.info(f"\nDataset: {adapter.name}")
    logger.info(f"  Correctness weight: {adapter.correctness_weight}")
    logger.info(f"  Success threshold: {adapter.success_threshold}")

    # Initialize profiler
    profiler = get_profiler()

    # Determine device
    if args.device:
        train_device = torch.device(args.device)
        if train_device.type == "cuda":
            torch.cuda.set_device(train_device)
        logger.info(f"  Using device: {train_device}")
    else:
        train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    logger.info("\nLoading model and tokenizer...")
    with profiler.track("model_loading"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=train_device,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Flash Attention 2 unavailable ({type(e).__name__}), using SDPA")
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=train_device,
            )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # SFT uses right padding
        tokenizer.chat_template = QWEN_CHAT_TEMPLATE

    # Prepare datasets
    logger.info("\nPreparing datasets...")
    with profiler.track("dataset_preparation"):
        train_ds = prepare_sft_dataset(adapter, tokenizer)
        _, eval_data = prepare_eval_dataset(adapter)
        # Limit eval samples for faster baseline evaluation
        if cfg.num_eval_samples is not None:
            eval_data = eval_data[: cfg.num_eval_samples]
            logger.info(f"  Limited eval to {len(eval_data)} samples")

    # WandB setup
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        effective_project = args.wandb_project or os.getenv("WANDB_PROJECT", "grail")
        logger.info(f"  WandB logged in (project: {effective_project})")

    # Build run identifiers
    run_id = f"seed{args.seed}" if not args.run_suffix else args.run_suffix
    output_base = os.getenv("GRAIL_OUTPUT_BASE", ".")

    # Configure SFTTrainer
    sft_config = SFTConfig(
        output_dir=f"{output_base}/outputs/trl_{adapter.name}_sft_{run_id}",
        # Learning rate & schedule
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="constant_with_warmup",
        max_steps=cfg.max_steps,
        # Batch size
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_grad_norm=cfg.grad_clip,
        # Sequence length
        max_length=cfg.max_length,
        # Optimizer
        optim="adamw_torch",
        weight_decay=0.0,
        # Logging & checkpointing
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        # Memory optimization
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=get_gradient_checkpointing_kwargs(cfg.model_id),
        report_to=["wandb"],
        eval_strategy="no",
        run_name=f"trl_{adapter.name}_sft_{cfg.model_id.replace('/', '_')}_{run_id}",
        # Dataset config
        dataset_text_field="text",
    )

    # Initialize sparsity analysis
    sparsity_config = AnalysisConfig(
        interval=1,
        param_change_enabled=True,
        param_change_thresholds=[0.0],
        sparse_quality_enabled=False,
        snapshot_dtype="bfloat16",
        gradient_enabled=True,
    )
    sparsity_analyzer = ModelAnalysisManager.create(sparsity_config)

    gradient_sparsity = GradientSparsityMetrics(
        thresholds=[0.0, 10, 1, 1e-4, 1e-8, 1e-16, 1e-20],
        track_per_layer=False,
    )
    sparsity_analyzer.add_metric(gradient_sparsity)

    sparsity_callback = SparsityCallback(sparsity_analyzer)
    logger.info(f"  Sparsity analysis enabled (interval={sparsity_config.interval})")

    # Initialize delta checkpoint callback
    delta_checkpoint_callback = DeltaCheckpointCallback(
        output_dir=f"{output_base}/checkpoints/deltas_{adapter.name}_sft_{run_id}",
        enabled=cfg.delta_checkpoint_enabled,
        snapshot_dtype=cfg.delta_checkpoint_dtype,
        profiler=profiler,
    )
    if cfg.delta_checkpoint_enabled:
        logger.info(f"  Delta checkpointing enabled (dtype={cfg.delta_checkpoint_dtype})")

    # Initialize evaluation callback
    eval_callback = SFTEvalCallback(
        adapter=adapter,
        eval_data=eval_data,
        tokenizer=tokenizer,
        eval_every_n_steps=args.eval_every,
        max_new_tokens=1024,
    )

    # Initialize WandB
    if wandb.run is None and sft_config.report_to and "wandb" in sft_config.report_to:
        wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "grail")
        wandb_tags_str = args.wandb_tags or os.getenv("WANDB_TAGS", "")
        wandb_tags = [t.strip() for t in wandb_tags_str.split(",") if t.strip()]

        wandb_config = {
            "grail/model_id": cfg.model_id,
            "grail/learning_rate": cfg.lr,
            "grail/batch_size": cfg.batch_size,
            "grail/grad_accum_steps": cfg.grad_accum_steps,
            "grail/effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
            "grail/max_length": cfg.max_length,
            "grail/grad_clip": cfg.grad_clip,
            "grail/warmup_steps": cfg.warmup_steps,
            "grail/max_steps": cfg.max_steps,
            "grail/gradient_checkpointing": cfg.gradient_checkpointing,
            "grail/seed": args.seed,
            "grail/dataset": args.dataset,
            "grail/eval_every": args.eval_every,
            "grail/trainer_type": "sft",
            **{f"trl/{k}": v for k, v in sft_config.to_dict().items()},
        }

        wandb.init(
            project=wandb_project,
            name=sft_config.run_name,
            config=wandb_config,
            tags=wandb_tags if wandb_tags else None,
        )
        logger.info(f"  WandB initialized (project: {wandb_project}, tags: {wandb_tags})")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[eval_callback, sparsity_callback, delta_checkpoint_callback],
    )

    # Baseline evaluation
    logger.info("\nTraining with SFT...")
    eval_callback.run_and_log(model, step=0, label="BASELINE EVAL")

    # Train
    with profiler.track("training"):
        trainer.train()

    # Final evaluation
    final_step = trainer.state.global_step if hasattr(trainer, "state") else cfg.max_steps
    final_metrics = eval_callback.run_and_log(model, step=final_step, label="FINAL EVAL")

    # Print profiling summary
    profiler.print_summary()
    profiler.log_to_wandb()

    # Print results summary
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL RESULTS SUMMARY ({adapter.name.upper()} SFT)")
    logger.info("=" * 60)
    logger.info(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    logger.info(f"  Mean reward: {final_metrics['mean_reward']:.3f}")
    logger.info(f"  Correct: {final_metrics['correct']}/{final_metrics['total']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
