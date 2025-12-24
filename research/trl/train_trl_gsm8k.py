#!/usr/bin/env python3
"""Minimal TRL GRPO training script for GSM8K matching GRAIL hyperparameters."""

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from grail.shared.chat_templates import build_qwen_chat_template
from grail.trainer.metrics import KMetricsAggregator, TaskReplicateResult

# Force unbuffered output for better logging in nohup mode
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# Load environment from .env for WandB
load_dotenv("/root/grail/.env")  # Load WandB API key and project

sys.path.append("/root/grail")


# ────────────────  HYPERPARAMETERS (from .env GRAIL config)  ────────────────
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
    report_ks: tuple = (1, 5, 10)
    # Evaluation optimization (for multi-GPU with 8 A100s)
    eval_batch_size: int = 128  # Large batch for parallel generation
    eval_num_workers: int = 4  # Dataloader workers
    # Max groups for GRPO (from GRPO_MAX_GROUPS)
    max_groups: int = 128


cfg = Config()

# ────────────────  SYSTEM PROMPT & TAGS  ────────────────
# Unbracketed tag tokens (used by parsers and regex)
REASONING_START_TOKEN = "start_working_out"
REASONING_END_TOKEN = "end_working_out"
SOLUTION_START_TOKEN = "SOLUTION"
SOLUTION_END_TOKEN = "SOLUTION"

# Bracketed forms (used in prompts/templates)
REASONING_START = f"<{REASONING_START_TOKEN}>"
REASONING_END = f"</{REASONING_END_TOKEN}>"
SOLUTION_START = f"<{SOLUTION_START_TOKEN}>"
SOLUTION_END = f"</{SOLUTION_END_TOKEN}>"

# Canonical system prompt referencing the tags above
SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}."
)

# Qwen chat template with system prompt and reasoning start
QWEN_CHAT_TEMPLATE = build_qwen_chat_template(
    system_prompt=SYSTEM_PROMPT, reasoning_start=REASONING_START
)


# ────────────────  REWARD PARSER (from GSM8KEnv)  ────────────────
def parse_gsm8k_golden(text: str) -> str:
    """Extract gold answer from GSM8K dataset format."""
    match = re.search(r"####\s*(.+)", text or "")
    if match:
        return match.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:[\.,]\d+)?", text or "")
    return nums[-1].replace(",", "").strip() if nums else ""


def parse_completion(text: str) -> dict:
    """Parse completion for thinking/answer tags and numeric content."""
    # Be robust to case in tags (model may emit <solution> instead of <SOLUTION>)
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
        num_match = re.search(r"[-+]?\d+(?:[\.,]\d+)?", inside)
        if num_match:
            answer_text = num_match.group(0).replace(",", "").strip()
            is_numeric_only = bool(re.match(r"^[-+]?[\d.,\s]+$", inside))
        trailing = len(text) - answer_match.end()

    return {
        "answer_text": answer_text,
        "has_thinking": has_thinking,
        "has_answer": has_answer,
        "is_numeric_only": is_numeric_only,
        "trailing": trailing,
    }


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    return re.sub(r"[\s\.]+$", "", s.strip().lower())


def compute_reward(completion: str, gold_answer: str) -> float:
    """Compute decomposed reward matching GSM8KEnv.

    Components (weights):
    - Correctness (0.6): exact match
    - Strict format (0.15): numeric-only + no trailing
    - Thinking (0.1): has thinking block
    - Answer (0.1): has answer block
    - No trailing (0.05): penalty for trailing text
    """
    parsed = parse_completion(completion)
    gold_parsed = parse_gsm8k_golden(gold_answer)

    # Correctness
    pred_norm = normalize_answer(parsed["answer_text"])
    gold_norm = normalize_answer(gold_parsed)
    correctness = 0.6 if pred_norm == gold_norm else 0.0

    # Strict format (numeric-only + no trailing)
    strict_format = (
        0.15
        if (parsed["has_answer"] and parsed["is_numeric_only"] and parsed["trailing"] == 0)
        else 0.0
    )

    # Thinking format
    thinking = 0.1 if parsed["has_thinking"] else 0.0

    # Answer format
    answer = 0.1 if parsed["has_answer"] else 0.0

    # No trailing penalty
    no_trailing = 0.05 if parsed["trailing"] == 0 else 0.0

    return correctness + strict_format + thinking + answer + no_trailing


# ────────────────  DATA PREPARATION  ────────────────
def prepare_dataset(tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load and format GSM8K dataset for TRL GRPO.

    Produces:
      - prompt: chat-formatted string (system + user) for generation
      - gold_answer: raw gold solution text for reward computation
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if cfg.num_train_samples is not None:
        ds = ds.select(range(cfg.num_train_samples))

    def format_prompt(example: dict[str, Any]) -> dict[str, str]:
        question = example["question"]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "gold_answer": example["answer"]}

    print(f"  Training dataset: {len(ds)} samples")
    return ds.map(format_prompt, remove_columns=ds.column_names)


def prepare_eval_dataset() -> Dataset:
    """Load eval dataset."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if cfg.num_eval_samples is not None:
        ds = ds.select(range(cfg.num_eval_samples))
    print(f"  Eval dataset: {len(ds)} samples")
    return ds


# ────────────────  VLLM EVALUATION CALLBACK  ────────────────
class VLLMEvalCallback(TrainerCallback):
    """Lightweight evaluation using TRL vLLM server /chat/ endpoint with KMetricsAggregator."""

    def __init__(
        self,
        eval_ds: Dataset,
        tokenizer: PreTrainedTokenizer,
        vllm_base_url: str,
        model_name: str,
        eval_every_n_steps: int = 30,
        max_concurrent: int = 512,
    ) -> None:
        self.eval_ds = eval_ds
        self.tokenizer = tokenizer
        self.eval_every_n = eval_every_n_steps
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.base_url = vllm_base_url.rstrip("/")
        self._metrics_defined = False

        print(
            f"✓ VLLMEvalCallback initialized: url={vllm_base_url}, eval_every={eval_every_n_steps}"
        )

    def run_and_log(self, step: int, label: str = "VLLM EVAL") -> dict[str, float]:
        """Run evaluation and log to WandB (DRY helper for baseline/periodic/final eval)."""
        print(f"\n{'=' * 80}")
        print(f"[{label}] Step {step}: Starting evaluation...")
        print(f"{'=' * 80}")

        metrics = asyncio.run(self._run_eval())

        # Log to WandB with independent eval_step axis
        try:
            import wandb

            if wandb.run is not None:
                # Define metrics once on first log
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
        """Run evaluation using vLLM chat completions API and compute metrics."""
        import time

        from tqdm import tqdm

        start_time = time.time()

        aggregator = KMetricsAggregator(report_ks=cfg.report_ks)

        # Process in batches
        total_tasks = len(self.eval_ds)
        batch_size = cfg.eval_batch_size

        with tqdm(total=total_tasks, desc="Eval", unit="task") as pbar:
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch = self.eval_ds[batch_start:batch_end]
                batch_questions = batch["question"]
                batch_golds = batch["answer"]

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

                # Generate completions using async chat API with concurrency control
                completions = await self._generate_batch(tasks_to_generate)

                # Log sample completions (first 3 from first batch)
                if batch_start == 0:
                    print("\n  ━━━ Sample Completions ━━━")
                    for i in range(min(3, len(completions))):
                        question = tasks_to_generate[i]
                        completion = completions[i]
                        metadata = task_metadata[i]
                        gold = batch_golds[metadata["task_idx"]]
                        reward = compute_reward(completion, gold)

                        # Truncate for display
                        q_display = question[:150] + "..." if len(question) > 150 else question
                        c_display = (
                            completion[:300] + "..." if len(completion) > 300 else completion
                        )
                        print(f"\n  Sample {i + 1}:")
                        print(f"    Question: {q_display}")
                        print(f"    Completion: {c_display}")
                        print(f"    Reward: {reward:.3f} | Gold: {parse_gsm8k_golden(gold)}")
                    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                # Compute rewards and aggregate
                for completion_text, metadata in zip(completions, task_metadata, strict=False):
                    task_id = metadata["task_id"]
                    task_idx = metadata["task_idx"]
                    replicate_idx = metadata["replicate_idx"]
                    gold = batch_golds[task_idx]

                    # Compute reward
                    reward = compute_reward(completion_text, gold)
                    success = reward >= 0.6  # Correctness threshold

                    # Add to aggregator
                    aggregator.add(
                        TaskReplicateResult(
                            task_id=task_id,
                            replicate_idx=replicate_idx,
                            reward=reward,
                            success=success,
                        )
                    )

                # Update progress bar
                pbar.update(len(batch_questions))

        # Compute final metrics
        metrics = aggregator.summarize()
        elapsed = time.time() - start_time
        throughput = (total_tasks * cfg.eval_replicates) / elapsed if elapsed > 0 else 0

        print(
            f"  ✓ Evaluated {total_tasks} tasks × {cfg.eval_replicates} reps in {elapsed:.2f}s "
            f"({throughput:.1f} completions/sec)"
        )

        return metrics

    async def _generate_batch(self, questions: list[str]) -> list[str]:
        """Generate completions using TRL /chat/ endpoint with true batching.

        Key: TRL's /chat/ accepts multiple message lists per request, enabling
        vLLM's continuous batching for parallel generation across all prompts.
        """
        import asyncio

        import aiohttp

        # Batch size per request (vLLM processes all in parallel)
        # Optimal: 64-128 for 4xGPU setup (balance memory and throughput)
        vllm_batch_size = 64
        total = len(questions)
        num_requests = (total + vllm_batch_size - 1) // vllm_batch_size
        print(
            f"    Generating {total} completions via {num_requests} batched requests (batch_size={vllm_batch_size})"
        )

        async def generate_batch_request(
            session: aiohttp.ClientSession, batch_questions: list[str], start_idx: int
        ) -> tuple[int, list[list[int]]]:
            """Send one batched request with multiple message lists."""
            max_retries = 3
            base_backoff = 1.0

            # Build batch: list of message lists (one per question)
            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                for q in batch_questions
            ]

            payload = {
                "messages": messages,  # Multiple conversations in ONE request
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
                            # Returns: {"completion_ids": [[...], [...], ...]}
                            # One completion per input message list
                            completion_ids_batch = data["completion_ids"]
                            return (start_idx, completion_ids_batch)
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        backoff = base_backoff * (2**attempt)
                        await asyncio.sleep(backoff)
                    else:
                        print(
                            f"  ⚠️  Batch request starting at {start_idx} failed: {type(e).__name__}"
                        )
                        # Return empty completions for failed batch
                        return (start_idx, [[] for _ in batch_questions])
            return (start_idx, [[] for _ in batch_questions])

        # Split into batches and send concurrent batched requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for batch_start in range(0, total, vllm_batch_size):
                batch_end = min(batch_start + vllm_batch_size, total)
                batch_questions = questions[batch_start:batch_end]
                tasks.append(generate_batch_request(session, batch_questions, batch_start))

            # Execute all batched requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=False)

        # Flatten results and decode
        all_completion_ids: list[list[int]] = [[] for _ in range(total)]
        for start_idx, completion_ids_batch in results:
            for offset, comp_ids in enumerate(completion_ids_batch):
                all_completion_ids[start_idx + offset] = comp_ids

        # Decode all completions
        completions = []
        for comp_ids in all_completion_ids:
            if comp_ids:
                completion_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions.append(completion_text)
            else:
                completions.append("")

        return completions


# ────────────────  MAIN TRAINING  ────────────────
def main() -> None:
    print("🚀 Loading model and tokenizer...")
    # Remove device_map="auto" when using single GPU training (CUDA_VISIBLE_DEVICES=0)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except (ImportError, RuntimeError) as e:
        print(f"⚠️  Flash Attention 2 unavailable ({type(e).__name__}), using default attention")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models during generation
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE  # Use exact GRAIL chat template

    print("📊 Preparing datasets...")
    train_ds = prepare_dataset(tokenizer)
    eval_ds = prepare_eval_dataset()  # For VLLMEvalCallback
    prompt_to_answer = {row["prompt"]: row["gold_answer"] for row in train_ds}

    print("⚙️  Configuring GRPO trainer...")

    # Login to WandB with API key from .env
    import wandb

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print(f"  ✓ WandB logged in (project: {os.getenv('WANDB_PROJECT', 'grail')})")

    # Calculate max_prompt_length: total max_length minus max_completion_tokens
    max_prompt_length = cfg.max_length - cfg.max_new_tokens  # 2048 - 1024 = 1024

    grpo_config = GRPOConfig(
        output_dir="./outputs/trl_gsm8k",
        # Learning rate (GRAIL_TRAINER_LR=3e-6)
        learning_rate=cfg.lr,
        # Epochs (GRAIL_TRAINER_EPOCHS=1)
        num_train_epochs=cfg.epochs,
        # Batch size (GRAIL_TRAINER_BATCH_SIZE=4)
        per_device_train_batch_size=cfg.batch_size,
        # Gradient accumulation (GRAIL_TRAINER_GRAD_ACCUM_STEPS=128)
        gradient_accumulation_steps=cfg.grad_accum_steps,
        # Gradient clipping (GRAIL_TRAINER_GRAD_CLIP=1.0)
        max_grad_norm=cfg.grad_clip,
        # Warmup steps (GRAIL_TRAINER_WARMUP_STEPS=50)
        warmup_steps=cfg.warmup_steps,
        # KL coefficient (GRAIL_TRAINER_KL_COEF=0.0)
        beta=cfg.kl_coef,
        # PPO clip epsilon
        epsilon=cfg.ppo_clip_eps,
        epsilon_high=cfg.ppo_clip_eps_upper,
        # Max prompt length (derived from GRAIL_TRAINER_MAX_LENGTH - GRPO_MAX_COMPLETION_TOKENS)
        max_prompt_length=max_prompt_length,
        # Max completion tokens (GRPO_MAX_COMPLETION_TOKENS=1024)
        max_completion_length=cfg.max_new_tokens,
        # Generation parameters
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=1.1,
        # Group size: 16 completions per prompt (rollouts_per_problem)
        num_generations=cfg.rollouts_per_problem,
        generation_batch_size=16,
        steps_per_generation=None,
        logging_steps=1,
        # Enable logging a sample of (prompt, completion) pairs each logging step
        log_completions=True,
        num_completions_to_print=1,
        wandb_log_unique_prompts=True,
        save_strategy="no",
        bf16=True,
        report_to=["wandb"],
        eval_strategy="no",  # Disable TRL's internal eval (using VLLMEvalCallback instead)
        run_name="trl_gsm8k_grpo_qwen15b_env_matched",
        # Loss type (GRAIL_GRPO_VARIANT=dapo)
        loss_type="dapo",
        # vLLM configuration for offloading generation to separate GPUs
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url="http://127.0.0.1:8000",
        # Importance sampling (GRAIL_TRAINER_IS_RATIO_MAX=2.5)
        vllm_importance_sampling_correction=False,
        vllm_importance_sampling_cap=cfg.is_ratio_max,
    )

    # Reward function wrapper
    def reward_fn(completions: list[str], prompts: list[str], **kwargs: Any) -> list[float]:
        # TRL passes all non-reserved dataset columns to reward_fn as lists
        if "gold_answer" in kwargs and kwargs["gold_answer"]:
            golds = kwargs["gold_answer"]
            return [compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]
        if "metadatas" in kwargs and kwargs["metadatas"]:
            golds = [m.get("gold_answer", "") for m in kwargs["metadatas"]]
            return [compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]
        golds = [prompt_to_answer.get(p, "") for p in prompts]
        return [compute_reward(c, g) for c, g in zip(completions, golds, strict=False)]

    print("🏋️  Training with GRPO...")

    # Initialize vLLM evaluation callback
    vllm_eval_callback = VLLMEvalCallback(
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        vllm_base_url=grpo_config.vllm_server_base_url,
        model_name=cfg.model_id,
        eval_every_n_steps=30,
        max_concurrent=512,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[vllm_eval_callback],
    )

    # Baseline evaluation before training (step=0, after WandB init)
    vllm_eval_callback.run_and_log(step=0, label="BASELINE EVAL")

    trainer.train()

    # Final evaluation after training (use max steps as step number)
    final_step = trainer.state.global_step if hasattr(trainer, "state") else 9999
    final_metrics = vllm_eval_callback.run_and_log(step=final_step, label="FINAL EVAL")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
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