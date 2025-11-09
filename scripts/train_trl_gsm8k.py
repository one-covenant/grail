#!/usr/bin/env python3
"""Minimal TRL GRPO training script for GSM8K matching GRAIL hyperparameters."""

import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# Force unbuffered output for better logging in nohup mode
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Load environment from .env for WandB
from dotenv import load_dotenv
load_dotenv("/root/grail/.env")  # Load WandB API key and project

# Import chat template builder and prompt constants
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.prompt_constants import SYSTEM_PROMPT, REASONING_START


# ────────────────  HYPERPARAMETERS (from GRAIL config)  ────────────────
@dataclass
class Config:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lr: float = 2e-6
    epochs: int = 1
    batch_size: int = 16  # 16 groups (prompts) per step
    grad_accum_steps: int = 16
    max_length: int = 1536
    grad_clip: float = 1.0
    warmup_steps: int = 50
    kl_coef: float = 0.0
    entropy_coef: float = 0.0005
    ppo_clip_eps: float = 0.2
    ppo_clip_eps_upper: float = 0.28
    is_ratio_max: float = 2.5
    logratio_clamp: float = 0.92
    num_train_samples: int | None = None  # None = use all training samples
    num_eval_samples: int | None = None  # None = use all test samples
    rollouts_per_problem: int = 16
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512
    eval_replicates: int = 5
    report_ks: tuple = (1, 5, 10)
    # Evaluation optimization (for multi-GPU with 8 A100s)
    eval_batch_size: int = 512  # Large batch for parallel generation
    eval_num_workers: int = 4  # Dataloader workers


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
    system_prompt=SYSTEM_PROMPT,
    reasoning_start=REASONING_START
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
    has_thinking = bool(re.search(rf"<{REASONING_START_TOKEN}>.*?</{REASONING_END_TOKEN}>", text, re.DOTALL))
    answer_match = re.search(rf"<{SOLUTION_START_TOKEN}>\s*(.+?)\s*</{SOLUTION_END_TOKEN}>", text, re.DOTALL)
    
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
    strict_format = 0.15 if (parsed["has_answer"] and parsed["is_numeric_only"] and parsed["trailing"] == 0) else 0.0
    
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

    def format_prompt(example: Dict[str, Any]) -> Dict[str, str]:
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
    """Load eval dataset.
    
    NOTE: TRL's evaluation expects raw messages (conversational format),
    not pre-formatted prompts. The prompt will be formatted by TRL internally.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if cfg.num_eval_samples is not None:
        ds = ds.select(range(cfg.num_eval_samples))
    
    def format_for_trl(example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question"]
        # TRL expects "prompt" to be the chat messages (not templated string)
        # TRL will apply the chat template internally
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "gold_answer": example["answer"],
        }
    
    print(f"  Eval dataset: {len(ds)} samples")
    return ds.map(format_for_trl, remove_columns=ds.column_names)


# ────────────────  EVALUATION (OPTIMIZED)  ────────────────
def evaluate_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_ds: Dataset,
    batch_size: int = 64, num_workers: int = 4
) -> Dict[str, float]:
    """Compute pass@k, mean@k, best@k metrics with optimized batch processing.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for encoding
        eval_ds: Evaluation dataset
        batch_size: Batch size for generation (large for multi-GPU)
        num_workers: Number of dataloader workers
    """
    model.eval()
    device = model.device
    
    # Create DataLoader for efficient batching
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        eval_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 0 to avoid pickling issues with HF datasets
        pin_memory=True if "cuda" in str(device) else False
    )
    
    results = []
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch_questions = batch["question"]
            batch_golds = batch["answer"]
            batch_size_actual = len(batch_questions)
            
            # Process all replicates for this batch in parallel
            # Create replicated batch: (batch_size * replicates,)
            replicated_questions = batch_questions * cfg.eval_replicates
            
            # Format prompts with chat template
            prompts = []
            for question in replicated_questions:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
            
            # Tokenize entire batch at once
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,  # Prompt only, not completions
            ).to(device)
            
            # Batch generate (leverages multi-GPU parallelism)
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode all completions
            completions = []
            for i, output in enumerate(outputs):
                # Skip prompt tokens
                prompt_len = inputs.input_ids[i].shape[0]
                generated = output[prompt_len:]
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                completions.append(completion)
            
            # Group by original batch and compute rewards
            # For each problem in batch, collect exactly 5 replicates
            for sample_idx in range(batch_size_actual):
                rewards: List[float] = []
                for rep_idx in range(cfg.eval_replicates):  # Always 5
                    # Index into flattened completions: [q0_r0, q0_r1, ..., q0_r4, q1_r0, q1_r1, ...]
                    completion_idx = sample_idx + rep_idx * batch_size_actual
                    completion = completions[completion_idx]
                    gold = batch_golds[sample_idx]
                    reward = compute_reward(completion, gold)
                    rewards.append(reward)
                
                # Verify exactly 5 rewards per problem
                assert len(rewards) == cfg.eval_replicates, f"Expected {cfg.eval_replicates} rewards, got {len(rewards)}"
                
                # Sort rewards descending for top-k
                results.append(sorted(rewards, reverse=True))
            
            if (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                print(f"  Progress: {batch_idx + 1}/{len(dataloader)} batches")
    
    # Compute metrics
    metrics = {}
    for k in cfg.report_ks:
        if k > cfg.eval_replicates:
            continue
        
        # pass@k: at least one correct in top-k (correctness threshold = 0.6)
        pass_at_k = sum(max(r[:k]) >= 0.6 for r in results) / len(results)
        
        # mean@k: average of top-k rewards
        mean_at_k = sum(sum(r[:k]) / k for r in results) / len(results)
        
        # best@k: best reward in top-k
        best_at_k = sum(max(r[:k]) for r in results) / len(results)
        
        metrics[f"pass@{k}"] = pass_at_k
        metrics[f"mean@{k}"] = mean_at_k
        metrics[f"best@{k}"] = best_at_k
    
    return metrics


# ────────────────  MAIN TRAINING  ────────────────
def main() -> None:
    print("🚀 Loading model and tokenizer...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    except (ImportError, RuntimeError, ValueError) as e:
        print(f"⚠️  Flash Attention 2 unavailable ({type(e).__name__}), using default attention")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For decoder-only models during generation
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE  # Use exact GRAIL chat template
    
    print("📊 Preparing datasets...")
    train_ds = prepare_dataset(tokenizer)
    eval_ds = prepare_eval_dataset()
    prompt_to_answer = {row["prompt"]: row["gold_answer"] for row in train_ds}
    
    print("⚙️  Configuring GRPO trainer...")
    
    # Login to WandB with API key from .env
    import wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print(f"  ✓ WandB logged in (project: {os.getenv('WANDB_PROJECT', 'grail')})")
    
    grpo_config = GRPOConfig(
        output_dir="./outputs/trl_gsm8k",
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=16,  # Must be >= num_generations (16)
        gradient_accumulation_steps=cfg.grad_accum_steps,
        max_grad_norm=cfg.grad_clip,
        warmup_steps=cfg.warmup_steps,
        beta=cfg.kl_coef,  # Beta is KL coefficient in GRPO
        epsilon=cfg.ppo_clip_eps,  # PPO epsilon
        epsilon_high=cfg.ppo_clip_eps_upper,  # Upper PPO epsilon
        max_prompt_length=512,  # Reasonable prompt limit
        max_completion_length=cfg.max_new_tokens,  # Max new tokens
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,  # Match loop.py: 50 highest probability tokens
        repetition_penalty=1.1,  # Match loop.py: penalize repeating tokens
        num_generations=16,  # group size: 16 completions per prompt
        generation_batch_size=16,  # generate 16 prompts per batch for generation
        steps_per_generation=None,
        logging_steps=1,
        # Enable logging a small sample of (prompt, completion) pairs each logging step.
        # Prints to console if `rich` is installed and logs a WandB table named "completions".
        log_completions=True,
        num_completions_to_print=3,
        wandb_log_unique_prompts=True,
        save_strategy="no",
        bf16=True,  # Disable if no GPU or GPU doesn't support it
        report_to=["wandb"],
        eval_strategy="no",  # Disable built-in eval; we'll use custom FullEvalEveryN callback
        # eval_steps=30,  # Not needed with eval_strategy="no"
        run_name="trl_gsm8k_grpo_qwen15b_g16x16",
        loss_type="dapo",  # Match config.py GRPO_VARIANT
    )
    
    # Reward function wrapper
    def reward_fn(completions: List[str], prompts: List[str], **kwargs: Any) -> List[float]:
        # TRL passes all non-reserved dataset columns to reward_fn as lists
        if "gold_answer" in kwargs and kwargs["gold_answer"]:
            golds = kwargs["gold_answer"]
            return [compute_reward(c, g) for c, g in zip(completions, golds)]
        if "metadatas" in kwargs and kwargs["metadatas"]:
            golds = [m.get("gold_answer", "") for m in kwargs["metadatas"]]
            return [compute_reward(c, g) for c, g in zip(completions, golds)]
        golds = [prompt_to_answer.get(p, "") for p in prompts]
        return [compute_reward(c, g) for c, g in zip(completions, golds)]
    
    print("🎯 Baseline evaluation (optimized batching)...")
    # baseline_metrics = evaluate_model(
    #     model, tokenizer, eval_ds,
    #     batch_size=cfg.eval_batch_size,
    #     num_workers=cfg.eval_num_workers
    # )
    # print("Baseline metrics:", baseline_metrics)
    
    print("🏋️  Training with GRPO...")
    # Callback to run full evaluation every 30 steps and log to wandb
    try:
        import wandb  # type: ignore
        _wandb_available = True
    except Exception:
        _wandb_available = False

    class FullEvalEveryN(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            if state.global_step > 0 and state.global_step % 30 == 0:
                metrics = evaluate_model(
                    model,
                    tokenizer,
                    eval_ds,
                    batch_size=cfg.eval_batch_size,
                    num_workers=cfg.eval_num_workers,
                )
                if _wandb_available:
                    data = {f"eval/{k}": v for k, v in metrics.items()}
                    data["global_step"] = state.global_step
                    wandb.log(data, step=state.global_step)
                print(f"[Eval @ step {state.global_step}] {metrics}")
                
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        eval_dataset=eval_ds,
        callbacks=[FullEvalEveryN()],
    )
    
    trainer.train()
    
    print("📈 Final evaluation (optimized batching)...")
    final_metrics = evaluate_model(
        model, tokenizer, eval_ds,
        batch_size=cfg.eval_batch_size,
        num_workers=cfg.eval_num_workers
    )
    print("Final metrics:", final_metrics)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for k in cfg.report_ks:
        if k > cfg.eval_replicates:
            continue
        print(f"\nMetrics @ k={k}:")
        print(f"  pass@{k}:  {baseline_metrics[f'pass@{k}']:.3f} → {final_metrics[f'pass@{k}']:.3f}")
        print(f"  mean@{k}:  {baseline_metrics[f'mean@{k}']:.3f} → {final_metrics[f'mean@{k}']:.3f}")
        print(f"  best@{k}:  {baseline_metrics[f'best@{k}']:.3f} → {final_metrics[f'best@{k}']:.3f}")


if __name__ == "__main__":
    main()

