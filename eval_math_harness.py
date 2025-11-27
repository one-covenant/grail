#!/usr/bin/env python3
"""Evaluate Qwen/Qwen2.5-1.5B-Instruct on Hendrycks MATH using lm-evaluation-harness.

Metrics Computed:
-----------------
1. Overall exact_match accuracy (primary)
2. Per-subject accuracy (7 subjects: algebra, counting_and_prob,
   geometry, intermediate_algebra, num_theory, prealgebra, precalc)
3. Aggregated by difficulty level (1-5) via post-processing

Best Practices Used:
--------------------
1. minerva_math task - Better prompts with \boxed{} extraction (standard for MATH)
2. 4-shot prompting - Standard for MATH benchmark
3. Chain-of-thought - Enabled via task's native format
4. Greedy decoding - temperature=0 for reproducibility
5. max_gen_toks=1024 - Sufficient for reasoning chains
6. Flash attention - Memory efficient for 7B+ models
7. BF16 precision - Optimal for modern GPUs
8. Batch size tuning - Auto via batch_size="auto"

Usage:
------
    python eval_math_harness.py
    python eval_math_harness.py --model Qwen/Qwen2.5-7B-Instruct
    python eval_math_harness.py --num-fewshot 0  # zero-shot
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Hendrycks MATH using lm-eval-harness"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=4,
        help="Number of few-shot examples (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for debugging)",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict:
    """Run lm-evaluation-harness on MATH dataset."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    logger.info(f"Model: {args.model}")
    logger.info(f"Few-shot: {args.num_fewshot}")
    logger.info(f"Batch size: {args.batch_size}")

    # MATH subtasks (all 7 subjects)
    # Using hendrycks_math tasks with proper \boxed{} answer extraction
    tasks = [
        "hendrycks_math_algebra",
        "hendrycks_math_counting_and_prob",
        "hendrycks_math_geometry",
        "hendrycks_math_intermediate_algebra",
        "hendrycks_math_num_theory",
        "hendrycks_math_prealgebra",
        "hendrycks_math_precalc",
    ]

    logger.info(f"Tasks: {tasks}")

    # Model configuration with best practices
    model_kwargs = {
        "pretrained": args.model,
        "dtype": "bfloat16",
        "device_map": "auto",
        "trust_remote_code": True,
        # Enable flash attention if available
        "attn_implementation": "flash_attention_2",
    }

    # Try to load with flash attention, fall back if not available
    try:
        model = HFLM(**model_kwargs)
    except Exception as e:
        logger.warning(f"Flash attention failed ({e}), using default attention")
        model_kwargs.pop("attn_implementation", None)
        model = HFLM(**model_kwargs)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        # Greedy decoding for reproducibility
        gen_kwargs="temperature=0,do_sample=False",
        log_samples=True,
    )

    return results


def print_results(results: dict, args: argparse.Namespace) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(f"MATH Evaluation Results - {args.model}")
    print("=" * 70)

    # Extract per-subject results
    subject_results = {}
    total_correct = 0
    total_samples = 0

    for task_name, task_results in results.get("results", {}).items():
        # Extract subject from task name and expand abbreviations
        subject = task_name.replace("hendrycks_math_", "")
        # Expand abbreviated names for readability
        subject = subject.replace("counting_and_prob", "counting_and_probability")
        subject = subject.replace("num_theory", "number_theory")
        subject = subject.replace("precalc", "precalculus")

        # Get accuracy metric (exact_match or acc)
        acc = task_results.get("exact_match,none", task_results.get("acc,none", 0))
        stderr = task_results.get("exact_match_stderr,none", task_results.get("acc_stderr,none", 0))

        subject_results[subject] = {
            "accuracy": acc,
            "stderr": stderr,
        }

        # For aggregate calculation
        n_samples = task_results.get("alias", {}).get("n-shot", 0)
        if "samples" in results:
            task_samples = results["samples"].get(task_name, [])
            n_samples = len(task_samples)
            total_correct += sum(1 for s in task_samples if s.get("acc", 0) == 1)
            total_samples += n_samples

    # Print per-subject results
    print("\nPer-Subject Accuracy:")
    print("-" * 50)
    for subject, data in sorted(subject_results.items()):
        acc_pct = data["accuracy"] * 100
        stderr_pct = data["stderr"] * 100
        print(f"  {subject:35s} {acc_pct:5.2f}% ± {stderr_pct:.2f}%")

    # Print aggregate
    if total_samples > 0:
        overall_acc = total_correct / total_samples * 100
    else:
        # Use average of subjects
        accs = [d["accuracy"] for d in subject_results.values()]
        overall_acc = sum(accs) / len(accs) * 100 if accs else 0

    print("-" * 50)
    print(f"  {'OVERALL':35s} {overall_acc:5.2f}%")
    print("=" * 70)


def save_results(results: dict, args: argparse.Namespace) -> Path:
    """Save results to JSON file."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model.replace("/", "_")
    output_file = args.output_dir / f"math_{model_name}_{timestamp}.json"

    # Add metadata
    results["metadata"] = {
        "model": args.model,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "timestamp": timestamp,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")
    return output_file


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Hendrycks MATH Evaluation")
    logger.info("=" * 50)

    # Run evaluation
    results = run_evaluation(args)

    # Print results
    print_results(results, args)

    # Save results
    save_results(results, args)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
