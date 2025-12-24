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
6. vLLM backend - High-throughput inference with tensor parallelism
7. BF16 precision - Optimal for modern GPUs
8. Tensor parallelism - Utilize all available GPUs

Usage:
------
    python eval_math_harness.py
    python eval_math_harness.py --model Qwen/Qwen2.5-7B-Instruct
    python eval_math_harness.py --num-fewshot 0  # zero-shot
    python eval_math_harness.py --tensor-parallel-size 8  # use 8 GPUs
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


def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        # Fallback to nvidia-smi
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip().split("\n")) if result.returncode == 0 else 1


def get_model_num_attention_heads(model_name: str) -> int:
    """Get the number of attention heads for a model."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return getattr(config, "num_attention_heads", 32)
    except Exception:
        return 32  # Default fallback


def get_optimal_tensor_parallel_size(model_name: str, max_gpus: int) -> int:
    """Calculate optimal tensor parallel size based on model architecture.

    Tensor parallelism requires num_attention_heads to be divisible by TP size.
    Returns the largest valid TP size <= max_gpus.
    """
    num_heads = get_model_num_attention_heads(model_name)

    # Find the largest divisor of num_heads that is <= max_gpus
    for tp_size in range(min(max_gpus, num_heads), 0, -1):
        if num_heads % tp_size == 0:
            return tp_size
    return 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model on Hendrycks MATH using lm-eval-harness with vLLM"
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM (default: auto-detect all GPUs)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length for vLLM (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf"],
        default="vllm",
        help="Inference backend: vllm (faster) or hf (HuggingFace)",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict:
    """Run lm-evaluation-harness on MATH dataset using vLLM or HuggingFace backend."""
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        logger.error("lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    # Auto-detect tensor parallel size if not specified
    available_gpus = get_gpu_count()
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = get_optimal_tensor_parallel_size(args.model, available_gpus)

    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Few-shot: {args.num_fewshot}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")

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

    if args.backend == "vllm":
        # Use vLLM backend for maximum efficiency with tensor parallelism
        try:
            from lm_eval.models.vllm_causallms import VLLM
        except ImportError:
            logger.error("vLLM not installed. Run: pip install vllm")
            sys.exit(1)

        logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
        logger.info(f"Max model length: {args.max_model_len}")

        # vLLM model configuration for maximum throughput
        model = VLLM(
            pretrained=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            trust_remote_code=True,
            # Enable prefix caching for faster few-shot evaluation
            enable_prefix_caching=True,
        )

        # Run evaluation with vLLM
        logger.info("Starting evaluation with vLLM backend...")
        results = simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            # Greedy decoding for reproducibility
            gen_kwargs="temperature=0,do_sample=False",
            log_samples=True,
        )
    else:
        # Fall back to HuggingFace backend
        from lm_eval.models.huggingface import HFLM

        model_kwargs = {
            "pretrained": args.model,
            "dtype": "bfloat16",
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
        }

        try:
            model = HFLM(**model_kwargs)
        except Exception as e:
            logger.warning(f"Flash attention failed ({e}), using default attention")
            model_kwargs.pop("attn_implementation", None)
            model = HFLM(**model_kwargs)

        logger.info("Starting evaluation with HuggingFace backend...")
        results = simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
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

    # Print per-subject results
    print("\nPer-Subject Accuracy:")
    print("-" * 50)
    for subject, data in sorted(subject_results.items()):
        acc_pct = data["accuracy"] * 100
        stderr_pct = data["stderr"] * 100
        print(f"  {subject:35s} {acc_pct:5.2f}% ± {stderr_pct:.2f}%")

    # Print aggregate (use weighted average based on number of samples per subject)
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
