#!/usr/bin/env python3
"""Compute pass@1, pass@5, pass@10 on MATH using vLLM with multiple samples."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Codex paper.

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} format."""
    # Find the last \boxed{...}
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace and common LaTeX formatting
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\!", "")
    return answer


def is_correct(pred: str, target: str) -> bool:
    """Check if prediction matches target."""
    pred_norm = normalize_answer(extract_boxed_answer(pred) or pred)
    target_norm = normalize_answer(extract_boxed_answer(target) or target)
    return pred_norm == target_norm


def build_prompt(problem: str, few_shot_examples: list[dict] = None) -> str:
    """Build prompt for MATH problem (zero-shot or few-shot)."""
    prompt = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Problem: {ex['problem']}\nSolution: {ex['solution']}\n\n"
    prompt += f"Problem: {problem}\nSolution:"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per problem (>= max k)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for testing)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    # Import vLLM
    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        max_num_seqs=512,  # More concurrent sequences
        enable_prefix_caching=True,
    )

    # Load MATH dataset (all 7 subjects)
    print("Loading MATH dataset...")
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    # Load all subjects
    test_data = []
    for subject in subjects:
        test_data.extend(load_dataset("EleutherAI/hendrycks_math", subject, split="test"))

    dataset = test_data[: args.limit] if args.limit else test_data
    print(f"Loaded {len(dataset)} test problems ({args.num_fewshot}-shot)")

    # Load few-shot examples if needed
    few_shot_examples = []
    if args.num_fewshot > 0:
        train_data = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        few_shot_examples = [train_data[i] for i in range(args.num_fewshot)]

    # Build prompts
    prompts = [build_prompt(ex["problem"], few_shot_examples) for ex in dataset]
    targets = [ex["solution"] for ex in dataset]

    # Sampling params for multiple samples
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,  # Generate n samples per prompt
    )

    print(f"Generating {args.n_samples} samples per problem for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate
    results = []
    for _idx, (output, target) in enumerate(
        tqdm(zip(outputs, targets, strict=False), total=len(outputs))
    ):
        # Check each sample
        correct_count = sum(
            1 for completion in output.outputs if is_correct(completion.text, target)
        )
        results.append(
            {
                "n_samples": args.n_samples,
                "n_correct": correct_count,
            }
        )

    # Compute pass@k for k=1,5
    k_values = [1, 5]
    pass_at_k_results = {}

    for k in k_values:
        if k <= args.n_samples:
            scores = [pass_at_k(r["n_samples"], r["n_correct"], k) for r in results]
            pass_at_k_results[f"pass@{k}"] = np.mean(scores) * 100

    # Print results
    print("\n" + "=" * 50)
    print(f"MATH pass@k Results - {args.model}")
    print(f"Temperature: {args.temperature}, Samples: {args.n_samples}")
    print("=" * 50)
    for k, score in pass_at_k_results.items():
        print(f"  {k}: {score:.2f}%")
    print("=" * 50)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"pass_at_k_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "temperature": args.temperature,
                "n_samples": args.n_samples,
                "num_problems": len(results),
                "results": pass_at_k_results,
                "per_problem": results,
            },
            f,
            indent=2,
        )

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
