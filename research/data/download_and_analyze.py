#!/usr/bin/env python3
"""
Download Triton/PyTorch datasets and analyze token distributions.
Creates histograms for user prompts, completions, and total tokens.
"""

import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Base paths
DATA_DIR = Path(__file__).parent / "raw"
FIGURES_DIR = Path(__file__).parent / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# User prompt template
USER_PROMPT_TEMPLATE = """Optimize the following PyTorch code with Triton kernels:
```python
{pytorch_code}
```

Think about your optimization strategy, then provide the Triton implementation.

Output format:
<think>
[Your reasoning about how to optimize this operation]
</think>

<triton_kernel>
[Your complete Triton implementation]
</triton_kernel>
"""

# Initialize tokenizer (using Qwen3 8B Instruct)
print("Loading Qwen3-8B tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


def create_user_prompt(pytorch_code: str) -> str:
    """Create the user prompt with injected PyTorch code."""
    return USER_PROMPT_TEMPLATE.format(pytorch_code=pytorch_code)


def plot_histograms(
    dataset_name: str,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    total_tokens: list[int],
):
    """Create and save histograms for token distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{dataset_name} - Token Distribution (n={len(prompt_tokens)})", fontsize=14)

    # User prompt histogram
    axes[0].hist(prompt_tokens, bins=50, edgecolor="black", alpha=0.7, color="blue")
    axes[0].set_title("User Prompt Tokens")
    axes[0].set_xlabel("Token Count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(
        np.median(prompt_tokens),
        color="red",
        linestyle="--",
        label=f"Median: {int(np.median(prompt_tokens))}",
    )
    axes[0].axvline(
        np.mean(prompt_tokens),
        color="green",
        linestyle="--",
        label=f"Mean: {int(np.mean(prompt_tokens))}",
    )
    axes[0].legend()

    # Completion histogram
    axes[1].hist(completion_tokens, bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Completion Tokens")
    axes[1].set_xlabel("Token Count")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(
        np.median(completion_tokens),
        color="red",
        linestyle="--",
        label=f"Median: {int(np.median(completion_tokens))}",
    )
    axes[1].axvline(
        np.mean(completion_tokens),
        color="green",
        linestyle="--",
        label=f"Mean: {int(np.mean(completion_tokens))}",
    )
    axes[1].legend()

    # Total histogram
    axes[2].hist(total_tokens, bins=50, edgecolor="black", alpha=0.7, color="green")
    axes[2].set_title("Total Tokens (Prompt + Completion)")
    axes[2].set_xlabel("Token Count")
    axes[2].set_ylabel("Frequency")
    axes[2].axvline(
        np.median(total_tokens),
        color="red",
        linestyle="--",
        label=f"Median: {int(np.median(total_tokens))}",
    )
    axes[2].axvline(
        np.mean(total_tokens),
        color="green",
        linestyle="--",
        label=f"Mean: {int(np.mean(total_tokens))}",
    )
    axes[2].legend()

    plt.tight_layout()
    output_path = FIGURES_DIR / f"{dataset_name.replace(' ', '_').lower()}_histograms.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved histogram to {output_path}")


def analyze_dataset(
    dataset_name: str,
    pytorch_codes: list[str],
    completions: list[str],
):
    """Analyze a dataset and create histograms."""
    print(f"\nAnalyzing {dataset_name}...")

    prompt_tokens = []
    completion_tokens = []
    total_tokens = []

    for pytorch_code, completion in tqdm(
        zip(pytorch_codes, completions, strict=True),
        total=len(pytorch_codes),
        desc="  Counting tokens",
    ):
        user_prompt = create_user_prompt(pytorch_code)
        p_tokens = count_tokens(user_prompt)
        c_tokens = count_tokens(completion) if completion else 0

        prompt_tokens.append(p_tokens)
        completion_tokens.append(c_tokens)
        total_tokens.append(p_tokens + c_tokens)

    # Print statistics
    print(f"  Samples: {len(prompt_tokens)}")
    print(
        f"  Prompt tokens - Mean: {np.mean(prompt_tokens):.0f}, Median: {np.median(prompt_tokens):.0f}, Max: {max(prompt_tokens)}"
    )
    print(
        f"  Completion tokens - Mean: {np.mean(completion_tokens):.0f}, Median: {np.median(completion_tokens):.0f}, Max: {max(completion_tokens)}"
    )
    print(
        f"  Total tokens - Mean: {np.mean(total_tokens):.0f}, Median: {np.median(total_tokens):.0f}, Max: {max(total_tokens)}"
    )

    # Create histograms
    plot_histograms(dataset_name, prompt_tokens, completion_tokens, total_tokens)

    return {
        "dataset": dataset_name,
        "samples": len(prompt_tokens),
        "prompt_mean": np.mean(prompt_tokens),
        "prompt_median": np.median(prompt_tokens),
        "prompt_max": max(prompt_tokens),
        "completion_mean": np.mean(completion_tokens),
        "completion_median": np.median(completion_tokens),
        "completion_max": max(completion_tokens),
        "total_mean": np.mean(total_tokens),
        "total_median": np.median(total_tokens),
        "total_max": max(total_tokens),
    }


# ============================================================================
# Dataset Downloaders
# ============================================================================


def download_kernelbook():
    """Download KernelBook from HuggingFace (GPUMODE/KernelBook)."""
    print("\n" + "=" * 60)
    print("Downloading KernelBook...")
    print("=" * 60)

    ds = load_dataset("GPUMODE/KernelBook", split="train")

    # Explore the structure
    print(f"  Columns: {ds.column_names}")
    print(f"  Rows: {len(ds)}")

    # Extract pytorch code and triton code
    # KernelBook has 'python_code' and 'triton_code' columns
    pytorch_codes = []
    completions = []

    for row in ds:
        pytorch_code = row.get("python_code", "")
        triton_code = row.get("triton_code", "")

        if pytorch_code:
            pytorch_codes.append(pytorch_code)
            completions.append(triton_code or "")

    print(f"  Found {len(pytorch_codes)} samples with python_code")
    return analyze_dataset("KernelBook", pytorch_codes, completions)


def download_tritonbench():
    """Download TritonBench from GitHub (thunlp/TritonBench).

    Processes each subset separately:
    - TritonBench_T_v1: PyTorch-crawled test set (166 items)
    - TritonBench_G_v1: GitHub-crawled test set (184 items)
    - train_crawl: Training data crawled from GitHub (4024 items)
    - train_synth: Synthesized training data (4133 items)
    """
    print("\n" + "=" * 60)
    print("Downloading TritonBench...")
    print("=" * 60)

    repo_dir = DATA_DIR / "TritonBench"

    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/thunlp/TritonBench.git",
                str(repo_dir),
            ],
            check=True,
        )

    data_path = repo_dir / "data"
    results = []

    # Process TritonBench_T_v1.jsonl (PyTorch-crawled test set)
    t_v1_file = data_path / "TritonBench_T_v1.jsonl"
    if t_v1_file.exists():
        try:
            with open(t_v1_file) as f:
                data = json.load(f)
            print(f"  TritonBench_T_v1 (test): {len(data)} items")

            pytorch_codes = []
            completions = []
            for item in data:
                torch_code = item.get("torch_code", "")
                example = item.get("example", "")

                if example and "```python" in example:
                    code_blocks = re.findall(r"```python\n(.*?)```", example, re.DOTALL)
                    if code_blocks:
                        pytorch_code = code_blocks[0]
                    else:
                        pytorch_code = torch_code
                else:
                    pytorch_code = torch_code

                if pytorch_code:
                    pytorch_codes.append(pytorch_code)
                    completions.append("")

            if pytorch_codes:
                result = analyze_dataset("TritonBench-T (Test)", pytorch_codes, completions)
                results.append(result)
        except Exception as e:
            print(f"    Error reading TritonBench_T_v1.jsonl: {e}")

    # Process TritonBench_G_v1.json (GitHub-crawled test set)
    # Format: file, repo, simp_instru, comp_instru, output (triton code)
    g_v1_file = data_path / "TritonBench_G_v1.json"
    if g_v1_file.exists():
        try:
            with open(g_v1_file) as f:
                data = json.load(f)
            print(f"  TritonBench_G_v1 (test): {len(data)} items")

            pytorch_codes = []
            completions = []
            for item in data:
                # Use output (triton code) as the reference code
                triton_code = item.get("output", "")
                if triton_code:
                    pytorch_codes.append(triton_code)
                    completions.append("")

            if pytorch_codes:
                result = analyze_dataset("TritonBench-G (Test)", pytorch_codes, completions)
                results.append(result)
        except Exception as e:
            print(f"    Error reading TritonBench_G_v1.json: {e}")

    # Process train_crawl.json (training data crawled from GitHub)
    crawl_file = data_path / "train_crawl.json"
    if crawl_file.exists():
        try:
            with open(crawl_file) as f:
                data = json.load(f)
            print(f"  train_crawl (train): {len(data)} items")

            pytorch_codes = []
            completions = []
            for item in data:
                triton_code = item.get("code", "")
                if triton_code:
                    pytorch_codes.append(triton_code)
                    completions.append("")

            if pytorch_codes:
                result = analyze_dataset("TritonBench-Crawl (Train)", pytorch_codes, completions)
                results.append(result)
        except Exception as e:
            print(f"    Error reading train_crawl.json: {e}")

    # Process train_synth.json (synthesized training data)
    # Format: instruction, input (prompt), output (triton code)
    synth_file = data_path / "train_synth.json"
    if synth_file.exists():
        try:
            with open(synth_file) as f:
                data = json.load(f)
            print(f"  train_synth (train): {len(data)} items")

            pytorch_codes = []
            completions = []
            for item in data:
                # Use output (triton code) as the reference
                triton_code = item.get("output", "")
                if triton_code:
                    pytorch_codes.append(triton_code)
                    completions.append("")

            if pytorch_codes:
                result = analyze_dataset("TritonBench-Synth (Train)", pytorch_codes, completions)
                results.append(result)
        except Exception as e:
            print(f"    Error reading train_synth.json: {e}")

    if not results:
        print("  Warning: No data found in TritonBench repo")
        return None

    return results


def download_kernelbench():
    """Download KernelBench from HuggingFace (ScalingIntelligence/KernelBench)."""
    print("\n" + "=" * 60)
    print("Downloading KernelBench...")
    print("=" * 60)

    # Load level 1 and level 2 from HuggingFace (ignoring level 3 & 4 as per user request)
    ds1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    ds2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2")

    print(f"  Columns: {ds1.column_names}")
    print(f"  Level 1: {len(ds1)} rows")
    print(f"  Level 2: {len(ds2)} rows")

    pytorch_codes = []
    completions = []

    # Process both levels
    for ds in [ds1, ds2]:
        for row in ds:
            code = row.get("code", "")
            if code:
                pytorch_codes.append(code)
                completions.append("")  # No reference completions in this dataset

    print(f"  Total: {len(pytorch_codes)} samples")
    return analyze_dataset("KernelBench", pytorch_codes, completions)


def download_categorized_triton():
    """Download Categorized Triton Data Permissive from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading Categorized Triton Data Permissive...")
    print("=" * 60)

    ds = load_dataset("GPUMODE/categorized_triton_data_permissive", split="train")

    print(f"  Columns: {ds.column_names}")
    print(f"  Rows: {len(ds)}")

    pytorch_codes = []
    completions = []

    for row in ds:
        # This dataset has 'input' column which contains the triton code
        triton_code = row.get("input", "")

        # Use triton code as the "input" (since this is what we're measuring)
        if triton_code:
            pytorch_codes.append(triton_code)
            completions.append("")  # No separate completion

    return analyze_dataset("Categorized Triton Permissive", pytorch_codes, completions)


def download_liger_kernel():
    """Download Liger Kernel from GitHub (linkedin/Liger-Kernel)."""
    print("\n" + "=" * 60)
    print("Downloading Liger Kernel...")
    print("=" * 60)

    repo_dir = DATA_DIR / "Liger-Kernel"

    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/linkedin/Liger-Kernel.git",
                str(repo_dir),
            ],
            check=True,
        )

    pytorch_codes = []
    completions = []

    # Look for triton kernel implementations
    kernel_dirs = [
        repo_dir / "src" / "liger_kernel" / "ops",
        repo_dir / "liger_kernel" / "ops",
        repo_dir / "src",
    ]

    for kernel_dir in kernel_dirs:
        if not kernel_dir.exists():
            continue

        for py_file in kernel_dir.rglob("*.py"):
            try:
                with open(py_file) as f:
                    code = f.read()
                # Look for files with triton.jit decorators
                if "@triton.jit" in code or "triton.language" in code:
                    pytorch_codes.append(code)
                    completions.append("")  # These ARE the triton implementations
            except Exception as e:
                print(f"    Error reading {py_file}: {e}")

    if not pytorch_codes:
        print("  Warning: No data found in Liger-Kernel repo")
        return None

    return analyze_dataset("Liger Kernel", pytorch_codes, completions)


def download_triton_eval_benchmark():
    """Download Triton Kernel Eval Benchmark (ROCm)."""
    print("\n" + "=" * 60)
    print("Downloading Triton Kernel Eval Benchmark...")
    print("=" * 60)

    # This benchmark consists of 30 kernels from various ROCm repositories
    # The data is part of the GEAK paper and uses modified TritonBench-G kernels
    # For now, skip as the 30-kernel benchmark data is not directly available
    # (it's integrated into GEAK agent code, not as a standalone dataset)
    print("  Note: ROCm Triton Eval Benchmark (30 kernels) is part of GEAK - skipping")
    return None


def download_multikernelbench():
    """Download MultiKernelBench from GitHub (wzzll123/MultiKernelBench)."""
    print("\n" + "=" * 60)
    print("Downloading MultiKernelBench...")
    print("=" * 60)

    repo_dir = DATA_DIR / "MultiKernelBench"

    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/wzzll123/MultiKernelBench.git",
                str(repo_dir),
            ],
            check=True,
        )

    pytorch_codes = []
    completions = []

    # MultiKernelBench has reference implementations in reference/ folder
    # organized by category (activation, matmul, normalization, etc.)
    reference_dir = repo_dir / "reference"
    if reference_dir.exists():
        for category_dir in reference_dir.iterdir():
            if not category_dir.is_dir():
                continue
            for py_file in category_dir.glob("*.py"):
                try:
                    with open(py_file) as f:
                        code = f.read()
                    if code.strip() and "torch" in code:
                        pytorch_codes.append(code)
                        completions.append("")  # No reference Triton output
                except Exception as e:
                    print(f"    Error reading {py_file}: {e}")

    if not pytorch_codes:
        print("  Warning: No data found in MultiKernelBench repo")
        return None

    print(f"  Found {len(pytorch_codes)} samples across categories")
    return analyze_dataset("MultiKernelBench", pytorch_codes, completions)


def main():
    """Main function to download and analyze all datasets."""
    print("=" * 60)
    print("Triton/PyTorch Dataset Token Analysis")
    print("=" * 60)

    results = []

    # Download and analyze each dataset
    try:
        result = download_kernelbook()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with KernelBook: {e}")

    try:
        tritonbench_results = download_tritonbench()
        if tritonbench_results:
            # TritonBench returns a list of results (one per subset)
            if isinstance(tritonbench_results, list):
                results.extend(tritonbench_results)
            else:
                results.append(tritonbench_results)
    except Exception as e:
        print(f"  Error with TritonBench: {e}")

    try:
        result = download_kernelbench()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with KernelBench: {e}")

    try:
        result = download_categorized_triton()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with Categorized Triton: {e}")

    try:
        result = download_liger_kernel()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with Liger Kernel: {e}")

    try:
        result = download_triton_eval_benchmark()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with Triton Eval Benchmark: {e}")

    try:
        result = download_multikernelbench()
        if result:
            results.append(result)
    except Exception as e:
        print(f"  Error with MultiKernelBench: {e}")

    # Save summary
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = FIGURES_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 60}")
        print("Summary saved to", summary_path)
        print("=" * 60)
        print(summary_df.to_string())

    print("\n\nDone! Histograms saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
