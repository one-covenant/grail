#!/usr/bin/env python3
"""
Combine Triton/PyTorch datasets into a unified format.

Target columns:
- instruction: verbal description of the kernel to be developed
- pytorch_code: reference PyTorch code
- triton_code: reference Triton implementation
- dataset_name: source dataset name
"""

import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "raw"
OUTPUT_DIR = Path(__file__).parent


def load_kernelbook() -> list[dict]:
    """
    KernelBook (GPUMODE/KernelBook)

    Columns used:
    - instruction: None (no instruction available)
    - pytorch_code: 'python_code'
    - triton_code: 'triton_code'
    """
    print("Loading KernelBook...")
    ds = load_dataset("GPUMODE/KernelBook", split="train")

    rows = []
    for item in tqdm(ds, desc="  Processing"):
        rows.append(
            {
                "instruction": None,
                "pytorch_code": item.get("python_code") or None,
                "triton_code": item.get("triton_code") or None,
                "dataset_name": "KernelBook",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_kernelbench() -> list[dict]:
    """
    KernelBench (ScalingIntelligence/KernelBench)

    Columns used:
    - instruction: 'name' (problem name as basic description)
    - pytorch_code: 'code'
    - triton_code: None (benchmark doesn't include triton references)
    """
    print("Loading KernelBench (levels 1 & 2)...")
    ds1 = load_dataset("ScalingIntelligence/KernelBench", split="level_1")
    ds2 = load_dataset("ScalingIntelligence/KernelBench", split="level_2")

    rows = []
    for ds in [ds1, ds2]:
        for item in tqdm(ds, desc="  Processing"):
            rows.append(
                {
                    "instruction": item.get("name") or None,
                    "pytorch_code": item.get("code") or None,
                    "triton_code": None,
                    "dataset_name": "KernelBench",
                }
            )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_categorized_triton() -> list[dict]:
    """
    Categorized Triton Data Permissive (GPUMODE/categorized_triton_data_permissive)

    Columns used:
    - instruction: None (no instruction available)
    - pytorch_code: None (no PyTorch code)
    - triton_code: 'input' (contains the triton code)
    """
    print("Loading Categorized Triton Permissive...")
    ds = load_dataset("GPUMODE/categorized_triton_data_permissive", split="train")

    rows = []
    for item in tqdm(ds, desc="  Processing"):
        rows.append(
            {
                "instruction": None,
                "pytorch_code": None,
                "triton_code": item.get("input") or None,
                "dataset_name": "CategorizedTritonPermissive",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_tritonbench_t() -> list[dict]:
    """
    TritonBench-T (PyTorch-crawled test set)

    Columns used:
    - instruction: 'description' + 'func_inputs' (combined description)
    - pytorch_code: 'torch_code' (compact form) or extracted from 'example'
    - triton_code: None (test set - no ground truth triton)
    """
    print("Loading TritonBench-T (Test)...")
    file_path = DATA_DIR / "TritonBench" / "data" / "TritonBench_T_v1.jsonl"

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return []

    with open(file_path) as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc="  Processing"):
        # Build instruction from description and function inputs
        description = item.get("description", "")
        func_inputs = item.get("func_inputs", "")
        instruction = (
            f"{description}\n\nFunction signature: {func_inputs}".strip() if description else None
        )

        # Get PyTorch code - prefer torch_code, fallback to example
        pytorch_code = item.get("torch_code") or None

        rows.append(
            {
                "instruction": instruction,
                "pytorch_code": pytorch_code,
                "triton_code": None,  # Test set - no ground truth
                "dataset_name": "TritonBench-T",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_tritonbench_g() -> list[dict]:
    """
    TritonBench-G (GitHub-crawled test set)

    Columns used:
    - instruction: 'comp_instru' (comprehensive instruction, more detailed)
    - pytorch_code: None (GitHub-crawled, no PyTorch reference)
    - triton_code: 'output' (ground truth triton code)
    """
    print("Loading TritonBench-G (Test)...")
    file_path = DATA_DIR / "TritonBench" / "data" / "TritonBench_G_v1.json"

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return []

    with open(file_path) as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc="  Processing"):
        # Use comprehensive instruction (more detailed than simple)
        instruction = item.get("comp_instru") or item.get("simp_instru") or None

        rows.append(
            {
                "instruction": instruction.strip() if instruction else None,
                "pytorch_code": None,
                "triton_code": item.get("output") or None,
                "dataset_name": "TritonBench-G",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_tritonbench_train_crawl() -> list[dict]:
    """
    TritonBench train_crawl (GitHub-crawled training data)

    Columns used:
    - instruction: 'description_1' (primary description)
    - pytorch_code: None (crawled triton code only)
    - triton_code: 'code' (crawled triton implementation)
    """
    print("Loading TritonBench train_crawl...")
    file_path = DATA_DIR / "TritonBench" / "data" / "train_crawl.json"

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return []

    with open(file_path) as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc="  Processing"):
        # Use description_1 as primary instruction
        instruction = item.get("description_1") or item.get("description_2") or None

        rows.append(
            {
                "instruction": instruction,
                "pytorch_code": None,
                "triton_code": item.get("code") or None,
                "dataset_name": "TritonBench-Crawl",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def load_tritonbench_train_synth() -> list[dict]:
    """
    TritonBench train_synth (Synthesized training data)

    Columns used:
    - instruction: 'input' (contains the instruction/prompt)
    - pytorch_code: None (synthesized data, no PyTorch reference)
    - triton_code: 'output' (synthesized triton implementation)
    """
    print("Loading TritonBench train_synth...")
    file_path = DATA_DIR / "TritonBench" / "data" / "train_synth.json"

    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return []

    with open(file_path) as f:
        data = json.load(f)

    rows = []
    for item in tqdm(data, desc="  Processing"):
        # 'input' contains the instruction/prompt
        instruction = item.get("input") or None

        rows.append(
            {
                "instruction": instruction,
                "pytorch_code": None,
                "triton_code": item.get("output") or None,
                "dataset_name": "TritonBench-Synth",
            }
        )

    print(f"  Loaded {len(rows)} samples")
    return rows


def main():
    """Load all datasets and combine into unified format."""
    print("=" * 60)
    print("Combining Triton/PyTorch Datasets")
    print("=" * 60)

    all_rows = []

    # Load each dataset
    loaders = [
        load_kernelbook,
        load_kernelbench,
        load_categorized_triton,
        load_tritonbench_t,
        load_tritonbench_g,
        load_tritonbench_train_crawl,
        load_tritonbench_train_synth,
    ]

    for loader in loaders:
        try:
            rows = loader()
            all_rows.extend(rows)
        except Exception as e:
            print(f"  Error loading {loader.__name__}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print("\nSamples per dataset:")
    print(df["dataset_name"].value_counts().to_string())
    print("\nColumn coverage:")
    print(
        f"  - instruction: {df['instruction'].notna().sum()} ({df['instruction'].notna().mean() * 100:.1f}%)"
    )
    print(
        f"  - pytorch_code: {df['pytorch_code'].notna().sum()} ({df['pytorch_code'].notna().mean() * 100:.1f}%)"
    )
    print(
        f"  - triton_code: {df['triton_code'].notna().sum()} ({df['triton_code'].notna().mean() * 100:.1f}%)"
    )

    # Save as parquet (efficient for large datasets)
    parquet_path = OUTPUT_DIR / "combined_triton_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nSaved to: {parquet_path}")

    # Also save as HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df)
    hf_path = OUTPUT_DIR / "combined_triton_dataset_hf"
    hf_dataset.save_to_disk(str(hf_path))
    print(f"Saved HuggingFace dataset to: {hf_path}")

    # Print detailed mapping report
    print("\n" + "=" * 60)
    print("Column Mapping Report")
    print("=" * 60)

    report = """
Dataset                      | instruction           | pytorch_code    | triton_code
-----------------------------|----------------------|-----------------|------------------
KernelBook                   | None                 | python_code     | triton_code
KernelBench                  | name                 | code            | None
CategorizedTritonPermissive  | None                 | None            | input
TritonBench-T                | description+func_in  | torch_code      | None
TritonBench-G                | comp_instru          | None            | output
TritonBench-Crawl            | description_1        | None            | code
TritonBench-Synth            | input                | None            | output

Notes:
- KernelBook: High-quality PyTorch-to-Triton pairs, no natural language instructions
- KernelBench: PyTorch benchmarks with problem names as basic instructions, no Triton ground truth
- CategorizedTritonPermissive: Triton code only, categorized by operation type
- TritonBench-T: Test set with detailed descriptions, PyTorch reference, no Triton ground truth
- TritonBench-G: Test set with comprehensive instructions and Triton ground truth
- TritonBench-Crawl: Training data with descriptions and Triton code
- TritonBench-Synth: Synthesized training data with prompts and Triton outputs
"""
    print(report)

    return df


if __name__ == "__main__":
    main()
