"""KernelBook Dataset Analysis

This script loads the GPUMODE/KernelBook dataset and explores its structure.

Dataset Overview:
- 18,162 PyTorch → Triton code pairs
- Sourced from real GitHub repositories
- Includes metadata: licenses, stars, repo links

Test Infrastructure:
- get_inputs(): Returns sample input tensors (100% of samples)
- get_init_inputs(): Returns module initialization arguments (100% of samples)
- No assert statements or test frameworks included
"""

import importlib.util
import traceback

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset

# ============================================================================
# Load KernelBook Dataset
# ============================================================================

print("Loading KernelBook dataset...")
ds = load_dataset("GPUMODE/KernelBook", split="train")
print(f"Loaded {len(ds)} samples")

# Convert to pandas DataFrame
df = ds.to_pandas()

# Display basic info
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)

# Preview first few rows (metadata columns only)
print("\nFirst 10 rows (metadata):")
print(df[["entry_point", "repo_name", "module_name", "synthetic", "licenses", "stars"]].head(10))

# ============================================================================
# Export Options (uncomment to use)
# ============================================================================

# Save to parquet (recommended for large datasets with code)
# df.to_parquet("kernelbook.parquet", index=False)

# Save to CSV (warning: code columns may have formatting issues)
# df.to_csv("kernelbook.csv", index=False)

# Save just metadata
# df[["entry_point", "repo_name", "module_name", "synthetic", "licenses", "stars", "sha", "repo_link"]].to_csv("kernelbook_metadata.csv", index=False)

print("Uncomment above lines to export the DataFrame")


# ============================================================================
# Helper Functions for Testing
# ============================================================================


def test_module_from_code(
    python_code: str, module_name: str = "TestModule"
) -> tuple[torch.nn.Module, list, torch.Tensor]:
    """Execute Python code and test the module using get_inputs() and get_init_inputs().

    Args:
        python_code: The Python code to execute
        module_name: Name for the module namespace

    Returns:
        Tuple of:
        - module instance
        - test inputs
        - forward pass output
    """
    # Create a module namespace
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)

    # Execute the code in the module namespace
    exec(python_code, module.__dict__)

    # Get the module class (first class found that's a nn.Module)
    module_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj != torch.nn.Module:
            module_class = obj
            break

    if module_class is None:
        raise ValueError("No nn.Module class found in code")

    # Get initialization args
    init_args, init_kwargs = module.get_init_inputs()

    # Instantiate module
    instance = module_class(*init_args, **init_kwargs)

    # Get test inputs
    test_inputs = module.get_inputs()

    # Run forward pass
    with torch.no_grad():
        output = instance(*test_inputs)

    return instance, test_inputs, output


def compare_pytorch_vs_triton(
    row_idx: int, rtol: float = 1e-5, atol: float = 1e-8
) -> tuple[bool, torch.Tensor, torch.Tensor, float]:
    """Compare PyTorch and Triton implementations from a dataset row.

    Args:
        row_idx: Index of the row to compare
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of:
        - True if outputs match within tolerance
        - PyTorch output
        - Triton output
        - Max difference
    """
    row = df.iloc[row_idx]

    # Execute PyTorch code
    pytorch_code = row["python_code"]
    pytorch_module, inputs, pytorch_output = test_module_from_code(
        pytorch_code, f"PyTorch_{row['entry_point']}"
    )

    # Execute Triton code (extract the call function)
    triton_code = row["triton_code"]

    # Create a namespace for Triton code
    triton_namespace = {"torch": torch}
    exec(triton_code, triton_namespace)

    # Get the call function
    if "call" not in triton_namespace:
        raise ValueError("Triton code does not contain 'call' function")

    triton_call = triton_namespace["call"]

    # Run Triton forward pass
    with torch.no_grad():
        triton_output = triton_call(inputs)[0]  # call returns tuple

    # Compare outputs
    if pytorch_output.shape != triton_output.shape:
        return False, pytorch_output, triton_output, float("inf")

    max_diff = (pytorch_output - triton_output).abs().max().item()
    is_close = torch.allclose(pytorch_output, triton_output, rtol=rtol, atol=atol)

    return is_close, pytorch_output, triton_output, max_diff


# ============================================================================
# Test Module Functionality
# ============================================================================

print("\n" + "=" * 80)
print("Testing Module from Code")
print("=" * 80)

# Test with example 0
example_code = df.iloc[0]["python_code"]
try:
    module, inputs, output = test_module_from_code(example_code, "SumAggregator")
    print(f"✅ Successfully tested: {df.iloc[0]['entry_point']}")
    print(f"   Input shape: {inputs[0].shape}")
    print(f"   Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Error: {e}")


# ============================================================================
# PyTorch vs Triton Comparison
# ============================================================================

print("\n" + "=" * 80)
print("PyTorch vs Triton Comparison")
print("=" * 80)

try:
    match, pytorch_out, triton_out, max_diff = compare_pytorch_vs_triton(0)
    print(f"✅ PyTorch vs Triton comparison for {df.iloc[0]['entry_point']}:")
    print(f"   Match: {match}")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   PyTorch output shape: {pytorch_out.shape}")
    print(f"   Triton output shape: {triton_out.shape}")
except Exception as e:
    print(f"❌ Error comparing: {e}")
    traceback.print_exc()


# ============================================================================
# Code Statistics Analysis
# ============================================================================

print("\n" + "=" * 80)
print("Code Statistics")
print("=" * 80)

# Add columns for code lengths (in characters)
df["python_code_len"] = df["python_code"].str.len()
df["triton_code_len"] = df["triton_code"].str.len()
df["original_triton_code_len"] = df["original_triton_code"].str.len()

# Summary statistics
print("\nCode Length Statistics:")
print(df[["python_code_len", "triton_code_len", "original_triton_code_len"]].describe())

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# PyTorch code length distribution
axes[0].hist(df["python_code_len"], bins=50, edgecolor="black", alpha=0.7, color="blue")
axes[0].set_title("PyTorch Code Length (chars)")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Frequency")
axes[0].axvline(
    df["python_code_len"].median(),
    color="red",
    linestyle="--",
    label=f"Median: {df['python_code_len'].median():.0f}",
)
axes[0].legend()

# Triton code length distribution
axes[1].hist(df["triton_code_len"], bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_title("Triton Code Length (chars)")
axes[1].set_xlabel("Characters")
axes[1].set_ylabel("Frequency")
axes[1].axvline(
    df["triton_code_len"].median(),
    color="red",
    linestyle="--",
    label=f"Median: {df['triton_code_len'].median():.0f}",
)
axes[1].legend()

# Code expansion ratio (Triton / PyTorch)
df["expansion_ratio"] = df["triton_code_len"] / df["python_code_len"]
axes[2].hist(df["expansion_ratio"], bins=50, edgecolor="black", alpha=0.7, color="green")
axes[2].set_title("Code Expansion Ratio (Triton/PyTorch)")
axes[2].set_xlabel("Ratio")
axes[2].set_ylabel("Frequency")
axes[2].axvline(
    df["expansion_ratio"].median(),
    color="red",
    linestyle="--",
    label=f"Median: {df['expansion_ratio'].median():.1f}x",
)
axes[2].legend()

plt.tight_layout()
plt.show()


# ============================================================================
# Synthetic vs Real Sample Analysis
# ============================================================================

print("\n" + "=" * 80)
print("Synthetic vs Real Samples")
print("=" * 80)

synthetic_counts = df["synthetic"].value_counts()
print("Synthetic vs Real:")
print(
    f"  Real code: {synthetic_counts.get(False, 0)} "
    f"({synthetic_counts.get(False, 0) / len(df) * 100:.1f}%)"
)
print(
    f"  Synthetic: {synthetic_counts.get(True, 0)} "
    f"({synthetic_counts.get(True, 0) / len(df) * 100:.1f}%)"
)


# ============================================================================
# Kernel Type Distribution Analysis
# ============================================================================

print("\n" + "=" * 80)
print("Kernel Type Distribution")
print("=" * 80)

# Count unique kernel types
print(f"Total samples: {len(df)}")
print(f"Unique kernel types (entry_points): {df['entry_point'].nunique()}")

# Count occurrences of each kernel type
kernel_counts = df["entry_point"].value_counts()

print("\n=== Top 30 Most Common Kernel Types ===")
for i, (name, count) in enumerate(kernel_counts.head(30).items(), 1):
    print(f"{i:3}. {name}: {count}")

print("\n=== Kernel Type Distribution ===")
print(f"Kernels appearing once: {(kernel_counts == 1).sum()}")
print(f"Kernels appearing 2-5 times: {((kernel_counts >= 2) & (kernel_counts <= 5)).sum()}")
print(f"Kernels appearing 6-10 times: {((kernel_counts >= 6) & (kernel_counts <= 10)).sum()}")
print(f"Kernels appearing 11-50 times: {((kernel_counts >= 11) & (kernel_counts <= 50)).sum()}")
print(f"Kernels appearing 51+ times: {(kernel_counts > 50).sum()}")
