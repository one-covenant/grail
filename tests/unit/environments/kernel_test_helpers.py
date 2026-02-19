"""Test helpers for GPU kernel environment tests."""

from __future__ import annotations

# Sample PyTorch code (simple enough for tests, matches KernelBench format)
SAMPLE_PYTORCH_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(4, 4)]

def get_init_inputs():
    return []
'''

# Valid Triton completion
VALID_TRITON_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    x = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, tl.maximum(x, 0.0))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = torch.empty_like(x)
        relu_kernel[(x.numel(),)](x, out, x.numel())
        return out
'''

# Structurally invalid (missing ModelNew)
MISSING_MODEL_NEW_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n: tl.constexpr):
    idx = tl.program_id(0)
    x = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, tl.maximum(x, 0.0))
'''

# Missing triton imports but has ModelNew
NO_TRITON_IMPORT_CODE = '''
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
'''

SYNTAX_ERROR_CODE = "def broken(:\n    return"


def build_kernel_completion(thinking: str, code: str, trailing: str = "") -> str:
    """Build properly formatted kernel completion with thinking and solution tags."""
    from grail.shared.thinking import get_thinking_config

    cfg = get_thinking_config()
    return (
        f"{cfg.thinking_open}\n{thinking}\n{cfg.thinking_close}\n"
        f"{cfg.solution_open}\n{code}\n{cfg.solution_close}{trailing}"
    )


def make_task_payload(
    pytorch_code: str = SAMPLE_PYTORCH_CODE,
    test_code: str | None = None,
) -> dict:
    """Create a task payload dict matching TaskSpec.payload structure."""
    return {
        "pytorch_code": pytorch_code,
        "test_code": test_code or pytorch_code,  # Fallback to pytorch_code
        "problem_name": "test_problem",
    }
