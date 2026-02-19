"""Parser for Triton kernel code completions.

Extracts and validates Triton kernel code from model completions.
Checks for structural requirements: ModelNew class, @triton.jit kernels, proper imports.
"""

from __future__ import annotations

import re
from typing import Any

from ..base import ThinkingParser


class TritonKernelParser(ThinkingParser):
    """Parser for Triton kernel completions with structural validation.

    Inherits thinking tag detection from ThinkingParser base class.
    Extracts Triton kernel code from <SOLUTION>...</SOLUTION> tags and validates:
    - Python syntax validity (Triton kernels are Python code)
    - Presence of ModelNew class (required by KernelBench)
    - Presence of @triton.jit decorated functions
    - Proper imports (torch, triton)

    Expected completion format (tags depend on thinking mode):
        <thinking_open>reasoning about optimization strategy...<thinking_close>
        <SOLUTION>
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def my_kernel(...):
            ...

        class ModelNew(nn.Module):
            def forward(self, ...):
                ...
        </SOLUTION>
    """

    def parse(self, completion: str, context: Any) -> dict[str, Any]:
        """Parse completion for thinking tags, Triton code, and structural validity.

        Returns dict with:
            - code: extracted code from <SOLUTION> tags (empty string if none)
            - has_thinking: bool, True if thinking block present
            - has_solution: bool, True if <SOLUTION> tags present
            - trailing_after_solution: int, chars after </SOLUTION>
            - syntax_valid: bool, True if code compiles without SyntaxError
            - has_model_new: bool, True if ModelNew class defined
            - has_triton_jit: bool, True if @triton.jit decorator found
            - has_triton_import: bool, True if triton is imported
            - has_torch_import: bool, True if torch is imported
            - structure_valid: bool, True if all structural checks pass
        """
        if completion is None:
            text = ""
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)

        # Use inherited methods from ThinkingParser
        has_thinking = self._detect_thinking_block(text)
        has_solution = self._detect_answer_block(text)

        # Defaults
        code = ""
        trailing_after_solution = 0
        syntax_valid = False
        has_model_new = False
        has_triton_jit = False
        has_triton_import = False
        has_torch_import = False

        if has_solution:
            try:
                content, trailing, _ = self._get_answer_with_thinking_check(text)
                if content is not None:
                    code = content.strip()
                    trailing_after_solution = max(0, trailing)

                    if code:
                        # Check Python syntax (Triton is Python)
                        try:
                            compile(code, "<triton_kernel>", "exec")
                            syntax_valid = True
                        except SyntaxError:
                            syntax_valid = False

                        # Structural checks (work even if syntax is invalid)
                        has_model_new = _check_has_model_new(code)
                        has_triton_jit = _check_has_triton_jit(code)
                        has_triton_import = _check_has_triton_import(code)
                        has_torch_import = _check_has_torch_import(code)
            except Exception:
                code = ""
                trailing_after_solution = 0

        structure_valid = (
            syntax_valid
            and has_model_new
            and has_triton_jit
            and has_triton_import
            and has_torch_import
        )

        return {
            "code": code,
            "has_thinking": has_thinking,
            "has_solution": has_solution,
            "trailing_after_solution": trailing_after_solution,
            "syntax_valid": syntax_valid,
            "has_model_new": has_model_new,
            "has_triton_jit": has_triton_jit,
            "has_triton_import": has_triton_import,
            "has_torch_import": has_torch_import,
            "structure_valid": structure_valid,
        }


# =============================================================================
# Structural validation helpers
# =============================================================================

# Pattern for class ModelNew definition
_MODEL_NEW_PATTERN = re.compile(
    r"class\s+ModelNew\s*\(", re.MULTILINE
)

# Pattern for @triton.jit decorator
_TRITON_JIT_PATTERN = re.compile(
    r"@triton\.jit", re.MULTILINE
)

# Pattern for triton imports
_TRITON_IMPORT_PATTERN = re.compile(
    r"(?:import\s+triton|from\s+triton)", re.MULTILINE
)

# Pattern for torch imports
_TORCH_IMPORT_PATTERN = re.compile(
    r"(?:import\s+torch|from\s+torch)", re.MULTILINE
)


def _check_has_model_new(code: str) -> bool:
    """Check if code defines a ModelNew class."""
    return bool(_MODEL_NEW_PATTERN.search(code))


def _check_has_triton_jit(code: str) -> bool:
    """Check if code has @triton.jit decorated functions."""
    return bool(_TRITON_JIT_PATTERN.search(code))


def _check_has_triton_import(code: str) -> bool:
    """Check if code imports triton."""
    return bool(_TRITON_IMPORT_PATTERN.search(code))


def _check_has_torch_import(code: str) -> bool:
    """Check if code imports torch."""
    return bool(_TORCH_IMPORT_PATTERN.search(code))


def extract_anti_hacking_signals(code: str) -> dict[str, bool]:
    """Extract signals that may indicate reward hacking attempts.

    Based on documented hacking patterns from Kevin, Dr. Kernel, CUDA-L1:
    - Using torch.nn directly instead of Triton kernels
    - Wrapping code in try-except with PyTorch fallback
    - Using pass in class body (inheriting reference implementation)
    - Not actually calling any Triton kernel in forward()

    Returns dict of boolean flags for each check.
    """
    checks: dict[str, bool] = {}

    # Check: does forward() call any triton-related function?
    # If ModelNew.forward just calls torch operations, it's not using Triton
    checks["forward_calls_triton"] = bool(
        re.search(r"def\s+forward\s*\(.*?\).*?(?:triton_|_kernel|_launch)", code, re.DOTALL)
    )

    # Check: try-except wrapping (potential PyTorch fallback)
    checks["has_try_except"] = bool(
        re.search(r"try\s*:.*?except", code, re.DOTALL)
    )

    # Check: class body is just 'pass'
    checks["model_is_pass_only"] = bool(
        re.search(r"class\s+ModelNew.*?:\s*\n\s+pass\s*$", code, re.MULTILINE)
    )

    # Check: uses torch.nn.functional in forward (potential passthrough)
    checks["uses_nn_functional_in_forward"] = bool(
        re.search(
            r"def\s+forward\s*\(.*?\).*?(?:torch\.nn\.functional|F\.)",
            code,
            re.DOTALL,
        )
    )

    return checks
