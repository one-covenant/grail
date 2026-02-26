"""Single-turn Triton kernel generation environment using KernelBench.

This environment serves GPU kernel optimization problems and evaluates
generated Triton kernels by:
- Parsing code from <SOLUTION> tags
- Validating Python syntax and Triton structure (ModelNew, @triton.jit, imports)
- Optionally executing on GPU via pluggable eval backend to verify correctness
- Computing rewards based on compilation, structure, gpu_compilation, correctness, and format

Key features:
- KernelBench-backed (250 problems across 4 difficulty levels)
- Unified kernel dataset support (10K+ rows from JSONL)
- Structural validation works without GPU (provides training signal)
- GPU correctness checking via pluggable backends (subprocess, affinetes, modal)
- Anti-reward-hacking detection signals
- Decomposed reward: compilation, structure, gpu_compilation, correctness, format, thinking

Expected completion format (tags depend on thinking mode):
    <thinking_open>
    Analysis of the PyTorch operations and optimization strategy...
    <thinking_close>
    <SOLUTION>
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def optimized_kernel(...):
        ...

    class ModelNew(nn.Module):
        def forward(self, ...):
            ...
    </SOLUTION>

RLVR/GRPO Design Principles:
- Rewards are deterministic given (completion, context)
- GPU execution happens at most once per step
- Reward bounds are explicit and achievable (max=1.0)
- Parser extracts all features in a single pass
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from ..base import Parser
from ..core import ChatMessage, Observation, Rubric, SingleTurnEnv, TaskSource
from ..providers import TaskSpec
from .eval_backends import KernelEvalBackend, get_global_backend
from .parser import TritonKernelParser, extract_anti_hacking_signals
from .rewards import TritonKernelRubric
from .task_sources import KernelBenchTaskSource

logger = logging.getLogger(__name__)

# Prompt template based on KernelBench/KernelLLM format with one-shot example
_PROMPT_TEMPLATE = """\
You write custom Triton kernels to replace the PyTorch operators in the given \
architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. \
You may replace multiple operators with custom implementations, consider \
operator fusion opportunities (combining multiple operators into a single \
kernel, for example, combining matmul+relu), or algorithmic changes (such as \
online softmax). You are only limited by your imagination.

Here is an example of how to write a Triton kernel to replace a PyTorch operator:

Input architecture:
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(4, 4), torch.randn(4, 4)]

def get_init_inputs():
    return []
```

Output:
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        output = torch.empty_like(a)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=1024)
        return output
```

Now, you are given the following architecture:

```python
{pytorch_code}
```

Optimize the architecture named Model with custom Triton kernels! \
Name your optimized output architecture ModelNew. Output the new code. \
Please generate real code, NOT pseudocode, make sure the code compiles and \
is fully functional. Just output the new model code, no other text, and \
NO testing code (no if __name__ == '__main__')!

Your output MUST:
1. Include all necessary imports (torch, torch.nn, triton, triton.language)
2. Define at least one @triton.jit kernel function
3. Define a class called ModelNew that inherits from nn.Module
4. Implement a forward() method that uses your custom Triton kernel(s)

IMPORTANT: Wrap your final code in <SOLUTION> and </SOLUTION> tags. Code outside these tags is ignored."""


class TritonKernelEnv(SingleTurnEnv):
    """Single-turn Triton kernel generation environment.

    Serves kernel optimization problems from KernelBench or unified dataset
    and evaluates generated Triton kernels for structural validity and
    optionally GPU correctness via pluggable eval backends.

    Difficulty levels (KernelBench):
        - Level 1: Single operators (100 problems) - matmul, conv, relu, etc.
        - Level 2: Fusion patterns (100 problems) - Conv+Bias+ReLU, etc.
        - Level 3: Complete architectures (50 problems) - MobileNet, VGG, etc.
        - Level 4: HuggingFace models (20 problems)

    Reward components:
        - Compilation (5%): Valid Python syntax
        - Structure (10%): ModelNew class, @triton.jit, imports
        - GPU Compilation (15%): Code compiles and runs on GPU
        - Correctness (50%): GPU execution matches reference
        - Format (10%): Proper <SOLUTION> tags
        - Thinking (10%): Reasoning block present

    Example:
        env = TritonKernelEnv(task_source=source, gpu_eval=True, eval_backend=backend)
        obs = env.reset(seed=42)
        obs, reward, done, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )
    """

    def __init__(
        self,
        *,
        split: str = "train",
        level: int | None = None,
        task_source: TaskSource | None = None,
        parser: Parser | None = None,
        rubric: Rubric | None = None,
        gpu_eval: bool = False,
        eval_backend: KernelEvalBackend | None = None,
    ):
        """Initialize Triton kernel generation environment.

        Args:
            split: Dataset split ('train', 'val', or 'all')
            level: KernelBench difficulty level (1-4, or None for all)
            task_source: Custom task source (overrides split/level)
            parser: Custom parser (defaults to TritonKernelParser)
            rubric: Custom rubric (defaults to TritonKernelRubric)
            gpu_eval: Whether to run GPU execution for correctness checking.
                      Requires a configured eval backend. Default False for safety.
            eval_backend: Eval backend instance (if None, falls back to global backend).
        """
        super().__init__()

        self._split = split
        self._level = level
        self._gpu_eval = gpu_eval
        self._eval_backend = eval_backend

        if task_source is None:
            task_source = KernelBenchTaskSource(split=split, level=level)

        self._source = task_source
        self._parser = parser or TritonKernelParser()
        self._rubric = rubric or TritonKernelRubric()
        self._task: TaskSpec | None = None

    def _do_reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Reset environment and sample a new kernel optimization problem.

        Returns initial observation with the PyTorch code to optimize.
        """
        self._task = self._source.next(seed=seed, task_id=task_id)
        assert self._task is not None, "TaskSource.next() must return a TaskSpec"

        # Build prompt from template
        pytorch_code = self._task.payload["pytorch_code"]
        prompt = _PROMPT_TEMPLATE.format(pytorch_code=pytorch_code)

        obs = Observation(
            messages=[ChatMessage(role="user", content=prompt)],
            available_tools=[],
            turn_index=0,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )
        return obs

    def _do_step(self, action: ChatMessage) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute single turn: parse Triton code, validate, optionally run on GPU.

        Steps:
        1. Parse completion (extract code, check syntax, detect tags)
        2. Validate Triton structure (ModelNew, @triton.jit, imports)
        3. Optionally execute on GPU via eval backend
        4. Compute multi-component reward
        5. Run anti-hacking detection (informational only)

        Returns:
            (observation, reward, truncated, info)
        """
        assert self._task is not None, "Must call reset() before step()"

        completion_text = action.content if action.content is not None else ""

        # Step 1-2: Parse and validate structure
        parsed = self._parser.parse(completion_text, self._task.payload)
        code = parsed.get("code", "") or ""

        # Step 3: GPU execution via eval backend (optional)
        exec_result: dict[str, Any] | None = None
        if self._gpu_eval and code and parsed.get("structure_valid", False):
            backend = self._eval_backend or get_global_backend()
            if backend is None:
                raise RuntimeError(
                    "gpu_eval=True but no eval backend configured. "
                    "Set up a backend via set_global_backend() or pass "
                    "eval_backend to TritonKernelEnv."
                )

            test_code = self._task.payload.get("test_code", "")
            if not test_code:
                exec_result = {
                    "correct": False,
                    "compiled": False,
                    "error": "missing_test_code",
                }
            else:
                t0 = time.monotonic()
                result = backend.evaluate(test_code, code)
                eval_duration = time.monotonic() - t0
                exec_result = asdict(result)
                logger.info(
                    "GPU kernel eval task=%s: compiled=%s correct=%s error=%s duration=%.2fs",
                    self._task.id,
                    result.compiled,
                    result.correct,
                    result.error,
                    eval_duration,
                )

        # Augment parsed dict with execution results
        parsed_with_exec = {
            **parsed,
            "exec_result": exec_result,
        }

        # Step 4: Compute reward
        reward, components = self._rubric.step_reward(
            parsed=parsed_with_exec,
            context=self._task.payload,
            turn_index=1,
        )

        # Step 5: Anti-hacking signals (informational, not used in reward)
        hacking_signals: dict[str, bool] = {}
        if code:
            hacking_signals = extract_anti_hacking_signals(code)

        # Determine success
        if exec_result is not None:
            success = bool(exec_result.get("correct", False))
        else:
            # Without GPU, success = structure is valid
            success = bool(parsed.get("structure_valid", False))

        # Build final observation
        obs = Observation(
            messages=[
                ChatMessage(role="user", content=self._task.payload["pytorch_code"]),
                ChatMessage(role="assistant", content=completion_text),
            ],
            available_tools=[],
            turn_index=1,
            task_meta={"task_id": self._task.id, **self._task.metadata},
        )

        info: dict[str, Any] = {
            "reward_components": components,
            "termination_cause": "final",
            "success": success,
            "has_code": bool(code),
            "syntax_valid": parsed.get("syntax_valid", False),
            "structure_valid": parsed.get("structure_valid", False),
            "has_model_new": parsed.get("has_model_new", False),
            "has_triton_jit": parsed.get("has_triton_jit", False),
            "gpu_eval": self._gpu_eval,
            "exec_result": exec_result,
            "hacking_signals": hacking_signals,
        }

        return obs, float(reward), False, info
