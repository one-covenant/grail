"""Reward functions for Triton kernel generation environment.

Multi-component reward vector for GRPO training (6 components):
- Compilation (5%): code is valid Python syntax (prerequisite for Triton)
- Structure (10%): has ModelNew class, @triton.jit, proper imports
- GPU Compilation (15%): code compiles and runs on GPU without crashing
- Correctness (50%): passes GPU execution tests (when available)
- Format (10%): proper <SOLUTION> tags with minimal trailing text
- Thinking (10%): presence of reasoning block

Reward hierarchy (natural curriculum):
  Format -> Compilation -> Structure -> GPU Compilation -> Correctness

Without GPU (gpu_eval=False): max reward = 0.35 (compilation + structure + format + thinking)
With GPU (gpu_eval=True): max reward = 1.0

GRPO Design:
- All component bounds are [0.0, 1.0] for clean normalization
- Correctness uses pre-computed test results (no double execution)
- Rewards are deterministic given (completion, context)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from ..base import RewardVector
from ..core import Rubric

logger = logging.getLogger(__name__)

_warned_no_gpu = False

# =============================================================================
# Individual reward functions
# =============================================================================


def _compilation_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for syntactically valid Python code (prerequisite for Triton).

    Returns 1.0 if code compiles, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0
    return 1.0 if parsed.get("syntax_valid", False) else 0.0


def _structure_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for proper Triton kernel structure.

    Checks four structural requirements (0.25 each):
    - ModelNew class defined
    - @triton.jit decorator present
    - triton imported
    - torch imported

    Returns value in [0.0, 1.0].
    """
    if not isinstance(parsed, dict):
        return 0.0

    score = 0.0
    if parsed.get("has_model_new", False):
        score += 0.25
    if parsed.get("has_triton_jit", False):
        score += 0.25
    if parsed.get("has_triton_import", False):
        score += 0.25
    if parsed.get("has_torch_import", False):
        score += 0.25
    return score


def _gpu_compilation_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for successful GPU compilation (code runs without crashing).

    Returns 1.0 if code compiled and executed on GPU, 0.0 otherwise.
    Returns 0.0 when GPU execution is unavailable.
    """
    if not isinstance(parsed, dict):
        return 0.0

    exec_result = parsed.get("exec_result")
    if exec_result is not None and isinstance(exec_result, dict):
        return 1.0 if exec_result.get("compiled", False) else 0.0

    return 0.0


def _correctness_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for GPU execution correctness (pre-computed).

    Uses cached execution results from parsed dict.
    Returns 1.0 if kernel produces correct outputs, 0.0 otherwise.

    When GPU execution is not available, this returns 0.0
    (the environment skips execution and doesn't populate exec_result).
    """
    if not isinstance(parsed, dict):
        return 0.0

    exec_result = parsed.get("exec_result")
    if exec_result is not None and isinstance(exec_result, dict):
        return 1.0 if exec_result.get("correct", False) else 0.0

    return 0.0


def _solution_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for proper <SOLUTION> tag usage.

    Returns 1.0 if tags present with minimal trailing text.
    """
    if not isinstance(parsed, dict):
        return 0.0

    has_solution = parsed.get("has_solution", False)
    trailing = int(parsed.get("trailing_after_solution", 0))

    if has_solution and trailing < 50:
        return 1.0
    return 0.0


def _thinking_format_reward(parsed: dict[str, Any], context: Any) -> float:
    """Reward for having a thinking/reasoning block.

    Returns 1.0 if thinking block present, 0.0 otherwise.
    """
    if not isinstance(parsed, dict):
        return 0.0
    return 1.0 if parsed.get("has_thinking", False) else 0.0


# =============================================================================
# Reward vector factory
# =============================================================================


def create_triton_kernel_reward_vector() -> RewardVector:
    """Create Triton kernel reward vector with 6 decomposed components.

    Components (all bounded [0.0, 1.0]):
        1. Compilation (0.05): Valid Python syntax
        2. Structure (0.10): Proper Triton kernel structure
        3. GPU Compilation (0.15): Code compiles and runs on GPU
        4. Correctness (0.50): GPU execution correctness
        5. Format (0.10): Proper <SOLUTION> tags
        6. Thinking (0.10): Presence of reasoning block

    Total weight: 1.0
    Max achievable reward: 1.0

    Without GPU: max reward is 0.35 (compilation + structure + format + thinking).
    With GPU: max reward is 1.0.
    """
    reward_functions = cast(
        list[Callable[[Any, Any], float]],
        [
            _compilation_reward,
            _structure_reward,
            _gpu_compilation_reward,
            _correctness_reward,
            _solution_format_reward,
            _thinking_format_reward,
        ],
    )
    weights = [0.05, 0.10, 0.15, 0.50, 0.10, 0.10]

    return RewardVector(
        reward_functions,
        weights,
        parser=None,
        bounds=[
            (0.0, 1.0),  # compilation
            (0.0, 1.0),  # structure
            (0.0, 1.0),  # gpu_compilation
            (0.0, 1.0),  # correctness
            (0.0, 1.0),  # format
            (0.0, 1.0),  # thinking
        ],
    )


# =============================================================================
# Rubric adapter
# =============================================================================


class TritonKernelRubric(Rubric):
    """Rubric that computes reward from pre-parsed dict with execution results.

    Avoids double-execution by using cached exec_result from parsed dict.
    """

    def __init__(self) -> None:
        self._reward_vector = create_triton_kernel_reward_vector()

    def step_reward(
        self, *, parsed: Any, context: Any, turn_index: int
    ) -> tuple[float, dict[str, float]]:
        if not isinstance(parsed, dict):
            return 0.0, {}

        try:
            rewards = []
            for fn in self._reward_vector.reward_functions:
                rewards.append(fn(parsed, context))

            total = sum(
                r * w
                for r, w in zip(rewards, self._reward_vector.weights, strict=False)
            )

            components = {
                "compilation": rewards[0],
                "structure": rewards[1],
                "gpu_compilation": rewards[2],
                "correctness": rewards[3],
                "format": rewards[4],
                "thinking": rewards[5],
            }

            # Log warning if no GPU eval on first call
            global _warned_no_gpu
            if not _warned_no_gpu and parsed.get("exec_result") is None:
                _warned_no_gpu = True
                logger.info(
                    "Kernel rewards: gpu_eval not active, max reward = 0.35 "
                    "(compilation + structure + format + thinking)"
                )

            return float(total), components
        except Exception:
            return 0.0, {}
