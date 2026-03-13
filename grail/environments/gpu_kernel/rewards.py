"""Reward functions for Triton kernel generation environment.

Sigmoid reward formulation (GPT-5 style):
  R = 0.80 * kernel_quality + 0.10 * format + 0.10 * thinking

Where kernel_quality = sigmoid(1{correct} + min(speedup, 3.0) - 1.8):
  - Incorrect kernel: 0.0 (gated)
  - Correct, no timing: sigmoid(1 - 1.8) = 0.310
  - Correct, 1x speedup: sigmoid(1 + 1 - 1.8) = 0.550
  - Correct, 2x speedup: sigmoid(1 + 2 - 1.8) = 0.769
  - Correct, 3x+ speedup: sigmoid(1 + 3 - 1.8) = 0.900

Without GPU (gpu_eval=False): max reward = 0.20 (format + thinking only)
With GPU (gpu_eval=True, correct, 3x speedup): max reward = 0.92

GRPO Design:
- Continuous speedup signal prevents reward collapse once correctness is learned
- Sigmoid diminishing returns avoid reward hacking via trivial speedups
- Format/thinking bonuses are independent of correctness gate
- Rewards are deterministic given (completion, context)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ...shared.constants import SIGMOID_DELTA, SIGMOID_KERNEL_WEIGHT, SPEEDUP_CLIP
from ..core import Rubric

logger = logging.getLogger(__name__)

_warned_no_gpu = False


# =============================================================================
# Individual reward functions
# =============================================================================


def _kernel_quality_reward(parsed: dict[str, Any], context: Any) -> float:
    """Sigmoid reward gated on correctness, incorporating speedup.

    Returns sigmoid(1 + min(speedup, SPEEDUP_CLIP) - SIGMOID_DELTA) when correct,
    0.0 when incorrect. If correct but no timing data, speedup is treated as 0.
    """
    if not isinstance(parsed, dict):
        return 0.0

    exec_result = parsed.get("exec_result")
    if exec_result is None or not isinstance(exec_result, dict):
        return 0.0
    if not exec_result.get("correct", False):
        return 0.0

    speedup = exec_result.get("speedup_ratio")
    if speedup is None:
        speedup = 0.0
    else:
        speedup = min(float(speedup), SPEEDUP_CLIP)

    x = 1.0 + speedup - SIGMOID_DELTA
    return 1.0 / (1.0 + math.exp(-x))


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
# Rubric
# =============================================================================


class TritonKernelRubric(Rubric):
    """Sigmoid-based rubric with speedup reward for Triton kernel generation.

    R = 0.80 * kernel_quality + 0.10 * format + 0.10 * thinking
    """

    def step_reward(
        self, *, parsed: Any, context: Any, turn_index: int
    ) -> tuple[float, dict[str, float]]:
        if not isinstance(parsed, dict):
            return 0.0, {}

        try:
            kernel_q = _kernel_quality_reward(parsed, context)
            fmt = _solution_format_reward(parsed, context)
            think = _thinking_format_reward(parsed, context)

            total = SIGMOID_KERNEL_WEIGHT * kernel_q + 0.10 * fmt + 0.10 * think

            components = {
                "kernel_quality": kernel_q,
                "format": fmt,
                "thinking": think,
            }

            global _warned_no_gpu
            if not _warned_no_gpu and parsed.get("exec_result") is None:
                _warned_no_gpu = True
                logger.info(
                    "Kernel rewards: first kernel has no exec_result "
                    "(no valid code or gpu_eval disabled), reward = 0.20 "
                    "for this rollout (format + thinking only)"
                )

            return float(total), components
        except Exception:
            return 0.0, {}
