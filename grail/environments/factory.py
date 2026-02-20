"""Environment factory for clean instantiation and dependency injection."""

from __future__ import annotations

import logging
from typing import Any

from ..shared.constants import CURRENT_ENV_ID
from .core import MultiTurnEnv

logger = logging.getLogger(__name__)

# Module-level cache for task sources to avoid redundant dataset downloads
# Cache key: (env_id, split) -> TaskSource instance
_TASK_SOURCE_CACHE: dict[tuple[str, str], Any] = {}

# Minimal test_code for GPU warmup (exercises torch + basic model pattern)
_WARMUP_TEST_CODE = """
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

def check_correctness(model_new_class):
    model = model_new_class()
    if torch.cuda.is_available():
        model = model.cuda()
    inputs = get_inputs()
    if torch.cuda.is_available():
        inputs = [x.cuda() for x in inputs]
    output = model(*inputs)
    return output.shape == inputs[0].shape
"""


def get_or_create_task_source(
    env_id: str,
    split: str = "train",
) -> Any:
    """Get cached task source or create new one.

    Caches task sources by (env_id, split) to avoid redundant dataset downloads.
    Thread-safe for read-heavy workloads (protected by Python GIL).

    Args:
        env_id: Environment identifier (sat, gsm8k, math, mbpp, python_code, humaneval)
        split: Dataset split (train, test, validation)

    Returns:
        Cached or newly created TaskSource instance

    Raises:
        ValueError: If env_id is unknown

    Examples:
        >>> source = get_or_create_task_source("gsm8k", "train")
        >>> # Second call returns cached instance
        >>> same_source = get_or_create_task_source("gsm8k", "train")
        >>> assert source is same_source
    """
    cache_key = (env_id, split)

    if cache_key not in _TASK_SOURCE_CACHE:
        logger.debug("Creating new task source: env_id=%s, split=%s", env_id, split)

        if env_id == "sat":
            from .providers import SATTaskSource

            _TASK_SOURCE_CACHE[cache_key] = SATTaskSource()

        elif env_id == "gsm8k":
            from .providers import GSM8KTaskSource

            _TASK_SOURCE_CACHE[cache_key] = GSM8KTaskSource(split=split)

        elif env_id == "math":
            from .providers import MATHTaskSource

            _TASK_SOURCE_CACHE[cache_key] = MATHTaskSource(split=split)

        elif env_id in ("mbpp", "python_code"):
            from .providers import MBPPTaskSource

            # Map 'val' to 'validation' for consistency
            actual_split = "validation" if split == "val" else split
            _TASK_SOURCE_CACHE[cache_key] = MBPPTaskSource(split=actual_split)

        elif env_id == "humaneval":
            from .providers import HumanEvalTaskSource

            _TASK_SOURCE_CACHE[cache_key] = HumanEvalTaskSource()

        elif env_id == "affine_trace":
            from .affinetes.task_sources import TraceTaskSource

            _TASK_SOURCE_CACHE[cache_key] = TraceTaskSource()

        elif env_id == "affine_logic":
            from .affinetes.task_sources import LogicTaskSource

            _TASK_SOURCE_CACHE[cache_key] = LogicTaskSource()

        elif env_id == "triton_kernel":
            from .gpu_kernel.task_sources import KernelBenchTaskSource

            _TASK_SOURCE_CACHE[cache_key] = KernelBenchTaskSource(split=split)

        else:
            raise ValueError(f"Unknown environment ID: {env_id}")

        logger.info(
            "✅ Task source cached: env_id=%s, split=%s (cache_size=%d)",
            env_id,
            split,
            len(_TASK_SOURCE_CACHE),
        )
    else:
        logger.debug("Using cached task source: env_id=%s, split=%s", env_id, split)

    return _TASK_SOURCE_CACHE[cache_key]


def clear_task_source_cache() -> None:
    """Clear task source cache (useful for testing).

    Examples:
        >>> clear_task_source_cache()
        >>> assert len(_TASK_SOURCE_CACHE) == 0
    """
    _TASK_SOURCE_CACHE.clear()
    logger.debug("Task source cache cleared")


def create_env(
    env_id: str | None = None,
    *,
    task_source: Any | None = None,
    split: str = "train",
    env_params: dict[str, Any] | None = None,
) -> MultiTurnEnv:
    """Create environment instance with clean dependency injection.

    Task sources are automatically cached by (env_id, split) to avoid redundant
    dataset downloads. Pass a custom task_source to override caching behavior.

    Args:
        env_id: Environment identifier (defaults to CURRENT_ENV_ID)
        task_source: Optional task source (if None, uses cached source)
        split: Dataset split for GSM8K/MATH (train/test/validation)
        env_params: Optional environment parameters (overrides defaults like split)

    Returns:
        Initialized environment instance

    Raises:
        ValueError: If env_id is unknown

    Examples:
        >>> env = create_env()  # Uses CURRENT_ENV_ID with cached source
        >>> env = create_env("sat")
        >>> env = create_env("gsm8k", split="test")  # Caches test split
        >>> env = create_env("math", split="train")  # Caches train split
        >>> # Custom source (bypasses cache)
        >>> custom_source = MATHTaskSource(split="validation")
        >>> env = create_env("math", task_source=custom_source)
        >>> # With runtime params from checkpoint
        >>> env = create_env("mbpp", env_params={"split": "validation"})
    """
    env_id = env_id or CURRENT_ENV_ID

    # Override split with env_params if provided (check for non-empty dict)
    if env_params:
        split = env_params.get("split", split)

    if env_id == "sat":
        from .sat_env import SATEnv

        return SATEnv()

    if env_id == "gsm8k":
        from .gsm8k_env import GSM8KEnv

        # Use provided source OR get from cache
        source = (
            task_source if task_source is not None else get_or_create_task_source(env_id, split)
        )
        return GSM8KEnv(task_source=source)

    if env_id == "math":
        from .math_hendrycks_env import MATHEnv

        # Use provided source OR get from cache
        source = (
            task_source if task_source is not None else get_or_create_task_source(env_id, split)
        )
        return MATHEnv(task_source=source)

    if env_id in ("mbpp", "python_code"):
        from .python_code_env import PythonCodeEnv

        # Map 'val' to 'validation' for consistency
        actual_split = "validation" if split == "val" else split

        # Use provided source OR get from cache
        source = (
            task_source
            if task_source is not None
            else get_or_create_task_source(env_id, actual_split)
        )
        return PythonCodeEnv(dataset="mbpp", split=actual_split, task_source=source)

    if env_id == "humaneval":
        from .python_code_env import PythonCodeEnv

        # Use provided source OR get from cache
        source = (
            task_source if task_source is not None else get_or_create_task_source(env_id, "test")
        )
        return PythonCodeEnv(dataset="humaneval", split="test", task_source=source)

    if env_id == "affine_trace":
        from .affinetes.trace_env import AffineTraceEnv

        source = (
            task_source if task_source is not None else get_or_create_task_source(env_id, split)
        )
        return AffineTraceEnv(task_source=source)

    if env_id == "affine_logic":
        from .affinetes.logic_env import AffineLogicEnv

        source = (
            task_source if task_source is not None else get_or_create_task_source(env_id, split)
        )
        return AffineLogicEnv(task_source=source)

    if env_id == "triton_kernel":
        from .gpu_kernel.env import TritonKernelEnv

        if task_source is not None:
            source = task_source
        else:
            # If dataset_path provided, use UnifiedKernelTaskSource (training);
            # otherwise use KernelBenchTaskSource from HF (miners/validators).
            dataset_path = env_params.get("dataset_path", "") if env_params else ""
            if dataset_path:
                from .gpu_kernel.task_sources import UnifiedKernelTaskSource

                mode = env_params.get("mode", "rl") if env_params else "rl"
                exclude_sources = env_params.get("exclude_sources") if env_params else None
                weighted_sampling = (
                    env_params.get("weighted_sampling", False) if env_params else False
                )
                source = UnifiedKernelTaskSource(
                    dataset_path=dataset_path,
                    split=split,
                    mode=mode,
                    exclude_sources=exclude_sources,
                    weighted_sampling=weighted_sampling,
                )
            else:
                source = get_or_create_task_source(env_id, split)

        level = env_params.get("level") if env_params else None
        # gpu_eval from env_params (checkpoint metadata) or GRAIL_GPU_EVAL env var
        gpu_eval = env_params.get("gpu_eval", False) if env_params else False
        if not gpu_eval:
            import os

            gpu_eval = os.environ.get("GRAIL_GPU_EVAL", "false").lower() in ("1", "true", "yes")
        eval_backend = env_params.get("eval_backend") if env_params else None

        # Auto-initialize global eval backend if gpu_eval=True and none exists
        if gpu_eval and eval_backend is None:
            from .gpu_kernel.eval_backends import (
                create_backend,
                get_global_backend,
                parse_gpu_ids,
                set_global_backend,
            )

            if get_global_backend() is None:
                gpu_ids = parse_gpu_ids()
                if gpu_ids:
                    backend = create_backend(gpu_ids=gpu_ids)
                    backend.start()
                    # Warmup JIT cache on eval GPUs to avoid cold compilation latency
                    try:
                        warmup_codes = [_WARMUP_TEST_CODE]
                        backend.warmup(warmup_codes)
                    except Exception:
                        logger.warning("Eval backend warmup failed, continuing without warmup")
                    set_global_backend(backend)
                    logger.info("Auto-initialized eval backend on GPUs %s", gpu_ids)

        return TritonKernelEnv(
            split=split,
            level=level,
            task_source=source,
            gpu_eval=gpu_eval,
            eval_backend=eval_backend,
        )

    raise ValueError(f"Unknown environment ID: {env_id}")


def create_env_factory(
    env_id: str | None = None,
    *,
    task_source: Any | None = None,
    split: str = "train",
    env_params: dict[str, Any] | None = None,
) -> Any:
    """Create environment factory function for deferred instantiation.

    Useful when you need to pass a factory to components that create
    multiple environment instances (e.g., vectorized evaluation).

    Args:
        env_id: Environment identifier (defaults to CURRENT_ENV_ID)
        task_source: Optional task source for dataset-backed envs
        split: Dataset split for GSM8K
        env_params: Optional environment parameters (overrides defaults)

    Returns:
        Callable that creates new environment instances

    Examples:
        >>> factory = create_env_factory("gsm8k", split="test")
        >>> env1 = factory()
        >>> env2 = factory()
        >>> # With runtime params from checkpoint
        >>> factory = create_env_factory("mbpp", env_params={"split": "validation"})
    """
    env_id = env_id or CURRENT_ENV_ID

    def _factory() -> MultiTurnEnv:
        return create_env(
            env_id=env_id, task_source=task_source, split=split, env_params=env_params
        )

    return _factory
