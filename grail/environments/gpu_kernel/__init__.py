"""GPU Kernel RL Environment for Triton kernel generation.

Provides a single-turn environment for training models to write optimized
Triton GPU kernels from PyTorch reference implementations using KernelBench.

Public API:
    - TritonKernelEnv: Main environment (SingleTurnEnv subclass)
    - KernelBenchTaskSource: Dataset provider for KernelBench problems
    - UnifiedKernelTaskSource: Dataset provider for the unified GPU kernel dataset
    - TritonKernelParser: Parser for Triton code extraction and validation
    - TritonKernelRubric: Multi-component reward rubric
    - EvalResult: Result of kernel evaluation
    - KernelEvalBackend: Protocol for evaluation backends
    - create_backend: Factory for evaluation backends
    - get_global_backend / set_global_backend: Global backend management
"""

from .env import TritonKernelEnv
from .eval_backends import (
    EvalResult,
    KernelEvalBackend,
    create_backend,
    get_global_backend,
    set_global_backend,
    validate_gpu_config,
)
from .parser import TritonKernelParser
from .rewards import TritonKernelRubric, create_triton_kernel_reward_vector
from .task_sources import KernelBenchTaskSource, UnifiedKernelTaskSource

__all__ = [
    "TritonKernelEnv",
    "KernelBenchTaskSource",
    "UnifiedKernelTaskSource",
    "TritonKernelParser",
    "TritonKernelRubric",
    "create_triton_kernel_reward_vector",
    "EvalResult",
    "KernelEvalBackend",
    "create_backend",
    "get_global_backend",
    "set_global_backend",
    "validate_gpu_config",
]
