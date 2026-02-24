"""Pluggable GPU evaluation backends for Triton kernel correctness checking.

Provides a protocol-based abstraction for kernel evaluation with three backends:
- SubprocessBackend: Per-eval subprocess isolation, GPU-pinned (default)
- AffinetesBackend: Docker container pool via vendored Affinetes
- ModalBackend: Serverless GPU via Modal

Configuration via environment variables:
    KERNEL_EVAL_BACKEND=subprocess|affinetes|modal
    KERNEL_EVAL_GPU_IDS=0,1,2   (comma-separated GPU indices)
    KERNEL_EVAL_TIMEOUT=60      (per-kernel timeout in seconds)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating a generated Triton kernel against reference.

    Attributes:
        correct: Whether kernel outputs match reference within tolerance.
        compiled: Whether kernel code compiled and ran without crashing.
        error: Error message if evaluation failed, None on success.
        max_diff: Maximum absolute difference between outputs, None if not computed.
    """

    correct: bool
    compiled: bool
    error: str | None = None
    max_diff: float | None = None


@runtime_checkable
class KernelEvalBackend(Protocol):
    """Protocol for GPU kernel evaluation backends.

    All backends must implement evaluate() for single-kernel evaluation.
    Batch, warmup, and lifecycle methods have default implementations.
    """

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single Triton kernel against its test harness.

        Args:
            test_code: Test code containing Model, get_inputs(), get_init_inputs(),
                       and check_correctness() functions.
            triton_code: Generated Triton kernel code defining ModelNew class.

        Returns:
            EvalResult with correctness and compilation status.
        """
        ...

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels. Default: sequential evaluate() calls."""
        ...

    def warmup(self, sample_test_codes: list[str]) -> None:
        """Warm up JIT cache and CUDA context on eval GPUs."""
        ...

    def start(self) -> None:
        """Start the backend (create workers, containers, etc.)."""
        ...

    def shutdown(self) -> None:
        """Shut down the backend and release resources."""
        ...


# ---------------------------------------------------------------------------
# Global backend management
# ---------------------------------------------------------------------------

_global_backend: KernelEvalBackend | None = None


def get_global_backend() -> KernelEvalBackend | None:
    """Get the globally configured eval backend, or None if not set."""
    return _global_backend


def set_global_backend(backend: KernelEvalBackend) -> None:
    """Set the global eval backend used by TritonKernelEnv and adapters."""
    global _global_backend
    _global_backend = backend
    logger.info("Global kernel eval backend set: %s", type(backend).__name__)


# ---------------------------------------------------------------------------
# GPU configuration validation
# ---------------------------------------------------------------------------


def parse_gpu_ids(gpu_ids_str: str | None = None) -> list[int]:
    """Parse GPU IDs from string (e.g. '0,1,2') or KERNEL_EVAL_GPU_IDS env var.

    Returns empty list if no GPU IDs configured.
    """
    raw = gpu_ids_str or os.environ.get("KERNEL_EVAL_GPU_IDS", "")
    if not raw.strip():
        return []
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        logger.warning("Invalid KERNEL_EVAL_GPU_IDS: %r, ignoring", raw)
        return []


def validate_gpu_config(gpu_ids: list[int], gpu_eval: bool) -> None:
    """Validate GPU configuration at startup. Raises RuntimeError with actionable message.

    Args:
        gpu_ids: List of GPU device indices to use for evaluation.
        gpu_eval: Whether GPU evaluation is enabled.

    Raises:
        RuntimeError: If gpu_eval=True but GPU configuration is invalid.
    """
    if not gpu_eval:
        return

    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "gpu_eval=True but PyTorch is not installed.\n"
            "Options:\n"
            "  - Install PyTorch: pip install torch\n"
            "  - Set gpu_eval=False (max reward = 0.35)\n"
            "  - Use KERNEL_EVAL_BACKEND=modal (no local GPU needed)"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpu_eval=True but CUDA is not available.\n"
            "Options:\n"
            "  - Set gpu_eval=False (max reward = 0.35)\n"
            "  - Use KERNEL_EVAL_BACKEND=modal (no local GPU needed)"
        )

    n_gpus = torch.cuda.device_count()

    if not gpu_ids:
        raise RuntimeError(
            f"gpu_eval=True but no KERNEL_EVAL_GPU_IDS configured.\n"
            f"  torch.cuda.device_count() = {n_gpus}\n\n"
            f"To fix:\n"
            f"  - Set KERNEL_EVAL_GPU_IDS to valid GPU indices (e.g. '0' or '0,1')\n"
            f"  - Use KERNEL_EVAL_BACKEND=modal (no local GPU needed)\n"
            f"  - Set gpu_eval=False (max reward = 0.35)"
        )

    for gid in gpu_ids:
        if gid >= n_gpus:
            device_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
            raise RuntimeError(
                f"KERNEL_EVAL_GPU_IDS contains {gid} but only {n_gpus} GPUs detected.\n"
                f"  Available: {device_names}\n\n"
                f"To fix:\n"
                f"  - Set KERNEL_EVAL_GPU_IDS to valid indices (0-{n_gpus - 1})\n"
                f"  - Use KERNEL_EVAL_BACKEND=modal (no local GPU needed)"
            )

    device_names = [torch.cuda.get_device_name(i) for i in gpu_ids]
    logger.info("Kernel eval GPUs: %s (%s)", gpu_ids, device_names)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def get_eval_timeout() -> float:
    """Get per-kernel evaluation timeout from env var or default."""
    try:
        return float(os.environ.get("KERNEL_EVAL_TIMEOUT", "60"))
    except ValueError:
        return 60.0


def create_backend(
    name: str | None = None,
    *,
    gpu_ids: list[int] | None = None,
    timeout: float | None = None,
    **kwargs: object,
) -> KernelEvalBackend:
    """Create a kernel evaluation backend by name.

    Args:
        name: Backend name ('subprocess', 'affinetes', 'modal').
              Defaults to KERNEL_EVAL_BACKEND env var or 'subprocess'.
        gpu_ids: GPU device indices. Defaults to KERNEL_EVAL_GPU_IDS env var.
        timeout: Per-kernel timeout. Defaults to KERNEL_EVAL_TIMEOUT env var or 60s.
        **kwargs: Additional backend-specific arguments.

    Returns:
        Configured KernelEvalBackend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    name = name or os.environ.get("KERNEL_EVAL_BACKEND", "subprocess")
    if gpu_ids is None:
        gpu_ids = parse_gpu_ids()
    if timeout is None:
        timeout = get_eval_timeout()

    if name == "subprocess":
        from .subprocess_backend import SubprocessBackend

        return SubprocessBackend(gpu_ids=gpu_ids, timeout=timeout, **kwargs)

    if name == "affinetes":
        from .affinetes_backend import AffinetesBackend

        return AffinetesBackend(gpu_ids=gpu_ids, timeout=timeout, **kwargs)

    if name == "modal":
        from .modal_backend import ModalBackend

        return ModalBackend(timeout=timeout, **kwargs)

    raise ValueError(
        f"Unknown kernel eval backend: {name!r}. Must be one of: subprocess, affinetes, modal"
    )
