"""Modal serverless GPU backend for kernel evaluation.

Uses Modal's @modal.function(gpu=...) for serverless GPU evaluation.
No local GPU needed - evaluation runs on Modal's infrastructure.
"""

from __future__ import annotations

import logging
from typing import Any

from . import EvalResult

logger = logging.getLogger(__name__)


class ModalBackend:
    """GPU kernel evaluation via Modal serverless GPU.

    Args:
        timeout: Per-kernel evaluation timeout in seconds.
        gpu_type: Modal GPU type (e.g. 'T4', 'A10G', 'A100').
    """

    def __init__(
        self,
        timeout: float = 60.0,
        gpu_type: str = "T4",
        **kwargs: Any,
    ) -> None:
        self._timeout = timeout
        self._gpu_type = gpu_type
        self._started = False

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        if not self._started:
            raise RuntimeError("ModalBackend not started. Call start() first.")
        raise NotImplementedError(
            "ModalBackend.evaluate() not yet implemented. "
            "Use KERNEL_EVAL_BACKEND=subprocess instead."
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        return [self.evaluate(tc, tr) for tc, tr in items]

    def warmup(self, sample_test_codes: list[str]) -> None:
        logger.info("ModalBackend warmup: %d samples (stub)", len(sample_test_codes))

    def start(self) -> None:
        self._started = True
        logger.info("ModalBackend started (stub)")

    def shutdown(self) -> None:
        self._started = False
        logger.info("ModalBackend shut down (stub)")
