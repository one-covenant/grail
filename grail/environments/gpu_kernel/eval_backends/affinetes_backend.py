"""Affinetes Docker container pool backend for GPU kernel evaluation.

Uses vendored Affinetes instance pool with Docker GPU containers
for isolated kernel evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from . import EvalResult

logger = logging.getLogger(__name__)


class AffinetesBackend:
    """GPU kernel evaluation via Affinetes Docker container pool.

    Args:
        gpu_ids: GPU device indices to expose to containers.
        timeout: Per-kernel evaluation timeout in seconds.
    """

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        self._gpu_ids = gpu_ids or []
        self._timeout = timeout
        self._started = False

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        if not self._started:
            raise RuntimeError("AffinetesBackend not started. Call start() first.")
        raise NotImplementedError(
            "AffinetesBackend.evaluate() not yet implemented. "
            "Use KERNEL_EVAL_BACKEND=subprocess instead."
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        return [self.evaluate(tc, tr) for tc, tr in items]

    def warmup(self, sample_test_codes: list[str]) -> None:
        logger.info("AffinetesBackend warmup: %d samples (stub)", len(sample_test_codes))

    def start(self) -> None:
        self._started = True
        logger.info("AffinetesBackend started (stub)")

    def shutdown(self) -> None:
        self._started = False
        logger.info("AffinetesBackend shut down (stub)")
