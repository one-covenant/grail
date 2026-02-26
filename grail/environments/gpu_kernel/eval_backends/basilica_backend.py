"""Basilica cloud GPU backend for kernel evaluation.

Uses Basilica's cloud platform with persistent GPU workers for remote
kernel evaluation. No local GPU needed.
"""

from __future__ import annotations

import logging
from typing import Any

from . import EvalResult

logger = logging.getLogger(__name__)


class BasilicaBackend:
    """GPU kernel evaluation via Basilica cloud GPU workers.

    Args:
        timeout: Per-kernel evaluation timeout in seconds.
        gpu_type: GPU type to request (e.g. 'A100', 'H100').
    """

    def __init__(
        self,
        timeout: float = 60.0,
        gpu_type: str = "A100",
        **kwargs: Any,
    ) -> None:
        self._timeout = timeout
        self._gpu_type = gpu_type
        self._started = False

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        if not self._started:
            raise RuntimeError("BasilicaBackend not started. Call start() first.")
        raise NotImplementedError(
            "BasilicaBackend.evaluate() not yet implemented. "
            "Use KERNEL_EVAL_BACKEND=persistent or subprocess instead."
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        return [self.evaluate(tc, tr) for tc, tr in items]

    def warmup(self, sample_test_codes: list[str]) -> None:
        logger.info("BasilicaBackend warmup: %d samples (stub)", len(sample_test_codes))

    def start(self) -> None:
        self._started = True
        logger.info("BasilicaBackend started (stub)")

    def shutdown(self) -> None:
        self._started = False
        logger.info("BasilicaBackend shut down (stub)")
