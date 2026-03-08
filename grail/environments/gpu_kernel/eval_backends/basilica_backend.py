"""Basilica cloud GPU backend for kernel evaluation.

Uses Basilica's kernel-bench service for remote GPU evaluation.
No local GPU needed — kernels are compiled, correctness-checked,
and benchmarked on Basilica's persistent A100 worker pool.

Configuration:
    BASILICA_EVAL_URL: Base URL of the kernel-bench deployment.
    KERNEL_EVAL_TIMEOUT: Per-kernel timeout in seconds (default 60).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from . import EvalResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0
_CONNECT_TIMEOUT = 10.0
_MAX_BATCH_WORKERS = 8
_MAX_RETRIES = 2
_RETRY_STATUSES = {502, 503, 504}

# Application-level errors from the kernel-bench worker pool that indicate
# infrastructure failure (worker crash, pool exhaustion), NOT genuine kernel
# evaluation results. These should be retried and, if persistent, marked as
# infra_error so the miner discards the rollout.
_INFRA_ERROR_PATTERNS = (
    "Worker connection lost",
    "Worker timed out",
    "No workers available",
    "Worker connection reset",
    "worker_send_failed",
    "worker_recv_failed",
    "worker_init_failed",
)


class BasilicaBackend:
    """GPU kernel evaluation via Basilica cloud GPU workers.

    Calls the ``/evaluate_raw`` endpoint which accepts grail's native
    ``(test_code, triton_code)`` format directly. The service runs
    ``check_correctness(ModelNew)`` on the GPU, then benchmarks
    ``Model.forward`` vs ``ModelNew.forward`` if correct.

    Only correctness fields are used for the returned ``EvalResult``.
    Full timing data is logged for observability.

    Args:
        timeout: Per-kernel evaluation timeout in seconds.
        url: Base URL of the kernel-bench service. Falls back to
            ``BASILICA_EVAL_URL`` env var.
        max_retries: Retries on transient HTTP errors (502/503/504).
    """

    def __init__(
        self,
        timeout: float = _DEFAULT_TIMEOUT,
        url: str | None = None,
        max_retries: int = _MAX_RETRIES,
        **kwargs: Any,
    ) -> None:
        self._url = url or os.environ.get("BASILICA_EVAL_URL")
        if not self._url:
            raise ValueError(
                "BasilicaBackend requires a URL. Set BASILICA_EVAL_URL env var "
                "or pass url= to constructor."
            )
        self._url = self._url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._session: requests.Session | None = None
        self._started = False

    def start(self) -> None:
        """Create HTTP session and verify the service is reachable."""
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

        try:
            resp = self._session.get(
                f"{self._url}/health",
                timeout=_CONNECT_TIMEOUT,
            )
            resp.raise_for_status()
            health = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"BasilicaBackend failed to connect to {self._url}/health: {e}"
            ) from e

        status = health.get("status", "unknown")
        workers = health.get("workers_alive", 0)
        if status == "unhealthy" or workers == 0:
            raise RuntimeError(
                f"BasilicaBackend service unhealthy: status={status}, workers_alive={workers}"
            )

        self._started = True
        logger.info(
            "BasilicaBackend started: url=%s status=%s workers=%d",
            self._url,
            status,
            workers,
        )

    @staticmethod
    def _is_infra_error(error: str | None) -> bool:
        """Check if an error message indicates infrastructure failure."""
        if not error:
            return False
        return any(pattern in error for pattern in _INFRA_ERROR_PATTERNS)

    def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single kernel via Basilica's /evaluate_raw endpoint.

        Retries on infrastructure errors (worker crashes, pool exhaustion).
        If all retries fail, returns EvalResult with infra_error=True so the
        miner can discard the rollout rather than uploading an unreliable reward.
        """
        if not self._started or self._session is None:
            raise RuntimeError("BasilicaBackend not started. Call start() first.")

        payload = {
            "test_code": test_code,
            "triton_code": triton_code,
        }

        last_data: dict | None = None
        for attempt in range(1 + self._max_retries):
            data = self._post_with_retry("/evaluate_raw", payload)
            last_data = data

            error = data.get("error")
            if not self._is_infra_error(error):
                break  # Genuine result (success or real kernel failure)

            if attempt < self._max_retries:
                logger.warning(
                    "Basilica infra error (attempt %d/%d): %s, retrying...",
                    attempt + 1,
                    self._max_retries + 1,
                    error,
                )
                time.sleep(1.0 * (attempt + 1))
            else:
                logger.error(
                    "Basilica infra error persisted after %d attempts: %s",
                    self._max_retries + 1,
                    error,
                )

        assert last_data is not None
        correct = last_data.get("correct", False)
        compiled = last_data.get("compiled", False)
        error = last_data.get("error")
        max_diff = last_data.get("max_diff")
        infra_error = self._is_infra_error(error)

        # Log timing for observability (not used in EvalResult)
        timing = last_data.get("timing")
        if timing:
            logger.info(
                "Basilica timing: kernel=%.4fms ref=%.4fms speedup=%s device=%s",
                timing.get("kernel_median_ms", 0),
                timing.get("reference_median_ms", 0),
                timing.get("speedup", "N/A"),
                last_data.get("device", "unknown"),
            )

        return EvalResult(
            correct=correct,
            compiled=compiled,
            error=error,
            max_diff=max_diff,
            infra_error=infra_error,
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels in parallel via thread pool."""
        if not items:
            return []

        results: list[EvalResult | None] = [None] * len(items)
        max_workers = min(len(items), _MAX_BATCH_WORKERS)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.evaluate, tc, tr): i for i, (tc, tr) in enumerate(items)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Batch eval item %d failed: %s", idx, e)
                    results[idx] = EvalResult(
                        correct=False,
                        compiled=False,
                        error=f"BasilicaBackend error: {e}",
                    )

        return results  # type: ignore[return-value]

    def warmup(self, sample_test_codes: list[str]) -> None:
        """No-op — Basilica workers stay warm via persistent process pool."""
        logger.info("BasilicaBackend warmup: %d samples (skipped)", len(sample_test_codes))

    def shutdown(self) -> None:
        """Close HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._started = False
        logger.info("BasilicaBackend shut down")

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with retry on transient HTTP errors."""
        assert self._session is not None
        url = f"{self._url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    timeout=self._timeout + 5,
                )

                if resp.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                    logger.warning(
                        "Basilica %s returned %d, retrying (%d/%d)",
                        path,
                        resp.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    time.sleep(1.0 * (attempt + 1))
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.ConnectionError as e:
                last_exc = e
                if attempt < self._max_retries:
                    logger.warning(
                        "Basilica connection error, retrying (%d/%d): %s",
                        attempt + 1,
                        self._max_retries,
                        e,
                    )
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise

            except requests.Timeout as e:
                raise RuntimeError(f"Basilica request timed out after {self._timeout + 5}s") from e

        raise RuntimeError(
            f"Basilica request failed after {self._max_retries + 1} attempts: {last_exc}"
        )
