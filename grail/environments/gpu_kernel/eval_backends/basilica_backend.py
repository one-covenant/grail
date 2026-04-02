"""Basilica cloud GPU backend for kernel evaluation.

Uses Basilica's kernel-bench service for remote GPU evaluation.
No local GPU needed -- kernels are compiled, correctness-checked,
and benchmarked on Basilica's persistent A100 worker pool.

Two client implementations:
- BasilicaBackend: synchronous requests.Session with connection pooling.
  Conforms to KernelEvalBackend protocol. Called from ThreadPoolExecutor
  in the mining engine's env.step() path.
- AsyncBasilicaBackend: httpx.AsyncClient with HTTP/2 multiplexing.
  For use when evaluate_batch can be called from an async context directly,
  eliminating thread overhead entirely.

Connection pooling strategy:
  The synchronous backend creates a single requests.Session with
  urllib3 connection pooling (pool_connections=4, pool_maxsize=16).
  All threads in evaluate_batch share this session since urllib3's
  PoolManager is thread-safe. This eliminates the per-thread Session
  creation overhead in the original implementation.

Configuration:
    BASILICA_EVAL_URL: Base URL of the kernel-bench deployment.
    KERNEL_EVAL_TIMEOUT: Per-kernel timeout in seconds (default 60).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import EvalResult

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0
_CONNECT_TIMEOUT = 10.0
_MAX_BATCH_WORKERS = 16
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


def _create_pooled_session(
    pool_connections: int = 4,
    pool_maxsize: int = 16,
) -> requests.Session:
    """Create a requests.Session with connection pooling optimized for Basilica.

    urllib3's PoolManager is thread-safe, so one Session can be shared
    across all threads in evaluate_batch(). This eliminates per-thread
    Session creation overhead (~5-10ms each) and enables TCP connection
    reuse across requests.

    Args:
        pool_connections: Number of urllib3 connection pools to cache.
        pool_maxsize: Maximum connections per pool (per host:port).
    """
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"

    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=Retry(total=0),  # We handle retries at the application level
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


class BasilicaBackend:
    """GPU kernel evaluation via Basilica cloud GPU workers.

    Calls the ``/evaluate_raw`` endpoint which accepts grail's native
    ``(test_code, triton_code)`` format directly. The service runs
    ``check_correctness(ModelNew)`` on the GPU, then benchmarks
    ``Model.forward`` vs ``ModelNew.forward`` if correct.

    Uses a single requests.Session with urllib3 connection pooling
    (thread-safe). All threads in evaluate_batch() share the same
    session and connection pool, enabling TCP connection reuse and
    eliminating per-thread setup overhead.

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
        """Create pooled HTTP session and verify the service is reachable."""
        self._session = _create_pooled_session(
            pool_connections=4,
            pool_maxsize=_MAX_BATCH_WORKERS,
        )

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
            "BasilicaBackend started: url=%s status=%s workers=%d pool_maxsize=%d",
            self._url,
            status,
            workers,
            _MAX_BATCH_WORKERS,
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

        Thread-safe: uses the shared pooled session (urllib3 is thread-safe).
        """
        if not self._started:
            raise RuntimeError("BasilicaBackend not started. Call start() first.")

        if self._session is None:
            raise RuntimeError("No HTTP session available.")

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

        if last_data is None:
            raise RuntimeError("Basilica evaluate returned no data after all attempts")

        correct = last_data.get("correct", False)
        compiled = last_data.get("compiled", False)
        error = last_data.get("error")
        max_diff = last_data.get("max_diff")
        infra_error = self._is_infra_error(error)

        # Extract timing data
        timing = last_data.get("timing")
        speedup_ratio = None
        kernel_median_ms = None
        reference_median_ms = None
        if timing and correct:
            speedup_ratio = timing.get("speedup_ratio")
            kernel_median_ms = timing.get("kernel_median_ms")
            reference_median_ms = timing.get("reference_median_ms")
            logger.info(
                "Basilica timing: kernel=%.4fms ref=%.4fms speedup=%.3fx device=%s",
                kernel_median_ms or 0,
                reference_median_ms or 0,
                speedup_ratio or 0,
                last_data.get("device", "unknown"),
            )

        return EvalResult(
            correct=correct,
            compiled=compiled,
            error=error,
            max_diff=max_diff,
            infra_error=infra_error,
            speedup_ratio=speedup_ratio,
            kernel_median_ms=kernel_median_ms,
            reference_median_ms=reference_median_ms,
        )

    def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels in parallel via thread pool.

        All threads share the same pooled session. urllib3's PoolManager
        is thread-safe and maintains a pool of TCP connections, so
        concurrent requests reuse connections without per-thread overhead.
        """
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
        """No-op. Basilica workers stay warm via persistent process pool."""
        logger.info("BasilicaBackend warmup: %d samples (skipped)", len(sample_test_codes))

    def shutdown(self) -> None:
        """Close HTTP session and connection pool."""
        if self._session is not None:
            self._session.close()
            self._session = None
        self._started = False
        logger.info("BasilicaBackend shut down")

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with retry on transient HTTP errors.

        Thread-safe: uses the shared pooled session.
        """
        if self._session is None:
            raise RuntimeError("No HTTP session available.")

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


# --------------------------------------------------------------------------- #
#                     Async Backend (HTTP/2 multiplexed)                       #
# --------------------------------------------------------------------------- #

_ASYNC_MAX_CONCURRENCY = 16


class AsyncBasilicaBackend:
    """Async GPU kernel evaluation via httpx with HTTP/2 multiplexing.

    For use when the eval path can be called from an async context directly.
    All concurrent eval requests share a single TCP connection via HTTP/2
    stream multiplexing, eliminating per-request TLS handshakes.

    This is the recommended backend when the mining engine supports async
    eval dispatch (future refactor). Until then, BasilicaBackend with
    connection pooling is the production default.
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
                "AsyncBasilicaBackend requires a URL. Set BASILICA_EVAL_URL env var "
                "or pass url= to constructor."
            )
        self._url = self._url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any | None = None  # httpx.AsyncClient, lazy import
        self._started = False

    async def start(self) -> None:
        """Create HTTP/2 client and verify the service is reachable."""
        import httpx

        if self._url is None:
            raise RuntimeError("AsyncBasilicaBackend URL not configured")
        self._client = httpx.AsyncClient(
            http2=True,
            base_url=self._url,
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT,
                read=self._timeout + 10,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=300,
            ),
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            health = resp.json()
        except Exception as e:
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"AsyncBasilicaBackend failed to connect to {self._url}/health: {e}"
            ) from e

        status = health.get("status", "unknown")
        workers = health.get("workers_alive", 0)
        if status == "unhealthy" or workers == 0:
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"AsyncBasilicaBackend service unhealthy: status={status}, workers_alive={workers}"
            )

        http_version = getattr(resp, "http_version", "unknown")
        self._started = True
        logger.info(
            "AsyncBasilicaBackend started: url=%s status=%s workers=%d http=%s",
            self._url,
            status,
            workers,
            http_version,
        )

    @staticmethod
    def _is_infra_error(error: str | None) -> bool:
        """Check if an error message indicates infrastructure failure."""
        if not error:
            return False
        return any(pattern in error for pattern in _INFRA_ERROR_PATTERNS)

    async def evaluate(self, test_code: str, triton_code: str) -> EvalResult:
        """Evaluate a single kernel via async HTTP/2 request."""
        if not self._started or self._client is None:
            raise RuntimeError("AsyncBasilicaBackend not started. Call start() first.")

        payload = {"test_code": test_code, "triton_code": triton_code}

        last_data: dict | None = None
        for attempt in range(1 + self._max_retries):
            data = await self._post_with_retry("/evaluate_raw", payload)
            last_data = data

            error = data.get("error")
            if not self._is_infra_error(error):
                break

            if attempt < self._max_retries:
                logger.warning(
                    "Async Basilica infra error (attempt %d/%d): %s, retrying...",
                    attempt + 1,
                    self._max_retries + 1,
                    error,
                )
                await asyncio.sleep(1.0 * (attempt + 1))
            else:
                logger.error(
                    "Async Basilica infra error persisted after %d attempts: %s",
                    self._max_retries + 1,
                    error,
                )

        if last_data is None:
            raise RuntimeError("Basilica evaluate returned no data after all attempts")

        correct = last_data.get("correct", False)
        compiled = last_data.get("compiled", False)
        error = last_data.get("error")
        max_diff = last_data.get("max_diff")
        infra_error = self._is_infra_error(error)

        timing = last_data.get("timing")
        speedup_ratio = None
        kernel_median_ms = None
        reference_median_ms = None
        if timing and correct:
            speedup_ratio = timing.get("speedup_ratio")
            kernel_median_ms = timing.get("kernel_median_ms")
            reference_median_ms = timing.get("reference_median_ms")
            logger.info(
                "Basilica timing: kernel=%.4fms ref=%.4fms speedup=%.3fx device=%s",
                kernel_median_ms or 0,
                reference_median_ms or 0,
                speedup_ratio or 0,
                last_data.get("device", "unknown"),
            )

        return EvalResult(
            correct=correct,
            compiled=compiled,
            error=error,
            max_diff=max_diff,
            infra_error=infra_error,
            speedup_ratio=speedup_ratio,
            kernel_median_ms=kernel_median_ms,
            reference_median_ms=reference_median_ms,
        )

    async def evaluate_batch(self, items: list[tuple[str, str]]) -> list[EvalResult]:
        """Evaluate multiple kernels concurrently via HTTP/2 multiplexing."""
        if not items:
            return []

        semaphore = asyncio.Semaphore(_ASYNC_MAX_CONCURRENCY)

        async def _eval_one(idx: int, tc: str, tr: str) -> tuple[int, EvalResult]:
            async with semaphore:
                try:
                    return idx, await self.evaluate(tc, tr)
                except Exception as e:
                    logger.error("Async batch eval item %d failed: %s", idx, e)
                    return idx, EvalResult(
                        correct=False,
                        compiled=False,
                        error=f"AsyncBasilicaBackend error: {e}",
                    )

        tasks = [_eval_one(i, tc, tr) for i, (tc, tr) in enumerate(items)]
        completed = await asyncio.gather(*tasks)

        results: list[EvalResult | None] = [None] * len(items)
        for idx, result in completed:
            results[idx] = result
        return results  # type: ignore[return-value]

    def warmup(self, sample_test_codes: list[str]) -> None:
        """No-op. Basilica workers stay warm via persistent process pool."""
        logger.info("AsyncBasilicaBackend warmup: %d samples (skipped)", len(sample_test_codes))

    async def shutdown(self) -> None:
        """Close HTTP/2 client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._started = False
        logger.info("AsyncBasilicaBackend shut down")

    async def _post_with_retry(self, path: str, payload: dict) -> dict:
        """POST with retry on transient HTTP errors."""
        import httpx

        if self._client is None:
            raise RuntimeError("No HTTP client available.")

        last_exc: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                resp = await self._client.post(path, json=payload)

                if resp.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                    logger.warning(
                        "Async Basilica %s returned %d, retrying (%d/%d)",
                        path,
                        resp.status_code,
                        attempt + 1,
                        self._max_retries,
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                resp.raise_for_status()
                return resp.json()

            except httpx.ConnectError as e:
                last_exc = e
                if attempt < self._max_retries:
                    logger.warning(
                        "Async Basilica connection error, retrying (%d/%d): %s",
                        attempt + 1,
                        self._max_retries,
                        e,
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise

            except httpx.TimeoutException as e:
                raise RuntimeError(
                    f"Async Basilica request timed out after {self._timeout + 10}s"
                ) from e

        raise RuntimeError(
            f"Async Basilica request failed after {self._max_retries + 1} attempts: {last_exc}"
        )
