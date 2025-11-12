"""Server utilities for offline trainer."""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


def wait_for_vllm_ready(base_url: str, timeout_s: float = 180.0) -> bool:
    """Wait for vLLM server to be ready.

    Polls the /v1/models endpoint until it responds with 200 OK.

    Args:
        base_url: Base URL of vLLM server (e.g., http://127.0.0.1:30001)
        timeout_s: Maximum time to wait in seconds

    Returns:
        True if server became ready, False if timeout
    """
    ready_url = f"{base_url}/v1/models"
    deadline = time.time() + timeout_s
    poll_count = 0
    last_error: str | None = None

    logger.info("Waiting for vLLM server at %s (timeout: %.0fs)", base_url, timeout_s)

    while time.time() < deadline:
        try:
            resp = requests.get(ready_url, timeout=3.0)
            if resp.status_code == 200:
                logger.info("✓ vLLM server ready at %s", base_url)
                return True
            last_error = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_error = str(exc)

        poll_count += 1
        if poll_count % 10 == 0:
            elapsed = time.time() - (deadline - timeout_s)
            logger.info("  Still waiting... (%.0fs elapsed, last: %s)", elapsed, last_error)

        time.sleep(2.0)

    logger.error("✗ vLLM server not ready after %.0fs: %s", timeout_s, last_error)
    return False
