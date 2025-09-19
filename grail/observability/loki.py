"""Minimal Grafana Loki logging handler.
This handler ships log records to a Loki `push` endpoint using requests.
It buffers logs in a background thread to avoid blocking the main thread.
Environment variables (consumed by the CLI wiring):
- GRAIL_LOKI_URL: Loki push URL, e.g.
  http://localhost:3100/loki/api/v1/push
- GRAIL_LOKI_TENANT_ID: Optional tenant header (X-Scope-OrgID)
- GRAIL_LOKI_USERNAME / GRAIL_LOKI_PASSWORD: Optional basic auth
- GRAIL_LOKI_LABELS: Comma-separated labels,
  e.g. "env=dev,service=grail"
- GRAIL_LOKI_BATCH_SIZE: Int (default 1) – set >1 to batch
- GRAIL_LOKI_BATCH_INTERVAL_S: Float seconds (default 1.0)
- GRAIL_LOKI_TIMEOUT_S: Float seconds HTTP timeout (default 2.5)
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

import requests


class LokiHandler(logging.Handler):
    """A logging handler that pushes logs to Grafana Loki.

    Notes:
        - Uses a background thread and an in-memory queue to avoid blocking.
        - Sends logs in a single stream defined by the provided labels.
        - Avoids internal logging to prevent recursion; errors are
          silently dropped.
    """

    def __init__(
        self,
        url: str,
        labels: dict[str, str],
        tenant_id: str | None = None,
        auth: tuple[str, str] | None = None,
        timeout: float = 2.5,
        batch_size: int = 1,
        batch_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.labels = {k: str(v) for k, v in labels.items()}
        self.tenant_id = tenant_id
        self.auth = auth
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.batch_interval = max(0.1, float(batch_interval))

        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="LokiHandlerThread",
            daemon=True,
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            # Loki expects timestamp in nanoseconds as a string
            ts_ns = str(int(record.created * 1_000_000_000))
            self._queue.put_nowait((ts_ns, message))
        except Exception:
            # Drop on failure to avoid interfering with application logging
            pass

    def _run(self) -> None:
        session = requests.Session()
        values: list[tuple[str, str]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set() or not self._queue.empty() or values:
            timeout = max(0.0, self.batch_interval - (time.monotonic() - last_flush))
            try:
                item = self._queue.get(timeout=timeout)
                values.append(item)
            except queue.Empty:
                # Timeout – fall through to flush check
                pass

            now = time.monotonic()
            should_flush = len(values) >= self.batch_size or (
                values and (now - last_flush) >= self.batch_interval
            )
            if should_flush:
                try:
                    self._push(session, values)
                except Exception:
                    # Swallow errors; avoid recursive logging
                    pass
                finally:
                    values.clear()
                    last_flush = now

        # Final flush on shutdown
        if values:
            try:
                self._push(session, values)
            except Exception:
                pass

    def _push(self, session: requests.Session, values: list[tuple[str, str]]) -> None:
        if not values:
            return
        # Build Loki push payload
        payload = {
            "streams": [
                {
                    "stream": self.labels,
                    "values": values,
                }
            ]
        }

        headers = {"Content-Type": "application/json"}
        if self.tenant_id:
            headers["X-Scope-OrgID"] = self.tenant_id

        # Loki commonly responds with 204 No Content on success
        session.post(
            self.url,
            data=json.dumps(payload),
            headers=headers,
            auth=self.auth,
            timeout=self.timeout,
        )

    def close(self) -> None:
        try:
            self._stop_event.set()
            # Nudge the queue to unblock the thread
            try:
                ts = str(int(time.time() * 1_000_000_000))
                self._queue.put_nowait((ts, ""))
            except Exception:
                pass
            self._thread.join(timeout=self.timeout + self.batch_interval + 0.5)
        except Exception:
            pass
        finally:
            super().close()
