"""Logging formatters for observability.

Provides a minimal JSON formatter suitable for shipping logs to Loki or other
structured log backends.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Format log records as compact JSON.

    Includes timestamp, level, logger name, message, and selected context
    attributes if present on the record.
    """

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Optional context that a ContextFilter may attach
        for key in (
            "service",
            "env",
            "version",
            "network",
            "netuid",
            "wallet",
            "hotkey",
            "run_id",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        # Exception information, if any
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Millisecond precision timestamp as ISO-like string (optional)
        try:
            payload["time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
        except Exception:
            pass

        return json.dumps(payload, separators=(",", ":"))


# end
