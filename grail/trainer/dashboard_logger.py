"""Structured JSON loggers for the Grafana trainer dashboard.

Each logger emits a single JSON object per log line, which Promtail ships
to Loki. The Grafana dashboard queries these by ``{logger="grail.trainer.<category>"}``.

Usage::

    from grail.trainer.dashboard_logger import step_logger, eval_logger, ...

    step_logger.emit({
        "optimizer_step": 5,
        "window": 100,
        "loss_total": 0.1234,
        ...
    })

Log lines in the file look like::

    2026-03-06 18:30:45 INFO     [grail.trainer.step] {"optimizer_step": 5, ...}

The Grafana query extracts the JSON via::

    {logger="grail.trainer.step"} | regexp `\\] (?P<raw>\\{.+\\})` | line_format `{{.raw}}` | json
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

_LOKI_KEY_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_key(key: str) -> str:
    """Sanitize a JSON key for Loki field extraction.

    Loki's ``| json`` replaces non-alphanumeric characters with ``_``.
    We pre-sanitize with the same rule so dashboard queries match exactly.
    Example: ``pass@1`` -> ``pass_1``, ``mean@5`` -> ``mean_5``.
    """
    return _LOKI_KEY_RE.sub("_", key)


def _sanitize_value(v: Any) -> Any:
    """Ensure value is JSON-serializable and finite.

    Uses 6 significant digits (not 6 decimal places) so small values like
    LR=2e-8 are preserved instead of rounding to 0.0.
    """
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{v:.6g}")
    if isinstance(v, dict):
        return {_sanitize_key(k): _sanitize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(item) for item in v]
    return v


class DashboardLogger:
    """Thin wrapper around a stdlib logger that emits JSON payloads."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def emit(self, payload: dict[str, Any]) -> None:
        """Emit a JSON log line at INFO level."""
        sanitized = _sanitize_value(payload)
        self._logger.info(json.dumps(sanitized, separators=(",", ":")))


# Dedicated loggers for each metric category.
# Logger names become Loki labels via Promtail's regex pipeline stage.
step_logger = DashboardLogger("grail.trainer.step")
eval_logger = DashboardLogger("grail.trainer.eval")
params_logger = DashboardLogger("grail.trainer.params")
sparse_logger = DashboardLogger("grail.trainer.sparse")
upload_logger = DashboardLogger("grail.trainer.upload")
data_logger = DashboardLogger("grail.trainer.data")
state_logger = DashboardLogger("grail.trainer.state")
