"""Sanity checks for protocol constants and test environment defaults."""

from __future__ import annotations

from grail.monitoring.config import MonitoringConfig
from grail.protocol.constants import (
    ROLLOUTS_PER_PROBLEM,
    SUPERLINEAR_EXPONENT,
    WINDOW_LENGTH,
)


def test_constants_are_reasonable() -> None:
    """Protocol constants have valid types and ranges (catches accidental edits)."""
    assert isinstance(WINDOW_LENGTH, int) and WINDOW_LENGTH > 0
    assert isinstance(ROLLOUTS_PER_PROBLEM, int) and ROLLOUTS_PER_PROBLEM >= 1
    assert isinstance(SUPERLINEAR_EXPONENT, float)
    assert SUPERLINEAR_EXPONENT > 1.0


def test_monitoring_config_disabled_env() -> None:
    """Conftest autouse fixture disables monitoring; verify it takes effect."""
    assert MonitoringConfig.is_monitoring_enabled() is False
    cfg = MonitoringConfig.from_environment()
    assert cfg["backend_type"] == "null"
    assert cfg["mode"] in {"disabled", "online", "offline"}
