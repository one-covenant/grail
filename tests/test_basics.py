from __future__ import annotations

from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import (
    ROLLOUTS_PER_PROBLEM,
    SUPERLINEAR_EXPONENT,
    WINDOW_LENGTH,
)


def test_constants_are_reasonable() -> None:
    # Basic sanity checks on shared constants used broadly across the system
    assert isinstance(WINDOW_LENGTH, int) and WINDOW_LENGTH > 0
    assert isinstance(ROLLOUTS_PER_PROBLEM, int) and ROLLOUTS_PER_PROBLEM >= 1
    assert isinstance(SUPERLINEAR_EXPONENT, float)
    assert SUPERLINEAR_EXPONENT > 1.0


def test_monitoring_config_disabled_env() -> None:
    # With defaults from conftest, monitoring should be effectively disabled
    assert MonitoringConfig.is_monitoring_enabled() is False
    cfg = MonitoringConfig.from_environment()
    # Ensure the parsed configuration reflects disabled/NULL backend
    assert cfg["backend_type"] == "null"
    assert cfg["mode"] in {"disabled", "online", "offline"}
