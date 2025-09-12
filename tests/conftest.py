import pathlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide safe default environment for tests.

    Keeps tests deterministic and avoids accidental network/monitoring.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("GRAIL_MONITORING_BACKEND", "null")
    monkeypatch.setenv("BT_NETWORK", "test")
    monkeypatch.setenv("NETUID", "1")
    monkeypatch.setenv("GRAIL_ROLLOUTS_PER_PROBLEM", "4")
    # Ensure project root is on path when running from different CWDs
    root = str(pathlib.Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
