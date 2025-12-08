"""Integration tests for the `grail train` CLI.

These tests execute the CLI entrypoint with lightweight fakes to ensure
the end-to-end orchestration (context building → trainer neuron →
training/upload worker processes) remains functional while keeping runtime
under a second.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from click import Command, Context

from grail.cli import train as train_cli

# ------------------------------------------------------------------------------
# Helper Fakes
# ------------------------------------------------------------------------------


class _FakeWallet:
    """Minimal wallet stub used by the CLI tests."""

    def __init__(self, name: str, hotkey: str) -> None:
        self.name = name
        self.hotkey_str = hotkey
        self.path = "/tmp"
        self.hotkey = SimpleNamespace(ss58_address="fake_ss58")


class _FakeCredentials:
    """Lightweight credentials container used by tests."""

    access_key: str = "fake_access"
    secret_key: str = "fake_secret"


class _FakeCheckpointPublisher:
    """CheckpointPublisher stub (satisfies constructor only)."""

    def __init__(self, *, credentials: Any, wallet: Any) -> None:  # pragma: no cover - trivial
        self.credentials = credentials
        self.wallet = wallet


class _FakeArtifact:
    """Model/tokenizer stub with save_pretrained + CPU/GPU helpers."""

    def __init__(self, name: str) -> None:
        self.name_or_path = name

    def save_pretrained(self, path: str | Path, **_: Any) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")

    # The real training process moves models between CPU/GPU. Return self.
    def cpu(self) -> _FakeArtifact:  # pragma: no cover - trivial
        return self

    def cuda(self) -> _FakeArtifact:  # pragma: no cover - trivial
        return self


def _write_snapshot(snapshot_manager: Any, status: str = "initial_upload") -> None:
    """Create a minimal snapshot on disk for upload worker tests."""
    fake_model = _FakeArtifact("fake-train")
    fake_tokenizer = _FakeArtifact("fake-tokenizer")
    snapshot_manager.save_snapshot_atomic(
        fake_model,
        fake_tokenizer,
        {"epoch": 0, "timestamp": time.time(), "status": status},
    )


def _fake_run_training_process(
    train_spec: Any,
    ref_spec: Any,
    config: Any,
    snapshot_manager: Any,
    credentials: Any,
    wallet_args: dict[str, str],
    monitor_config: dict[str, Any],
    ipc: Any,
    verbosity: int = 1,
) -> None:
    """Simulate the training child process."""
    _ = (train_spec, ref_spec, config, credentials, wallet_args, monitor_config, verbosity)
    _write_snapshot(snapshot_manager)
    snapshot_manager.set_training_heartbeat()
    while not ipc.stop.is_set():
        snapshot_manager.set_training_heartbeat()
        time.sleep(0.05)


def _fake_run_upload_worker(
    snapshot_manager: Any,
    credentials: Any,
    wallet_args: dict[str, str],
    ipc: Any,
    poll_interval: int = 30,
    verbosity: int = 1,
) -> None:
    """Simulate upload worker copying a snapshot once."""
    _ = (credentials, wallet_args, poll_interval, verbosity)
    deadline = time.time() + 2
    while time.time() < deadline and not ipc.stop.is_set():
        if snapshot_manager.check_snapshot_ready():
            snapshot_manager.copy_snapshot_to_staging()
            snapshot_manager.cleanup_staging()
            break
        time.sleep(0.05)
    ipc.stop.set()


async def _fake_initialize_chain_manager(self: Any) -> None:  # pragma: no cover - simple stub
    self._context.chain_manager = object()


async def _fast_orchestration_loop(self: Any) -> None:
    """Wait for heartbeat then trigger cooperative shutdown."""
    deadline = time.time() + 5
    while (
        self._snapshot_manager.get_training_heartbeat_age() == float("inf")
        and time.time() < deadline
    ):
        await asyncio.sleep(0.05)

    snapshot_path = self._snapshot_manager.get_latest_snapshot_path()
    assert snapshot_path is not None and snapshot_path.exists()

    self._ipc.stop.set()
    self.stop_event.set()


# ------------------------------------------------------------------------------
# Shared fixture for CLI tests
# ------------------------------------------------------------------------------


@pytest.fixture
def train_cli_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Patch expensive dependencies so the CLI can run in a unit test."""

    # Minimal wallet/credential stubs
    monkeypatch.setattr(train_cli.bt, "wallet", lambda name, hotkey: _FakeWallet(name, hotkey))
    monkeypatch.setattr(train_cli, "load_r2_credentials", lambda: _FakeCredentials())
    monkeypatch.setattr(train_cli, "CheckpointPublisher", _FakeCheckpointPublisher)
    monkeypatch.setattr(train_cli, "get_monitoring_manager", lambda: None)

    # Deterministic environment variables for ModelLoadSpec parsing
    monkeypatch.setenv("GRAIL_TRAIN_MODEL_MODE", "hf")
    monkeypatch.setenv("GRAIL_TRAIN_MODEL_ID", "fake-train-id")
    monkeypatch.setenv("GRAIL_REF_MODEL_MODE", "hf")
    monkeypatch.setenv("GRAIL_REF_MODEL_ID", "fake-ref-id")
    monkeypatch.setenv("BT_WALLET_COLD", "test")
    monkeypatch.setenv("BT_WALLET_HOT", "test_hk")
    monkeypatch.setenv("GRAIL_LOG_FILE", "")  # disable rotating file handler noise

    # Keep snapshot/cache writes inside the temp directory
    cache_root = tmp_path / "cli-cache"
    monkeypatch.setattr(
        "grail.neurons.trainer.default_checkpoint_cache_root",
        lambda: cache_root,
    )

    return {"cache_root": cache_root}


# ------------------------------------------------------------------------------
# CLI Smoke Test
# ------------------------------------------------------------------------------


def test_train_cli_smoke(
    monkeypatch: pytest.MonkeyPatch, train_cli_test_env: dict[str, Path]
) -> None:
    """Full CLI smoke test with lightweight training/upload workers."""

    cache_root = train_cli_test_env["cache_root"]
    snapshots_dir = cache_root / "async_trainer" / "snapshots"
    heartbeats = cache_root / "async_trainer" / "locks" / "TRAINING_HEARTBEAT"

    monkeypatch.setattr("grail.neurons.trainer.run_training_process", _fake_run_training_process)
    monkeypatch.setattr("grail.neurons.trainer.run_upload_worker", _fake_run_upload_worker)
    monkeypatch.setattr(
        "grail.neurons.trainer.TrainerNeuron._initialize_chain_manager",
        _fake_initialize_chain_manager,
    )
    monkeypatch.setattr(
        "grail.neurons.trainer.TrainerNeuron._orchestration_loop",
        _fast_orchestration_loop,
    )

    # Create a mock Typer context with parent context for verbosity
    click_ctx = Context(Command("train"))
    typer_ctx = typer.Context(click_ctx)
    typer_ctx.parent = SimpleNamespace(params={"verbose": 1})
    train_cli.train(ctx=typer_ctx)

    # Assertions: snapshot + metadata exist and heartbeat file was written
    snapshot_path = snapshots_dir / "latest"
    metadata_file = snapshot_path / "snapshot_metadata.json"
    assert snapshot_path.exists()
    assert metadata_file.exists()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["status"] == "initial_upload"
    assert heartbeats.exists()


# ------------------------------------------------------------------------------
# CLI Regression Test: Startup Error Handling
# ------------------------------------------------------------------------------


def test_train_cli_reports_env_errors(
    monkeypatch: pytest.MonkeyPatch, train_cli_test_env: dict[str, Path]
) -> None:
    """CLI should exit with code 1 when model env parsing fails."""

    # Cause parse_train_env to raise so we exercise the error path
    def raise_value_error() -> Any:
        raise ValueError("bad env")

    monkeypatch.setattr("grail.cli.train.parse_train_env", raise_value_error)

    # Create a mock Typer context with parent context for verbosity
    click_ctx = Context(Command("train"))
    typer_ctx = typer.Context(click_ctx)
    typer_ctx.parent = SimpleNamespace(params={"verbose": 1})

    with pytest.raises(typer.Exit) as excinfo:
        train_cli.train(ctx=typer_ctx)

    assert excinfo.value.exit_code == 1
