"""Tests for distributed training integration into TrainerNeuron.

Verifies the Phase 1 integration: mp.Process spawn, IPC wiring,
snapshot adopt, and orchestrator lifecycle. All tests run on CPU
without requiring NCCL or multiple GPUs.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from grail.trainer.ipc import IPCChannels, create_ipc_channels
from grail.trainer.snapshot_manager import SnapshotManager

if TYPE_CHECKING:
    from grail.neurons.trainer import TrainerNeuron

# IPC cross-process tests need a consistent mp context. On Linux the default
# is "fork"; on macOS 3.12+ it's "spawn". Production code calls
# set_start_method("spawn") early.  We pick the safest option per platform:
# "fork" on Linux (avoids the SemLock fork/spawn mismatch), "spawn" elsewhere.
_MP_CTX = multiprocessing.get_context("fork" if os.name != "nt" else "spawn")


# ---------------------------------------------------------------------------
# adopt_snapshot_atomic
# ---------------------------------------------------------------------------


class TestAdoptSnapshotAtomic:
    """Test SnapshotManager.adopt_snapshot_atomic() method."""

    def test_adopt_creates_latest_from_source(self) -> None:
        """adopt_snapshot_atomic moves source_dir to snapshots/latest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Simulate FSDP2 save_full_checkpoint writing to a temp dir
            source = manager.snapshot_dir / "epoch.tmp.0"
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"fake-weights")
            (source / "config.json").write_text('{"model_type": "qwen2"}')

            metadata = {"epoch": 1, "timestamp": time.time(), "window": 100}
            manager.adopt_snapshot_atomic(source, metadata)

            # Source should be renamed (no longer exists at original path)
            assert not source.exists()

            # latest/ should exist with the files
            latest = manager.snapshot_dir / "latest"
            assert latest.exists()
            assert (latest / "model.safetensors").read_bytes() == b"fake-weights"
            assert (latest / "config.json").exists()

            # Metadata should be written
            meta_path = latest / "snapshot_metadata.json"
            assert meta_path.exists()
            saved_meta = json.loads(meta_path.read_text())
            assert saved_meta["epoch"] == 1
            assert saved_meta["window"] == 100

    def test_adopt_sets_snapshot_ready_marker(self) -> None:
        """adopt_snapshot_atomic touches the SNAPSHOT_READY marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            source = manager.snapshot_dir / "initial.tmp.123"
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"weights")

            assert not manager.check_snapshot_ready()

            manager.adopt_snapshot_atomic(source, {"epoch": 0})

            assert manager.check_snapshot_ready()

    def test_adopt_replaces_existing_latest(self) -> None:
        """adopt_snapshot_atomic atomically replaces an existing latest/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Create initial latest/
            mock_model = Mock()
            mock_model.save_pretrained = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.save_pretrained = Mock()
            manager.save_snapshot_atomic(
                mock_model, mock_tokenizer, {"epoch": 0, "timestamp": time.time()}
            )

            # Now adopt a new checkpoint
            source = manager.snapshot_dir / "epoch.tmp.1"
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"new-weights")

            manager.adopt_snapshot_atomic(source, {"epoch": 1})

            latest = manager.snapshot_dir / "latest"
            assert latest.exists()
            assert (latest / "model.safetensors").read_bytes() == b"new-weights"

            meta = json.loads((latest / "snapshot_metadata.json").read_text())
            assert meta["epoch"] == 1

    def test_adopt_raises_on_missing_source(self) -> None:
        """adopt_snapshot_atomic raises FileNotFoundError for missing source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                manager.adopt_snapshot_atomic(Path(tmpdir) / "nonexistent", {"epoch": 0})

    def test_adopt_then_copy_to_staging(self) -> None:
        """Full pipeline: adopt -> check_ready -> copy_to_staging -> cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(Path(tmpdir))

            # Simulate FSDP2 checkpoint write
            source = manager.snapshot_dir / "epoch.tmp.42"
            source.mkdir(parents=True)
            (source / "model.safetensors").write_bytes(b"checkpoint-data")
            (source / "tokenizer.json").write_text('{"type": "test"}')

            # Adopt
            manager.adopt_snapshot_atomic(source, {"epoch": 42, "timestamp": time.time()})
            assert manager.check_snapshot_ready()

            # Upload worker: copy to staging
            staging = manager.copy_snapshot_to_staging()
            assert staging.exists()
            assert (staging / "model.safetensors").read_bytes() == b"checkpoint-data"
            assert (staging / "snapshot_metadata.json").exists()
            assert not manager.check_snapshot_ready()

            # Upload worker: cleanup
            manager.cleanup_staging()
            assert not staging.exists()


# ---------------------------------------------------------------------------
# IPC across processes (verifies ForkingPickler works with spawn)
# ---------------------------------------------------------------------------


def _child_heartbeat_worker(ipc: IPCChannels, ready: multiprocessing.Event) -> None:  # type: ignore[type-arg]
    """Child process that updates heartbeat and signals ready."""
    ipc.update_heartbeat()
    ready.set()  # type: ignore[union-attr]


def _child_stop_worker(ipc: IPCChannels, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Child process that checks stop signal."""
    # Wait for stop signal (with timeout)
    for _ in range(50):
        if ipc.stop.is_set():
            result_queue.put("stopped")
            return
        time.sleep(0.01)
    result_queue.put("timeout")


def _child_snapshot_worker(
    ipc: IPCChannels,
    snapshot_manager: SnapshotManager,
) -> None:
    """Child process that saves a snapshot and queues it via IPC."""
    # Simulate writing a checkpoint
    source = snapshot_manager.snapshot_dir / "child.tmp"
    source.mkdir(parents=True, exist_ok=True)
    (source / "model.safetensors").write_bytes(b"child-weights")

    metadata = {"epoch": 1, "timestamp": time.time(), "window": 200}
    snapshot_manager.adopt_snapshot_atomic(source, metadata)

    path = snapshot_manager.get_latest_snapshot_path()
    if path:
        ipc.queue_snapshot(str(path), metadata, 200)


class TestIPCAcrossProcesses:
    """Test IPC primitives work across multiprocessing.Process (spawn method)."""

    def test_heartbeat_crosses_process_boundary(self) -> None:
        """Heartbeat updated in child process is visible in parent."""
        ctx = _MP_CTX
        ipc = create_ipc_channels()
        ready = ctx.Event()

        # Initially no heartbeat
        assert ipc.get_heartbeat_age() == float("inf")

        p = ctx.Process(target=_child_heartbeat_worker, args=(ipc, ready))
        p.start()
        ready.wait(timeout=10)
        p.join(timeout=10)

        # Heartbeat should be recent
        age = ipc.get_heartbeat_age()
        assert age < 5.0, f"Heartbeat age {age}s is too old"

    def test_stop_signal_crosses_process_boundary(self) -> None:
        """Stop event set in parent is visible in child process."""
        ctx = _MP_CTX
        ipc = create_ipc_channels()
        result_queue = ctx.Queue()

        p = ctx.Process(target=_child_stop_worker, args=(ipc, result_queue))
        p.start()

        # Set stop from parent
        time.sleep(0.05)
        ipc.stop.set()

        p.join(timeout=10)
        result = result_queue.get(timeout=5)
        assert result == "stopped"

    def test_snapshot_queue_crosses_process_boundary(self) -> None:
        """Snapshot queued in child process is consumable in parent."""
        ctx = _MP_CTX
        ipc = create_ipc_channels()

        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SnapshotManager(Path(tmpdir))

            p = ctx.Process(target=_child_snapshot_worker, args=(ipc, sm))
            p.start()
            p.join(timeout=10)

            # Parent should see the snapshot message
            msg = ipc.snapshot_queue.get(timeout=5)
            assert msg["type"] == "snapshot_ready"
            assert msg["window"] == 200
            assert "epoch" in msg["metadata"]

            # Snapshot should be at latest/
            latest = sm.get_latest_snapshot_path()
            assert latest is not None
            assert (latest / "model.safetensors").read_bytes() == b"child-weights"


# ---------------------------------------------------------------------------
# TrainerNeuron distributed spawning logic
# ---------------------------------------------------------------------------


class TestTrainerDistributedSpawning:
    """Test TrainerNeuron._start_distributed_training() logic."""

    def test_nproc_1_uses_single_gpu_path(self) -> None:
        """GRAIL_DIST_NPROC=1 or unset uses the single-GPU training process."""

        with patch.dict(os.environ, {"GRAIL_DIST_NPROC": "1"}, clear=False):
            neuron = self._make_neuron()

            with patch.object(neuron, "_start_distributed_training") as mock_dist:
                with patch("grail.neurons.trainer.run_training_process"):
                    with patch("multiprocessing.Process") as mock_proc:
                        mock_proc.return_value.start = Mock()
                        mock_proc.return_value.pid = 12345
                        neuron._start_training_process()

                mock_dist.assert_not_called()

    def test_nproc_2_uses_distributed_path(self) -> None:
        """GRAIL_DIST_NPROC=2 dispatches to _start_distributed_training."""

        with patch.dict(os.environ, {"GRAIL_DIST_NPROC": "2"}, clear=False):
            neuron = self._make_neuron()

            with patch.object(neuron, "_start_distributed_training") as mock_dist:
                neuron._start_training_process()
                mock_dist.assert_called_once_with(2)

    def test_distributed_disables_eval(self) -> None:
        """_start_distributed_training disables local evaluation."""

        neuron = self._make_neuron()
        # Force eval enabled (server .env may have it disabled)
        neuron._eval_cfg.enabled = True
        assert neuron._eval_cfg.enabled

        with patch("grail.trainer.distributed.launcher.run_distributed_training_process"):
            with patch("multiprocessing.Process") as mock_proc:
                mock_proc.return_value.start = Mock()
                mock_proc.return_value.pid = 99999
                neuron._start_distributed_training(2)

        assert not neuron._eval_cfg.enabled

    def test_distributed_spawns_n_processes(self) -> None:
        """_start_distributed_training spawns exactly N processes."""

        neuron = self._make_neuron()

        with patch("grail.trainer.distributed.launcher.run_distributed_training_process"):
            with patch("multiprocessing.Process") as mock_proc:
                mock_instance = Mock()
                mock_instance.start = Mock()
                mock_instance.pid = 10000
                mock_proc.return_value = mock_instance
                neuron._start_distributed_training(3)

        assert mock_proc.call_count == 3

        # Verify rank 0 is used as _training_process for health checks
        assert neuron._training_process is not None

    @pytest.mark.asyncio
    async def test_shutdown_terminates_all_distributed_processes(self) -> None:
        """_shutdown_processes handles all distributed processes."""

        neuron = self._make_neuron()

        # Simulate distributed processes
        procs = []
        for _ in range(3):
            p = Mock()
            p.pid = 10000
            p.is_alive.return_value = False
            p.join = Mock()
            procs.append(p)

        neuron._distributed_processes = procs
        neuron._upload_process = Mock()
        neuron._upload_process.pid = 20000
        neuron._upload_process.is_alive.return_value = False
        neuron._upload_process.join = Mock()

        await neuron._shutdown_processes()

        # All processes should have been joined
        for p in procs:
            p.join.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_detects_dead_distributed_rank(self) -> None:
        """_check_process_health logs error when a distributed rank dies."""

        neuron = self._make_neuron()

        # Rank 1 is dead
        procs = []
        for i in range(2):
            p = Mock()
            p.pid = 10000 + i
            p.is_alive.return_value = i == 0  # rank 0 alive, rank 1 dead
            procs.append(p)

        neuron._distributed_processes = procs
        neuron._upload_process = Mock()
        neuron._upload_process.is_alive.return_value = True

        with patch("grail.neurons.trainer.logger") as mock_logger:
            await neuron._check_process_health()
            # Should log error for dead rank
            mock_logger.error.assert_called()
            error_msg = mock_logger.error.call_args[0][0]
            assert "rank" in error_msg.lower() or "Distributed" in error_msg

    def _make_neuron(self) -> TrainerNeuron:
        """Create a TrainerNeuron with all dependencies mocked."""
        from grail.model.train_loading import ModelLoadSpec
        from grail.neurons.trainer import TrainerContext, TrainerNeuron

        context = TrainerContext(
            wallet=Mock(),
            credentials=Mock(),
            checkpoint_publisher=None,
            monitor=None,
            train_spec=ModelLoadSpec(mode="hf", hf_id="test/model"),
            ref_spec=ModelLoadSpec(mode="hf", hf_id="test/model"),
        )

        tmpdir = Path(tempfile.mkdtemp())

        with (
            patch("multiprocessing.set_start_method"),
            patch.dict(os.environ, {"GRAIL_CACHE_DIR": str(tmpdir)}, clear=False),
        ):
            neuron = TrainerNeuron(context)

        return neuron


# ---------------------------------------------------------------------------
# run_distributed_training_process argument validation
# ---------------------------------------------------------------------------


class TestRunDistributedTrainingProcess:
    """Test the run_distributed_training_process entry point."""

    def test_function_signature_accepts_required_args(self) -> None:
        """Verify the function has the expected signature."""
        import inspect

        from grail.trainer.distributed.launcher import run_distributed_training_process

        sig = inspect.signature(run_distributed_training_process)
        params = list(sig.parameters.keys())

        # Core params that must exist
        assert "rank" in params
        assert "world_size" in params
        assert "ipc" in params
        assert "snapshot_manager" in params
        assert "train_cfg" in params
        assert "dist_cfg" in params
        assert "credentials" in params
        assert "wallet_args" in params

    def test_ipc_channels_work_across_spawn_process(self) -> None:
        """IPCChannels work across process boundary (spawn method).

        Uses the module-level _child_heartbeat_worker which is picklable.
        """
        ctx = _MP_CTX
        ipc = create_ipc_channels()
        ready = ctx.Event()

        p = ctx.Process(target=_child_heartbeat_worker, args=(ipc, ready))
        p.start()
        ready.wait(timeout=10)
        p.join(timeout=10)

        assert ipc.get_heartbeat_age() < 5.0

    def test_snapshot_manager_is_picklable(self) -> None:
        """SnapshotManager can be pickled (required for spawn start method)."""
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SnapshotManager(Path(tmpdir))

            data = pickle.dumps(sm)
            restored = pickle.loads(data)

            assert restored.cache_root == sm.cache_root

    def test_training_config_is_picklable(self) -> None:
        """TrainingConfig can be pickled (required for spawn start method)."""
        import pickle

        from grail.trainer.config import TrainingConfig

        cfg = TrainingConfig()
        cfg.grpo_variant = "grpo"

        data = pickle.dumps(cfg)
        restored = pickle.loads(data)

        assert restored.grpo_variant == "grpo"

    def test_distributed_config_is_picklable(self) -> None:
        """DistributedConfig can be pickled (required for spawn start method)."""
        import pickle

        from grail.trainer.distributed.config import DistributedConfig

        cfg = DistributedConfig(tp_degree=1)

        data = pickle.dumps(cfg)
        restored = pickle.loads(data)

        assert restored.tp_degree == 1


# ---------------------------------------------------------------------------
# _run_training IPC integration (mocked, no NCCL)
# ---------------------------------------------------------------------------


class TestRunTrainingIPCIntegration:
    """Test that _run_training() properly uses IPC when provided."""

    @pytest.mark.asyncio
    async def test_stop_flag_breaks_loop(self) -> None:
        """Training loop exits when ipc.stop is set before first iteration.

        Uses rank=1 (non-zero) to skip rank 0's chain initialization which
        requires a real subtensor connection.
        """
        from grail.trainer.distributed.launcher import _run_training

        ipc = create_ipc_channels()
        ipc.stop.set()  # Pre-set stop

        service = Mock()
        device = Mock()

        # rank=1 skips chain init. _should_stop() returns True immediately.
        with patch("grail.trainer.distributed.launcher.dist") as mock_dist:
            mock_dist.broadcast = Mock()
            await _run_training(
                service=service,
                rank=1,
                device=device,
                train_config=Mock(),
                snapshot_manager=None,
                wallet=None,
                credentials=None,
                stop_event=ipc.stop,
                test_mode=False,
                ipc=ipc,
            )

        # Should not have called _train_epoch (loop never entered)
        service._train_epoch.assert_not_called()
