"""Distributed GRPO training launcher.

Two entry points:

1. ``torchrun --nproc_per_node=N -m grail.trainer.distributed.launcher``
   Standalone mode: ``main()`` handles process group init, model loading, and
   the full training loop.  IPC is unavailable (torchrun uses subprocess.Popen).

2. ``run_distributed_training_process(rank, world_size, ...)``
   Orchestrated mode: called as ``multiprocessing.Process`` target from
   ``TrainerNeuron._start_distributed_training()``.  Receives ``IPCChannels``
   and ``SnapshotManager`` for heartbeat, stop signal, and snapshot upload.

Both paths call the same ``_run_training()`` loop.  When ``ipc`` is provided,
rank 0 updates the heartbeat, checks the stop signal, saves an initial
checkpoint, and queues snapshots for the upload worker.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import pickle
import re
import subprocess
import sys
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import bittensor as bt
import torch
import torch.distributed as dist

from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.credentials import load_r2_credentials
from grail.infrastructure.network import create_subtensor
from grail.protocol.constants import WINDOW_LENGTH
from grail.shared.config import NETUID
from grail.trainer.algorithms.grpo import GRPOGroup, load_grpo_groups
from grail.trainer.config import TrainingConfig
from grail.trainer.distributed.config import DistributedConfig
from grail.trainer.distributed.parallelism import create_device_mesh
from grail.trainer.distributed.training_service import DistributedTrainingService
from grail.trainer.ipc import IPCChannels
from grail.trainer.replay_buffer import create_replay_buffer
from grail.trainer.snapshot_manager import SnapshotManager

logger = logging.getLogger(__name__)

# Training loop constants
NO_DATA_SLEEP_SECONDS = 60
LOOP_ERROR_SLEEP_SECONDS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_model_params_b(model_id: str) -> float:
    """Extract approximate model size in billions from a HuggingFace model ID."""
    match = re.search(r"(\d+(?:\.\d+)?)[bB]", model_id)
    if match:
        return float(match.group(1))
    return 7.0


def _resolve_tp_degree(dist_cfg: DistributedConfig, world_size: int) -> int:
    """Determine TP degree based on strategy and config.

    DDP and DILOCO do not support tensor parallelism (each GPU holds a full
    model replica), so tp_degree is forced to 1. For FSDP2, use the configured
    or auto-detected value.
    """
    if dist_cfg.strategy in ("ddp", "diloco"):
        return 1

    if dist_cfg.tp_degree > 0:
        return dist_cfg.tp_degree

    model_id = os.getenv("GRAIL_TRAIN_MODEL_ID", "")
    params_b = _estimate_model_params_b(model_id) if model_id else 7.0
    tp_degree, _ = DistributedConfig.auto_detect_parallelism(world_size, params_b)
    return tp_degree


def _configure_logging(rank: int, verbosity: int = 1) -> None:
    """Configure logging for the distributed worker process."""
    if rank == 0:
        level = logging.DEBUG if verbosity >= 2 else logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        format=f"[rank {rank}] %(asctime)s %(name)s %(levelname)s %(message)s",
        level=level,
        force=True,
    )


def _broadcast_groups(
    groups: list[GRPOGroup] | None,
    rank: int,
    device: torch.device,
) -> list[GRPOGroup]:
    """Broadcast GRPO groups from rank 0 to all ranks via pickle + NCCL."""
    if rank == 0:
        buf = pickle.dumps(groups or [])
        size_tensor = torch.tensor([len(buf)], dtype=torch.int64, device=device)
    else:
        size_tensor = torch.tensor([0], dtype=torch.int64, device=device)

    dist.broadcast(size_tensor, src=0)
    nbytes = int(size_tensor.item())

    if nbytes == 0:
        return []

    if rank == 0:
        data_tensor = torch.frombuffer(bytearray(buf), dtype=torch.uint8).to(device)  # type: ignore[arg-type]
    else:
        data_tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)

    dist.broadcast(data_tensor, src=0)

    if rank != 0:
        return pickle.loads(data_tensor.cpu().numpy().tobytes())  # type: ignore[return-value]

    assert groups is not None
    return groups


# ---------------------------------------------------------------------------
# Production training loop
# ---------------------------------------------------------------------------


async def _run_training(
    service: DistributedTrainingService,
    rank: int,
    device: torch.device,
    train_config: TrainingConfig,
    snapshot_manager: SnapshotManager | None,
    wallet: bt.wallet | None,
    credentials: Any | None,
    stop_event: Event,
    test_mode: bool = False,
    ipc: IPCChannels | None = None,
    monitor: Any | None = None,
) -> None:
    """Main distributed training loop.

    Rank 0: chain queries, R2 data loading, replay buffer, snapshots.
    All ranks: FSDP2 training (forward/backward/optimizer step).

    When *ipc* is provided (orchestrated mode), rank 0 also:
    - Updates the heartbeat each iteration
    - Checks ``ipc.stop`` for graceful shutdown
    - Saves an initial checkpoint before the first epoch
    - Queues snapshots for the upload worker via ``ipc.queue_snapshot()``
    - Logs training metrics to W&B via *monitor*
    """
    subtensor = None
    chain_manager = None
    replay_buffer = None
    last_loaded_window = -1
    epoch_counter = 0
    current_block = 0
    current_window = 0

    if rank == 0:
        from types import SimpleNamespace

        subtensor = await create_subtensor(resilient=True)
        metagraph = await subtensor.metagraph(NETUID)  # type: ignore[misc]
        chain_config = SimpleNamespace(netuid=NETUID)
        assert wallet is not None
        assert credentials is not None
        chain_manager = GrailChainManager(
            chain_config,
            wallet,
            metagraph,
            subtensor,  # type: ignore[arg-type]  # ResilientSubtensor compat
            credentials,  # type: ignore[arg-type]
        )
        await chain_manager.initialize()
        replay_buffer = create_replay_buffer(
            buffer_type="recency_weighted",
            max_windows=train_config.replay_buffer_max_windows,
            recent_window_fraction=train_config.replay_buffer_recent_fraction,
            decay_factor=train_config.replay_buffer_decay_factor,
        )
        logger.info("Rank 0: chain + replay buffer initialized")

    # ── Initial checkpoint bootstrap (orchestrated mode only) ──
    is_fsdp2 = service.dist_config.strategy == "fsdp2"
    if ipc is not None:
        try:
            if is_fsdp2:
                # FSDP2: collective save (all ranks must participate)
                from grail.trainer.distributed.checkpoint import save_full_checkpoint

                if rank == 0 and snapshot_manager is not None:
                    initial_dir = (
                        snapshot_manager.cache_root / "snapshots" / f"initial.tmp.{os.getpid()}"
                    )
                    save_full_checkpoint(service.train_model, service.tokenizer, initial_dir, rank)
                else:
                    save_full_checkpoint(
                        service.train_model, service.tokenizer, Path("/tmp/unused"), rank
                    )
            else:
                # DDP/DILOCO: rank-0-only save (no collective, no barrier)
                initial_dir = None
                if rank == 0 and snapshot_manager is not None:
                    from grail.trainer.distributed.checkpoint import save_ddp_checkpoint

                    initial_dir = (
                        snapshot_manager.cache_root / "snapshots" / f"initial.tmp.{os.getpid()}"
                    )
                    save_ddp_checkpoint(service.train_model, service.tokenizer, initial_dir, rank)

            # Rank 0: adopt and queue snapshot for upload
            if rank == 0 and snapshot_manager is not None:
                initial_dir_path = (
                    snapshot_manager.cache_root / "snapshots" / f"initial.tmp.{os.getpid()}"
                )
                snapshot_metadata = {
                    "epoch": 0,
                    "timestamp": time.time(),
                    "status": "initial_upload",
                    "window": current_window,
                }
                snapshot_manager.adopt_snapshot_atomic(initial_dir_path, snapshot_metadata)
                snapshot_path = snapshot_manager.get_latest_snapshot_path()
                if snapshot_path:
                    ipc.queue_snapshot(str(snapshot_path), snapshot_metadata, current_window)
                logger.info("Initial checkpoint saved and queued for upload")
        except Exception as exc:
            if rank == 0:
                logger.error("Initial checkpoint failed: %s", exc)

    def _should_stop() -> bool:
        if stop_event.is_set():
            return True
        if ipc is not None and ipc.stop.is_set():
            return True
        return False

    while not _should_stop():
        try:
            # ── Rank 0: heartbeat + stop broadcast ──
            if rank == 0 and ipc is not None:
                ipc.update_heartbeat()

            # Broadcast stop signal from rank 0 to all ranks
            stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
            if rank == 0 and _should_stop():
                stop_flag.fill_(1)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                logger.info("Stop signal received (rank=%d)", rank)
                break

            # ── Rank 0: fetch data ──
            groups: list[GRPOGroup] | None = None
            if rank == 0:
                assert subtensor is not None
                assert replay_buffer is not None
                assert chain_manager is not None

                try:
                    current_block = await subtensor.get_current_block()  # type: ignore[misc]
                    current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH
                except Exception as exc:
                    logger.warning("Failed to get current block: %s", exc)

                if monitor is not None:
                    monitor.set_block_context(current_block, current_window)

                target_data_window = current_window - WINDOW_LENGTH

                # Load new data when window changes
                if target_data_window != last_loaded_window and target_data_window >= 0:
                    try:
                        # Refresh metagraph for trust computation
                        metagraph = await subtensor.metagraph(NETUID)  # type: ignore[misc]

                        # Get trusted miners
                        from grail.trainer.trust import (
                            get_trust_list_from_validator,
                            get_trusted_miner_hotkeys,
                        )

                        trusted: set[str] | None = None
                        try:
                            trusted = await get_trust_list_from_validator(
                                metagraph, chain_manager, target_data_window
                            )
                        except Exception:
                            pass

                        if not trusted:
                            trusted = await get_trusted_miner_hotkeys(
                                metagraph,
                                min_trusted_miners=train_config.min_trusted_miners,
                                timeout=5.0,
                            )

                        if trusted:
                            new_groups = await load_grpo_groups(
                                window=target_data_window,
                                advantage_tolerance=train_config.group_adv_sum_tolerance,
                                trusted_miner_hotkeys=trusted,
                                credentials=credentials,
                                chain_manager=chain_manager,
                                config=train_config,
                                monitor=monitor,
                                eos_token_id=service.tokenizer.eos_token_id,
                            )
                            if new_groups:
                                replay_buffer.add_window(target_data_window, new_groups)
                                logger.info(
                                    "Loaded %d groups from window %d",
                                    len(new_groups),
                                    target_data_window,
                                )
                                last_loaded_window = target_data_window
                    except Exception as exc:
                        logger.exception("Data loading failed: %s", exc)

                # Sample from replay buffer
                seed = current_window + epoch_counter
                groups = replay_buffer.sample_groups(
                    train_config.replay_buffer_max_groups_per_epoch, seed
                )
                if not groups:
                    logger.warning("No training data, sleeping %ds", NO_DATA_SLEEP_SECONDS)

            # ── Broadcast groups to all ranks ──
            groups = _broadcast_groups(groups, rank, device)

            if not groups:
                await asyncio.sleep(NO_DATA_SLEEP_SECONDS)
                continue

            if rank == 0:
                total_rollouts = sum(len(g.rollouts) for g in groups)
                logger.info(
                    "Epoch %d: %d groups, %d rollouts",
                    epoch_counter + 1,
                    len(groups),
                    total_rollouts,
                )

            # ── All ranks: train ──
            t0 = time.time()
            metrics = await service._train_epoch(  # noqa: SLF001
                groups,
                monitor=monitor if rank == 0 else None,
                window=current_window,
            )
            elapsed = time.time() - t0

            # ── Save checkpoint ──
            try:
                strategy = service.dist_config.strategy

                # DILOCO: only save model weights and update snapshots/latest
                # after outer sync, when local params == global consensus.
                # Between outer syncs each GPU has diverged params; overwriting
                # snapshots/latest would expose non-consensus weights to the
                # upload worker and eval resource loader.
                diloco_synced = service._diloco_sync_happened  # noqa: SLF001
                should_save_weights = strategy != "diloco" or diloco_synced

                temp_ckpt: Path | None = None
                if should_save_weights:
                    if rank == 0 and snapshot_manager is not None:
                        temp_ckpt = (
                            snapshot_manager.cache_root / "snapshots" / f"epoch.tmp.{epoch_counter}"
                        )

                    if strategy == "fsdp2":
                        # FSDP2: collective gather (all ranks must participate)
                        from grail.trainer.distributed.checkpoint import save_full_checkpoint

                        save_full_checkpoint(
                            service.train_model,
                            service.tokenizer,
                            temp_ckpt or Path("/tmp/fsdp2_unused"),
                            rank,
                        )
                    elif rank == 0 and temp_ckpt is not None:
                        # DDP/DILOCO: rank 0 saves directly, no collective
                        from grail.trainer.distributed.checkpoint import save_ddp_checkpoint

                        save_ddp_checkpoint(
                            service.train_model,
                            service.tokenizer,
                            temp_ckpt,
                            rank,
                        )

                # Rank 0: adopt snapshot and queue for upload (only when
                # weights were saved, i.e. not a DILOCO non-sync epoch)
                if rank == 0 and snapshot_manager is not None and temp_ckpt is not None:
                    snapshot_metadata = {
                        "epoch": epoch_counter + 1,
                        "timestamp": time.time(),
                        "window": current_window,
                        "metrics": {
                            "loss_total": metrics.get("loss_total", 0.0),
                        },
                    }
                    snapshot_manager.adopt_snapshot_atomic(temp_ckpt, snapshot_metadata)

                    if ipc is not None:
                        snapshot_path = snapshot_manager.get_latest_snapshot_path()
                        if snapshot_path:
                            ipc.queue_snapshot(
                                str(snapshot_path),
                                snapshot_metadata,
                                current_window,
                            )

                    upload_status = "uploaded"
                else:
                    upload_status = "skipped (awaiting DILOCO sync)"

                # Always save resume state for crash recovery (rank 0 only).
                # This is cheap (optimizer + scheduler + DILOCO state, no model
                # weights) and ensures we can restart from the last sync point.
                if rank == 0:
                    try:
                        service._save_resume_checkpoint()  # noqa: SLF001
                    except Exception as resume_exc:
                        logger.warning("Resume checkpoint save failed: %s", resume_exc)

                if rank == 0:
                    logger.info(
                        "Epoch %d complete in %.1fs: loss=%.4f (%s)",
                        epoch_counter + 1,
                        elapsed,
                        metrics.get("loss_total", 0.0),
                        upload_status,
                    )

            except Exception as exc:
                if rank == 0:
                    logger.error("Checkpoint save failed: %s", exc)

            # ── Rank 0: log epoch-level metrics to W&B ──
            if rank == 0 and monitor is not None and metrics:
                try:
                    await monitor.log_gauge("training/epoch_duration", elapsed)
                    await monitor.log_gauge("training/epoch_loss", metrics.get("loss_total", 0.0))
                    lr = (
                        service.scheduler.get_last_lr()[0] if service.scheduler else train_config.lr
                    )
                    await monitor.log_gauge("training/lr", lr)
                    await monitor.log_gauge("training/epoch", epoch_counter + 1)
                    await monitor.flush_metrics()
                except Exception as exc:
                    logger.debug("Failed to log epoch metrics to W&B: %s", exc)

            if service.scheduler is not None:
                service.scheduler.step()
            epoch_counter += 1
            service.epoch_counter = epoch_counter

            dist.barrier()

        except asyncio.CancelledError:
            logger.info("Training loop cancelled (rank=%d)", rank)
            break
        except Exception as exc:
            logger.exception("Training loop error (rank=%d): %s", rank, exc)
            # All ranks must hit the same collectives, so broadcast a "skip" signal
            try:
                _broadcast_groups(None if rank == 0 else None, rank, device)
            except Exception:
                pass
            await asyncio.sleep(LOOP_ERROR_SLEEP_SECONDS)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``torchrun --nproc_per_node=N -m grail.trainer.distributed.launcher``."""
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    dist.init_process_group(backend="cuda:nccl,cpu:gloo")

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        verbosity = int(os.getenv("GRAIL_VERBOSITY", "2"))
        _configure_logging(rank, verbosity)

        train_cfg = TrainingConfig()
        dist_cfg = DistributedConfig.from_env()
        dist_cfg.validate(world_size)

        tp_degree = _resolve_tp_degree(dist_cfg, world_size)
        mesh = create_device_mesh(world_size, tp_degree)
        device = torch.device("cuda", local_rank)

        if rank == 0:
            logger.info(
                "Distributed launcher: strategy=%s, rank=%d, world=%d, tp=%d, dp=%d",
                dist_cfg.strategy,
                rank,
                world_size,
                tp_degree,
                world_size // tp_degree,
            )

        # Safety: override DAPO variant (its inner-loop gather causes NCCL mismatch)
        train_cfg.grpo_variant = "grpo"

        # Build service
        model_id = os.getenv("GRAIL_TRAIN_MODEL_ID", "Qwen/Qwen3-8B")
        service = DistributedTrainingService(
            train_config=train_cfg,
            dist_config=dist_cfg,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            mesh=mesh,
            model_name_or_path=model_id,
        )

        logger.info("Initializing resources (rank=%d)...", rank)
        service._initialize_resources()  # noqa: SLF001

        # Rank 0 only: wallet, credentials, snapshot manager
        wallet = None
        credentials = None
        snapshot_manager = None

        if rank == 0:
            coldkey = os.getenv("BT_WALLET_COLD", "default")
            hotkey = os.getenv("BT_WALLET_HOT", "default")
            wallet = bt.wallet(name=coldkey, hotkey=hotkey)
            credentials = load_r2_credentials()

            snapshot_dir = Path(
                os.getenv(
                    "GRAIL_SNAPSHOT_DIR",
                    os.path.expanduser("~/grail_cache/checkpoints/async_trainer/snapshots"),
                )
            )
            snapshot_manager = SnapshotManager(snapshot_dir)
            logger.info("Rank 0: wallet=%s/%s, snapshots=%s", coldkey, hotkey, snapshot_dir)

        stop_event = multiprocessing.Event()
        dist.barrier()

        test_mode = os.getenv("GRAIL_TEST_MODE", "0") == "1"
        asyncio.run(
            _run_training(
                service,
                rank,
                device,
                train_cfg,
                snapshot_manager,
                wallet,
                credentials,
                stop_event,
                test_mode,
            )
        )

    except KeyboardInterrupt:
        if dist.is_initialized():
            logger.info("Interrupted (rank=%d)", dist.get_rank())
    except Exception as exc:
        r = dist.get_rank() if dist.is_initialized() else -1
        logger.exception("Launcher failed (rank=%d): %s", r, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Orchestrated entry point (called from TrainerNeuron via mp.Process)
# ---------------------------------------------------------------------------


def run_distributed_training_process(
    rank: int,
    world_size: int,
    ipc: IPCChannels,
    snapshot_manager: SnapshotManager,
    train_cfg: TrainingConfig,
    dist_cfg: DistributedConfig,
    credentials: Any,
    wallet_args: dict[str, str],
    monitor_config: dict[str, Any],
    verbosity: int = 1,
    test_mode: bool = False,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """Entry point for distributed training via ``multiprocessing.Process``.

    Called by ``TrainerNeuron._start_distributed_training()``.  Unlike
    ``main()`` (torchrun), this receives ``IPCChannels`` and
    ``SnapshotManager`` directly so the upload worker can consume
    snapshots and the orchestrator can monitor heartbeat/stop.

    Args:
        rank: Process rank (0-based).
        world_size: Total number of training processes.
        ipc: IPC channels shared with the orchestrator and upload worker.
        snapshot_manager: Snapshot storage manager (rank 0 writes, upload worker reads).
        train_cfg: Training hyperparameters.
        dist_cfg: Distributed parallelism config (tp_degree, etc.).
        credentials: R2 credentials (rank 0 only, for chain manager).
        wallet_args: Serialized wallet (name, hotkey, path).
        monitor_config: Monitoring/W&B configuration.
        verbosity: Logging verbosity (0=silent, 1=INFO, >=2=DEBUG).
        test_mode: If True, train only on TRAINER_UID data.
        master_addr: NCCL rendezvous address.
        master_port: NCCL rendezvous port.
    """
    # ── Environment setup (replaces torchrun's env injection) ──
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_TIMEOUT"] = "120"

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    _configure_logging(rank, verbosity)

    try:
        dist.init_process_group(backend="cuda:nccl,cpu:gloo")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

        logger.info(
            "Distributed worker started: rank=%d, world=%d, master=%s:%d",
            rank,
            world_size,
            master_addr,
            master_port,
        )

        dist_cfg.validate(world_size)
        tp_degree = _resolve_tp_degree(dist_cfg, world_size)
        mesh = create_device_mesh(world_size, tp_degree)

        # Safety: override DAPO variant (its inner-loop gather causes NCCL mismatch)
        train_cfg.grpo_variant = "grpo"

        model_id = os.getenv("GRAIL_TRAIN_MODEL_ID", "Qwen/Qwen3-8B")
        service = DistributedTrainingService(
            train_config=train_cfg,
            dist_config=dist_cfg,
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            mesh=mesh,
            model_name_or_path=model_id,
        )

        logger.info("Initializing resources (rank=%d)...", rank)
        service._initialize_resources()  # noqa: SLF001

        # Rank 0: wallet and credentials for chain data loading
        wallet = None
        creds = None
        sm = None

        if rank == 0:
            wallet = bt.wallet(
                name=wallet_args["name"],
                hotkey=wallet_args["hotkey"],
                path=wallet_args.get("path", "~/.bittensor/wallets"),
            )
            creds = credentials
            sm = snapshot_manager

        dist.barrier()

        async def _run_with_monitor() -> None:
            # Rank 0: init W&B monitoring (shared mode, attaches to orchestrator's run)
            rank0_monitor = None
            if rank == 0 and monitor_config:
                from grail.monitoring import initialize_subprocess_monitoring

                rank0_monitor = await initialize_subprocess_monitoring(
                    monitor_config,
                    "distributed_rank0",
                    test_connection=True,
                )
                if rank0_monitor:
                    logger.info("Rank 0: W&B monitoring initialized")
                else:
                    logger.warning("Rank 0: W&B monitoring init failed, metrics will not be logged")

            try:
                await _run_training(
                    service,
                    rank,
                    device,
                    train_cfg,
                    sm,
                    wallet,
                    creds,
                    ipc.stop,
                    test_mode,
                    ipc,
                    monitor=rank0_monitor,
                )
            finally:
                if rank0_monitor is not None:
                    try:
                        await rank0_monitor.shutdown()
                    except Exception as exc:
                        logger.debug("Monitor shutdown error: %s", exc)

        asyncio.run(_run_with_monitor())

    except KeyboardInterrupt:
        if dist.is_initialized():
            logger.info("Interrupted (rank=%d)", dist.get_rank())
    except Exception as exc:
        r = dist.get_rank() if dist.is_initialized() else rank
        logger.exception("Distributed worker failed (rank=%d): %s", r, exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Subprocess launcher (kept for standalone torchrun usage)
# ---------------------------------------------------------------------------


def launch_distributed(
    nproc: int,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch distributed training as a subprocess via torchrun."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        "--master_port=29500",
        "-m",
        "grail.trainer.distributed.launcher",
    ]

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    if "PYTORCH_CUDA_ALLOC_CONF" not in merged_env:
        merged_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    logger.info("Launching distributed training: nproc=%d", nproc)
    return subprocess.Popen(cmd, env=merged_env)


if __name__ == "__main__":
    main()
