from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from types import SimpleNamespace

import bittensor as bt
import torch

from grail.cli.mine import (
    MiningTimers,
    generate_rollouts_for_window,
    get_conf,
    get_window_randomness,
    has_time_for_next_generation,
    upload_inferences_with_metrics,
)
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import TRAINER_UID, WINDOW_LENGTH
from grail.shared.subnet import get_own_uid_on_subnet
from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    # (heartbeat is now handled by BaseNeuron.heartbeat())

    async def run(self) -> None:
        """Main mining loop mirrored from the CLI implementation."""
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")

        # Model and tokenizer will be loaded from checkpoint
        model = None
        tokenizer = None
        current_checkpoint_window: int | None = None
        window_wait_tracker = WindowWaitTracker(log_interval_secs=120)

        async def _run() -> None:
            nonlocal model, tokenizer, current_checkpoint_window
            last_window_start = -1
            timers = MiningTimers()

            # Load R2 credentials
            try:
                credentials = load_r2_credentials()
                logger.info("✅ Loaded R2 credentials")
            except Exception as e:
                logger.error(f"Failed to load R2 credentials: {e}")
                raise

            # Initialize heartbeat (watchdog will monitor for stalls)
            self.heartbeat()
            logger.info("✅ Initialized watchdog heartbeat")

            # Get subtensor and metagraph for chain manager (use shared base method)
            subtensor = await self.get_subtensor()
            self.heartbeat()
            netuid = int(get_conf("BT_NETUID", get_conf("NETUID", 200)))
            metagraph = await subtensor.metagraph(netuid)
            self.heartbeat()

            # Initialize chain manager for credential commitments
            config = SimpleNamespace(netuid=netuid)
            chain_manager = GrailChainManager(config, wallet, metagraph, subtensor, credentials)
            await chain_manager.initialize()
            logger.info("✅ Initialized chain manager and committed read credentials")
            # Ensure background chain worker stops on shutdown
            self.register_shutdown_callback(chain_manager.stop)
            self.heartbeat()

            # Use trainer UID's committed read credentials for checkpoints
            trainer_bucket = chain_manager.get_bucket(TRAINER_UID)
            if trainer_bucket is not None:
                logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")
                checkpoint_credentials = trainer_bucket
            else:
                logger.warning(
                    f"⚠️ Trainer UID {TRAINER_UID} bucket not found, using local credentials"
                )
                checkpoint_credentials = credentials

            checkpoint_manager = CheckpointManager(
                cache_root=default_checkpoint_cache_root(),
                credentials=checkpoint_credentials,
                keep_limit=2,  # Keep only current + previous window
            )

            # Initialize monitoring for mining operations
            monitor = get_monitoring_manager()
            if monitor:
                mining_config = MonitoringConfig.for_mining(wallet.name)
                try:
                    subtensor_for_uid = await self.get_subtensor()
                    self.heartbeat()
                except Exception:
                    subtensor_for_uid = None
                uid = None
                if subtensor_for_uid is not None:
                    uid = await get_own_uid_on_subnet(
                        subtensor_for_uid, netuid, wallet.hotkey.ss58_address
                    )
                    self.heartbeat()
                run_name = f"miner-{uid}" if uid is not None else f"mining_{wallet.name}"
                run_id = await monitor.start_run(run_name, mining_config.get("hyperparameters", {}))
                self.heartbeat()
                logger.info(f"Started monitoring run: {run_id} (name={run_name})")

            while not self.stop_event.is_set():
                try:
                    # Update heartbeat at start of each iteration
                    self.heartbeat()

                    # Use shared subtensor from base class
                    subtensor = await self.get_subtensor()

                    current_block = await subtensor.get_current_block()
                    window_start = self.calculate_window(current_block)

                    # Set monitoring context for metrics (use block_number for x-axis)
                    if monitor:
                        monitor.set_block_context(current_block, None)

                    if window_start <= last_window_start:
                        if window_wait_tracker.should_log_initial():
                            log_window_wait_initial(
                                current_block=current_block,
                                last_processed_window=last_window_start,
                                window_length=WINDOW_LENGTH,
                            )
                        elif window_wait_tracker.should_log_periodic():
                            next_window = calculate_next_window(last_window_start, WINDOW_LENGTH)
                            log_window_wait_periodic(
                                next_window=next_window,
                                elapsed_seconds=window_wait_tracker.get_elapsed_seconds(),
                            )

                        await asyncio.sleep(2)
                        continue

                    # Window is available - reset tracker
                    window_wait_tracker.reset()

                    # Load or update checkpoint (unified fast/slow path)
                    timer_ctx = (
                        monitor.timer("profiling/checkpoint_load")
                        if monitor
                        else contextlib.nullcontext()
                    )
                    with timer_ctx:
                        result, checkpoint_path = await checkpoint_manager.load_or_update_model(
                            window_start, model, current_checkpoint_window
                        )
                    self.heartbeat()

                    if result.success:
                        if result.is_fast_path:
                            # Fast path: model already updated in-place
                            current_checkpoint_window = result.window
                            logger.info("⚡ Model updated in-place to window %s", result.window)
                        elif checkpoint_path is not None:
                            # Slow path: load from disk
                            logger.info(
                                "🔁 Loading checkpoint for window %s from %s",
                                result.window,
                                checkpoint_path,
                            )
                            try:
                                model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
                                model = get_model(str(checkpoint_path), device=None, eval_mode=True)
                                tokenizer = get_tokenizer(str(checkpoint_path))
                                current_checkpoint_window = result.window

                                if torch.cuda.is_available():
                                    logger.info(
                                        f"GPU Memory: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
                                        f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                                    )
                                    torch.cuda.empty_cache()
                            except Exception:
                                logger.exception(
                                    "Failed to load checkpoint for window %s", result.window
                                )
                                raise
                    elif model is None or tokenizer is None:
                        logger.error("No checkpoint available and no model loaded, cannot mine")
                        await asyncio.sleep(60)
                        continue

                    # Ensure model and tokenizer are loaded before mining
                    if model is None or tokenizer is None:
                        logger.error("Model or tokenizer not loaded, cannot mine")
                        await asyncio.sleep(30)
                        continue

                    logger.info(
                        f"🔥 Starting inference generation for window "
                        f"{window_start}-{window_start + WINDOW_LENGTH - 1}"
                    )

                    if not await has_time_for_next_generation(subtensor, timers, window_start):
                        last_window_start = window_start
                        await asyncio.sleep(5)
                        continue

                    window_block_hash, combined_randomness = await get_window_randomness(
                        subtensor,
                        window_start,
                        self.use_drand,
                    )

                    inferences = await generate_rollouts_for_window(
                        wallet,
                        model,
                        tokenizer,
                        subtensor,
                        window_start,
                        window_block_hash,
                        combined_randomness,
                        timers,
                        monitor,
                        self.use_drand,
                        current_checkpoint_window,
                    )

                    if inferences:
                        logger.info(
                            f"📤 Uploading {len(inferences)} rollouts to R2 "
                            f"for window {window_start}..."
                        )
                        try:
                            upload_duration = await upload_inferences_with_metrics(
                                wallet, window_start, inferences, credentials, monitor
                            )
                            timers.update_upload_time_ema(upload_duration)
                            logger.info(
                                f"✅ Successfully uploaded window {window_start} "
                                f"with {len(inferences)} rollouts"
                            )
                            self.heartbeat()
                            if monitor:
                                await monitor.log_counter("mining/successful_uploads")
                                await monitor.log_gauge("mining/uploaded_rollouts", len(inferences))

                        except Exception as e:
                            logger.error(f"❌ Failed to upload window {window_start}: {e}")
                            logger.error(traceback.format_exc())
                            if monitor:
                                await monitor.log_counter("mining/failed_uploads")
                    else:
                        logger.warning(f"No inferences generated for window {window_start}")
                        if monitor:
                            await monitor.log_counter("mining/empty_windows")

                    last_window_start = window_start
                    await checkpoint_manager.cleanup_local(window_start)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in miner loop: {e}. Continuing ...")
                    self.reset_subtensor()  # Force reconnect on next iteration
                    await asyncio.sleep(10)
                    continue

        # Start process-level watchdog (handled by BaseNeuron)
        self.start_watchdog(timeout_seconds=(60 * 10))
        await _run()
