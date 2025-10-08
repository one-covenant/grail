from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
from types import SimpleNamespace

import bittensor as bt
import torch

import grail.cli.mine as cli_mine
from grail.cli.mine import (
    MiningTimers,
    generate_rollouts_for_window,
    get_conf,
    get_subtensor,
    get_window_randomness,
    has_time_for_next_generation,
    upload_inferences_with_metrics,
)
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.checkpoints import CheckpointManager, default_checkpoint_cache_root
from grail.infrastructure.credentials import load_r2_credentials
from grail.model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import TRAINER_UID, WINDOW_LENGTH
from grail.shared.subnet import get_own_uid_on_subnet

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    def _update_heartbeat(self) -> None:
        """Update heartbeat to signal progress to watchdog.

        Called at key points in the mining loop to prove the process is
        making progress. Now that we have explicit timeouts on blockchain
        calls, we don't need continuous background updates.
        """
        cli_mine.HEARTBEAT = time.monotonic()

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
            self._update_heartbeat()
            logger.info("✅ Initialized watchdog heartbeat")

            # Get subtensor and metagraph for chain manager (use shared base method)
            subtensor = await self.get_subtensor()
            self._update_heartbeat()
            netuid = int(get_conf("BT_NETUID", get_conf("NETUID", 200)))
            metagraph = await subtensor.metagraph(netuid)
            self._update_heartbeat()

            # Initialize chain manager for credential commitments
            config = SimpleNamespace(netuid=netuid)
            chain_manager = GrailChainManager(config, wallet, metagraph, subtensor, credentials)
            await chain_manager.initialize()
            # Store chain manager globally for watchdog cleanup
            cli_mine.CHAIN_MANAGER = chain_manager

            logger.info("✅ Initialized chain manager and committed read credentials")
            self._update_heartbeat()

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
                    subtensor_for_uid = await get_subtensor()
                    self._update_heartbeat()
                except Exception:
                    subtensor_for_uid = None
                uid = None
                if subtensor_for_uid is not None:
                    uid = await get_own_uid_on_subnet(
                        subtensor_for_uid, 81, wallet.hotkey.ss58_address
                    )
                    self._update_heartbeat()
                run_name = f"miner-{uid}" if uid is not None else f"mining_{wallet.name}"
                run_id = await monitor.start_run(run_name, mining_config.get("hyperparameters", {}))
                self._update_heartbeat()
                logger.info(f"Started monitoring run: {run_id} (name={run_name})")

            while not self.stop_event.is_set():
                try:
                    # Update heartbeat at start of each iteration
                    self._update_heartbeat()

                    # Use shared subtensor from base class
                    subtensor = await self.get_subtensor()

                    current_block = await subtensor.get_current_block()
                    window_start = self.calculate_window(current_block)
                    # Miners use the checkpoint published after the previous window finished
                    checkpoint_window = window_start - WINDOW_LENGTH

                    if window_start <= last_window_start:
                        await asyncio.sleep(2)
                        continue

                    # Load checkpoint (required - miners always use checkpoints)
                    if checkpoint_window is not None and checkpoint_window >= 0:
                        if current_checkpoint_window != checkpoint_window:
                            # Time checkpoint download/retrieval
                            timer_ctx = (
                                monitor.timer("mining/checkpoint_download")
                                if monitor
                                else contextlib.nullcontext()
                            )
                            with timer_ctx:
                                checkpoint_path = await checkpoint_manager.get_checkpoint(
                                    checkpoint_window
                                )
                            self._update_heartbeat()

                            # checkpoint_path = 'Qwen/Qwen3-4B-Instruct-2507'
                            if checkpoint_path is not None:
                                logger.info(
                                    "🔁 Loading checkpoint for window %s from %s",
                                    checkpoint_window,
                                    checkpoint_path,
                                )
                                try:
                                    # Pre-load cleanup to prevent VRAM growth when swapping checkpoints
                                    model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
                                    model = get_model(
                                        str(checkpoint_path), device=None, eval_mode=True
                                    )
                                    tokenizer = get_tokenizer(str(checkpoint_path))
                                    current_checkpoint_window = checkpoint_window
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception:
                                    logger.exception(
                                        "Failed to load checkpoint for window %s",
                                        checkpoint_window,
                                    )
                                    raise
                            else:
                                logger.error(
                                    "No checkpoint available for window %s - cannot mine",
                                    checkpoint_window,
                                )
                                await asyncio.sleep(30)
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
                            self._update_heartbeat()
                            if monitor:
                                await monitor.log_counter("mining.successful_uploads")
                                await monitor.log_gauge("mining.uploaded_rollouts", len(inferences))

                        except Exception as e:
                            logger.error(f"❌ Failed to upload window {window_start}: {e}")
                            logger.error(traceback.format_exc())
                            if monitor:
                                await monitor.log_counter("mining.failed_uploads")
                    else:
                        logger.warning(f"No inferences generated for window {window_start}")
                        if monitor:
                            await monitor.log_counter("mining.empty_windows")

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

        async def _main() -> None:
            await asyncio.gather(_run(), cli_mine.watchdog())

        await _main()
