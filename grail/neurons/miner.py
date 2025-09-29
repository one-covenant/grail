from __future__ import annotations

import asyncio
import logging
import time
import traceback
from types import SimpleNamespace

import bittensor as bt

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
from grail.infrastructure.credentials import load_r2_credentials
from grail.model.provider import get_model, get_tokenizer
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import MODEL_NAME, WINDOW_LENGTH
from grail.shared.subnet import get_own_uid_on_subnet

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    async def run(self) -> None:
        """Main mining loop mirrored from the CLI implementation."""
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        # Initialize model and tokenizer via provider
        logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")
        logger.info(f"Loading base model: {MODEL_NAME}")
        model = get_model(MODEL_NAME, device=None, eval_mode=True)
        tokenizer = get_tokenizer(MODEL_NAME)

        async def _run() -> None:
            subtensor = None
            last_window_start = -1
            timers = MiningTimers()

            # Load R2 credentials
            try:
                credentials = load_r2_credentials()
                logger.info("✅ Loaded R2 credentials")
            except Exception as e:
                logger.error(f"Failed to load R2 credentials: {e}")
                raise

            # Initialize chain manager for credential commitments
            config = SimpleNamespace(netuid=int(get_conf("BT_NETUID", get_conf("NETUID", 200))))
            chain_manager = GrailChainManager(config, wallet, credentials)
            await chain_manager.initialize()
            logger.info("✅ Initialized chain manager and committed read credentials")

            # Initialize monitoring for mining operations
            monitor = get_monitoring_manager()
            if monitor:
                mining_config = MonitoringConfig.for_mining(wallet.name)
                try:
                    subtensor_for_uid = await get_subtensor()
                except Exception:
                    subtensor_for_uid = None
                uid = None
                if subtensor_for_uid is not None:
                    uid = await get_own_uid_on_subnet(
                        subtensor_for_uid, 81, wallet.hotkey.ss58_address
                    )
                run_name = f"miner-{uid}" if uid is not None else f"mining_{wallet.name}"
                run_id = await monitor.start_run(run_name, mining_config.get("hyperparameters", {}))
                logger.info(f"Started monitoring run: {run_id} (name={run_name})")

            while not self.stop_event.is_set():
                try:
                    # Update the heartbeat used by the CLI watchdog
                    cli_mine.HEARTBEAT = time.monotonic()
                    if subtensor is None:
                        subtensor = await get_subtensor()

                    current_block = await subtensor.get_current_block()
                    window_start = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

                    if window_start <= last_window_start:
                        await asyncio.sleep(2)
                        continue

                    logger.info(f"🚀 Using base model for window {window_start}")
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
                            cli_mine.HEARTBEAT = time.monotonic()
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
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in miner loop: {e}. Continuing ...")
                    subtensor = None
                    await asyncio.sleep(10)
                    continue

        async def _main() -> None:
            await asyncio.gather(_run(), cli_mine.watchdog())

        await _main()
