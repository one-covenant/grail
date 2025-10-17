"""Trainer neuron orchestrating window selection and delegating training."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from grail.infrastructure.chain import GrailChainManager
from grail.shared.constants import NETUID, WINDOW_LENGTH
from grail.trainer.service import TrainerService

from .base import BaseNeuron

if TYPE_CHECKING:
    import bittensor as bt

    from grail.infrastructure.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainerContext:
    """Resources required to run the trainer neuron."""

    wallet: bt.wallet
    credentials: Any
    checkpoint_manager: CheckpointManager
    monitor: Any | None
    train_model: Any
    ref_model: Any
    tokenizer: Any
    chain_manager: Any | None = None


class TrainerNeuron(BaseNeuron):
    """Runs training cycles by delegating to the TrainerService."""

    def __init__(self, context: TrainerContext) -> None:
        super().__init__()
        self._context = context

    async def run(self) -> None:
        # Start the built-in watchdog (15 minute timeout)
        self.start_watchdog(timeout_seconds=60 * 15, grace_seconds=10)

        # Initialize chain manager once for the lifetime of the trainer
        await self._initialize_chain_manager()

        last_processed_window = -1

        while not self.stop_event.is_set():
            try:
                # Update heartbeat from BaseNeuron
                self.heartbeat()

                # Use shared subtensor from base class
                subtensor = await self.get_subtensor()

                current_block = await subtensor.get_current_block()
                current_window = self.calculate_window(current_block)
                target_window = current_window - WINDOW_LENGTH

                if target_window <= last_processed_window or target_window < 0:
                    await asyncio.sleep(10)
                    continue

                logger.info("🎓 Training window %s", target_window)
                success = await self._train_window(target_window)

                if success:
                    logger.info("✅ Trained window %s", target_window)
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training.success")
                else:
                    logger.warning("⚠️ Training issue (w=%s)", target_window)
                    logger.warning("Retrying next window")
                    if self._context.monitor:
                        await self._context.monitor.log_counter("training.failed")

                # Mark window as processed regardless of outcome
                last_processed_window = target_window

                # Sleep before checking for next window to avoid tight loops
                await asyncio.sleep(10)

            except asyncio.CancelledError:  # pragma: no cover - coop shutdown
                break
            except Exception:
                logger.exception("Trainer loop error", exc_info=True)
                # Force reconnect on next iteration
                self.reset_subtensor()
                await asyncio.sleep(30)

    async def _initialize_chain_manager(self) -> None:
        """Initialize chain manager for miner data fetching."""
        try:
            subtensor = await self.get_subtensor()
            metagraph = await subtensor.metagraph(NETUID)

            config = SimpleNamespace(netuid=NETUID)
            chain_manager = GrailChainManager(
                config,
                self._context.wallet,
                metagraph,
                subtensor,
                self._context.credentials,
            )

            await chain_manager.initialize()
            self._context.chain_manager = chain_manager
            logger.info("Initialized chain manager for trainer lifetime")

            # Register cleanup callback
            self.register_shutdown_callback(self._cleanup_chain_manager)

        except Exception as exc:
            logger.warning(
                "Failed to initialize chain manager: %s; will continue with default credentials",
                exc,
            )
            self._context.chain_manager = None

    def _cleanup_chain_manager(self) -> None:
        """Clean up chain manager on shutdown."""
        if self._context.chain_manager:
            try:
                self._context.chain_manager.stop()
                logger.info("Stopped chain manager")
            except Exception as exc:
                logger.warning("Error stopping chain manager: %s", exc)

    async def _train_window(self, window: int) -> bool:
        ctx = self._context
        # Update heartbeat before long operation
        self.heartbeat()

        # Get subtensor for metagraph queries
        subtensor = await self.get_subtensor()

        service = TrainerService(
            wallet=ctx.wallet,
            credentials=ctx.credentials,
            checkpoint_manager=ctx.checkpoint_manager,
            monitor=ctx.monitor,
            train_model=ctx.train_model,
            ref_model=ctx.ref_model,
            tokenizer=ctx.tokenizer,
            chain_manager=ctx.chain_manager,
        )
        return await service.train_window(window, subtensor)
