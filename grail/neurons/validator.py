"""Validator neuron using new service-based validation architecture.

Uses ValidationService for clean separation of concerns.
"""

from __future__ import annotations

import logging
import os

import bittensor as bt

from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.comms import login_huggingface
from grail.infrastructure.credentials import load_r2_credentials
from grail.logging_utils import flush_all_logs
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.scoring import WeightComputer
from grail.shared.constants import (
    GRAIL_BURN_PERCENTAGE,
    GRAIL_BURN_UID,
    NETUID,
    SUPERLINEAR_EXPONENT,
    UNIQUE_ROLLOUTS_CAP_ENABLED,
    WINDOW_LENGTH,
)
from grail.validation import create_env_validation_pipeline
from grail.validation.service import (
    WEIGHT_ROLLING_WINDOWS,
    ValidationService,
)

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class ValidatorNeuron(BaseNeuron):
    """Runs validation using new service-based architecture."""

    def __init__(
        self,
        use_drand: bool = True,
        test_mode: bool = False,
    ) -> None:
        super().__init__()
        self.use_drand = use_drand
        self.test_mode = test_mode

    async def run(self) -> None:
        """Run validation with new service architecture."""
        logger = logging.getLogger("grail")

        # Get wallet from environment
        coldkey = os.getenv("BT_WALLET_COLD", "default")
        hotkey = os.getenv("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
        logger.info("Validator will load model from checkpoint")

        logger.info("🤗 Logging into Hugging Face for dataset uploads...")
        login_huggingface()

        # Get shared subtensor instance from BaseNeuron
        subtensor = await self.get_subtensor()
        logger.info("✅ Connected to Bittensor network")

        # Fail fast if hotkey is not registered on the subnet
        await self.ensure_registered(wallet, NETUID, role="validator")

        # Load credentials
        credentials = load_r2_credentials()

        # Initialize monitoring
        monitor = get_monitoring_manager()
        if monitor:
            validation_config = MonitoringConfig.for_validation(wallet.name)
            # Use start_run instead of initialize to avoid creating multiple wandb runs
            # (CLI already initialized the backend globally)
            run_name = validation_config.get("run_name", f"validation_{wallet.name}")
            hyperparams = validation_config.get("hyperparameters", {})
            await monitor.start_run(run_name, hyperparams)

        # Create validation pipeline (env-agnostic)
        validation_pipeline = create_env_validation_pipeline()
        logger.info(
            "✅ Created environment validation pipeline with %s validators",
            len(validation_pipeline.validators),
        )

        # Create weight computer
        weight_computer = WeightComputer(
            rolling_windows=WEIGHT_ROLLING_WINDOWS,
            window_length=WINDOW_LENGTH,
            superlinear_exponent=SUPERLINEAR_EXPONENT,
            burn_uid=GRAIL_BURN_UID,
            burn_percentage=GRAIL_BURN_PERCENTAGE,
            cap_enabled=UNIQUE_ROLLOUTS_CAP_ENABLED,
        )

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            cache_root=default_checkpoint_cache_root(),
            credentials=credentials,
            keep_limit=3,
        )

        validation_service = ValidationService(
            wallet=wallet,
            netuid=NETUID,
            validation_pipeline=validation_pipeline,
            weight_computer=weight_computer,
            credentials=credentials,
            checkpoint_manager=checkpoint_manager,
            monitor=monitor,
        )

        # Start process-level watchdog (30 minutes stall timeout)
        self.start_watchdog(timeout_seconds=(60 * 30))

        try:
            # Ensure chain manager is stopped during cooperative shutdown
            self.register_shutdown_callback(validation_service.cleanup)

            # Run validation loop and feed heartbeats to watchdog
            await validation_service.run_validation_loop(
                subtensor=subtensor,
                use_drand=self.use_drand,
                test_mode=self.test_mode,
                heartbeat_callback=self.heartbeat,
            )
        except Exception:
            logger.exception("Validator crashed due to unhandled exception")
            flush_all_logs()
            raise
        finally:
            # Cleanup handled via registered shutdown callback and BaseNeuron
            pass
