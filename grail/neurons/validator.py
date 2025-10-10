"""Validator neuron using new service-based validation architecture.

Uses ValidationService for clean separation of concerns.
"""

from __future__ import annotations

import logging
import os

import bittensor as bt

from grail.environments import get_sat_reward_bounds
from grail.infrastructure.checkpoints import (
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
    WINDOW_LENGTH,
)
from grail.validation import ValidationService, create_sat_validation_pipeline

from .base import BaseNeuron

logger = logging.getLogger(__name__)

# Weight submission interval (same as validate.py)
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 360
WEIGHT_ROLLING_WINDOWS = int(WEIGHT_SUBMISSION_INTERVAL_BLOCKS / WINDOW_LENGTH)


class ValidatorNeuron(BaseNeuron):
    """Runs validation using new service-based architecture."""

    def __init__(self, use_drand: bool = True, test_mode: bool = False) -> None:
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

        # Load credentials
        credentials = load_r2_credentials()

        # Initialize monitoring
        monitor = get_monitoring_manager()
        if monitor:
            validation_config = MonitoringConfig.for_validation(wallet.name)
            monitor.initialize(validation_config)

        # Create validation pipeline
        sat_pipeline = create_sat_validation_pipeline()
        logger.info(
            f"✅ Created SAT validation pipeline with {len(sat_pipeline.validators)} validators"
        )

        # Create weight computer
        weight_computer = WeightComputer(
            rolling_windows=WEIGHT_ROLLING_WINDOWS,
            window_length=WINDOW_LENGTH,
            superlinear_exponent=SUPERLINEAR_EXPONENT,
            burn_uid=GRAIL_BURN_UID,
            burn_percentage=GRAIL_BURN_PERCENTAGE,
        )

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            cache_root=default_checkpoint_cache_root(),
            credentials=credentials,
            keep_limit=3,
        )

        # Get SAT reward bounds
        sat_reward_low, sat_reward_high = get_sat_reward_bounds()

        # Create validation service
        validation_service = ValidationService(
            wallet=wallet,
            netuid=NETUID,
            sat_pipeline=sat_pipeline,
            weight_computer=weight_computer,
            credentials=credentials,
            checkpoint_manager=checkpoint_manager,
            sat_reward_bounds=(sat_reward_low, sat_reward_high),
            monitor=monitor,
        )

        try:
            # Run validation loop
            await validation_service.run_validation_loop(
                subtensor=subtensor,
                use_drand=self.use_drand,
                test_mode=self.test_mode,
            )
        except Exception:
            logger.exception("Validator crashed due to unhandled exception")
            flush_all_logs()
            raise
        finally:
            validation_service.cleanup()
