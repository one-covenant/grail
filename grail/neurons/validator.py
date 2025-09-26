"""Validator neuron wrapper.

Stage 1 delegates to the existing CLI entry to avoid behavior changes.
In later stages, this neuron will orchestrate the validation service directly.
"""

from __future__ import annotations

import logging

import bittensor as bt

from grail.cli.validate import (
    _flush_all_logs,
    _get_sat_reward_bounds,
    _run_validation_service,
    get_conf,
)
from grail.grail import Verifier
from grail.infrastructure.comms import login_huggingface
from grail.shared.constants import MODEL_NAME

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class ValidatorNeuron(BaseNeuron):
    """Runs the validation loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True, test_mode: bool = False) -> None:
        super().__init__()
        self.use_drand = use_drand
        self.test_mode = test_mode

    async def run(self) -> None:
        # Orchestrate validation using existing helpers (no behavior change).
        logger = logging.getLogger("grail")

        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
        logger.info(f"Loading base model for validation: {MODEL_NAME}")
        verifier = Verifier(model_name=MODEL_NAME)

        logger.info("🤗 Logging into Hugging Face for dataset uploads...")
        login_huggingface()

        sat_reward_low, sat_reward_high = _get_sat_reward_bounds()

        try:
            await _run_validation_service(
                wallet=wallet,
                verifier=verifier,
                sat_reward_low=sat_reward_low,
                sat_reward_high=sat_reward_high,
                use_drand=self.use_drand,
                test_mode=self.test_mode,
            )
        except Exception:
            logger.exception("Validator crashed due to unhandled exception")
            _flush_all_logs()
            raise
