"""Validator neuron using new context-based validation architecture.

Uses ValidationContext, validation pipeline, and separate scoring.
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
from grail.infrastructure.comms import login_huggingface
from grail.mining.rollout_generator import REASONING_START, SYSTEM_PROMPT
from grail.model.provider import get_model, get_tokenizer
from grail.scoring import WeightComputer
from grail.shared.chat_templates import build_qwen_chat_template
from grail.shared.constants import (
    GRAIL_BURN_PERCENTAGE,
    GRAIL_BURN_UID,
    MODEL_NAME,
    SUPERLINEAR_EXPONENT,
    WINDOW_LENGTH,
)
from grail.validation import create_sat_validation_pipeline

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class ValidatorNeuron(BaseNeuron):
    """Runs validation using new context-based architecture."""

    def __init__(self, use_drand: bool = True, test_mode: bool = False) -> None:
        super().__init__()
        self.use_drand = use_drand
        self.test_mode = test_mode

    async def run(self) -> None:
        """Run validation with new architecture: pipeline + scoring."""
        logger = logging.getLogger("grail")

        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Validator hotkey: {wallet.hotkey.ss58_address}")
        logger.info(f"Loading model for validation: {MODEL_NAME}")

        # Load model once via provider
        chat_template = build_qwen_chat_template(SYSTEM_PROMPT, REASONING_START)
        model = get_model(MODEL_NAME, eval_mode=True)
        tokenizer = get_tokenizer(MODEL_NAME, chat_template=chat_template)

        logger.info("🤗 Logging into Hugging Face for dataset uploads...")
        login_huggingface()

        # Create validation pipeline
        sat_pipeline = create_sat_validation_pipeline()
        logger.info(
            f"✅ Created SAT validation pipeline with {len(sat_pipeline.validators)} validators"
        )

        # Create weight computer
        weight_computer = WeightComputer(
            rolling_windows=12,
            window_length=WINDOW_LENGTH,
            superlinear_exponent=SUPERLINEAR_EXPONENT,
            burn_uid=GRAIL_BURN_UID,
            burn_percentage=GRAIL_BURN_PERCENTAGE,
        )

        sat_reward_low, sat_reward_high = _get_sat_reward_bounds()

        try:
            await _run_validation_service(
                wallet=wallet,
                model=model,
                tokenizer=tokenizer,
                sat_pipeline=sat_pipeline,
                weight_computer=weight_computer,
                sat_reward_low=sat_reward_low,
                sat_reward_high=sat_reward_high,
                use_drand=self.use_drand,
                test_mode=self.test_mode,
            )
        except Exception:
            logger.exception("Validator crashed due to unhandled exception")
            _flush_all_logs()
            raise
