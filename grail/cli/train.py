#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#                             Imports                                         #
# --------------------------------------------------------------------------- #
import asyncio
import logging
import os
from typing import Any

import bittensor as bt
import typer

from grail.infrastructure.checkpoints import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.model.train_loading import (
    ModelLoadSpec,
    load_training_artifacts,
    parse_ref_env,
    parse_train_env,
)
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig

from . import console

logger = logging.getLogger("grail")


def get_conf(key: str, default: Any | None = None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        console.print(f"[red]{key} not set.[/red]\nRun:\n    af set {key} <value>")
        raise typer.Exit(code=1)
    return v or default


def register(app: typer.Typer) -> None:
    app.command("train")(train)


def train() -> None:
    """Run the training process via TrainerNeuron orchestration."""
    from grail.neurons import TrainerNeuron
    from grail.neurons.trainer import TrainerContext

    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    logger.info(f"🔑 Trainer hotkey: {wallet.hotkey.ss58_address}")

    async def _setup_and_run() -> None:
        # Credentials
        credentials = load_r2_credentials()

        # Checkpoints
        checkpoint_manager = CheckpointManager(
            cache_root=default_checkpoint_cache_root(),
            credentials=credentials,
        )

        # Monitoring
        monitor = get_monitoring_manager()
        if monitor:
            training_config = MonitoringConfig.for_training(wallet.name)
            run_id = await monitor.start_run(
                f"trainer_{wallet.name}",
                training_config.get("hyperparameters", {}),
            )
            logger.info(f"Started monitoring run: {run_id}")

        # Parse env and load training artifacts (strict; no defaults)
        try:
            train_spec: ModelLoadSpec = parse_train_env()
            ref_spec: ModelLoadSpec = parse_ref_env()

            # Log chosen configuration
            logger.info("🚀 Trainer model loading configuration:")
            logger.info(f"  Train mode: {train_spec.mode}")
            if train_spec.hf_id:
                logger.info(f"  Train HF ID: {train_spec.hf_id}")
            if train_spec.window is not None:
                logger.info(f"  Train checkpoint window: {train_spec.window}")
            logger.info(f"  Reference mode: {ref_spec.mode}")
            if ref_spec.hf_id:
                logger.info(f"  Reference HF ID: {ref_spec.hf_id}")
            if ref_spec.window is not None:
                logger.info(f"  Reference checkpoint window: {ref_spec.window}")

            train_model, ref_model, tokenizer = await load_training_artifacts(
                train_spec, ref_spec, checkpoint_manager
            )
            logger.info("✅ Successfully loaded training artifacts")

            # Store model paths for potential reloading after evaluation
            train_model_path = getattr(train_model, "name_or_path", None)
            ref_model_path = getattr(ref_model, "name_or_path", None)
        except Exception as exc:
            logger.error("Trainer startup configuration error: %s", exc)
            raise typer.Exit(code=1) from exc

        # Context
        context = TrainerContext(
            wallet=wallet,
            credentials=credentials,
            checkpoint_manager=checkpoint_manager,
            monitor=monitor,
            train_model=train_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_model_path=train_model_path,
            ref_model_path=ref_model_path,
        )

        # Run neuron (watchdog is managed by BaseNeuron)
        trainer = TrainerNeuron(context)
        await trainer.main()

    asyncio.run(_setup_and_run())


def main() -> None:
    train()
