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

from grail.infrastructure.credentials import load_r2_credentials
from grail.model.train_loading import (
    ModelLoadSpec,
    parse_ref_env,
    parse_train_env,
)
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.trainer.checkpoint_publisher import CheckpointPublisher

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


def train(
    ctx: typer.Context,
) -> None:
    """Run the training process via TrainerNeuron orchestration."""
    from grail.neurons import TrainerNeuron
    from grail.neurons.trainer import TrainerContext

    # Get verbosity from parent context (set by main callback)
    verbosity = getattr(ctx.parent, "params", {}).get("verbose", 1) if ctx.parent else 1

    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    logger.info(f"🔑 Trainer hotkey: {wallet.hotkey.ss58_address}")

    async def _setup_and_run() -> None:
        # Credentials
        credentials = load_r2_credentials()

        # Checkpoint publisher (producer - trainer only)
        checkpoint_publisher = CheckpointPublisher(
            credentials=credentials,
            wallet=wallet,
        )

        # Monitoring
        monitor = get_monitoring_manager()
        if monitor:
            training_config = MonitoringConfig.for_training(wallet.name)
            run_id = await monitor.start_run(
                f"trainer_{wallet.name}",
                training_config,  # Pass full config (includes wandb_shared_mode)
            )
            logger.info(f"Started monitoring run: {run_id}")
            
            # CRITICAL: Wait for WandB run to fully sync to cloud before spawning subprocess
            # Shared mode requires the primary run to be API-accessible before workers connect
            # Without this delay, workers timeout trying to connect to a not-yet-synced run
            if training_config.get("wandb_shared_mode"):
                import asyncio
                logger.info("Waiting 5s for WandB run to sync to cloud (shared mode requirement)...")
                await asyncio.sleep(5)
                logger.info("WandB run sync complete, safe to spawn worker processes")

        # Parse env (strict; no defaults)
        try:
            train_spec: ModelLoadSpec = parse_train_env()
            ref_spec: ModelLoadSpec = parse_ref_env()
        except Exception as exc:
            logger.error("Trainer startup configuration error: %s", exc)
            raise typer.Exit(code=1) from exc

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

        # Context
        context = TrainerContext(
            wallet=wallet,
            credentials=credentials,
            checkpoint_publisher=checkpoint_publisher,
            monitor=monitor,
            train_spec=train_spec,
            ref_spec=ref_spec,
            verbosity=verbosity,
        )

        # Run neuron (watchdog is managed by BaseNeuron)
        trainer = TrainerNeuron(context)
        await trainer.main()

    asyncio.run(_setup_and_run())


def main() -> None:
    train()
