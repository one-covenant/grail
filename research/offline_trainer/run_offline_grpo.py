"""Offline GRPO training runner (backward compatibility wrapper).

This file provides backward compatibility for existing scripts.
New code should use bin/run_offline_grpo.py or import from grail_offline.pipelines.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Ensure repo root and src are importable
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = _THIS_FILE.parent / "src"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from grail.monitoring import (  # noqa: E402, I001
    MonitoringConfig,
    get_monitoring_manager,
    initialize_monitoring,
)
from grail_offline.pipelines.offline_grpo import run_training  # noqa: E402

logger = logging.getLogger(__name__)


def _initialize_monitoring(cfg: DictConfig) -> Any | None:
    """Initialize monitoring if enabled in config."""
    if not bool(cfg.logging.wandb.enabled):
        logger.info("WandB disabled - using in-memory metrics only")
        return None

    logger.info("Initializing monitoring system (WandB enabled)")
    monitoring_config = MonitoringConfig.for_training()
    monitoring_config.update(
        {
            "tags": ["offline_trainer", "grpo", str(cfg.data.environment)],
            "hyperparameters": {
                "model": str(cfg.model.train_id),
                "environment": str(cfg.data.environment),
                "iterations": int(cfg.train.iterations),
                "batch_size": int(cfg.train.batch_size),
                "learning_rate": float(cfg.train.lr),
                "kl_coef": float(cfg.train.kl_coef),
                "entropy_coef": float(cfg.train.entropy_coef),
                "rollouts_per_problem": int(cfg.data.rollouts_per_problem),
                "problems_per_iteration": int(cfg.data.problems_per_iteration),
            },
        }
    )
    backend_type = monitoring_config.pop("backend_type", "wandb")
    initialize_monitoring(backend_type=backend_type, **monitoring_config)
    return get_monitoring_manager()


async def _run(cfg: DictConfig) -> None:
    """Main training loop."""
    monitor = _initialize_monitoring(cfg)
    workdir = Path(HydraConfig.get().runtime.output_dir)

    try:
        await run_training(cfg, workdir, monitor)
    finally:
        if monitor:
            logger.info("Shutting down monitoring")
            try:
                await monitor.shutdown()
            except Exception as e:
                logger.warning("Error during monitoring shutdown", extra={"error": str(e)})


@hydra.main(config_path="conf", config_name="offline_grpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point for offline GRPO training."""
    try:
        asyncio.run(_run(cfg))
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
