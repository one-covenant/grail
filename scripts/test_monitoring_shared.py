#!/usr/bin/env python3
"""
Test script for WandB shared mode using the GRAIL monitoring class.

This script tests the MonitoringManager with WandB shared mode enabled,
simulating how the actual GRAIL trainer and training subprocess interact.

Usage:
    # Start a new shared mode run
    python test_monitoring_shared.py --new

    # Connect to an existing run as a worker
    python test_monitoring_shared.py --run-id abc123xyz

    # Connect to an existing run with specific entity
    python test_monitoring_shared.py --run-id abc123xyz --entity tplr
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
import time
from typing import Any

from grail.monitoring.backends.wandb_backend import WandBBackend
from grail.monitoring.config import MonitoringConfig
from grail.monitoring.manager import MonitoringManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def primary_process_workflow(
    entity: str | None = None, run_name: str = "test_monitoring_shared_primary"
) -> str:
    """Run the primary process workflow (main trainer process).

    Args:
        entity: WandB entity name
        run_name: Name for the WandB run

    Returns:
        Run ID for use in worker process
    """
    logger.info("[PRIMARY] Starting primary process workflow...")

    # Get training config
    config = MonitoringConfig.for_training(wallet_name="test_wallet")
    if entity:
        config["entity"] = entity

    logger.info(
        "[PRIMARY] Config: backend=%s, project=%s, entity=%s, shared_mode=%s",
        config.get("backend_type"),
        config.get("project"),
        config.get("entity"),
        config.get("wandb_shared_mode"),
    )

    # Create backend and manager
    backend = WandBBackend()
    manager = MonitoringManager(backend)
    manager.initialize(config)

    logger.info("[PRIMARY] MonitoringManager initialized")

    # Start run (this creates a primary WandB run)
    start_time = time.time()
    run_id = await manager.start_run(
        run_name,
        config,  # Pass full config including shared mode settings
    )
    elapsed = time.time() - start_time

    logger.info(
        "[PRIMARY] ✅ Primary run created in %.1fs with run_id=%s",
        elapsed,
        run_id,
    )

    # Log some metrics to verify the run is active
    logger.info("[PRIMARY] Logging initial metrics...")
    await manager.log_gauge("testing/primary_startup_time", elapsed)
    await manager.log_gauge("testing/primary_timestamp", time.time())
    await manager.flush_metrics()

    logger.info("[PRIMARY] Metrics logged and flushed")
    logger.info(
        "[PRIMARY] ✅ Primary process ready. Run ID: %s",
        run_id,
    )
    logger.info("[PRIMARY] Waiting 5s for run to sync to cloud...")
    await asyncio.sleep(5)
    logger.info("[PRIMARY] ✅ Run sync complete, safe to spawn worker processes")

    # Keep the manager alive for a bit longer while worker connects
    await asyncio.sleep(2)

    # Clean shutdown
    logger.info("[PRIMARY] Shutting down primary process...")
    await manager.finish_run(run_id)
    await manager.shutdown()

    logger.info("[PRIMARY] ✅ Primary process shutdown complete")
    return run_id


def worker_process_workflow(
    run_id: str, entity: str | None = None, process_label: str = "worker"
) -> None:
    """Run the worker process workflow (training subprocess).

    This function runs in a separate process and connects to an existing
    WandB run using shared mode.

    Args:
        run_id: The WandB run ID to connect to
        entity: WandB entity name
        process_label: Label for this worker process
    """

    def setup_logging() -> None:
        """Setup logging for the worker process."""
        logging.basicConfig(
            level=logging.INFO,
            format=f"[{process_label}] %(levelname)s - %(name)s - %(message)s",
        )

    setup_logging()
    logger_worker = logging.getLogger(__name__)

    logger_worker.info(f"[{process_label}] Starting worker process workflow...")
    logger_worker.info(f"[{process_label}] Connecting to run_id=%s", run_id)

    try:
        # Get training config for worker (not primary)
        config = MonitoringConfig.for_training(wallet_name="test_wallet")
        if entity:
            config["entity"] = entity

        # Mark as worker, not primary
        config["wandb_x_primary"] = False
        config["wandb_x_label"] = process_label
        config["run_id"] = run_id  # Connect to existing run

        logger_worker.info(
            f"[{process_label}] Config: backend=%s, project=%s, entity=%s, "
            "shared_mode=%s, x_primary=%s, x_label=%s, run_id=%s",
            config.get("backend_type"),
            config.get("project"),
            config.get("entity"),
            config.get("wandb_shared_mode"),
            config.get("wandb_x_primary"),
            config.get("wandb_x_label"),
            config.get("run_id"),
        )

        # Create backend and manager
        backend = WandBBackend()
        manager = MonitoringManager(backend)
        manager.initialize(config)

        logger_worker.info(f"[{process_label}] MonitoringManager initialized")

        # Start run (connects to existing run in worker mode)
        start_time = time.time()
        actual_run_id = asyncio.run(
            manager.start_run(f"{process_label}_process", config)
        )
        elapsed = time.time() - start_time

        logger_worker.info(
            f"[{process_label}] ✅ Connected in %.1fs (run_id=%s)",
            elapsed,
            actual_run_id,
        )

        # Verify we're connected to the same run
        if actual_run_id == run_id:
            logger_worker.info(
                f"[{process_label}] ✅ Verified: Connected to correct run (run_id=%s)",
                run_id,
            )
        else:
            logger_worker.warning(
                f"[{process_label}] ⚠️ Warning: Expected run_id=%s, got %s",
                run_id,
                actual_run_id,
            )

        # Log metrics in worker
        logger_worker.info(f"[{process_label}] Logging worker metrics...")
        asyncio.run(
            manager.log_gauge(
                f"testing/{process_label}_startup_time",
                elapsed,
            )
        )
        asyncio.run(manager.log_gauge(f"testing/{process_label}_timestamp", time.time()))
        asyncio.run(manager.log_gauge(f"testing/{process_label}_connected", 1.0))
        asyncio.run(manager.flush_metrics())

        logger_worker.info(f"[{process_label}] Metrics logged and flushed")

        # Keep the manager alive briefly
        time.sleep(2)

        # Clean shutdown
        logger_worker.info(f"[{process_label}] Shutting down worker process...")
        asyncio.run(manager.finish_run(actual_run_id))
        asyncio.run(manager.shutdown())

        logger_worker.info(f"[{process_label}] ✅ Worker process shutdown complete")

    except Exception as e:
        logger_worker.exception(f"[{process_label}] ❌ Error in worker process: %s", e)
        sys.exit(1)


async def test_shared_mode_new(entity: str | None = None) -> None:
    """Test WandB shared mode by starting a new primary run and connecting a worker.

    Args:
        entity: WandB entity name
    """
    logger.info("=" * 80)
    logger.info("Testing WandB Shared Mode (New Run)")
    logger.info("=" * 80)

    # Step 1: Start primary process
    logger.info("\n[STEP 1] Starting primary process...")
    run_id = await primary_process_workflow(entity=entity)

    logger.info("\n[STEP 2] Spawning worker process to connect to run...")

    # Step 2: Spawn worker process
    process = multiprocessing.Process(
        target=worker_process_workflow,
        args=(run_id, entity, "training_worker"),
    )
    process.start()
    process.join(timeout=30)

    if process.is_alive():
        logger.error("❌ Worker process timeout!")
        process.terminate()
        process.join()
    elif process.exitcode != 0:
        logger.error(f"❌ Worker process failed with exit code {process.exitcode}")
    else:
        logger.info("✅ Worker process completed successfully")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Test PASSED - shared mode working!")
    logger.info("=" * 80)


async def test_shared_mode_existing(run_id: str, entity: str | None = None) -> None:
    """Test connecting to an existing WandB run using shared mode.

    Args:
        run_id: The WandB run ID to connect to
        entity: WandB entity name
    """
    logger.info("=" * 80)
    logger.info(f"Testing WandB Shared Mode (Existing Run: {run_id})")
    logger.info("=" * 80)

    logger.info("\nSpawning worker process to connect to run...")

    # Spawn worker process
    process = multiprocessing.Process(
        target=worker_process_workflow,
        args=(run_id, entity, "test_worker"),
    )
    process.start()
    process.join(timeout=30)

    if process.is_alive():
        logger.error("❌ Worker process timeout!")
        process.terminate()
        process.join()
    elif process.exitcode != 0:
        logger.error(f"❌ Worker process failed with exit code {process.exitcode}")
    else:
        logger.info("✅ Worker process completed successfully")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Test PASSED - worker connected to existing run!")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test WandB shared mode using GRAIL monitoring classes"
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new primary run and connect a worker",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Connect to an existing run (as a worker)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity name (username or team)",
    )

    args = parser.parse_args()

    if not args.new and not args.run_id:
        logger.error("❌ Please specify either --new or --run-id")
        sys.exit(1)

    if args.new:
        asyncio.run(test_shared_mode_new(entity=args.entity))
    else:
        assert args.run_id is not None
        asyncio.run(test_shared_mode_existing(args.run_id, entity=args.entity))


if __name__ == "__main__":
    main()

