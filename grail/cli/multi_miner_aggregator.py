#!/usr/bin/env python3
"""
Multi-Miner Aggregator for GRAIL

Coordinates multiple miners running on the same machine and aggregates
their results into a single window upload to R2.

Usage:
    python -m grail.cli.multi_miner_aggregator \
        --hotkeys miner_1 miner_2 miner_3 miner_4 \
        --aggregation-hotkey aggregator_hotkey \
        --mode watch  # or 'batch'
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bittensor as bt
import typer

from ..infrastructure.comms import (
    upload_file_chunked,
)
from ..infrastructure.credentials import load_r2_credentials
from ..shared.constants import WINDOW_LENGTH
from . import console

logger = logging.getLogger("grail.aggregator")


# --------------------------------------------------------------------------- #
#                       Configuration & State                                #
# --------------------------------------------------------------------------- #


@dataclass
class AggregatorConfig:
    """Configuration for multi-miner aggregation."""

    hotkeys: list[str]
    aggregation_hotkey: str
    cold_wallet: str = "default"
    results_dir: Path = Path("/tmp/grail_miner_results")
    poll_interval: float = 5.0  # seconds between polls
    window_timeout: float = 300.0  # seconds to wait for all miners per window
    credentials: Any | None = None


class WindowAggregator:
    """Aggregates results from multiple miners for a specific window."""

    def __init__(self, window_start: int, config: AggregatorConfig):
        self.window_start = window_start
        self.config = config
        self.results: dict[str, list[dict]] = {hotkey: [] for hotkey in config.hotkeys}
        self.collected_hotkeys: set[str] = set()
        self.start_time = time.time()

    async def collect_results(self) -> dict[str, list[dict]]:
        """Poll for results from all miners until timeout or all collected."""
        logger.info(
            f"🔍 Collecting results for window {self.window_start} "
            f"from {len(self.config.hotkeys)} miners..."
        )

        while time.time() - self.start_time < self.config.window_timeout:
            # Check each miner's result directory
            for hotkey in self.config.hotkeys:
                if hotkey in self.collected_hotkeys:
                    continue  # Already collected

                result_file = self._get_result_path(hotkey)
                if result_file.exists():
                    try:
                        inferences = await self._load_and_parse(result_file, hotkey)
                        self.results[hotkey] = inferences
                        self.collected_hotkeys.add(hotkey)
                        logger.info(f"  ✓ {hotkey}: {len(inferences)} inferences collected")
                    except Exception as e:
                        logger.warning(f"  ✗ {hotkey}: Failed to load results - {e}")

            # Check if we have all results
            if len(self.collected_hotkeys) == len(self.config.hotkeys):
                logger.info(
                    f"✅ All {len(self.config.hotkeys)} miners reported for window "
                    f"{self.window_start}"
                )
                break

            # Log progress
            elapsed = time.time() - self.start_time
            remaining = self.config.window_timeout - elapsed
            pending = len(self.config.hotkeys) - len(self.collected_hotkeys)
            if pending > 0:
                logger.debug(f"  ⏳ Waiting for {pending} miners ({remaining:.0f}s remaining)...")

            await asyncio.sleep(self.config.poll_interval)

        # Log final status
        if len(self.collected_hotkeys) < len(self.config.hotkeys):
            missing = set(self.config.hotkeys) - self.collected_hotkeys
            logger.warning(
                f"⚠️ Timeout: Missing results from {missing}. "
                f"Uploading partial results ({len(self.collected_hotkeys)}/{len(self.config.hotkeys)})"
            )

        return self.results

    async def aggregate_and_upload(self, wallet: bt.wallet) -> bool:
        """Aggregate all collected results and upload to R2."""
        # Flatten all inferences
        all_inferences: list[dict] = []
        for inferences in self.results.values():
            all_inferences.extend(inferences)

        if not all_inferences:
            logger.warning(f"No inferences to upload for window {self.window_start}")
            return False

        # Create window data with aggregation metadata
        window_data = {
            "wallet": wallet.hotkey.ss58_address,
            "window_start": self.window_start,
            "window_length": WINDOW_LENGTH,
            "inference_count": len(all_inferences),
            "inferences": all_inferences,
            "timestamp": time.time(),
            "aggregated": True,
            "miner_count": len(self.collected_hotkeys),
            "miner_hotkeys": list(self.collected_hotkeys),
            "collection_time_seconds": time.time() - self.start_time,
        }

        # Upload to R2
        key = (
            f"grail/windows/aggregated/{wallet.hotkey.ss58_address}-window-{self.window_start}.json"
        )
        body = json.dumps(window_data).encode()

        logger.info(
            f"📤 Uploading aggregated window {self.window_start} "
            f"({len(all_inferences)} inferences from {len(self.collected_hotkeys)} miners)..."
        )

        success = await upload_file_chunked(
            key,
            body,
            credentials=self.config.credentials,
            use_write=True,
        )

        if success:
            logger.info(f"✅ Successfully uploaded aggregated window {self.window_start} to R2")
            # Clean up local result files
            await self._cleanup_results()
        else:
            logger.error(f"❌ Failed to upload aggregated window {self.window_start}")

        return success

    def _get_result_path(self, hotkey: str) -> Path:
        """Get path where miner should write results."""
        return self.config.results_dir / f"{hotkey}-window-{self.window_start}.json"

    async def _load_and_parse(self, result_file: Path, hotkey: str) -> list[dict]:
        """Load and parse inferences from result file."""
        try:
            with open(result_file) as f:
                data = json.load(f)
                inferences = data.get("inferences", [])
                if not isinstance(inferences, list):
                    raise ValueError(f"Expected list of inferences, got {type(inferences)}")
                return inferences
        except Exception as e:
            logger.debug(f"Failed to parse {result_file}: {e}")
            raise

    async def _cleanup_results(self) -> None:
        """Remove processed result files."""
        for hotkey in self.collected_hotkeys:
            result_file = self._get_result_path(hotkey)
            try:
                if result_file.exists():
                    result_file.unlink()
                    logger.debug(f"Cleaned up {result_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {result_file}: {e}")


class MultiMinerAggregatorService:
    """Main service for coordinating multi-miner aggregation."""

    def __init__(self, config: AggregatorConfig):
        self.config = config
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = asyncio.Event()

    async def watch_and_aggregate(self) -> None:
        """Watch for window completions and aggregate results."""
        logger.info(f"🚀 Starting multi-miner aggregator for {len(self.config.hotkeys)} miners")
        logger.info(f"   Miners: {', '.join(self.config.hotkeys)}")
        logger.info(f"   Results directory: {self.config.results_dir}")
        logger.info(f"   Poll interval: {self.config.poll_interval}s")
        logger.info(f"   Window timeout: {self.config.window_timeout}s")

        wallet = bt.wallet(name=self.config.cold_wallet, hotkey=self.config.aggregation_hotkey)
        last_window = -1

        try:
            while not self.stop_event.is_set():
                # Get current window
                subtensor = bt.subtensor()
                current_block = await asyncio.to_thread(subtensor.get_current_block)
                current_window = (current_block // WINDOW_LENGTH) * WINDOW_LENGTH

                # New window detected
                if current_window > last_window:
                    logger.info(f"📍 New window detected: {current_window} (block {current_block})")
                    last_window = current_window

                    # Process previous window if we have results
                    if current_window > WINDOW_LENGTH:
                        prev_window = current_window - WINDOW_LENGTH
                        await self._process_window(wallet, prev_window)

                await asyncio.sleep(self.config.poll_interval)

        except KeyboardInterrupt:
            logger.info("Stopping aggregator...")
        except Exception as e:
            logger.error(f"Error in aggregator: {e}", exc_info=True)
            raise

    async def _process_window(self, wallet: bt.wallet, window_start: int) -> None:
        """Process and upload a specific window."""
        aggregator = WindowAggregator(window_start, self.config)
        results = await aggregator.collect_results()

        # Check if we have any results
        total_inferences = sum(len(inf) for inf in results.values())
        if total_inferences == 0:
            logger.info(f"⊘ No results for window {window_start}, skipping")
            return

        # Upload aggregated results
        await aggregator.aggregate_and_upload(wallet)

    async def batch_process_window(self, window_start: int) -> bool:
        """Process a single window in batch mode."""
        wallet = bt.wallet(name=self.config.cold_wallet, hotkey=self.config.aggregation_hotkey)
        await self._process_window(wallet, window_start)
        return True


# --------------------------------------------------------------------------- #
#                              CLI Interface                                 #
# --------------------------------------------------------------------------- #


def register(app: typer.Typer) -> None:
    """Register aggregator command with CLI."""
    app.command("aggregate")(aggregate)


def aggregate(
    hotkeys: list[str] = typer.Option(
        ...,
        "--hotkey",
        help="Miner hotkeys to aggregate (can specify multiple times)",
    ),
    aggregation_hotkey: str = typer.Option(
        ...,
        "--aggregation-hotkey",
        help="Hotkey to use for uploading aggregated results",
    ),
    cold_wallet: str = typer.Option(
        "default",
        "--cold-wallet",
        help="Cold wallet name",
    ),
    results_dir: str = typer.Option(
        "/tmp/grail_miner_results",
        "--results-dir",
        help="Directory where miners write results",
    ),
    poll_interval: float = typer.Option(
        5.0,
        "--poll-interval",
        help="Seconds between polls for new results",
    ),
    window_timeout: float = typer.Option(
        300.0,
        "--window-timeout",
        help="Seconds to wait for all miners per window",
    ),
    mode: str = typer.Option(
        "watch",
        "--mode",
        help="'watch' for continuous monitoring or 'batch' for single window",
    ),
    window: int | None = typer.Option(
        None,
        "--window",
        help="Window to process (required for batch mode)",
    ),
) -> None:
    """Aggregate results from multiple miners and upload to R2.

    Example:
        python -m grail.cli.multi_miner_aggregator \
            --hotkey miner_1 --hotkey miner_2 --hotkey miner_3 \
            --aggregation-hotkey aggregator \
            --mode watch

        python -m grail.cli.multi_miner_aggregator \
            --hotkey miner_1 --hotkey miner_2 \
            --aggregation-hotkey aggregator \
            --mode batch --window 12345
    """
    try:
        # Validate inputs
        if not hotkeys:
            console.print("[red]Error: At least one --hotkey must be specified[/red]")
            raise typer.Exit(code=1)

        if mode not in ("watch", "batch"):
            console.print(f"[red]Error: mode must be 'watch' or 'batch', got {mode}[/red]")
            raise typer.Exit(code=1)

        if mode == "batch" and window is None:
            console.print("[red]Error: --window required for batch mode[/red]")
            raise typer.Exit(code=1)

        # Load credentials
        try:
            credentials = load_r2_credentials()
        except Exception as e:
            console.print(f"[red]Failed to load R2 credentials: {e}[/red]")
            raise typer.Exit(code=1) from None

        # Create config
        config = AggregatorConfig(
            hotkeys=hotkeys,
            aggregation_hotkey=aggregation_hotkey,
            cold_wallet=cold_wallet,
            results_dir=Path(results_dir),
            poll_interval=poll_interval,
            window_timeout=window_timeout,
            credentials=credentials,
        )

        # Run aggregator
        service = MultiMinerAggregatorService(config)

        if mode == "watch":
            console.print("[bold green]Starting multi-miner aggregator in watch mode[/bold green]")
            asyncio.run(service.watch_and_aggregate())
        else:  # batch
            console.print(f"[bold green]Processing window {window}[/bold green]")
            asyncio.run(service.batch_process_window(window))

    except KeyboardInterrupt:
        console.print("[yellow]Aggregator stopped by user[/yellow]")
        raise typer.Exit(code=0) from None
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                  #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Main entry point for aggregator CLI."""

    app = typer.Typer()
    register(app)
    app()


if __name__ == "__main__":
    main()
