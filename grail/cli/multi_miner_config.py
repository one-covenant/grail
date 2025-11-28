"""
Multi-Miner Configuration and Helper Utilities

Provides common configurations and helper functions for running multiple
miners on the same machine with window-based result aggregation.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MinerConfig:
    """Configuration for a single miner instance."""

    hotkey: str
    gpu_index: int | None = None
    batch_size: int = 2
    safety_blocks: int = 3
    use_drand: bool = True
    extra_env: dict[str, str] | None = None


@dataclass
class MultiMinerSetup:
    """Complete setup for multiple miners."""

    miners: list[MinerConfig]
    cold_wallet: str = "default"
    use_aggregator: bool = True
    aggregator_hotkey: str | None = None
    results_directory: Path = Path("/tmp/grail_miner_results")
    poll_interval_seconds: float = 5.0
    window_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.miners:
            raise ValueError("At least one miner config is required")
        if self.use_aggregator and not self.aggregator_hotkey:
            import time

            self.aggregator_hotkey = f"aggregator_{int(time.time())}"


class MultiMinerBuilder:
    """Builder for creating multi-miner configurations."""

    @staticmethod
    def from_hotkeys(
        hotkeys: list[str],
        gpus: list[int] | None = None,
        batch_size: int = 2,
        use_aggregator: bool = True,
    ) -> MultiMinerSetup:
        """Create multi-miner setup from list of hotkeys and optional GPU assignments.

        Args:
            hotkeys: List of miner hotkeys
            gpus: Optional list of GPU indices (cycles if fewer than hotkeys)
            batch_size: Generation batch size per miner
            use_aggregator: Whether to enable result aggregation

        Returns:
            MultiMinerSetup ready to launch
        """
        if not hotkeys:
            raise ValueError("At least one hotkey required")

        # Build miner configs
        miners = []
        for i, hotkey in enumerate(hotkeys):
            gpu = gpus[i % len(gpus)] if gpus else None
            miners.append(
                MinerConfig(
                    hotkey=hotkey,
                    gpu_index=gpu,
                    batch_size=batch_size,
                )
            )

        return MultiMinerSetup(
            miners=miners,
            use_aggregator=use_aggregator,
        )

    @staticmethod
    def from_environment() -> MultiMinerSetup:
        """Create multi-miner setup from environment variables.

        Environment variables:
            GRAIL_MINERS: Comma-separated hotkey list (e.g., "miner_1,miner_2,miner_3")
            GRAIL_GPUS: Comma-separated GPU indices (optional, e.g., "0,1,2")
            GRAIL_BATCH_SIZE: Generation batch size (default: 2)
            GRAIL_USE_AGGREGATOR: "true" or "false" (default: true)
            GRAIL_AGGREGATOR_HOTKEY: Aggregator identity (auto-generated if not set)
            GRAIL_RESULTS_DIR: Results directory (default: /tmp/grail_miner_results)

        Returns:
            MultiMinerSetup from environment configuration
        """
        # Parse miners
        miners_str = os.getenv("GRAIL_MINERS", "miner_1")
        hotkeys = [h.strip() for h in miners_str.split(",") if h.strip()]

        if not hotkeys:
            raise ValueError("GRAIL_MINERS environment variable is empty")

        # Parse GPUs (optional)
        gpus_str = os.getenv("GRAIL_GPUS", "")
        gpus = None
        if gpus_str:
            gpus = [int(g.strip()) for g in gpus_str.split(",") if g.strip()]

        # Other settings
        batch_size = int(os.getenv("GRAIL_BATCH_SIZE", "2"))
        use_aggregator = os.getenv("GRAIL_USE_AGGREGATOR", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        aggregator_hotkey = os.getenv("GRAIL_AGGREGATOR_HOTKEY", None)
        results_dir = Path(os.getenv("GRAIL_RESULTS_DIR", "/tmp/grail_miner_results"))

        setup = MultiMinerBuilder.from_hotkeys(
            hotkeys=hotkeys,
            gpus=gpus,
            batch_size=batch_size,
            use_aggregator=use_aggregator,
        )

        if aggregator_hotkey:
            setup.aggregator_hotkey = aggregator_hotkey

        setup.results_directory = results_dir

        return setup


class MinerLauncher:
    """Helper for launching miner processes with proper environment."""

    @staticmethod
    def get_env_for_miner(config: MinerConfig, cold_wallet: str = "default") -> dict[str, str]:
        """Get environment variables for a miner process.

        Args:
            config: MinerConfig for this miner
            cold_wallet: Cold wallet name

        Returns:
            Dictionary of environment variables to set
        """
        env = os.environ.copy()

        # Set wallet
        env["BT_WALLET_COLD"] = cold_wallet
        env["BT_WALLET_HOT"] = config.hotkey

        # Set GPU if specified
        if config.gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_index)
        else:
            # Remove GPU constraint if not specified
            env.pop("CUDA_VISIBLE_DEVICES", None)

        # Set generation parameters
        env["GRAIL_GENERATION_BATCH_SIZE"] = str(config.batch_size)
        env["GRAIL_MINER_SAFETY_BLOCKS"] = str(config.safety_blocks)

        # Add any extra environment variables
        if config.extra_env:
            env.update(config.extra_env)

        return env

    @staticmethod
    def get_command_for_miner(config: MinerConfig) -> list[str]:
        """Get command to launch a miner.

        Args:
            config: MinerConfig for this miner

        Returns:
            Command as list of strings (suitable for subprocess)
        """
        return [
            "python",
            "-m",
            "grail.cli.mine",
            "--use-drand" if config.use_drand else "--no-drand",
        ]


class AggregatorLauncher:
    """Helper for launching aggregator with proper arguments."""

    @staticmethod
    def get_command_for_aggregator(setup: MultiMinerSetup, mode: str = "watch") -> list[str]:
        """Get command to launch aggregator.

        Args:
            setup: MultiMinerSetup configuration
            mode: "watch" or "batch"

        Returns:
            Command as list of strings
        """
        if not setup.aggregator_hotkey:
            raise ValueError("aggregator_hotkey not set")

        cmd = [
            "python",
            "-m",
            "grail.cli.multi_miner_aggregator",
        ]

        # Add miner hotkeys
        for miner in setup.miners:
            cmd.extend(["--hotkey", miner.hotkey])

        # Add aggregator settings
        cmd.extend(
            [
                "--aggregation-hotkey",
                setup.aggregator_hotkey,
                "--cold-wallet",
                setup.cold_wallet,
                "--results-dir",
                str(setup.results_directory),
                "--poll-interval",
                str(setup.poll_interval_seconds),
                "--window-timeout",
                str(setup.window_timeout_seconds),
                "--mode",
                mode,
            ]
        )

        return cmd


def print_setup_summary(setup: MultiMinerSetup) -> None:
    """Pretty-print the multi-miner setup configuration."""
    print("\n" + "=" * 60)
    print("Multi-Miner Setup Configuration")
    print("=" * 60)

    print(f"\n📊 Miners: {len(setup.miners)}")
    for i, miner in enumerate(setup.miners, 1):
        gpu_info = f"GPU {miner.gpu_index}" if miner.gpu_index is not None else "Any GPU"
        print(f"  {i}. {miner.hotkey:20s} [{gpu_info}] batch_size={miner.batch_size}")

    print(f"\n💼 Wallet: {setup.cold_wallet}")

    if setup.use_aggregator:
        print(f"\n🔄 Aggregator: {setup.aggregator_hotkey}")
        print(f"   Poll interval: {setup.poll_interval_seconds}s")
        print(f"   Window timeout: {setup.window_timeout_seconds}s")
    else:
        print("\n🔄 Aggregator: Disabled")

    print(f"\n📁 Results directory: {setup.results_directory}")
    print("\n" + "=" * 60 + "\n")
