#!/usr/bin/env python3
"""
Minimal Tier 3 test runner that uses existing grail infrastructure.
Starts multiple miners and a validator with different models for testing.
By default, uses hotkeys hk1, hk2, hk3 for the miners.

Usage:
    # Test with different models (uses default hotkeys hk1, hk2)
    python scripts/run_tier3_test.py \
        --miners "Qwen/Qwen2-0.5B-Instruct,google/gemma-3-1b-it" \
        --validator "Qwen/Qwen2-1.5B-Instruct"

    # Test with same model (3 miners with default hotkeys hk1, hk2, hk3)
    python scripts/run_tier3_test.py \
        --n-miners 3 \
        --miner-model "Qwen/Qwen2-0.5B-Instruct" \
        --validator "google/gemma-3-1b-it"

    # Custom hotkeys
    python scripts/run_tier3_test.py \
        --n-miners 3 \
        --hotkeys "custom1,custom2,custom3"
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Tier3TestRunner:
    """Minimal test runner for Tier 3 integration testing."""

    def __init__(self, start_gpu: int = 0):
        self.processes: dict[str, subprocess.Popen] = {}
        self.base_env = self._load_env()
        self.running = True
        self.gpu_assignments: dict[str, int] = {}  # Track GPU assignments
        self.next_gpu = start_gpu  # Next GPU to assign
        self.start_gpu = start_gpu  # Remember starting GPU
        self.log_dir: Path = self._init_log_dir()
        self.log_files: dict[str, IO[str]] = {}
        self.log_locks: dict[str, threading.Lock] = {}

        # Handle signals for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_env(self) -> dict[str, str]:
        """Load environment from .env file."""
        env = os.environ.copy()
        env_file = Path(__file__).parent.parent / ".env"

        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip().strip("\"'")

        logger.info(f"Loaded environment from {env_file}")
        logger.info(f"Using wallet: {env.get('BT_WALLET_COLD', 'unknown')}")
        logger.info(f"WandB project: {env.get('WANDB_PROJECT', 'unknown')}")

        return env

    def _init_log_dir(self) -> Path:
        """Create a timestamped log directory for this test run."""
        root_dir = Path(__file__).parent.parent
        logs_root = root_dir / "logs" / "tier3"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = logs_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log directory: {run_dir}")
        return run_dir

    def _ensure_log_file(self, name: str) -> Path:
        """Ensure a log file and lock exist for a given process name."""
        if name not in self.log_files:
            log_path = self.log_dir / f"{name}.log"
            f = open(log_path, "a", encoding="utf-8")
            self.log_files[name] = f
            self.log_locks[name] = threading.Lock()
            f.write(f"==== {name} log started at {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
            f.flush()
            logger.info(f"{name} log file: {log_path}")
        return self.log_dir / f"{name}.log"

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info("\nReceived shutdown signal, stopping all services...")
        self.running = False
        self.stop_all()
        sys.exit(0)

    def _get_next_gpu(self) -> int:
        """Get the next available GPU index."""
        gpu = self.next_gpu
        self.next_gpu = (self.next_gpu + 1) % 8  # Cycle through 8 GPUs
        return gpu

    def _stream_output(self, name: str, process: subprocess.Popen) -> None:
        """Stream process output to console with prefix."""

        def read_stream(stream: Any, prefix: str) -> None:
            try:
                for line in iter(stream.readline, b""):
                    if line and self.running:
                        text = line.decode(errors="replace")
                        print(f"[{name}] {text.rstrip()}")
                        # Write to per-process log file
                        if name in self.log_files:
                            lock = self.log_locks.get(name)
                            if lock:
                                with lock:
                                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                                    self.log_files[name].write(f"{ts} [{prefix}] {text}")
                                    self.log_files[name].flush()
            except Exception as e:
                if self.running:
                    logger.error(f"Error reading {prefix} for {name}: {e}")

        # Start threads for stdout and stderr
        stdout_thread = threading.Thread(
            target=read_stream, args=(process.stdout, "OUT"), daemon=True
        )
        stderr_thread = threading.Thread(
            target=read_stream, args=(process.stderr, "ERR"), daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

    def start_miner(
        self, index: int, model_name: str, hotkey: Optional[str] = None
    ) -> subprocess.Popen:
        """Start a miner with specific model and optionally specific hotkey."""
        name = f"miner-{index}"
        gpu_id = self._get_next_gpu()
        self.gpu_assignments[name] = gpu_id

        env = self.base_env.copy()
        env.update(
            {
                "GRAIL_MODEL_NAME": model_name,
                "WANDB_RUN_NAME": (f"tier3-miner-{index}-{model_name.split('/')[-1]}"),
                "WANDB_TAGS": f"tier3,miner,{model_name.split('/')[-1]}",
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
            }
        )

        # Set hotkey if provided
        if hotkey:
            env["BT_WALLET_HOT"] = hotkey
            logger.info(f"Using hotkey '{hotkey}' for {name}")

        cmd = ["uv", "run", "grail", "-vv", "mine"]

        hotkey_info = f" with hotkey '{hotkey}'" if hotkey else ""
        logger.info(f"Starting {name} with model: {model_name} on GPU {gpu_id}" + hotkey_info)

        # Ensure per-process log file exists and announce its path
        miner_log_path = self._ensure_log_file(name)
        logger.info(f"Logging output for {name} to {miner_log_path}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )

        self.processes[name] = process
        self._stream_output(name, process)

        return process

    def start_validator(self, model_name: str, test_mode: bool = False) -> subprocess.Popen:
        """Start validator with specific model.

        Args:
            model_name: Model to use for validation
            test_mode: If True, run in test mode (only validate own files)
        """
        name = "validator"
        gpu_id = self._get_next_gpu()
        self.gpu_assignments[name] = gpu_id

        env = self.base_env.copy()
        env.update(
            {
                "GRAIL_MODEL_NAME": model_name,
                "WANDB_RUN_NAME": f"tier3-validator-{model_name.split('/')[-1]}",
                "WANDB_TAGS": f"tier3,validator,{model_name.split('/')[-1]}",
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
            }
        )

        env["BT_WALLET_HOT"] = "grail-hotkey"

        # Build command with appropriate flags
        cmd = ["uv", "run", "grail", "-vv", "validate"]
        if not test_mode:
            cmd.append("--no-test-mode")
            logger.info("Running validator in production mode (will check all miners)")
        else:
            logger.info("Running validator in test mode (will only check own files)")

        logger.info(f"Starting {name} with model: {model_name} on GPU {gpu_id}")

        # Ensure per-process log file exists and announce its path
        validator_log_path = self._ensure_log_file(name)
        logger.info(f"Logging output for {name} to {validator_log_path}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )

        self.processes[name] = process
        self._stream_output(name, process)

        return process

    def stop_all(self) -> None:
        """Stop all running processes."""
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    process.kill()
                    process.wait()

        self.processes.clear()
        logger.info("All services stopped")
        # Close log files
        for f in self.log_files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self.log_files.clear()
        self.log_locks.clear()

    def run(
        self,
        miner_models: list[str],
        validator_model: str,
        miner_hotkeys: Optional[list[str]] = None,
        validator_delay: int = 30,
    ) -> None:
        """Run the test with specified models and optionally specific hotkeys.

        Args:
            miner_models: List of model names for miners
            validator_model: Model name for validator
            miner_hotkeys: Optional list of hotkey names for miners
            validator_delay: Seconds to wait after miners before starting
                validator
                (default: 30)
        """
        logger.info("=" * 60)
        logger.info("Starting Tier 3 Integration Test")
        logger.info(f"Miners: {len(miner_models)} x {miner_models}")
        if miner_hotkeys:
            logger.info(f"Hotkeys: {miner_hotkeys}")
        logger.info(f"Validator: {validator_model}")
        logger.info(f"Validator delay: {validator_delay} seconds")
        logger.info("=" * 60)

        # Start all miners first
        logger.info("\n🚀 Starting miners...")
        for i, model in enumerate(miner_models):
            hotkey = miner_hotkeys[i] if miner_hotkeys and i < len(miner_hotkeys) else None
            self.start_miner(i, model, hotkey)
            if i < len(miner_models) - 1:  # Don't sleep after last miner
                time.sleep(2)  # Stagger startup

        # Wait for miners to fully initialize and register on network
        logger.info(
            f"\n⏳ Waiting {validator_delay} seconds for miners to initialize and register..."
        )
        logger.info("   This ensures miners are ready before validator starts checking")

        # Show countdown for long waits
        if validator_delay > 10:
            for remaining in range(validator_delay, 0, -10):
                logger.info(f"   {remaining} seconds remaining...")
                time.sleep(min(10, remaining))
        else:
            time.sleep(validator_delay)

        # Start validator after delay
        logger.info("\n🎯 Starting validator...")
        # Run in production mode to see all miners
        self.start_validator(validator_model, test_mode=False)

        logger.info("\nAll services started. Press Ctrl+C to stop.")
        logger.info("Check WandB for detailed metrics and logs.")
        logger.info(f"WandB project: {self.base_env.get('WANDB_PROJECT', 'unknown')}")
        logger.info("=" * 60 + "\n")

        # Wait for processes
        try:
            while self.running and any(p.poll() is None for p in self.processes.values()):
                time.sleep(1)

                # Check for crashed processes
                for name, process in self.processes.items():
                    if process.poll() is not None and self.running:
                        logger.warning(f"{name} exited with code: {process.poll()}")
        except KeyboardInterrupt:
            pass
        finally:
            if self.running:
                self.stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run Tier 3 integration test with multiple miners and a validator"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Different models for each miner (uses default hotkeys hk1, hk2)
  %(prog)s --miners "Qwen/Qwen2-0.5B-Instruct,google/gemma-3-1b-it"

  # Same model for 3 miners (uses default hotkeys hk1, hk2, hk3)
  %(prog)s --n-miners 3 --miner-model "Qwen/Qwen2-0.5B-Instruct"

  # Custom validator model with 2 miners (uses hk1, hk2)
  %(prog)s --n-miners 2 --validator "Qwen/Qwen2-1.5B-Instruct"

  # Custom hotkeys (override defaults)
  %(prog)s --n-miners 3 --hotkeys "custom1,custom2,custom3"

  # Quick testing with short validator delay
  %(prog)s --n-miners 3 --validator-delay 10
        """,
    )

    # Miner configuration
    miner_group = parser.add_mutually_exclusive_group(required=True)
    miner_group.add_argument(
        "--miners",
        type=str,
        help="Comma-separated list of model names for miners",
    )
    miner_group.add_argument("--n-miners", type=int, help="Number of miners to run with same model")

    parser.add_argument(
        "--miner-model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Model to use for all miners (with --n-miners)",
    )

    # Validator configuration
    parser.add_argument(
        "--validator",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model to use for validator",
    )

    # GPU configuration
    parser.add_argument(
        "--start-gpu",
        type=int,
        default=0,
        help="Starting GPU index (default: 0, useful to skip busy GPUs)",
    )

    # Hotkey configuration
    parser.add_argument(
        "--hotkeys",
        type=str,
        default="hk1,hk2,hk3",
        help=('Comma-separated list of hotkey names for miners (default: "hk1,hk2,hk3")'),
    )

    # Validator delay configuration
    parser.add_argument(
        "--validator-delay",
        type=int,
        default=30,
        help=("Seconds to wait after starting miners before starting validator (default: 30)"),
    )

    args = parser.parse_args()

    # Determine miner models
    if args.miners:
        miner_models = [m.strip() for m in args.miners.split(",")]
    else:
        miner_models = [args.miner_model] * args.n_miners

    # Parse hotkeys
    miner_hotkeys = [h.strip() for h in args.hotkeys.split(",")]

    # Check if number of hotkeys matches number of miners
    if len(miner_hotkeys) < len(miner_models):
        logger.error(
            f"Error: Not enough hotkeys ({len(miner_hotkeys)}) for {len(miner_models)} miners"
        )
        logger.error(f"Provided hotkeys: {miner_hotkeys}")
        logger.error("Either provide more hotkeys with --hotkeys or reduce number of miners")
        sys.exit(1)
    elif len(miner_hotkeys) > len(miner_models):
        # Just use the first N hotkeys
        miner_hotkeys = miner_hotkeys[: len(miner_models)]
        logger.info(f"Using first {len(miner_models)} hotkeys: {miner_hotkeys}")

    # Validate number of services vs GPUs
    total_services = len(miner_models) + 1  # +1 for validator
    available_gpus = 8 - args.start_gpu

    if total_services > available_gpus:
        logger.error(
            f"Error: Cannot run {len(miner_models)} miners + 1 validator = "
            f"{total_services} services"
        )
        logger.error(f"Only {available_gpus} GPUs available starting from GPU {args.start_gpu}")
        sys.exit(1)
    elif total_services > available_gpus - 2:
        logger.warning(
            f"Warning: Running {len(miner_models)} miners + 1 validator = {total_services} services"
        )
        logger.warning(f"Using GPUs {args.start_gpu} through {args.start_gpu + total_services - 1}")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)

    # Run the test
    runner = Tier3TestRunner(start_gpu=args.start_gpu)
    runner.run(miner_models, args.validator, miner_hotkeys, args.validator_delay)


if __name__ == "__main__":
    main()
