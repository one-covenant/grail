#!/usr/bin/env python3
"""Minimal Tier 3 test runner that utilizes existing GRAIL infrastructure.

This script starts multiple miners and a validator, optionally with
custom hotkeys.  It is primarily intended for local integration testing
on a single machine with multiple GPUs.

Examples
--------
Run with two different miner models (default hotkeys ``hk1`` and ``hk2``)::

    python scripts/run_tier3_test.py \
        --miners "Qwen/Qwen2-0.5B-Instruct,google/gemma-3-1b-it" \
        --validator "Qwen/Qwen2-1.5B-Instruct"

Run three miners that share the same model (default hotkeys
``hk1``, ``hk2``, ``hk3``)::

    python scripts/run_tier3_test.py \
        --n-miners 3 \
        --miner-model "Qwen/Qwen2-0.5B-Instruct" \
        --validator "google/gemma-3-1b-it"

Specify custom hotkeys::

    python scripts/run_tier3_test.py \
        --n-miners 3 \
        --hotkeys "custom1,custom2,custom3"
"""
from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, IO, List, Optional

# ----------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ----------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Runner implementation -------------------------------------------------------
# ----------------------------------------------------------------------------


class Tier3TestRunner:
    """Minimal test-runner for Tier 3 integration testing."""

    def __init__(self, start_gpu: int = 0) -> None:
        self.processes: Dict[str, subprocess.Popen[bytes]] = {}
        self.base_env: Dict[str, str] = self._load_env()
        self.running: bool = True
        self.gpu_assignments: Dict[str, int] = {}
        self.next_gpu: int = start_gpu
        self.start_gpu: int = start_gpu

        # Handle signals for clean shutdown.
        signal.signal(signal.SIGINT, self._signal_handler)  # type: ignore[arg-type]
        signal.signal(signal.SIGTERM, self._signal_handler)  # type: ignore[arg-type]

    # ---------------------------------------------------------------------
    # Internal helpers ----------------------------------------------------
    # ---------------------------------------------------------------------

    def _load_env(self) -> Dict[str, str]:
        """Load environment variables from a local ``.env`` file (if present)."""
        env: Dict[str, str] = os.environ.copy()
        env_file = Path(__file__).resolve().parent.parent / ".env"

        if env_file.exists():
            with env_file.open("r", encoding="utf-8") as fp:
                for line in fp:
                    stripped = line.strip()
                    if stripped and "=" in stripped and not stripped.startswith("#"):
                        key, value = stripped.split("=", 1)
                        env[key.strip()] = value.strip().strip("\"' ")

        logger.info("Loaded environment from %s", env_file)
        logger.info("Using wallet: %s", env.get("BT_WALLET_COLD", "unknown"))
        logger.info("WandB project: %s", env.get("WANDB_PROJECT", "unknown"))
        return env

    # ------------------------------------------------------------------
    # Signal handling ---------------------------------------------------
    # ------------------------------------------------------------------

    def _signal_handler(self, signum: int, frame: Optional[object]) -> None:  # noqa: D401
        """Handle *nix shutdown signals."""
        _ = signum  # ‑ unused but retained for signature clarity
        _ = frame
        logger.info("Received shutdown signal, stopping all services …")
        self.running = False
        self.stop_all()
        # Terminate the interpreter to propagate the signal exit status.
        sys.exit(0)

    # ------------------------------------------------------------------
    # GPU helpers -------------------------------------------------------
    # ------------------------------------------------------------------

    def _get_next_gpu(self) -> int:
        """Round-robin scheduling across the eight GPUs present."""
        gpu = self.next_gpu
        self.next_gpu = (self.next_gpu + 1) % 8
        return gpu

    # ------------------------------------------------------------------
    # Process I/O -------------------------------------------------------
    # ------------------------------------------------------------------

    def _stream_output(self, name: str, process: subprocess.Popen[bytes]) -> None:
        """Continuously stream *stdout*/*stderr* from *process* with prefix."""

        def read_stream(stream: IO[bytes], prefix: str) -> None:
            try:
                for line in iter(stream.readline, b""):
                    if not self.running:
                        break
                    if line:
                        print(f"[{name}][{prefix}] {line.decode().rstrip()}")
            except Exception as exc:  # pylint: disable=broad-except
                if self.running:
                    logger.error("Error reading %s for %s: %s", prefix, name, exc)

        threading.Thread(target=read_stream, args=(process.stdout, "OUT"), daemon=True).start()  # type: ignore[arg-type]
        threading.Thread(target=read_stream, args=(process.stderr, "ERR"), daemon=True).start()  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------

    def start_miner(self, index: int, model_name: str, hotkey: Optional[str] = None) -> subprocess.Popen[bytes]:
        """Spawn a miner instance.

        Parameters
        ----------
        index:
            Numerical index to distinguish multiple miner processes.
        model_name:
            Hugging-Face ID of the model to load for this miner.
        hotkey:
            Optional wallet hotkey to bind to the miner process.
        """
        name = f"miner-{index}"
        gpu_id = self._get_next_gpu()
        self.gpu_assignments[name] = gpu_id

        env = self.base_env.copy()
        env.update(
            {
                "GRAIL_MODEL_NAME": model_name,
                "WANDB_RUN_NAME": f"tier3-miner-{index}-{model_name.split('/')[-1]}",
                "WANDB_TAGS": f"tier3,miner,{model_name.split('/')[-1]}",
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
            }
        )

        if hotkey is not None:
            env["BT_WALLET_HOT"] = hotkey
            logger.info("Using hotkey '%s' for %s", hotkey, name)

        cmd: List[str] = ["uv", "run", "grail", "-vv", "mine"]
        logger.info("Starting %s with model: %s on GPU %s", name, model_name, gpu_id)
        process = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        self.processes[name] = process
        self._stream_output(name, process)
        return process

    def start_validator(self, model_name: str) -> subprocess.Popen[bytes]:
        """Spawn a validator instance."""
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

        cmd: List[str] = ["uv", "run", "grail", "-vv", "validate"]
        logger.info("Starting %s with model: %s on GPU %s", name, model_name, gpu_id)
        process = subprocess.Popen(  # pylint: disable=subprocess-popen-preexec-fn
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
        )
        self.processes[name] = process
        self._stream_output(name, process)
        return process

    # ------------------------------------------------------------------
    # Lifecycle management ---------------------------------------------
    # ------------------------------------------------------------------

    def stop_all(self) -> None:
        """Terminate all child processes gracefully (SIGTERM)."""
        for name, process in list(self.processes.items()):
            if process.poll() is None:
                logger.info("Stopping %s …", name)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing %s …", name)
                    process.kill()
                    process.wait()
        self.processes.clear()
        logger.info("All services stopped")

    # ------------------------------------------------------------------
    # Orchestration -----------------------------------------------------
    # ------------------------------------------------------------------

    def run(
        self,
        miner_models: List[str],
        validator_model: str,
        miner_hotkeys: Optional[List[str]] | None = None,
    ) -> None:
        """Start miners and a validator, then wait until interrupted."""
        logger.info("=" * 60)
        logger.info("Starting Tier 3 integration test")
        logger.info("Miners (%d): %s", len(miner_models), miner_models)
        if miner_hotkeys:
            logger.info("Hotkeys: %s", miner_hotkeys)
        logger.info("Validator: %s", validator_model)
        logger.info("=" * 60)

        for idx, model in enumerate(miner_models):
            hotkey = miner_hotkeys[idx] if miner_hotkeys and idx < len(miner_hotkeys) else None
            self.start_miner(idx, model, hotkey)
            time.sleep(2)  # Stagger start-up to avoid contention.

        logger.info("Waiting for miners to initialise …")
        time.sleep(5)

        self.start_validator(validator_model)

        logger.info("
All services started. Press Ctrl+C to stop.")
        logger.info("Check WandB for detailed metrics and logs.")
        logger.info("=" * 60)

        try:
            while self.running and any(proc.poll() is None for proc in self.processes.values()):
                time.sleep(1)
                for name, proc in list(self.processes.items()):
                    if proc.poll() is not None and self.running:
                        logger.warning("%s exited with code: %s", name, proc.poll())
        finally:
            if self.running:
                self.stop_all()


# ----------------------------------------------------------------------------
# CLI entrypoint -------------------------------------------------------------
# ----------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Tier 3 integration test with multiple miners and a validator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Miner configuration ----------------------------------------------
    # ------------------------------------------------------------------

    miner_group = parser.add_mutually_exclusive_group(required=True)
    miner_group.add_argument(
        "--miners",
        type=str,
        help="Comma-separated list of model names for miners.",
    )
    miner_group.add_argument(
        "--n-miners",
        type=int,
        help="Number of miners to run with the same model.",
    )

    parser.add_argument(
        "--miner-model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Model to use for all miners (used with --n-miners).",
    )

    # ------------------------------------------------------------------
    # Validator configuration ------------------------------------------
    # ------------------------------------------------------------------

    parser.add_argument(
        "--validator",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model to use for the validator.",
    )

    # ------------------------------------------------------------------
    # GPU configuration -------------------------------------------------
    # ------------------------------------------------------------------

    parser.add_argument(
        "--start-gpu",
        type=int,
        default=0,
        help="GPU index to start allocation from (allows skipping busy GPUs).",
    )

    # ------------------------------------------------------------------
    # Hotkey configuration ---------------------------------------------
    # ------------------------------------------------------------------

    parser.add_argument(
        "--hotkeys",
        type=str,
        default="hk1,hk2,hk3",
        help="Comma-separated list of hotkeys for miners.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] | None = None) -> None:  # noqa: D401
    """Program entry-point."""
    args = _parse_args(argv)

    miner_models: List[str]
    if args.miners:
        miner_models = [m.strip() for m in args.miners.split(",") if m.strip()]
    else:
        miner_models = [args.miner_model] * args.n_miners  # type: ignore[attr-defined]

    miner_hotkeys = [hk.strip() for hk in args.hotkeys.split(",") if hk.strip()]

    if len(miner_hotkeys) < len(miner_models):
        logger.error("Not enough hotkeys (%d) for %d miners", len(miner_hotkeys), len(miner_models))
        sys.exit(1)
    if len(miner_hotkeys) > len(miner_models):
        miner_hotkeys = miner_hotkeys[: len(miner_models)]
        logger.info("Using first %d hotkeys: %s", len(miner_models), miner_hotkeys)

    total_services = len(miner_models) + 1  # +1 for validator
    available_gpus = 8 - args.start_gpu

    if total_services > available_gpus:
        logger.error("Cannot run %d miners + 1 validator = %d services", len(miner_models), total_services)
        logger.error("Only %d GPUs available starting from GPU %d", available_gpus, args.start_gpu)
        sys.exit(1)

    runner = Tier3TestRunner(start_gpu=args.start_gpu)
    runner.run(miner_models, args.validator, miner_hotkeys)


if __name__ == "__main__":
    main()