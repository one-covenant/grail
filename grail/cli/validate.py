#!/usr/bin/env python3
"""Validation CLI entry point.

This module provides the CLI command for running validation.
All validation logic is in grail.validation.ValidationService.
"""

import asyncio
import atexit
import faulthandler
import logging
import signal
import sys
from types import TracebackType

import typer

from ..logging_utils import flush_all_logs

# --------------------------------------------------------------------------- #
#                              Logging Setup                                  #
# --------------------------------------------------------------------------- #

logger = logging.getLogger("grail")


# --------------------------------------------------------------------------- #
#                           Crash Diagnostics                                 #
# --------------------------------------------------------------------------- #


def _install_crash_diagnostics() -> None:
    """Enable faulthandler and global exception logging for silent crashes."""
    # Dump Python tracebacks on fatal signals and C-level faults
    try:
        faulthandler.enable(all_threads=True)
        # Register common termination signals to dump tracebacks before exit
        for sig in (
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGABRT", None),
            getattr(signal, "SIGSEGV", None),
        ):
            if sig is not None:
                try:
                    faulthandler.register(sig, chain=True)
                except Exception:
                    pass
    except Exception:
        pass

    # Ensure unhandled exceptions get logged
    def _excepthook(
        exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
    ) -> None:
        try:
            if exc_type is KeyboardInterrupt:
                # Let standard handling occur for Ctrl-C
                return sys.__excepthook__(exc_type, exc, tb)
            logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        finally:
            flush_all_logs()

    try:
        sys.excepthook = _excepthook
    except Exception:
        pass

    # Flush logs on normal interpreter exit
    try:
        atexit.register(flush_all_logs)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#                       Validator Startup GPU Probe                           #
# --------------------------------------------------------------------------- #


def _probe_validator_gpus() -> None:
    """Verify a CUDA device is visible before any other validator setup runs.

    Validators load a multi-billion-parameter checkpoint and run proof
    verification forward passes. A silent CPU fallback pins every vCPU at
    100% and exhausts host RAM, which has frozen production hosts.

    This probe is intentionally scoped to the validator CLI entry only.
    Miners and other grail entry points (mine, train) stay device-agnostic
    and are free to run on CPU, MLX, MacBook M4, etc.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Validator startup GPU probe failed: torch.cuda.is_available() is False. "
            "The validator requires a visible CUDA device. "
            "If running via Docker, run the preflight "
            "`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` "
            "on the host. If that fails, reinstall or reconfigure "
            "nvidia-container-toolkit (`sudo nvidia-ctk runtime configure "
            "--runtime=docker && sudo systemctl restart docker`). "
            "See docs/validator.md troubleshooting for the full diagnostic flow."
        )

    device_count = torch.cuda.device_count()
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    logger.info(
        "Validator startup GPU probe: count=%d names=%s",
        device_count,
        device_names,
    )


# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #


def register(app: typer.Typer) -> None:
    """Register the validate command with the CLI app."""
    app.command("validate")(validate)


# --------------------------------------------------------------------------- #
#                               Validator                                     #
# --------------------------------------------------------------------------- #


def validate(
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Include drand in challenge randomness (default: True)",
        show_default=True,
    ),
    test_mode: bool = typer.Option(
        False,
        "--test-mode/--no-test-mode",
        help="Test mode: validate only own files (default: False)",
        show_default=True,
    ),
) -> None:
    """Run the validation neuron.

    This is the CLI entry point for validation. All logic is delegated to
    ValidatorNeuron, which uses ValidationService for the actual validation work.

    Args:
        use_drand: Include drand in challenge randomness
        test_mode: Validate only own files for testing
    """
    # Install crash diagnostics early to catch silent failures
    _install_crash_diagnostics()

    # Fail loud and early if no CUDA device is visible to this process.
    # Runs before any subtensor, R2, pool, or checkpoint work so a broken
    # GPU setup is diagnosed at second 0 instead of ~2 minutes in.
    _probe_validator_gpus()

    from ..neurons import ValidatorNeuron

    asyncio.run(ValidatorNeuron(use_drand=use_drand, test_mode=test_mode).main())
