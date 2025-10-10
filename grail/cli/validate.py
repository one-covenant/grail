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

from ..logging_utils import MinerPrefixFilter, flush_all_logs

# --------------------------------------------------------------------------- #
#                              Logging Setup                                  #
# --------------------------------------------------------------------------- #

logger = logging.getLogger("grail")
logger.addFilter(MinerPrefixFilter())


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
    from ..neurons import ValidatorNeuron

    asyncio.run(ValidatorNeuron(use_drand=use_drand, test_mode=test_mode).main())
