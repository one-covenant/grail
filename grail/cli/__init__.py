#!/usr/bin/env python3
"""Unified Typer CLI for GRAIL with Rich logging.

This module creates a single Typer application and registers subcommands
from `mine`, `validate`, and `train` modules. It also configures a Rich
logger and a global verbosity option.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler


console = Console()


# Custom TRACE level consistent with existing modules
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def configure_logging(verbosity: int) -> None:
    """Configure root logger using Rich based on -v count.

    Levels: 0 -> CRITICAL+1 (silent), 1 -> INFO, 2 -> DEBUG, >=3 -> TRACE
    """
    level = (
        TRACE
        if verbosity >= 3
        else (
            logging.DEBUG
            if verbosity == 2
            else (logging.INFO if verbosity == 1 else logging.CRITICAL + 1)
        )
    )

    # Quiet noisy libraries
    for noisy in [
        "websockets",
        "bittensor",
        "bittensor-cli",
        "btdecode",
        "asyncio",
        "aiobotocore.regions",
        "botocore",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    # GRAIL debug details only visible with -vv or higher
    if verbosity < 2:
        logging.getLogger("grail").setLevel(logging.INFO)


app = typer.Typer(
    name="grail",
    no_args_is_help=True,
    add_completion=False,
    help="GRAIL – Guaranteed Rollout Authenticity via Inference Ledger",
)


@app.callback()
def _main_callback(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Increase verbosity (-v INFO, -vv DEBUG, -vvv TRACE)",
    )
) -> None:
    """Configure logging once for all subcommands."""
    configure_logging(verbose)


@app.command("version")
def version() -> None:
    """Show GRAIL version."""
    # Lazy import to avoid circulars
    try:
        from importlib.metadata import version as _v

        console.print(f"grail {_v('grail')}")
    except Exception:
        # Fallback to package attribute if metadata is unavailable
        try:
            from .. import __version__ as _pv

            console.print(f"grail {_pv}")
        except Exception:
            console.print("grail (version unknown)")


def main() -> None:
    """Entry point for console_scripts in pyproject."""
    app()


# Register subcommands from sibling modules
def _register_subcommands() -> None:
    import importlib
    from typing import Callable, Optional

    for mod_name in (
        "grail.cli.mine",
        "grail.cli.validate",
        "grail.cli.train",
    ):
        module = importlib.import_module(mod_name)
        register: Optional[Callable[[typer.Typer], None]] = getattr(module, "register", None)
        if callable(register):
            register(app)


_register_subcommands()
