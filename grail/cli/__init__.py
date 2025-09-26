#!/usr/bin/env python3
"""Unified Typer CLI for GRAIL with Rich logging.

This module creates a single Typer application and registers subcommands
from `mine`, `validate`, and `train` modules. It also configures a Rich
logger and a global verbosity option.
"""

from __future__ import annotations

import atexit
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from typing import Callable, cast

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from ..monitoring import initialize_monitoring
from ..monitoring.config import MonitoringConfig
from ..shared.constants import NETUID, NETWORK
from ..shared.logging import ContextFilter

# Load environment variables once for the whole CLI at import time so that
# modules imported during subcommand registration can read them.
try:
    load_dotenv(override=True)
except Exception:
    # Safe to ignore dotenv issues; continue with system env
    pass


console = Console()
logger = logging.getLogger(__name__)
TRACE_ID = str(uuid.uuid4())


def configure_logging(verbosity: int) -> None:
    """Configure root logger using Rich based on -v count.

    Levels: 0 -> CRITICAL+1 (silent), 1 -> INFO, >=2 -> DEBUG
    """
    level = (
        logging.DEBUG
        if verbosity >= 2
        else (logging.INFO if verbosity == 1 else logging.CRITICAL + 1)
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

    console_handler = RichHandler(
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
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console_handler)

    # Optional file logging for Promtail tailing
    try:
        log_file = os.environ.get("GRAIL_LOG_FILE", "").strip()
        if log_file:

            def _parse_size_to_bytes(size_text: str) -> int:
                """Parse sizes like '100MB', '1G', '52428800' into bytes.

                Accepts optional suffixes: B, KB, MB, GB (case-insensitive).
                Falls back to integers (already in bytes) when no suffix.
                """
                text = (size_text or "").strip().upper()
                if not text:
                    return 0
                try:
                    if text.endswith("GB"):
                        return int(float(text[:-2]) * 1024 * 1024 * 1024)
                    if text.endswith("MB"):
                        return int(float(text[:-2]) * 1024 * 1024)
                    if text.endswith("KB"):
                        return int(float(text[:-2]) * 1024)
                    if text.endswith("B"):
                        return int(float(text[:-1]))
                    # No suffix -> assume bytes
                    return int(float(text))
                except Exception:
                    # On parse error, default to 100 MB
                    return 100 * 1024 * 1024

            max_size_text = os.environ.get("GRAIL_LOG_MAX_SIZE", "100MB").strip()
            backup_count_text = os.environ.get("GRAIL_LOG_BACKUP_COUNT", "5").strip()

            max_bytes = _parse_size_to_bytes(max_size_text)
            try:
                backup_count = int(backup_count_text)
            except Exception:
                backup_count = 5

            rotating_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            rotating_handler.setFormatter(formatter)
            root.addHandler(rotating_handler)
            msg_parts = [
                "✅ File logging enabled (rotating):",
                f"path={log_file}",
                f"max={max_bytes}B",
                f"backups={backup_count}",
            ]
            logger.info(" ".join(msg_parts))
        else:
            logger.info("File logging disabled (GRAIL_LOG_FILE not set)")
    except Exception as e:
        # Never fail app due to logging file errors
        logger.warning(f"Failed to enable file logging: {e}")

    # Attach filters so all records carry consistent fields and a level tag
    root.addFilter(ContextFilter())

    class LokiLevelTagFilter(logging.Filter):
        """Ensure each record carries a 'tags' dict with a 'level' key.

        The python-logging-loki handler already adds a 'severity' label. We
        add an explicit 'level' tag too to make queries more ergonomic
        (e.g., {level="INFO"}).
        """

        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            try:
                tags = getattr(record, "tags", None)
                if not isinstance(tags, dict):
                    tags = {}
                if "level" not in tags:
                    tags["level"] = record.levelname
                record.tags = tags
            except Exception:
                # Never fail logging due to tagging issues
                pass
            return True

    root.addFilter(LokiLevelTagFilter())

    # Log shipping mode indication
    promtail_enabled = os.environ.get("PROMTAIL_ENABLE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if promtail_enabled:
        logger.info("📤 Promtail-based log shipping enabled")
    else:
        logger.info("📤  In-process log shipping mode")

    # GRAIL debug details only visible with -vv or higher
    if verbosity < 2:
        logging.getLogger("grail").setLevel(logging.INFO)

    # Log selected network at startup
    def _network_label(n: str) -> str:
        if n == "test":
            return "public testnet"
        if n == "finney":
            return "mainnet"
        return "custom"

    msg = f"Network: {NETWORK} ({_network_label(NETWORK)}), NETUID={NETUID}"
    logger.info(msg)

    # Initialize monitoring system based on environment
    _initialize_monitoring(verbosity)

    # Ensure handlers flush at exit
    def _shutdown_logging() -> None:
        try:
            logging.shutdown()
        except Exception:
            pass

    atexit.register(_shutdown_logging)


def _initialize_monitoring(verbosity: int) -> None:
    """Initialize monitoring system based on environment configuration.

    Args:
        verbosity: CLI verbosity level
    """
    try:
        # Check if monitoring is enabled
        if not MonitoringConfig.is_monitoring_enabled():
            logger.debug("Monitoring disabled by configuration")
            return

        # Get base configuration from environment
        config = MonitoringConfig.from_environment()

        # Adjust configuration based on verbosity
        if verbosity >= 3:
            # High verbosity - use debug configuration
            debug_config = MonitoringConfig.get_debug_config()
            config.update(debug_config)
        elif verbosity == 0:
            # Silent mode - but still allow monitoring if explicitly enabled
            if not MonitoringConfig.is_monitoring_enabled():
                config["backend_type"] = "null"

        # Validate configuration
        errors = MonitoringConfig.validate_config(config)
        if errors:
            logger.warning(f"Invalid monitoring configuration: {errors}")
            config["backend_type"] = "null"  # Fall back to null backend

        # Initialize monitoring
        backend_type = config.pop("backend_type", "wandb")
        initialize_monitoring(backend_type, **config)

        logger.info(f"Monitoring initialized with {backend_type} backend")

    except Exception as e:
        # Don't let monitoring failures break the CLI
        logger.warning(f"Failed to initialize monitoring: {e}")


app = typer.Typer(
    name="grail",
    no_args_is_help=True,
    add_completion=False,
    help=("GRAIL – Guaranteed Rollout Authenticity via Inference Ledger"),
)


# Provide typed wrappers to satisfy mypy's disallow_untyped_decorators
_Callback = Callable[[Callable[..., None]], Callable[..., None]]
_callback_decorator: _Callback = cast(_Callback, app.callback())


@_callback_decorator
def _main_callback(
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Increase verbosity (-v INFO, -vv DEBUG)",
    ),
) -> None:
    """Configure logging once for all subcommands."""
    configure_logging(verbose)


_version_decorator: _Callback = cast(_Callback, app.command("version"))


@_version_decorator
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

    for mod_name in (
        "grail.cli.mine",
        "grail.cli.validate",
        "grail.cli.train",
    ):
        module = importlib.import_module(mod_name)
        register: Callable[[typer.Typer], None] | None = getattr(module, "register", None)
        if callable(register):
            register(app)


_register_subcommands()
