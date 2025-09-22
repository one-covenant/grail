"""Shared logging utilities for GRAIL.

Currently provides a `ContextFilter` that enriches each `LogRecord` with
standard fields sourced from environment variables so logs are consistent
across modules, miners, and validators.
"""

from __future__ import annotations

import logging
import os


class ContextFilter(logging.Filter):
    """Inject standard fields into each `LogRecord` if missing.

    Fields: service, env, version, network, netuid, wallet, hotkey, run_id.
    Values are sourced from environment variables when available.
    """

    def __init__(
        self,
        service: str | None = None,
        env_name: str | None = None,
        version: str | None = None,
        network: str | None = None,
        netuid: str | None = None,
        wallet: str | None = None,
        hotkey: str | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.service = service or os.environ.get("GRAIL_SERVICE", "grail")
        self.env_name = env_name or os.environ.get("GRAIL_ENV", "dev")
        self.version = version or os.environ.get("GRAIL_VERSION")
        self.network = network or os.environ.get("NETWORK")
        self.netuid = netuid or os.environ.get("NETUID")
        self.wallet = wallet or os.environ.get("BT_WALLET_COLD")
        self.hotkey = hotkey or os.environ.get("BT_WALLET_HOTKEY")
        self.run_id = run_id or os.environ.get("WANDB_RUN_ID")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if not hasattr(record, "service"):
            record.service = self.service
        if not hasattr(record, "env"):
            record.env = self.env_name
        if self.version and not hasattr(record, "version"):
            record.version = self.version
        if self.network and not hasattr(record, "network"):
            record.network = self.network
        if self.netuid and not hasattr(record, "netuid"):
            record.netuid = self.netuid
        if self.wallet and not hasattr(record, "wallet"):
            record.wallet = self.wallet
        if self.hotkey and not hasattr(record, "hotkey"):
            record.hotkey = self.hotkey
        if self.run_id and not hasattr(record, "run_id"):
            record.run_id = self.run_id
        return True


__all__ = ["ContextFilter"]
