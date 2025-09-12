#!/usr/bin/env python3
"""
Utilities for subnet-related helpers shared across miner and validator.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("grail")


async def get_own_uid_on_subnet(subtensor: Any, netuid: int, hotkey_ss58: str) -> Optional[int]:
    """Return the UID for a given hotkey on a subnet, or None if not registered.

    Args:
        subtensor: Initialized async subtensor client.
        netuid: Subnet identifier.
        hotkey_ss58: SS58 address for the wallet hotkey.

    Returns:
        UID as int if present in the metagraph; otherwise None.
    """
    try:
        meta = await subtensor.metagraph(netuid)
        uid_by_hotkey = dict(zip(meta.hotkeys, meta.uids))
        uid = uid_by_hotkey.get(hotkey_ss58)
        return int(uid) if uid is not None else None
    except Exception as e:
        logger.debug("Failed to resolve own UID on subnet %s: %s", netuid, e)
        return None
