"""Shared utilities for fetching miner window data.

Used by both validator (for validation) and trainer (for training data).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .comms import file_exists, get_file

if TYPE_CHECKING:
    from ..shared.schemas import BucketCredentials
    from .chain import GrailChainManager

logger = logging.getLogger(__name__)


async def fetch_miner_window_data(
    miner_hotkey: str,
    window: int,
    credentials: BucketCredentials | Any,
    chain_manager: GrailChainManager | None = None,
) -> dict | None:
    """Fetch a single miner's window file from their bucket.

    Args:
        miner_hotkey: Miner's hotkey address
        window: Window start block number
        credentials: Default R2 credentials (fallback if no miner bucket)
        chain_manager: Optional chain manager for miner bucket lookup

    Returns:
        Window data dict with 'inferences' list, or None
    """
    filename = f"grail/windows/{miner_hotkey}-window-{window}.json"

    # Try to get miner-specific bucket if chain manager available
    bucket_to_use = credentials
    if chain_manager:
        miner_bucket = chain_manager.get_bucket_for_hotkey(miner_hotkey)
        if miner_bucket:
            bucket_to_use = miner_bucket

    try:
        # Check if file exists
        exists = await file_exists(filename, credentials=bucket_to_use, use_write=False)
        if not exists:
            logger.debug(
                "No file found at %s for miner %s",
                filename,
                miner_hotkey[:12],
            )
            return None

        # Download file
        window_data = await get_file(filename, credentials=bucket_to_use, use_write=False)
        if not window_data:
            logger.warning(
                "Could not download %s for miner %s",
                filename,
                miner_hotkey[:12],
            )
            return None

        return window_data

    except Exception as exc:
        logger.debug(
            "Failed to fetch window data for miner %s: %s",
            miner_hotkey[:12],
            exc,
        )
        return None


async def fetch_multiple_miners_data(
    miner_hotkeys: set[str] | list[str],
    window: int,
    credentials: BucketCredentials | Any,
    chain_manager: GrailChainManager | None = None,
    max_concurrent: int = 10,
) -> dict[str, dict]:
    """Fetch window data from multiple miners in parallel.

    Args:
        miner_hotkeys: Set or list of miner hotkey addresses
        window: Window start block number
        credentials: Default R2 credentials
        chain_manager: Optional chain manager for miner bucket lookup
        max_concurrent: Maximum concurrent downloads

    Returns:
        Dict mapping hotkey -> window_data for successfully fetched
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_with_limit(hotkey: str) -> tuple[str, dict | None]:
        async with semaphore:
            data = await fetch_miner_window_data(hotkey, window, credentials, chain_manager)
            return hotkey, data

    # Fetch all miners in parallel with concurrency limit
    tasks = [_fetch_with_limit(hotkey) for hotkey in miner_hotkeys]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful results
    fetched_data = {}
    for result in results:
        if isinstance(result, BaseException):
            logger.debug("Miner fetch task failed: %s", result)
            continue

        # Type narrowed: result is tuple[str, dict | None]
        hotkey, data = result
        if data is not None:
            fetched_data[hotkey] = data

    logger.info(
        "Fetched data from %d/%d miners for window %s",
        len(fetched_data),
        len(miner_hotkeys),
        window,
    )

    return fetched_data
