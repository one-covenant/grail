"""Separate process worker for blockchain operations to avoid GIL contention."""

import logging
import multiprocessing
import os
import time
from typing import Any

import bittensor as bt
from bittensor.core.chain_data import decode_account_id
from pydantic import ValidationError

from ..shared.schemas import Bucket

logger = logging.getLogger(__name__)


def chain_commitment_worker(
    queue: multiprocessing.Queue,
    netuid: int,
    wallet_name: str,
    wallet_hotkey: str,
    fetch_interval: int,
) -> None:
    """Worker process that fetches commitments and sends them via queue.

    This runs in a completely separate process with its own Python interpreter,
    avoiding any GIL contention with the main mining loop.

    Args:
        queue: Queue to send commitment updates
        netuid: Network UID
        wallet_name: Wallet coldkey name
        wallet_hotkey: Wallet hotkey name
        fetch_interval: Seconds between fetches
    """
    # Set up logging in worker process
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Chain worker process started (PID: {os.getpid()})")

    try:
        # Initialize bittensor connection in worker process
        network = os.getenv("BT_NETWORK", "finney")
        chain_endpoint = os.getenv("BT_CHAIN_ENDPOINT")  # No default

        if chain_endpoint:
            # Custom endpoint
            logger.info(f"Worker using custom endpoint: {chain_endpoint}")
            subtensor = bt.subtensor(network=chain_endpoint)
        else:
            # Named network (finney, test, local)
            logger.info(f"Worker using network: {network}")
            subtensor = bt.subtensor(network=network)

        metagraph = subtensor.metagraph(netuid)
        logger.info(f"Worker initialized subtensor for netuid {netuid}")

        while True:
            try:
                # Sync metagraph
                metagraph.sync(subtensor=subtensor)

                # Fetch commitments (blocking is OK - we're in our own process!)
                commitments = _fetch_commitments_sync(subtensor, metagraph, netuid)

                # Send to main process (non-blocking)
                try:
                    queue.get_nowait()  # Clear old value if present
                except Exception:
                    pass
                queue.put(commitments, block=False)

                logger.info(f"Worker fetched {len(commitments)} commitments")

                # Sleep before next fetch
                time.sleep(fetch_interval)

            except Exception as e:
                logger.error(f"Worker error during fetch: {e}")
                # Try to reinitialize connection
                try:
                    subtensor.substrate.close()
                    subtensor.substrate.initialize()
                except Exception:
                    pass
                time.sleep(60)  # Wait before retry

    except Exception as e:
        logger.error(f"Worker fatal error: {e}")
        raise


def _fetch_commitments_sync(
    subtensor: bt.subtensor,
    metagraph: Any,
    netuid: int,
) -> dict[int, Bucket]:
    """Synchronous commitment fetch - safe because we're in a separate process."""
    try:
        substrate = subtensor.substrate

        # Query commitments via substrate (blocking call - but we're isolated!)
        query_result = substrate.query_map(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid],
            block_hash=None,
        )

        # Create hotkey to UID mapping
        hotkey_to_uid = dict(zip(metagraph.hotkeys, metagraph.uids, strict=False))
        commitments = {}
        total_entries = 0
        skipped_entries = 0

        if query_result is None:
            logger.warning("Chain query returned None, no commitments available")
            return commitments

        # query_map returns an iterable at runtime but Pyright stubs may not reflect this
        for key, value in query_result:  # type: ignore[union-attr]
            total_entries += 1
            try:
                # Decode the commitment
                decoded_ss58 = decode_account_id(key[0])
                commitment = value.value["info"]["fields"][0][0]
                bytes_tuple = commitment[next(iter(commitment.keys()))][0]
                commitment_str = bytes(bytes_tuple).decode()

            except Exception as e:
                logger.debug(f"Failed to decode metadata for key {key}: {e}")
                skipped_entries += 1
                continue

            # Skip if hotkey not in metagraph
            if decoded_ss58 not in hotkey_to_uid:
                skipped_entries += 1
                continue

            uid = hotkey_to_uid[decoded_ss58]

            # Validate commitment length (new format only)
            if len(commitment_str) != 128:
                logger.debug(f"Invalid commitment length for UID {uid}: {len(commitment_str)}")
                skipped_entries += 1
                continue

            try:
                # 32 account_id + 32 access_key_id + 64 secret_access_key
                account_id = commitment_str[:32]
                access_key_id = commitment_str[32:64]
                secret_access_key = commitment_str[64:128]

                # Bucket name equals account id by assumption
                bucket = Bucket(
                    name=account_id,
                    account_id=account_id,
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key,
                )
                commitments[uid] = bucket

            except ValidationError as e:
                logger.debug(f"Failed to validate bucket for UID {uid}: {e}")
                skipped_entries += 1
                continue

        # Log summary of fetch operation
        if skipped_entries > 0:
            logger.debug(
                f"Commitment fetch summary: {len(commitments)} valid, "
                f"{skipped_entries} skipped out of {total_entries} total"
            )

        return commitments

    except Exception as e:
        logger.error(f"Error querying commitments from chain: {e}")
        return {}
