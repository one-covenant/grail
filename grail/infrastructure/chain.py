"""GRAIL-specific chain manager for R2 credentials management."""

import asyncio
import logging
import os
from typing import Any, Optional

import bittensor as bt
from bittensor.core.chain_data import decode_account_id
from pydantic import ValidationError

from ..shared.schemas import Bucket, BucketCredentials

logger = logging.getLogger(__name__)


class GrailChainManager:
    """GRAIL-specific chain manager for handling R2 credential commitments"""

    def __init__(
        self,
        config: Any,
        wallet: bt.wallet,
        credentials: BucketCredentials,
        fetch_interval: int = 600,  # 10 minutes default
    ):
        """
        Initialize GRAIL chain manager.

        Args:
            config: Bittensor config object
            wallet: Wallet for signing commitments
            credentials: Dual R2 credentials
            fetch_interval: Interval in seconds between fetching commitments
        """
        self.config = config
        self.netuid = config.netuid
        self.wallet = wallet
        self.credentials = credentials
        self.fetch_interval = fetch_interval

        network = os.getenv("BT_NETWORK", "finney")
        chain_endpoint = os.getenv("BT_CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")

        if chain_endpoint:
            # When using a custom chain endpoint, pass it as the network parameter
            # This preserves the hostname (e.g., ws://alice:9944) in Docker environments
            self.subtensor = bt.subtensor(network=chain_endpoint)
        else:
            self.subtensor = bt.subtensor(network=network)

        self.metagraph = self.subtensor.metagraph(self.netuid)

        # Commitment tracking
        self.commitments: dict[int, Bucket] = {}
        self._fetch_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized GrailChainManager for netuid {self.netuid}")

    async def initialize(self) -> None:
        """Initialize chain manager and verify/commit credentials"""
        logger.info("Initializing GRAIL chain manager...")

        # Fetch existing commitments
        await self.fetch_commitments()

        # Check if our commitment matches what's on chain
        try:
            uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            existing = self.commitments.get(uid)

            # Compare existing commitment with our read credentials
            our_commitment = self._create_bucket_from_read_creds()

            if not existing or not self._commitments_match(existing, our_commitment):
                if not existing:
                    logger.info(
                        "No existing commitment found, committing read credentials to chain"
                    )
                else:
                    logger.info("Commitment mismatch, updating read credentials on chain")
                self.commit_read_credentials()
            else:
                logger.info("Existing commitment matches current read credentials")

        except ValueError:
            logger.warning(
                f"Hotkey {self.wallet.hotkey.ss58_address} not found in metagraph, will commit when registered"
            )

        # Start periodic fetching
        self.start_commitment_fetcher()

    def _create_bucket_from_read_creds(self) -> Bucket:
        """Create a Bucket object from read credentials for comparison"""
        return Bucket(
            name=self.credentials.bucket_name[:32].ljust(32),
            account_id=self.credentials.account_id[:32].ljust(32),
            access_key_id=self.credentials.read_access_key_id[:32].ljust(32),
            secret_access_key=self.credentials.read_secret_access_key[:64].ljust(64),
        )

    def _commitments_match(self, bucket1: Bucket, bucket2: Bucket) -> bool:
        """Check if two bucket commitments match"""
        return (
            bucket1.name.strip() == bucket2.name.strip()
            and bucket1.access_key_id.strip() == bucket2.access_key_id.strip()
            and bucket1.secret_access_key.strip() == bucket2.secret_access_key.strip()
        )

    def commit_read_credentials(self) -> None:
        """Commit read credentials to the chain"""
        commitment = self.credentials.read_commitment

        if self.netuid is not None:
            try:
                self.subtensor.commit(self.wallet, self.netuid, commitment)
                logger.info(
                    f"Successfully committed read credentials to chain for hotkey {self.wallet.hotkey.ss58_address}"
                )
            except Exception as e:
                logger.error(f"Failed to commit read credentials: {e}")
                # Try to reinitialize substrate connection
                self.subtensor.substrate.close()
                self.subtensor.substrate.initialize()
                raise

    def start_commitment_fetcher(self) -> None:
        """Start background task to periodically fetch commitments"""
        if self._fetch_task is None:
            self._fetch_task = asyncio.create_task(self._fetch_commitments_periodically())
            logger.info(f"Started commitment fetcher with {self.fetch_interval}s interval")

    async def _fetch_commitments_periodically(self) -> None:
        """Background task to periodically fetch commitments"""
        while True:
            try:
                await asyncio.sleep(self.fetch_interval)

                # Sync metagraph
                await asyncio.to_thread(lambda: self.metagraph.sync(subtensor=self.subtensor))

                # Fetch commitments
                await self.fetch_commitments()

            except Exception as e:
                logger.error(f"Error in periodic commitment fetch: {e}")
                # Try to reinitialize substrate connection
                try:
                    self.subtensor.substrate.close()
                    self.subtensor.substrate.initialize()
                except Exception:
                    pass

    async def fetch_commitments(self) -> None:
        """Fetch all bucket commitments from the chain"""
        try:
            commitments = await self.get_commitments()
            if commitments:
                self.commitments = commitments
                logger.debug(f"Fetched {len(commitments)} commitments from chain")
            else:
                logger.warning("No commitments fetched from chain")
        except Exception as e:
            logger.error(f"Failed to fetch commitments: {e}")

    async def get_commitments(self, block: Optional[int] = None) -> dict[int, Bucket]:
        """
        Retrieve all bucket commitments from the chain.

        Args:
            block: Optional block number to query at

        Returns:
            Dictionary mapping UIDs to Bucket configurations
        """
        try:
            substrate = self.subtensor.substrate

            # Query commitments via substrate
            query_result = substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.netuid],
                block_hash=(None if block is None else substrate.get_block_hash(block)),
            )

            # Create hotkey to UID mapping
            hotkey_to_uid = dict(zip(self.metagraph.hotkeys, self.metagraph.uids))
            commitments = {}

            for key, value in query_result:
                try:
                    # Decode the commitment
                    decoded_ss58, commitment_str = self._decode_metadata(key, value.value)
                except Exception as e:
                    logger.error(f"Failed to decode metadata for key {key.value}: {e}")
                    continue

                # Skip if hotkey not in metagraph
                if decoded_ss58 not in hotkey_to_uid:
                    continue

                uid = hotkey_to_uid[decoded_ss58]

                # Validate commitment length (new format only)
                if len(commitment_str) != 128:
                    logger.error(
                        "Invalid commitment length for UID %s: %s",
                        uid,
                        len(commitment_str),
                    )
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
                    logger.debug(f"Retrieved bucket commitment for UID {uid}")

                except ValidationError as e:
                    logger.error(f"Failed to validate bucket for UID {uid}: {e}")
                    continue

            return commitments

        except Exception as e:
            logger.error(f"Error querying commitments from chain: {e}")
            # Try to reinitialize substrate connection
            self.subtensor.substrate.close()
            self.subtensor.substrate.initialize()
            return {}

    def _decode_metadata(self, encoded_ss58: tuple, metadata: dict) -> tuple[str, str]:
        """Decode metadata from chain query result"""
        # Decode the key into an SS58 address
        decoded_key = decode_account_id(encoded_ss58[0])

        # Get the commitment from the metadata
        commitment = metadata["info"]["fields"][0][0]
        bytes_tuple = commitment[next(iter(commitment.keys()))][0]

        return decoded_key, bytes(bytes_tuple).decode()

    def get_bucket(self, uid: int) -> Optional[Bucket]:
        """
        Get the bucket configuration for a given UID.

        Args:
            uid: The UID to retrieve the bucket for

        Returns:
            Bucket configuration or None if not found
        """
        return self.commitments.get(uid)

    def get_all_buckets(self) -> dict[int, Optional[Bucket]]:
        """
        Get all bucket configurations for all UIDs.

        Returns:
            Dictionary mapping UIDs to their bucket configurations
        """
        return {uid: self.get_bucket(uid) for uid in self.metagraph.uids}

    def get_bucket_for_hotkey(self, hotkey: str) -> Optional[Bucket]:
        """
        Get bucket configuration for a specific hotkey.

        Args:
            hotkey: SS58 address of the hotkey

        Returns:
            Bucket configuration or None if not found
        """
        try:
            uid = self.metagraph.hotkeys.index(hotkey)
            return self.get_bucket(uid)
        except ValueError:
            logger.warning(f"Hotkey {hotkey} not found in metagraph")
            return None

    def stop(self) -> None:
        """Stop the background fetcher task"""
        if self._fetch_task:
            self._fetch_task.cancel()
            self._fetch_task = None
            logger.info("Stopped commitment fetcher")
