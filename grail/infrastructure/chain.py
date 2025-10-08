"""GRAIL-specific chain manager for R2 credentials management."""

import asyncio
import logging
import multiprocessing
from typing import Any, Optional

import bittensor as bt

from ..shared.schemas import Bucket, BucketCredentials
from .chain_worker import chain_commitment_worker

logger = logging.getLogger(__name__)


class GrailChainManager:
    """GRAIL-specific chain manager for handling R2 credential commitments"""

    def __init__(
        self,
        config: Any,
        wallet: bt.wallet,
        metagraph: Any,
        subtensor: bt.subtensor,
        credentials: BucketCredentials,
        fetch_interval: int = 600,  # 10 minutes default
    ):
        """
        Initialize GRAIL chain manager.

        Args:
            config: Bittensor config object
            wallet: Wallet for signing commitments
            metagraph: Metagraph instance from caller
            subtensor: Async subtensor instance from caller
            credentials: Dual R2 credentials
            fetch_interval: Interval in seconds between fetching commitments
        """
        self.config = config
        self.netuid = config.netuid
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.credentials = credentials
        self.fetch_interval = fetch_interval

        # Commitment tracking
        self.commitments: dict[int, Bucket] = {}
        self._fetch_task: Optional[asyncio.Task] = None

        # Worker process for commitment fetching (avoids GIL contention)
        self._worker_queue: Optional[multiprocessing.Queue] = None
        self._worker_process: Optional[multiprocessing.Process] = None

        logger.info(f"Initialized GrailChainManager for netuid {self.netuid}")

    async def initialize(self) -> None:
        """Initialize chain manager and verify/commit credentials"""
        logger.info("Initializing GRAIL chain manager...")

        # Start worker process for periodic fetching
        self.start_commitment_worker()

        # Wait for first commitment fetch from worker (with timeout)
        logger.info("Waiting for initial commitments from worker...")
        await self._wait_for_initial_commitments(timeout=30.0)

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
                await self.commit_read_credentials()
            else:
                logger.info("Existing commitment matches current read credentials")

        except ValueError:
            logger.warning(
                f"Hotkey {self.wallet.hotkey.ss58_address} not found in "
                "metagraph, will commit when registered"
            )

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

    async def commit_read_credentials(self) -> None:
        """Commit read credentials to the chain using async subtensor"""
        commitment = self.credentials.read_commitment

        if self.netuid is not None:
            try:
                # Use async subtensor commit
                await self.subtensor.commit(self.wallet, self.netuid, commitment)
                logger.info(
                    "Successfully committed read credentials to chain "
                    f"for hotkey {self.wallet.hotkey.ss58_address}"
                )
            except Exception as e:
                logger.error(f"Failed to commit read credentials: {e}")
                raise

    def start_commitment_worker(self) -> None:
        """Start worker process for commitment fetching.

        Uses a separate process to completely isolate blockchain operations
        from the main event loop, avoiding any GIL contention issues.
        """
        if self._worker_process is not None:
            logger.warning("Worker process already started")
            return

        # Create queue for receiving commitment updates
        self._worker_queue = multiprocessing.Queue(maxsize=1)

        # Start worker process
        self._worker_process = multiprocessing.Process(
            target=chain_commitment_worker,
            args=(
                self._worker_queue,
                self.netuid,
                self.wallet.name,
                self.wallet.hotkey.ss58_address,
                self.fetch_interval,
            ),
            daemon=True,  # Process will terminate when main process exits
        )
        self._worker_process.start()

        # Start async task to poll queue
        self._fetch_task = asyncio.create_task(self._poll_worker_queue())

        logger.info(f"Started commitment worker process (PID: {self._worker_process.pid})")

    async def _wait_for_initial_commitments(self, timeout: float = 30.0) -> None:
        """Wait for first commitment payload from worker with timeout."""
        start_time = asyncio.get_event_loop().time()

        while True:
            if self._worker_queue is not None:
                try:
                    # Non-blocking get
                    commitments = self._worker_queue.get_nowait()
                    self.commitments = commitments
                    logger.info(f"Received initial {len(commitments)} commitments from worker")
                    return
                except Exception:
                    # Queue empty
                    pass

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"Timeout waiting for initial commitments after "
                    f"{timeout}s; continuing with empty set"
                )
                return

            await asyncio.sleep(0.5)

    async def _poll_worker_queue(self) -> None:
        """Poll the worker queue for commitment updates."""
        while True:
            try:
                # Check queue non-blockingly
                await asyncio.sleep(1)  # Poll every second

                if self._worker_queue is not None:
                    try:
                        # Non-blocking get
                        commitments = self._worker_queue.get_nowait()
                        self.commitments = commitments
                        logger.debug(f"Received {len(commitments)} commitments from worker")
                    except Exception:
                        # Queue empty, continue
                        pass

            except Exception as e:
                logger.error(f"Error polling worker queue: {e}")

    def get_bucket(self, uid: int) -> Optional[Bucket]:
        """
        Get the bucket configuration for a given UID.

        Args:
            uid: The UID to retrieve the bucket for

        Returns:
            Bucket configuration or None if not found
        """
        return self.commitments.get(uid)

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
        """Stop the background fetcher task and worker process"""
        if self._fetch_task:
            self._fetch_task.cancel()
            self._fetch_task = None

        if self._worker_process:
            self._worker_process.terminate()
            self._worker_process.join(timeout=5)
            if self._worker_process.is_alive():
                self._worker_process.kill()
            self._worker_process = None
            logger.info("Stopped commitment worker process")

        if self._worker_queue:
            self._worker_queue.close()
            self._worker_queue = None
