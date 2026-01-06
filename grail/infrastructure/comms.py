"""S3/R2 communication utilities for GRAIL."""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Any

import bittensor as bt
from aiobotocore.session import get_session
from botocore.config import Config
from datasets import Dataset
from huggingface_hub import HfFolder
from safetensors.torch import load_file, save_file
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM

from ..shared.constants import UPLOAD_TIMEOUT, WINDOW_LENGTH
from ..shared.schemas import Bucket, BucketCredentials

logger = logging.getLogger(__name__)

# Protocol version for dataset versioning
PROTOCOL_VERSION = "v1.0.0"

# --------------------------------------------------------------------------- #
#                   S3/R2 Configuration                                       #
# --------------------------------------------------------------------------- #

# Singleton aiobotocore session for connection pooling across all
# clients. Rationale: Creating a new session for each client creates
# isolated connection pools, leading to resource exhaustion when handling
# 100+ concurrent requests. Reusing a single session allows proper
# connection pooling (max_pool_connections applies across all clients)
# and reduces overhead from session initialization.
_AIOBOTO_SESSION: Any | None = None

# Client cache for reusing S3 clients across operations. During training/validation,
# we make hundreds of S3 operations per window with the same credentials. Reusing
# clients eliminates repeated setup overhead (credential validation, config init).
# Key: (credentials_hash, use_write) -> Value: aiobotocore client
_CLIENT_CACHE: dict[tuple[str, bool], Any] = {}
_CLIENT_CACHE_LOCK = asyncio.Lock()


def _get_aioboto_session() -> Any:
    """Get or create the singleton aiobotocore session.

    This session is reused across all S3 client creations to enable
    proper connection pooling and reduce resource consumption.
    Thread-safe for asyncio single-threaded execution model.

    Returns:
        Aiobotocore session instance
    """
    global _AIOBOTO_SESSION
    if _AIOBOTO_SESSION is None:
        _AIOBOTO_SESSION = get_session()
        logger.debug("Created singleton aiobotocore session for R2 pooling")
    return _AIOBOTO_SESSION


def _make_cache_key(
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
) -> tuple[str, bool]:
    """Create a cache key from credentials for client reuse.

    Args:
        credentials: Bucket credentials
        use_write: Whether write credentials are used

    Returns:
        Tuple of (credentials_hash, use_write) for cache lookup
    """
    # Create deterministic hash from credential components
    if credentials is not None:
        if isinstance(credentials, BucketCredentials):
            account_id = credentials.account_id
            if use_write:
                key_id = credentials.write_access_key_id
            else:
                key_id = credentials.read_access_key_id
        elif isinstance(credentials, Bucket):
            account_id = credentials.account_id.strip()
            key_id = credentials.access_key_id.strip()
        elif isinstance(credentials, dict):
            account_id = credentials.get("account_id", "").strip()
            key_id = credentials.get("access_key_id", "").strip()
    else:
        # Use environment variables for hash
        account_id = os.getenv("R2_ACCOUNT_ID", "")
        if use_write:
            key_id = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
        else:
            key_id = os.getenv("R2_READ_ACCESS_KEY_ID", "")

    # Create hash from credential identifiers (not secrets)
    cred_hash = hashlib.md5(f"{account_id}:{key_id}".encode()).hexdigest()
    return (cred_hash, use_write)


async def _get_cached_client(
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
) -> Any:
    """Get or create a cached S3 client for reuse across operations.

    This significantly reduces overhead during high-frequency operations by:
    - Eliminating repeated client setup (credential validation, config init)
    - Reusing connection pools via singleton session
    - Maintaining client state across operations

    Clients are cached per unique (credentials, use_write) combination.

    Args:
        credentials: Bucket credentials (BucketCredentials, Bucket, or dict)
        use_write: Whether to use write credentials

    Returns:
        Cached or newly created aiobotocore S3 client
    """
    cache_key = _make_cache_key(credentials, use_write)

    async with _CLIENT_CACHE_LOCK:
        if cache_key not in _CLIENT_CACHE:
            # Create new client and enter its context to initialize
            client_ctx = get_client_ctx(credentials, use_write)
            client = await client_ctx.__aenter__()
            _CLIENT_CACHE[cache_key] = (client, client_ctx)
            logger.debug(f"Created and cached S3 client for {cache_key[0][:8]}...")
        else:
            client, _client_ctx = _CLIENT_CACHE[cache_key]

    return client


async def clear_client_cache() -> None:
    """Clear all cached S3 clients and close their connections.

    Call this when:
    - Credentials have changed
    - Cleaning up at shutdown
    - Recovering from connection errors
    """
    async with _CLIENT_CACHE_LOCK:
        for cache_key, (_client, client_ctx) in _CLIENT_CACHE.items():
            try:
                await client_ctx.__aexit__(None, None, None)
                logger.debug(f"Closed cached client {cache_key[0][:8]}...")
            except Exception as e:
                logger.debug(f"Error closing cached client: {e}")

        _CLIENT_CACHE.clear()
        logger.info("Cleared S3 client cache")


def get_conf(key: str, default: str | None = None) -> str:
    """Get configuration from environment variables."""
    v = os.getenv(key)
    if not v and default is None:
        raise ValueError(f"{key} not set. Please set the environment variable.")
    return v or default or ""


def get_client_ctx(
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
) -> Any:
    """Create an S3 client for Cloudflare R2 or a compatible endpoint.

    Args:
        credentials: Either BucketCredentials, Bucket, or dict with credential fields.
                    If None, falls back to environment variables (backwards compatibility).
        use_write: If True and credentials is BucketCredentials, use write creds.
                  Ignored for Bucket/dict types.

    Environment overrides for hermetic/integration tests:
      - R2_ENDPOINT_URL: full endpoint (e.g., http://s3:9000 for MinIO)
      - R2_REGION: region name (default: us-east-1)
      - R2_FORCE_PATH_STYLE: "true" to force path-style addressing for MinIO
    """
    # Determine credentials to use
    if credentials is not None:
        if isinstance(credentials, BucketCredentials):
            # Same account and bucket, different access keys
            account_id = credentials.account_id
            if use_write:
                access_key = credentials.write_access_key_id
                secret_key = credentials.write_secret_access_key
            else:
                access_key = credentials.read_access_key_id
                secret_key = credentials.read_secret_access_key
        elif isinstance(credentials, Bucket):
            # Bucket objects are typically read credentials from chain
            account_id = credentials.account_id.strip()
            access_key = credentials.access_key_id.strip()
            secret_key = credentials.secret_access_key.strip()
            credentials.name.strip()
        elif isinstance(credentials, dict):
            # Handle dict format (from chain commitments or legacy)
            account_id = credentials.get("account_id", "").strip()
            access_key = credentials.get("access_key_id", "").strip()
            secret_key = credentials.get("secret_access_key", "").strip()
            credentials.get("name", credentials.get("bucket_name", "")).strip()
        else:
            raise ValueError(f"Unsupported credentials type: {type(credentials)}")
    else:
        # Fall back to environment variables (backwards compatibility)
        account_id = get_conf("R2_ACCOUNT_ID", None)
        if account_id:
            # Old single-credential mode
            access_key = get_conf("R2_WRITE_ACCESS_KEY_ID")
            secret_key = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
        else:
            # Try new dual-credential mode (same bucket/account, different keys)
            account_id = get_conf("R2_ACCOUNT_ID")
            if use_write:
                access_key = get_conf("R2_WRITE_ACCESS_KEY_ID")
                secret_key = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
            else:
                # Try read credentials first, fall back to write if not found
                try:
                    access_key = get_conf("R2_READ_ACCESS_KEY_ID")
                    secret_key = get_conf("R2_READ_SECRET_ACCESS_KEY")
                except ValueError:
                    # Fall back to write credentials for backwards compatibility
                    access_key = get_conf("R2_WRITE_ACCESS_KEY_ID")
                    secret_key = get_conf("R2_WRITE_SECRET_ACCESS_KEY")

    # Resolve endpoint
    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    if not endpoint_url:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    region_name = os.getenv("R2_REGION", "us-east-1")
    force_path_style = os.getenv("R2_FORCE_PATH_STYLE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    s3_config: dict[str, Any] = {}
    if force_path_style:
        s3_config["addressing_style"] = "path"

    # Configure transport-level timeouts and modest retries to avoid
    # indefinite hangs. Set these higher than our asyncio timeout (30s)
    # to let asyncio.wait_for handle the timeout cleanly.
    # Root cause: botocore and asyncio timeouts conflict when stacked.
    config = Config(
        connect_timeout=3,
        read_timeout=30,  # Higher than asyncio timeout
        retries={"max_attempts": 2, "mode": "standard"},
        max_pool_connections=256,
        region_name=region_name,
        signature_version="s3v4",
        s3=s3_config or None,
    )

    # Use singleton session for proper connection pooling across all clients
    return _get_aioboto_session().create_client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=config,
    )


def get_bucket_id(
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
) -> str:
    """Get the bucket ID from credentials or environment.

    Note: Since we use the same bucket for both read and write operations,
    the use_write parameter doesn't affect the bucket ID, only the access keys.
    """
    if credentials is not None:
        if isinstance(credentials, BucketCredentials):
            return str(credentials.bucket_name)
        elif isinstance(credentials, Bucket):
            return str(credentials.name).strip()
        elif isinstance(credentials, dict):
            name_raw = credentials.get("name", credentials.get("bucket_name", ""))
            name = str(name_raw).strip()
            if name:
                return name

    # Fall back to environment variable (same bucket for read and write)
    bucket_id = get_conf("R2_BUCKET_ID")
    if bucket_id is None:
        raise ValueError("R2_BUCKET_ID environment variable not set")
    return bucket_id


# --------------------------------------------------------------------------- #
#                   Progress Tracking                                         #
# --------------------------------------------------------------------------- #


class TransferProgress:
    """Track upload/download progress and speed with tqdm-style visualization"""

    def __init__(self, total_size: int, operation: str):
        self.total_size = total_size
        self.operation = operation
        self.transferred = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_transferred = 0

        # Create tqdm progress bar that outputs to logger
        # Use file=None and disable=True to prevent TTY output
        self.pbar = tqdm(
            total=total_size,
            desc=f"📊 {operation}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=True,  # Don't output to stderr
            file=None,
        )

    def update(self, bytes_transferred: int) -> None:
        self.transferred += bytes_transferred
        self.pbar.update(bytes_transferred)
        now = time.time()

        # Log progress every 2 seconds or on completion
        if now - self.last_log_time >= 2.0 or self.transferred >= self.total_size:
            elapsed = now - self.start_time
            speed_mbps = (self.transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            progress_pct = (self.transferred / self.total_size) * 100 if self.total_size > 0 else 0

            # Calculate ETA
            if speed_mbps > 0 and self.transferred < self.total_size:
                remaining_mb = (self.total_size - self.transferred) / (1024 * 1024)
                eta_seconds = remaining_mb / speed_mbps
                eta_str = self._format_time(eta_seconds)
            else:
                eta_str = "00:00"

            # Create a visual progress bar for logs
            bar_length = 30
            filled_length = (
                int(bar_length * self.transferred // self.total_size) if self.total_size > 0 else 0
            )
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            # Format sizes
            transferred_str = self._format_bytes(self.transferred)
            total_str = self._format_bytes(self.total_size)

            logger.info(
                f"📊 {self.operation}: |{bar}| "
                f"{progress_pct:.1f}% [{transferred_str}/{total_str}] "
                f"@ {speed_mbps:.2f} MB/s ETA: {eta_str}"
            )
            self.last_log_time = now
            self.last_transferred = self.transferred

    def close(self) -> None:
        """Close the progress bar and log final stats"""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        speed_mbps = (self.transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        transferred_str = self._format_bytes(self.transferred)
        logger.info(
            f"✅ {self.operation} completed: {transferred_str} "
            f"in {self._format_time(elapsed)} @ {speed_mbps:.2f} MB/s"
        )

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        """Format bytes to human-readable string"""
        size = float(num_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to MM:SS"""
        mins, secs = divmod(int(seconds), 60)
        return f"{mins:02d}:{secs:02d}"


# --------------------------------------------------------------------------- #
#                   Upload Functions                                          #
# --------------------------------------------------------------------------- #


async def upload_file_chunked(
    key: str,
    data: bytes,
    chunk_size: int | None = None,
    max_retries: int = 3,
    compress: bool = False,
    credentials: BucketCredentials | None = None,
    use_write: bool = False,
    upload_timeout: float = UPLOAD_TIMEOUT,
) -> bool:
    """Upload file in chunks optimized for H100 high-bandwidth - 100MB chunks.

    Args:
        key: S3 object key
        data: File data to upload
        chunk_size: Size of each chunk (adaptive if None)
        max_retries: Max retry attempts per chunk (default: 3)
        compress: Whether to compress JSON files (default: False, disabled due to
            negligible bandwidth savings and download path complexity)
        credentials: R2 credentials
        use_write: Whether to use write credentials
        upload_timeout: Timeout in seconds per chunk upload (default: 600s/10min)

    Returns:
        True if upload succeeded, False otherwise
    """

    # NOTE: JSON gzip compression disabled by default. The bandwidth savings are
    # negligible (<0.01% of total upload for typical checkpoints) and it adds
    # complexity to download paths (handling .gz vs non-.gz filenames).
    # Model weights use zstandard compression which provides the real savings.
    if compress and key.endswith(".json") and len(data) < 10 * 1024 * 1024:
        original_size = len(data)
        data = gzip.compress(data, compresslevel=6)
        key = key + ".gz"
        logger.info(
            f"🗜️ Compressed {original_size} → {len(data)} bytes ({100 * (1 - len(data) / original_size):.1f}% reduction)"
        )

    total_size = len(data)
    # Adaptive chunk sizing based on total size
    if chunk_size is None:
        if total_size < 50 * 1024 * 1024:
            chunk_size = total_size
        elif total_size < 500 * 1024 * 1024:
            chunk_size = 50 * 1024 * 1024
        elif total_size < 5 * 1024 * 1024 * 1024:
            chunk_size = 500 * 1024 * 1024
        else:
            chunk_size = 500 * 1024 * 1024
    progress = TransferProgress(total_size, f"Upload {key}")

    # For small files, use single upload
    if total_size <= chunk_size:
        logger.info(f"📤 Uploading {key} ({total_size} bytes)")
        return await _upload_single_chunk(
            key, data, progress, max_retries, credentials, use_write, upload_timeout
        )

    # For large files, use multipart upload
    logger.info(
        f"📤 Starting chunked upload of {key} ({total_size} bytes, {(total_size + chunk_size - 1) // chunk_size} chunks)"
    )

    client = None
    upload_id = None
    try:
        client = await _get_cached_client(credentials, use_write)

        # Initiate multipart upload
        bucket_id = get_bucket_id(credentials, use_write)
        response = await client.create_multipart_upload(Bucket=bucket_id, Key=key)
        upload_id = response["UploadId"]

        # Upload chunks concurrently with limited concurrency (fewer for larger chunks)
        semaphore = asyncio.Semaphore(3 if chunk_size >= 200 * 1024 * 1024 else 15)
        tasks = []

        for i in range(0, total_size, chunk_size):
            chunk_data = data[i : i + chunk_size]
            part_number = (i // chunk_size) + 1
            task = _upload_chunk_with_semaphore(
                semaphore,
                client,
                key,
                upload_id,
                part_number,
                chunk_data,
                progress,
                max_retries,
                credentials,
                use_write,
                upload_timeout,
            )
            tasks.append(task)

        # Wait for all chunks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        parts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {i + 1} failed: {result}")
                bucket_id = get_bucket_id(credentials, use_write)
                try:
                    await client.abort_multipart_upload(
                        Bucket=bucket_id, Key=key, UploadId=upload_id
                    )
                except Exception as e:
                    logger.debug(f"Failed to abort upload: {e}")
                return False
            parts.append(result)

        # Complete multipart upload
        bucket_id = get_bucket_id(credentials, use_write)
        await client.complete_multipart_upload(
            Bucket=bucket_id,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        elapsed = time.time() - progress.start_time
        speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        logger.info(
            f"✅ Upload completed: {key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s"
        )
        return True

    except asyncio.TimeoutError as e:
        logger.error(f"❌ Upload timeout for {key} after {upload_timeout}s: {e}")
        if client is not None and upload_id is not None:
            try:
                bucket_id = get_bucket_id(credentials, use_write)
                await client.abort_multipart_upload(Bucket=bucket_id, Key=key, UploadId=upload_id)
            except Exception as cleanup_err:
                logger.debug(f"Failed to cleanup after timeout: {cleanup_err}")
        return False
    except Exception as e:
        logger.error(f"❌ Upload failed for {key}: {e}")
        if client is not None and upload_id is not None:
            try:
                bucket_id = get_bucket_id(credentials, use_write)
                await client.abort_multipart_upload(Bucket=bucket_id, Key=key, UploadId=upload_id)
            except Exception as cleanup_err:
                logger.debug(f"Failed to cleanup after error: {cleanup_err}")
        return False


def _is_throttling_error(error: Exception) -> bool:
    """Detect if error is R2/S3 throttling (SlowDown, 429, RequestLimitExceeded)."""
    error_str = str(error).lower()
    return any(code in error_str for code in ["slowdown", "429", "requestlimitexceeded", "throttl"])


async def _exponential_backoff(
    attempt: int,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 32.0,
    is_throttle: bool = False,
) -> float:
    """Calculate exponential backoff with jitter for retries.

    Args:
        attempt: Current attempt number (0-indexed)
        max_retries: Total retry attempts
        base_delay: Base delay in seconds (default 1s)
        max_delay: Maximum delay in seconds (default 32s)
        is_throttle: If True, use longer delays for throttling

    Returns:
        Delay in seconds before next retry
    """
    import random

    # For throttling, use longer base delay and cap
    if is_throttle:
        base_delay = 2.0
        max_delay = 60.0

    # Exponential backoff: 2^attempt * base_delay
    delay = min(base_delay * (2**attempt), max_delay)

    # Add jitter: ±25% of delay to avoid thundering herd
    jitter = delay * 0.25 * (2 * random.random() - 1)
    final_delay = max(0.1, delay + jitter)

    return final_delay


async def _upload_single_chunk(
    key: str,
    data: bytes,
    progress: TransferProgress,
    max_retries: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
    upload_timeout: float = UPLOAD_TIMEOUT,
) -> bool:
    """Upload single chunk with retry logic and timeout protection"""

    for attempt in range(max_retries):
        try:
            client = await _get_cached_client(credentials, use_write)
            bucket_id = get_bucket_id(credentials, use_write)

            # Wrap upload with timeout
            await asyncio.wait_for(
                client.put_object(Bucket=bucket_id, Key=key, Body=data),
                timeout=upload_timeout,
            )
            progress.update(len(data))
            return True
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                wait_time = await _exponential_backoff(attempt, max_retries)
                logger.warning(
                    f"Upload attempt {attempt + 1} timeout for {key} (waited {upload_timeout}s), "
                    f"retrying in {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    f"Upload timeout for {key} after {max_retries} attempts "
                    f"({upload_timeout}s per attempt)"
                )
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                is_throttle = _is_throttling_error(e)
                wait_time = await _exponential_backoff(
                    attempt, max_retries, is_throttle=is_throttle
                )
                logger.warning(
                    f"Upload attempt {attempt + 1} failed for {key}, retrying in {wait_time:.1f}s "
                    f"({'throttled' if is_throttle else 'error'}): {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Upload failed after {max_retries} attempts for {key}: {e}")
                return False
    return False


async def _upload_chunk_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: Any,
    key: str,
    upload_id: str,
    part_number: int,
    data: bytes,
    progress: TransferProgress,
    max_retries: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
    upload_timeout: float = UPLOAD_TIMEOUT,
) -> dict[str, Any] | None:
    """Upload a single chunk with concurrency control, retry logic, and timeout protection"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                bucket_id = get_bucket_id(credentials, use_write)

                # Wrap upload_part with timeout to catch stalled transfers
                response = await asyncio.wait_for(
                    client.upload_part(
                        Bucket=bucket_id,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data,
                    ),
                    timeout=upload_timeout,
                )
                progress.update(len(data))
                return {"PartNumber": part_number, "ETag": response["ETag"]}
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = await _exponential_backoff(attempt, max_retries)
                    logger.warning(
                        f"Chunk {part_number} attempt {attempt + 1} timeout "
                        f"(waited {upload_timeout}s), retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Chunk {part_number} timeout after {max_retries} attempts "
                        f"({upload_timeout}s per attempt)"
                    )
                    raise TimeoutError(
                        f"Chunk {part_number} upload timeout after {upload_timeout}s"
                    ) from None
            except Exception as e:
                if attempt < max_retries - 1:
                    is_throttle = _is_throttling_error(e)
                    wait_time = await _exponential_backoff(
                        attempt, max_retries, is_throttle=is_throttle
                    )
                    logger.warning(
                        f"Chunk {part_number} attempt {attempt + 1} failed, retrying in {wait_time:.1f}s "
                        f"({'throttled' if is_throttle else 'error'}): {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    return None


# --------------------------------------------------------------------------- #
#                   Download Functions                                        #
# --------------------------------------------------------------------------- #


async def download_file_chunked(
    key: str,
    max_retries: int = 3,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
    chunk_size: int | None = None,
) -> bytes | None:
    """Download file in chunks with automatic decompression"""
    actual_key = key
    is_compressed = False

    for attempt in range(max_retries):
        try:
            client = await _get_cached_client(credentials, use_write)

            # Try compressed version first if not already .gz
            if not key.endswith(".gz"):
                try:
                    compressed_key = key + ".gz"
                    bucket_id = get_bucket_id(credentials, use_write)
                    head_response = await client.head_object(Bucket=bucket_id, Key=compressed_key)
                    actual_key = compressed_key
                    is_compressed = True
                    logger.debug(f"Found compressed version: {compressed_key}")
                except Exception:
                    # Fallback to uncompressed
                    bucket_id = get_bucket_id(credentials, use_write)
                    head_response = await client.head_object(Bucket=bucket_id, Key=key)
                    actual_key = key
            else:
                bucket_id = get_bucket_id(credentials, use_write)
                head_response = await client.head_object(Bucket=bucket_id, Key=key)
                is_compressed = key.endswith(".gz")
            total_size = head_response["ContentLength"]

            logger.debug(
                f"📥 Downloading {actual_key} ({total_size} bytes){' (compressed)' if is_compressed else ''}"
            )
            progress = TransferProgress(total_size, f"Download {actual_key}")

            # For small files, download in one go
            if chunk_size is None:
                if total_size < 50 * 1024 * 1024:
                    chunk_size = total_size
                elif total_size < 500 * 1024 * 1024:
                    chunk_size = 50 * 1024 * 1024
                elif total_size < 5 * 1024 * 1024 * 1024:
                    chunk_size = 200 * 1024 * 1024
                else:
                    chunk_size = 500 * 1024 * 1024

            if total_size <= chunk_size:
                bucket_id = get_bucket_id(credentials, use_write)
                response = await client.get_object(Bucket=bucket_id, Key=actual_key)
                try:
                    payload: bytes = await response["Body"].read()
                    progress.update(len(payload))
                    progress.close()  # Log final stats

                    # Decompress if needed
                    if is_compressed:
                        payload = gzip.decompress(payload)
                        logger.debug(f"🗜️ Decompressed to {len(payload)} bytes")
                    return payload
                finally:
                    # Close response body
                    response["Body"].close()

            # For large files, download in chunks
            chunks: list[bytes] = []
            assert chunk_size is not None, "chunk_size must be set for chunked downloads"
            semaphore = asyncio.Semaphore(10 if chunk_size >= 200 * 1024 * 1024 else 30)
            tasks = []

            for start in range(0, total_size, chunk_size):
                end = min(start + chunk_size - 1, total_size - 1)
                task = _download_chunk_with_semaphore(
                    semaphore,
                    client,
                    actual_key,
                    start,
                    end,
                    progress,
                    max_retries,
                    credentials,
                    use_write,
                )
                tasks.append(task)

            # Wait for all chunks
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures and reassemble
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"Download chunk {i} failed: {result}")
                    raise result
                if not isinstance(result, (bytes, bytearray)):
                    raise TypeError("Invalid chunk type")
                chunks.append(bytes(result))

            assembled: bytes = b"".join(chunks)
            progress.close()  # Log final stats

            # Decompress if needed
            if is_compressed:
                assembled = gzip.decompress(assembled)
                logger.debug(f"🗜️ Decompressed to {len(assembled)} bytes")
            return assembled

        except Exception as e:
            # Don't retry for 404 errors - file simply doesn't exist
            is_not_found = "404" in str(e) or "Not Found" in str(e)
            if is_not_found:
                logger.debug(f"File not found: {key}")
                return None

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"Download attempt {attempt + 1} failed for {key}, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts for {key}: {e}")

    # If all retries failed, return None
    return None


async def _download_chunk_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: Any,
    key: str,
    start: int,
    end: int,
    progress: TransferProgress,
    max_retries: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> bytes | None:
    """Download a single chunk with concurrency control and retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            response = None
            try:
                bucket_id = get_bucket_id(credentials, use_write)
                response = await client.get_object(
                    Bucket=bucket_id, Key=key, Range=f"bytes={start}-{end}"
                )
                payload: bytes = await response["Body"].read()
                progress.update(len(payload))
                return payload
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Download chunk {start}-{end} attempt {attempt + 1} "
                        f"failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise e
            finally:
                # Explicitly close the response body to avoid unclosed
                # connection warnings
                if response is not None and "Body" in response:
                    try:
                        response["Body"].close()
                    except Exception:
                        pass  # Best effort cleanup
    return None


# --------------------------------------------------------------------------- #
#                   File Operations                                           #
# --------------------------------------------------------------------------- #


async def file_exists(
    key: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
    max_upload_time: float | None = None,
) -> bool:
    """Check if a file exists (compressed or uncompressed) with optional deadline check.

    This is a convenience wrapper around file_exists_with_deadline that returns
    a simple boolean for backward compatibility with existing code.

    Args:
        key: S3 object key to check
        credentials: R2/S3 credentials
        use_write: Whether to use write credentials
        max_upload_time: If provided, reject files uploaded after this timestamp

    Returns:
        True if file exists and meets upload time constraint (if specified)
    """
    exists, _was_late, _too_small, _upload_time = await file_exists_with_deadline(
        key=key,
        credentials=credentials,
        use_write=use_write,
        max_upload_time=max_upload_time,
    )
    return exists


async def file_exists_with_deadline(
    key: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
    max_upload_time: float | None = None,
    min_size_bytes: int = 0,
) -> tuple[bool, bool, bool, float | None]:
    """Check existence, size, and whether the object violates an upload deadline.

    Returns
    -------
    (exists_and_valid, was_late, too_small, upload_time)
        exists_and_valid: True if object exists, size >= min_size_bytes, and upload_time <= max_upload_time (when provided)
        was_late: True if object exists and is large enough but upload_time exceeded max_upload_time
        too_small: True if object exists but size < min_size_bytes
        upload_time: Float unix timestamp of LastModified if available

    Args:
        key: S3 object key to check
        credentials: R2/S3 credentials
        use_write: Whether to use write credentials
        max_upload_time: If provided, reject files uploaded after this timestamp
        min_size_bytes: Minimum file size in bytes (default: 0, no size check)
    """
    try:
        client = await _get_cached_client(credentials, use_write)
        bucket_id = get_bucket_id(credentials, use_write)

        # Try both compressed and uncompressed versions
        checked_keys = []
        if not key.endswith(".gz"):
            checked_keys.append(key + ".gz")
            checked_keys.append(key)  # Also check uncompressed
        else:
            checked_keys.append(key)
            checked_keys.append(key[:-3])  # Remove .gz and check uncompressed

        for candidate in checked_keys:
            try:
                response = await client.head_object(Bucket=bucket_id, Key=candidate)
                logger.debug(f"file_exists_with_deadline: Found {candidate}")
            except Exception:
                continue

            # Check size requirement
            if min_size_bytes > 0:
                size = response.get("ContentLength")
                if size is None or size < min_size_bytes:
                    logger.debug(
                        f"file_exists_with_deadline: {candidate} too small ({size} < {min_size_bytes})"
                    )
                    return False, False, True, None

            last_modified = response.get("LastModified")
            upload_time = float(last_modified.timestamp()) if last_modified else None
            if max_upload_time is not None and upload_time is not None:
                if upload_time > float(max_upload_time):
                    return False, True, False, upload_time
            return True, False, False, upload_time

        return False, False, False, None
    except Exception as e:
        logger.debug(
            "file_exists_with_deadline failed for key=%s: %s %s", key, type(e).__name__, str(e)
        )
        return False, False, False, None


async def list_bucket_files(
    prefix: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> list[str]:
    """List files in bucket with given prefix.

    Handles pagination to return all files even when >1000 exist.
    """
    try:
        client = await _get_cached_client(credentials, use_write)
        bucket_id = get_bucket_id(credentials, use_write)

        all_keys: list[str] = []
        continuation_token: str | None = None

        while True:
            # Build request parameters
            params: dict[str, Any] = {"Bucket": bucket_id, "Prefix": prefix}
            if continuation_token:
                params["ContinuationToken"] = continuation_token

            response = await client.list_objects_v2(**params)

            if "Contents" in response:
                all_keys.extend(obj["Key"] for obj in response["Contents"])

            # Check if there are more results
            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
                if not continuation_token:
                    break  # Safety: shouldn't happen but avoid infinite loop
            else:
                break

        return all_keys
    except Exception:
        logger.error("Failed to list bucket files with prefix %s", prefix, exc_info=True)
        return []


async def get_file_size(
    key: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> int | None:
    """Get the size of a file on R2. Returns None if file doesn't exist or on error."""
    try:
        client = await _get_cached_client(credentials, use_write)
        bucket_id = get_bucket_id(credentials, use_write)

        # Try exact key first
        try:
            response = await client.head_object(Bucket=bucket_id, Key=key)
            size = response.get("ContentLength")
            return int(size) if isinstance(size, int) else None
        except Exception:
            # Try compressed version if key doesn't end with .gz
            if not key.endswith(".gz"):
                try:
                    compressed_key = key + ".gz"
                    response = await client.head_object(Bucket=bucket_id, Key=compressed_key)
                    size2 = response.get("ContentLength")
                    return int(size2) if isinstance(size2, int) else None
                except Exception:
                    pass
        return None
    except Exception as e:
        logger.debug(f"Failed to get file size for {key}: {e}")
        return None


async def delete_prefix(
    prefix: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = True,
) -> None:
    """Delete all objects under a prefix.

    Designed for cleanup of checkpoint directories. Uses batched delete
    operations to minimize round trips.
    """
    continuation_token: str | None = None
    while True:
        try:
            client = await _get_cached_client(credentials, use_write)
            bucket_id = get_bucket_id(credentials, use_write)
            list_kwargs: dict[str, Any] = {"Bucket": bucket_id, "Prefix": prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = await client.list_objects_v2(**list_kwargs)
            objects = response.get("Contents", [])
            if objects:
                delete_payload = {"Objects": [{"Key": obj["Key"]} for obj in objects]}
                await client.delete_objects(Bucket=bucket_id, Delete=delete_payload)

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")
        except Exception:
            logger.error("Failed to delete prefix %s", prefix, exc_info=True)
            break


async def get_file(
    key: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> dict[str, Any] | None:
    """Download and parse JSON file with improved error handling"""
    try:
        data = await download_file_chunked(key, credentials=credentials, use_write=use_write)
        if data:
            loaded: Any = json.loads(data.decode())
            if isinstance(loaded, dict):
                return loaded
            else:
                logger.debug(f"Expected dict JSON in {key}, found {type(loaded).__name__}")
                return None
        return None
    except Exception as e:
        logger.debug(f"Failed to get file {key}: {e}")
        return None


async def get_parquet_file(
    key: str,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> dict[str, Any] | None:
    """Download and parse Parquet file containing window data.

    Downloads a Parquet file from S3/R2 and deserializes it to a window
    dictionary with inferences list.

    Args:
        key: S3/R2 key for the Parquet file
        credentials: Bucket credentials for authentication
        use_write: Whether to use write credentials (typically False for reads)

    Returns:
        Window data dictionary or None if download/parse fails
    """
    from .parquet_io import ParquetError, deserialize_parquet_to_window

    try:
        data = await download_file_chunked(key, credentials=credentials, use_write=use_write)
        if data:
            return deserialize_parquet_to_window(data)
        return None
    except ParquetError as e:
        logger.warning(f"Corrupt Parquet file {key}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Failed to get parquet file {key}: {e}")
        return None


# --------------------------------------------------------------------------- #
#                   GRAIL-specific Storage Functions                          #
# --------------------------------------------------------------------------- #


async def sink_window_inferences(
    wallet: bt.wallet,
    window_start: int,
    inferences: list[dict],
    credentials: BucketCredentials | None = None,
) -> None:
    """Upload window of inferences to S3 in Parquet format.

    Uses Apache Parquet for efficient columnar storage with snappy compression,
    providing better compression ratios and faster serialization than JSON.
    """
    from .parquet_io import serialize_window_to_parquet

    key = f"grail/windows/{wallet.hotkey.ss58_address}-window-{window_start}.parquet"

    # Pack all inferences into window data
    window_data = {
        "wallet": wallet.hotkey.ss58_address,
        "window_start": window_start,
        "window_length": WINDOW_LENGTH,
        "inference_count": len(inferences),
        "inferences": inferences,
        "timestamp": time.time(),
    }

    body = serialize_window_to_parquet(window_data)
    logger.debug(f"[SINK] window={window_start} count={len(inferences)} → key={key}")

    success = await upload_file_chunked(key, body, credentials=credentials, use_write=True)
    if success:
        logger.info(
            f"📤 Uploaded window data for window {window_start} ({len(inferences)} inferences)"
        )
    else:
        logger.error(f"❌ Failed to upload window data for window {window_start}")


# TODO(v2): Re-enable model state management for training
async def save_model_state(
    model: AutoModelForCausalLM,
    hotkey: str,
    window: int,
    credentials: BucketCredentials | None = None,
) -> bool:
    # Save model state as safetensors to S3 with chunked upload and progress logging
    key = f"grail/models/{hotkey}-{window}.safetensors"

    # Create temporary file for safetensors
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        logger.info(f"💾 Preparing model state for {hotkey} window {window}")
        # Save to temporary file
        save_file(model.state_dict(), temp_path)  # type: ignore[attr-defined]

        # Read file content as bytes
        with open(temp_path, "rb") as f:
            body = f.read()

        file_size_mb = len(body) / (1024 * 1024)
        logger.info(f"📦 Model state prepared: {file_size_mb:.1f} MB")

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    logger.debug(f"[MODEL] Saving model state for {hotkey} window {window} → {key}")

    # Use chunked upload with retry logic
    success = await upload_file_chunked(key, body, credentials=credentials, use_write=True)

    if success:
        logger.info(f"✅ Successfully uploaded model state for window {window}")
    else:
        logger.error(f"❌ Failed to upload model state for window {window}")

    return success


async def load_model_state(
    model: AutoModelForCausalLM,
    hotkey: str,
    window: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> bool:
    """Load model state from S3 with chunked download and progress logging"""
    key = f"grail/models/{hotkey}-{window}.safetensors"

    logger.info(f"🔍 Loading model state for {hotkey} window {window}")

    # Use chunked download with retry logic
    data = await download_file_chunked(key, credentials=credentials, use_write=use_write)

    if data is None:
        logger.debug(f"Model state not found for {key}")
        return False

    try:
        # Load safetensors from bytes using temporary file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(data)

        try:
            # Load from temporary file
            state_dict = load_file(temp_path)
            model.load_state_dict(state_dict)  # type: ignore[attr-defined]

            file_size_mb = len(data) / (1024 * 1024)
            logger.info(
                f"✅ Successfully loaded model state for window {window} ({file_size_mb:.1f} MB)"
            )
            return True

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"❌ Failed to load model state for window {window}: {e}")
        return False


async def model_state_exists(
    hotkey: str,
    window: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> bool:
    # Check if model state exists for given hotkey and window
    key = f"grail/models/{hotkey}-{window}.safetensors"
    return await file_exists(key, credentials=credentials, use_write=use_write)


async def upload_valid_rollouts(
    window: int,
    valid_rollouts: list[dict],
    credentials: BucketCredentials | None = None,
) -> bool:
    """Upload validated SAT rollouts for training with chunked upload and progress logging"""
    key = f"grail/valid_rollouts/{window}.json"

    data = {
        "window": window,
        "count": len(valid_rollouts),
        "rollouts": valid_rollouts,
        "timestamp": time.time(),
    }

    body = json.dumps(data).encode()
    logger.debug(f"[VALID] Uploading {len(valid_rollouts)} valid rollouts for window {window}")

    success = await upload_file_chunked(key, body, credentials=credentials, use_write=True)

    if success:
        logger.info(f"📤 Uploaded {len(valid_rollouts)} valid rollouts for window {window}")
    else:
        logger.error(f"❌ Failed to upload valid rollouts for window {window}")

    return success


async def get_valid_rollouts(
    window: int,
    credentials: BucketCredentials | Bucket | dict | None = None,
    use_write: bool = False,
) -> list[dict]:
    """
    Download valid SAT rollouts for training.

    These rollouts have already been:
    - Verified by validators using verify_rollout()
    - Confirmed to have valid GRAIL proofs (model identity verified)
    - Checked for SAT problem correctness and solution validity

    The trainer can safely use these for GRPO training.
    """
    key = f"grail/valid_rollouts/{window}.json"

    try:
        data = await get_file(key, credentials=credentials, use_write=use_write)
        if data and "rollouts" in data:
            logger.info(f"Downloaded {len(data['rollouts'])} verified rollouts for window {window}")
            return data["rollouts"]  # type: ignore[no-any-return]
        # Backward compatibility: check old format
        elif data and "inferences" in data:
            logger.info(
                f"Downloaded {len(data['inferences'])} verified rollouts (legacy format) for window {window}"
            )
            return data["inferences"]  # type: ignore[no-any-return]
        return []
    except Exception:
        logger.debug("No valid rollouts found for window %s", window)
        return []


# --------------------------------------------------------------------------- #
#                   Hugging Face Dataset Upload                               #
# --------------------------------------------------------------------------- #


def login_huggingface() -> bool:
    """
    Login to Hugging Face using token from environment or cache.
    This should be called once at startup.
    """
    try:
        from huggingface_hub import HfFolder, login

        # Check if already logged in
        existing_token = HfFolder.get_token()
        if existing_token:
            logger.info("Already logged into Hugging Face")
            return True

        # Try to get token from environment using get_conf pattern
        token = get_conf("HF_TOKEN", None)
        if token:
            login(token=token, add_to_git_credential=False)
            logger.info("✅ Successfully logged into Hugging Face")
            return True
        else:
            logger.info("No HF_TOKEN found. Set HF_TOKEN to enable dataset uploads.")
            logger.info("Get your token at: https://huggingface.co/settings/tokens")
            return False

    except Exception as e:
        logger.warning(f"Failed to login to Hugging Face: {e}")
        return False


async def upload_to_huggingface(
    rollouts: list[dict], window: int, version: str | None = None
) -> bool:
    """
    Upload rollouts to unified Hugging Face dataset.

    Dataset: grail/sat-rollouts (single dataset for all windows)
    Each rollout is versioned and includes window metadata.

    Args:
        rollouts: List of validated rollout dictionaries
        window: Window number for temporal tracking
        version: Protocol version (defaults to PROTOCOL_VERSION)
    """
    if not rollouts:
        logger.debug("No rollouts to upload to Hugging Face")
        return False

    if version is None:
        version = PROTOCOL_VERSION

    # Use the HF username from environment or default to a common one
    hf_username = get_conf("HF_USERNAME", "fatheroffire")
    dataset_name = f"{hf_username}/grail-sat-rollouts"

    try:
        # Prepare rollouts with metadata
        processed_rollouts = []
        for rollout in rollouts:
            # Generate unique ID for each rollout
            rollout_id = hashlib.sha256(
                f"{rollout.get('hotkey', '')}_{window}_{rollout.get('nonce', '')}".encode()
            ).hexdigest()[:16]

            # Extract key information
            commit = rollout.get("commit", {})
            sat_problem = commit.get("sat_problem", {})
            rollout_data = commit.get("rollout", {})
            proof = rollout.get("proof", {})

            # Create flattened structure for dataset
            processed_rollout = {
                "id": rollout_id,
                "version": version,
                "window": window,
                "timestamp": rollout.get("timestamp", time.time()),
                "uploaded_at": datetime.utcnow().isoformat(),
                # Miner info
                "miner": rollout.get("hotkey", ""),
                "nonce": rollout.get("nonce", 0),
                # SAT problem
                "sat_seed": sat_problem.get("seed", ""),
                "sat_num_vars": sat_problem.get("num_vars", 0),
                "sat_num_clauses": len(sat_problem.get("clauses", [])),
                "sat_difficulty": sat_problem.get("difficulty", 0.5),
                "sat_clauses": json.dumps(sat_problem.get("clauses", [])),  # Store as JSON string
                # Solution
                "solution_success": rollout_data.get("success", False),
                "solution_assignment": json.dumps(rollout_data.get("assignment", [])),
                "solution_trajectory": json.dumps(rollout_data.get("trajectory", [])),
                "solution_satisfied_clauses": rollout_data.get("satisfied_clauses", 0),
                "solution_total_reward": rollout_data.get("total_reward", 0.0),
                # GRAIL proof (store as JSON strings for complex fields)
                "grail_tokens": json.dumps(commit.get("tokens", [])),
                "grail_s_vals": json.dumps(commit.get("s_vals", [])),
                "grail_signature": commit.get("signature", ""),
                "grail_beacon": json.dumps(commit.get("beacon", {})),
                "grail_indices": json.dumps(proof.get("indices", [])),
                # Metrics
                "token_count": len(commit.get("tokens", [])),
                "inference_count": rollout.get("inference_count", 1),
            }

            processed_rollouts.append(processed_rollout)

        # Create dataset from rollouts
        dataset = Dataset.from_list(processed_rollouts)

        # Check if we're logged in
        token = HfFolder.get_token()
        if not token:
            # Try to login if not already
            if not login_huggingface():
                logger.debug("Cannot upload to Hugging Face without login")
                return False
            token = HfFolder.get_token()

        # Initialize Hugging Face API
        from huggingface_hub import create_repo, repo_exists

        # Check if repository exists before trying to create it
        try:
            repo_exists_flag = repo_exists(repo_id=dataset_name, repo_type="dataset", token=token)
            if not repo_exists_flag:
                # Only create if it doesn't exist
                logger.debug(f"Creating new dataset repository: {dataset_name}")
                create_repo(
                    repo_id=dataset_name,
                    token=token,
                    private=False,
                    repo_type="dataset",
                )
                logger.info(f"✅ Created new dataset repository: {dataset_name}")
            else:
                logger.debug(f"Dataset repository {dataset_name} already exists")
        except Exception as e:
            # If we can't check or create, just continue - the push might still work
            logger.debug(f"Note about repo check/creation: {e}")

        # Push to Hugging Face Hub
        dataset_pushed = False

        # First, try to just push/overwrite the dataset directly
        # This works whether the dataset exists or not
        try:
            logger.debug(f"Pushing dataset to: {dataset_name}")
            dataset.push_to_hub(
                dataset_name,
                token=token,
                private=False,  # Make it public for community access
                split="train",  # Use main train split
            )
            logger.info(
                f"📤 Successfully pushed {len(processed_rollouts)} rollouts to HF dataset {dataset_name}"
            )
            dataset_pushed = True
        except Exception as push_error:
            logger.debug(f"Direct push failed, trying append approach: {push_error}")

            # If direct push fails, try to append to existing dataset
            try:
                from datasets import concatenate_datasets, load_dataset

                logger.debug(f"Attempting to load and append to existing dataset: {dataset_name}")

                # Try without force_redownload first (use cache if available)
                try:
                    existing_dataset = load_dataset(dataset_name, split="train", token=token)
                except Exception:
                    # If that fails, try forcing redownload
                    existing_dataset = load_dataset(
                        dataset_name,
                        split="train",
                        token=token,
                        download_mode="force_redownload",
                    )

                # Append new data to existing
                combined_dataset = concatenate_datasets([existing_dataset, dataset])  # type: ignore[arg-type]  # HF dataset type stub issue
                combined_dataset.push_to_hub(
                    dataset_name, token=token, private=False, split="train"
                )
                logger.info(
                    f"📤 Appended {len(processed_rollouts)} rollouts to existing HF dataset"
                )
                dataset_pushed = True
            except Exception as append_error:
                logger.error(f"Failed to push dataset via both methods: {append_error}")
                return False

        if not dataset_pushed:
            return False

        logger.info(
            f"📤 Successfully uploaded {len(processed_rollouts)} rollouts to HF dataset {dataset_name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        return False


async def download_from_huggingface(
    version: str | None = None,
    window: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Download rollouts from Hugging Face dataset with optional filtering.

    Args:
        version: Filter by protocol version (e.g., "v1.0.0")
        window: Filter by specific window number
        limit: Maximum number of rollouts to return

    Returns:
        List of rollout dictionaries
    """
    # Use the same dataset name as upload
    hf_username = get_conf("HF_USERNAME", "fatheroffire")
    dataset_name = f"{hf_username}/grail-sat-rollouts"

    try:
        from datasets import load_dataset

        # Load dataset
        dataset = load_dataset(dataset_name, split="train")

        # Apply filters
        if version:
            dataset = dataset.filter(lambda x: x["version"] == version)

        if window is not None:
            dataset = dataset.filter(lambda x: x["window"] == window)

        # Convert to list of dicts
        rollouts = dataset.to_list()  # type: ignore[attr-defined]  # HF dataset method exists but not in stubs

        # Apply limit if specified
        if limit:
            rollouts = rollouts[:limit]

        # Decode JSON strings back to objects
        for rollout in rollouts:
            rollout["sat_clauses"] = json.loads(rollout.get("sat_clauses", "[]"))
            rollout["solution_assignment"] = json.loads(rollout.get("solution_assignment", "[]"))
            rollout["solution_trajectory"] = json.loads(rollout.get("solution_trajectory", "[]"))
            rollout["grail_tokens"] = json.loads(rollout.get("grail_tokens", "[]"))
            rollout["grail_s_vals"] = json.loads(rollout.get("grail_s_vals", "[]"))
            rollout["grail_beacon"] = json.loads(rollout.get("grail_beacon", "{}"))
            rollout["grail_indices"] = json.loads(rollout.get("grail_indices", "[]"))

        logger.info(f"Downloaded {len(rollouts)} rollouts from HF dataset")
        return rollouts  # type: ignore[no-any-return]

    except Exception as e:
        logger.error(f"Failed to download from Hugging Face: {e}")
        return []
