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
from typing import Any, Optional, Union

import bittensor as bt
from aiobotocore.session import get_session
from botocore.config import Config
from datasets import Dataset
from huggingface_hub import HfFolder
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from ..shared.constants import WINDOW_LENGTH
from ..shared.schemas import Bucket, BucketCredentials

logger = logging.getLogger(__name__)

# Protocol version for dataset versioning
PROTOCOL_VERSION = "v1.0.0"

# --------------------------------------------------------------------------- #
#                   S3/R2 Configuration                                       #
# --------------------------------------------------------------------------- #


def get_conf(key: str, default: Optional[str] = None) -> str:
    """Get configuration from environment variables."""
    v = os.getenv(key)
    if not v and default is None:
        raise ValueError(f"{key} not set. Please set the environment variable.")
    return v or default or ""


def get_client_ctx(
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
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
    # indefinite hangs. Set these higher than our asyncio timeout (6s)
    # to let asyncio.wait_for handle the timeout cleanly.
    # Root cause: botocore and asyncio timeouts conflict when stacked.
    config = Config(
        connect_timeout=3,
        read_timeout=10,  # Higher than asyncio timeout
        retries={"max_attempts": 2, "mode": "standard"},
        max_pool_connections=256,
        region_name=region_name,
        signature_version="s3v4",
        s3=s3_config or None,
    )

    return get_session().create_client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=config,
    )


def get_bucket_id(
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = True,
) -> str:
    """Get the bucket ID from credentials or environment.

    Note: Since we use the same bucket for both read and write operations,
    the use_write parameter doesn't affect the bucket ID, only the access keys.
    """
    if credentials is not None:
        if isinstance(credentials, BucketCredentials):
            return credentials.bucket_name
        elif isinstance(credentials, Bucket):
            return credentials.name.strip()
        elif isinstance(credentials, dict):
            name = credentials.get("name", credentials.get("bucket_name", "")).strip()
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
    """Track upload/download progress and speed"""

    def __init__(self, total_size: int, operation: str):
        self.total_size = total_size
        self.operation = operation
        self.transferred = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_transferred = 0

    def update(self, bytes_transferred: int) -> None:
        self.transferred += bytes_transferred
        now = time.time()

        # Log progress every 2 seconds or on completion
        if now - self.last_log_time >= 2.0 or self.transferred >= self.total_size:
            elapsed = now - self.start_time
            speed_mbps = (self.transferred / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            progress_pct = (self.transferred / self.total_size) * 100 if self.total_size > 0 else 0

            logger.info(
                f"📊 {self.operation}: {progress_pct:.1f}% ({self.transferred}/{self.total_size} bytes) @ {speed_mbps:.2f} MB/s"
            )
            self.last_log_time = now
            self.last_transferred = self.transferred


# --------------------------------------------------------------------------- #
#                   Upload Functions                                          #
# --------------------------------------------------------------------------- #


async def upload_file_chunked(
    key: str,
    data: bytes,
    chunk_size: int = 100 * 1024 * 1024,
    max_retries: int = 3,
    compress: bool = True,
    credentials: Optional[BucketCredentials] = None,
    use_write: bool = False,
) -> bool:
    """Upload file in chunks optimized for H100 high-bandwidth - 100MB chunks with compression"""
    # Compress data if enabled and it's JSON
    if compress and key.endswith(".json"):
        original_size = len(data)
        data = gzip.compress(data, compresslevel=1)  # Fast compression
        key = key + ".gz"
        logger.info(
            f"🗜️ Compressed {original_size} → {len(data)} bytes ({100 * (1 - len(data) / original_size):.1f}% reduction)"
        )

    total_size = len(data)
    progress = TransferProgress(total_size, f"Upload {key}")

    # For small files, use single upload
    if total_size <= chunk_size:
        logger.info(f"📤 Uploading {key} ({total_size} bytes)")
        return await _upload_single_chunk(key, data, progress, max_retries, credentials, use_write)

    # For large files, use multipart upload
    logger.info(
        f"📤 Starting chunked upload of {key} ({total_size} bytes, {(total_size + chunk_size - 1) // chunk_size} chunks)"
    )

    try:
        async with get_client_ctx(credentials, use_write) as client:
            # Initiate multipart upload
            bucket_id = get_bucket_id(credentials, use_write)
            response = await client.create_multipart_upload(Bucket=bucket_id, Key=key)
            upload_id = response["UploadId"]

            # Upload chunks concurrently with limited concurrency
            semaphore = asyncio.Semaphore(30)  # High concurrency for H100 bandwidth
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
                    await client.abort_multipart_upload(
                        Bucket=bucket_id, Key=key, UploadId=upload_id
                    )
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

    except Exception as e:
        logger.error(f"❌ Upload failed for {key}: {e}")
        return False


async def _upload_single_chunk(
    key: str,
    data: bytes,
    progress: TransferProgress,
    max_retries: int,
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = True,
) -> bool:
    """Upload single chunk with retry logic"""
    for attempt in range(max_retries):
        try:
            async with get_client_ctx(credentials, use_write) as client:
                bucket_id = get_bucket_id(credentials, use_write)
                await client.put_object(Bucket=bucket_id, Key=key, Body=data)
            progress.update(len(data))
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"Upload attempt {attempt + 1} failed for {key}, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Upload failed after {max_retries} attempts for {key}: {e}")
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = True,
) -> Optional[dict[str, Any]]:
    """Upload a single chunk with concurrency control and retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                bucket_id = get_bucket_id(credentials, use_write)
                response = await client.upload_part(
                    Bucket=bucket_id,
                    Key=key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=data,
                )
                progress.update(len(data))
                return {"PartNumber": part_number, "ETag": response["ETag"]}
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Chunk {part_number} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> Optional[bytes]:
    """Download file in chunks with automatic decompression"""
    actual_key = key
    is_compressed = False

    for attempt in range(max_retries):
        try:
            async with get_client_ctx(credentials, use_write) as client:
                # Try compressed version first if not already .gz
                if not key.endswith(".gz"):
                    try:
                        compressed_key = key + ".gz"
                        bucket_id = get_bucket_id(credentials, use_write)
                        head_response = await client.head_object(
                            Bucket=bucket_id, Key=compressed_key
                        )
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

                logger.info(
                    f"📥 Downloading {actual_key} ({total_size} bytes){' (compressed)' if is_compressed else ''}"
                )
                progress = TransferProgress(total_size, f"Download {actual_key}")

                # For small files, download in one go
                chunk_size = 100 * 1024 * 1024  # 100MB chunks for H100 bandwidth
                if total_size <= chunk_size:
                    bucket_id = get_bucket_id(credentials, use_write)
                    response = await client.get_object(Bucket=bucket_id, Key=actual_key)
                    data = await response["Body"].read()
                    progress.update(len(data))
                    elapsed = time.time() - progress.start_time
                    speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"✅ Download completed: {actual_key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s"
                    )
                    # Decompress if needed
                    if is_compressed:
                        data = gzip.decompress(data)
                        logger.debug(f"🗜️ Decompressed to {len(data)} bytes")
                    return data  # type: ignore[no-any-return]

                # For large files, download in chunks
                chunks = []
                semaphore = asyncio.Semaphore(30)  # High concurrency for H100 bandwidth
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
                    chunks.append(result)

                data = b"".join(chunks)  # type: ignore[arg-type]
                elapsed = time.time() - progress.start_time
                speed_mbps = (total_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"✅ Download completed: {actual_key} ({total_size} bytes) in {elapsed:.1f}s @ {speed_mbps:.2f} MB/s"
                )

                # Decompress if needed
                if is_compressed:
                    data = gzip.decompress(data)
                    logger.debug(f"🗜️ Decompressed to {len(data)} bytes")
                return data

        except Exception as e:
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> Optional[bytes]:
    """Download a single chunk with concurrency control and retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                bucket_id = get_bucket_id(credentials, use_write)
                response = await client.get_object(
                    Bucket=bucket_id, Key=key, Range=f"bytes={start}-{end}"
                )
                data = await response["Body"].read()
                progress.update(len(data))
                return data  # type: ignore[no-any-return]
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        f"Download chunk {start}-{end} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    return None


# --------------------------------------------------------------------------- #
#                   File Operations                                           #
# --------------------------------------------------------------------------- #


async def file_exists(
    key: str,
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> bool:
    """Check if a file exists (compressed or uncompressed) with bounded HEAD.

    Each HEAD call is wrapped in a small timeout so one stalled request
    cannot block upstream gather calls. Root cause: rare transport stalls
    (no response) on R2 under concurrency; without timeouts, awaits can
    hang indefinitely. We treat timeouts as "not found" to keep progress.
    """

    async def _head_exists(client: Any, bucket_id: str, key_to_check: str) -> bool:
        try:
            # Remove nested asyncio.wait_for - let the outer timeout handle it
            # to avoid timeout conflicts between botocore and asyncio
            await client.head_object(Bucket=bucket_id, Key=key_to_check)
            return True
        except Exception:
            # NOTE: uncomment this for debugging later on
            # elapsed = time.time() - start_time
            # logger.warning(
            #     "R2 HEAD failed after %.2fs for key=%s bucket=%s: %s %s",
            #     elapsed,
            #     key_to_check,
            #     bucket_id,
            #     type(e).__name__,
            #     str(e),
            # )
            return False

    try:
        async with get_client_ctx(credentials, use_write) as client:
            bucket_id = get_bucket_id(credentials, use_write)

            # Check for compressed first
            if not key.endswith(".gz"):
                if await _head_exists(client, bucket_id, key + ".gz"):
                    return True

            # Fallback to original key
            return await _head_exists(client, bucket_id, key)
    except Exception as e:
        logger.warning("file_exists failed for key=%s: %s %s", key, type(e).__name__, str(e))
        return False


async def list_bucket_files(
    prefix: str,
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> list[str]:
    """List files in bucket with given prefix"""
    try:
        async with get_client_ctx(credentials, use_write) as client:
            bucket_id = get_bucket_id(credentials, use_write)
            response = await client.list_objects_v2(Bucket=bucket_id, Prefix=prefix)
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            return []
    except Exception:
        logger.error("Failed to list bucket files with prefix %s", prefix, exc_info=True)
        return []


async def get_file(
    key: str,
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> Optional[dict[str, Any]]:
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


# --------------------------------------------------------------------------- #
#                   GRAIL-specific Storage Functions                          #
# --------------------------------------------------------------------------- #


async def sink_window_inferences(
    wallet: bt.wallet,
    window_start: int,
    inferences: list[dict],
    credentials: Optional[BucketCredentials] = None,
) -> None:
    """Upload window of inferences to S3 with improved logging"""
    key = f"grail/windows/{wallet.hotkey.ss58_address}-window-{window_start}.json"

    # Pack all inferences into window data
    window_data = {
        "wallet": wallet.hotkey.ss58_address,
        "window_start": window_start,
        "window_length": WINDOW_LENGTH,
        "inference_count": len(inferences),
        "inferences": inferences,
        "timestamp": time.time(),
    }

    body = json.dumps(window_data).encode()
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
    credentials: Optional[BucketCredentials] = None,
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
    use_write: bool = False,
) -> bool:
    # Check if model state exists for given hotkey and window
    key = f"grail/models/{hotkey}-{window}.safetensors"
    return await file_exists(key, credentials=credentials, use_write=use_write)


async def upload_valid_rollouts(
    window: int,
    valid_rollouts: list[dict],
    credentials: Optional[BucketCredentials] = None,
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
    credentials: Optional[Union[BucketCredentials, Bucket, dict]] = None,
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
    rollouts: list[dict], window: int, version: Optional[str] = None
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
                combined_dataset = concatenate_datasets([existing_dataset, dataset])
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
    version: Optional[str] = None,
    window: Optional[int] = None,
    limit: Optional[int] = None,
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
        rollouts = dataset.to_list()

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
