"""R2 uploader utility for experiment artifacts.

Uploads training artifacts (logs, outputs, checkpoints) to Cloudflare R2
using boto3 S3-compatible API with retry logic and progress tracking.
"""

from __future__ import annotations

import fnmatch
import logging
import re
import time
from pathlib import Path
from typing import Any

import boto3
from boto3.exceptions import Boto3Error
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BUCKET_ID_LIKE_RE = re.compile(r"^[0-9a-f]{32}$")


def _warn_if_bucket_looks_like_id(bucket_name: str) -> None:
    if _BUCKET_ID_LIKE_RE.fullmatch(bucket_name):
        logger.warning(
            "R2 bucket value looks like an ID. Cloudflare R2's S3 API expects the bucket *name*.",
            extra={"bucket_value": bucket_name},
        )


def _should_exclude(rel_path: Path, exclude_patterns: list[str]) -> bool:
    """Check whether a relative path should be excluded.

    Notes:
    - Patterns without "/" are applied to the basename and directory components
      (e.g., "__pycache__" excludes anything under that directory).
    - Patterns with "/" are matched against the full relative POSIX path.
    """
    if not exclude_patterns:
        return False

    rel_posix = rel_path.as_posix()
    for pattern in exclude_patterns:
        if "/" in pattern:
            if fnmatch.fnmatch(rel_posix, pattern):
                return True
            continue

        if fnmatch.fnmatch(rel_path.name, pattern):
            return True

        # Allow excluding by directory name (e.g., "__pycache__")
        if pattern in rel_path.parts:
            return True

    return False


class R2Uploader:
    """Upload files and directories to Cloudflare R2 storage.

    Uses boto3 S3-compatible API with automatic retries and progress tracking.

    Example:
        >>> uploader = R2Uploader(
        ...     account_id="your-account-id",
        ...     access_key="your-access-key",
        ...     secret_key="your-secret-key",
        ... )
        >>> uploader.upload_directory(
        ...     local_dir=Path("./outputs"),
        ...     bucket_name="my-bucket",
        ...     r2_prefix="experiment-1/outputs",
        ... )
    """

    def __init__(
        self,
        account_id: str,
        access_key: str,
        secret_key: str,
        max_retries: int = 3,
        timeout: int = 300,
    ) -> None:
        """Initialize R2 uploader.

        Args:
            account_id: Cloudflare R2 account ID
            access_key: R2 access key ID
            secret_key: R2 secret access key
            max_retries: Maximum number of upload retries (default: 3)
            timeout: Timeout in seconds for each upload (default: 300)
        """
        self.account_id: str = account_id
        self.max_retries: int = max_retries

        # Configure boto3 client for R2
        self.endpoint_url: str = f"https://{account_id}.r2.cloudflarestorage.com"

        boto_config = Config(
            # We manage retries ourselves (for clearer logs/backoff). Avoid "double retry".
            retries={"max_attempts": 1, "mode": "standard"},
            connect_timeout=timeout,
            read_timeout=timeout,
            # R2 is S3-compatible but works most reliably with path-style addressing.
            s3={"addressing_style": "path"},
            signature_version="s3v4",
        )

        self.s3_client: Any = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            region_name="auto",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto_config,
        )

        logger.info(
            "R2 uploader initialized",
            extra={"account_id": account_id, "endpoint_url": self.endpoint_url},
        )

    def upload_file(
        self,
        local_file: Path,
        bucket_name: str,
        r2_key: str,
        retry_delay: float = 2.0,
    ) -> bool:
        """Upload a single file to R2.

        Args:
            local_file: Path to local file
            bucket_name: R2 bucket name
            r2_key: Destination key in R2 (e.g., "experiments/run1/file.txt")
            retry_delay: Delay between retries in seconds (default: 2.0)

        Returns:
            True if upload succeeded, False otherwise
        """
        if not local_file.exists():
            logger.error("File not found", extra={"local_file": str(local_file)})
            return False

        _warn_if_bucket_looks_like_id(bucket_name)
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Uploading file",
                    extra={
                        "local_file": str(local_file),
                        "bucket": bucket_name,
                        "key": r2_key,
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                    },
                )

                self.s3_client.upload_file(
                    Filename=str(local_file),
                    Bucket=bucket_name,
                    Key=r2_key,
                )

                logger.info("Uploaded", extra={"key": r2_key})
                return True

            except (ClientError, BotoCoreError, Boto3Error) as e:
                error_code = type(e).__name__
                error_message: str | None = None
                if isinstance(e, ClientError):
                    err = e.response.get("Error", {})
                    error_code = err.get("Code", "Unknown")
                    error_message = err.get("Message")

                logger.warning(
                    "Upload failed",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error_code": error_code,
                        "error_message": error_message,
                        "local_file": str(local_file),
                    },
                )

                if attempt < self.max_retries:
                    time.sleep(retry_delay * (2 ** (attempt - 1)))  # Exponential backoff
                    continue

                logger.error(
                    "Failed to upload after retries",
                    extra={"local_file": str(local_file), "max_retries": self.max_retries},
                )
                return False
            except OSError as e:
                logger.error(
                    "Local file error uploading",
                    extra={
                        "local_file": str(local_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                return False

        return False

    def upload_directory(
        self,
        local_dir: Path,
        bucket_name: str,
        r2_prefix: str,
        exclude_patterns: list[str] | None = None,
    ) -> tuple[int, int]:
        """Upload entire directory to R2 recursively.

        Args:
            local_dir: Path to local directory
            bucket_name: R2 bucket name
            r2_prefix: Prefix for all uploaded files (e.g., "experiments/run1")
            exclude_patterns: List of glob patterns to exclude (e.g., ["*.tmp", "__pycache__"])

        Returns:
            Tuple of (successful_uploads, failed_uploads)
        """
        if not local_dir.exists():
            logger.error("Directory not found", extra={"local_dir": str(local_dir)})
            return (0, 0)

        if not local_dir.is_dir():
            logger.error("Not a directory", extra={"local_dir": str(local_dir)})
            return (0, 0)

        # Collect all files to upload
        files_to_upload: list[tuple[Path, str]] = []
        exclude_patterns = exclude_patterns or []

        for file_path in local_dir.rglob("*"):
            if not file_path.is_file() or file_path.is_symlink():
                continue

            rel_path = file_path.relative_to(local_dir)
            if _should_exclude(rel_path, exclude_patterns):
                continue

            r2_key = f"{r2_prefix.rstrip('/')}/{rel_path.as_posix()}"
            files_to_upload.append((file_path, r2_key))

        if not files_to_upload:
            logger.warning("No files found to upload", extra={"local_dir": str(local_dir)})
            return (0, 0)

        _warn_if_bucket_looks_like_id(bucket_name)
        logger.info(
            "Uploading directory",
            extra={
                "file_count": len(files_to_upload),
                "local_dir": str(local_dir),
                "bucket": bucket_name,
                "prefix": r2_prefix,
            },
        )

        # Upload with progress bar
        success_count = 0
        fail_count = 0

        if tqdm is None:
            for local_file, r2_key in files_to_upload:
                if self.upload_file(local_file, bucket_name, r2_key):
                    success_count += 1
                else:
                    fail_count += 1
        else:
            with tqdm(total=len(files_to_upload), desc="Uploading", unit="file") as pbar:
                for local_file, r2_key in files_to_upload:
                    if self.upload_file(local_file, bucket_name, r2_key):
                        success_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)

        logger.info(
            "Upload complete",
            extra={
                "succeeded": success_count,
                "failed": fail_count,
                "total": len(files_to_upload),
            },
        )

        return (success_count, fail_count)

    def verify_bucket_access(self, bucket_name: str) -> bool:
        """Verify that we can access the R2 bucket.

        Args:
            bucket_name: R2 bucket name to verify

        Returns:
            True if bucket is accessible, False otherwise
        """
        _warn_if_bucket_looks_like_id(bucket_name)
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info("Verified bucket access", extra={"bucket": bucket_name})
            return True
        except (ClientError, BotoCoreError, Boto3Error) as e:
            error_code = type(e).__name__
            error_message: str | None = None
            if isinstance(e, ClientError):
                err = e.response.get("Error", {})
                error_code = err.get("Code", "Unknown")
                error_message = err.get("Message")
            logger.error(
                "Cannot access bucket",
                extra={
                    "bucket": bucket_name,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
            return False


def upload_experiment_artifacts(
    local_base_dir: Path,
    experiment_name: str,
    bucket_id: str,
    account_id: str,
    access_key: str,
    secret_key: str,
    artifact_dirs: list[str] | None = None,
) -> bool:
    """Upload all experiment artifacts to R2.

    Convenience function for uploading standard experiment artifacts
    (logs, outputs, checkpoints) to R2.

    Args:
        local_base_dir: Base directory containing artifact subdirectories
        experiment_name: Name of experiment (used as R2 prefix)
        bucket_id: R2 bucket name (legacy arg/env var name: R2_BUCKET_ID)
        account_id: R2 account ID
        access_key: R2 access key ID
        secret_key: R2 secret access key
        artifact_dirs: List of subdirectories to upload (default: ["logs", "outputs", "checkpoints"])

    Returns:
        True if all uploads succeeded, False otherwise

    Example:
        >>> upload_experiment_artifacts(
        ...     local_base_dir=Path("/home/ubuntu/grail/research/trl"),
        ...     experiment_name="qwen2.5-0.5b-iter1",
        ...     bucket_id="your-bucket-name",
        ...     account_id="your-account-id",
        ...     access_key="your-access-key",
        ...     secret_key="your-secret-key",
        ... )
    """
    artifact_dirs = artifact_dirs or ["logs", "outputs", "checkpoints"]
    bucket_name = bucket_id
    _warn_if_bucket_looks_like_id(bucket_name)

    uploader = R2Uploader(
        account_id=account_id,
        access_key=access_key,
        secret_key=secret_key,
    )

    # Verify bucket access first
    if not uploader.verify_bucket_access(bucket_name):
        logger.error("Bucket access verification failed, aborting upload")
        return False

    all_success = True

    for artifact_dir in artifact_dirs:
        local_dir = local_base_dir / artifact_dir

        if not local_dir.exists():
            logger.warning(f"Artifact directory not found, skipping: {local_dir}")
            continue

        r2_prefix = f"experiments/{experiment_name}/{artifact_dir}"

        success, failed = uploader.upload_directory(
            local_dir=local_dir,
            bucket_name=bucket_name,
            r2_prefix=r2_prefix,
            exclude_patterns=["*.tmp", "__pycache__", "*.pyc"],
        )

        if failed > 0:
            all_success = False

    return all_success


if __name__ == "__main__":
    # Example usage
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 3:
        logger.error("Usage: python r2_uploader.py <local_dir> <experiment_name>")
        logger.error(
            "Requires env vars: R2_BUCKET_NAME (preferred) or R2_BUCKET_ID (legacy), "
            "R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY"
        )
        raise SystemExit(1)

    local_dir = Path(sys.argv[1])
    experiment_name = sys.argv[2]
    bucket_name = os.getenv("R2_BUCKET_NAME") or os.getenv("R2_BUCKET_ID")
    if not bucket_name:
        logger.error(
            "Missing bucket env var: set R2_BUCKET_NAME (preferred) or R2_BUCKET_ID (legacy)"
        )
        raise SystemExit(1)

    success = upload_experiment_artifacts(
        local_base_dir=local_dir,
        experiment_name=experiment_name,
        bucket_id=bucket_name,
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key=os.environ["R2_WRITE_ACCESS_KEY_ID"],
        secret_key=os.environ["R2_WRITE_SECRET_ACCESS_KEY"],
    )

    raise SystemExit(0 if success else 1)
