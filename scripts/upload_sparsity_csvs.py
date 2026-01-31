#!/usr/bin/env python3
"""Upload sparsity CSV files to R2 storage."""

import logging
import os
import sys
from pathlib import Path

# Add research/infrastructure to path for R2Uploader
sys.path.insert(0, str(Path(__file__).parent.parent / "research" / "infrastructure"))

from dotenv import load_dotenv
from r2_uploader import R2Uploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def upload_sparsity_csvs(csv_files: list[Path], r2_prefix: str = "sparsity_analysis") -> bool:
    """Upload sparsity CSV files to R2.

    Args:
        csv_files: List of CSV file paths to upload
        r2_prefix: R2 prefix for uploaded files

    Returns:
        True if all uploads succeeded
    """
    # Load environment variables
    load_dotenv(Path(__file__).parent.parent / ".env")

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_WRITE_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_WRITE_SECRET_ACCESS_KEY")
    # Bucket name is the same as R2_BUCKET_ID per project convention
    bucket_name = os.environ.get("R2_BUCKET_NAME") or os.environ.get("R2_BUCKET_ID")

    if not all([account_id, access_key, secret_key, bucket_name]):
        logger.error("Missing R2 credentials in environment")
        return False

    logger.info(f"Using bucket: {bucket_name}")

    uploader = R2Uploader(
        account_id=account_id,
        access_key=access_key,
        secret_key=secret_key,
    )

    # Verify bucket access
    if not uploader.verify_bucket_access(bucket_name):
        logger.error(f"Cannot access bucket: {bucket_name}")
        return False

    all_success = True
    for csv_file in csv_files:
        if not csv_file.exists():
            logger.warning(f"File not found: {csv_file}")
            continue

        if csv_file.stat().st_size == 0:
            logger.warning(f"Skipping empty file: {csv_file}")
            continue

        r2_key = f"{r2_prefix}/{csv_file.name}"
        logger.info(f"Uploading {csv_file.name} -> {r2_key}")

        if not uploader.upload_file(csv_file, bucket_name, r2_key):
            logger.error(f"Failed to upload: {csv_file}")
            all_success = False
        else:
            logger.info(f"Successfully uploaded: {csv_file.name}")

    return all_success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload sparsity CSVs to R2")
    parser.add_argument("files", nargs="+", type=Path, help="CSV files to upload")
    parser.add_argument("--prefix", default="sparsity_analysis", help="R2 prefix")
    args = parser.parse_args()

    success = upload_sparsity_csvs(args.files, args.prefix)
    sys.exit(0 if success else 1)
