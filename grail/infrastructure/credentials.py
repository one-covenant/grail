"""Credential loading utilities for GRAIL R2 integration."""

import logging
import os
from typing import Optional

from ..shared.schemas import BucketCredentials

logger = logging.getLogger(__name__)


def get_conf(key: str, default: Optional[str] = None) -> str:
    """
    Get configuration from environment variables.

    Args:
        key: Environment variable key
        default: Default value if not set

    Returns:
        Configuration value

    Raises:
        ValueError: If required key is not set and no default provided
    """
    value = os.getenv(key)
    if not value and default is None:
        raise ValueError(f"{key} not set. Please set the environment variable.")
    return value or default or ""


def load_r2_credentials(fallback_to_single: bool = True) -> BucketCredentials:
    """
    Load R2 credentials from environment variables.

    Supports both dual-credential mode (separate read/write keys for same
    bucket)
    and backwards-compatible single-credential mode.

    Args:
        fallback_to_single: If True, fall back to single credential mode
                          if dual credentials are not found

    Returns:
        BucketCredentials object with read and write credentials

    Raises:
        ValueError: If required credentials are not found
    """
    # Try to load dual credentials first
    try:
        logger.info("Attempting to load dual R2 credentials...")

        # Same bucket and account for both read and write
        bucket_id = get_conf("R2_BUCKET_ID")
        account_id = get_conf("R2_ACCOUNT_ID")

        credentials = BucketCredentials(
            # Shared bucket configuration
            bucket_name=bucket_id,
            account_id=account_id,
            # Read-only credentials (to be shared on chain)
            read_access_key_id=get_conf("R2_READ_ACCESS_KEY_ID"),
            read_secret_access_key=get_conf("R2_READ_SECRET_ACCESS_KEY"),
            # Read-write credentials (private, never shared)
            write_access_key_id=get_conf("R2_WRITE_ACCESS_KEY_ID"),
            write_secret_access_key=get_conf("R2_WRITE_SECRET_ACCESS_KEY"),
        )

        logger.info(f"Successfully loaded dual R2 credentials for bucket: {bucket_id}")
        return credentials

    except ValueError as e:
        if not fallback_to_single:
            raise

        logger.warning(f"Dual credentials not found: {e}")
        logger.info("Falling back to single credential mode (backwards compatibility)")

    # Fall back to single credential mode for backwards compatibility
    try:
        # Load single set of credentials
        account_id = get_conf("R2_ACCOUNT_ID")
        access_key = get_conf("R2_WRITE_ACCESS_KEY_ID")
        secret_key = get_conf("R2_WRITE_SECRET_ACCESS_KEY")
        bucket_id = get_conf("R2_BUCKET_ID")

        # Use the same credentials for both read and write
        credentials = BucketCredentials(
            # Shared bucket configuration
            bucket_name=bucket_id,
            account_id=account_id,
            # Use write credentials for read (backwards compatibility)
            read_access_key_id=access_key,
            read_secret_access_key=secret_key,
            # Write credentials
            write_access_key_id=access_key,
            write_secret_access_key=secret_key,
        )

        logger.warning(
            "Using single credential mode (deprecated). "
            "Please migrate to dual credentials for improved security. "
            "Set R2_READ_ACCESS_KEY_ID, R2_READ_SECRET_ACCESS_KEY, "
            "R2_WRITE_ACCESS_KEY_ID, and R2_WRITE_SECRET_ACCESS_KEY "
            "environment variables."
        )

        return credentials

    except ValueError as e:
        logger.error(
            "Failed to load R2 credentials. Please set either:\n"
            "1. Dual credentials: R2_BUCKET_ID, R2_ACCOUNT_ID, "
            "R2_READ_ACCESS_KEY_ID, R2_READ_SECRET_ACCESS_KEY, "
            "R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY\n"
            "2. Single credentials (deprecated): R2_BUCKET_ID, R2_ACCOUNT_ID, "
            "R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY"
        )
        raise ValueError(f"Failed to load R2 credentials: {e}") from e


def validate_credentials(credentials: BucketCredentials) -> bool:
    """
    Validate that credentials have required fields.

    Args:
        credentials: Credentials to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check shared configuration
        if not all([credentials.bucket_name, credentials.account_id]):
            logger.error("Missing bucket name or account ID")
            return False

        # Check read credentials
        if not all(
            [
                credentials.read_access_key_id,
                credentials.read_secret_access_key,
            ]
        ):
            logger.error("Missing required read credentials")
            return False

        # Check write credentials
        if not all(
            [
                credentials.write_access_key_id,
                credentials.write_secret_access_key,
            ]
        ):
            logger.error("Missing required write credentials")
            return False

        # Validate commitment length
        commitment = credentials.read_commitment
        if len(commitment) != 128:
            logger.error(f"Invalid commitment length: {len(commitment)} (expected 128)")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating credentials: {e}")
        return False
