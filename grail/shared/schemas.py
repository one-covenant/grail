"""Data schemas for GRAIL R2 credentials management."""

from pydantic import BaseModel, ConfigDict, Field


class BucketCredentials(BaseModel):
    """Dual credential configuration for R2 buckets

    Same bucket and account, but different access keys:
    - Read credentials: read-only access (shared on chain)
    - Write credentials: read-write access (kept private)
    """

    # Shared bucket configuration
    bucket_name: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)

    # Read-only credentials (shared on chain)
    read_access_key_id: str = Field(..., min_length=1)
    read_secret_access_key: str = Field(..., min_length=1)

    # Read-write credentials (local only, never shared)
    write_access_key_id: str = Field(..., min_length=1)
    write_secret_access_key: str = Field(..., min_length=1)

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    @property
    def read_commitment(self) -> str:
        """Generate commitment string for chain (128 chars total)

        Layout:
          32 account_id + 32 read_access_key_id + 64 read_secret_access_key
        Assumption: bucket name equals account_id.
        """
        return (
            self.account_id[:32].ljust(32)
            + self.read_access_key_id[:32].ljust(32)
            + self.read_secret_access_key[:64].ljust(64)
        )

    def get_read_dict(self) -> dict:
        """Get read credentials as dict for compatibility"""
        return {
            "name": self.bucket_name,
            "account_id": self.account_id,
            "access_key_id": self.read_access_key_id,
            "secret_access_key": self.read_secret_access_key,
        }

    def get_write_dict(self) -> dict:
        """Get write credentials as dict for compatibility"""
        return {
            "name": self.bucket_name,
            "account_id": self.account_id,
            "access_key_id": self.write_access_key_id,
            "secret_access_key": self.write_secret_access_key,
        }


class Bucket(BaseModel):
    """Configuration for a bucket (used for chain commitments)"""

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.account_id,
                self.access_key_id,
                self.secret_access_key,
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Bucket):
            return self.model_dump() == other.model_dump()
        return False

    name: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)
    access_key_id: str = Field(..., min_length=1)
    secret_access_key: str = Field(..., min_length=1)

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )
