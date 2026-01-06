#!/usr/bin/env python3
"""
Script to remove all files and folders from R2 bucket.
Uses Cloudflare R2 credentials from environment variables.
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

# Get R2 credentials from environment
R2_BUCKET_ID = os.getenv("R2_BUCKET_ID")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_WRITE_ACCESS_KEY_ID = os.getenv("R2_WRITE_ACCESS_KEY_ID")
R2_WRITE_SECRET_ACCESS_KEY = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
R2_FORCE_PATH_STYLE = os.getenv("R2_FORCE_PATH_STYLE", "true").lower() == "true"

# Validate credentials
if not all([R2_BUCKET_ID, R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY]):
    print("Error: Missing R2 credentials in environment variables")
    sys.exit(1)

# R2 API endpoint
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Create S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_WRITE_ACCESS_KEY_ID,
    aws_secret_access_key=R2_WRITE_SECRET_ACCESS_KEY,
    region_name="auto",
    config=boto3.session.Config(
        s3={"addressing_style": "path" if R2_FORCE_PATH_STYLE else "virtual"}
    ),
)


def delete_all_objects(bucket_name):
    """Delete all objects in the bucket."""
    print(f"Starting deletion of all objects in bucket: {bucket_name}")

    try:
        # List all objects
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name)

        total_deleted = 0

        for page in pages:
            if "Contents" not in page:
                print("Bucket is already empty!")
                return 0

            # Prepare deletion list
            objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]

            # Delete objects in batch (max 1000 per request)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i : i + 1000]
                response = s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": batch})

                deleted_count = len(response.get("Deleted", []))
                total_deleted += deleted_count
                print(f"Deleted {deleted_count} objects (Total: {total_deleted})")

                # Print any errors
                if "Errors" in response:
                    for error in response["Errors"]:
                        print(f"  Error deleting {error['Key']}: {error['Message']}")

        return total_deleted

    except ClientError as e:
        print(f"Error: {e}")
        sys.exit(1)


def delete_all_versions(bucket_name):
    """Delete all versions of all objects (for versioned buckets)."""
    print(f"Checking for versioned objects in bucket: {bucket_name}")

    try:
        paginator = s3_client.get_paginator("list_object_versions")
        pages = paginator.paginate(Bucket=bucket_name)

        total_deleted = 0

        for page in pages:
            # Delete all versions
            if "Versions" in page:
                objects_to_delete = [
                    {"Key": obj["Key"], "VersionId": obj["VersionId"]} for obj in page["Versions"]
                ]

                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i : i + 1000]
                    response = s3_client.delete_objects(
                        Bucket=bucket_name, Delete={"Objects": batch}
                    )

                    deleted_count = len(response.get("Deleted", []))
                    total_deleted += deleted_count
                    print(f"Deleted {deleted_count} versioned objects (Total: {total_deleted})")

            # Delete all delete markers
            if "DeleteMarkers" in page:
                objects_to_delete = [
                    {"Key": obj["Key"], "VersionId": obj["VersionId"]}
                    for obj in page["DeleteMarkers"]
                ]

                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i : i + 1000]
                    response = s3_client.delete_objects(
                        Bucket=bucket_name, Delete={"Objects": batch}
                    )

                    deleted_count = len(response.get("Deleted", []))
                    total_deleted += deleted_count
                    print(f"Deleted {deleted_count} delete markers (Total: {total_deleted})")

        return total_deleted

    except ClientError as e:
        print(f"Error: {e}")
        return 0


def main():
    bucket_name = R2_BUCKET_ID

    print("R2 Configuration:")
    print(f"  Account ID: {R2_ACCOUNT_ID}")
    print(f"  Bucket ID: {R2_BUCKET_ID}")
    print(f"  Endpoint: {R2_ENDPOINT}")
    print(f"  Force Path Style: {R2_FORCE_PATH_STYLE}")
    print()

    # Confirm deletion (skip if --force flag is passed)
    if "--force" not in sys.argv:
        response = input(
            f"WARNING: This will delete ALL objects in bucket '{bucket_name}'. Continue? (yes/no): "
        )
        if response.lower() != "yes":
            print("Operation cancelled.")
            sys.exit(0)
    else:
        print(f"WARNING: This will delete ALL objects in bucket '{bucket_name}'.")
        print("Proceeding with --force flag...")

    print()

    # Delete all objects
    count1 = delete_all_objects(bucket_name)

    # Delete all versions (in case of versioned bucket)
    count2 = delete_all_versions(bucket_name)

    total = count1 + count2
    print()
    print(f"✓ Successfully deleted {total} total objects from R2 bucket")


if __name__ == "__main__":
    main()
