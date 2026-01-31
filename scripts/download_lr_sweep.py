#!/usr/bin/env python3
"""Download LR sweep experiments from R2."""

import os
import sys
from pathlib import Path

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from botocore.config import Config
from botocore.session import get_session

# R2 credentials
R2_ACCOUNT_ID = os.getenv(
    "R2_ACCOUNT_ID", "91561e574629960f78e985efa5a37e59"
)
R2_BUCKET_ID = os.getenv(
    "R2_BUCKET_ID", "91561e574629960f78e985efa5a37e59"
)
R2_ACCESS_KEY = os.getenv(
    "R2_WRITE_ACCESS_KEY_ID", "5961758bc74f3554506f2ba05390a6dd"
)
R2_SECRET_KEY = os.getenv(
    "R2_WRITE_SECRET_ACCESS_KEY",
    "0a1fbf3a324e889d44d2a235eb58de661758aeba08a0c23d8f744dfd9fc3566a",
)
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Experiments to download
EXPERIMENTS = [
    "experiments/qwen2.5-1.5b-lr1e-6",
    "experiments/qwen2.5-1.5b-lr5e-6",
    "experiments/qwen2.5-1.5b-lr5e-7",
]

# Local destination
LOCAL_BASE = Path("/root/grail/research/sparsity_analysis")


def create_s3_client():
    """Create S3 client for R2."""
    session = get_session()
    config = Config(s3={"addressing_style": "path"}, region_name="auto")
    return session.create_client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=config,
    )


def list_objects(s3, prefix: str) -> list[dict]:
    """List all objects under a prefix."""
    objects = []
    paginator_token = None

    while True:
        kwargs = {"Bucket": R2_BUCKET_ID, "Prefix": prefix}
        if paginator_token:
            kwargs["ContinuationToken"] = paginator_token

        response = s3.list_objects_v2(**kwargs)
        objects.extend(response.get("Contents", []))

        if response.get("IsTruncated"):
            paginator_token = response.get("NextContinuationToken")
        else:
            break

    return objects


def download_experiment(s3, experiment: str):
    """Download a single experiment."""
    print(f"\n{'='*60}")
    print(f"Downloading: {experiment}")
    print("=" * 60)

    # List all objects in the experiment
    objects = list_objects(s3, f"{experiment}/")
    print(f"Found {len(objects)} files")

    if not objects:
        print("WARNING: No files found!")
        return

    # Calculate total size
    total_size = sum(obj["Size"] for obj in objects)
    print(f"Total size: {total_size / 1024**3:.2f} GB")

    # Download each file
    downloaded = 0
    for i, obj in enumerate(objects):
        key = obj["Key"]
        local_path = LOCAL_BASE / key

        # Create parent directory
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists and same size
        if local_path.exists() and local_path.stat().st_size == obj["Size"]:
            downloaded += obj["Size"]
            continue

        # Download using get_object
        try:
            response = s3.get_object(Bucket=R2_BUCKET_ID, Key=key)
            with open(local_path, "wb") as f:
                # Stream the body to file in chunks
                body = response["Body"]
                while chunk := body.read(8 * 1024 * 1024):  # 8MB chunks
                    f.write(chunk)

            downloaded += obj["Size"]

            if (i + 1) % 50 == 0 or i == len(objects) - 1:
                pct = downloaded / total_size * 100
                print(f"  Progress: {i+1}/{len(objects)} files ({pct:.1f}%)")

        except Exception as e:
            print(f"  ERROR downloading {key}: {e}")

    print(f"Done: {experiment}")


def main():
    print("LR Sweep Experiment Downloader")
    print(f"Destination: {LOCAL_BASE}")

    s3 = create_s3_client()

    for experiment in EXPERIMENTS:
        download_experiment(s3, experiment)

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
