#!/usr/bin/env python3
"""Copy objects between two R2 buckets on different accounts.

Parallel streaming transfers with per-object verification, retries, and reconciliation.
Resume-safe: skips objects already in destination with matching size.
"""

import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
from botocore.config import Config

# --- Source bucket (read credentials) ---
SRC_ACCOUNT_ID = "91561e574629960f78e985efa5a37e59"
SRC_BUCKET = "91561e574629960f78e985efa5a37e59"
SRC_ACCESS_KEY = "2a4e12f622668457a871d2e80de3439e"
SRC_SECRET_KEY = "eea24b0551188a57a90bbe3b83de32880030b1108955022f3d78d7f33895c058"

# --- Destination bucket (write credentials) ---
DST_ACCOUNT_ID = "8af7f92a8a0661cf7f1ac0420c932980"
DST_BUCKET = "sparsity-experiments"
DST_ACCESS_KEY = "f43049ae162d2b216da683e68926c43d"
DST_SECRET_KEY = "a727dd36a56120d12aac0f9ec70bb00d5da861b5ad072e28302cbfcbdda1c3a9"

# Folders to exclude
EXCLUDE_PREFIXES = ["grail/"]

# Parallelism
NUM_WORKERS = 16

# Multipart config
MULTIPART_THRESHOLD = 50 * 1024 * 1024   # 50 MB
MULTIPART_CHUNKSIZE = 25 * 1024 * 1024   # 25 MB

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds

# Thread-local boto3 clients (not thread-safe, need one per thread)
_thread_local = threading.local()


def get_clients():
    """Get or create per-thread boto3 clients."""
    if not hasattr(_thread_local, "src"):
        _thread_local.src = boto3.client(
            "s3",
            endpoint_url=f"https://{SRC_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=SRC_ACCESS_KEY,
            aws_secret_access_key=SRC_SECRET_KEY,
            region_name="auto",
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
        _thread_local.dst = boto3.client(
            "s3",
            endpoint_url=f"https://{DST_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=DST_ACCESS_KEY,
            aws_secret_access_key=DST_SECRET_KEY,
            region_name="auto",
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
    return _thread_local.src, _thread_local.dst


def list_all_objects(client, bucket, exclude_prefixes=None):
    """List all objects, optionally excluding specified prefixes."""
    exclude_prefixes = exclude_prefixes or []
    objects = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.startswith(p) for p in exclude_prefixes):
                continue
            objects[key] = obj["Size"]
    return objects


def copy_single_object(key, expected_size, counter, total, lock):
    """Copy one object with retries and verification. Called from thread pool."""
    src, dst = get_clients()
    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD,
        multipart_chunksize=MULTIPART_CHUNKSIZE,
        max_concurrency=1,  # per-file concurrency=1 since we parallelize across files
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Stream from source to destination
            response = src.get_object(Bucket=SRC_BUCKET, Key=key)
            dst.upload_fileobj(
                response["Body"],
                DST_BUCKET,
                key,
                Config=transfer_config,
            )

            # Verify size
            head = dst.head_object(Bucket=DST_BUCKET, Key=key)
            actual_size = head["ContentLength"]
            if actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch: expected {expected_size}, got {actual_size}"
                )

            with lock:
                counter["done"] += 1
                counter["bytes"] += expected_size
                n = counter["done"] + counter["skipped"]
                size_mb = expected_size / (1024**2)
                elapsed = time.time() - counter["start"]
                rate = counter["bytes"] / (1024**2) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{n}/{total}] OK: {key} ({size_mb:.1f} MB) "
                    f"[{rate:.1f} MB/s, {counter['done']} copied, {counter['skipped']} skipped]",
                    flush=True,
                )
            return True

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                with lock:
                    counter["failed"].append((key, expected_size, str(e)))
                    n = counter["done"] + counter["skipped"] + len(counter["failed"])
                    print(
                        f"  [{n}/{total}] FAILED: {key} after {MAX_RETRIES} attempts: {e}",
                        flush=True,
                    )
                return False


def main():
    test_mode = "--test" in sys.argv
    start_time = time.time()

    # Use a shared client for listing (single-threaded)
    src = boto3.client(
        "s3",
        endpoint_url=f"https://{SRC_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=SRC_ACCESS_KEY,
        aws_secret_access_key=SRC_SECRET_KEY,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    dst = boto3.client(
        "s3",
        endpoint_url=f"https://{DST_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=DST_ACCESS_KEY,
        aws_secret_access_key=DST_SECRET_KEY,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )

    # ── Phase 1: List source objects ──
    print(f"[{datetime.now(timezone.utc).isoformat()}] Phase 1: Listing source objects (excluding grail/)...")
    src_objects = list_all_objects(src, SRC_BUCKET, EXCLUDE_PREFIXES)
    total_size = sum(src_objects.values())
    print(f"  Found {len(src_objects)} objects, total: {total_size / (1024**3):.2f} GB")

    if test_mode:
        first_key = next(iter(src_objects))
        src_objects = {first_key: src_objects[first_key]}
        print(f"  [TEST MODE] Copying only: {first_key}")

    # ── Phase 2: Detect already-copied objects ──
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Phase 2: Listing destination for resume detection...")
    dst_objects = list_all_objects(dst, DST_BUCKET)
    print(f"  Destination has {len(dst_objects)} existing objects")

    # Split into to-copy and to-skip
    to_copy = []
    skipped = 0
    for key in sorted(src_objects.keys()):
        expected = src_objects[key]
        if key in dst_objects and dst_objects[key] == expected:
            skipped += 1
            continue
        to_copy.append((key, expected))

    total = len(to_copy) + skipped
    print(f"  Skipping {skipped} already-copied objects")
    print(f"  Need to copy {len(to_copy)} objects")

    # ── Phase 3: Parallel copy ──
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Phase 3: Copying with {NUM_WORKERS} parallel workers...")
    lock = threading.Lock()
    counter = {
        "done": 0,
        "skipped": skipped,
        "bytes": 0,
        "failed": [],
        "start": time.time(),
    }

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {
            pool.submit(copy_single_object, key, size, counter, total, lock): key
            for key, size in to_copy
        }
        for future in as_completed(futures):
            # Exceptions already handled inside copy_single_object
            pass

    # ── Phase 4: Final reconciliation ──
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Phase 4: Final reconciliation...")
    dst_objects_final = list_all_objects(dst, DST_BUCKET)

    missing = []
    size_mismatch = []
    for key, expected_size in src_objects.items():
        if key not in dst_objects_final:
            missing.append(key)
        elif dst_objects_final[key] != expected_size:
            size_mismatch.append((key, expected_size, dst_objects_final[key]))

    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    avg_rate = counter["bytes"] / (1024**2) / (time.time() - counter["start"]) if counter["bytes"] > 0 else 0

    # ── Summary ──
    print("\n" + "=" * 70)
    print("TRANSFER SUMMARY")
    print("=" * 70)
    print(f"  Workers:                      {NUM_WORKERS}")
    print(f"  Source objects (excl grail/):  {len(src_objects)}")
    print(f"  Newly copied:                 {counter['done']}")
    print(f"  Skipped (already existed):    {skipped}")
    print(f"  Failed:                       {len(counter['failed'])}")
    print(f"  Bytes transferred:            {counter['bytes'] / (1024**3):.2f} GB")
    print(f"  Avg throughput:               {avg_rate:.1f} MB/s")
    print(f"  Elapsed time:                 {int(hours)}h {int(mins)}m {int(secs)}s")
    print()
    print("RECONCILIATION:")
    print(f"  Missing in destination:       {len(missing)}")
    print(f"  Size mismatches:              {len(size_mismatch)}")

    if not missing and not size_mismatch and not counter["failed"]:
        print("\n  *** ALL FILES VERIFIED SUCCESSFULLY ***")
    else:
        if missing:
            print(f"\n  Missing files ({len(missing)}):")
            for k in missing[:50]:
                print(f"    {k}")
            if len(missing) > 50:
                print(f"    ... and {len(missing) - 50} more")
        if size_mismatch:
            print(f"\n  Size mismatches ({len(size_mismatch)}):")
            for k, exp, got in size_mismatch[:50]:
                print(f"    {k}: expected={exp}, got={got}")
        if counter["failed"]:
            print(f"\n  Failed transfers ({len(counter['failed'])}):")
            for k, sz, err in counter["failed"]:
                print(f"    {k} ({sz} bytes): {err}")

    # Write manifest
    manifest_path = "/ephemeral/r2_copy_manifest.json"
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_bucket": SRC_BUCKET,
        "dest_bucket": DST_BUCKET,
        "workers": NUM_WORKERS,
        "total_source_objects": len(src_objects),
        "copied": counter["done"],
        "skipped": skipped,
        "failed": [{"key": k, "size": s, "error": e} for k, s, e in counter["failed"]],
        "missing": missing,
        "size_mismatches": [{"key": k, "expected": e, "got": g} for k, e, g in size_mismatch],
        "elapsed_seconds": elapsed,
        "avg_throughput_mbps": avg_rate,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to: {manifest_path}")
    print("=" * 70)

    sys.exit(1 if (missing or size_mismatch or counter["failed"]) else 0)


if __name__ == "__main__":
    main()
