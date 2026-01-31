#!/usr/bin/env python3
"""Download missing Qwen 7B and Gemma 4B experiments."""

import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from botocore.config import Config
from botocore.session import get_session

R2_ACCOUNT_ID = "91561e574629960f78e985efa5a37e59"
R2_BUCKET = "91561e574629960f78e985efa5a37e59"
R2_ACCESS_KEY = "5961758bc74f3554506f2ba05390a6dd"
R2_SECRET_KEY = "0a1fbf3a324e889d44d2a235eb58de661758aeba08a0c23d8f744dfd9fc3566a"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

LOCAL_BASE = Path("/root/grail/research/sparsity_analysis")
MIN_CHECKPOINT = 350
NUM_WORKERS = 16

EXPERIMENTS = [
    ("qwen2.5-7b-grpo-math-lr3e-06/", "experiments/qwen2.5-7b-grpo-math-lr3e-06/", "Qwen 7B GRPO"),
    ("experiments/gemma3-4b-iter1/", "experiments/gemma3-4b-iter1/", "Gemma 4B iter1"),
]

thread_local = threading.local()

def get_s3_client():
    if not hasattr(thread_local, "s3"):
        session = get_session()
        config = Config(s3={"addressing_style": "path"}, region_name="auto", max_pool_connections=50)
        thread_local.s3 = session.create_client("s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY, config=config)
    return thread_local.s3

def create_s3_client():
    session = get_session()
    config = Config(s3={"addressing_style": "path"}, region_name="auto")
    return session.create_client("s3", endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY, aws_secret_access_key=R2_SECRET_KEY, config=config)

def list_objects(s3, prefix):
    objects = []
    token = None
    while True:
        kwargs = {"Bucket": R2_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        objects.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return objects

def should_download(key):
    if "delta_" in key and key.endswith(".pt"):
        return True
    if "/logs/" in key or key.endswith(".log") or key.endswith(".json"):
        return True
    if "metadata" in key.lower():
        return True
    checkpoint_match = re.search(r"checkpoint-(\d+)", key)
    if checkpoint_match:
        return int(checkpoint_match.group(1)) >= MIN_CHECKPOINT
    return False

@dataclass
class DownloadTask:
    key: str
    local_path: Path
    size: int

def download_file(task):
    try:
        task.local_path.parent.mkdir(parents=True, exist_ok=True)
        if task.local_path.exists() and task.local_path.stat().st_size == task.size:
            return True, 0
        s3 = get_s3_client()
        response = s3.get_object(Bucket=R2_BUCKET, Key=task.key)
        with open(task.local_path, "wb") as f:
            for chunk in response["Body"].iter_chunks(chunk_size=8*1024*1024):
                f.write(chunk)
        return True, task.size
    except Exception as e:
        print(f"\nERROR downloading {task.key}: {e}", flush=True)
        return False, 0

class ProgressTracker:
    def __init__(self, total_files, total_bytes):
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.completed_files = 0
        self.downloaded_bytes = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, success, bytes_downloaded):
        with self.lock:
            self.completed_files += 1
            if success:
                self.downloaded_bytes += bytes_downloaded
            else:
                self.errors += 1
            if self.completed_files % 10 == 0 or self.completed_files == self.total_files:
                self._print_progress()

    def _print_progress(self):
        elapsed = time.time() - self.start_time
        pct = self.completed_files * 100 // self.total_files
        gb = self.downloaded_bytes / 1e9
        speed = self.downloaded_bytes / elapsed / 1e6 if elapsed > 0 else 0
        if self.downloaded_bytes > 0 and elapsed > 0:
            bytes_remaining = self.total_bytes - self.downloaded_bytes
            eta_min = int(bytes_remaining / (self.downloaded_bytes / elapsed) // 60)
            eta_str = f"ETA={eta_min}m"
        else:
            eta_str = "ETA=..."
        print(f"\rProgress: {self.completed_files}/{self.total_files} ({pct}%) | "
              f"Downloaded: {gb:.1f}GB | Speed: {speed:.1f}MB/s | "
              f"Errors: {self.errors} | {eta_str}    ", end="", flush=True)

def download_experiment(r2_prefix, local_prefix, description):
    print(f"\n{'='*70}")
    print(f"Downloading: {description}")
    print(f"  R2: {r2_prefix}")
    print(f"  Local: {LOCAL_BASE / local_prefix}")
    print("="*70)

    s3 = create_s3_client()
    print("Listing objects...")
    objects = list_objects(s3, r2_prefix)
    print(f"Found {len(objects)} total objects")

    tasks = []
    for obj in objects:
        key = obj["Key"]
        if should_download(key):
            relative_path = key[len(r2_prefix):]
            local_path = LOCAL_BASE / local_prefix / relative_path
            tasks.append(DownloadTask(key=key, local_path=local_path, size=obj["Size"]))

    deltas = [t for t in tasks if "delta_" in t.key]
    checkpoints = [t for t in tasks if "checkpoint-" in t.key]
    logs = [t for t in tasks if t not in deltas and t not in checkpoints]

    total_size = sum(t.size for t in tasks)
    print(f"  Deltas: {len(deltas)} files")
    print(f"  Checkpoints (>={MIN_CHECKPOINT}): {len(checkpoints)} files")
    print(f"  Logs/metadata: {len(logs)} files")
    print(f"  Total size: {total_size/1e9:.2f} GB")

    need_download = []
    already_size = 0
    for task in tasks:
        if task.local_path.exists() and task.local_path.stat().st_size == task.size:
            already_size += task.size
        else:
            need_download.append(task)

    if already_size > 0:
        print(f"  Already downloaded: {len(tasks) - len(need_download)} files ({already_size/1e9:.1f} GB)")

    if not need_download:
        print("  All files already exist, skipping!")
        return len(tasks)

    remaining_size = sum(t.size for t in need_download)
    print(f"  Need to download: {len(need_download)} files ({remaining_size/1e9:.1f} GB)")
    print()

    progress = ProgressTracker(len(need_download), remaining_size)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_file, task): task for task in need_download}
        for future in as_completed(futures):
            success, size = future.result()
            progress.update(success, size)

    print()
    print(f"  Complete! Downloaded {progress.downloaded_bytes/1e9:.2f} GB, Errors: {progress.errors}")
    return len(tasks)

def main():
    print("="*70)
    print("Downloading Qwen 7B and Gemma 4B")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Checkpoint filter: >= {MIN_CHECKPOINT}")
    print("="*70)

    total_files = 0
    for r2_prefix, local_prefix, description in EXPERIMENTS:
        count = download_experiment(r2_prefix, local_prefix, description)
        total_files += count

    print(f"\n{'='*70}")
    print(f"DONE! Total files processed: {total_files}")
    print("="*70)

if __name__ == "__main__":
    main()
