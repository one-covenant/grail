#!/usr/bin/env python3
"""
Extract sparsity from SFT delta files in R2.
Sparsity is pre-computed in the metadata.
"""

import os
import sys
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import boto3
from botocore.config import Config
import torch
import pandas as pd

# R2 credentials
R2_ACCOUNT_ID = "91561e574629960f78e985efa5a37e59"
R2_BUCKET = "91561e574629960f78e985efa5a37e59"
R2_ACCESS_KEY = "5961758bc74f3554506f2ba05390a6dd"
R2_SECRET_KEY = "0a1fbf3a324e889d44d2a235eb58de661758aeba08a0c23d8f744dfd9fc3566a"
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# SFT experiments
SFT_EXPERIMENTS = [
    ("qwen2.5-1.5b-sft-math-lr3e-06", 3e-6),
    ("qwen2.5-1.5b-sft-math-lr2e-05", 2e-5),
]

thread_local = threading.local()


def get_s3_client():
    """Get thread-local S3 client."""
    if not hasattr(thread_local, "s3"):
        thread_local.s3 = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            config=Config(signature_version='s3v4'),
        )
    return thread_local.s3


def list_delta_files(experiment: str) -> dict[int, list[int]]:
    """List all delta files by seed."""
    s3 = get_s3_client()
    paginator = s3.get_paginator('list_objects_v2')

    by_seed = {}
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=f'{experiment}/'):
        for obj in page.get('Contents', []):
            match = re.search(r'seed(\d+)/deltas/delta_(\d+)\.pt', obj['Key'])
            if match:
                seed = int(match.group(1))
                step = int(match.group(2))
                if seed not in by_seed:
                    by_seed[seed] = []
                by_seed[seed].append(step)

    for seed in by_seed:
        by_seed[seed] = sorted(by_seed[seed])

    return by_seed


def extract_sparsity(experiment: str, lr: float, seed: int, step: int) -> dict | None:
    """Download delta file and extract sparsity from metadata."""
    s3 = get_s3_client()
    key = f"{experiment}/seed{seed}/deltas/delta_{step:06d}.pt"

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as tmp:
        try:
            s3.download_file(R2_BUCKET, key, tmp.name)
            delta = torch.load(tmp.name, map_location='cpu', weights_only=False)

            metadata = delta.get('metadata', {})
            sparsity = metadata.get('sparsity', 0) * 100  # Convert to percentage

            return {
                'model_family': 'Qwen',
                'model_size': '1.5B',
                'experiment_type': 'SFT',
                'seed': seed,
                'step': step,
                'k': 1,
                'sparsity': sparsity,
                'learning_rate': lr,
                'iteration_num': 1,
            }
        except Exception as e:
            print(f"Error processing {key}: {e}")
            return None


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    all_results = []

    for experiment, lr in SFT_EXPERIMENTS:
        print(f"\n=== Processing {experiment} (lr={lr:.0e}) ===")

        by_seed = list_delta_files(experiment)
        print(f"Seeds: {list(by_seed.keys())}")

        for seed, steps in sorted(by_seed.items()):
            print(f"  Seed {seed}: {len(steps)} steps", flush=True)

            # Process in parallel
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {
                    executor.submit(extract_sparsity, experiment, lr, seed, step): step
                    for step in steps
                }

                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)
                    completed += 1
                    if completed % 20 == 0:
                        print(f"    Processed {completed}/{len(steps)}", flush=True)

            seed_results = [r for r in all_results if r['seed'] == seed and r['learning_rate'] == lr]
            mean_sparsity = sum(r['sparsity'] for r in seed_results) / len(seed_results) if seed_results else 0
            print(f"    Done. Mean sparsity: {mean_sparsity:.2f}%")

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(['learning_rate', 'seed', 'step'])
        output_path = output_dir / "sparsity_sft.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} rows to {output_path}")

        # Print summary
        print("\n=== Summary ===")
        for lr in sorted(df['learning_rate'].unique()):
            subset = df[df['learning_rate'] == lr]
            print(f"lr={lr:.0e}: {len(subset)} samples, mean sparsity={subset['sparsity'].mean():.2f}% ± {subset['sparsity'].std():.2f}%")
    else:
        print("No results computed")


if __name__ == "__main__":
    main()
