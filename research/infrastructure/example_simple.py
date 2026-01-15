#!/usr/bin/env python3
"""Simple example: Deploy a single pod and run a quick experiment.

This example demonstrates the basic usage pattern for the Lium infrastructure
system. It deploys a single pod and runs a short training experiment.

Usage:
    python example_simple.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from experiment_runner import ExperimentConfig, run_experiments_on_pod
from lium_manager import LiumInfra, PodSpec


def main():
    """Run a simple example experiment."""

    print("=" * 80)
    print("SIMPLE EXAMPLE: Single Pod, Single Experiment")
    print("=" * 80)
    print()

    # Initialize Lium infrastructure
    infra = LiumInfra(state_file=".lium_state_example.json")

    # Define a single pod
    pod_spec = PodSpec(
        name="example-pod",
        gpu_type="A100",
        gpu_count=8,
        min_upload_mbps=500,
        min_download_mbps=500,
        ttl_hours=2,  # Auto-terminate after 2 hours
    )

    # Define a quick experiment
    experiment = ExperimentConfig(
        name="quick_test",
        dataset="gsm8k",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        learning_rate=3e-6,
        batch_size=4,
        grad_accum_steps=128,
        total_steps=5,  # Just 5 steps for quick testing
        eval_every=5,
    )

    # Step 1: Deploy the pod
    print("\n[1/3] Deploying pod...")
    infra.apply([pod_spec])

    # Step 2: Get pod info
    pod_info = infra.get_pod_info("example-pod")
    if not pod_info:
        print("Error: Failed to deploy pod")
        return 1

    print("\nPod deployed successfully:")
    print(f"  SSH: {pod_info['ssh']['host']}:{pod_info['ssh']['port']}")
    print(
        f"  Bandwidth: ↑{pod_info['bandwidth']['upload']:.0f} ↓{pod_info['bandwidth']['download']:.0f} Mbps"
    )

    # Step 3: Run the experiment
    print("\n[2/3] Running experiment...")

    # Auto-detect project root (go up from this script)
    local_code_path = Path(__file__).parent.parent.parent

    # Run experiment
    asyncio.run(
        run_experiments_on_pod(
            pod_info=pod_info,
            experiments=[experiment],
            local_code_path=local_code_path,
            sync_code=True,
            setup_env=True,
        )
    )

    # Step 4: Cleanup
    print("\n[3/3] Cleaning up...")
    response = input("Destroy the pod? (y/N): ")
    if response.lower() == "y":
        infra.destroy()
        print("✅ Pod destroyed")
    else:
        print("⚠️  Pod still running. To destroy later, run:")
        print("    python deploy.py --destroy --state-file .lium_state_example.json")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
