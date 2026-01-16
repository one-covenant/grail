#!/usr/bin/env python3
"""Deploy only the Qwen models."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from lium_manager import LiumInfra, PodSpec
from nohup_experiment_runner import NohupExperimentRunner

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_CONFIGS = {
    "qwen2.5-0.5b-iter1": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "qwen-0.5b,iter1,lium",
    },
    "qwen2.5-1.5b-iter1": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "qwen-1.5b,iter1,lium",
    },
}


async def main():
    lium = LiumInfra(state_file=".lium_state_qwen_only.json")

    pods = [
        PodSpec(name=name, gpu_type="A100", gpu_count=8, ttl_hours=124)
        for name in MODEL_CONFIGS.keys()
    ]

    print("=" * 70)
    print("Creating Qwen pods...")
    print("=" * 70)

    # Use apply method (synchronous)
    created = lium.apply(pods)

    # Run experiments
    tasks = []
    for pod_name, cfg in MODEL_CONFIGS.items():
        pod_info = created.get(pod_name)
        if not pod_info:
            print(f"❌ Pod {pod_name} not available")
            continue

        ssh_host = pod_info["ssh"]["host"]
        ssh_port = pod_info["ssh"]["port"]

        print(f"Starting experiment on {pod_name} ({ssh_host}:{ssh_port})")

        runner = NohupExperimentRunner(
            pod_name=pod_name,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            model_id=cfg["model"],
            dataset="math",
            eval_every=40,
            num_iterations=cfg["num_iterations"],
            wandb_project=cfg["wandb_project"],
            wandb_tags=cfg["wandb_tags"],
        )

        tasks.append(runner.run_experiment())

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pod_name, result in zip(MODEL_CONFIGS.keys(), results, strict=False):
            if isinstance(result, Exception):
                print(f"❌ {pod_name} failed: {result}")
            else:
                print(f"✅ {pod_name} completed")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
