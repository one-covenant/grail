#!/usr/bin/env python3
"""Deployment script for parallel nohup training experiments on Lium.

This script orchestrates multi-model GRPO experiments across Lium pods:
1. Creates pods based on experiment configuration
2. Runs nohup training scripts on each pod (4 seeds per pod)
3. Monitors completion
4. Downloads artifacts
5. Uploads to R2

Usage:
    python deploy_parallel.py --config test_qwen_0.5b
    python deploy_parallel.py --config multi_model
    python deploy_parallel.py --config multi_model --deploy-only
    python deploy_parallel.py --destroy
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from experiment_configs import get_config, list_configs
from lium_manager import LiumInfra
from nohup_experiment_runner import ExperimentConfig, NohupExperimentRunner

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# Model configuration mapping for nohup experiments
# Maps pod names to (model_id, num_iterations, wandb config)
MODEL_CONFIGS = {
    # Test config
    "test-qwen-0.5b": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "test,qwen-0.5b,lium",
    },
    # Full multi-model configs
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
    "qwen2.5-7b-iter1": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "qwen-7b,iter1,lium",
    },
    "qwen2.5-7b-iter8": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_iterations": 8,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "qwen-7b,iter8,lium",
    },
    "qwen2.5-7b-iter16": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "num_iterations": 16,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "qwen-7b,iter16,lium",
    },
    "llama3.2-1b-iter1": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "llama-1b,iter1,lium",
    },
    "llama3.2-3b-iter1": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "llama-3b,iter1,lium",
    },
    "gemma3-1b-iter1": {
        "model": "google/gemma-3-1b-it",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "gemma-1b,iter1,lium",
    },
    "gemma3-4b-iter1": {
        "model": "google/gemma-3-4b-it",
        "num_iterations": 1,
        "wandb_project": "grail-lium-sweep",
        "wandb_tags": "gemma-4b,iter1,lium",
    },
}


async def run_pod_experiment(
    pod_name: str,
    pod_info: dict,
    r2_config: dict[str, str],
    local_code_path: Path,
    local_env_path: Path,
    dataset: str = "math",
    eval_every: int = 40,
    sync_code: bool = True,
    setup_env: bool = True,
) -> bool:
    """Run experiment on a single pod.

    Args:
        pod_name: Name of the pod
        pod_info: Pod information from lium_manager
        r2_config: R2 configuration dict
        local_code_path: Local path to grail repository
        local_env_path: Local path to .env file
        dataset: Dataset to train on (default: math)
        eval_every: Evaluation interval (default: 40)
        sync_code: Whether to sync code (default: True)
        setup_env: Whether to setup environment (default: True)

    Returns:
        True if experiment succeeded, False otherwise
    """
    # Get model config for this pod
    if pod_name not in MODEL_CONFIGS:
        logger.error(f"No model configuration found for pod: {pod_name}")
        return False

    model_config = MODEL_CONFIGS[pod_name]

    # Create experiment config with W&B settings
    config = ExperimentConfig(
        name=pod_name,
        model_id=model_config["model"],
        num_iterations=model_config["num_iterations"],
        dataset=dataset,
        eval_every=eval_every,
        wandb_project=model_config.get("wandb_project", "grail-lium-sweep"),
        wandb_tags=model_config.get("wandb_tags", ""),
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment on pod: {pod_name}")
    logger.info(f"  Model: {config.model_id}")
    logger.info(f"  Num Iterations: {config.num_iterations}")
    logger.info(f"  Dataset: {config.dataset}")
    logger.info(f"{'='*80}\n")

    # Create runner
    ssh_info = pod_info["ssh"]
    runner = NohupExperimentRunner(
        ssh_host=ssh_info["host"],
        ssh_port=ssh_info["port"],
        r2_config=r2_config,
    )

    # Run experiment
    success = await runner.run_experiment(
        config=config,
        local_code_path=local_code_path,
        local_env_path=local_env_path,
        sync_code=sync_code,
        setup_env=setup_env,
        upload_to_r2=True,
        cleanup_local=False,  # Keep local copies for debugging
    )

    if success:
        logger.info(f"✓ Experiment {pod_name} completed successfully")
    else:
        logger.error(f"✗ Experiment {pod_name} failed")

    return success


async def deploy_and_run(
    config_name: str,
    dataset: str = "math",
    eval_every: int = 40,
    deploy_only: bool = False,
    no_deploy: bool = False,
    sync_code: bool = True,
    setup_env: bool = True,
    state_file: str = ".lium_state.json",
) -> bool:
    """Deploy pods and run experiments.

    Args:
        config_name: Name of experiment configuration
        dataset: Dataset to train on (default: math)
        eval_every: Evaluation interval (default: 40)
        deploy_only: Only deploy pods, don't run experiments (default: False)
        no_deploy: Skip pod deployment, use existing pods (default: False)
        sync_code: Whether to sync code to pods (default: True)
        setup_env: Whether to setup environment on pods (default: True)
        state_file: Path to lium state file (default: .lium_state.json)

    Returns:
        True if all experiments succeeded, False otherwise
    """
    # Load experiment configuration
    logger.info(f"Loading configuration: {config_name}")
    pods, _ = get_config(config_name)

    # Initialize lium manager
    infra = LiumInfra(state_file=state_file)

    # Deploy pods (unless skipped)
    if not no_deploy:
        logger.info(f"\n{'='*80}")
        logger.info(f"Deploying {len(pods)} pods...")
        logger.info(f"{'='*80}\n")

        deployed_pods = infra.apply(pods)

        if not deployed_pods:
            logger.error("Failed to deploy pods")
            return False

        logger.info(f"\n✓ Deployed {len(deployed_pods)} pods successfully")

        # Print pod information
        for pod_name, pod_info in deployed_pods.items():
            ssh_info = pod_info["ssh"]
            logger.info(f"\n  Pod: {pod_name}")
            logger.info(f"    SSH: {ssh_info['host']}:{ssh_info['port']}")
            logger.info(f"    GPUs: {pod_info['spec']['gpu_count']} × {pod_info['spec']['gpu_type']}")

        if deploy_only:
            logger.info("\n✓ Deploy-only mode: pods created, skipping experiments")
            return True

    else:
        logger.info("Skipping pod deployment (using existing pods)")
        deployed_pods = {pod.name: infra.get_pod(pod.name) for pod in pods}

    # Load R2 configuration from environment
    r2_config = {
        "bucket_id": os.getenv("R2_BUCKET_ID"),
        "account_id": os.getenv("R2_ACCOUNT_ID"),
        "access_key": os.getenv("R2_WRITE_ACCESS_KEY_ID"),
        "secret_key": os.getenv("R2_WRITE_SECRET_ACCESS_KEY"),
    }

    # Validate R2 config
    if not all(r2_config.values()):
        logger.error("Missing R2 configuration in environment variables")
        logger.error("Required: R2_BUCKET_ID, R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY")
        return False

    # Paths
    local_code_path = PROJECT_ROOT
    local_env_path = PROJECT_ROOT / ".env"

    # Run experiments on all pods in parallel
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiments on {len(deployed_pods)} pods in parallel...")
    logger.info(f"{'='*80}\n")

    tasks = [
        run_pod_experiment(
            pod_name=pod_name,
            pod_info=pod_info,
            r2_config=r2_config,
            local_code_path=local_code_path,
            local_env_path=local_env_path,
            dataset=dataset,
            eval_every=eval_every,
            sync_code=sync_code,
            setup_env=setup_env,
        )
        for pod_name, pod_info in deployed_pods.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check results
    success_count = sum(1 for r in results if r is True)
    failed_count = len(results) - success_count

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {success_count} succeeded, {failed_count} failed")
    logger.info(f"{'='*80}\n")

    return failed_count == 0


def destroy_pods(state_file: str = ".lium_state.json") -> bool:
    """Destroy all managed pods.

    Args:
        state_file: Path to lium state file (default: .lium_state.json)

    Returns:
        True if all pods destroyed successfully, False otherwise
    """
    logger.info("Destroying all managed pods...")

    infra = LiumInfra(state_file=state_file)
    success = infra.destroy()

    if success:
        logger.info("✓ All pods destroyed successfully")
    else:
        logger.error("✗ Failed to destroy some pods")

    return success


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy and run parallel nohup training experiments on Lium"
    )

    # Main commands
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=str,
        choices=list_configs(),
        help="Experiment configuration to run",
    )
    group.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations",
    )
    group.add_argument(
        "--destroy",
        action="store_true",
        help="Destroy all managed pods and exit",
    )

    # Experiment options
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to train on (default: math)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=40,
        help="Evaluation interval in steps (default: 40)",
    )

    # Deployment options
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="Only deploy pods, don't run experiments",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip pod deployment, use existing pods",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip code synchronization",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip environment setup",
    )

    # State management
    parser.add_argument(
        "--state-file",
        type=str,
        default=".lium_state.json",
        help="Path to lium state file (default: .lium_state.json)",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Configure logging with immediate flush for nohup visibility
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Reconfigure if already set
    )
    # Set logging handlers to flush immediately (critical for nohup)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.INFO)
    # Force unbuffered stdout/stderr for nohup
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    # List configurations
    if args.list_configs:
        print("\nAvailable configurations:")
        for config_name in list_configs():
            print(f"  - {config_name}")
        return 0

    # Destroy pods
    if args.destroy:
        success = destroy_pods(state_file=args.state_file)
        return 0 if success else 1

    # Require config for deployment
    if not args.config:
        print("Error: --config required (or use --list-configs, --destroy)")
        return 1

    # Deploy and run
    success = await deploy_and_run(
        config_name=args.config,
        dataset=args.dataset,
        eval_every=args.eval_every,
        deploy_only=args.deploy_only,
        no_deploy=args.no_deploy,
        sync_code=not args.no_sync,
        setup_env=not args.no_setup,
        state_file=args.state_file,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
