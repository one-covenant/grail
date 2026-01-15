#!/usr/bin/env python3
"""Main deployment script for running distributed experiments on Lium.

Usage:
    # List available configurations
    python deploy.py --list-configs

    # Deploy and run a predefined configuration
    python deploy.py --config lr_sweep

    # Run experiments without deploying new pods (use existing)
    python deploy.py --config lr_sweep --no-deploy

    # Deploy pods only (don't run experiments)
    python deploy.py --config lr_sweep --deploy-only

    # Cleanup all managed pods
    python deploy.py --destroy

    # Inspect executor specs
    python deploy.py --inspect-executors --gpu-type A100
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from experiment_configs import get_config, list_configs
from experiment_runner import run_experiments_parallel
from lium_manager import LiumInfra


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy and run distributed experiments on Lium infrastructure"
    )

    # Configuration selection
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration name to deploy and run",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available predefined configurations",
    )

    # Deployment options
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Skip pod deployment (use existing pods)",
    )

    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="Only deploy pods, don't run experiments",
    )

    parser.add_argument(
        "--destroy",
        action="store_true",
        help="Destroy all managed pods and exit",
    )

    # Code sync options
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

    # Infrastructure options
    parser.add_argument(
        "--state-file",
        type=str,
        default=".lium_state.json",
        help="Path to Lium state file (default: .lium_state.json)",
    )

    parser.add_argument(
        "--local-code-path",
        type=str,
        default=None,
        help="Local code path for rsync (only needed if --use-rsync)",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file to copy to pods (default: <repo>/.env)",
    )

    parser.add_argument(
        "--git-repo",
        type=str,
        default="https://github.com/manifold-inc/grail.git",
        help="Git repository URL (default: manifold-inc/grail)",
    )

    parser.add_argument(
        "--git-branch",
        type=str,
        default="main",
        help="Git branch to checkout (default: main)",
    )

    parser.add_argument(
        "--use-rsync",
        action="store_true",
        help="Use rsync instead of git clone (requires --local-code-path)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of in parallel on each pod",
    )

    # Debugging
    parser.add_argument(
        "--inspect-executors",
        action="store_true",
        help="Inspect executor specs and bandwidth info",
    )

    parser.add_argument(
        "--gpu-type",
        type=str,
        default="A100",
        help="GPU type for executor inspection (default: A100)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List configurations (no Lium connection needed)
    if args.list_configs:
        print("Available configurations:")
        for name in list_configs():
            print(f"  - {name}")
        return 0

    # Initialize Lium infrastructure (only when needed)
    infra = LiumInfra(state_file=args.state_file)

    # Inspect executors
    if args.inspect_executors:
        print(f"\n=== Executor Specs ({args.gpu_type}) ===\n")
        infra.inspect_executor_specs(gpu_type=args.gpu_type)
        print(f"\n=== Executors with Bandwidth Info ===\n")
        infra.list_executors_with_bandwidth(
            gpu_type=args.gpu_type,
            min_upload=0,
            min_download=0,
        )
        return 0

    # Destroy pods
    if args.destroy:
        print("\n🗑️  Destroying all managed pods...\n")
        infra.destroy()
        print("\n✅ All pods destroyed\n")
        return 0

    # Validate config argument
    if not args.config:
        print("Error: --config is required (use --list-configs to see available options)")
        return 1

    # Get configuration
    print(f"\n📋 Loading configuration: {args.config}\n")
    try:
        pod_specs, pod_experiments = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Configuration loaded:")
    print(f"  - Pods: {len(pod_specs)}")
    print(f"  - Total experiments: {sum(len(exps) for exps in pod_experiments.values())}")

    # Deploy pods
    if not args.no_deploy:
        print(f"\n{'='*80}")
        print("🚀 DEPLOYING PODS")
        print(f"{'='*80}\n")
        infra.apply(pod_specs)
        print(f"\n✅ Pod deployment complete\n")

    if args.deploy_only:
        print("Pod deployment complete (--deploy-only specified, skipping experiments)")
        return 0

    # Get pod info for running experiments
    managed_pods = infra.list_pods()
    pod_info_map = {}

    for pod_name, experiments in pod_experiments.items():
        if pod_name not in managed_pods:
            print(f"⚠️  Warning: Pod '{pod_name}' not found in managed pods, skipping experiments")
            continue
        pod_info_map[pod_name] = (managed_pods[pod_name], experiments)

    if not pod_info_map:
        print("Error: No pods available for running experiments")
        return 1

    # Determine local code path (only needed for rsync)
    local_code_path = None
    if args.use_rsync:
        if args.local_code_path:
            local_code_path = Path(args.local_code_path)
        else:
            # Auto-detect: go up from this script to project root
            script_dir = Path(__file__).parent
            local_code_path = script_dir.parent.parent
            print(f"Auto-detected local code path: {local_code_path}")

        if not local_code_path.exists():
            print(f"Error: Local code path does not exist: {local_code_path}")
            return 1

    # Determine .env file path
    local_env_path = None
    if args.env_file:
        local_env_path = Path(args.env_file)
        if not local_env_path.exists():
            print(f"⚠️  Warning: .env file not found at {local_env_path}")
    else:
        # Try default location
        script_dir = Path(__file__).parent
        default_env = script_dir.parent.parent / ".env"
        if default_env.exists():
            local_env_path = default_env
            print(f"Found .env file at: {local_env_path}")

    # Run experiments
    print(f"\n{'='*80}")
    print("🧪 RUNNING EXPERIMENTS")
    print(f"{'='*80}")
    if args.use_rsync:
        print(f"Code deployment: rsync from {local_code_path}")
    else:
        print(f"Code deployment: git clone {args.git_repo} (branch: {args.git_branch})")
    print(f"Environment setup: {'Yes' if not args.no_setup else 'No'}")
    print(f"Execution mode: {'Sequential' if args.sequential else 'Parallel (per pod)'}")
    if local_env_path:
        print(f".env file: {local_env_path}")
    print()

    asyncio.run(
        run_experiments_parallel(
            pod_experiments=pod_info_map,
            local_code_path=local_code_path,
            local_env_path=local_env_path,
            git_repo=args.git_repo,
            git_branch=args.git_branch,
            use_git=not args.use_rsync,
            setup_env=not args.no_setup,
            run_parallel_per_pod=not args.sequential,
        )
    )

    print(f"\n{'='*80}")
    print("✅ ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
