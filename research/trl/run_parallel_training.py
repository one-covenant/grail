#!/usr/bin/env python3
"""Parallel training launcher for TRL GRPO with multiple seeds.

Runs 4 parallel training instances, each with:
- Dedicated VLLM server GPU
- Dedicated training GPU
- Unique random seed
- Unique port

NOTE: This script is designed to run in nohup mode. Use:
    ./run_parallel_training_nohup.sh [dataset] [eval_every]

GPU allocation (8 GPUs total):
- Instance 0: VLLM GPU 0, Training GPU 1, Port 8000, Seed 42
- Instance 1: VLLM GPU 2, Training GPU 3, Port 8001, Seed 1337
- Instance 2: VLLM GPU 4, Training GPU 5, Port 8002, Seed 2024
- Instance 3: VLLM GPU 6, Training GPU 7, Port 8003, Seed 9999

Usage:
    python run_parallel_training.py --dataset gsm8k
    python run_parallel_training.py --dataset math --eval-every 50
"""

from __future__ import annotations

# Force unbuffered output for nohup mode
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


# Fixed seeds for reproducibility across experiments
SEEDS = [42, 1337, 2024, 9999]

# GPU allocation: (vllm_gpu, training_gpu)
GPU_PAIRS = [
    (0, 1),  # Instance 0
    (2, 3),  # Instance 1
    (4, 5),  # Instance 2
    (6, 7),  # Instance 3
]

# Base port for VLLM servers
BASE_PORT = 8000

# Base port for NCCL group coordination (each instance needs unique port)
BASE_GROUP_PORT = 51216


class ProcessManager:
    """Manages VLLM servers and training processes."""

    def __init__(
        self,
        dataset: str,
        eval_every: int,
        model_id: str,
        num_iterations: int = 1,
        wandb_project: str | None = None,
        wandb_tags: str | None = None,
    ):
        self.dataset = dataset
        self.eval_every = eval_every
        self.model_id = model_id
        self.num_iterations = num_iterations
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags
        self.vllm_processes: list[subprocess.Popen] = []
        self.training_processes: list[subprocess.Popen] = []
        self.log_dir = Path("./logs/parallel_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals gracefully."""
        print(f"\n\n🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown_all()
        sys.exit(0)

    def start_vllm_server(self, instance_id: int, gpu_id: int, port: int) -> subprocess.Popen:
        """Start VLLM server on specified GPU and port.

        Args:
            instance_id: Instance identifier (0-3)
            gpu_id: GPU ID for VLLM server
            port: Port number for server

        Returns:
            Popen object for the server process
        """
        log_file = self.log_dir / f"vllm_instance{instance_id}_gpu{gpu_id}_port{port}.log"

        # Use tools/vllm-server venv for vLLM (isolated environment with compatible versions)
        repo_root = Path(__file__).parent.parent.parent
        trl_path = repo_root / "tools" / "vllm-server" / ".venv" / "bin" / "trl"
        cmd = [
            str(trl_path), "vllm-serve",
            "--model", self.model_id,
            "--port", str(port),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Unique cache directories per instance to avoid race conditions
        cache_base = f"/tmp/vllm_cache_instance{instance_id}"
        env["VLLM_CACHE_ROOT"] = cache_base
        env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_base}/inductor"
        env["TORCH_COMPILE_CACHE_DIR"] = f"{cache_base}/torch_compile"

        print(f"[Instance {instance_id}] Starting VLLM server:")
        print(f"  GPU: {gpu_id}, Port: {port}, Cache: {cache_base}, Log: {log_file}")

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group for clean shutdown
            )

        self.vllm_processes.append(process)
        return process

    def wait_for_vllm_ready(
        self, port: int, timeout: int = 60, instance_id: int = 0, process: subprocess.Popen = None
    ) -> bool:
        """Wait for VLLM server to be ready.

        Uses fixed wait time + process alive check (TRL's vllm-serve doesn't expose /v1/models).

        Args:
            port: Port to check
            timeout: Wait time in seconds (default 60s for model loading)
            instance_id: Instance identifier for logging
            process: The vLLM subprocess to check if still alive

        Returns:
            True if server is ready, False if process died
        """
        print(f"[Instance {instance_id}] Waiting {timeout}s for VLLM server on port {port}...")
        time.sleep(timeout)

        # Check if process is still alive
        if process and process.poll() is not None:
            print(f"[Instance {instance_id}] ✗ VLLM server died during startup (exit: {process.returncode})")
            return False

        print(f"[Instance {instance_id}] ✓ VLLM server ready!")
        return True

    def start_training(
        self,
        instance_id: int,
        gpu_id: int,
        port: int,
        seed: int,
    ) -> subprocess.Popen:
        """Start training script.

        Args:
            instance_id: Instance identifier (0-3)
            gpu_id: GPU ID for training
            port: VLLM server port to connect to
            seed: Random seed

        Returns:
            Popen object for the training process
        """
        log_file = self.log_dir / f"training_instance{instance_id}_gpu{gpu_id}_seed{seed}.log"

        # Use local .venv python
        python_path = Path(__file__).parent / ".venv" / "bin" / "python"
        group_port = BASE_GROUP_PORT + instance_id  # Unique port per instance
        cmd = [
            str(python_path), "train_trl_grpo.py",
            "--dataset", self.dataset,
            "--eval-every", str(self.eval_every),
            "--num-iterations", str(self.num_iterations),
            "--seed", str(seed),
            "--vllm-port", str(port),
            "--group-port", str(group_port),
            "--run-suffix", f"instance{instance_id}_seed{seed}",
        ]
        # Add W&B args if provided (CLI takes precedence over env vars)
        if self.wandb_project:
            cmd.extend(["--wandb-project", self.wandb_project])
        if self.wandb_tags:
            cmd.extend(["--wandb-tags", self.wandb_tags])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[Instance {instance_id}] Starting training:")
        print(f"  GPU: {gpu_id}, Seed: {seed}, VLLM Port: {port}, Group Port: {group_port}, Log: {log_file}")

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

        self.training_processes.append(process)
        return process

    def monitor_processes(self) -> None:
        """Monitor all processes and report status."""
        print("\n" + "=" * 80)
        print("📊 MONITORING ACTIVE PROCESSES")
        print("=" * 80)

        while True:
            # Check if any training process has finished
            all_finished = True
            any_failed = False

            for i, proc in enumerate(self.training_processes):
                poll = proc.poll()
                if poll is None:
                    all_finished = False
                elif poll != 0:
                    any_failed = True
                    print(f"\n⚠️  [Instance {i}] Training process failed with code {poll}")

            if all_finished:
                if any_failed:
                    print("\n❌ Some processes failed. Check logs in ./logs/parallel_training/")
                else:
                    print("\n✅ All training processes completed successfully!")
                break

            # Status update every 60 seconds
            time.sleep(60)

            # Print brief status
            running = sum(1 for p in self.training_processes if p.poll() is None)
            print(f"[{time.strftime('%H:%M:%S')}] Training processes: {running}/{len(self.training_processes)} running")

    def shutdown_all(self) -> None:
        """Shutdown all processes gracefully."""
        print("\n🧹 Shutting down all processes...")

        # Shutdown training processes first
        for i, proc in enumerate(self.training_processes):
            if proc.poll() is None:
                print(f"  Stopping training instance {i}...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception as e:
                    print(f"    Force killing: {e}")
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        # Shutdown VLLM servers
        for i, proc in enumerate(self.vllm_processes):
            if proc.poll() is None:
                print(f"  Stopping VLLM instance {i}...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception as e:
                    print(f"    Force killing: {e}")
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        print("✓ All processes stopped")

    def run(self) -> None:
        """Run the full parallel training pipeline."""
        print("=" * 80)
        print("🚀 PARALLEL TRAINING LAUNCHER")
        print("=" * 80)
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model_id}")
        print(f"Num Iterations: {self.num_iterations}")
        print(f"Instances: {len(SEEDS)}")
        print(f"Seeds: {SEEDS}")
        print(f"GPU pairs: {GPU_PAIRS}")
        print(f"Ports: {[BASE_PORT + i for i in range(len(SEEDS))]}")
        print(f"Logs: {self.log_dir}")
        print("=" * 80)

        try:
            # Phase 1: Start all VLLM servers
            print("\n📡 Phase 1: Starting VLLM servers...")
            for i, (vllm_gpu, _) in enumerate(GPU_PAIRS):
                port = BASE_PORT + i
                self.start_vllm_server(i, vllm_gpu, port)
                time.sleep(2)  # Stagger startup

            # Phase 2: Wait for all servers to be ready (60s for model loading)
            print("\n⏳ Phase 2: Waiting 60s for VLLM servers to load models...")
            time.sleep(60)

            # Check all servers are still alive
            all_ready = True
            for i, proc in enumerate(self.vllm_processes):
                if proc.poll() is not None:
                    print(f"[Instance {i}] ✗ VLLM server died (exit: {proc.returncode})")
                    all_ready = False
                else:
                    print(f"[Instance {i}] ✓ VLLM server running")

            if not all_ready:
                print("\n❌ Some VLLM servers failed to start. Aborting.")
                self.shutdown_all()
                sys.exit(1)

            print("\n✅ All VLLM servers ready!")

            # Phase 3: Start all training processes
            print("\n🏋️  Phase 3: Starting training processes...")
            for i, (_, training_gpu) in enumerate(GPU_PAIRS):
                port = BASE_PORT + i
                seed = SEEDS[i]
                self.start_training(i, training_gpu, port, seed)
                time.sleep(5)  # Stagger startup

            print("\n✅ All training processes started!")

            # Phase 4: Monitor until completion
            self.monitor_processes()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown_all()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel TRL GRPO training with multiple seeds"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to use for training (default: gsm8k)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=40,
        help="Run evaluation every N steps (default: 40)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model ID to use (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of training updates per batch of rollouts (default: 1)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (overrides env var)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags (overrides env var)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Verify we have 8 GPUs
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus < 8:
        print(f"⚠️  Warning: Only {num_gpus} GPUs detected. This script expects 8 GPUs.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)

    manager = ProcessManager(
        dataset=args.dataset,
        eval_every=args.eval_every,
        model_id=args.model,
        num_iterations=args.num_iterations,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
    )
    manager.run()


if __name__ == "__main__":
    main()
