#!/usr/bin/env python3
"""Parallel training launcher for TRL GRPO with multiple seeds.

Runs 4 parallel training instances, each with:
- Dedicated VLLM server GPU
- Dedicated training GPU
- Unique random seed
- Unique port
- NCCL weight sync between trainer and vLLM server (via TCP store)

NOTE: This script is designed to run in nohup mode. Use:
    ./run_parallel_training_nohup.sh [dataset] [eval_every]

GPU allocation (8 GPUs total):
- Instance 0: VLLM on GPU 0, Training on GPU 1, Port 8000, Seed 42
- Instance 1: VLLM on GPU 2, Training on GPU 3, Port 8001, Seed 1337
- Instance 2: VLLM on GPU 4, Training on GPU 5, Port 8002, Seed 2024
- Instance 3: VLLM on GPU 6, Training on GPU 7, Port 8003, Seed 9999

NCCL weight sync between trainer and vLLM server requires both GPUs to be visible
in each process. To keep HuggingFace Trainer from accidentally using DataParallel
on the vLLM GPU, the training script forces `n_gpu=1` after device setup.

Usage:
    python run_parallel_training.py --dataset gsm8k
    python run_parallel_training.py --dataset math --eval-every 50
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Force unbuffered output for nohup mode
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


# Fixed seeds for reproducibility across experiments
SEEDS = [42, 1337, 2024, 9999]

# GPU allocation: (vllm_gpu, training_gpu)
GPU_PAIRS = [
    (0, 1),  # Instance 0
    (2, 3),  # Instance 1
    (4, 5),  # Instance 2
    (6, 7),  # Instance 3
]

# Base port for VLLM servers (can be overridden via GRAIL_BASE_PORT env var)
BASE_PORT = int(os.environ.get("GRAIL_BASE_PORT", 8000))

# Base port for NCCL group coordination (each instance needs unique port)
# Use large spacing (100 ports apart) to avoid any potential conflicts
# Can be overridden via GRAIL_BASE_GROUP_PORT env var
BASE_GROUP_PORT = int(os.environ.get("GRAIL_BASE_GROUP_PORT", 51200))
GROUP_PORT_SPACING = 100  # Port increment per instance

# Base ports for vLLM internal communication (can be overridden via env vars)
VLLM_NIXL_PORT_BASE = int(os.environ.get("GRAIL_VLLM_NIXL_PORT_BASE", 5557))
VLLM_MASTER_PORT_BASE = int(os.environ.get("GRAIL_VLLM_MASTER_PORT_BASE", 29500))

# Run prefix for unique run names (used to differentiate parallel experiments)
RUN_PREFIX = os.environ.get("GRAIL_RUN_PREFIX", "")


class ProcessManager:
    """Manages VLLM servers and training processes."""

    def __init__(
        self,
        dataset: str,
        eval_every: int,
        model_id: str,
        num_iterations: int = 1,
        num_instances: int = 1,
        wandb_project: str | None = None,
        wandb_tags: str | None = None,
        output_base: str | None = None,
        batch_size: int | None = None,
        grad_accum_steps: int | None = None,
        resume_from_checkpoint: str | None = None,
        seed_override: int | None = None,
        lr: float | None = None,
    ):
        self.dataset = dataset
        self.eval_every = eval_every
        self.model_id = model_id
        self.num_iterations = num_iterations
        self.num_instances = num_instances
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.seed_override = seed_override
        self.lr = lr
        # Output base directory for checkpoints, wandb, etc.
        # Use /ephemeral on cloud instances with limited home storage
        self.output_base = output_base or os.getenv("GRAIL_OUTPUT_BASE", ".")
        self.vllm_processes: list[subprocess.Popen] = []
        self.training_processes: list[subprocess.Popen] = []
        self.log_dir = Path("./logs/parallel_training")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Start instance index (for running on different GPU pairs)
        self.start_instance = int(os.environ.get("GRAIL_START_INSTANCE", 0))

        # Limit GPU_PAIRS and SEEDS to num_instances, starting from start_instance
        self.gpu_pairs = GPU_PAIRS[self.start_instance:self.start_instance + num_instances]
        # Override seeds if seed_override is provided (for single-instance resume)
        if self.seed_override is not None and num_instances == 1:
            self.seeds = [self.seed_override]
        else:
            self.seeds = SEEDS[self.start_instance:self.start_instance + num_instances]

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle termination signals gracefully."""
        print(f"\n\n🛑 Received signal {signum}, shutting down gracefully...")
        self.shutdown_all()
        sys.exit(0)

    def start_vllm_server(
        self, instance_id: int, vllm_gpu: int, training_gpu: int, port: int
    ) -> subprocess.Popen:
        """Start VLLM server on specified GPU and port.

        Args:
            instance_id: Instance identifier (0-3)
            vllm_gpu: Physical GPU ID for VLLM server
            training_gpu: Physical GPU ID for training (needed for NCCL visibility)
            port: Port number for server

        Returns:
            Popen object for the server process
        """
        log_file = self.log_dir / f"vllm_instance{instance_id}_gpu{vllm_gpu}_port{port}.log"

        # Use tools/vllm-server venv for vLLM (isolated environment with compatible versions)
        repo_root = Path(__file__).parent.parent.parent
        trl_path = repo_root / "tools" / "vllm-server" / ".venv" / "bin" / "trl"
        # VLLM will use device 0 by default (first GPU in CUDA_VISIBLE_DEVICES)
        cmd = [
            str(trl_path), "vllm-serve",
            "--model", self.model_id,
            "--port", str(port),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.9",
            "--tensor-parallel-size", "1",
        ]

        env = os.environ.copy()
        # NCCL weight sync requires both GPUs visible to enable peer access.
        # Order matters: vLLM must run on cuda:0 in this process.
        env["CUDA_VISIBLE_DEVICES"] = f"{vllm_gpu},{training_gpu}"
        # Unique cache directories per instance to avoid race conditions
        cache_base = f"/tmp/vllm_cache_instance{instance_id}"
        env["VLLM_CACHE_ROOT"] = cache_base
        env["TORCHINDUCTOR_CACHE_DIR"] = f"{cache_base}/inductor"
        env["TORCH_COMPILE_CACHE_DIR"] = f"{cache_base}/torch_compile"
        # CRITICAL: Unique ports per instance to avoid NCCL/collective conflicts
        # Without this, multiple vLLM instances try to bind to the same ports
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(VLLM_NIXL_PORT_BASE + instance_id * 10)
        env["VLLM_DP_MASTER_PORT"] = str(VLLM_MASTER_PORT_BASE + instance_id * 10)
        env["MASTER_PORT"] = str(VLLM_MASTER_PORT_BASE + instance_id * 10)

        print(f"[Instance {instance_id}] Starting VLLM server:")
        print(f"  GPUs visible: {vllm_gpu},{training_gpu}, Using: cuda:0, Port: {port}, Log: {log_file}")

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
        self,
        port: int,
        timeout: int = 60,
        instance_id: int = 0,
        process: subprocess.Popen[str] | None = None,
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
        vllm_gpu: int,
        training_gpu: int,
        port: int,
        seed: int,
    ) -> subprocess.Popen:
        """Start training script.

        Args:
            instance_id: Instance identifier (0-3)
            vllm_gpu: Physical GPU ID for VLLM (needed for NCCL visibility)
            training_gpu: Physical GPU ID for training
            port: VLLM server port to connect to
            seed: Random seed

        Returns:
            Popen object for the training process
        """
        log_file = self.log_dir / f"training_instance{instance_id}_gpu{training_gpu}_seed{seed}.log"

        # Use local .venv python
        python_path = Path(__file__).parent / ".venv" / "bin" / "python"
        group_port = BASE_GROUP_PORT + (instance_id * GROUP_PORT_SPACING)  # Unique port per instance
        cmd = [
            str(python_path), "train_trl_grpo.py",
            "--dataset", self.dataset,
            "--eval-every", str(self.eval_every),
            "--num-iterations", str(self.num_iterations),
            "--seed", str(seed),
            "--vllm-port", str(port),
            "--group-port", str(group_port),
            "--run-suffix", f"{RUN_PREFIX}instance{instance_id}_seed{seed}",
            "--model", self.model_id,
            # Ensure training uses cuda:0 in this process (the training GPU).
            "--device", "cuda:0",
        ]
        # Add batch size and grad accum steps if provided
        if self.batch_size is not None:
            cmd.extend(["--batch-size", str(self.batch_size)])
        if self.grad_accum_steps is not None:
            cmd.extend(["--grad-accum-steps", str(self.grad_accum_steps)])
        # Add resume checkpoint if provided
        if self.resume_from_checkpoint:
            cmd.extend(["--resume-from-checkpoint", self.resume_from_checkpoint])
        # Add W&B args if provided (CLI takes precedence over env vars)
        if self.wandb_project:
            cmd.extend(["--wandb-project", self.wandb_project])
        if self.wandb_tags:
            cmd.extend(["--wandb-tags", self.wandb_tags])
        # Add learning rate if provided
        if self.lr is not None:
            cmd.extend(["--lr", str(self.lr)])

        env = os.environ.copy()
        # NCCL weight sync requires both GPUs visible to enable peer access.
        # Order matters: training must run on cuda:0 in this process, so we put the
        # training GPU first.
        env["CUDA_VISIBLE_DEVICES"] = f"{training_gpu},{vllm_gpu}"
        # Set output directories for checkpoints, wandb, etc.
        env["GRAIL_OUTPUT_BASE"] = self.output_base
        env["WANDB_DIR"] = f"{self.output_base}/wandb"

        print(f"[Instance {instance_id}] Starting training:")
        print(f"  GPUs visible: {training_gpu},{vllm_gpu}, Using: cuda:0, Seed: {seed}, Log: {log_file}")
        print(f"  Output base: {self.output_base}")

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
        print(f"Instances: {len(self.seeds)} (starting at index {self.start_instance})")
        print(f"Seeds: {self.seeds}")
        print(f"GPU pairs: {self.gpu_pairs}")
        print(f"vLLM Ports: {[BASE_PORT + i for i in range(len(self.seeds))]}")
        print(f"Group Ports: {[BASE_GROUP_PORT + i * GROUP_PORT_SPACING for i in range(len(self.seeds))]}")
        print(f"Run Prefix: {RUN_PREFIX or '(none)'}")
        print(f"Logs: {self.log_dir}")
        print("=" * 80)

        try:
            # Phase 1: Start all VLLM servers
            print("\n📡 Phase 1: Starting VLLM servers...")
            for i, (vllm_gpu, training_gpu) in enumerate(self.gpu_pairs):
                port = BASE_PORT + i
                self.start_vllm_server(i, vllm_gpu, training_gpu, port)
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
            # IMPORTANT: NCCL communicator initialization between trainer and VLLM
            # is prone to race conditions. We stagger startups by 30s to ensure
            # each instance fully initializes before the next one starts.
            print("\n🏋️  Phase 3: Starting training processes...")
            for i, (vllm_gpu, training_gpu) in enumerate(self.gpu_pairs):
                port = BASE_PORT + i
                seed = self.seeds[i]
                self.start_training(i, vllm_gpu, training_gpu, port, seed)
                if i < len(self.gpu_pairs) - 1:  # Don't sleep after the last one
                    # Longer delay (45s) to ensure init_communicator completes fully
                    # before next instance starts (vLLM uses fire_and_forget async RPC)
                    print(f"   ⏳ Waiting 45s for NCCL init before next instance...")
                    time.sleep(45)

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
        "--num-instances",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Number of parallel training instances (default: 1). "
             "Higher values risk NCCL conflicts.",
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
    parser.add_argument(
        "--output-base",
        type=str,
        default=None,
        help="Base directory for outputs, checkpoints, wandb (default: current dir). "
             "Set to /ephemeral on cloud instances with limited home storage.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size per device. Effective batch = batch_size * grad_accum_steps.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps. Effective batch = batch_size * grad_accum_steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to TRL checkpoint to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed for single-instance runs (for resuming specific seed).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: 3e-6).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Verify we have enough GPUs for requested instances
    import torch
    num_gpus = torch.cuda.device_count()
    required_gpus = args.num_instances * 2
    if num_gpus < required_gpus:
        print(f"⚠️  Warning: Only {num_gpus} GPUs detected, but {required_gpus} required for {args.num_instances} instance(s).")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)

    manager = ProcessManager(
        dataset=args.dataset,
        eval_every=args.eval_every,
        model_id=args.model,
        num_iterations=args.num_iterations,
        num_instances=args.num_instances,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        output_base=args.output_base,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed_override=args.seed,
        lr=args.lr,
    )
    manager.run()


if __name__ == "__main__":
    main()
