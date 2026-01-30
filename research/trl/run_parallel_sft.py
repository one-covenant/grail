#!/usr/bin/env python3
"""Parallel SFT training launcher with multiple seeds.

Runs up to 8 parallel SFT training instances, each with:
- Dedicated GPU (one GPU per instance)
- Unique random seed
- No vLLM server needed (supervised fine-tuning doesn't require generation)

NOTE: This script is designed to run in nohup mode. Use:
    ./run_parallel_sft_nohup.sh [dataset] [eval_every] [model] [max_steps] [num_instances]

GPU allocation (8 GPUs total, one per instance):
- Instance 0: GPU 0, Seed 42
- Instance 1: GPU 1, Seed 1337
- Instance 2: GPU 2, Seed 2024
- Instance 3: GPU 3, Seed 9999
- Instance 4: GPU 4, Seed 7777
- Instance 5: GPU 5, Seed 5555
- Instance 6: GPU 6, Seed 3333
- Instance 7: GPU 7, Seed 1111

Usage:
    python run_parallel_sft.py --dataset math
    python run_parallel_sft.py --dataset gsm8k --num-instances 4
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
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


# Fixed seeds for reproducibility across experiments (8 total for 8 GPUs)
SEEDS = [42, 1337, 2024, 9999, 7777, 5555, 3333, 1111]

# GPU IDs (one per instance)
GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

# Run prefix for unique run names (used to differentiate parallel experiments)
RUN_PREFIX = os.environ.get("GRAIL_RUN_PREFIX", "")


class SFTProcessManager:
    """Manages parallel SFT training processes.

    Unlike GRPO, SFT doesn't need vLLM servers for generation during training,
    so we can run one training instance per GPU (up to 8 instances on 8-GPU node).
    """

    def __init__(
        self,
        dataset: str,
        eval_every: int,
        model_id: str,
        max_steps: int = 400,
        num_instances: int = 4,
        wandb_project: str | None = None,
        wandb_tags: str | None = None,
        output_base: str | None = None,
        batch_size: int | None = None,
        grad_accum_steps: int | None = None,
        seed_override: int | None = None,
        lr: float | None = None,
    ):
        self.dataset = dataset
        self.eval_every = eval_every
        self.model_id = model_id
        self.max_steps = max_steps
        self.num_instances = num_instances
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.seed_override = seed_override
        self.lr = lr
        # Output base directory for checkpoints, wandb, etc.
        # Use /ephemeral on cloud instances with limited home storage
        self.output_base = output_base or os.getenv("GRAIL_OUTPUT_BASE", ".")
        self.training_processes: list[subprocess.Popen] = []
        self.log_dir = Path("./logs/parallel_sft")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Start instance index (for running on different GPUs)
        self.start_instance = int(os.environ.get("GRAIL_START_INSTANCE", 0))

        # Limit GPUs and seeds to num_instances, starting from start_instance
        self.gpu_ids = GPU_IDS[self.start_instance : self.start_instance + num_instances]
        # Override seeds if seed_override is provided (for single-instance resume)
        if self.seed_override is not None and num_instances == 1:
            self.seeds = [self.seed_override]
        else:
            self.seeds = SEEDS[self.start_instance : self.start_instance + num_instances]

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:  # noqa: ARG002, ARG001
        """Handle termination signals gracefully."""
        print(f"\n\n[SHUTDOWN] Received signal {signum}, shutting down gracefully...")
        self.shutdown_all()
        sys.exit(0)

    def start_training(
        self,
        instance_id: int,
        gpu_id: int,
        seed: int,
    ) -> subprocess.Popen:
        """Start SFT training script on specified GPU.

        Args:
            instance_id: Instance identifier (0-7)
            gpu_id: Physical GPU ID for training
            seed: Random seed

        Returns:
            Popen object for the training process
        """
        log_file = self.log_dir / f"sft_instance{instance_id}_gpu{gpu_id}_seed{seed}.log"

        # Use local .venv python
        python_path = Path(__file__).parent / ".venv" / "bin" / "python"
        cmd = [
            str(python_path),
            "train_trl_sft.py",
            "--dataset",
            self.dataset,
            "--eval-every",
            str(self.eval_every),
            "--max-steps",
            str(self.max_steps),
            "--seed",
            str(seed),
            "--run-suffix",
            f"{RUN_PREFIX}instance{instance_id}_seed{seed}",
            "--model",
            self.model_id,
            # Ensure training uses cuda:0 (the only visible GPU)
            "--device",
            "cuda:0",
        ]

        # Add batch size if provided
        if self.batch_size is not None:
            cmd.extend(["--batch-size", str(self.batch_size)])

        # Add grad accum steps if provided
        if self.grad_accum_steps is not None:
            cmd.extend(["--grad-accum-steps", str(self.grad_accum_steps)])

        # Add W&B args if provided (CLI takes precedence over env vars)
        if self.wandb_project:
            cmd.extend(["--wandb-project", self.wandb_project])
        if self.wandb_tags:
            cmd.extend(["--wandb-tags", self.wandb_tags])

        # Add learning rate if provided
        if self.lr is not None:
            cmd.extend(["--lr", str(self.lr)])

        env = os.environ.copy()
        # Set single GPU visibility for this instance
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Set output directories for checkpoints, wandb, etc.
        env["GRAIL_OUTPUT_BASE"] = self.output_base
        env["WANDB_DIR"] = f"{self.output_base}/wandb"

        print(f"[Instance {instance_id}] Starting SFT training:")
        print(f"  GPU: {gpu_id}, Seed: {seed}, Log: {log_file}")
        print(f"  Output base: {self.output_base}")

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group for clean shutdown
            )

        self.training_processes.append(process)
        return process

    def monitor_processes(self) -> None:
        """Monitor all processes and report status."""
        print("\n" + "=" * 80)
        print("MONITORING SFT TRAINING PROCESSES")
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
                    print(f"\n[WARNING] Instance {i} failed with exit code {poll}")

            if all_finished:
                if any_failed:
                    print("\n[ERROR] Some processes failed. Check logs in ./logs/parallel_sft/")
                else:
                    print("\n[SUCCESS] All SFT training processes completed!")
                break

            # Status update every 60 seconds
            time.sleep(60)

            # Print brief status
            running = sum(1 for p in self.training_processes if p.poll() is None)
            print(
                f"[{time.strftime('%H:%M:%S')}] Training processes: "
                f"{running}/{len(self.training_processes)} running"
            )

    def shutdown_all(self) -> None:
        """Shutdown all processes gracefully."""
        print("\n[CLEANUP] Shutting down all processes...")

        for i, proc in enumerate(self.training_processes):
            if proc.poll() is None:
                print(f"  Stopping training instance {i}...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=10)
                except Exception as e:
                    print(f"    Force killing: {e}")
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        print("[CLEANUP] All processes stopped")

    def run(self) -> None:
        """Run the full parallel SFT training pipeline."""
        print("=" * 80)
        print("PARALLEL SFT TRAINING LAUNCHER")
        print("=" * 80)
        print(f"Dataset: {self.dataset}")
        print(f"Model: {self.model_id}")
        print(f"Max Steps: {self.max_steps}")
        print(f"Instances: {len(self.seeds)} (starting at index {self.start_instance})")
        print(f"Seeds: {self.seeds}")
        print(f"GPUs: {self.gpu_ids}")
        print(f"Run Prefix: {RUN_PREFIX or '(none)'}")
        print(f"Logs: {self.log_dir}")
        print("=" * 80)

        try:
            # Start all training processes with staggered startup
            print("\n[STARTUP] Starting SFT training processes...")
            for i, (gpu_id, seed) in enumerate(zip(self.gpu_ids, self.seeds)):
                self.start_training(i, gpu_id, seed)
                # Brief delay to avoid race conditions in logging/wandb init
                if i < len(self.gpu_ids) - 1:
                    print("   Waiting 5s before next instance...")
                    time.sleep(5)

            print(f"\n[SUCCESS] All {len(self.seeds)} training processes started!")

            # Monitor until completion
            self.monitor_processes()

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.shutdown_all()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parallel TRL SFT training with multiple seeds"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=["gsm8k", "math", "mbpp"],
        help="Dataset to use for training (default: math)",
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
        "--max-steps",
        type=int,
        default=400,
        help="Maximum training steps (default: 400)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="Number of parallel training instances (default: 4)",
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
        "--seed",
        type=int,
        default=None,
        help="Override the seed for single-instance runs (for resuming specific seed).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: 2e-5 for SFT).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Verify we have enough GPUs for requested instances
    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus < args.num_instances:
        print(
            f"[WARNING] Only {num_gpus} GPUs detected, "
            f"but {args.num_instances} requested."
        )
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)

    manager = SFTProcessManager(
        dataset=args.dataset,
        eval_every=args.eval_every,
        model_id=args.model,
        max_steps=args.max_steps,
        num_instances=args.num_instances,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        output_base=args.output_base,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        seed_override=args.seed,
        lr=args.lr,
    )
    manager.run()


if __name__ == "__main__":
    main()
