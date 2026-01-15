"""Experiment runner for distributed training on Lium pods.

Orchestrates running training experiments with different hyperparameter
configurations across multiple pods.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncssh


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment.

    Attributes:
        name: Unique experiment name
        dataset: Dataset to train on (gsm8k, math, mbpp)
        model_id: HuggingFace model ID
        learning_rate: Learning rate
        batch_size: Batch size per device
        grad_accum_steps: Gradient accumulation steps
        total_steps: Total training steps
        eval_every: Evaluation interval
        vllm_gpu: GPU for VLLM server (e.g., "0")
        train_gpu: GPU for training (e.g., "1")
        vllm_port: Port for VLLM server (default: 8000 + experiment_id)
        custom_env: Additional environment variables
        custom_args: Additional arguments to train script
    """

    name: str
    dataset: str = "math"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Training hyperparameters
    learning_rate: float = 3e-6
    batch_size: int = 4
    grad_accum_steps: int = 128
    total_steps: int = 100
    eval_every: int = 40

    # GPU configuration (1 GPU for VLLM + 1 GPU for training)
    vllm_gpu: str = "0"
    train_gpu: str = "1"
    vllm_port: int = 8000

    # Optional overrides
    custom_env: dict[str, str] = field(default_factory=dict)
    custom_args: dict[str, Any] = field(default_factory=dict)

    def to_env_vars(self) -> dict[str, str]:
        """Convert config to environment variables for the training script."""
        env = {
            "DATASET": self.dataset,
            "MODEL_ID": self.model_id,
            "LEARNING_RATE": str(self.learning_rate),
            "BATCH_SIZE": str(self.batch_size),
            "GRAD_ACCUM_STEPS": str(self.grad_accum_steps),
            "TOTAL_STEPS": str(self.total_steps),
            "EVAL_EVERY": str(self.eval_every),
            "CUDA_VISIBLE_DEVICES_VLLM": self.vllm_gpu,
            "CUDA_VISIBLE_DEVICES_TRAIN": self.train_gpu,
            "VLLM_PORT": str(self.vllm_port),
        }
        env.update(self.custom_env)
        return env

    def to_train_args(self) -> str:
        """Convert config to training script arguments."""
        args = [
            f"--dataset {self.dataset}",
            f"--eval-every {self.eval_every}",
        ]

        for key, value in self.custom_args.items():
            args.append(f"--{key} {value}")

        return " ".join(args)


class ExperimentRunner:
    """Manages running experiments on remote Lium pods via SSH.

    Handles:
    - SSH connection management
    - Code deployment (git clone or rsync)
    - Environment setup (uv installation, dependencies)
    - Remote command execution
    - Log streaming and collection
    """

    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str = "root",
        ssh_key_path: str | None = None,
    ):
        """Initialize experiment runner.

        Args:
            ssh_host: SSH host address
            ssh_port: SSH port
            ssh_user: SSH username
            ssh_key_path: Path to SSH private key (optional)
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self._conn: asyncssh.SSHClientConnection | None = None

    async def connect(self):
        """Establish SSH connection to the pod."""
        print(f"🔌 Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")

        connect_kwargs = {
            "host": self.ssh_host,
            "port": self.ssh_port,
            "username": self.ssh_user,
            "known_hosts": None,  # Accept any host key (for cloud VMs)
        }

        if self.ssh_key_path:
            connect_kwargs["client_keys"] = [self.ssh_key_path]

        self._conn = await asyncssh.connect(**connect_kwargs)
        print(f"✅ Connected to {self.ssh_host}:{self.ssh_port}")

    async def disconnect(self):
        """Close SSH connection."""
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None

    async def run_command(
        self,
        command: str,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Run a command on the remote pod.

        Args:
            command: Command to execute
            env: Environment variables
            timeout: Command timeout in seconds

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        if not self._conn:
            await self.connect()

        print(f"🔧 Running: {command[:100]}...")

        result = await self._conn.run(
            command,
            env=env,
            timeout=timeout,
            check=False,
        )

        return result.stdout, result.stderr, result.exit_status

    async def setup_code(
        self,
        local_path: Path | None = None,
        git_repo: str = "https://github.com/your-org/grail.git",
        git_branch: str = "main",
        use_git: bool = True,
        remote_path: str = "~/grail",
    ):
        """Setup code on remote pod (git clone or rsync).

        Args:
            local_path: Local path for rsync (if use_git=False)
            git_repo: Git repository URL (if use_git=True)
            git_branch: Git branch to checkout
            use_git: Whether to use git clone (True) or rsync (False)
            remote_path: Remote destination path
        """
        if use_git:
            print(f"📦 Cloning repository: {git_repo} (branch: {git_branch})")

            # Remove existing directory if it exists
            await self.run_command(f"rm -rf {remote_path}")

            # Clone repository
            stdout, stderr, code = await self.run_command(
                f"git clone -b {git_branch} {git_repo} {remote_path}",
                timeout=300,
            )

            if code != 0:
                print(f"❌ Git clone failed: {stderr}")
                raise RuntimeError(f"Git clone failed: {stderr}")

            print(f"✅ Repository cloned to {remote_path}")
        else:
            if not local_path:
                raise ValueError("local_path required when use_git=False")

            print(f"📦 Syncing code from {local_path} to {remote_path}")

            # Use rsync for efficient syncing
            rsync_cmd = [
                "rsync",
                "-avz",
                "--delete",
                "--exclude='.git'",
                "--exclude='__pycache__'",
                "--exclude='*.pyc'",
                "--exclude='.venv'",
                "--exclude='outputs'",
                "--exclude='wandb'",
                f"-e 'ssh -p {self.ssh_port} -o StrictHostKeyChecking=no'",
                f"{local_path}/",
                f"{self.ssh_user}@{self.ssh_host}:{remote_path}/",
            ]

            # Run rsync locally
            result = subprocess.run(
                " ".join(rsync_cmd),
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("✅ Code synced successfully")
            else:
                print(f"❌ Sync failed: {result.stderr}")
                raise RuntimeError(f"Code sync failed: {result.stderr}")

    async def copy_env_file(
        self,
        local_env_path: Path,
        remote_path: str = "~/grail",
    ):
        """Copy .env file to remote pod.

        Args:
            local_env_path: Local path to .env file
            remote_path: Remote code directory
        """
        if not local_env_path.exists():
            print(f"⚠️  .env file not found at {local_env_path}, skipping")
            return

        print(f"📋 Copying .env file from {local_env_path}")

        # Read local .env
        env_content = local_env_path.read_text()

        # Write to remote
        await self.run_command(
            f"cat > {remote_path}/.env << 'ENV_FILE_EOF'\n{env_content}\nENV_FILE_EOF"
        )

        print("✅ .env file copied")

    async def setup_environment(self, remote_path: str = "~/grail"):
        """Setup Python environment on remote pod.

        Steps:
        1. Install uv (if not present)
        2. Run uv sync in research/trl (installs TRL deps + grail as editable)
        3. Run uv sync in tools/vllm-server

        Args:
            remote_path: Remote code directory
        """
        print("🐍 Setting up Python environment")

        # Step 1: Install uv
        print("  [1/3] Installing uv...")
        stdout, stderr, code = await self.run_command(
            "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh",
            timeout=180,
        )

        if code != 0 and "command not found" not in stderr:
            print(f"⚠️  uv installation warning: {stderr[:200]}")

        # Make sure uv is in PATH
        await self.run_command('export PATH="$HOME/.cargo/bin:$PATH"')

        # Step 2: Sync TRL research dependencies (includes grail as editable dependency)
        print("  [2/3] Installing TRL research dependencies...")
        stdout, stderr, code = await self.run_command(
            f"cd {remote_path}/research/trl && $HOME/.cargo/bin/uv sync",
            timeout=600,
        )

        if code != 0:
            print(f"❌ TRL dependency installation failed:\n{stderr}")
            raise RuntimeError(f"TRL dependency installation failed: {stderr}")

        print("  ✅ TRL dependencies installed (includes grail)")

        # Step 3: Sync vllm-server dependencies
        print("  [3/3] Installing VLLM server dependencies...")
        stdout, stderr, code = await self.run_command(
            f"cd {remote_path}/tools/vllm-server && $HOME/.cargo/bin/uv sync",
            timeout=600,
        )

        if code != 0:
            print(f"❌ VLLM dependency installation failed:\n{stderr}")
            raise RuntimeError(f"VLLM dependency installation failed: {stderr}")

        print("  ✅ VLLM dependencies installed")
        print("✅ Environment setup complete")

    async def run_experiment(
        self,
        config: ExperimentConfig,
        remote_path: str = "~/grail",
        log_file: str | None = None,
        background: bool = False,
    ) -> int:
        """Run a training experiment with the given configuration.

        Args:
            config: Experiment configuration
            remote_path: Remote code directory
            log_file: Optional path to save logs locally
            background: If True, run in background and return immediately

        Returns:
            Exit code of the training process (0 if background=True)
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 Starting experiment: {config.name}")
        print(f"   VLLM GPU: {config.vllm_gpu} | Training GPU: {config.train_gpu}")
        print(f"   Port: {config.vllm_port}")
        print(f"{'=' * 80}\n")

        # Create experiment script
        env_vars = config.to_env_vars()
        train_args = config.to_train_args()

        # Export environment variables
        env_exports = "\n".join([f"export {k}={v}" for k, v in env_vars.items()])

        # Create the training script
        script = f"""#!/bin/bash
set -euo pipefail

# Change to TRL research directory
cd {remote_path}/research/trl

# Add uv to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Export environment variables
{env_exports}

# Start VLLM server
echo "[{config.name}] Starting VLLM server on GPU $CUDA_VISIBLE_DEVICES_VLLM, port $VLLM_PORT..."
VLLM_TRL_BIN="{remote_path}/tools/vllm-server/.venv/bin/trl"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VLLM nohup "$VLLM_TRL_BIN" vllm-serve \\
  --model $MODEL_ID \\
  --tensor-parallel-size 1 \\
  --host 127.0.0.1 \\
  --port $VLLM_PORT \\
  --gpu-memory-utilization 0.9 \\
  > {remote_path}/research/trl/vllm_server_{config.name}.log 2>&1 &

VLLM_PID=$!
echo "[{config.name}] VLLM server started (PID: $VLLM_PID)"

# Wait for VLLM to be ready
echo "[{config.name}] Waiting for VLLM server to be ready..."
sleep 60

# Check if VLLM is still running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "[{config.name}] ERROR: VLLM server died during startup"
    tail -50 {remote_path}/research/trl/vllm_server_{config.name}.log
    exit 1
fi

# Use TRL research environment python
TRL_PYTHON_BIN="{remote_path}/research/trl/.venv/bin/python"

# Start training
echo "[{config.name}] Starting training on GPU $CUDA_VISIBLE_DEVICES_TRAIN..."
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_TRAIN "$TRL_PYTHON_BIN" train_trl_grpo.py \\
  {train_args} \\
  2>&1 | tee train_{config.name}.log

TRAIN_EXIT=${{PIPESTATUS[0]}}

# Cleanup: kill VLLM server
echo "[{config.name}] Training finished, cleaning up VLLM server..."
kill $VLLM_PID 2>/dev/null || true

exit $TRAIN_EXIT
"""

        # Write script to remote
        script_path = f"/tmp/experiment_{config.name}.sh"
        await self.run_command(
            f"cat > {script_path} << 'EXPERIMENT_SCRIPT_EOF'\n{script}\nEXPERIMENT_SCRIPT_EOF"
        )
        await self.run_command(f"chmod +x {script_path}")

        if background:
            # Run in background
            await self.run_command(
                f"nohup bash {script_path} > /tmp/experiment_{config.name}_output.log 2>&1 &",
                timeout=10,
            )
            print(f"✅ Experiment {config.name} started in background")
            return 0
        else:
            # Run in foreground
            start_time = time.time()
            stdout, stderr, exit_code = await self.run_command(
                f"bash {script_path}",
                timeout=None,  # No timeout for long training runs
            )
            elapsed = time.time() - start_time

            print(f"\n{'=' * 80}")
            print(f"Experiment {config.name} completed in {elapsed / 60:.1f} minutes")
            print(f"Exit code: {exit_code}")
            print(f"{'=' * 80}\n")

            # Save logs if requested
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "w") as f:
                    f.write(f"=== STDOUT ===\n{stdout}\n\n")
                    f.write(f"=== STDERR ===\n{stderr}\n")
                print(f"📝 Logs saved to {log_file}")

            return exit_code

    async def cleanup_experiment(self, config: ExperimentConfig):
        """Cleanup experiment artifacts (stop servers, remove temp files).

        Args:
            config: Experiment configuration
        """
        print(f"🧹 Cleaning up experiment: {config.name}")

        # Kill VLLM server on specific port
        await self.run_command(
            f"lsof -ti:{config.vllm_port} | xargs kill -9 2>/dev/null || true",
            timeout=10,
        )

        # Remove temp script
        await self.run_command(f"rm -f /tmp/experiment_{config.name}.sh", timeout=10)

        print("✅ Cleanup complete")


async def run_experiments_on_pod(
    pod_info: dict,
    experiments: list[ExperimentConfig],
    local_code_path: Path | None = None,
    local_env_path: Path | None = None,
    git_repo: str | None = None,
    git_branch: str = "main",
    use_git: bool = True,
    setup_env: bool = True,
    run_parallel: bool = True,
):
    """Run multiple experiments on a single pod.

    Args:
        pod_info: Pod information from Lium manager
        experiments: List of experiment configurations
        local_code_path: Local path to code repository (for rsync)
        local_env_path: Local path to .env file
        git_repo: Git repository URL (for git clone)
        git_branch: Git branch to checkout
        use_git: Whether to use git clone (True) or rsync (False)
        setup_env: Whether to setup environment before running
        run_parallel: If True, run experiments in parallel (background)
    """
    ssh_info = pod_info["ssh"]
    runner = ExperimentRunner(
        ssh_host=ssh_info["host"],
        ssh_port=ssh_info["port"],
    )

    try:
        await runner.connect()

        # Setup code (git clone or rsync)
        await runner.setup_code(
            local_path=local_code_path,
            git_repo=git_repo,
            git_branch=git_branch,
            use_git=use_git,
        )

        # Copy .env file if provided
        if local_env_path:
            await runner.copy_env_file(local_env_path)

        # Setup environment
        if setup_env:
            await runner.setup_environment()

        # Run experiments
        if run_parallel:
            # Start all experiments in background
            print(f"\n🚀 Starting {len(experiments)} experiments in parallel...\n")
            for exp in experiments:
                try:
                    await runner.run_experiment(exp, background=True)
                except Exception as e:
                    print(f"❌ Failed to start experiment {exp.name}: {e}")

            # Wait for all to complete (check every minute)
            print("\n⏳ Waiting for experiments to complete...")
            print("   (Use SSH to monitor: tail -f ~/grail/train_*.log)\n")

        else:
            # Run sequentially
            for exp in experiments:
                log_file = f"logs/{pod_info['spec']['name']}_{exp.name}.log"
                try:
                    exit_code = await runner.run_experiment(exp, log_file=log_file)
                    if exit_code != 0:
                        print(f"⚠️  Experiment {exp.name} failed with exit code {exit_code}")
                except Exception as e:
                    print(f"❌ Experiment {exp.name} failed with error: {e}")
                finally:
                    await runner.cleanup_experiment(exp)

    finally:
        await runner.disconnect()


async def run_experiments_parallel(
    pod_experiments: dict[str, tuple[dict, list[ExperimentConfig]]],
    local_code_path: Path | None = None,
    local_env_path: Path | None = None,
    git_repo: str | None = None,
    git_branch: str = "main",
    use_git: bool = True,
    setup_env: bool = True,
    run_parallel_per_pod: bool = True,
):
    """Run experiments in parallel across multiple pods.

    Args:
        pod_experiments: Dict mapping pod names to (pod_info, experiments) tuples
        local_code_path: Local path to code repository (for rsync)
        local_env_path: Local path to .env file
        git_repo: Git repository URL (for git clone)
        git_branch: Git branch to checkout
        use_git: Whether to use git clone (True) or rsync (False)
        setup_env: Whether to setup environment before running
        run_parallel_per_pod: If True, run experiments in parallel on each pod
    """
    tasks = []
    for pod_name, (pod_info, experiments) in pod_experiments.items():
        print(f"\n📋 Scheduling {len(experiments)} experiments on pod: {pod_name}")
        task = run_experiments_on_pod(
            pod_info=pod_info,
            experiments=experiments,
            local_code_path=local_code_path,
            local_env_path=local_env_path,
            git_repo=git_repo,
            git_branch=git_branch,
            use_git=use_git,
            setup_env=setup_env,
            run_parallel=run_parallel_per_pod,
        )
        tasks.append(task)

    print(f"\n🚀 Running experiments on {len(tasks)} pods in parallel...\n")
    await asyncio.gather(*tasks)
    print("\n✅ All experiments completed!")
