"""Nohup experiment runner for remote Lium pods.

Runs run_parallel_training_nohup.sh on remote instances, monitors completion,
downloads artifacts, and uploads to R2.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import asyncssh

from r2_uploader import upload_experiment_artifacts

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str  # Experiment name (used for R2 folder)
    model_id: str  # HuggingFace model ID
    num_iterations: int  # Number of GRPO training iterations (ignored for SFT)
    trainer_type: str = "grpo"  # Trainer type: "grpo" or "sft"
    max_steps: int = 400  # Max training steps (used by SFT, ignored by GRPO)
    dataset: str = "math"  # Dataset to train on
    eval_every: int = 40  # Evaluation interval
    wandb_project: str = "grail-lium-sweep"  # W&B project for logging
    wandb_tags: str = ""  # Comma-separated W&B tags
    batch_size: int | None = None  # Optional: batch size per device
    grad_accum_steps: int | None = None  # Optional: gradient accumulation steps
    num_instances: int = 4  # Number of parallel training instances (4 for 8-GPU nodes)
    # New: support for running multiple experiments on same node
    run_prefix: str | None = None  # GRAIL_RUN_PREFIX: unique prefix for run names
    seed: int | None = None  # GRAIL_SEED: override seed
    start_instance: int = 0  # GRAIL_START_INSTANCE: GRPO uses pair index; SFT uses GPU index
    base_port: int = 8000  # GRAIL_BASE_PORT: vLLM server port
    base_group_port: int = 51200  # GRAIL_BASE_GROUP_PORT: NCCL group port
    vllm_nixl_port_base: int = 5557  # GRAIL_VLLM_NIXL_PORT_BASE
    vllm_master_port_base: int = 29500  # GRAIL_VLLM_MASTER_PORT_BASE
    learning_rate: float | None = None  # GRAIL_TRAINER_LR: override learning rate


class NohupExperimentRunner:
    """Runs nohup training on remote pod and uploads results to R2.

    This class handles:
    1. SSH connection to pod
    2. Code synchronization
    3. Environment setup
    4. Running nohup training script
    5. Monitoring for completion
    6. Downloading artifacts
    7. Uploading to R2

    Example:
        >>> runner = NohupExperimentRunner(
        ...     ssh_host="1.2.3.4",
        ...     ssh_port=22,
        ...     r2_config={
        ...         "bucket_id": "...",
        ...         "account_id": "...",
        ...         "access_key": "...",
        ...         "secret_key": "...",
        ...     },
        ... )
        >>> await runner.run_experiment(
        ...     config=ExperimentConfig(
        ...         name="qwen2.5-0.5b-iter1",
        ...         model_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...         num_iterations=1,
        ...     ),
        ... )
    """

    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        r2_config: dict[str, str],
        ssh_user: str = "root",
        ssh_key_path: Optional[str] = None,
        remote_path: str = "~/grail",
    ):
        """Initialize experiment runner.

        Args:
            ssh_host: SSH host address
            ssh_port: SSH port
            r2_config: R2 configuration dict with keys: bucket_id, account_id, access_key, secret_key
            ssh_user: SSH username (default: root)
            ssh_key_path: Path to SSH private key (optional)
            remote_path: Remote code directory (default: ~/grail)
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.remote_path = remote_path
        self.r2_config = r2_config
        self._conn: Optional[asyncssh.SSHClientConnection] = None

    async def connect(self):
        """Establish SSH connection to the pod with keepalive."""
        logger.info(f"Connecting to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")

        connect_kwargs = {
            "host": self.ssh_host,
            "port": self.ssh_port,
            "username": self.ssh_user,
            "known_hosts": None,  # Accept any host key (for cloud VMs)
            # SSH keepalive to prevent connection drops during long training
            "keepalive_interval": 30,  # Send keepalive every 30 seconds
            "keepalive_count_max": 5,  # Allow 5 missed keepalives before disconnect
        }

        if self.ssh_key_path:
            connect_kwargs["client_keys"] = [self.ssh_key_path]

        self._conn = await asyncssh.connect(**connect_kwargs)
        logger.info(f"✓ Connected to {self.ssh_host}:{self.ssh_port}")

    async def disconnect(self):
        """Close SSH connection."""
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            logger.info("✓ SSH connection closed")

    async def run_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> tuple[str, str, int]:
        """Run a command on the remote pod.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds (None = no timeout)
            check: If True, raise error on non-zero exit code

        Returns:
            Tuple of (stdout, stderr, exit_code)

        Raises:
            RuntimeError: If check=True and command fails
        """
        if not self._conn:
            await self.connect()

        logger.debug(f"Running: {command[:100]}...")

        result = await self._conn.run(
            command,
            timeout=timeout,
            check=False,
        )

        if check and result.exit_status != 0:
            raise RuntimeError(
                f"Command failed with exit code {result.exit_status}:\n"
                f"Command: {command}\n"
                f"Stderr: {result.stderr}"
            )

        return result.stdout, result.stderr, result.exit_status

    async def sync_code(self, local_path: Path):
        """Sync local code to remote pod using rsync.

        Args:
            local_path: Local path to grail repository
        """
        logger.info(f"Syncing code from {local_path} to {self.remote_path}")

        # First, ensure rsync is installed on the remote pod
        logger.info("  Ensuring rsync is installed on remote...")
        await self.run_command(
            "command -v rsync || (apt-get update -qq && apt-get install -y -qq rsync)",
            timeout=120,
            check=False,
        )

        # Build rsync command with relaxed SSH options and comprehensive exclusions
        # Exclude large directories: checkpoints, outputs, wandb, .venv, caches
        rsync_cmd = [
            "rsync",
            "-avz",
            "--delete",
            # Version control and Python cache
            "--exclude=.git",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
            "--exclude=.pytest_cache",
            "--exclude=.mypy_cache",
            "--exclude=.ruff_cache",
            # Virtual environments (can be huge)
            "--exclude=.venv",
            "--exclude=venv",
            "--exclude=.uv",
            # Training artifacts (often 10s-100s of GB)
            "--exclude=outputs",
            "--exclude=checkpoints",
            "--exclude=wandb",
            "--exclude=logs",
            # Data and model caches
            "--exclude=*.safetensors",
            "--exclude=*.bin",
            "--exclude=*.pt",
            "--exclude=*.pth",
            "--exclude=*.ckpt",
            "--exclude=.cache",
            "--exclude=huggingface_cache",
            # Build artifacts
            "--exclude=*.egg-info",
            "--exclude=build",
            "--exclude=dist",
            "--exclude=*.so",
            # IDE and OS files
            "--exclude=.idea",
            "--exclude=.vscode",
            "--exclude=.DS_Store",
            # Lock files (will be regenerated)
            "--exclude=uv.lock",
            "-e", f"ssh -p {self.ssh_port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            f"{local_path}/",
            f"{self.ssh_user}@{self.ssh_host}:{self.remote_path}/",
        ]

        # Run rsync locally
        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("✓ Code synced successfully")
        else:
            raise RuntimeError(f"Code sync failed: {result.stderr}")

    async def setup_environment(self):
        """Setup Python environment on remote pod.

        Steps:
        0. Fix /ephemeral permissions (if exists)
        1. Install uv (if not present)
        2. Run uv sync in research/trl
        3. Install Flash Attention 2 (for faster training)
        4. Run uv sync in tools/vllm-server
        5. Login to HuggingFace (for gated models like Llama)
        """
        logger.info("Setting up Python environment")

        # Step 0: Fix /ephemeral permissions if the directory exists
        # Many cloud instances mount /ephemeral as root-owned
        logger.info("  [0/5] Fixing /ephemeral permissions (if exists)...")
        await self.run_command(
            "test -d /ephemeral && sudo chown -R $(whoami):$(whoami) /ephemeral || true",
            timeout=30,
            check=False,
        )

        # Step 1: Install uv (new installer puts it in ~/.local/bin)
        logger.info("  [1/5] Installing uv...")
        await self.run_command(
            "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh",
            timeout=180,
            check=False,
        )

        # Determine uv path (could be ~/.local/bin/uv or ~/.cargo/bin/uv)
        uv_cmd = "$(command -v uv || echo $HOME/.local/bin/uv)"

        # Step 2: Sync TRL research dependencies (with all extras)
        logger.info("  [2/5] Installing TRL research dependencies (with --all-extras)...")
        await self.run_command(
            f"cd {self.remote_path}/research/trl && {uv_cmd} sync --all-extras",
            timeout=600,
        )

        # Step 3: Install Flash Attention 2 for faster training
        # Uses --no-build-isolation as recommended for flash-attn with uv
        # See: https://github.com/astral-sh/uv/issues/6437
        logger.info("  [3/5] Installing Flash Attention 2...")
        await self._install_flash_attention(uv_cmd)

        # Step 4: Sync vllm-server dependencies (with all extras)
        logger.info("  [4/5] Installing VLLM server dependencies (with --all-extras)...")
        await self.run_command(
            f"cd {self.remote_path}/tools/vllm-server && {uv_cmd} sync --all-extras",
            timeout=600,
        )

        # Step 5: Login to HuggingFace if token is available (required for gated models like Llama)
        logger.info("  [5/5] Setting up HuggingFace authentication...")
        await self._setup_huggingface_auth()

        logger.info("✓ Environment setup complete")

    async def _install_flash_attention(self, uv_cmd: str):
        """Install Flash Attention 2 for faster transformer training.

        Flash Attention 2 provides significant speedups for attention computation
        on NVIDIA GPUs. Installation requires --no-build-isolation flag with uv.

        See: https://github.com/Dao-AILab/flash-attention
        See: https://github.com/astral-sh/uv/issues/6437
        """
        # Check if flash-attn is already installed
        check_cmd = (
            f"cd {self.remote_path}/research/trl && "
            f"source .venv/bin/activate 2>/dev/null && "
            f"python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null"
        )
        stdout, _, exit_code = await self.run_command(check_cmd, check=False)

        if exit_code == 0 and stdout.strip():
            logger.info(f"  ✓ Flash Attention already installed (v{stdout.strip()})")
            return

        # Install flash-attn with --no-build-isolation (required for uv)
        # This allows flash-attn to find torch during build
        install_cmd = (
            f"cd {self.remote_path}/research/trl && "
            f"{uv_cmd} pip install flash-attn --no-build-isolation"
        )

        try:
            await self.run_command(install_cmd, timeout=600, check=True)
            logger.info("  ✓ Flash Attention 2 installed successfully")
        except Exception as e:
            # Flash attention is optional - training will work without it (just slower)
            logger.warning(
                f"  ⚠ Flash Attention installation failed (training will use default attention): {e}"
            )

    async def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication on remote pod.

        Reads HF_TOKEN from the .env file and logs in using huggingface-cli.
        This is required for downloading gated models like Llama.
        """
        # Extract HF_TOKEN from .env file on remote pod
        hf_token_cmd = f"grep -E '^HF_TOKEN=' {self.remote_path}/.env 2>/dev/null | cut -d'=' -f2 | tr -d '\"' | tr -d \"'\""
        stdout, _, exit_code = await self.run_command(hf_token_cmd, check=False)

        hf_token = stdout.strip()
        if not hf_token:
            logger.warning("HF_TOKEN not found in .env - gated models may not be accessible")
            return

        # Login to HuggingFace using the token
        # Use huggingface-cli which should be installed with transformers
        login_cmd = (
            f"cd {self.remote_path}/research/trl && "
            f"source .venv/bin/activate 2>/dev/null || true && "
            f"python -c \"from huggingface_hub import login; login(token='{hf_token}', add_to_git_credential=False)\""
        )

        try:
            await self.run_command(login_cmd, timeout=60, check=True)
            logger.info("✓ HuggingFace authentication configured")
        except Exception as e:
            logger.warning(f"HuggingFace login failed (gated models may not work): {e}")

    async def start_training(self, config: ExperimentConfig) -> str:
        """Start nohup training on remote pod.

        Args:
            config: Experiment configuration

        Returns:
            Path to PID file on remote pod
        """
        logger.info(f"Starting training: {config.name}")
        logger.info(f"  Trainer type: {config.trainer_type}")
        logger.info(f"  Model: {config.model_id}")
        if config.trainer_type == "grpo":
            logger.info(f"  Num Iterations: {config.num_iterations}")
        else:
            logger.info(f"  Max Steps: {config.max_steps}")
        logger.info(f"  Dataset: {config.dataset}")
        logger.info(f"  W&B Project: {config.wandb_project}")
        if config.run_prefix:
            logger.info(f"  Run Prefix: {config.run_prefix}")
        if config.seed is not None:
            logger.info(f"  Seed: {config.seed}")
        if config.learning_rate is not None:
            logger.info(f"  Learning Rate: {config.learning_rate}")

        # GPU logging differs by trainer type
        if config.trainer_type == "grpo":
            # GRPO uses 2 GPUs per instance (vLLM + trainer)
            logger.info(f"  Start Instance: {config.start_instance} (GPUs {config.start_instance*2},{config.start_instance*2+1})")
            logger.info(f"  Base Port: {config.base_port}")
        else:
            # SFT uses 1 GPU per instance
            logger.info(f"  Start Instance: {config.start_instance} (GPU {config.start_instance})")

        # Build environment variable exports
        env_exports = f"export WANDB_PROJECT='{config.wandb_project}' && "
        if config.wandb_tags:
            env_exports += f"export WANDB_TAGS='{config.wandb_tags}' && "
        # Use /ephemeral for outputs (large disk on Basilica instances)
        env_exports += "export GRAIL_OUTPUT_BASE='/ephemeral' && "

        # Port and instance configuration for running multiple experiments on same node
        if config.run_prefix:
            env_exports += f"export GRAIL_RUN_PREFIX='{config.run_prefix}' && "
        if config.seed is not None:
            env_exports += f"export GRAIL_SEED='{config.seed}' && "
        env_exports += f"export GRAIL_START_INSTANCE='{config.start_instance}' && "

        # GRPO-specific port configuration (SFT doesn't need these)
        if config.trainer_type == "grpo":
            env_exports += f"export GRAIL_BASE_PORT='{config.base_port}' && "
            env_exports += f"export GRAIL_BASE_GROUP_PORT='{config.base_group_port}' && "
            env_exports += f"export GRAIL_VLLM_NIXL_PORT_BASE='{config.vllm_nixl_port_base}' && "
            env_exports += f"export GRAIL_VLLM_MASTER_PORT_BASE='{config.vllm_master_port_base}' && "

        if config.learning_rate is not None:
            env_exports += f"export GRAIL_TRAINER_LR='{config.learning_rate}' && "

        # Source .env file to get HF_TOKEN and other environment variables
        # This ensures HuggingFace token is available for gated model downloads
        source_env = (
            f"set -a && "
            f"[ -f {self.remote_path}/.env ] && source {self.remote_path}/.env && "
            f"set +a && "
        )

        # Build script command based on trainer type
        batch_size_arg = str(config.batch_size) if config.batch_size is not None else ""
        grad_accum_arg = str(config.grad_accum_steps) if config.grad_accum_steps is not None else ""

        if config.trainer_type == "grpo":
            # GRPO: run_parallel_training_nohup.sh
            # Args: dataset, eval_every, model_id, num_iterations, num_instances, batch_size, grad_accum_steps
            script_cmd = (
                f"cd {self.remote_path}/research/trl && "
                f"{source_env}"
                f"{env_exports}"
                f"./run_parallel_training_nohup.sh "
                f"{config.dataset} {config.eval_every} {config.model_id} {config.num_iterations} {config.num_instances} "
                f"{batch_size_arg} {grad_accum_arg}"
            )
            pid_log_dir = "parallel_training"
        else:
            # SFT: run_parallel_sft_nohup.sh
            # Args: dataset, eval_every, model_id, max_steps, num_instances, batch_size, grad_accum_steps
            script_cmd = (
                f"cd {self.remote_path}/research/trl && "
                f"{source_env}"
                f"{env_exports}"
                f"./run_parallel_sft_nohup.sh "
                f"{config.dataset} {config.eval_every} {config.model_id} {config.max_steps} {config.num_instances} "
                f"{batch_size_arg} {grad_accum_arg}"
            )
            pid_log_dir = "parallel_sft"

        _ = await self.run_command(
            script_cmd,
            timeout=30,
            check=True,
        )

        # PID file path uses run_prefix if provided
        pid_suffix = config.run_prefix if config.run_prefix else "default"
        pid_file = f"{self.remote_path}/research/trl/logs/{pid_log_dir}/launcher_{pid_suffix}.pid"

        logger.info(f"✓ Training started (PID file: {pid_file})")
        return pid_file

    async def monitor_training(
        self,
        pid_file: str,
        check_interval: int = 60,
        max_retries: int = 5,
    ):
        """Monitor training until completion with connection recovery.

        Polls the PID file to check if the process is still running.
        Automatically reconnects if SSH connection is lost.

        Args:
            pid_file: Path to PID file on remote pod
            check_interval: Seconds between checks (default: 60)
            max_retries: Maximum reconnection attempts (default: 5)
        """
        logger.info(f"Monitoring training (checking every {check_interval}s)")
        # Flush logs for visibility under nohup
        for handler in logging.getLogger().handlers:
            handler.flush()

        consecutive_failures = 0

        while True:
            try:
                # Check if PID file exists
                stdout, _, exit_code = await self.run_command(
                    f"test -f {pid_file} && cat {pid_file}",
                    check=False,
                )

                if exit_code != 0:
                    logger.warning("PID file not found, assuming training complete")
                    break

                pid = stdout.strip()

                # Check if process is still running
                _, _, ps_exit = await self.run_command(
                    f"ps -p {pid} > /dev/null 2>&1",
                    check=False,
                )

                if ps_exit != 0:
                    logger.info(f"✓ Training complete (PID {pid} no longer running)")
                    break

                logger.info(f"Training still running (PID {pid})")
                # Reset failure counter on successful check
                consecutive_failures = 0

            except (asyncssh.Error, OSError) as e:
                consecutive_failures += 1
                logger.warning(
                    f"SSH connection error (attempt {consecutive_failures}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                # Flush error logs
                for handler in logging.getLogger().handlers:
                    handler.flush()

                if consecutive_failures >= max_retries:
                    logger.error("Max reconnection attempts reached, giving up")
                    raise

                # Attempt to reconnect
                logger.info("Attempting to reconnect...")
                await self.disconnect()
                await asyncio.sleep(5)  # Brief pause before reconnect
                await self.connect()
                logger.info("Reconnected successfully")

            # Flush logs periodically for visibility under nohup
            for handler in logging.getLogger().handlers:
                handler.flush()

            await asyncio.sleep(check_interval)

    async def download_artifacts(self, local_download_dir: Path, output_base: str = "/ephemeral") -> Path:
        """Download training artifacts from remote pod.

        Downloads logs/, outputs/, and checkpoints/ directories.

        Args:
            local_download_dir: Local directory to download artifacts to
            output_base: Remote base directory for outputs (default: /ephemeral for Basilica)

        Returns:
            Path to downloaded artifacts directory
        """
        logger.info(f"Downloading artifacts to {local_download_dir}")
        logger.info(f"  Remote output base: {output_base}")

        # Create local directory
        local_download_dir.mkdir(parents=True, exist_ok=True)

        # Download each artifact directory from the output base
        for artifact_dir in ["logs", "outputs", "checkpoints"]:
            remote_dir = f"{output_base}/{artifact_dir}"
            local_dir = local_download_dir / artifact_dir

            # Check if remote directory exists
            _, _, exists = await self.run_command(
                f"test -d {remote_dir}",
                check=False,
            )

            if exists != 0:
                logger.warning(f"Remote directory not found, skipping: {remote_dir}")
                continue

            # Use rsync to download
            rsync_cmd = [
                "rsync",
                "-avz",
                f"-e", f"ssh -p {self.ssh_port} -o StrictHostKeyChecking=no",
                f"{self.ssh_user}@{self.ssh_host}:{remote_dir}/",
                f"{local_dir}/",
            ]

            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"✓ Downloaded: {artifact_dir}")
            else:
                logger.error(f"✗ Failed to download {artifact_dir}: {result.stderr}")

        logger.info(f"✓ Artifacts downloaded to {local_download_dir}")
        return local_download_dir

    async def upload_to_r2(self, local_dir: Path, experiment_name: str) -> bool:
        """Upload artifacts to R2 from local directory.

        Args:
            local_dir: Local directory containing artifacts
            experiment_name: Experiment name (used as R2 prefix)

        Returns:
            True if upload succeeded, False otherwise
        """
        logger.info(f"Uploading artifacts to R2: experiments/{experiment_name}/")

        success = upload_experiment_artifacts(
            local_base_dir=local_dir,
            experiment_name=experiment_name,
            bucket_id=self.r2_config["bucket_id"],
            account_id=self.r2_config["account_id"],
            access_key=self.r2_config["access_key"],
            secret_key=self.r2_config["secret_key"],
        )

        if success:
            logger.info("✓ Upload to R2 complete")
        else:
            logger.error("✗ R2 upload failed")

        return success

    async def upload_to_r2_remote(
        self, experiment_name: str, output_base: str = "/ephemeral"
    ) -> bool:
        """Upload artifacts directly from remote server to R2.

        This method runs the r2_uploader.py script on the remote server,
        avoiding the need to download artifacts to the local machine first.

        Args:
            experiment_name: Experiment name (used as R2 prefix)
            output_base: Remote base directory for outputs (default: /ephemeral)

        Returns:
            True if upload succeeded, False otherwise
        """
        logger.info(f"Uploading artifacts directly from remote to R2: experiments/{experiment_name}/")
        logger.info(f"  Remote output base: {output_base}")

        # First, ensure boto3 is installed in the TRL venv (required for r2_uploader)
        logger.info("  Ensuring boto3 is installed...")
        uv_cmd = "$(command -v uv || echo $HOME/.local/bin/uv)"
        install_cmd = (
            f"cd {self.remote_path}/research/trl && "
            f"{uv_cmd} pip install boto3 tqdm 2>/dev/null || "
            f"pip install boto3 tqdm 2>/dev/null || true"
        )
        await self.run_command(install_cmd, timeout=120, check=False)

        # Build the upload command using r2_uploader.py on the remote server
        # The r2_uploader expects: python r2_uploader.py <local_dir> <experiment_name>
        # where local_dir contains logs/, outputs/, checkpoints/ subdirectories
        upload_cmd = (
            f"cd {self.remote_path}/research/infrastructure && "
            f"source ../trl/.venv/bin/activate && "
            f"export R2_BUCKET_NAME='{self.r2_config['bucket_id']}' && "
            f"export R2_ACCOUNT_ID='{self.r2_config['account_id']}' && "
            f"export R2_WRITE_ACCESS_KEY_ID='{self.r2_config['access_key']}' && "
            f"export R2_WRITE_SECRET_ACCESS_KEY='{self.r2_config['secret_key']}' && "
            f"python r2_uploader.py {output_base} {experiment_name}"
        )

        try:
            stdout, stderr, exit_code = await self.run_command(
                upload_cmd,
                timeout=3600,  # 1 hour timeout for large uploads
                check=False,
            )

            if exit_code == 0:
                logger.info("✓ Remote upload to R2 complete")
                return True
            else:
                logger.error(f"✗ Remote R2 upload failed (exit code {exit_code})")
                logger.error(f"  stdout: {stdout[:1000] if stdout else 'empty'}")
                logger.error(f"  stderr: {stderr[:1000] if stderr else 'empty'}")
                return False

        except Exception as e:
            logger.error(f"✗ Remote R2 upload failed: {type(e).__name__}: {e}")
            return False

    async def run_experiment(
        self,
        config: ExperimentConfig,
        local_code_path: Path,
        local_env_path: Optional[Path] = None,
        sync_code: bool = True,
        setup_env: bool = True,
        download_dir: Optional[Path] = None,
        upload_to_r2: bool = True,
        cleanup_local: bool = False,
        remote_upload: bool = True,
    ) -> bool:
        """Run complete experiment workflow.

        Steps:
        1. Connect to pod
        2. Sync code (if sync_code=True)
        3. Setup environment (if setup_env=True)
        4. Start training
        5. Monitor until completion
        6. Upload to R2 directly from remote (if upload_to_r2=True and remote_upload=True)
           OR download artifacts then upload (if remote_upload=False)
        7. Cleanup (if cleanup_local=True)

        Args:
            config: Experiment configuration
            local_code_path: Local path to grail repository
            local_env_path: Local path to .env file (optional)
            sync_code: Whether to sync code to pod (default: True)
            setup_env: Whether to setup environment (default: True)
            download_dir: Local directory for downloads (default: ./downloads/{experiment_name})
            upload_to_r2: Whether to upload results to R2 (default: True)
            cleanup_local: Whether to delete local artifacts after R2 upload (default: False)
            remote_upload: Whether to upload directly from remote server (default: True)
                          If False, downloads to local first then uploads.

        Returns:
            True if experiment completed successfully, False otherwise
        """
        try:
            await self.connect()

            # Sync code
            if sync_code:
                await self.sync_code(local_code_path)

            # Copy .env file if provided
            if local_env_path and local_env_path.exists():
                logger.info("Copying .env file to remote pod")
                env_content = local_env_path.read_text()
                await self.run_command(
                    f"cat > {self.remote_path}/.env << 'ENV_EOF'\n{env_content}\nENV_EOF"
                )

            # Setup environment
            if setup_env:
                await self.setup_environment()

            # Start training
            pid_file = await self.start_training(config)

            # Monitor until completion
            await self.monitor_training(pid_file)

            # Upload to R2
            if upload_to_r2:
                if remote_upload:
                    # Upload directly from remote server (no local download needed)
                    upload_success = await self.upload_to_r2_remote(
                        experiment_name=config.name,
                        output_base="/ephemeral",
                    )
                else:
                    # Legacy: download to local, then upload
                    download_dir = download_dir or Path(f"./downloads/{config.name}")
                    local_artifacts_dir = await self.download_artifacts(download_dir)
                    upload_success = await self.upload_to_r2(local_artifacts_dir, config.name)

                    # Cleanup local files if requested
                    if cleanup_local and upload_success:
                        logger.info(f"Cleaning up local artifacts: {local_artifacts_dir}")
                        shutil.rmtree(local_artifacts_dir)

                if not upload_success:
                    logger.error("R2 upload failed")
                    return False

            logger.info(f"✓ Experiment {config.name} completed successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Experiment {config.name} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Flush logs to ensure error is captured (especially under nohup)
            for handler in logging.getLogger().handlers:
                handler.flush()
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            return False

        finally:
            await self.disconnect()


if __name__ == "__main__":
    # Example usage
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        if len(sys.argv) < 3:
            print("Usage: python nohup_experiment_runner.py <ssh_host> <ssh_port>")
            print(
                "  Requires env vars: R2_BUCKET_NAME (preferred) or R2_BUCKET_ID (legacy), "
                "R2_ACCOUNT_ID, R2_WRITE_ACCESS_KEY_ID, R2_WRITE_SECRET_ACCESS_KEY"
            )
            sys.exit(1)

        bucket_name = os.getenv("R2_BUCKET_NAME") or os.getenv("R2_BUCKET_ID")
        if not bucket_name:
            print("Missing env var: set R2_BUCKET_NAME (preferred) or R2_BUCKET_ID (legacy)")
            sys.exit(1)

        runner = NohupExperimentRunner(
            ssh_host=sys.argv[1],
            ssh_port=int(sys.argv[2]),
            r2_config={
                # Legacy key name; value must be the *bucket name* for the S3 API.
                "bucket_id": bucket_name,
                "account_id": os.environ["R2_ACCOUNT_ID"],
                "access_key": os.environ["R2_WRITE_ACCESS_KEY_ID"],
                "secret_key": os.environ["R2_WRITE_SECRET_ACCESS_KEY"],
            },
        )

        config = ExperimentConfig(
            name="test-run",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            num_iterations=1,
        )

        success = await runner.run_experiment(
            config=config,
            local_code_path=Path("/home/ubuntu/grail"),
            local_env_path=Path("/home/ubuntu/grail/.env"),
        )

        sys.exit(0 if success else 1)

    asyncio.run(main())
