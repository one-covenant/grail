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
    num_iterations: int  # Number of GRPO training iterations
    dataset: str = "math"  # Dataset to train on
    eval_every: int = 40  # Evaluation interval
    wandb_project: str = "grail-lium-sweep"  # W&B project for logging
    wandb_tags: str = ""  # Comma-separated W&B tags


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
        1. Install uv (if not present)
        2. Run uv sync in research/trl
        3. Run uv sync in tools/vllm-server
        """
        logger.info("Setting up Python environment")

        # Step 1: Install uv (new installer puts it in ~/.local/bin)
        logger.info("  [1/3] Installing uv...")
        await self.run_command(
            "command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh",
            timeout=180,
            check=False,
        )

        # Determine uv path (could be ~/.local/bin/uv or ~/.cargo/bin/uv)
        uv_cmd = "$(command -v uv || echo $HOME/.local/bin/uv)"

        # Step 2: Sync TRL research dependencies (with all extras)
        logger.info("  [2/3] Installing TRL research dependencies (with --all-extras)...")
        await self.run_command(
            f"cd {self.remote_path}/research/trl && {uv_cmd} sync --all-extras",
            timeout=600,
        )

        # Step 3: Sync vllm-server dependencies (with all extras)
        logger.info("  [3/3] Installing VLLM server dependencies (with --all-extras)...")
        await self.run_command(
            f"cd {self.remote_path}/tools/vllm-server && {uv_cmd} sync --all-extras",
            timeout=600,
        )

        logger.info("✓ Environment setup complete")

    async def start_training(self, config: ExperimentConfig) -> str:
        """Start nohup training on remote pod.

        Args:
            config: Experiment configuration

        Returns:
            Path to PID file on remote pod
        """
        logger.info(f"Starting training: {config.name}")
        logger.info(f"  Model: {config.model_id}")
        logger.info(f"  Num Iterations: {config.num_iterations}")
        logger.info(f"  Dataset: {config.dataset}")
        logger.info(f"  W&B Project: {config.wandb_project}")

        # Build environment variable exports for W&B
        env_exports = f"export WANDB_PROJECT='{config.wandb_project}' && "
        if config.wandb_tags:
            env_exports += f"export WANDB_TAGS='{config.wandb_tags}' && "

        # Run the nohup script with W&B config
        script_cmd = (
            f"cd {self.remote_path}/research/trl && "
            f"{env_exports}"
            f"./run_parallel_training_nohup.sh "
            f"{config.dataset} {config.eval_every} {config.model_id} {config.num_iterations}"
        )

        stdout, stderr, exit_code = await self.run_command(
            script_cmd,
            timeout=30,
            check=True,
        )

        # Extract PID file path from output
        pid_file = f"{self.remote_path}/research/trl/logs/parallel_training/launcher.pid"

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

    async def download_artifacts(self, local_download_dir: Path) -> Path:
        """Download training artifacts from remote pod.

        Downloads logs/, outputs/, and checkpoints/ directories.

        Args:
            local_download_dir: Local directory to download artifacts to

        Returns:
            Path to downloaded artifacts directory
        """
        logger.info(f"Downloading artifacts to {local_download_dir}")

        # Create local directory
        local_download_dir.mkdir(parents=True, exist_ok=True)

        # Download each artifact directory
        for artifact_dir in ["logs", "outputs", "checkpoints"]:
            remote_dir = f"{self.remote_path}/research/trl/{artifact_dir}"
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
        """Upload artifacts to R2.

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
    ) -> bool:
        """Run complete experiment workflow.

        Steps:
        1. Connect to pod
        2. Sync code (if sync_code=True)
        3. Setup environment (if setup_env=True)
        4. Start training
        5. Monitor until completion
        6. Download artifacts
        7. Upload to R2 (if upload_to_r2=True)
        8. Cleanup (if cleanup_local=True)

        Args:
            config: Experiment configuration
            local_code_path: Local path to grail repository
            local_env_path: Local path to .env file (optional)
            sync_code: Whether to sync code to pod (default: True)
            setup_env: Whether to setup environment (default: True)
            download_dir: Local directory for downloads (default: ./downloads/{experiment_name})
            upload_to_r2: Whether to upload results to R2 (default: True)
            cleanup_local: Whether to delete local artifacts after R2 upload (default: False)

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

            # Download artifacts
            download_dir = download_dir or Path(f"./downloads/{config.name}")
            local_artifacts_dir = await self.download_artifacts(download_dir)

            # Upload to R2
            if upload_to_r2:
                upload_success = await self.upload_to_r2(local_artifacts_dir, config.name)
                if not upload_success:
                    logger.error("R2 upload failed, keeping local artifacts")
                    return False

            # Cleanup local files if requested
            if cleanup_local and upload_to_r2:
                logger.info(f"Cleaning up local artifacts: {local_artifacts_dir}")
                shutil.rmtree(local_artifacts_dir)

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
