"""Basilica infrastructure manager for GPU instance provisioning.

Provides declarative infrastructure management using the Basilica CLI,
with automatic health verification and instance replacement.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Timeouts (seconds)
# GPU instances can take 10-20 minutes to fully provision
SSH_CONNECT_TIMEOUT = 900  # 15 minutes - wait for SSH port to open
SSH_VERIFY_TIMEOUT = 60  # 1 minute - verify SSH connection works
PROVISION_TIMEOUT = 600  # 10 minutes - wait for rental creation
SSH_POLL_INTERVAL = 15  # Poll every 15 seconds
MAX_PROVISION_POLLS = 80  # 80 * 15 = 20 minutes max wait for SSH info


@dataclass
class GpuConfig:
    """GPU instance configuration."""

    name: str
    gpu_type: str = "A100"
    gpu_count: int = 1
    compute_type: str = "secure-cloud"
    country: str | None = None
    max_price_per_gpu: float | None = None

    @property
    def total_hourly_cost(self) -> float | None:
        """Estimated total hourly cost if max_price_per_gpu is set."""
        if self.max_price_per_gpu:
            return self.max_price_per_gpu * self.gpu_count
        return None


@dataclass
class SshConnection:
    """SSH connection details."""

    host: str
    port: int
    user: str = "ubuntu"

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "user": self.user}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SshConnection:
        return cls(host=data["host"], port=data["port"], user=data.get("user", "root"))


@dataclass
class Instance:
    """A provisioned GPU instance."""

    id: str
    name: str
    config: GpuConfig
    ssh: SshConnection

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "config": {
                "gpu_type": self.config.gpu_type,
                "gpu_count": self.config.gpu_count,
                "compute_type": self.config.compute_type,
            },
            "ssh": self.ssh.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        config = GpuConfig(
            name=data["name"],
            gpu_type=data["config"]["gpu_type"],
            gpu_count=data["config"]["gpu_count"],
            compute_type=data["config"].get("compute_type", "secure-cloud"),
        )
        return cls(
            id=data["id"],
            name=data["name"],
            config=config,
            ssh=SshConnection.from_dict(data["ssh"]),
        )


@dataclass
class AvailableGpu:
    """An available GPU offering from basilica ls."""

    id: str
    provider: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    region: str
    hourly_rate_per_gpu: float
    available: bool
    interconnect: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AvailableGpu:
        return cls(
            id=data["id"],
            provider=data["provider"],
            gpu_type=data["gpu_type"],
            gpu_count=data["gpu_count"],
            gpu_memory_gb=data.get("gpu_memory_gb_per_gpu", 0),
            region=data["region"],
            hourly_rate_per_gpu=float(data["hourly_rate_per_gpu"]),
            available=data.get("availability", False),
            interconnect=data.get("interconnect"),
        )


class BasilicaCli:
    """Wrapper for basilica CLI commands."""

    @staticmethod
    def run(args: list[str], timeout: int = 120) -> dict[str, Any] | list[Any]:
        """Execute a basilica command and return JSON result."""
        cmd = ["basilica", *args, "--json"]
        logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"basilica {args[0]} failed: {error}")

        return json.loads(result.stdout)

    @staticmethod
    def run_interactive(
        args: list[str], timeout: int = 120, select_first: bool = True
    ) -> dict[str, Any] | list[Any]:
        """Execute a basilica command that may require interactive selection.

        Uses `expect` to automate selection when the CLI prompts for user input.
        This is necessary for commands like `basilica up` when multiple GPU
        offerings match the criteria.

        Args:
            args: Command arguments (without 'basilica' prefix)
            timeout: Command timeout in seconds
            select_first: If True, automatically select the first option

        Returns:
            Parsed JSON response from the command
        """
        import shutil
        import tempfile

        cmd_str = " ".join(["basilica", *args, "--json"])
        logger.debug(f"Running interactive: {cmd_str}")

        # Check if expect is available
        expect_path = shutil.which("expect")
        if not expect_path:
            raise RuntimeError("'expect' command not found - required for non-interactive provisioning")

        # Create expect script to auto-select first option
        # The script waits for the selection prompt and sends Enter to select first option
        expect_script = f'''
set timeout {timeout}
spawn {cmd_str}
expect {{
    "Select offering" {{
        send "\\r"
        exp_continue
    }}
    eof
}}
catch wait result
exit [lindex $result 3]
'''

        # Write expect script to temp file and execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".exp", delete=False) as f:
            f.write(expect_script)
            expect_file = f.name

        try:
            result = subprocess.run(
                [expect_path, expect_file],
                capture_output=True,
                text=True,
                timeout=timeout + 30,  # Extra time for expect overhead
            )

            # Clean ANSI escape codes and control characters from output
            import re
            clean_output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\[\?[0-9]*[a-zA-Z]', '', result.stdout)
            clean_output = re.sub(r'[\x00-\x1f\x7f]', '', clean_output)

            # Extract rental ID from the success message
            # Format: "Successfully started rental <uuid>"
            rental_match = re.search(r'Successfully started rental ([a-f0-9-]{36})', clean_output)
            if rental_match:
                rental_id = rental_match.group(1)
                logger.info(f"Extracted rental ID from output: {rental_id}")
                # Return a properly structured response
                return {"rental": {"id": rental_id}}

            # Fallback: try to find JSON in output
            json_match = re.search(r'\{[^{}]*"rental"[^{}]*\{[^{}]*\}[^{}]*\}', clean_output)
            if not json_match:
                # Try simpler JSON pattern
                json_match = re.search(r'\{.*\}', clean_output, re.DOTALL)

            if result.returncode != 0 or not json_match:
                error = result.stderr.strip() or clean_output.strip()
                raise RuntimeError(f"basilica {args[0]} failed: {error}")

            return json.loads(json_match.group())

        finally:
            Path(expect_file).unlink(missing_ok=True)

    @classmethod
    def list_gpus(
        cls,
        gpu_type: str | None = None,
        gpu_min: int | None = None,
        compute_type: str = "secure-cloud",
    ) -> list[AvailableGpu]:
        """List available GPU instances."""
        args = ["ls", "--compute", compute_type]
        if gpu_type:
            args.append(gpu_type.lower())
        if gpu_min:
            args.extend(["--gpu-min", str(gpu_min)])

        data = cls.run(args)

        # Handle different response formats
        items = data if isinstance(data, list) else data.get("secure_cloud", [])
        return [AvailableGpu.from_json(item) for item in items]

    @classmethod
    def list_rentals(cls, compute_type: str | None = None) -> list[dict[str, Any]]:
        """List active rentals."""
        args = ["ps"]
        if compute_type:
            args.extend(["--compute", compute_type])

        data = cls.run(args)

        # Handle different response formats
        if "rentals" in data:
            return data["rentals"]

        rentals = []
        for cloud in ["secure_cloud", "community_cloud"]:
            rentals.extend(data.get(cloud, {}).get("rentals", []))
        return rentals

    @classmethod
    def start_rental(
        cls,
        gpu_type: str,
        gpu_count: int,
        name: str | None = None,
        compute_type: str = "secure-cloud",
    ) -> dict[str, Any]:
        """Start a new GPU rental.

        Uses run_interactive to handle cases where multiple GPU offerings
        match the criteria and the CLI prompts for selection.
        """
        args = [
            "up",
            gpu_type.lower(),
            "--compute", compute_type,
            "--gpu-count", str(gpu_count),
            "-d",  # detached mode
        ]
        if name:
            args.extend(["--name", name])

        # Use run_interactive to handle selection prompts
        return cls.run_interactive(args, timeout=PROVISION_TIMEOUT)

    @classmethod
    def stop_rental(cls, rental_id: str) -> None:
        """Stop a rental."""
        subprocess.run(
            ["basilica", "down", rental_id],
            capture_output=True,
            timeout=60,
        )

    @classmethod
    def get_status(cls, rental_id: str) -> dict[str, Any] | None:
        """Get rental status including SSH info."""
        try:
            return cls.run(["status", rental_id])
        except Exception:
            return None


def wait_for_ssh(host: str, port: int, timeout: int = SSH_CONNECT_TIMEOUT) -> bool:
    """Wait for SSH port to become accessible."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except (socket.error, OSError):
            time.sleep(5)
    return False


def verify_ssh(host: str, port: int, user: str = "root", timeout: int = SSH_VERIFY_TIMEOUT) -> bool:
    """Verify SSH connection works with a test command.

    Args:
        host: SSH hostname or IP
        port: SSH port
        user: SSH username (defaults to root for Basilica instances)
        timeout: Maximum time to wait for connection

    Returns:
        True if SSH connection succeeded
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
                "-p", str(port),
                f"{user}@{host}",
                "echo SSH_OK",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "SSH_OK" in result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.debug(f"SSH verification failed for {user}@{host}:{port}: {e}")
        return False


class BasilicaInfra:
    """Declarative infrastructure manager for Basilica GPU instances."""

    def __init__(self, state_file: str = ".basilica_state.json") -> None:
        self.state_file = Path(state_file)
        self._instances: dict[str, Instance] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk with corruption recovery."""
        if not self.state_file.exists():
            return

        try:
            data = json.loads(self.state_file.read_text())
            for name, inst_data in data.get("instances", {}).items():
                try:
                    self._instances[name] = Instance.from_dict(inst_data)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping corrupted instance {name}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"State file corrupted: {e}")
            # Try backup file
            backup = Path(str(self.state_file) + ".bak")
            if backup.exists():
                try:
                    data = json.loads(backup.read_text())
                    for name, inst_data in data.get("instances", {}).items():
                        self._instances[name] = Instance.from_dict(inst_data)
                    logger.info(f"Recovered state from backup")
                except Exception as e2:
                    logger.error(f"Backup also corrupted: {e2}")
                    logger.warning("Starting with empty state - may have orphaned instances!")

    def _save_state(self) -> None:
        """Persist current state to disk with atomic write and backup."""
        data = {"instances": {n: i.to_dict() for n, i in self._instances.items()}}
        content = json.dumps(data, indent=2)

        # Create backup of existing state
        if self.state_file.exists():
            backup = Path(str(self.state_file) + ".bak")
            try:
                backup.write_text(self.state_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Atomic write: write to temp file then rename
        temp_file = Path(str(self.state_file) + ".tmp")
        try:
            temp_file.write_text(content)
            temp_file.rename(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            raise

    def list_available(
        self,
        gpu_type: str | None = None,
        min_gpus: int | None = None,
    ) -> list[AvailableGpu]:
        """List available GPU instances."""
        return BasilicaCli.list_gpus(gpu_type=gpu_type, gpu_min=min_gpus)

    def get_instance(self, name: str) -> Instance | None:
        """Get a managed instance by name."""
        return self._instances.get(name)

    def list_instances(self) -> dict[str, Instance]:
        """List all managed instances."""
        return dict(self._instances)

    def _find_rental_by_name(self, name: str) -> dict | None:
        """Find an existing rental by name."""
        try:
            rentals = BasilicaCli.list_rentals()
            for rental in rentals:
                if rental.get("name") == name:
                    return rental
        except Exception as e:
            logger.warning(f"Failed to list rentals: {e}")
        return None

    def provision(self, config: GpuConfig, max_retries: int = 3) -> Instance | None:
        """Provision a new GPU instance.

        IMPORTANT: Once a rental is successfully created, we NEVER retry.
        We wait for that rental to be ready, even if it takes a long time.
        Retries only happen if the rental creation itself fails.

        Args:
            config: GPU configuration specifying type and count
            max_retries: Maximum attempts to CREATE a rental (not to wait for it)

        Returns:
            Instance if successful, None otherwise
        """
        # Remove existing instance with same name from our state
        if config.name in self._instances:
            self.terminate(config.name)

        rental_id = None

        # Try to create a rental (retry only if creation fails)
        for attempt in range(max_retries):
            print(f"   🔄 Attempt {attempt + 1}/{max_retries}: {config.gpu_count}x {config.gpu_type}")

            try:
                # Start rental
                response = BasilicaCli.start_rental(
                    gpu_type=config.gpu_type,
                    gpu_count=config.gpu_count,
                    name=config.name,
                    compute_type=config.compute_type,
                )

                rental_id = response.get("rental", response).get("id")
                if rental_id:
                    print(f"   🆔 Rental created: {rental_id}")
                    break  # SUCCESS - rental created, exit retry loop
                else:
                    print(f"   ❌ No rental ID in response: {response}")

            except Exception as e:
                print(f"   ⚠️  Error during rental creation: {e}")
                # Don't retry immediately - the rental might have been created
                # Wait a moment and check
                time.sleep(2)

        if not rental_id:
            print(f"   ❌ Failed to create rental after {max_retries} attempts")
            return None

        # Rental created - now wait for it to be ready (no retries, just wait)
        print(f"   ⏳ Waiting for rental {rental_id[:8]}... to provision and become ready...")
        instance = self._wait_for_rental_ready(rental_id, config)

        if not instance:
            print(f"   ❌ Rental {rental_id[:8]}... failed to become ready")
            # Optionally terminate the failed rental
            print(f"   🗑️  Terminating failed rental...")
            BasilicaCli.stop_rental(rental_id)

        return instance

    def _wait_for_rental_ready(self, rental_id: str, config: GpuConfig) -> Instance | None:
        """Wait for a rental to be ready and return an Instance.

        Args:
            rental_id: The rental ID to wait for
            config: GPU configuration

        Returns:
            Instance if ready, None if failed
        """
        print(f"   ⏳ Waiting for rental {rental_id[:8]}... to be ready...")

        try:
            # Wait for SSH info (polls until provisioning complete)
            ssh_info = self._wait_for_ssh_info(rental_id)
            if not ssh_info:
                print(f"   ❌ SSH info not available after provisioning")
                return None

            host, port = ssh_info

            # Wait for SSH connectivity (port open)
            print(f"   ⏳ Waiting for SSH port ({host}:{port})...")
            if not wait_for_ssh(host, port):
                print(f"   ❌ SSH port connection timeout")
                return None

            # Verify SSH login works (retry for up to 20 minutes - system boot can take time)
            # Basilica instances often show "System is booting up" even after port is open
            print(f"   🔍 Verifying SSH login (may take 10-20 min for system boot)...")
            ssh_user = None
            max_ssh_verify_attempts = 80  # 80 * 15s = 20 minutes
            for attempt in range(max_ssh_verify_attempts):
                # Try ubuntu first (most common on Basilica), then root
                for user in ["ubuntu", "root"]:
                    if verify_ssh(host, port, user=user):
                        ssh_user = user
                        break
                if ssh_user:
                    break
                if attempt % 4 == 0:  # Log every minute
                    print(f"   ⏳ SSH login not ready yet ({attempt + 1}/{max_ssh_verify_attempts})...")
                time.sleep(15)

            if not ssh_user:
                print(f"   ❌ SSH verification failed after {max_ssh_verify_attempts} attempts")
                return None

            # Success
            instance = Instance(
                id=rental_id,
                name=config.name,
                config=config,
                ssh=SshConnection(host=host, port=port, user=ssh_user),
            )
            self._instances[config.name] = instance
            self._save_state()

            print(f"   ✅ Ready: {host}:{port}")
            return instance

        except Exception as e:
            print(f"   ❌ Error waiting for rental: {e}")
            return None

    def _wait_for_ssh_info(
        self,
        rental_id: str,
        max_polls: int = MAX_PROVISION_POLLS,
    ) -> tuple[str, int] | None:
        """Poll until SSH connection info is available."""
        for i in range(max_polls):
            status = BasilicaCli.get_status(rental_id)
            if not status:
                time.sleep(SSH_POLL_INTERVAL)
                continue

            # Check status - it's a string like "running", "provisioning", "failed"
            rental_status = status.get("status", "")
            if rental_status == "failed":
                print(f"   ❌ Rental failed")
                return None

            # Extract SSH info from ip_address field
            # ssh_command format: "ssh ubuntu@185.216.21.60" (port 22 implied)
            ip_address = status.get("ip_address")
            if ip_address:
                # Default to port 22 for basilica instances
                return ip_address, 22

            # Still provisioning
            if rental_status == "provisioning":
                print(f"   ⏳ Provisioning... ({i + 1}/{max_polls})")
            else:
                print(f"   ⏳ Waiting for SSH info ({i + 1}/{max_polls})... status={rental_status}")
            time.sleep(SSH_POLL_INTERVAL)

        return None

    def terminate(self, name: str) -> bool:
        """Terminate an instance by name."""
        instance = self._instances.get(name)
        if not instance:
            return True

        print(f"🗑️  Terminating: {name}")
        try:
            BasilicaCli.stop_rental(instance.id)
        except Exception as e:
            print(f"   ⚠️  {e}")

        del self._instances[name]
        self._save_state()
        return True

    def terminate_all(self) -> bool:
        """Terminate all managed instances."""
        success = True
        for name in list(self._instances.keys()):
            if not self.terminate(name):
                success = False
        return success

    def apply(self, configs: list[GpuConfig]) -> dict[str, Instance]:
        """Apply desired state - provision/terminate to match configs.

        Args:
            configs: List of desired GPU configurations

        Returns:
            Dict of successfully provisioned instances
        """
        desired = {c.name: c for c in configs}

        # Terminate instances not in desired state
        for name in list(self._instances.keys()):
            if name not in desired:
                self.terminate(name)

        # Find existing rentals by name
        existing_rentals = {r.get("name"): r for r in BasilicaCli.list_rentals() if r.get("name")}

        # Provision or verify each config
        result = {}
        for name, config in desired.items():
            # Check if already running
            if name in existing_rentals and name in self._instances:
                print(f"✅ Already running: {name}")
                result[name] = self._instances[name]
                continue

            # Provision new instance
            print(f"🚀 Provisioning: {name} ({config.gpu_count}x {config.gpu_type})")
            instance = self.provision(config)
            if instance:
                result[name] = instance

        return result

    def verify_health(self, configs: list[GpuConfig], max_retries: int = 3) -> dict[str, Instance]:
        """Verify all instances are healthy, replacing unhealthy ones.

        Args:
            configs: List of GPU configurations to verify
            max_retries: Max retries for unhealthy instances

        Returns:
            Dict of healthy instances
        """
        healthy = {}

        for config in configs:
            instance = self._instances.get(config.name)

            if not instance:
                print(f"🚀 Creating: {config.name}")
                new_instance = self.provision(config, max_retries=max_retries)
                if new_instance:
                    healthy[config.name] = new_instance
                continue

            # Verify existing instance
            print(f"🔍 Verifying: {config.name} ({instance.ssh.host}:{instance.ssh.port})")
            if verify_ssh(instance.ssh.host, instance.ssh.port, user=instance.ssh.user):
                print(f"   ✅ Healthy")
                healthy[config.name] = instance
            else:
                print(f"   ❌ Unhealthy - replacing...")
                new_instance = self.provision(config, max_retries=max_retries)
                if new_instance:
                    healthy[config.name] = new_instance

        return healthy


# Compatibility alias for deploy_parallel_basilica.py
PodSpec = GpuConfig
