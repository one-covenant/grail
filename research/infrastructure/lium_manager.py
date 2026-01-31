"""Lium infrastructure manager for distributed training experiments.

This module provides declarative infrastructure management for Lium pods,
including bandwidth filtering, state management, and pod lifecycle operations.

Features:
- Declarative pod management with automatic retry on SSH failures
- Failed pod replacement without affecting healthy pods
- Bandwidth filtering and executor selection
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from lium import Config, ExecutorInfo, Lium

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_SSH_TIMEOUT = 180
DEFAULT_SSH_VERIFY_TIMEOUT = 30


def _wait_for_ssh(host: str, port: int, timeout: int = 120, interval: int = 5) -> bool:
    """Wait for SSH port to be accessible.

    Args:
        host: SSH host
        port: SSH port
        timeout: Maximum wait time in seconds
        interval: Check interval in seconds

    Returns:
        True if SSH is accessible, False if timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except (socket.error, OSError):
            pass
        time.sleep(interval)
    return False


def _verify_ssh_connection(host: str, port: int, timeout: int = DEFAULT_SSH_VERIFY_TIMEOUT) -> bool:
    """Verify SSH connection works by running a simple command.

    Args:
        host: SSH host
        port: SSH port
        timeout: Command timeout in seconds

    Returns:
        True if SSH connection works, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                "-p", str(port),
                f"root@{host}",
                "echo 'SSH_OK'"
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0 and "SSH_OK" in result.stdout
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.debug(f"SSH verification failed for {host}:{port}: {e}")
        return False


@dataclass
class PodSpec:
    """Specification for a Lium pod.

    Attributes:
        name: Unique pod name identifier
        gpu_type: GPU type (e.g., "A100", "H100")
        gpu_count: Number of GPUs per pod
        country: Optional country filter for executor location
        volume_id: Optional persistent volume ID to attach
        ttl_hours: Optional time-to-live in hours (auto-terminate)
        min_upload_mbps: Minimum upload bandwidth in Mbps
        min_download_mbps: Minimum download bandwidth in Mbps
    """

    name: str
    gpu_type: str = "A100"
    gpu_count: int = 8
    country: Optional[str] = None
    volume_id: Optional[str] = None
    ttl_hours: Optional[int] = None
    # Bandwidth requirements (in Mbps)
    min_upload_mbps: Optional[float] = None
    min_download_mbps: Optional[float] = None


class LiumInfra:
    """Declarative infrastructure manager for Lium with bandwidth filtering.

    Manages pod lifecycle with state persistence and bandwidth requirements.
    Supports declarative apply/destroy operations for cluster management.

    Example:
        >>> infra = LiumInfra()
        >>> pods = [
        ...     PodSpec(name="trainer-0", gpu_count=8, min_upload_mbps=500),
        ...     PodSpec(name="trainer-1", gpu_count=8, min_upload_mbps=500),
        ... ]
        >>> infra.apply(pods)
        >>> # Later...
        >>> infra.destroy()
    """

    def __init__(self, api_key: str | None = None, state_file: str = ".lium_state.json") -> None:
        """Initialize Lium infrastructure manager.

        Args:
            api_key: Optional Lium API key (defaults to Config.load())
            state_file: Path to state file for tracking managed pods
        """
        config = Config(api_key=api_key) if api_key else Config.load()
        self.lium = Lium(config)
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file or create empty state."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"pods": {}}

    def _save_state(self):
        """Persist current state to file."""
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _get_bandwidth(self, executor: ExecutorInfo) -> tuple[float, float]:
        """Extract upload/download bandwidth from executor specs (in Mbps).

        Args:
            executor: Executor info from Lium API

        Returns:
            Tuple of (upload_mbps, download_mbps)
        """
        specs = executor.specs

        # Lium.io stores network speeds in specs.network.{upload,download}_speed
        network_info = specs.get("network", {})
        upload = network_info.get("upload_speed", 0.0)
        download = network_info.get("download_speed", 0.0)

        return float(upload), float(download)

    def _find_executor(self, spec: PodSpec) -> ExecutorInfo | None:
        """Find best executor matching spec including bandwidth requirements.

        Args:
            spec: Pod specification with requirements

        Returns:
            Best matching executor or None if no match found
        """
        executors = self.lium.ls(gpu_type=spec.gpu_type)

        print(f"   🔍 Found {len(executors)} executors for GPU type: {spec.gpu_type}")

        candidates = []
        rejected = []
        for e in executors:
            reasons = []

            # Basic filters
            if e.gpu_count < spec.gpu_count:
                reasons.append(f"gpu_count {e.gpu_count} < required {spec.gpu_count}")
            # Note: lium.io returns status="unknown" for available executors
            # Skip status check as it's unreliable
            # if e.status not in ["available", "unknown"]:
            #     reasons.append(f"status={e.status}")
            if spec.country and e.location.get("country") != spec.country:
                reasons.append(f"country mismatch")

            # Bandwidth filters
            upload, download = self._get_bandwidth(e)

            if spec.min_upload_mbps and upload < spec.min_upload_mbps:
                reasons.append(f"upload {upload} < {spec.min_upload_mbps}")
            if spec.min_download_mbps and download < spec.min_download_mbps:
                reasons.append(f"download {download} < {spec.min_download_mbps}")

            if reasons:
                rejected.append((e, reasons))
            else:
                candidates.append((e, upload, download))

        if not candidates:
            print(f"   ❌ No matching executors. Rejection reasons:")
            for e, reasons in rejected[:5]:  # Show first 5
                print(f"      Executor {e.id[:8]}... ({e.gpu_count}x{e.gpu_type}): {', '.join(reasons)}")
            return None

        # Sort by price (or by bandwidth if you prefer)
        candidates.sort(key=lambda x: x[0].price_per_hour)
        print(f"   ✓ Found {len(candidates)} matching executor(s)")
        return candidates[0][0]

    def list_executors_with_bandwidth(
        self,
        gpu_type: str | None = None,
        min_upload: float = 0,
        min_download: float = 0
    ) -> None:
        """List executors with their bandwidth info for debugging.

        Args:
            gpu_type: Filter by GPU type
            min_upload: Minimum upload speed in Mbps
            min_download: Minimum download speed in Mbps
        """
        executors = self.lium.ls(gpu_type=gpu_type)

        print(f"{'HUID':<12} {'GPU':<25} {'Upload':<12} {'Download':<12} {'Price/hr':<10} {'Location'}")
        print("-" * 100)

        for e in executors:
            upload, download = self._get_bandwidth(e)

            # Filter
            if upload < min_upload or download < min_download:
                continue

            loc = e.location.get("city", "?") + ", " + e.location.get("country", "?")
            print(f"{e.huid:<12} {e.gpu_count}x {e.gpu_type:<18} {upload:<12.1f} {download:<12.1f} ${e.price_per_hour:<9.2f} {loc}")

    def inspect_executor_specs(self, gpu_type: str | None = None) -> None:
        """Debug: Print raw specs to see available fields.

        Args:
            gpu_type: Filter by GPU type
        """
        executors = self.lium.ls(gpu_type=gpu_type)
        if executors:
            print("Sample executor specs structure:")
            print(json.dumps(executors[0].specs, indent=2, default=str))

    def apply(self, pods: list[PodSpec]) -> dict:
        """Apply desired state - creates/removes pods to match spec.

        This is a declarative operation: the cluster will be modified to match
        the desired state specified by the pods list.

        Args:
            pods: List of desired pod specifications

        Returns:
            Dictionary mapping pod names to their info (SSH, bandwidth, spec)
        """
        desired = {p.name: p for p in pods}
        current_pods = {p.name: p for p in self.lium.ps()}

        # Remove pods not in desired state
        for name in set(self.state["pods"].keys()) - set(desired.keys()):
            if name in current_pods:
                print(f"🗑️  Removing: {name}")
                self.lium.down(current_pods[name])
            del self.state["pods"][name]

        # Create new pods or update state for existing ones
        for name, spec in desired.items():
            if name in current_pods:
                existing_pod = current_pods[name]
                print(f"✅ Already running: {name}")

                # Update state with current pod info if not already tracked
                if name not in self.state["pods"]:
                    ssh_host = existing_pod.host or (
                        existing_pod.executor.ip if existing_pod.executor else None
                    )
                    # Get bandwidth from executor if available
                    upload, download = 0.0, 0.0
                    if existing_pod.executor:
                        upload, download = self._get_bandwidth(existing_pod.executor)

                    self.state["pods"][name] = {
                        "id": existing_pod.id,
                        "spec": spec.__dict__,
                        "executor_id": existing_pod.executor.id if existing_pod.executor else "",
                        "bandwidth": {"upload": upload, "download": download},
                        "ssh": {"host": ssh_host, "port": existing_pod.ssh_port},
                    }
                continue

            bw_info = ""
            if spec.min_upload_mbps or spec.min_download_mbps:
                bw_info = f", min ↑{spec.min_upload_mbps or 0}Mbps ↓{spec.min_download_mbps or 0}Mbps"
            print(f"🚀 Creating: {name} ({spec.gpu_count}x {spec.gpu_type}{bw_info})")

            executor = self._find_executor(spec)
            if not executor:
                print(f"   ❌ No executor found matching requirements")
                continue

            upload, download = self._get_bandwidth(executor)
            print(f"   📡 Selected: {executor.huid} (↑{upload:.0f} ↓{download:.0f} Mbps)")

            result = self.lium.up(
                executor_id=executor.id,
                name=name,
                volume_id=spec.volume_id
            )

            pod = self.lium.wait_ready(result, timeout=300)
            if not pod:
                print(f"   ❌ Pod creation failed or timed out")
                continue

            # Get SSH connection info - try multiple sources
            ssh_host = pod.host
            if not ssh_host and pod.executor:
                ssh_host = pod.executor.ip
            if not ssh_host:
                # Fallback to executor IP we selected earlier
                ssh_host = executor.ip if hasattr(executor, "ip") else None

            if not ssh_host:
                print(f"   ❌ Could not determine SSH host for pod")
                continue

            ssh_port = pod.ssh_port
            if not ssh_port:
                print(f"   ❌ Could not determine SSH port for pod")
                continue

            # Wait for SSH port to be accessible (crucial - API reports ready before SSH)
            print(f"   ⏳ Waiting for SSH ({ssh_host}:{ssh_port})...")
            if not _wait_for_ssh(ssh_host, ssh_port, timeout=180, interval=5):
                print(f"   ⚠️  SSH not accessible after 180s, continuing anyway...")

            print(f"   ✅ Ready: {ssh_host}:{ssh_port}")
            self.state["pods"][name] = {
                "id": pod.id,
                "spec": spec.__dict__,
                "executor_id": executor.id,
                "bandwidth": {"upload": upload, "download": download},
                "ssh": {"host": ssh_host, "port": ssh_port},
            }

            if spec.ttl_hours:
                term_time = datetime.now(timezone.utc) + timedelta(hours=spec.ttl_hours)
                self.lium.schedule_termination(pod, termination_time=term_time.isoformat())
                print(f"   ⏰ Auto-terminate in {spec.ttl_hours}h")

        self._save_state()

        # Return only the pods that were requested (may be subset of state)
        return {name: self.state["pods"][name] for name in desired if name in self.state["pods"]}

    def destroy(self) -> None:
        """Destroy all managed pods."""
        current_pods = {p.name: p for p in self.lium.ps()}
        for name in list(self.state["pods"].keys()):
            print(f"🗑️  Destroying: {name}")
            try:
                if name in current_pods:
                    self.lium.down(current_pods[name])
            except Exception as e:
                print(f"   ⚠️  {e}")
            del self.state["pods"][name]
        self._save_state()

    def destroy_pod(self, name: str) -> bool:
        """Destroy a single pod by name.

        Args:
            name: Pod name to destroy

        Returns:
            True if pod was destroyed successfully
        """
        current_pods = {p.name: p for p in self.lium.ps()}

        if name in current_pods:
            print(f"🗑️  Destroying: {name}")
            try:
                self.lium.down(current_pods[name])
            except Exception as e:
                print(f"   ⚠️  Failed to destroy {name}: {e}")
                return False

        # Remove from state
        if name in self.state["pods"]:
            del self.state["pods"][name]
            self._save_state()

        return True

    def replace_pod(
        self,
        spec: PodSpec,
        exclude_executor_ids: list[str] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict | None:
        """Replace a failed pod with a new one on a different executor.

        This method:
        1. Destroys the existing pod if it exists
        2. Finds a new executor (excluding previously failed ones)
        3. Creates a new pod
        4. Verifies SSH connectivity
        5. Retries with different executors if needed

        Args:
            spec: Pod specification
            exclude_executor_ids: List of executor IDs to exclude (previously failed)
            max_retries: Maximum number of retry attempts

        Returns:
            Pod info dict if successful, None if all retries failed
        """
        exclude_ids = set(exclude_executor_ids or [])

        # First, destroy existing pod if any
        if spec.name in self.state["pods"]:
            old_executor_id = self.state["pods"][spec.name].get("executor_id")
            if old_executor_id:
                exclude_ids.add(old_executor_id)
            self.destroy_pod(spec.name)

        for attempt in range(max_retries):
            print(f"   🔄 Attempt {attempt + 1}/{max_retries} to create pod {spec.name}")

            # Find executor, excluding failed ones
            executor = self._find_executor_excluding(spec, exclude_ids)
            if not executor:
                print(f"   ❌ No more executors available (excluded {len(exclude_ids)})")
                return None

            upload, download = self._get_bandwidth(executor)
            print(f"   📡 Trying executor: {executor.huid} (↑{upload:.0f} ↓{download:.0f} Mbps)")

            # Create pod
            try:
                result = self.lium.up(
                    executor_id=executor.id,
                    name=spec.name,
                    volume_id=spec.volume_id
                )

                pod = self.lium.wait_ready(result, timeout=300)
                if not pod:
                    print(f"   ❌ Pod creation timed out")
                    exclude_ids.add(executor.id)
                    continue

                # Get SSH info
                ssh_host = pod.host or (pod.executor.ip if pod.executor else executor.ip)
                ssh_port = pod.ssh_port

                if not ssh_host or not ssh_port:
                    print(f"   ❌ Could not get SSH info")
                    self._cleanup_failed_pod(pod.id, spec.name)
                    exclude_ids.add(executor.id)
                    continue

                # Wait for SSH port
                print(f"   ⏳ Waiting for SSH ({ssh_host}:{ssh_port})...")
                if not _wait_for_ssh(ssh_host, ssh_port, timeout=DEFAULT_SSH_TIMEOUT):
                    print(f"   ❌ SSH port not accessible after {DEFAULT_SSH_TIMEOUT}s")
                    self._cleanup_failed_pod(pod.id, spec.name)
                    exclude_ids.add(executor.id)
                    continue

                # Verify SSH actually works
                print(f"   🔍 Verifying SSH connection...")
                if not _verify_ssh_connection(ssh_host, ssh_port):
                    print(f"   ❌ SSH verification failed")
                    self._cleanup_failed_pod(pod.id, spec.name)
                    exclude_ids.add(executor.id)
                    continue

                # Success!
                print(f"   ✅ Pod ready: {ssh_host}:{ssh_port}")
                self.state["pods"][spec.name] = {
                    "id": pod.id,
                    "spec": spec.__dict__,
                    "executor_id": executor.id,
                    "bandwidth": {"upload": upload, "download": download},
                    "ssh": {"host": ssh_host, "port": ssh_port},
                }

                if spec.ttl_hours:
                    term_time = datetime.now(timezone.utc) + timedelta(hours=spec.ttl_hours)
                    self.lium.schedule_termination(pod, termination_time=term_time.isoformat())
                    print(f"   ⏰ Auto-terminate in {spec.ttl_hours}h")

                self._save_state()
                return self.state["pods"][spec.name]

            except Exception as e:
                print(f"   ❌ Error creating pod: {e}")
                exclude_ids.add(executor.id)
                continue

        print(f"   ❌ All {max_retries} attempts failed for {spec.name}")
        return None

    def _find_executor_excluding(
        self,
        spec: PodSpec,
        exclude_ids: set[str]
    ) -> ExecutorInfo | None:
        """Find executor matching spec, excluding certain IDs.

        Args:
            spec: Pod specification
            exclude_ids: Set of executor IDs to exclude

        Returns:
            Matching executor or None
        """
        executors = self.lium.ls(gpu_type=spec.gpu_type)

        candidates = []
        for e in executors:
            # Skip excluded executors
            if e.id in exclude_ids:
                continue

            # Basic filters
            if e.gpu_count < spec.gpu_count:
                continue
            if spec.country and e.location.get("country") != spec.country:
                continue

            # Bandwidth filters
            upload, download = self._get_bandwidth(e)
            if spec.min_upload_mbps and upload < spec.min_upload_mbps:
                continue
            if spec.min_download_mbps and download < spec.min_download_mbps:
                continue

            candidates.append((e, upload, download))

        if not candidates:
            return None

        # Sort by price
        candidates.sort(key=lambda x: x[0].price_per_hour)
        return candidates[0][0]

    def _cleanup_failed_pod(self, pod_id: str, name: str) -> None:
        """Clean up a failed pod.

        Args:
            pod_id: Pod ID to clean up
            name: Pod name
        """
        try:
            current_pods = {p.id: p for p in self.lium.ps()}
            if pod_id in current_pods:
                self.lium.down(current_pods[pod_id])
                print(f"   🧹 Cleaned up failed pod")
        except Exception as e:
            logger.debug(f"Failed to cleanup pod {name}: {e}")

        # Remove from state if present
        if name in self.state["pods"]:
            del self.state["pods"][name]
            self._save_state()

    def verify_and_replace_unhealthy(
        self,
        pods: list[PodSpec],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict:
        """Verify all pods are healthy, replacing any that fail SSH check.

        Args:
            pods: List of pod specifications to verify
            max_retries: Maximum retries per failed pod

        Returns:
            Dict mapping pod names to their info (only healthy pods)
        """
        healthy_pods = {}

        for spec in pods:
            pod_info = self.state["pods"].get(spec.name)

            if not pod_info:
                print(f"🚀 Creating new pod: {spec.name}")
                new_info = self.replace_pod(spec, max_retries=max_retries)
                if new_info:
                    healthy_pods[spec.name] = new_info
                continue

            # Verify existing pod
            ssh_host = pod_info["ssh"]["host"]
            ssh_port = pod_info["ssh"]["port"]

            print(f"🔍 Verifying pod: {spec.name} ({ssh_host}:{ssh_port})")

            if _verify_ssh_connection(ssh_host, ssh_port):
                print(f"   ✅ Healthy")
                healthy_pods[spec.name] = pod_info
            else:
                print(f"   ❌ Unhealthy - replacing...")
                exclude_ids = [pod_info.get("executor_id")] if pod_info.get("executor_id") else []
                new_info = self.replace_pod(spec, exclude_executor_ids=exclude_ids, max_retries=max_retries)
                if new_info:
                    healthy_pods[spec.name] = new_info

        return healthy_pods

    def get_pod_info(self, name: str) -> dict | None:
        """Get information about a managed pod.

        Args:
            name: Pod name

        Returns:
            Pod info dict or None if not found
        """
        return self.state["pods"].get(name)

    def list_pods(self) -> dict:
        """List all managed pods.

        Returns:
            Dict mapping pod names to their info
        """
        return self.state["pods"]
