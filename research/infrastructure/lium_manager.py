"""Lium infrastructure manager for distributed training experiments.

This module provides declarative infrastructure management for Lium pods,
including bandwidth filtering, state management, and pod lifecycle operations.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from lium import Config, ExecutorInfo, Lium


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
