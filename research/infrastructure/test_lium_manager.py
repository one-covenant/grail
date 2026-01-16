#!/usr/bin/env python3
"""Test script to validate lium_manager.py corrections."""

import logging
from pathlib import Path

from dotenv import load_dotenv

from lium_manager import LiumInfra

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def test_lium_infra() -> None:
    """Test LiumInfra manager with corrected API usage."""
    print("=" * 80)
    print("TESTING LIUMINFRA MANAGER")
    print("=" * 80)

    try:
        # Initialize manager
        infra = LiumInfra()
        print("✅ LiumInfra initialized successfully")

        # List executors with bandwidth info
        print("\n" + "-" * 80)
        print("Available A100 Executors (sorted by price):")
        print("-" * 80)
        infra.list_executors_with_bandwidth(gpu_type="A100")

        # Check executor specs structure
        print("\n" + "-" * 80)
        print("Sample Executor Specs:")
        print("-" * 80)
        infra.inspect_executor_specs(gpu_type="A100")

        # List managed pods
        print("\n" + "-" * 80)
        print("Managed Pods:")
        print("-" * 80)
        pods_info = infra.list_pods()
        if pods_info:
            for pod_name, pod_info in pods_info.items():
                print(f"  {pod_name}:")
                print(f"    SSH: {pod_info['ssh']['host']}:{pod_info['ssh']['port']}")
                print(
                    f"    Bandwidth: ↑{pod_info['bandwidth']['upload']:.0f} "
                    f"↓{pod_info['bandwidth']['download']:.0f} Mbps"
                )
        else:
            print("  No managed pods")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    test_lium_infra()
