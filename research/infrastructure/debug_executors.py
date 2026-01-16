#!/usr/bin/env python3
"""Debug script to inspect available Lium executors."""

import os

from lium import Config, Lium

# Initialize Lium
api_key = os.getenv("LIUM_API_KEY")
if not api_key:
    print("❌ LIUM_API_KEY not set")
    print("   Run: export LIUM_API_KEY='your-api-key'")
    exit(1)

config = Config(api_key=api_key)
lium = Lium(config)

print("=" * 80)
print("LIUM EXECUTOR DEBUG")
print("=" * 80)

# List all A100 executors
print("\n🔍 Searching for A100 executors...")
executors = lium.ls(gpu_type="A100")

print(f"   Found {len(executors)} A100 executors\n")

if not executors:
    print("❌ No A100 executors found")
    print("   Try listing all executors:")
    all_executors = lium.ls()
    print(f"   Total executors: {len(all_executors)}")
    for e in all_executors[:5]:
        print(f"   - {e.id[:12]}: {e.gpu_count}x{e.gpu_type} (status: {e.status})")
    exit(0)

# Show detailed info for each executor
for i, e in enumerate(executors[:10], 1):  # Show first 10
    print(f"{i}. Executor: {e.id}")
    print(f"   GPU: {e.gpu_count}x {e.gpu_type}")
    print(f"   Status: {e.status}")
    print(f"   Price: ${e.price_per_hour}/hr")
    print(f"   Location: {e.location}")

    # Check for bandwidth info
    specs = e.specs if hasattr(e, "specs") else {}
    print(f"   Specs keys: {list(specs.keys()) if specs else 'None'}")

    # Try different bandwidth field names
    bandwidth_fields = [
        "upload_speed",
        "download_speed",
        "network",
        "bandwidth",
        "upload_mbps",
        "download_mbps",
    ]
    for field in bandwidth_fields:
        if field in specs:
            print(f"   {field}: {specs[field]}")

    print()

print("=" * 80)
print("FILTERS TO TEST")
print("=" * 80)

# Test the filter that's failing
print("\n📋 Executors with 8+ GPUs and status='available':")
matching = []
for e in executors:
    if e.gpu_count >= 8 and e.status == "available":
        matching.append(e)
        print(f"   ✓ {e.id[:12]}: {e.gpu_count}x{e.gpu_type}, ${e.price_per_hour}/hr")

if not matching:
    print("   ❌ None found")
    print("\n   Available statuses:")
    statuses = {e.status for e in executors}
    for status in statuses:
        count = sum(1 for e in executors if e.status == status)
        print(f"   - {status}: {count} executor(s)")

    print("\n   GPU counts available:")
    gpu_counts = {}
    for e in executors:
        gpu_counts[e.gpu_count] = gpu_counts.get(e.gpu_count, 0) + 1
    for count, num in sorted(gpu_counts.items()):
        print(f"   - {count}x GPUs: {num} executor(s)")
