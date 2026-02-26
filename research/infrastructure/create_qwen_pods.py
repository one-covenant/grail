#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from lium_manager import LiumInfra, PodSpec

load_dotenv(Path("/root/grail/.env"))

print("Creating fresh Qwen pods...", flush=True)
lium = LiumInfra(state_file=".lium_state_qwen_fresh.json")

pods = [
    PodSpec(name="qwen2.5-0.5b-new", gpu_type="A100", gpu_count=8, ttl_hours=124),
    PodSpec(name="qwen2.5-1.5b-new", gpu_type="A100", gpu_count=8, ttl_hours=124),
]

created = lium.apply(pods)

print("\nCreated pods:", flush=True)
for name, info in created.items():
    if info:
        print(f"  ✅ {name}: {info['ssh']['host']}:{info['ssh']['port']}", flush=True)
    else:
        print(f"  ❌ {name}: Failed to create", flush=True)

import json
with open("qwen_pods_ssh.json", "w") as f:
    json.dump(created, f, indent=2)
print("\nSaved pod info to qwen_pods_ssh.json", flush=True)
