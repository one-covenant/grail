#!/usr/bin/env python3
"""Test wandb import in the offline trainer context."""

import sys
from pathlib import Path

# Add parent grail to path (same as run_offline_grpo.py)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

print("Python path:")
for p in sys.path[:5]:
    print(f"  {p}")

print("\n1. Testing direct wandb import:")
try:
    import wandb

    print(f"  ✓ wandb imported: {wandb}")
    print(f"  ✓ wandb.__file__: {wandb.__file__}")
    print(f"  ✓ wandb.__version__: {wandb.__version__}")
    print(f"  ✓ hasattr(wandb, 'init'): {hasattr(wandb, 'init')}")
    print(f"  ✓ type(wandb.init): {type(wandb.init)}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n2. Testing WandBBackend import:")
try:
    from grail.monitoring.backends.wandb_backend import WandBBackend

    print("  ✓ WandBBackend imported")

    backend = WandBBackend()
    print(f"  ✓ Backend created: {backend}")
    print(f"  ✓ backend._wandb_module before init: {backend._wandb_module}")

    # Initialize
    backend.initialize({"project": "test", "mode": "disabled"})
    print(f"  ✓ Backend initialized: {backend._initialized}")
    print(f"  ✓ backend._wandb_module after init: {backend._wandb_module}")

    if backend._wandb_module:
        print(f"  ✓ wandb module type: {type(backend._wandb_module)}")
        print(f"  ✓ wandb module has init: {hasattr(backend._wandb_module, 'init')}")
        if hasattr(backend._wandb_module, "__name__"):
            print(f"  ✓ wandb module __name__: {backend._wandb_module.__name__}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n3. Testing initialize_monitoring:")
try:
    from grail.monitoring import get_monitoring_manager, initialize_monitoring

    initialize_monitoring(
        backend_type="wandb",
        project="test",
        mode="disabled",
    )
    print("  ✓ initialize_monitoring succeeded")  # noqa: F541

    manager = get_monitoring_manager()
    print(f"  ✓ manager: {manager}")
    print(f"  ✓ manager.backend: {manager.backend}")
    if hasattr(manager.backend, "_wandb_module"):
        print(f"  ✓ manager.backend._wandb_module: {manager.backend._wandb_module}")
        if manager.backend._wandb_module:
            print(f"  ✓ has init: {hasattr(manager.backend._wandb_module, 'init')}")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ All tests completed")
