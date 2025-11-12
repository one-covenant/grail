"""Standalone test runner for offline trainer (no pytest dependency)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure repo root and src on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_SRC_DIR = _THIS_FILE.parents[1] / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from test_offline_trainer import (  # noqa: E402
    test_evaluation,
    test_rollout_generation,
    test_training_epoch,
    test_training_with_gpu,
)


async def run_all_tests() -> tuple[int, int]:
    """Run all tests and return (passed, total)."""
    tests = [
        ("rollout_generation", test_rollout_generation),
        ("training_epoch", test_training_epoch),
        ("evaluation", test_evaluation),
        ("training_with_gpu", test_training_with_gpu),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            await test_func()
            print(f"✅ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback

            traceback.print_exc()

    return passed, total


async def main() -> None:
    """Run all tests."""
    print("Running offline trainer tests...")
    passed, total = await run_all_tests()
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
