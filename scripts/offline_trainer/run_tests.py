"""Standalone test runner for offline trainer (no pytest dependency)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_REPO_ROOT))

from scripts.offline_trainer.test_offline_trainer import (
    test_evaluator_gpu,
    test_evaluator_smoke,
    test_rollout_generator_advantage_computation,
    test_rollout_generator_produces_valid_groups,
    test_rollout_groups_are_valid,
    test_train_epoch_cpu,
    test_train_epoch_gpu,
)


async def run_all_tests() -> tuple[int, int]:
    """Run all tests and return (passed, total)."""
    tests = [
        ("rollout_generator_advantage_computation", test_rollout_generator_advantage_computation),
        ("rollout_groups_are_valid", test_rollout_groups_are_valid),
        ("rollout_generator_produces_valid_groups", test_rollout_generator_produces_valid_groups),
        ("evaluator_smoke", test_evaluator_smoke),
        ("train_epoch_cpu", test_train_epoch_cpu),
    ]

    # GPU tests if available
    import torch

    if torch.cuda.is_available():
        tests.extend(
            [
                ("train_epoch_gpu", test_train_epoch_gpu),
                ("evaluator_gpu", test_evaluator_gpu),
            ]
        )

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print(f"{'=' * 60}")
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            print(f"✅ {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            import traceback

            traceback.print_exc()

    return passed, total


async def main() -> None:
    print("Running offline trainer tests...")
    passed, total = await run_all_tests()
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
