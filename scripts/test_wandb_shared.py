#!/usr/bin/env python3
"""Test WandB shared mode connection to existing run.

This script tests if subprocess can quickly connect to an EXISTING WandB run
(from the trainer) using shared mode.
"""

import multiprocessing
import time
import sys
import argparse


def subprocess_worker(run_id: str, process_label: str, entity: str | None) -> None:
    """Worker process that connects to existing run via shared mode.
    
    Args:
        run_id: The run ID from the primary process
        process_label: Label for this worker (e.g., "worker_1")
        entity: WandB entity (team or username)
    """
    print(f"\n[{process_label}] Starting subprocess...")
    print(f"[{process_label}] Will connect to EXISTING run_id: {run_id}")
    print(f"[{process_label}] Entity: {entity if entity else '(default)'}")
    print(f"[{process_label}] Using shared mode with x_primary=False")
    
    # Time the connection
    start_time = time.time()
    
    try:
        import wandb
        
        # Connect to existing run using shared mode (same as GRAIL training subprocess)
        print(f"[{process_label}] Calling wandb.init()...")
        
        init_kwargs = {
            "project": "grail",
            "id": run_id,
            "mode": "shared",
            "settings": wandb.Settings(
                x_primary=False,
                x_label=process_label,
                init_timeout=300.0,
            ),
        }
        
        # Add entity if provided (matches GRAIL behavior)
        if entity:
            init_kwargs["entity"] = entity
            print(f"[{process_label}] Using entity: {entity}")
        
        run = wandb.init(**init_kwargs)
        
        connect_time = time.time() - start_time
        print(f"[{process_label}] ✅ Connected in {connect_time:.1f}s (run_id={run.id})")
        
        # Log multiple test metrics to the "testing" namespace
        print(f"[{process_label}] Logging test metrics to 'testing/' namespace...")
        print(f"[{process_label}] These will appear in WandB UI: https://wandb.ai/tplr/grail/runs/{run_id}")
        
        for i in range(5):
            metrics = {
                "testing/connection_test": i,
                "testing/subprocess_metric": i * 100,
                "testing/timestamp": time.time(),
            }
            run.log(metrics)
            print(f"[{process_label}]   Logged batch {i+1}/5")
            time.sleep(0.5)  # Small delay between logs
        
        print(f"[{process_label}] ✅ Logged 5 test metric batches to 'testing/' namespace")
        print(f"[{process_label}] ✅ Check WandB UI - metrics should appear immediately!")
        
        # Don't call finish() - existing run might still be active
        print(f"[{process_label}] Worker complete (total time: {time.time() - start_time:.1f}s)")
        
    except Exception as e:
        connect_time = time.time() - start_time
        print(f"[{process_label}] ❌ Failed after {connect_time:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main test script."""
    parser = argparse.ArgumentParser(
        description="Test connecting to existing WandB run via shared mode"
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Existing WandB run ID to connect to (e.g., 3v0fkpxi from grail_train.log)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="tplr",
        help="WandB entity (team or username, default: tplr)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=360,
        help="Timeout in seconds for subprocess connection (default: 360)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WandB Shared Mode Connection Test (Existing Run)")
    print("=" * 80)
    
    try:
        import wandb
    except ImportError:
        print("❌ wandb not installed")
        sys.exit(1)
    
    print(f"WandB version: {wandb.__version__}")
    print(f"Target run ID: {args.run_id}")
    print(f"Entity: {args.entity}")
    print(f"Timeout: {args.timeout}s")
    print(f"Run URL: https://wandb.ai/{args.entity}/grail/runs/{args.run_id}")
    print()
    
    # Spawn subprocess to connect to existing run (NO primary creation)
    print(f"[MAIN] Spawning subprocess to connect to EXISTING run {args.run_id}...")
    print(f"[MAIN] This mimics the GRAIL training subprocess behavior")
    print()
    
    process = multiprocessing.Process(
        target=subprocess_worker,
        args=(args.run_id, "test_worker", args.entity),
    )
    
    start_time = time.time()
    process.start()
    
    # Wait for subprocess with timeout
    process.join(timeout=args.timeout)
    
    elapsed = time.time() - start_time
    
    if process.is_alive():
        print(f"\n[MAIN] ⚠️ Subprocess still running after {elapsed:.1f}s, terminating...")
        process.terminate()
        process.join(timeout=10)
        print(f"[MAIN] ❌ Test FAILED - subprocess timed out")
        print(f"[MAIN] This explains why GRAIL training subprocess times out!")
        sys.exit(1)
    elif process.exitcode == 0:
        print(f"\n[MAIN] ✅ Subprocess completed successfully in {elapsed:.1f}s")
        print(f"[MAIN] ✅ Test PASSED - shared mode should work in GRAIL!")
    else:
        print(f"\n[MAIN] ❌ Subprocess exited with code {process.exitcode} after {elapsed:.1f}s")
        print(f"[MAIN] ❌ Test FAILED")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
    print("\nTo test with your current trainer run:")
    print("  1. Find run_id in grail_train.log (e.g., 3v0fkpxi)")
    print("  2. Run: python test_wandb_shared.py 3v0fkpxi --entity tplr")
    print("     (Use --entity to match your WandB team/account)")


if __name__ == "__main__":
    # Required for multiprocessing with CUDA
    multiprocessing.set_start_method("spawn", force=True)
    main()

