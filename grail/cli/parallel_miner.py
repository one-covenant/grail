#!/usr/bin/env python3
"""
Parallel Multi-GPU Miner for GRAIL

Coordinates multiple GPU workers to generate rollouts in parallel, with each GPU
handling a distinct range of problem IDs. All results are gathered before a
single upload to maximize throughput while maintaining submission integrity.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Coordinator Process                       │
    │  - Assigns problem ranges: GPU0=[0-11], GPU1=[12-23], ...   │
    │  - Spawns N worker processes                                 │
    │  - Gathers results via temp files                           │
    │  - Single sink_window_inferences() call                      │
    └──────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────┬─────────┬─┴─────────┬─────────┐
         ▼         ▼         ▼           ▼         ▼
     ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐
     │GPU 0 │ │GPU 1 │ │GPU 2 │ ... │GPU N │ │GPU N │
     │P:0-11│ │P:12-23│ │P:24-35│    │      │ │      │
     └──────┘ └──────┘ └──────┘     └──────┘ └──────┘

Usage:
    python -m grail.cli.parallel_miner --num-gpus 8 --problems-per-gpu 12
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any

import bittensor as bt
import torch
import typer

from ..infrastructure.credentials import load_r2_credentials
from ..shared.constants import WINDOW_LENGTH
from . import console

logger = logging.getLogger("grail.parallel_miner")


# --------------------------------------------------------------------------- #
#                       Configuration                                         #
# --------------------------------------------------------------------------- #


@dataclass
class GPUWorkerConfig:
    """Configuration for a single GPU worker process."""

    gpu_id: int
    problem_offset: int
    max_problems: int
    results_dir: Path
    window_start: int
    window_block_hash: str
    combined_randomness: str
    use_drand: bool
    checkpoint_path: str | None
    # Wallet names read from environment in worker for subprocess isolation
    batch_size: int = 2
    safety_blocks: int = 3


@dataclass
class ParallelMinerConfig:
    """Configuration for parallel multi-GPU mining."""

    num_gpus: int = 8
    problems_per_gpu: int = 12
    batch_size: int = 2
    safety_blocks: int = 3
    use_drand: bool = True
    results_dir: Path = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="grail_parallel_"))
    )
    worker_timeout: float = 600.0  # 10 minutes max per window
    gpu_ids: list[int] | None = None  # Specific GPU IDs to use, None = [0, 1, ..., num_gpus-1]

    def get_gpu_ids(self) -> list[int]:
        """Return list of GPU IDs to use."""
        if self.gpu_ids is not None:
            return self.gpu_ids
        return list(range(self.num_gpus))


# --------------------------------------------------------------------------- #
#                       GPU Worker Process                                    #
# --------------------------------------------------------------------------- #


def _gpu_worker_main(
    config: GPUWorkerConfig,
    result_queue: mp.Queue,
) -> None:
    """Main function for GPU worker process.

    This runs in a separate process with CUDA_VISIBLE_DEVICES set to the
    assigned GPU. It generates rollouts for a specific problem range and
    writes results to a temp file.

    Args:
        config: Worker configuration with GPU assignment and problem range
        result_queue: Queue to signal completion status back to coordinator
    """
    worker_id = f"GPU-{config.gpu_id}"
    start_time = time.time()

    # Configure logging for worker process
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{worker_id}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    worker_logger = logging.getLogger(f"grail.worker.{config.gpu_id}")

    try:
        # Set GPU visibility BEFORE any CUDA operations
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # Import heavy modules after setting CUDA_VISIBLE_DEVICES
        from ..cli.mine import (
            MiningTimers,
            package_rollout_data,
        )
        from ..environments.factory import create_env
        from ..environments.loop import AgentEnvLoop
        from ..grail import derive_env_seed
        from ..model.provider import get_model, get_tokenizer
        from ..shared.constants import ROLLOUTS_PER_PROBLEM

        worker_logger.info(
            "Starting worker: problems %d-%d on GPU %d",
            config.problem_offset,
            config.problem_offset + config.max_problems - 1,
            config.gpu_id,
        )

        # Load wallet from environment (same env as coordinator)
        coldkey = os.getenv("BT_WALLET_COLD", "default")
        hotkey = os.getenv("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        # Load model and tokenizer
        if config.checkpoint_path:
            model = get_model(config.checkpoint_path, device="cuda", eval_mode=True)
            tokenizer = get_tokenizer(config.checkpoint_path)
        else:
            raise RuntimeError("checkpoint_path is required for parallel mining")

        device = model.device
        loop = AgentEnvLoop(model, tokenizer, str(device))

        # Generate rollouts for assigned problem range
        inferences: list[dict] = []
        timers = MiningTimers()

        for local_idx in range(config.max_problems):
            problem_index = config.problem_offset + local_idx
            gen_start = time.time()

            # Derive deterministic seed for this problem
            seed_int = derive_env_seed(
                wallet.hotkey.ss58_address,
                config.window_block_hash,
                problem_index,
            )

            worker_logger.debug(
                "Generating problem %d (seed=%d)",
                problem_index,
                seed_int,
            )

            # Generate GRPO rollouts
            def _env_factory():
                return create_env()

            grpo_rollouts = loop.run_grpo_group(
                _env_factory,
                ROLLOUTS_PER_PROBLEM,
                config.combined_randomness,
                wallet,
                batch_size=config.batch_size,
                seed=seed_int,
            )

            # Package rollouts with signatures
            base_nonce = problem_index
            for rollout_idx, rollout in enumerate(grpo_rollouts):
                rollout_data = package_rollout_data(
                    model,
                    wallet,
                    rollout,
                    base_nonce,
                    rollout_idx,
                    len(grpo_rollouts),
                    config.window_start,
                    config.window_start,  # current_block = window_start for parallel
                    config.window_block_hash,
                    config.combined_randomness,
                    config.use_drand,
                )
                inferences.append(rollout_data)

            gen_duration = time.time() - gen_start
            timers.update_gen_time_ema(gen_duration)

            worker_logger.info(
                "Problem %d: %d rollouts in %.2fs",
                problem_index,
                len(grpo_rollouts),
                gen_duration,
            )

        # Write results to temp file
        result_file = config.results_dir / f"gpu_{config.gpu_id}_results.json"
        result_data = {
            "gpu_id": config.gpu_id,
            "problem_offset": config.problem_offset,
            "max_problems": config.max_problems,
            "inference_count": len(inferences),
            "inferences": inferences,
            "duration_seconds": time.time() - start_time,
        }

        with open(result_file, "w") as f:
            json.dump(result_data, f)

        worker_logger.info(
            "Completed: %d rollouts from %d problems in %.2fs",
            len(inferences),
            config.max_problems,
            time.time() - start_time,
        )

        # Signal success
        result_queue.put(
            {
                "gpu_id": config.gpu_id,
                "status": "success",
                "inference_count": len(inferences),
                "result_file": str(result_file),
                "duration": time.time() - start_time,
            }
        )

    except Exception as e:
        worker_logger.exception("Worker failed: %s", e)
        result_queue.put(
            {
                "gpu_id": config.gpu_id,
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time,
            }
        )


# --------------------------------------------------------------------------- #
#                       Parallel Mining Coordinator                           #
# --------------------------------------------------------------------------- #


class ParallelMiningCoordinator:
    """Coordinates parallel rollout generation across multiple GPUs.

    Responsibilities:
    - Spawn GPU worker processes with non-overlapping problem ranges
    - Monitor worker progress and handle failures
    - Gather all results and perform single aggregated upload
    - Clean up temp files after successful upload
    """

    def __init__(
        self,
        config: ParallelMinerConfig,
        wallet: bt.wallet,
        credentials: Any,
    ) -> None:
        self.config = config
        self.wallet = wallet
        self.credentials = credentials
        self._workers: list[mp.Process] = []
        # Use spawn context for CUDA-safe queue
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set
        self._ctx = mp.get_context("spawn")
        self._result_queue: mp.Queue = self._ctx.Queue()
        self._shutdown_requested = False

    async def mine_window(
        self,
        window_start: int,
        window_block_hash: str,
        combined_randomness: str,
        checkpoint_path: str | None,
    ) -> list[dict]:
        """Generate rollouts for a window using all GPUs in parallel.

        Args:
            window_start: Start block of the mining window
            window_block_hash: Block hash at window start
            combined_randomness: Combined randomness for proof generation
            checkpoint_path: Path to model checkpoint

        Returns:
            Combined list of all rollout inferences from all GPUs
        """
        gpu_ids = self.config.get_gpu_ids()
        total_problems = self.config.problems_per_gpu * len(gpu_ids)

        logger.info(
            "🚀 Starting parallel mining: %d GPUs × %d problems = %d total problems",
            len(gpu_ids),
            self.config.problems_per_gpu,
            total_problems,
        )

        # Ensure results directory exists
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        # Create worker configs with non-overlapping problem ranges
        worker_configs: list[GPUWorkerConfig] = []
        for idx, gpu_id in enumerate(gpu_ids):
            problem_offset = idx * self.config.problems_per_gpu
            worker_config = GPUWorkerConfig(
                gpu_id=gpu_id,
                problem_offset=problem_offset,
                max_problems=self.config.problems_per_gpu,
                results_dir=self.config.results_dir,
                window_start=window_start,
                window_block_hash=window_block_hash,
                combined_randomness=combined_randomness,
                use_drand=self.config.use_drand,
                checkpoint_path=checkpoint_path,
                batch_size=self.config.batch_size,
                safety_blocks=self.config.safety_blocks,
            )
            worker_configs.append(worker_config)

        # Spawn worker processes using 'spawn' method for CUDA compatibility
        # This ensures each worker gets a fresh CUDA context without conflicts
        start_time = time.time()
        self._workers = []

        for worker_config in worker_configs:
            # Use spawn context to avoid CUDA context issues
            proc = self._ctx.Process(
                target=_gpu_worker_main,
                args=(worker_config, self._result_queue),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)
            logger.info(
                "  Started worker PID %d for GPU %d (problems %d-%d)",
                proc.pid,
                worker_config.gpu_id,
                worker_config.problem_offset,
                worker_config.problem_offset + worker_config.max_problems - 1,
            )

        # Wait for all workers to complete
        results = await self._wait_for_workers(len(gpu_ids))

        # Gather and combine results - ALL workers must succeed
        all_inferences, all_succeeded = await self._gather_results(results, len(gpu_ids))

        elapsed = time.time() - start_time

        if not all_succeeded:
            logger.error("❌ Parallel mining FAILED: Not all GPUs completed successfully")
            logger.error(
                "Returning empty results to prevent partial upload that would fail validation"
            )
            return []  # Return empty to prevent upload

        # Verify expected rollout count
        from ..shared.constants import ROLLOUTS_PER_PROBLEM

        expected_rollouts = len(gpu_ids) * self.config.problems_per_gpu * ROLLOUTS_PER_PROBLEM
        if len(all_inferences) != expected_rollouts:
            logger.error(
                "❌ Rollout count mismatch: got %d, expected %d (%d GPUs × %d problems × %d rollouts)",
                len(all_inferences),
                expected_rollouts,
                len(gpu_ids),
                self.config.problems_per_gpu,
                ROLLOUTS_PER_PROBLEM,
            )
            logger.error("Returning empty results to prevent validation failure")
            return []

        logger.info(
            "✅ Parallel mining complete: %d rollouts in %.2fs (%.1f rollouts/sec)",
            len(all_inferences),
            elapsed,
            len(all_inferences) / elapsed if elapsed > 0 else 0,
        )

        return all_inferences

    async def _wait_for_workers(self, expected_count: int) -> list[dict]:
        """Wait for all worker processes to complete.

        Args:
            expected_count: Number of workers expected to complete

        Returns:
            List of result dictionaries from each worker
        """
        results: list[dict] = []
        deadline = time.time() + self.config.worker_timeout

        while len(results) < expected_count and time.time() < deadline:
            try:
                # Non-blocking check with timeout
                result = await asyncio.to_thread(
                    self._result_queue.get,
                    timeout=5.0,
                )
                results.append(result)

                if result["status"] == "success":
                    logger.info(
                        "  GPU %d completed: %d rollouts in %.2fs",
                        result["gpu_id"],
                        result["inference_count"],
                        result["duration"],
                    )
                else:
                    logger.error(
                        "  GPU %d failed: %s",
                        result["gpu_id"],
                        result.get("error", "unknown error"),
                    )

            except Empty:
                # Check if any workers have crashed
                alive_count = sum(1 for w in self._workers if w.is_alive())
                if alive_count == 0 and len(results) < expected_count:
                    logger.error("All workers have exited but not all reported results")
                    break
                continue

        # Terminate any remaining workers
        for worker in self._workers:
            if worker.is_alive():
                logger.warning("Terminating hung worker PID %d", worker.pid)
                worker.terminate()
                worker.join(timeout=5.0)

        return results

    async def _gather_results(
        self, worker_results: list[dict], expected_gpu_count: int
    ) -> tuple[list[dict], bool]:
        """Gather and combine results from all workers.

        CRITICAL: All workers must succeed for upload to proceed.
        Missing any problem ID will cause validator proof failure.

        Args:
            worker_results: List of worker result status dictionaries
            expected_gpu_count: Number of GPUs that must succeed

        Returns:
            Tuple of (combined inferences, all_succeeded)
        """
        all_inferences: list[dict] = []
        successful_gpus = 0
        failed_gpus: list[int] = []

        for result in worker_results:
            if result["status"] != "success":
                failed_gpus.append(result["gpu_id"])
                logger.error(
                    "GPU %d FAILED: %s - Cannot upload partial results!",
                    result["gpu_id"],
                    result.get("error", "unknown error"),
                )
                continue

            result_file = Path(result["result_file"])
            if not result_file.exists():
                failed_gpus.append(result["gpu_id"])
                logger.error(
                    "GPU %d result file missing: %s - Cannot upload partial results!",
                    result["gpu_id"],
                    result_file,
                )
                continue

            try:
                with open(result_file) as f:
                    data = json.load(f)
                    inferences = data.get("inferences", [])
                    all_inferences.extend(inferences)
                    successful_gpus += 1
                    logger.info(
                        "  GPU %d: %d rollouts collected",
                        result["gpu_id"],
                        len(inferences),
                    )

                # Clean up temp file
                result_file.unlink()

            except Exception as e:
                failed_gpus.append(result["gpu_id"])
                logger.error("Failed to read results from GPU %d: %s", result["gpu_id"], e)

        # Check if ALL workers succeeded
        all_succeeded = (successful_gpus == expected_gpu_count) and len(failed_gpus) == 0

        if all_succeeded:
            logger.info(
                "✅ All %d GPUs succeeded: %d total rollouts ready for upload",
                successful_gpus,
                len(all_inferences),
            )
        else:
            logger.error(
                "❌ INCOMPLETE: Only %d/%d GPUs succeeded. Failed GPUs: %s",
                successful_gpus,
                expected_gpu_count,
                failed_gpus,
            )
            logger.error(
                "Cannot upload partial results - validator would reject due to missing problem IDs!"
            )

        return all_inferences, all_succeeded

    def cleanup(self) -> None:
        """Clean up resources and temp files."""
        # Terminate any remaining workers
        for worker in self._workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)

        # Clean up results directory
        try:
            if self.config.results_dir.exists():
                for f in self.config.results_dir.iterdir():
                    f.unlink()
                self.config.results_dir.rmdir()
        except Exception as e:
            logger.debug("Cleanup error (non-fatal): %s", e)


# --------------------------------------------------------------------------- #
#                       CLI Interface                                         #
# --------------------------------------------------------------------------- #


async def run_parallel_miner(
    config: ParallelMinerConfig,
    use_drand: bool = True,
) -> None:
    """Main entry point for parallel multi-GPU mining.

    Args:
        config: Parallel mining configuration
        use_drand: Whether to use drand for randomness
    """
    from types import SimpleNamespace

    from ..cli.mine import (
        MiningTimers,
        calculate_window_start,
        get_conf,
        get_window_randomness,
        upload_inferences_with_metrics,
    )
    from ..infrastructure.chain import GrailChainManager
    from ..infrastructure.checkpoints import CheckpointManager, default_checkpoint_cache_root
    from ..shared.constants import TRAINER_UID

    # Load configuration
    coldkey = get_conf("BT_WALLET_COLD", "default")
    hotkey = get_conf("BT_WALLET_HOT", "default")
    wallet = bt.wallet(name=coldkey, hotkey=hotkey)

    logger.info("🔑 Parallel Miner hotkey: %s", wallet.hotkey.ss58_address)
    logger.info("   GPUs: %d, Problems/GPU: %d", config.num_gpus, config.problems_per_gpu)

    # Load credentials
    credentials = load_r2_credentials()
    logger.info("✅ Loaded R2 credentials")

    # Initialize async subtensor (grail uses async bittensor wrapper)
    from ..infrastructure.network import create_subtensor

    subtensor = await create_subtensor()
    netuid = int(get_conf("BT_NETUID", get_conf("NETUID", 200)))

    # Get metagraph using async subtensor
    metagraph = await subtensor.metagraph(netuid)

    # Initialize chain manager for credential commitments
    chain_config = SimpleNamespace(netuid=netuid)
    chain_manager = GrailChainManager(chain_config, wallet, metagraph, subtensor, credentials)
    await chain_manager.initialize()
    logger.info("✅ Initialized chain manager")

    # Get trainer credentials for checkpoints
    trainer_bucket = chain_manager.get_bucket(TRAINER_UID)
    checkpoint_credentials = trainer_bucket if trainer_bucket else credentials

    checkpoint_manager = CheckpointManager(
        cache_root=default_checkpoint_cache_root(),
        credentials=checkpoint_credentials,
        keep_limit=2,
    )

    # Create coordinator
    coordinator = ParallelMiningCoordinator(config, wallet, credentials)

    # Main mining loop
    last_window_start = -1
    timers = MiningTimers()
    current_checkpoint_window: int | None = None
    checkpoint_path: str | None = None

    try:
        while True:
            current_block = await subtensor.get_current_block()
            window_start = calculate_window_start(current_block)
            checkpoint_window = window_start - WINDOW_LENGTH

            if window_start <= last_window_start:
                await asyncio.sleep(5)
                continue

            # Load checkpoint if needed
            if checkpoint_window >= 0 and current_checkpoint_window != checkpoint_window:
                logger.info("🔁 Loading checkpoint for window %s", checkpoint_window)
                checkpoint_path_obj = await checkpoint_manager.get_checkpoint(checkpoint_window)
                if checkpoint_path_obj:
                    checkpoint_path = str(checkpoint_path_obj)
                    current_checkpoint_window = checkpoint_window
                else:
                    logger.error("No checkpoint available for window %s", checkpoint_window)
                    await asyncio.sleep(30)
                    continue

            if not checkpoint_path:
                logger.error("No checkpoint loaded, cannot mine")
                await asyncio.sleep(30)
                continue

            # Check time budget - skip for parallel mode since workers manage their own time
            # The parallel coordinator ensures all workers complete before upload

            # Get window randomness
            window_block_hash, combined_randomness = await get_window_randomness(
                subtensor,
                window_start,
                use_drand,
            )

            logger.info(
                "🔥 Starting parallel mining for window %d-%d",
                window_start,
                window_start + WINDOW_LENGTH - 1,
            )

            # Run parallel mining
            inferences = await coordinator.mine_window(
                window_start,
                window_block_hash,
                combined_randomness,
                checkpoint_path,
            )

            # Upload aggregated results
            if inferences:
                logger.info(
                    "📤 Uploading %d aggregated rollouts for window %d",
                    len(inferences),
                    window_start,
                )
                upload_duration = await upload_inferences_with_metrics(
                    wallet,
                    window_start,
                    inferences,
                    credentials,
                    None,  # monitor
                )
                timers.update_upload_time_ema(upload_duration)
                logger.info("✅ Successfully uploaded window %d", window_start)
            else:
                logger.warning("No inferences generated for window %d", window_start)

            last_window_start = window_start
            await checkpoint_manager.cleanup_local(window_start)

    except KeyboardInterrupt:
        logger.info("Shutting down parallel miner...")
    finally:
        coordinator.cleanup()
        chain_manager.stop()


def register(app: typer.Typer) -> None:
    """Register parallel-mine command with CLI."""
    app.command("parallel-mine")(parallel_mine)


def parallel_mine(
    num_gpus: int = typer.Option(
        8,
        "--num-gpus",
        "-g",
        help="Number of GPUs to use for parallel mining",
    ),
    problems_per_gpu: int = typer.Option(
        12,
        "--problems-per-gpu",
        "-p",
        help="Minimum number of problems each GPU should generate",
    ),
    batch_size: int = typer.Option(
        2,
        "--batch-size",
        "-b",
        help="Rollout batch size within each problem (1-16)",
    ),
    safety_blocks: int = typer.Option(
        3,
        "--safety-blocks",
        help="Safety margin blocks before window end",
    ),
    use_drand: bool = typer.Option(
        True,
        "--use-drand/--no-drand",
        help="Use drand for randomness",
    ),
    gpu_ids: str = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). Default: 0 to num_gpus-1",
    ),
    worker_timeout: float = typer.Option(
        600.0,
        "--worker-timeout",
        help="Maximum seconds to wait for workers per window",
    ),
) -> None:
    """Run parallel multi-GPU miner for maximum throughput.

    Spawns multiple worker processes, each on a dedicated GPU, generating
    rollouts for non-overlapping problem ranges. Results are aggregated
    and uploaded as a single submission per window.

    Example:
        grail parallel-mine --num-gpus 8 --problems-per-gpu 12

    This generates 8 × 12 = 96 problems per window (1,536+ rollouts).
    """
    # Validate inputs
    if num_gpus < 1:
        console.print("[red]Error: --num-gpus must be at least 1[/red]")
        raise typer.Exit(code=1)

    if problems_per_gpu < 1:
        console.print("[red]Error: --problems-per-gpu must be at least 1[/red]")
        raise typer.Exit(code=1)

    if batch_size < 1 or batch_size > 16:
        console.print("[red]Error: --batch-size must be between 1 and 16[/red]")
        raise typer.Exit(code=1)

    # Parse GPU IDs if provided
    parsed_gpu_ids = None
    if gpu_ids:
        try:
            parsed_gpu_ids = [int(x.strip()) for x in gpu_ids.split(",")]
            if len(parsed_gpu_ids) != num_gpus:
                console.print(
                    f"[red]Error: --gpu-ids has {len(parsed_gpu_ids)} IDs "
                    f"but --num-gpus is {num_gpus}[/red]"
                )
                raise typer.Exit(code=1)
        except ValueError as err:
            console.print("[red]Error: --gpu-ids must be comma-separated integers[/red]")
            raise typer.Exit(code=1) from err

    # Check GPU availability
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        console.print(
            f"[yellow]Warning: Only {available_gpus} GPUs available, "
            f"but {num_gpus} requested[/yellow]"
        )

    config = ParallelMinerConfig(
        num_gpus=num_gpus,
        problems_per_gpu=problems_per_gpu,
        batch_size=batch_size,
        safety_blocks=safety_blocks,
        use_drand=use_drand,
        worker_timeout=worker_timeout,
        gpu_ids=parsed_gpu_ids,
    )

    total_problems = num_gpus * problems_per_gpu
    console.print("[bold green]Starting Parallel Miner[/bold green]")
    console.print(f"  GPUs: {num_gpus}")
    console.print(f"  Problems/GPU: {problems_per_gpu}")
    console.print(f"  Total problems/window: {total_problems}")
    console.print(f"  Expected rollouts/window: {total_problems * 16}")

    try:
        asyncio.run(run_parallel_miner(config, use_drand))
    except KeyboardInterrupt:
        console.print("[yellow]Parallel miner stopped by user[/yellow]")
        raise typer.Exit(code=0) from None
    except Exception as e:
        logger.exception("Fatal error in parallel miner")
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
#                          Main Entry Point                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    """Main entry point for parallel miner CLI."""
    app = typer.Typer()
    register(app)
    app()


if __name__ == "__main__":
    main()
