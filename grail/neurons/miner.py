from __future__ import annotations

import asyncio
import contextlib
import logging
import traceback
from types import SimpleNamespace

import bittensor as bt
import torch

from grail.cli.mine import (
    MiningTimers,
    _pipelined_generation_loop,
    generate_rollouts_for_window,
    get_conf,
    get_window_randomness,
    has_time_for_next_generation,
    upload_inferences_with_metrics,
)
from grail.environments.execution import (
    CodeExecutionPool,
    set_global_execution_pool,
)
from grail.infrastructure.chain import GrailChainManager
from grail.infrastructure.checkpoint_consumer import (
    CheckpointManager,
    default_checkpoint_cache_root,
)
from grail.infrastructure.credentials import load_r2_credentials
from grail.mining.config import PipelineConfig
from grail.model.provider import clear_model_and_tokenizer, get_model, get_tokenizer
from grail.monitoring import get_monitoring_manager
from grail.monitoring.config import MonitoringConfig
from grail.shared.constants import TRAINER_UID, WINDOW_LENGTH
from grail.shared.window_utils import (
    WindowWaitTracker,
    calculate_next_window,
    log_window_wait_initial,
    log_window_wait_periodic,
)

from .base import BaseNeuron

logger = logging.getLogger(__name__)


class MinerNeuron(BaseNeuron):
    """Runs the mining loop under a unified neuron lifecycle."""

    def __init__(self, use_drand: bool = True) -> None:
        super().__init__()
        self.use_drand = use_drand

    # (heartbeat is now handled by BaseNeuron.heartbeat())

    async def run(self) -> None:
        """Main mining loop mirrored from the CLI implementation."""
        coldkey = get_conf("BT_WALLET_COLD", "default")
        hotkey = get_conf("BT_WALLET_HOT", "default")
        wallet = bt.wallet(name=coldkey, hotkey=hotkey)

        logger.info(f"🔑 Miner hotkey: {wallet.hotkey.ss58_address}")

        # Model and tokenizer will be loaded from checkpoint
        model = None
        tokenizer = None
        current_checkpoint_window: int | None = None
        window_wait_tracker = WindowWaitTracker(log_interval_secs=120)

        async def _run() -> None:
            nonlocal model, tokenizer, current_checkpoint_window
            last_window_start = -1
            timers = MiningTimers()

            # Load R2 credentials
            try:
                credentials = load_r2_credentials()
                logger.info("✅ Loaded R2 credentials")
            except Exception as e:
                logger.error(f"Failed to load R2 credentials: {e}")
                raise

            # Initialize heartbeat (watchdog will monitor for stalls)
            self.heartbeat()
            logger.info("✅ Initialized watchdog heartbeat")

            # Get subtensor and metagraph for chain manager (use shared base method)
            subtensor = await self.get_subtensor()
            self.heartbeat()
            netuid = int(get_conf("BT_NETUID", get_conf("NETUID", 200)))

            # Fail fast if hotkey is not registered on the subnet
            miner_uid = await self.ensure_registered(wallet, netuid, role="miner")
            self.heartbeat()

            metagraph = await subtensor.metagraph(netuid)
            self.heartbeat()

            # Initialize chain manager for credential commitments
            config = SimpleNamespace(netuid=netuid)
            chain_manager = GrailChainManager(config, wallet, metagraph, subtensor, credentials)
            await chain_manager.initialize()
            logger.info("✅ Initialized chain manager and committed read credentials")
            # Ensure background chain worker stops on shutdown
            self.register_shutdown_callback(chain_manager.stop)
            self.heartbeat()

            # Initialize fast code execution pool for MBPP/HumanEval environments
            # This eliminates ~6s spawn overhead per code execution (7000x speedup)
            execution_pool: CodeExecutionPool | None = None
            try:
                execution_pool = CodeExecutionPool(
                    num_workers=4,  # Fewer workers for miner (less parallel load)
                    max_tasks_per_child=50,
                )
                execution_pool.start()
                set_global_execution_pool(execution_pool)
                logger.info("✅ Fast code execution pool initialized: %d workers", 4)
            except Exception as e:
                logger.warning("⚠️ Failed to init execution pool, using slow path: %s", e)
                execution_pool = None

            def _cleanup_execution_pool() -> None:
                nonlocal execution_pool
                if execution_pool is not None:
                    try:
                        logger.info("Shutting down code execution pool...")
                        set_global_execution_pool(None)
                        execution_pool.shutdown()
                        execution_pool = None
                        logger.info("✅ Code execution pool shutdown complete")
                    except Exception as e:
                        logger.warning(f"Error shutting down execution pool: {e}")

            self.register_shutdown_callback(_cleanup_execution_pool)
            self.heartbeat()

            # Use trainer UID's committed read credentials for checkpoints
            trainer_bucket = chain_manager.get_bucket(TRAINER_UID)
            if trainer_bucket is not None:
                logger.info(f"✅ Using trainer UID {TRAINER_UID} bucket for checkpoints")
                checkpoint_credentials = trainer_bucket
            else:
                logger.warning(
                    f"⚠️ Trainer UID {TRAINER_UID} bucket not found, using local credentials"
                )
                checkpoint_credentials = credentials

            checkpoint_manager = CheckpointManager(
                cache_root=default_checkpoint_cache_root(),
                credentials=checkpoint_credentials,
                keep_limit=2,  # Keep only current + previous window
            )

            # Pipeline state (initialized on first checkpoint load if enabled)
            pipeline_config = PipelineConfig.from_env()
            pipeline_engine = None
            weight_sync = None
            proof_worker = None

            if pipeline_config.enabled:
                logger.info(
                    "Pipeline mode ENABLED: backend=%s, gen_gpu=%d, proof_gpu=%d",
                    pipeline_config.backend,
                    pipeline_config.vllm_gpu,
                    pipeline_config.proof_gpu,
                )

            # Initialize monitoring for mining operations
            monitor = get_monitoring_manager()
            if monitor:
                mining_config = MonitoringConfig.for_mining(wallet.name)
                run_name = f"miner-{miner_uid}"
                run_id = await monitor.start_run(run_name, mining_config.get("hyperparameters", {}))
                self.heartbeat()
                logger.info(f"Started monitoring run: {run_id} (name={run_name})")

            while not self.stop_event.is_set():
                try:
                    # Update heartbeat at start of each iteration
                    self.heartbeat()

                    # Use shared subtensor from base class
                    subtensor = await self.get_subtensor()

                    current_block = await subtensor.get_current_block()  # type: ignore[misc]  # bittensor async stub
                    window_start = self.calculate_window(current_block)

                    # Set monitoring context for metrics (use block_number for x-axis)
                    if monitor:
                        monitor.set_block_context(current_block, None)

                    if window_start <= last_window_start:
                        if window_wait_tracker.should_log_initial():
                            log_window_wait_initial(
                                current_block=current_block,
                                last_processed_window=last_window_start,
                                window_length=WINDOW_LENGTH,
                            )
                        elif window_wait_tracker.should_log_periodic():
                            next_window = calculate_next_window(last_window_start, WINDOW_LENGTH)
                            log_window_wait_periodic(
                                next_window=next_window,
                                elapsed_seconds=window_wait_tracker.get_elapsed_seconds(),
                            )

                        await asyncio.sleep(2)
                        continue

                    # Window is available - reset tracker
                    window_wait_tracker.reset()

                    # Load or update checkpoint (unified fast/slow path)
                    timer_ctx = (
                        monitor.timer("profiling/checkpoint_load")
                        if monitor
                        else contextlib.nullcontext()
                    )
                    with timer_ctx:
                        result, checkpoint_path = await checkpoint_manager.load_or_update_model(
                            window_start, model, current_checkpoint_window
                        )
                    self.heartbeat()

                    if result.success:
                        if result.is_fast_path:
                            # Fast path: model already updated in-place
                            current_checkpoint_window = result.window
                            logger.info("⚡ Model updated in-place to window %s", result.window)

                            # Pipeline: sync proof worker (model is shared object,
                            # weights already updated in-place, but keep reference fresh)
                            if proof_worker is not None:
                                proof_worker.update_model_in_place(model)

                            # Pipeline: sync generation server weights
                            if pipeline_engine is not None and weight_sync is not None:
                                try:
                                    cache_path = await checkpoint_manager.await_cache_complete()
                                    if cache_path:
                                        await weight_sync.sync_weights(str(cache_path))
                                    else:
                                        logger.warning(
                                            "Pipeline: cache path not available after fast path"
                                        )
                                except Exception as sync_exc:
                                    logger.warning(
                                        "Pipeline: weight sync failed (mining continues "
                                        "with correct HF proofs): %s",
                                        sync_exc,
                                    )

                        elif checkpoint_path is not None:
                            # Slow path: load from disk
                            logger.info(
                                "🔁 Loading checkpoint for window %s from %s",
                                result.window,
                                checkpoint_path,
                            )
                            try:
                                if proof_worker is not None:
                                    # Pipeline slow-path: reload via proof_worker
                                    # (single model owner avoids duplicate on proof GPU)
                                    model = None
                                    tokenizer = None
                                    proof_worker.load_model(str(checkpoint_path))
                                    model = proof_worker._model
                                    tokenizer = proof_worker.tokenizer
                                else:
                                    model, tokenizer = clear_model_and_tokenizer(model, tokenizer)
                                    # Pipeline: load model to proof GPU to keep gen GPU free
                                    model_device = (
                                        f"cuda:{pipeline_config.proof_gpu}"
                                        if pipeline_config.enabled
                                        else None
                                    )
                                    model = get_model(
                                        str(checkpoint_path), device=model_device, eval_mode=True
                                    )
                                    tokenizer = get_tokenizer(str(checkpoint_path))

                                current_checkpoint_window = result.window

                                if torch.cuda.is_available():
                                    logger.info(
                                        f"GPU Memory: allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
                                        f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GB"
                                    )
                                    torch.cuda.empty_cache()

                                if weight_sync is not None:
                                    try:
                                        await weight_sync.sync_weights(str(checkpoint_path))
                                    except Exception as sync_exc:
                                        logger.warning(
                                            "Pipeline: weight sync failed after full load: %s",
                                            sync_exc,
                                        )

                            except Exception:
                                logger.exception(
                                    "Failed to load checkpoint for window %s", result.window
                                )
                                raise
                    elif model is None or tokenizer is None:
                        # No checkpoint and no model - skip this window, wait for next
                        logger.warning(
                            "No checkpoint for window %s, waiting for next window", window_start
                        )
                        last_window_start = window_start
                        continue

                    # Safety check: ensure model and tokenizer are loaded before mining
                    if model is None or tokenizer is None:
                        logger.error("Model or tokenizer not loaded, cannot mine")
                        last_window_start = window_start  # Prevent infinite loop
                        continue

                    # Initialize pipeline on first successful checkpoint load
                    if (
                        pipeline_config.enabled
                        and pipeline_engine is None
                        and checkpoint_path is not None
                    ):
                        try:
                            from grail.mining.engine import PipelinedMiningEngine
                            from grail.mining.proof_worker import ProofWorker
                            from grail.mining.weight_sync import (
                                SGLangWeightSync,
                                VLLMWeightSync,
                            )

                            proof_worker = ProofWorker(pipeline_config)
                            proof_worker.set_model(model, tokenizer)

                            if pipeline_config.backend == "sglang":
                                weight_sync = SGLangWeightSync(
                                    pipeline_config, proof_worker.tokenizer
                                )
                            else:
                                weight_sync = VLLMWeightSync(
                                    pipeline_config, proof_worker.tokenizer
                                )

                            await weight_sync.start(str(checkpoint_path))
                            pipeline_engine = PipelinedMiningEngine(
                                pipeline_config, weight_sync, proof_worker
                            )
                            logger.info("Pipeline engine initialized successfully")

                            # Register cleanup callbacks
                            async def _shutdown_pipeline(
                                _ws=weight_sync,
                                _pe=pipeline_engine,
                            ) -> None:
                                if _ws is not None:
                                    await _ws.shutdown()
                                if _pe is not None:
                                    _pe.shutdown()

                            def _schedule_pipeline_shutdown() -> None:
                                asyncio.ensure_future(_shutdown_pipeline())

                            self.register_shutdown_callback(_schedule_pipeline_shutdown)
                        except Exception as pipe_exc:
                            logger.error(
                                "Failed to initialize pipeline, falling back to "
                                "single-GPU mode: %s",
                                pipe_exc,
                                exc_info=True,
                            )
                            pipeline_engine = None
                            weight_sync = None
                            proof_worker = None

                    # Fetch checkpoint metadata for environment and generation configuration
                    env_id = None
                    env_params = {}
                    generation_params = {}

                    if current_checkpoint_window is not None:
                        try:
                            checkpoint_metadata = await checkpoint_manager.get_checkpoint_metadata(
                                current_checkpoint_window
                            )
                            if checkpoint_metadata:
                                missing = checkpoint_metadata.validate_metadata()
                                if missing:
                                    logger.error(
                                        "Checkpoint %s missing required metadata: %s. "
                                        "Skipping window — trainer may be misconfigured.",
                                        current_checkpoint_window,
                                        ", ".join(missing),
                                    )
                                    last_window_start = window_start
                                    continue

                                env_id = checkpoint_metadata.env_id
                                env_params = checkpoint_metadata.env_params or {}
                                generation_params = checkpoint_metadata.generation_params or {}
                                logger.info(
                                    f"Using checkpoint config: env_id={env_id}, "
                                    f"generation_params={generation_params}"
                                )
                            else:
                                logger.warning(
                                    f"No metadata found for checkpoint window {current_checkpoint_window}, "
                                    f"using defaults"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to fetch checkpoint metadata (window={current_checkpoint_window}): {e}",
                                exc_info=True,
                            )
                            # Continue with defaults on error
                    else:
                        logger.warning(
                            "current_checkpoint_window is None, using default env config"
                        )

                    logger.info(
                        f"🔥 Starting inference generation for window "
                        f"{window_start}-{window_start + WINDOW_LENGTH - 1}"
                    )

                    if not await has_time_for_next_generation(subtensor, timers, window_start):
                        last_window_start = window_start
                        await asyncio.sleep(5)
                        continue

                    window_block_hash, combined_randomness = await get_window_randomness(
                        subtensor,
                        window_start,
                        self.use_drand,
                    )

                    if pipeline_engine is not None:
                        inferences = await _pipelined_generation_loop(
                            pipeline_engine,
                            wallet,
                            model,
                            tokenizer,
                            subtensor,
                            window_start,
                            window_block_hash,
                            combined_randomness,
                            timers,
                            monitor,
                            self.use_drand,
                            current_checkpoint_window,
                            env_id=env_id,
                            env_params=env_params,
                        )
                    else:
                        inferences = await generate_rollouts_for_window(
                            wallet,
                            model,
                            tokenizer,
                            subtensor,
                            window_start,
                            window_block_hash,
                            combined_randomness,
                            timers,
                            monitor,
                            self.use_drand,
                            current_checkpoint_window,
                            env_id=env_id,
                            env_params=env_params,
                            generation_params=generation_params,
                        )

                    if inferences:
                        logger.info(
                            f"📤 Uploading {len(inferences)} rollouts to R2 "
                            f"for window {window_start}..."
                        )
                        try:
                            upload_duration = await upload_inferences_with_metrics(
                                wallet, window_start, inferences, credentials, monitor
                            )
                            timers.update_upload_time_ema(upload_duration)
                            logger.info(
                                f"✅ Successfully uploaded window {window_start} "
                                f"with {len(inferences)} rollouts"
                            )
                            self.heartbeat()
                            if monitor:
                                await monitor.log_counter("mining/successful_uploads")
                                await monitor.log_gauge("mining/uploaded_rollouts", len(inferences))

                        except Exception as e:
                            logger.error(f"❌ Failed to upload window {window_start}: {e}")
                            logger.error(traceback.format_exc())
                            if monitor:
                                await monitor.log_counter("mining/failed_uploads")
                    else:
                        logger.warning(f"No inferences generated for window {window_start}")
                        if monitor:
                            await monitor.log_counter("mining/empty_windows")

                    last_window_start = window_start
                    await checkpoint_manager.cleanup_local(window_start)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Error in miner loop: {e}. Continuing ...")
                    self.reset_subtensor()  # Force reconnect on next iteration
                    await asyncio.sleep(10)
                    continue

        # Start process-level watchdog (handled by BaseNeuron)
        self.start_watchdog(timeout_seconds=(60 * 30))
        await _run()
