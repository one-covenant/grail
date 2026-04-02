"""Distributed training service for FSDP2 + Tensor Parallelism GRPO training.

Replaces ``TrainingService`` from ``grail.trainer.training_process`` for
multi-GPU training across data-parallel and tensor-parallel dimensions.
Coordinates model setup (TP+SP, FSDP2, gradient checkpointing), training
loop orchestration, and distributed checkpointing.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from grail.model.provider import get_model, get_tokenizer
from grail.shared.config import TOTAL_TRAINING_WINDOWS, is_kl_enabled
from grail.trainer.algorithms import GRPOAlgorithm
from grail.trainer.algorithms.grpo import GRPOGroup
from grail.trainer.config import TrainingConfig
from grail.trainer.distributed.checkpoint import (
    load_hf_into_distributed,
    save_full_checkpoint,
)
from grail.trainer.distributed.compat import DistributedContext
from grail.trainer.distributed.config import DistributedConfig
from grail.trainer.distributed.parallelism import setup_ref_model, setup_training_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optimizer hyperparameters (mirrored from training_process.py)
# ---------------------------------------------------------------------------

OPTIMIZER_BETAS = (0.9, 0.95)
OPTIMIZER_WEIGHT_DECAY = 0.1

# Scheduler hyperparameters
WARMUP_FRACTION = float(os.getenv("GRAIL_WARMUP_FRACTION", "0.05"))
SCHEDULER_ETA_MIN = float(os.getenv("GRAIL_SCHEDULER_ETA_MIN", "1e-7"))
LR_SCHEDULER_TYPE = os.getenv("GRAIL_LR_SCHEDULER_TYPE", "constant").strip().lower()

# Training loop timing constants
NO_DATA_SLEEP_SECONDS = 60
EPOCH_FAILURE_SLEEP_SECONDS = 30
LOOP_ERROR_SLEEP_SECONDS = 30
PAUSE_CHECK_INTERVAL_SECONDS = 5

# Pause marker filename (rank 0 checks for this file)
PAUSE_MARKER_FILENAME = "PAUSE_TRAINING"


class DistributedTrainingService:
    """FSDP2 + TP distributed training service for GRPO.

    Manages the full lifecycle of distributed training: model initialization
    with tensor and data parallelism, training loop with gradient accumulation,
    checkpoint saving (both full HF and sharded DCP), and pause/resume
    coordination across ranks.
    """

    def __init__(
        self,
        train_config: TrainingConfig,
        dist_config: DistributedConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        mesh: DeviceMesh,
        *,
        snapshot_dir: str | Path | None = None,
        credentials: Any | None = None,
        model_name_or_path: str | None = None,
        ref_model_name_or_path: str | None = None,
    ) -> None:
        self.train_config = train_config
        self.dist_config = dist_config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.mesh = mesh

        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path("snapshots")
        self.credentials = credentials
        self.model_name_or_path = model_name_or_path or os.getenv("GRAIL_TRAIN_MODEL_ID", "")
        self.ref_model_name_or_path = ref_model_name_or_path or os.getenv("GRAIL_REF_MODEL_ID", "")

        # DP/TP sub-mesh references
        self.dp_mesh: DeviceMesh = mesh["dp"]
        self.tp_mesh: DeviceMesh = mesh["tp"]
        self.dp_rank: int = self.dp_mesh.get_local_rank()
        self.dp_size: int = self.dp_mesh.size()
        self.tp_size: int = self.tp_mesh.size()

        # TP process group for distributed logprobs (None when tp_size == 1)
        self._tp_group: dist.ProcessGroup | None = None
        if self.tp_size > 1:
            self._tp_group = self.tp_mesh.get_group()

        # Models and training state (initialized in _initialize_resources)
        self.train_model: Any = None
        self.ref_model: Any = None
        self.tokenizer: Any = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.algorithm: GRPOAlgorithm | None = None
        self.context: DistributedContext | None = None

        # Counters
        self.epoch_counter: int = 0

        self._is_rank0 = rank == 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, stop_event: Event) -> None:
        """Main entry point: initialize resources, then run the training loop.

        All ranks must call this method. Collective operations (FSDP2 gather,
        all-reduce, barrier) require participation from every rank.

        Args:
            stop_event: Multiprocessing event signalling shutdown.
        """
        logger.info(
            "DistributedTrainingService starting (rank=%d, world=%d, dp=%d, tp=%d)",
            self.rank,
            self.world_size,
            self.dp_size,
            self.tp_size,
        )

        self._initialize_resources()

        await self._training_loop(stop_event)

        logger.info(
            "DistributedTrainingService exiting (rank=%d, epochs=%d)",
            self.rank,
            self.epoch_counter,
        )

    # ------------------------------------------------------------------
    # Resource initialization
    # ------------------------------------------------------------------

    def _initialize_resources(self) -> None:
        """Load models, apply parallelism, create optimizer and algorithm.

        Execution order matters:
        1. Load model weights on CPU/meta device (no .to(device) yet)
        2. Apply TP+SP, gradient checkpointing, FSDP2 (setup_training_model)
        3. Load HF weights into the sharded model (load_hf_into_distributed)
        4. Create optimizer AFTER FSDP2 wrapping (parameters are sharded)
        5. Create scheduler, algorithm, and DistributedContext
        """
        device = torch.device("cuda", self.local_rank)

        # Step 1: Load training model on CPU (skip .to(device), FSDP2 handles placement)
        logger.info("Loading training model: %s", self.model_name_or_path)
        self.train_model = get_model(
            self.model_name_or_path,
            device="cpu",
            eval_mode=False,
        )
        self.tokenizer = get_tokenizer(self.model_name_or_path)

        # Step 2: Apply distributed parallelism (RMSNorm replacement, TP+SP, GC, FSDP2)
        logger.info("Applying distributed parallelism (rank=%d)", self.rank)
        setup_training_model(self.train_model, self.mesh, self.dist_config)

        # Step 3: Load HF weights into the sharded model
        # load_hf_into_distributed is collective: all ranks must call it.
        logger.info("Loading weights into distributed model (rank=%d)", self.rank)
        load_hf_into_distributed(self.train_model, self.model_name_or_path, self.rank)

        # Step 4: Reference model (if KL is enabled)
        kl_enabled = is_kl_enabled()
        if kl_enabled and self.ref_model_name_or_path:
            logger.info("Loading reference model: %s", self.ref_model_name_or_path)
            self.ref_model = get_model(
                self.ref_model_name_or_path,
                device="cpu",
                eval_mode=True,
            )
            setup_ref_model(self.ref_model)
            load_hf_into_distributed(self.ref_model, self.ref_model_name_or_path, self.rank)
            self.ref_model.eval()
        else:
            self.ref_model = None

        # Step 5: Optimizer (AFTER FSDP2, so parameters are sharded DTensors).
        # Use foreach=True for fused multi-tensor updates on CUDA.
        self.optimizer = torch.optim.AdamW(
            self.train_model.parameters(),
            lr=self.train_config.lr,
            betas=OPTIMIZER_BETAS,
            weight_decay=OPTIMIZER_WEIGHT_DECAY,
            foreach=True,
        )
        logger.info("Optimizer created (AdamW, lr=%.2e, foreach=True)", self.train_config.lr)

        # Step 6: Scheduler
        self.scheduler = self._create_scheduler()

        # Step 7: Algorithm
        self.algorithm = GRPOAlgorithm(config=self.train_config)

        # Step 8: DistributedContext (Accelerator shim for GRPOAlgorithm)
        self.context = DistributedContext(
            device=device,
            rank=self.rank,
            world_size=self.world_size,
            dp_group=self.dp_mesh.get_group(),
            dp_size=self.dp_size,
        )

        dist.barrier()
        logger.info("Resource initialization complete (rank=%d)", self.rank)

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create LR scheduler with warmup, matching training_process.py."""
        assert self.optimizer is not None
        warmup_steps = max(1, int(WARMUP_FRACTION * TOTAL_TRAINING_WINDOWS))

        def lr_lambda_warmup(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda_warmup)

        if LR_SCHEDULER_TYPE == "cosine":
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=TOTAL_TRAINING_WINDOWS - warmup_steps,
                eta_min=SCHEDULER_ETA_MIN,
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            logger.info(
                "Scheduler: warmup (%d steps) + cosine (T_max=%d, eta_min=%.2e)",
                warmup_steps,
                TOTAL_TRAINING_WINDOWS - warmup_steps,
                SCHEDULER_ETA_MIN,
            )
        else:
            scheduler = warmup_scheduler
            logger.info("Scheduler: warmup (%d steps) + constant", warmup_steps)

        return scheduler

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    async def _training_loop(self, stop_event: Event) -> None:
        """Main loop: fetch groups, train epochs, save checkpoints.

        The loop structure mirrors ``TrainingService._training_loop`` but
        replaces Accelerator with DistributedContext and adds rank-aware
        data splitting and checkpoint coordination.

        Args:
            stop_event: Multiprocessing event signalling shutdown.
        """
        logger.info("Training loop starting (rank=%d)", self.rank)

        while not stop_event.is_set():
            try:
                # Check for pause (rank 0 checks filesystem, broadcasts to others)
                if await self._handle_pause(stop_event):
                    if stop_event.is_set():
                        break
                    continue

                # Fetch training groups. In a full deployment this would query
                # the replay buffer or load from R2. For the initial distributed
                # implementation, callers supply groups via _run_epoch_with_groups.
                # The loop skeleton is kept for structural parity with the
                # single-GPU service.
                groups = await self._fetch_groups()
                if not groups:
                    if self._is_rank0:
                        logger.warning(
                            "No training groups available, sleeping %ds",
                            NO_DATA_SLEEP_SECONDS,
                        )
                    await asyncio.sleep(NO_DATA_SLEEP_SECONDS)
                    continue

                # Train one epoch
                metrics = await self._train_epoch(groups)

                # Save checkpoint
                self._save_snapshot(metrics)

                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self.epoch_counter += 1
                if self._is_rank0:
                    logger.info(
                        "Epoch %d complete: loss=%.4f",
                        self.epoch_counter,
                        metrics.get("loss_total", 0.0),
                    )

            except asyncio.CancelledError:
                logger.info("Training loop cancelled (rank=%d)", self.rank)
                break
            except Exception as exc:
                logger.exception("Training loop error (rank=%d): %s", self.rank, exc)
                await asyncio.sleep(LOOP_ERROR_SLEEP_SECONDS)

    async def _fetch_groups(self) -> list[GRPOGroup]:
        """Fetch GRPO groups for the current training window.

        Placeholder for the full data pipeline integration. Override or extend
        this method to wire up the replay buffer and R2 data loading.

        Returns:
            List of GRPO groups (empty if no data available).
        """
        # In the full integration, this would call load_grpo_groups and
        # replay buffer sampling, similar to TrainingService._load_grpo_groups
        # and the replay buffer logic in _training_loop.
        return []

    async def run_epoch_with_groups(
        self,
        groups: list[GRPOGroup],
        stop_event: Event | None = None,
    ) -> dict[str, float]:
        """Train a single epoch with pre-supplied groups.

        Convenience method for callers that manage data loading externally
        (e.g., research scripts, integration tests). Handles the TP logprob
        monkey-patch, trains, and saves a checkpoint.

        Args:
            groups: Pre-loaded GRPO groups.
            stop_event: Optional stop event (unused in single-epoch mode).

        Returns:
            Training metrics dictionary.
        """
        metrics = await self._train_epoch(groups)
        self._save_snapshot(metrics)
        if self.scheduler is not None:
            self.scheduler.step()
        self.epoch_counter += 1
        return metrics

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    async def _train_epoch(
        self,
        groups: list[GRPOGroup],
        monitor: Any | None = None,
        window: int = 0,
    ) -> dict[str, float]:
        """Train one epoch, splitting data across DP ranks.

        1. Split groups across data-parallel ranks.
        2. Monkey-patch the logprob functions in grpo module when TP is active
           so that compute_logprobs and compute_logprobs_packed use the
           distributed logsumexp path instead of materializing full vocab logits.
        3. Call GRPOAlgorithm.train_epoch with the DistributedContext shim.
        4. Restore original logprob functions.

        Args:
            groups: Full set of GRPO groups (before DP splitting).
            monitor: Optional monitoring manager for W&B metric logging (rank 0 only).
            window: Current training window number for metric context.

        Returns:
            Training metrics dictionary.
        """
        assert self.algorithm is not None
        assert self.optimizer is not None
        assert self.context is not None
        assert self.train_model is not None

        # Split groups across DP ranks (round-robin)
        local_groups = groups[self.dp_rank :: self.dp_size]
        if not local_groups:
            logger.warning(
                "DP rank %d received 0 groups (total=%d, dp_size=%d)",
                self.dp_rank,
                len(groups),
                self.dp_size,
            )

        # Check if ANY rank has 0 groups. If so, all ranks must skip this epoch
        # to avoid deadlock (FSDP2 collectives require all ranks to participate).
        has_groups = torch.tensor(
            [1 if len(local_groups) > 0 else 0],
            device=self.context.device,
            dtype=torch.int32,
        )
        dist.all_reduce(has_groups, op=dist.ReduceOp.MIN)
        if has_groups.item() == 0:
            if self._is_rank0:
                logger.warning("Skipping epoch: at least one DP rank has 0 groups")
            return {}

        if self._is_rank0:
            logger.info(
                "Training epoch %d: %d total groups, %d local (dp_rank=%d/%d)",
                self.epoch_counter + 1,
                len(groups),
                len(local_groups),
                self.dp_rank,
                self.dp_size,
            )

        epoch_start = time.time()

        # Monkey-patch logprob functions for TP-aware computation
        patched = self._patch_logprobs_for_tp()

        try:
            metrics = await self.algorithm.train_epoch(
                self.train_model,
                self.ref_model,
                self.tokenizer,
                local_groups,
                self.optimizer,
                self.context,  # type: ignore[arg-type]  # duck-type Accelerator shim
                monitor,
                window,
                self.train_config,
            )
        finally:
            self._restore_logprobs(patched)

        epoch_duration = time.time() - epoch_start
        if self._is_rank0:
            logger.info(
                "Epoch %d trained in %.1fs (loss=%.4f, reward_mean=%.4f)",
                self.epoch_counter + 1,
                epoch_duration,
                metrics.get("loss_total", 0.0),
                metrics.get("reward_mean", 0.0),
            )

        return metrics

    # ------------------------------------------------------------------
    # TP logprob monkey-patching
    # ------------------------------------------------------------------

    def _patch_logprobs_for_tp(self) -> dict[str, Any] | None:
        """Replace module-level logprob functions with TP-aware wrappers.

        When TP is active (tp_size > 1), the lm_head output is sharded across
        TP ranks. The standard compute_logprobs path would see partial vocab
        logits and produce incorrect results. We temporarily replace the
        module-level functions with wrappers that route through
        ``compute_logprobs_distributed`` which uses distributed logsumexp.

        Returns:
            A dict of original functions to restore, or None if no patching
            was needed (tp_size == 1).
        """
        if self._tp_group is None:
            return None

        import grail.trainer.algorithms.grpo as grpo_module
        from grail.trainer.distributed.logprobs import compute_logprobs_distributed

        originals: dict[str, Any] = {
            "compute_logprobs": grpo_module.compute_logprobs,
            "compute_logprobs_packed": grpo_module.compute_logprobs_packed,
        }

        tp_group = self._tp_group
        chunk_size = self.train_config.logit_chunk_size

        # Wrapper for the padded (non-packed) path
        def _tp_compute_logprobs(
            model: Any,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            prompt_lengths: list[int],
            completion_lengths: list[int],
            return_per_token: bool = False,
            return_entropy: bool = False,
            *,
            chunked: bool = False,
            chunk_size: int = chunk_size,
        ) -> (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            logprobs_sum, logprobs_pt, entropies = compute_logprobs_distributed(
                model,
                input_ids,
                attention_mask,
                prompt_lengths,
                completion_lengths,
                chunked=chunked,
                chunk_size=chunk_size,
                tp_group=tp_group,
                return_per_token=return_per_token,
                return_entropy=return_entropy,
            )

            # Match the return convention of the original function
            if return_per_token and return_entropy and logprobs_pt is not None:
                assert entropies is not None
                return logprobs_sum, logprobs_pt, entropies
            if return_per_token and logprobs_pt is not None:
                return logprobs_sum, logprobs_pt
            return logprobs_sum

        # Wrapper for the packed path. The packed path does not use TP-aware
        # logprobs (sequence packing is incompatible with TP sequence sharding
        # in the current implementation). Fall back to the original function,
        # which will work correctly when TP is 1 on this dimension, or raise
        # an informative error.
        def _tp_compute_logprobs_packed(
            model: Any,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            seq_lengths: list[int],
            prompt_lengths: list[int],
            completion_lengths: list[int],
            return_per_token: bool = False,
            return_entropy: bool = False,
            *,
            chunked: bool = False,
            chunk_size: int = chunk_size,
        ) -> (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ):
            # Packing + TP is not yet supported. Log a warning on first call
            # and delegate to the original implementation. This is safe when
            # use_sequence_packing is disabled (the default for distributed).
            logger.warning(
                "compute_logprobs_packed called under TP; "
                "falling back to standard packed path (TP logprobs not applied)"
            )
            return originals["compute_logprobs_packed"](
                model,
                input_ids,
                position_ids,
                seq_lengths,
                prompt_lengths,
                completion_lengths,
                return_per_token=return_per_token,
                return_entropy=return_entropy,
                chunked=chunked,
                chunk_size=chunk_size,
            )

        grpo_module.compute_logprobs = _tp_compute_logprobs  # type: ignore[assignment]
        grpo_module.compute_logprobs_packed = _tp_compute_logprobs_packed  # type: ignore[assignment]

        logger.debug("Logprob functions patched for TP (tp_size=%d)", self.tp_size)
        return originals

    def _restore_logprobs(self, originals: dict[str, Any] | None) -> None:
        """Restore original logprob functions after training epoch."""
        if originals is None:
            return

        import grail.trainer.algorithms.grpo as grpo_module

        grpo_module.compute_logprobs = originals["compute_logprobs"]  # type: ignore[assignment]
        grpo_module.compute_logprobs_packed = originals["compute_logprobs_packed"]  # type: ignore[assignment]
        logger.debug("Logprob functions restored to originals")

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_snapshot(self, metrics: dict[str, Any]) -> None:
        """Save checkpoint after an epoch.

        Uses ``save_full_checkpoint`` for HF-compatible safetensors output.
        All ranks must participate (the function performs collective gathers),
        but only rank 0 writes files to disk.

        Args:
            metrics: Training metrics for metadata.
        """
        if self.train_model is None or self.tokenizer is None:
            return

        checkpoint_dir = self.snapshot_dir / f"checkpoint-{self.epoch_counter}"

        try:
            save_full_checkpoint(
                self.train_model,
                self.tokenizer,
                checkpoint_dir,
                self.rank,
            )

            # Rank 0 writes metadata alongside the checkpoint
            if self._is_rank0:
                metadata = {
                    "epoch": self.epoch_counter,
                    "timestamp": time.time(),
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "lr": (
                        self.scheduler.get_last_lr()[0] if self.scheduler else self.train_config.lr
                    ),
                    "world_size": self.world_size,
                    "dp_size": self.dp_size,
                    "tp_size": self.tp_size,
                }
                metadata_path = checkpoint_dir / "snapshot_metadata.json"
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(
                    "Checkpoint saved: %s (epoch=%d)",
                    checkpoint_dir,
                    self.epoch_counter,
                )

        except Exception as exc:
            logger.exception("Checkpoint save failed (rank=%d): %s", self.rank, exc)

    # ------------------------------------------------------------------
    # Pause/resume coordination
    # ------------------------------------------------------------------

    async def _handle_pause(self, stop_event: Event) -> bool:
        """Check for pause marker and coordinate across ranks.

        Rank 0 polls the filesystem for a pause marker file. When detected,
        it broadcasts a flag to all ranks. All ranks then participate in a
        checkpoint save before entering a wait loop. Rank 0 polls for the
        marker to be removed (resume signal).

        Args:
            stop_event: Shutdown signal.

        Returns:
            True if a pause was handled (caller should ``continue`` the loop).
            False if no pause was requested.
        """
        # Rank 0 checks for the pause marker
        pause_flag = torch.zeros(1, dtype=torch.int32, device=f"cuda:{self.local_rank}")
        if self._is_rank0:
            marker = self.snapshot_dir / PAUSE_MARKER_FILENAME
            if marker.exists():
                pause_flag.fill_(1)

        # Broadcast decision to all ranks
        dist.broadcast(pause_flag, src=0)

        if pause_flag.item() == 0:
            return False

        logger.info("Pause requested (rank=%d), saving checkpoint before pause", self.rank)

        # All ranks participate in checkpoint save
        self._save_snapshot({"status": "paused"})

        # Synchronize before entering wait loop
        dist.barrier()

        if self._is_rank0:
            logger.info("All ranks paused, waiting for resume signal")

        # Wait for resume (rank 0 polls, broadcasts to others)
        while not stop_event.is_set():
            resume_flag = torch.zeros(1, dtype=torch.int32, device=f"cuda:{self.local_rank}")
            if self._is_rank0:
                marker = self.snapshot_dir / PAUSE_MARKER_FILENAME
                if not marker.exists():
                    resume_flag.fill_(1)

            dist.broadcast(resume_flag, src=0)
            if resume_flag.item() == 1:
                break

            await asyncio.sleep(PAUSE_CHECK_INTERVAL_SECONDS)

        if self._is_rank0:
            logger.info("Resume signal received, continuing training")

        dist.barrier()
        return True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release GPU memory and destroy process group.

        Should be called in a finally block after ``run()`` completes.
        """
        del self.train_model
        del self.ref_model
        del self.optimizer
        self.train_model = None
        self.ref_model = None
        self.optimizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup complete (rank=%d)", self.rank)
