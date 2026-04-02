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
    save_ddp_checkpoint,
    save_full_checkpoint,
)
from grail.trainer.distributed.compat import DistributedContext
from grail.trainer.distributed.config import DistributedConfig
from grail.trainer.distributed.parallelism import (
    apply_ddp,
    setup_ref_model,
    setup_training_model,
)

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

        # DILOCO state (initialized in _initialize_resources when strategy == "diloco")
        self._diloco_global_params: list[torch.Tensor] | None = None
        self._diloco_outer_optimizer: torch.optim.Optimizer | None = None
        self._diloco_inner_step_counter: int = 0
        self._diloco_sync_happened: bool = False

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

        # Attempt to load resume checkpoint (rank 0 restores optimizer/DILOCO state)
        if self._load_resume_checkpoint():
            logger.info(
                "Resumed from checkpoint (rank=%d, epoch=%d)", self.rank, self.epoch_counter
            )

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

        Strategy dispatch:
        - fsdp2: Load on CPU, apply TP+SP+FSDP2, load weights via DCP broadcast.
        - ddp: Load on CPU, move to device, wrap with DDP.
        - diloco: Same as ddp (each GPU holds a full model), plus initialize
          CPU-offloaded global params copy and outer Nesterov optimizer.
        """
        strategy = self.dist_config.strategy
        device = torch.device("cuda", self.local_rank)

        logger.info("Initializing resources: strategy=%s, rank=%d", strategy, self.rank)

        # Step 1: Load training model and tokenizer
        logger.info("Loading training model: %s", self.model_name_or_path)
        self.train_model = get_model(
            self.model_name_or_path,
            device="cpu",
            eval_mode=False,
        )
        self.tokenizer = get_tokenizer(self.model_name_or_path)

        # Step 2: Apply strategy-specific parallelism
        if strategy == "fsdp2":
            self._init_fsdp2(device)
        elif strategy == "ddp":
            self._init_ddp(device)
        elif strategy == "diloco":
            self._init_diloco_model(device)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step 3: Reference model (if KL is enabled)
        kl_enabled = is_kl_enabled()
        if kl_enabled and self.ref_model_name_or_path:
            logger.info("Loading reference model: %s", self.ref_model_name_or_path)
            self.ref_model = get_model(
                self.ref_model_name_or_path,
                device="cpu",
                eval_mode=True,
            )
            if strategy == "fsdp2":
                setup_ref_model(self.ref_model)
                load_hf_into_distributed(self.ref_model, self.ref_model_name_or_path, self.rank)
            else:
                # DDP/DILOCO: load weights normally, move to device
                self.ref_model.to(device)
            self.ref_model.eval()
        else:
            self.ref_model = None

        # Step 4: Optimizer (after parallelism wrapping)
        params = self.train_model.parameters()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.train_config.lr,
            betas=OPTIMIZER_BETAS,
            weight_decay=OPTIMIZER_WEIGHT_DECAY,
            foreach=True,
        )
        logger.info("Optimizer created (AdamW, lr=%.2e, foreach=True)", self.train_config.lr)

        # Step 5: Scheduler
        self.scheduler = self._create_scheduler()

        # Step 6: Algorithm
        self.algorithm = GRPOAlgorithm(config=self.train_config)

        # Step 7: DistributedContext (Accelerator shim)
        self.context = DistributedContext(
            device=device,
            rank=self.rank,
            world_size=self.world_size,
            dp_group=self.dp_mesh.get_group(),
            dp_size=self.dp_size,
            strategy=strategy,
        )

        # Step 8: DILOCO outer optimization state
        if strategy == "diloco":
            self._init_diloco()

        dist.barrier()
        logger.info("Resource initialization complete (strategy=%s, rank=%d)", strategy, self.rank)

    def _init_fsdp2(self, device: torch.device) -> None:
        """Initialize model with FSDP2 parallelism."""
        logger.info("Applying FSDP2 parallelism (rank=%d)", self.rank)
        setup_training_model(self.train_model, self.mesh, self.dist_config)
        load_hf_into_distributed(self.train_model, self.model_name_or_path, self.rank)

    def _init_ddp(self, device: torch.device) -> None:
        """Initialize model with DDP wrapping."""
        logger.info("Applying DDP parallelism (rank=%d)", self.rank)
        self.train_model.to(device)

        # Enable gradient checkpointing for memory savings
        if hasattr(self.train_model, "gradient_checkpointing_enable"):
            self.train_model.gradient_checkpointing_enable(  # type: ignore[operator]
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled (DDP path)")

        self.train_model = apply_ddp(self.train_model, self.local_rank)

    def _init_diloco_model(self, device: torch.device) -> None:
        """Initialize model for DILOCO: bare model, no DDP wrapping.

        DILOCO requires each GPU to train independently (no gradient sync)
        for H inner steps. DDP wraps the model and all-reduces gradients on
        every backward call, which would defeat DILOCO's purpose. Instead,
        we just move the model to the device and enable gradient checkpointing.
        """
        logger.info("Initializing bare model for DILOCO (rank=%d, no DDP)", self.rank)
        self.train_model.to(device)

        if hasattr(self.train_model, "gradient_checkpointing_enable"):
            self.train_model.gradient_checkpointing_enable(  # type: ignore[operator]
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled (DILOCO path)")

    def _init_diloco(self) -> None:
        """Initialize DILOCO outer optimization state.

        Creates:
        - CPU copy of global parameters (theta_global)
        - Nesterov SGD outer optimizer operating on the global params
        - Step counter for tracking inner steps between syncs
        """
        assert self.train_model is not None

        # Snapshot global params to CPU. These are the "consensus" parameters
        # that the outer optimizer updates every H inner steps.
        unwrapped = (
            self.train_model.module if hasattr(self.train_model, "module") else self.train_model
        )
        self._diloco_global_params = [p.detach().clone().cpu() for p in unwrapped.parameters()]

        # Outer optimizer: Nesterov SGD on the global params.
        # The .grad field will be filled with pseudo-gradients before each step.
        for p in self._diloco_global_params:
            p.requires_grad = True

        outer_momentum = self.dist_config.diloco_outer_momentum
        self._diloco_outer_optimizer = torch.optim.SGD(
            self._diloco_global_params,
            lr=self.dist_config.diloco_outer_lr,
            momentum=outer_momentum,
            nesterov=outer_momentum > 0,  # PyTorch requires nesterov=False when momentum=0
        )

        self._diloco_inner_step_counter = 0

        # PULSE-DiLoCo: residual buffer for BF16-gated sparse communication
        self._pulse_residual_buffer: torch.Tensor | None = None
        if self.dist_config.pulse_diloco:
            total_numel = sum(p.numel() for p in self._diloco_global_params)
            self._pulse_residual_buffer = torch.zeros(total_numel, dtype=torch.float32)
            logger.info(
                "PULSE-DiLoCo residual buffer initialized: %d params, %.1f MB",
                total_numel,
                total_numel * 4 / 1e6,
            )

        logger.info(
            "DILOCO initialized: H=%d, outer_lr=%.3f, outer_momentum=%.2f, "
            "pulse=%s, global_params on CPU (%d tensors)",
            self.dist_config.diloco_inner_steps,
            self.dist_config.diloco_outer_lr,
            self.dist_config.diloco_outer_momentum,
            self.dist_config.pulse_diloco,
            len(self._diloco_global_params),
        )

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

        # DILOCO: track inner steps and trigger outer sync when H is reached
        if self.dist_config.strategy == "diloco":
            self._diloco_sync_happened = False
            self._diloco_inner_step_counter += 1
            diloco_metrics = self._diloco_maybe_outer_step()
            if diloco_metrics:
                metrics.update(diloco_metrics)

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
    # DILOCO outer optimization
    # ------------------------------------------------------------------

    def _diloco_maybe_outer_step(self) -> dict[str, float]:
        """Check if H inner steps have elapsed and perform outer sync if so.

        Returns:
            Dictionary of DILOCO-specific metrics (empty if no sync happened).
        """
        H = self.dist_config.diloco_inner_steps
        if self._diloco_inner_step_counter % H != 0:
            return {}

        if self.dist_config.pulse_diloco:
            return self._pulse_diloco_outer_step()
        return self._diloco_outer_step()

    def _diloco_outer_step(self) -> dict[str, float]:
        """Perform the DILOCO outer optimization step.

        Algorithm:
        1. Compute pseudo-gradients: delta_i = theta_global - theta_local_i
        2. All-reduce pseudo-gradients across all workers (average)
        3. Apply outer Nesterov SGD update to global params
        4. Overwrite local model params with the updated global params

        Returns:
            Dictionary with DILOCO metrics (pseudo_grad_norm, etc.)
        """
        assert self._diloco_global_params is not None
        assert self._diloco_outer_optimizer is not None
        assert self.train_model is not None
        assert self.context is not None

        device = self.context.device
        dp_group = self.dp_mesh.get_group()
        dp_size = self.dp_size

        # DILOCO uses a bare model (no DDP wrapper), so no unwrapping needed
        local_params = list(self.train_model.parameters())

        pseudo_grad_norm_sq = 0.0
        sync_start = time.time()

        # Step 1-2: Compute pseudo-gradients and all-reduce
        for global_p, local_p in zip(self._diloco_global_params, local_params, strict=True):
            # pseudo_gradient = theta_global - theta_local (on GPU for NCCL)
            delta = global_p.to(device) - local_p.data
            # All-reduce: average across DP workers (not TP ranks)
            dist.all_reduce(delta, op=dist.ReduceOp.SUM, group=dp_group)
            delta.div_(dp_size)
            # Store as .grad on the CPU global param for the outer optimizer
            grad = delta.cpu()
            global_p.grad = grad
            pseudo_grad_norm_sq += grad.norm().item() ** 2

        # Step 3: Outer Nesterov SGD step
        self._diloco_outer_optimizer.step()
        self._diloco_outer_optimizer.zero_grad()

        # Step 4: Overwrite local model params with updated global params (consensus)
        with torch.no_grad():
            for global_p, local_p in zip(self._diloco_global_params, local_params, strict=True):
                local_p.data.copy_(global_p.to(device))

        self._diloco_sync_happened = True

        sync_duration = time.time() - sync_start
        pseudo_grad_norm = pseudo_grad_norm_sq**0.5

        if self._is_rank0:
            logger.info(
                "DILOCO outer step %d: pseudo_grad_norm=%.4f, sync_time=%.2fs",
                self._diloco_inner_step_counter // self.dist_config.diloco_inner_steps,
                pseudo_grad_norm,
                sync_duration,
            )

        return {
            "diloco/pseudo_grad_norm": pseudo_grad_norm,
            "diloco/sync_duration": sync_duration,
            "diloco/outer_step": float(
                self._diloco_inner_step_counter // self.dist_config.diloco_inner_steps
            ),
        }

    def _pulse_diloco_outer_step(self) -> dict[str, float]:
        """PULSE-DiLoCo outer step: BF16-gated sparse communication with residual buffer.

        Instead of dense all-reduce, computes BF16-visible pseudo-gradients,
        communicates only non-zero entries via sparse all-gather, and accumulates
        invisible residuals for future rounds.

        Algorithm:
        1. Compute residual-corrected BF16-visible pseudo-gradients
        2. Sparse all-gather (exchange only non-zero entries)
        3. Outer Nesterov SGD step on averaged pseudo-gradients
        4. Hard reset: overwrite local params with consensus

        Returns:
            Dictionary with PULSE-DiLoCo metrics.
        """
        assert self._diloco_global_params is not None
        assert self._diloco_outer_optimizer is not None
        assert self.train_model is not None
        assert self.context is not None
        assert self._pulse_residual_buffer is not None

        device = self.context.device
        dp_group = self.dp_mesh.get_group()
        dp_size = self.dp_size
        local_params = list(self.train_model.parameters())

        total_numel = self._pulse_residual_buffer.numel()
        sync_start = time.time()

        # ── Step 1: Residual-corrected BF16-visible pseudo-gradient ──
        flat_pseudo_grad = torch.zeros(total_numel, dtype=torch.float32)
        offset = 0
        for global_p, local_p in zip(self._diloco_global_params, local_params, strict=True):
            numel = global_p.numel()
            param_cpu = local_p.data.to("cpu").float()
            snapshot = global_p.data.view(-1)  # theta^(t-1), flat FP32 on CPU

            delta = snapshot - param_cpu.view(-1)  # exact FP32 pseudo-grad
            s = self._pulse_residual_buffer[offset : offset + numel] + delta
            w_tilde = snapshot - s  # virtual corrected model

            b_theta = snapshot.bfloat16().float()
            b_w_tilde = w_tilde.bfloat16().float()
            delta_hat = b_theta - b_w_tilde  # BF16-visible pseudo-grad (sparse)

            flat_pseudo_grad[offset : offset + numel] = delta_hat
            self._pulse_residual_buffer[offset : offset + numel] = s - delta_hat
            offset += numel

        # ── Step 2: Sparse all-gather ──
        nonzero_mask = flat_pseudo_grad != 0
        indices = nonzero_mask.nonzero(as_tuple=True)[0]  # int64
        values = flat_pseudo_grad[nonzero_mask]  # FP32
        nnz = indices.numel()

        # Exchange counts across workers
        nnz_tensor = torch.tensor([nnz], dtype=torch.int64, device=device)
        nnz_list = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(dp_size)]
        dist.all_gather(nnz_list, nnz_tensor, group=dp_group)
        max_nnz = max(int(t.item()) for t in nnz_list)

        comm_sparsity = 1.0 - nnz / total_numel if total_numel > 0 else 0.0

        # Memory guard: estimate GPU cost of sparse all-gather.
        # Total = (dp_size + 1) * 12 * max_nnz bytes (int64 idx + float32 val).
        # If this exceeds the budget, fall back to per-parameter dense all-reduce
        # of the BF16-gated pseudo-gradient. The residual buffer is already
        # updated correctly in step 1 regardless of the communication path.
        _PULSE_GPU_BUDGET_BYTES = 4 * 1024**3  # 4 GB
        estimated_gpu_bytes = (dp_size + 1) * 12 * max_nnz
        use_sparse_gather = max_nnz > 0 and estimated_gpu_bytes <= _PULSE_GPU_BUDGET_BYTES

        if max_nnz > 0 and not use_sparse_gather and self._is_rank0:
            logger.warning(
                "PULSE memory guard: max_nnz=%d would need %.1f GB (budget=%.1f GB), "
                "falling back to dense all-reduce for this outer step",
                max_nnz,
                estimated_gpu_bytes / 1e9,
                _PULSE_GPU_BUDGET_BYTES / 1e9,
            )

        if use_sparse_gather:
            # Pad and gather on GPU (NCCL requires equal-size tensors)
            idx_pad = torch.zeros(max_nnz, dtype=torch.int64, device=device)
            val_pad = torch.zeros(max_nnz, dtype=torch.float32, device=device)
            if nnz > 0:
                idx_pad[:nnz] = indices.to(device)
                val_pad[:nnz] = values.to(device)

            all_idx = [torch.zeros_like(idx_pad) for _ in range(dp_size)]
            all_val = [torch.zeros_like(val_pad) for _ in range(dp_size)]
            dist.all_gather(all_idx, idx_pad, group=dp_group)
            dist.all_gather(all_val, val_pad, group=dp_group)

            # Reconstruct averaged pseudo-gradient on CPU via scatter-add
            avg_pseudo_grad = torch.zeros(total_numel, dtype=torch.float32)
            for r in range(dp_size):
                n = int(nnz_list[r].item())
                if n > 0:
                    avg_pseudo_grad.index_add_(0, all_idx[r][:n].cpu().long(), all_val[r][:n].cpu())
            avg_pseudo_grad.div_(dp_size)
        elif max_nnz > 0:
            # Dense fallback: all-reduce BF16-gated pseudo-grad per parameter.
            # Same math as sparse path but bounded by largest single parameter.
            avg_pseudo_grad = torch.zeros(total_numel, dtype=torch.float32)
            offset_dr = 0
            for global_p in self._diloco_global_params:
                numel = global_p.numel()
                chunk = flat_pseudo_grad[offset_dr : offset_dr + numel].to(device)
                dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=dp_group)
                chunk.div_(dp_size)
                avg_pseudo_grad[offset_dr : offset_dr + numel] = chunk.cpu()
                offset_dr += numel
            comm_sparsity = 0.0  # dense fallback, no sparsity benefit
        else:
            avg_pseudo_grad = torch.zeros(total_numel, dtype=torch.float32)

        # ── Step 3: Outer optimizer ──
        # Snapshot global params before step (for weight-update sparsity metric)
        snapshots_before = [gp.data.clone() for gp in self._diloco_global_params]

        offset = 0
        for global_p in self._diloco_global_params:
            numel = global_p.numel()
            global_p.grad = avg_pseudo_grad[offset : offset + numel].view(global_p.shape)
            offset += numel

        self._diloco_outer_optimizer.step()
        self._diloco_outer_optimizer.zero_grad()

        # ── Step 4: Hard reset ──
        with torch.no_grad():
            for global_p, local_p in zip(self._diloco_global_params, local_params, strict=True):
                local_p.data.copy_(global_p.to(device))

        self._diloco_sync_happened = True

        # ── Metrics ──
        sync_duration = time.time() - sync_start
        pseudo_grad_norm = avg_pseudo_grad.norm().item()

        # Weight-update BF16 sparsity (PULSE paper definition)
        bf16_changed = 0
        for gp, snap in zip(self._diloco_global_params, snapshots_before, strict=True):
            bf16_changed += int(
                (gp.data.bfloat16().float() != snap.bfloat16().float()).sum().item()
            )
        weight_bf16_sparsity = 1.0 - bf16_changed / total_numel if total_numel > 0 else 0.0

        outer_step_num = self._diloco_inner_step_counter // self.dist_config.diloco_inner_steps
        if self._is_rank0:
            logger.info(
                "PULSE-DiLoCo outer step %d: pseudo_grad_norm=%.4f, "
                "comm_sparsity=%.2f%%, weight_bf16_sparsity=%.2f%%, "
                "nnz=%d/%d, sync_time=%.2fs",
                outer_step_num,
                pseudo_grad_norm,
                comm_sparsity * 100,
                weight_bf16_sparsity * 100,
                nnz,
                total_numel,
                sync_duration,
            )

        return {
            "diloco/pseudo_grad_norm": pseudo_grad_norm,
            "diloco/sync_duration": sync_duration,
            "diloco/outer_step": float(outer_step_num),
            "pulse/comm_sparsity": comm_sparsity,
            "pulse/weight_bf16_sparsity": weight_bf16_sparsity,
            "pulse/nnz": float(nnz),
            "pulse/total_numel": float(total_numel),
        }

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

        Dispatches to the correct checkpoint function based on strategy:
        - fsdp2: ``save_full_checkpoint`` (collective gather + rank-0 write)
        - ddp/diloco: ``save_ddp_checkpoint`` (rank-0 only, no collective)

        Args:
            metrics: Training metrics for metadata.
        """
        if self.train_model is None or self.tokenizer is None:
            return

        checkpoint_dir = self.snapshot_dir / f"checkpoint-{self.epoch_counter}"

        try:
            if self.dist_config.strategy == "fsdp2":
                save_full_checkpoint(
                    self.train_model,
                    self.tokenizer,
                    checkpoint_dir,
                    self.rank,
                )
            else:
                save_ddp_checkpoint(
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

            # Save resume state (rank 0 only, includes optimizer + DILOCO state)
            if self._is_rank0:
                try:
                    self._save_resume_checkpoint()
                except Exception as resume_exc:
                    logger.warning("Resume checkpoint save failed: %s", resume_exc)

        except Exception as exc:
            logger.exception("Checkpoint save failed (rank=%d): %s", self.rank, exc)

    # ------------------------------------------------------------------
    # Resume checkpoint (separate from weight sync)
    # ------------------------------------------------------------------

    def _save_resume_checkpoint(self) -> None:
        """Save full training state for crash recovery (rank 0 only).

        This is separate from the weight sync checkpoint (HF safetensors uploaded
        to R2). The resume checkpoint contains everything needed to restart
        training from the exact point where it left off, including optimizer
        states, scheduler, DILOCO outer optimizer, and PULSE residual buffer.

        Saved as a single .pt file to ``snapshot_dir/resume_state.pt``.
        """
        if not self._is_rank0:
            return

        state: dict[str, Any] = {
            "epoch_counter": self.epoch_counter,
        }

        # Inner optimizer (AdamW) state
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        # LR scheduler state
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        # DILOCO-specific state
        if self.dist_config.strategy == "diloco":
            state["diloco_inner_step_counter"] = self._diloco_inner_step_counter

            if self._diloco_global_params is not None:
                state["diloco_global_params"] = [p.data.clone() for p in self._diloco_global_params]

            if self._diloco_outer_optimizer is not None:
                state["diloco_outer_optimizer_state_dict"] = (
                    self._diloco_outer_optimizer.state_dict()
                )

            # PULSE residual buffer
            if self._pulse_residual_buffer is not None:
                state["pulse_residual_buffer"] = self._pulse_residual_buffer.clone()

        resume_path = self.snapshot_dir / "resume_state.pt"
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, resume_path)
        logger.info("Resume checkpoint saved: %s (epoch=%d)", resume_path, self.epoch_counter)

    def _load_resume_checkpoint(self) -> bool:
        """Load training state from a resume checkpoint if available.

        Must be called AFTER ``_initialize_resources()`` so that optimizer,
        scheduler, and DILOCO state exist and can receive the loaded state.

        Returns:
            True if a resume checkpoint was loaded, False otherwise.
        """
        resume_path = self.snapshot_dir / "resume_state.pt"
        if not resume_path.exists():
            return False

        if not self._is_rank0:
            # Only rank 0 loads; state will be synchronized via the next
            # outer step (DILOCO) or is already identical (DDP/FSDP2).
            return False

        try:
            state = torch.load(resume_path, map_location="cpu", weights_only=False)

            self.epoch_counter = state.get("epoch_counter", 0)

            if "optimizer_state_dict" in state and self.optimizer is not None:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])

            if "scheduler_state_dict" in state and self.scheduler is not None:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])

            # DILOCO state
            if self.dist_config.strategy == "diloco":
                self._diloco_inner_step_counter = state.get("diloco_inner_step_counter", 0)

                if "diloco_global_params" in state and self._diloco_global_params is not None:
                    for stored, current in zip(
                        state["diloco_global_params"],
                        self._diloco_global_params,
                        strict=True,
                    ):
                        current.data.copy_(stored)

                if (
                    "diloco_outer_optimizer_state_dict" in state
                    and self._diloco_outer_optimizer is not None
                ):
                    self._diloco_outer_optimizer.load_state_dict(
                        state["diloco_outer_optimizer_state_dict"]
                    )

                if "pulse_residual_buffer" in state and self._pulse_residual_buffer is not None:
                    self._pulse_residual_buffer.copy_(state["pulse_residual_buffer"])

            logger.info(
                "Resume checkpoint loaded: epoch=%d, inner_step=%d",
                self.epoch_counter,
                self._diloco_inner_step_counter,
            )
            return True

        except Exception as exc:
            logger.warning("Failed to load resume checkpoint: %s", exc)
            return False

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

        # DILOCO cleanup
        self._diloco_global_params = None
        self._diloco_outer_optimizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cleanup complete (rank=%d)", self.rank)
