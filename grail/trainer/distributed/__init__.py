"""Distributed training primitives for multi-strategy parallelism (FSDP2, DDP, DILOCO)."""

from grail.trainer.distributed.checkpoint import (
    async_save_sharded_checkpoint,
    create_checkpoint_stager,
    load_hf_into_distributed,
    load_sharded_checkpoint,
    save_ddp_checkpoint,
    save_full_checkpoint,
    save_sharded_checkpoint,
)
from grail.trainer.distributed.compat import DistributedContext
from grail.trainer.distributed.config import DistributedConfig, StrategyType
from grail.trainer.distributed.grad_utils import check_grad_nan_across_ranks, clip_grad_norm_
from grail.trainer.distributed.logprobs import (
    compute_logprobs_distributed,
    tp_chunked_logprobs,
)
from grail.trainer.distributed.parallelism import (
    apply_ddp,
    apply_fsdp2,
    apply_gradient_checkpointing,
    apply_tp_sp,
    create_device_mesh,
    replace_rmsnorm_for_sp,
    setup_ref_model,
    setup_training_model,
)
from grail.trainer.distributed.training_service import DistributedTrainingService

__all__ = [
    "DistributedConfig",
    "StrategyType",
    "DistributedContext",
    "DistributedTrainingService",
    "check_grad_nan_across_ranks",
    "clip_grad_norm_",
    "apply_ddp",
    "apply_fsdp2",
    "apply_gradient_checkpointing",
    "apply_tp_sp",
    "async_save_sharded_checkpoint",
    "compute_logprobs_distributed",
    "create_checkpoint_stager",
    "create_device_mesh",
    "load_hf_into_distributed",
    "load_sharded_checkpoint",
    "replace_rmsnorm_for_sp",
    "save_ddp_checkpoint",
    "save_full_checkpoint",
    "save_sharded_checkpoint",
    "setup_ref_model",
    "setup_training_model",
    "tp_chunked_logprobs",
]
