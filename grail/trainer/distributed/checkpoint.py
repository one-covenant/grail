"""Distributed checkpointing for FSDP2 + TP training.

Provides three checkpoint formats:

1. **Full HF checkpoint** (``save_full_checkpoint``): Gathers the complete state dict
   across all ranks, writes HuggingFace-compatible safetensors on rank 0.  Used for
   publishing weights to miners/validators.

2. **Sharded DCP checkpoint** (``async_save_sharded_checkpoint``, ``save_sharded_checkpoint``):
   Each rank saves its local shard via ``torch.distributed.checkpoint``.  Supports async
   writes (pinned + shared memory staging) for overlapping I/O with training.  Used for
   fast training resume.

3. **HF-to-distributed loader** (``load_hf_into_distributed``): Reads a full HF
   safetensors checkpoint on rank 0 and broadcasts into an already-sharded FSDP2+TP
   model, including non-parameter buffers (RoPE embeddings, etc.).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for DCP APIs that may not exist in older PyTorch versions.
# Each function guards its own imports so callers get a clear error message
# rather than an ImportError at module load time.
# ---------------------------------------------------------------------------

_DCP_AVAILABLE: bool | None = None


def _check_dcp_available() -> bool:
    """Return True if torch.distributed.checkpoint is usable."""
    global _DCP_AVAILABLE  # noqa: PLW0603
    if _DCP_AVAILABLE is not None:
        return _DCP_AVAILABLE
    try:
        import torch.distributed.checkpoint  # noqa: F401
        from torch.distributed.checkpoint.state_dict import (  # noqa: F401
            get_model_state_dict,
            set_model_state_dict,
        )

        _DCP_AVAILABLE = True
    except ImportError:
        _DCP_AVAILABLE = False
    return _DCP_AVAILABLE


# ---------------------------------------------------------------------------
# 1. Full HF checkpoint (collective gather + rank-0 write)
# ---------------------------------------------------------------------------


def save_full_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    path: str | Path,
    rank: int,
) -> None:
    """Gather full state dict on all ranks, save HF safetensors on rank 0.

    This is a **collective** operation: every rank must call this function
    because ``get_model_state_dict`` with ``full_state_dict=True`` performs
    all-gathers internally.

    Args:
        model: FSDP2-wrapped model.
        tokenizer: HuggingFace tokenizer (must support ``save_pretrained``).
        path: Directory where the checkpoint will be written (rank 0 only).
        rank: Current process rank.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    path = Path(path)

    # Collective: all ranks participate in the gather.
    logger.info("Gathering full state dict (rank=%d)...", rank)
    full_sd = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)

        # Pass the gathered state_dict directly to save_pretrained.
        # Do NOT use load_state_dict(assign=True) as that would mutate
        # the live training model, replacing CUDA DTensor params with CPU
        # tensors and breaking FSDP2 invariants.
        model.save_pretrained(  # type: ignore[attr-defined]
            str(path),
            state_dict=full_sd,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(path))

        logger.info("Full HF checkpoint saved to %s", path)

    dist.barrier()


# ---------------------------------------------------------------------------
# 1b. DDP checkpoint (rank-0 only, no collective gather)
# ---------------------------------------------------------------------------


def save_ddp_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    path: str | Path,
    rank: int,
) -> None:
    """Save checkpoint from a DDP-wrapped or bare model. Only rank 0 writes.

    Unlike ``save_full_checkpoint``, this is NOT a collective operation.
    Only rank 0 writes to disk. Callers are responsible for synchronization
    (e.g., placing a ``dist.barrier()`` after this call if needed).

    Args:
        model: DDP-wrapped or bare model.
        tokenizer: HuggingFace tokenizer (must support ``save_pretrained``).
        path: Directory where the checkpoint will be written (rank 0 only).
        rank: Current process rank.
    """
    if rank != 0:
        return

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # DDP wraps the model; .module is the original model. Bare models
    # (e.g., DILOCO) don't have .module.
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(  # type: ignore[attr-defined]
        str(path),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(path))
    logger.info("DDP/DILOCO checkpoint saved to %s", path)


# ---------------------------------------------------------------------------
# 2a. Async sharded DCP save
# ---------------------------------------------------------------------------


def create_checkpoint_stager() -> Any:
    """Create a ``DefaultStager`` configured for async DCP checkpointing.

    Uses pinned memory, shared memory, and async staging for maximum
    overlap with training computation.

    Returns:
        A ``DefaultStager`` instance.

    Raises:
        ImportError: If staging APIs are not available.
    """
    from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

    return DefaultStager(
        StagingOptions(
            use_pinned_memory=True,
            use_shared_memory=True,
            use_async_staging=True,
        )
    )


def async_save_sharded_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    stager: Any | None = None,
) -> Any:
    """Async DCP save for training resume. Each rank saves its own shard.

    The caller is responsible for tracking the returned ``AsyncSaveResponse``
    (call ``.result()`` to block until the background write completes, or
    poll ``.done()``).

    Args:
        model: FSDP2-wrapped model.
        optimizer: Optimizer whose state to checkpoint.
        path: Directory for the sharded checkpoint.
        stager: Optional ``DefaultStager``. Created via
            :func:`create_checkpoint_stager` if not provided.

    Returns:
        An ``AsyncSaveResponse`` (or similar future) from ``dcp.async_save``.

    Raises:
        ImportError: If async DCP APIs are not available.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict: dict[str, Any] = {
        "model": get_model_state_dict(model, options=opts),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=opts),
    }

    if stager is None:
        stager = create_checkpoint_stager()

    try:
        from torch.distributed.checkpoint.async_checkpoint import (  # type: ignore[import-not-found]
            AsyncCheckpointerType,
        )

        _async_save: Any = dcp.async_save  # type: ignore[attr-defined]
        _writer: Any = dcp.FileSystemWriter  # type: ignore[attr-defined]
        response = _async_save(
            state_dict,
            storage_writer=_writer(str(path)),
            async_checkpointer_type=AsyncCheckpointerType.PROCESS,
            async_stager=stager,
        )
    except ImportError:
        # Older PyTorch without AsyncCheckpointerType: fall back to the
        # simpler async_save signature.
        _async_save = dcp.async_save  # type: ignore[attr-defined]
        _writer = dcp.FileSystemWriter  # type: ignore[attr-defined]
        response = _async_save(
            state_dict,
            storage_writer=_writer(str(path)),
        )

    logger.info("Async sharded DCP save initiated to %s", path)
    return response


# ---------------------------------------------------------------------------
# 2b. Synchronous sharded DCP save (fallback)
# ---------------------------------------------------------------------------


def save_sharded_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
) -> None:
    """Synchronous DCP save for training resume.

    Each rank saves its own shard. Use this as a fallback when async
    checkpointing is not available or not desired.

    Args:
        model: FSDP2-wrapped model.
        optimizer: Optimizer whose state to checkpoint.
        path: Directory for the sharded checkpoint.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict: dict[str, Any] = {
        "model": get_model_state_dict(model, options=opts),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=opts),
    }

    _save = dcp.save  # type: ignore[attr-defined]
    _writer = dcp.FileSystemWriter  # type: ignore[attr-defined]
    _save(state_dict, storage_writer=_writer(str(path)))

    logger.info("Sharded DCP checkpoint saved to %s", path)


# ---------------------------------------------------------------------------
# 3. Sharded DCP load (resume)
# ---------------------------------------------------------------------------


def load_sharded_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
) -> None:
    """Load a sharded DCP checkpoint for training resume.

    Each rank loads only its own shard. DCP handles automatic resharding if
    the topology (number of ranks, TP/DP degrees) changed between the save
    and load.

    Args:
        model: FSDP2-wrapped model (must match architecture).
        optimizer: Optimizer to restore state into.
        path: Directory containing the sharded checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint directory does not exist.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    # cpu_offload=False: keep state on GPU during resume (no unnecessary roundtrip)
    opts = StateDictOptions(full_state_dict=False, cpu_offload=False)

    # Build empty state dicts with the correct structure for DCP to fill.
    model_sd = get_model_state_dict(model, options=opts)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=opts)

    state_dict: dict[str, Any] = {
        "model": model_sd,
        "optimizer": optim_sd,
    }

    _load = dcp.load  # type: ignore[attr-defined]
    _reader = dcp.FileSystemReader  # type: ignore[attr-defined]
    _load(state_dict, storage_reader=_reader(str(path)))

    # Apply the loaded state back into the live model and optimizer.
    set_model_state_dict(
        model,
        state_dict["model"],
        options=opts,
    )
    set_optimizer_state_dict(
        model,
        optimizer,
        state_dict["optimizer"],
        options=opts,
    )

    logger.info("Sharded DCP checkpoint loaded from %s", path)


# ---------------------------------------------------------------------------
# 4. Load HF safetensors into a distributed model
# ---------------------------------------------------------------------------


def load_hf_into_distributed(
    model: nn.Module,
    checkpoint_path: str | Path,
    rank: int,
) -> None:
    """Load an HF safetensors checkpoint into an already-sharded FSDP2+TP model.

    Rank 0 reads the full state dict from disk. ``set_model_state_dict`` with
    ``broadcast_from_rank0=True`` distributes the weights to all ranks and
    reshards them according to each rank's FSDP2/TP configuration.

    After parameters are set, non-parameter buffers (e.g. RoPE ``inv_freq``)
    are explicitly broadcast from rank 0 so that all ranks share identical
    values.

    Args:
        model: FSDP2+TP wrapped model (architecture must match checkpoint).
        checkpoint_path: Path to a directory containing HF-format safetensors
            (``model.safetensors`` or sharded ``model-*.safetensors``).
        rank: Current process rank.

    Raises:
        FileNotFoundError: If no model weights are found at ``checkpoint_path``.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    checkpoint_path = Path(checkpoint_path)

    # Resolve HuggingFace model IDs (e.g. "Qwen/Qwen3-8B") to local cache paths.
    # Only rank 0 needs the resolved path; others pass an empty state dict.
    if not checkpoint_path.exists() and "/" in str(checkpoint_path):
        try:
            from huggingface_hub import snapshot_download

            resolved = snapshot_download(str(checkpoint_path))
            checkpoint_path = Path(resolved)
            logger.info("Resolved HF model ID to cache: %s", checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to resolve HF model ID %s: %s", checkpoint_path, exc)

    # Rank 0 loads the full state dict; other ranks pass an empty dict.
    if rank == 0:
        from grail.shared.safetensors_utils import load_model_state_dict

        full_sd = load_model_state_dict(checkpoint_path)
        if full_sd is None:
            raise FileNotFoundError(
                f"No model weights found at {checkpoint_path}. "
                "Expected model.safetensors or model.safetensors.index.json."
            )
        logger.info(
            "Rank 0 loaded %d parameter tensors from %s",
            len(full_sd),
            checkpoint_path,
        )
    else:
        full_sd = {}

    # Distribute weights: DCP broadcasts from rank 0 and reshards automatically.
    set_model_state_dict(
        model,
        full_sd,  # type: ignore[arg-type]
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=True,
        ),
    )
    logger.info("Model parameters set from HF checkpoint (rank=%d)", rank)

    # Broadcast non-parameter buffers (RoPE inv_freq, layer norms, etc.).
    # set_model_state_dict handles parameters but buffers need manual sync.
    _broadcast_buffers(model, src_rank=0)

    dist.barrier()
    logger.info("HF checkpoint loaded into distributed model (rank=%d)", rank)


def _broadcast_buffers(model: nn.Module, src_rank: int = 0) -> None:
    """Broadcast all named buffers from ``src_rank`` to every other rank.

    This ensures non-parameter state like RoPE inverse frequency tensors are
    identical across all ranks after loading from a single-rank checkpoint.
    """
    if not dist.is_initialized():
        return

    for name, buf in model.named_buffers():
        if buf is None:
            continue
        try:
            dist.broadcast(buf, src=src_rank)
        except RuntimeError:
            # Buffer may be on CPU or have an unsupported dtype for NCCL.
            # Move to GPU, broadcast, then update in-place.
            device = torch.device("cuda", dist.get_rank() % torch.cuda.device_count())
            buf_gpu = buf.to(device)
            dist.broadcast(buf_gpu, src=src_rank)
            buf.copy_(buf_gpu.to(buf.device))
            logger.debug("Buffer %s broadcast via GPU fallback", name)
