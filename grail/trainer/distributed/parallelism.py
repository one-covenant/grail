"""FSDP2 + Tensor Parallelism setup for HuggingFace transformer models.

Handles DeviceMesh creation, TP+SP (Sequence Parallelism) application,
per-layer FSDP2 wrapping, gradient checkpointing, and RMSNorm replacement.
Supports Qwen2, Qwen3, and Llama model families.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.utils.checkpoint import checkpoint

from grail.trainer.distributed.config import DistributedConfig

logger = logging.getLogger(__name__)

# HuggingFace RMSNorm class names that need replacement for SP compatibility.
# torch.nn.RMSNorm supports DTensor sharding; the HF variants do not.
_HF_RMSNORM_NAMES = frozenset({"Qwen2RMSNorm", "Qwen3RMSNorm", "LlamaRMSNorm"})


def _get_base_model(model: nn.Module) -> nn.Module:
    """Return the inner transformer model (model.model for HF CausalLM wrappers)."""
    if hasattr(model, "model"):
        return cast(nn.Module, model.model)  # type: ignore[union-attr]
    return model


def _get_layers(base_model: nn.Module) -> nn.ModuleList:
    """Return the decoder layer list from the base transformer model.

    Raises:
        AttributeError: If the model has no .layers attribute.
    """
    if not hasattr(base_model, "layers"):
        raise AttributeError(
            f"{type(base_model).__name__} has no .layers attribute. "
            "Expected a HuggingFace Qwen2/Qwen3/Llama model."
        )
    return cast(nn.ModuleList, base_model.layers)  # type: ignore[union-attr]


def create_device_mesh(world_size: int, tp_degree: int) -> DeviceMesh:
    """Create a 2D DeviceMesh with (dp, tp) dimensions.

    Args:
        world_size: Total number of ranks in the process group.
        tp_degree: Tensor parallelism degree. Must evenly divide world_size.

    Returns:
        A 2D DeviceMesh with dimension names ("dp", "tp").

    Raises:
        ValueError: If tp_degree does not evenly divide world_size or is < 1.
    """
    if tp_degree < 1:
        raise ValueError(f"tp_degree must be >= 1, got {tp_degree}")
    if world_size % tp_degree != 0:
        raise ValueError(f"tp_degree={tp_degree} does not evenly divide world_size={world_size}")

    dp_degree = world_size // tp_degree
    logger.info(
        "Creating 2D device mesh: world_size=%d, dp=%d, tp=%d",
        world_size,
        dp_degree,
        tp_degree,
    )

    mesh = init_device_mesh(
        "cuda",
        (dp_degree, tp_degree),
        mesh_dim_names=("dp", "tp"),
    )
    return mesh


def replace_rmsnorm_for_sp(model: nn.Module) -> None:
    """Replace HuggingFace RMSNorm modules with torch.nn.RMSNorm.

    HuggingFace's Qwen2RMSNorm, Qwen3RMSNorm, and LlamaRMSNorm do not support
    DTensor inputs required by SequenceParallel. torch.nn.RMSNorm (added in
    PyTorch 2.4) handles sharded tensors natively. This function walks the
    module tree, finds matching RMSNorm submodules by class name, copies their
    weights and eps, and replaces them on the parent module.

    Args:
        model: The HuggingFace model to modify in-place.
    """
    replacements: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name not in _HF_RMSNORM_NAMES:
            continue

        # Extract parameters from the HF RMSNorm
        weight = cast(nn.Parameter, module.weight)  # type: ignore[union-attr]
        eps: float = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        normalized_shape: int = weight.shape[0]

        # Create the torch.nn.RMSNorm replacement
        new_norm = torch.nn.RMSNorm(normalized_shape, eps=eps, device=weight.device)
        new_norm.weight = weight  # Share the same parameter (no copy)

        replacements.append((name, new_norm))

    # Apply replacements by walking parent -> child relationships
    replaced = 0
    for name, new_module in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            attr_name = parts[0]
            parent = model
        setattr(parent, attr_name, new_module)
        replaced += 1

    if replaced > 0:
        logger.info(
            "Replaced %d HF RMSNorm modules with torch.nn.RMSNorm for SP compatibility",
            replaced,
        )


def apply_tp_sp(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Apply Tensor Parallelism + Sequence Parallelism to a HuggingFace model.

    Parallelization plan (matching Qwen2/Qwen3/Llama architectures):

    Top-level:
      - model.embed_tokens: RowwiseParallel (output sharded on seq dim for SP)
      - lm_head: ColwiseParallel (input sharded on seq dim, output sharded on vocab)
      - model.norm: SequenceParallel

    Per decoder layer:
      - input_layernorm, post_attention_layernorm: SequenceParallel
      - self_attn: PrepareModuleInput (Shard(1) -> Replicate for attention)
      - self_attn.{q,k,v}_proj: ColwiseParallel
      - self_attn.o_proj: RowwiseParallel (output Shard(1) for residual)
      - mlp: PrepareModuleInput (Shard(1) -> Replicate for MLP)
      - mlp.{gate,up}_proj: ColwiseParallel
      - mlp.down_proj: RowwiseParallel (output Shard(1) for residual)

    Notes:
      - o_proj and down_proj use output_layouts=Shard(1) so their outputs match
        the sharded residual connection (both Shard(1) on the sequence dimension).
      - self_attn uses input_kwarg_layouts because HuggingFace attention modules
        receive hidden_states as a keyword argument.
      - No num_heads patching is needed; HF uses -1 in reshape.

    Args:
        model: HuggingFace CausalLM model (e.g., Qwen2ForCausalLM).
        tp_mesh: The TP sub-mesh from the 2D device mesh.
    """
    from torch.distributed.tensor import Replicate, Shard

    # Resolve the base transformer (model.model for HF CausalLM wrappers)
    base_model = _get_base_model(model)

    # --- Top-level parallelization ---
    parallelize_module(
        base_model,
        tp_mesh,
        {
            "embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
        },
    )

    # lm_head lives on the outer CausalLM wrapper
    parallelize_module(
        model,
        tp_mesh,
        {
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
        },
    )

    # --- Per-layer parallelization ---
    layers = _get_layers(base_model)
    for layer in layers:
        plan: dict[str, Any] = {
            # Layer norms operate on sharded sequences
            "input_layernorm": SequenceParallel(),
            "post_attention_layernorm": SequenceParallel(),
            # Attention: gather sharded hidden_states before QKV projections.
            # HF passes hidden_states as a kwarg, so use input_kwarg_layouts.
            "self_attn": PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            # MLP: gather sharded hidden_states before gate/up projections.
            # HF MLP forward() takes positional args, so use input_layouts.
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }

        parallelize_module(layer, tp_mesh, plan)

    logger.info(
        "Applied TP+SP to %d transformer layers (tp_degree=%d)",
        len(layers),
        tp_mesh.size(),
    )


def apply_gradient_checkpointing(
    model: nn.Module,
    gc_every_n: int,
    skip_last_n: int,
) -> None:
    """Apply per-layer gradient checkpointing with selective layer skipping.

    Wraps each eligible decoder layer's forward method with
    ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``. Layers are
    checkpointed every ``gc_every_n`` layers, and the last ``skip_last_n``
    layers are always skipped (their activations are kept in memory for faster
    backward).

    Args:
        model: HuggingFace CausalLM model.
        gc_every_n: Checkpoint every Nth layer (1 = all eligible layers).
        skip_last_n: Number of final layers to skip checkpointing.
    """
    base_model = _get_base_model(model)
    if not hasattr(base_model, "layers"):
        logger.warning("Model has no .layers attribute, skipping gradient checkpointing")
        return

    layers = _get_layers(base_model)
    num_layers = len(layers)
    eligible_end = num_layers - skip_last_n

    if eligible_end <= 0:
        logger.warning(
            "skip_last_n=%d >= num_layers=%d, no layers will be checkpointed",
            skip_last_n,
            num_layers,
        )
        return

    checkpointed = 0
    for i in range(eligible_end):
        if i % gc_every_n != 0:
            continue

        layer = layers[i]
        original_forward = layer.forward

        # Use a factory to capture the correct original_forward per layer
        def _make_ckpt_forward(
            orig_fn: Any,
        ) -> Any:
            def ckpt_forward(*args: Any, **kwargs: Any) -> Any:
                return checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)

            return ckpt_forward

        layer.forward = _make_ckpt_forward(original_forward)  # type: ignore[method-assign]
        checkpointed += 1

    logger.info(
        "Gradient checkpointing: %d/%d layers (every %d, skipping last %d)",
        checkpointed,
        num_layers,
        gc_every_n,
        skip_last_n,
    )


def apply_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
) -> None:
    """Apply FSDP2 wrapping: per-layer sharding then root model sharding.

    Each decoder layer is individually wrapped with ``fully_shard`` to enable
    communication/computation overlap and fine-grained memory management. The
    root model is then wrapped to handle any remaining parameters (embeddings,
    final norm, lm_head).

    Args:
        model: HuggingFace CausalLM model (already TP-parallelized if applicable).
        dp_mesh: The DP sub-mesh from the 2D device mesh.
        mp_policy: Mixed precision policy (e.g., param bf16, reduce fp32).
        reshard_after_forward: If True, reshard parameters after forward (ZeRO-3).
            If False, keep unsharded (ZeRO-2 style, faster but uses more memory).
    """
    base_model = _get_base_model(model)

    # Per-layer FSDP wrapping
    if hasattr(base_model, "layers"):
        layer_list = _get_layers(base_model)
        for layer in layer_list:
            fully_shard(
                layer,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        logger.debug("FSDP2: wrapped %d decoder layers", len(layer_list))

    # Root model wrapping (covers embed_tokens, norm, lm_head, etc.)
    fully_shard(
        model,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )

    logger.info(
        "FSDP2 applied: reshard_after_forward=%s, mp_policy=%s",
        reshard_after_forward,
        mp_policy,
    )


def apply_ddp(
    model: nn.Module,
    local_rank: int,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    DDP replicates the full model on each GPU and all-reduces gradients during
    backward, overlapped with computation. Suitable when the model + optimizer
    states fit in a single GPU's memory.

    Args:
        model: Model already moved to the correct CUDA device.
        local_rank: Local GPU index for this process.

    Returns:
        The DDP-wrapped model.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    wrapped = DDP(
        model,
        device_ids=[local_rank],
        static_graph=False,
        gradient_as_bucket_view=True,
    )
    logger.info("DDP applied: device_ids=[%d], static_graph=False", local_rank)
    return wrapped


def setup_ref_model(model: nn.Module) -> None:
    """Freeze and FSDP2-wrap the reference model with CPU offloading.

    The reference model is used for KL divergence computation during GRPO
    training. It does not need TP (replicated across all ranks) and its
    parameters are offloaded to CPU with pinned memory to minimize GPU
    memory usage.

    Args:
        model: HuggingFace CausalLM model to use as the frozen reference.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply FSDP2 per-layer with CPU offload.
    # Match the training model's bf16 precision for consistent KL computation.
    base_model = _get_base_model(model)
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)

    if hasattr(base_model, "layers"):
        layer_list = _get_layers(base_model)
        for layer in layer_list:
            fully_shard(layer, offload_policy=offload_policy, mp_policy=mp_policy)

    # Root wrapping
    fully_shard(model, offload_policy=offload_policy, mp_policy=mp_policy)

    logger.info("Reference model frozen and wrapped with FSDP2 + CPU offload (pin_memory=True)")


def setup_training_model(
    model: nn.Module,
    mesh: DeviceMesh,
    config: DistributedConfig,
) -> None:
    """Full distributed setup: RMSNorm replacement, TP+SP, GC, and FSDP2.

    Orchestrates the complete parallelization pipeline for the training model:
    1. Replace HF RMSNorm with torch.nn.RMSNorm (SP compatibility)
    2. Apply Tensor Parallelism + Sequence Parallelism
    3. Apply selective gradient checkpointing
    4. Apply FSDP2 data-parallel sharding

    Args:
        model: HuggingFace CausalLM model (on meta device or CPU).
        mesh: 2D DeviceMesh with ("dp", "tp") dimensions.
        config: Distributed training configuration.
    """
    tp_mesh: DeviceMesh = mesh["tp"]
    dp_mesh: DeviceMesh = mesh["dp"]

    # Step 1–2: TP + SP (skip when tp_degree == 1 to avoid DTensor conflicts with FSDP2)
    if tp_mesh.size() > 1:
        replace_rmsnorm_for_sp(model)
        apply_tp_sp(model, tp_mesh)
    else:
        logger.info("Skipping TP+SP (tp_degree=1, pure data parallelism)")

    # Step 3: Gradient checkpointing via HF's built-in support.
    # The custom per-layer checkpoint wrapper (apply_gradient_checkpointing) is
    # incompatible with FSDP2: tensor metadata changes between forward and
    # recomputation cause CheckpointError. HF's implementation handles this
    # correctly by integrating with the model's own forward hooks.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # type: ignore[operator]  # HF dynamic method
        logger.info("Gradient checkpointing enabled (HF built-in, use_reentrant=False)")
    else:
        gc_every_n = 1
        apply_gradient_checkpointing(
            model, gc_every_n=gc_every_n, skip_last_n=config.gc_skip_last_n
        )

    # Step 4: FSDP2 wrapping with mixed precision
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    apply_fsdp2(model, dp_mesh, mp_policy, reshard_after_forward=config.reshard_after_forward)

    logger.info("Training model distributed setup complete")
