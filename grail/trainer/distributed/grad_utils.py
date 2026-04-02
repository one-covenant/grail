"""DTensor-safe gradient clipping for FSDP2 parameters.

Works around two upstream PyTorch bugs in ``torch.nn.utils.clip_grad_norm_``
when used with FSDP2 / DTensor parameters:

1. **Numeric epsilon bug** (`pytorch/pytorch#149768`): the stock implementation
   computes ``total_norm + 1e-6`` where ``total_norm`` is a DTensor. Each rank
   adds epsilon to its local shard independently, inflating the denominator by
   ``world_size * 1e-6`` instead of a single ``1e-6``.

2. **O(n^2) complexity bug** (`pytorch/pytorch#169445`): ``torch.stack`` on a
   list of DTensors has quadratic cost in the number of tensors, making gradient
   clipping prohibitively slow for models with many parameters.

The fix separates DTensor and regular-tensor gradients, computes partial norms
locally, manually all-reduces, and then clips uniformly.

Adapted from ``pytorch/torchtitan`` (``torchtitan/distributed/utils.py``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _local_norm(
    tensors: list[torch.Tensor],
    norm_type: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the (partial) p-norm of a flat list of local tensors.

    For ``norm_type == inf`` this returns the element-wise max.  Otherwise
    it returns ``(sum |t|^p)`` **without** the final ``^(1/p)`` root so that
    partial results can be summed across ranks before taking the root once.
    """
    if len(tensors) == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    device = tensors[0].device
    dtype = torch.float32  # always accumulate in fp32

    if math.isinf(norm_type):
        # max-norm: local max, to be all-reduced with MAX later
        return torch.stack([t.detach().abs().max().to(dtype) for t in tensors]).max().to(device)

    # p-norm: accumulate sum of |t|^p in a scalar
    partial = torch.tensor(0.0, device=device, dtype=dtype)
    for t in tensors:
        local = t.detach().to(dtype)
        partial += torch.norm(local, norm_type).pow(norm_type)
    return partial


def _get_fsdp_process_group(dtensor: DTensor) -> dist.ProcessGroup | None:
    """Extract the FSDP process group from a DTensor's device mesh.

    Returns ``None`` when the mesh does not contain an FSDP dimension (e.g. the
    tensor is only replicated), in which case no all-reduce is needed.
    """
    mesh = dtensor.device_mesh
    if mesh is None:
        return None

    mesh_dim_names = mesh.mesh_dim_names
    if mesh_dim_names is None:
        # Single-dim mesh: treat as the FSDP group
        return mesh.get_group(0)

    # Look for common FSDP mesh dimension names used by PyTorch FSDP2
    for name in ("dp", "dp_shard", "dp_replicate_and_shard", "fsdp"):
        if name in mesh_dim_names:
            idx = list(mesh_dim_names).index(name)
            return mesh.get_group(idx)

    # Fallback: flatten the entire mesh into one group
    return mesh.get_group()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool | None = None,
) -> torch.Tensor:
    """DTensor-safe gradient clipping that avoids upstream PyTorch bugs.

    Separates DTensor gradients (sharded by FSDP2) from plain tensor gradients,
    computes local partial norms, all-reduces across the FSDP process group,
    and clips all gradients uniformly.

    Args:
        parameters: Model parameters (or a single tensor) whose ``.grad`` will
            be clipped in-place.
        max_norm: Maximum allowed gradient norm.
        norm_type: Type of p-norm (default 2.0). Use ``float('inf')`` for the
            max-norm.
        foreach: Use the faster foreach-based norm for regular tensors.  When
            ``None``, auto-detects based on device type.

    Returns:
        The total gradient norm as a plain (non-DTensor) tensor.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # Partition gradients by type
    dtensor_grads: list[torch.Tensor] = []  # local data of DTensor .grad
    regular_grads: list[torch.Tensor] = []
    fsdp_pg: dist.ProcessGroup | None = None

    for p in parameters:
        if p.grad is None:
            continue
        if isinstance(p.grad, DTensor):
            # Extract local shard to avoid the O(n^2) torch.stack bug
            dtensor_grads.append(p.grad._local_tensor)
            if fsdp_pg is None:
                fsdp_pg = _get_fsdp_process_group(p.grad)
        else:
            regular_grads.append(p.grad)

    # ------------------------------------------------------------------
    # Compute norms for each group
    # ------------------------------------------------------------------

    # Determine a common device for zero-tensors (empty partition edge case)
    _dev: torch.device | None = None
    if dtensor_grads:
        _dev = dtensor_grads[0].device
    elif regular_grads:
        _dev = regular_grads[0].device

    # DTensor grads: compute local partial norm, then all-reduce
    dtensor_partial = _local_norm(dtensor_grads, norm_type, device=_dev)

    if fsdp_pg is not None and len(dtensor_grads) > 0:
        if math.isinf(norm_type):
            dist.all_reduce(dtensor_partial, op=dist.ReduceOp.MAX, group=fsdp_pg)
        else:
            dist.all_reduce(dtensor_partial, op=dist.ReduceOp.SUM, group=fsdp_pg)

    # Regular grads: standard local norm (no all-reduce needed)
    regular_partial = _local_norm(regular_grads, norm_type, device=_dev)

    # ------------------------------------------------------------------
    # Combine into total norm
    # ------------------------------------------------------------------

    if math.isinf(norm_type):
        total_norm = torch.maximum(dtensor_partial, regular_partial)
    else:
        total_norm = (dtensor_partial + regular_partial).pow(1.0 / norm_type)

    # ------------------------------------------------------------------
    # Clip all grads uniformly
    # ------------------------------------------------------------------

    clip_coef = max_norm / (total_norm + 1e-6)
    # Clamped to [0, 1] so we never scale gradients *up*
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    if foreach is None:
        foreach = all(g.device.type in ("cuda", "cpu") for g in (*dtensor_grads, *regular_grads))

    all_grads = dtensor_grads + regular_grads
    if len(all_grads) > 0:
        if foreach:
            torch._foreach_mul_(all_grads, clip_coef_clamped)
        else:
            for g in all_grads:
                g.mul_(clip_coef_clamped)

    return total_norm


def check_grad_nan_across_ranks(grad_norm: torch.Tensor) -> bool:
    """All-reduce NaN/Inf flag so all ranks agree on skip/step.

    Returns ``True`` if *any* rank observes a NaN or Inf gradient norm,
    ensuring all ranks make the same decision about whether to skip the
    optimizer step.  When ``torch.distributed`` is not initialized (single-GPU
    training), falls back to a local check.

    Args:
        grad_norm: Scalar gradient norm tensor (output of :func:`clip_grad_norm_`).

    Returns:
        ``True`` if the gradient norm is NaN or Inf on any rank.
    """
    has_bad = torch.isnan(grad_norm) | torch.isinf(grad_norm)

    if not dist.is_initialized():
        return bool(has_bad.item())

    # Convert to int for all_reduce (boolean reduce is not portable across backends)
    flag = has_bad.int().to(grad_norm.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0)
