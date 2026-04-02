"""Drop-in replacement for HuggingFace Accelerate's ``Accelerator``.

``DistributedContext`` exposes the minimal surface used by
``GRPOAlgorithm.train_epoch()`` (device, autocast, backward, gather,
unwrap_model) without pulling in Accelerate as a dependency.  When running
under FSDP2, mixed-precision is handled by ``MixedPrecisionPolicy`` rather
than autocast, so :meth:`autocast` returns a no-op context manager.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.distributed as dist


class DistributedContext:
    """Lightweight Accelerator-compatible context for FSDP2 training."""

    def __init__(
        self,
        device: torch.device | None = None,
        rank: int = 0,
        world_size: int = 1,
        dp_group: dist.ProcessGroup | None = None,
        dp_size: int = 1,
    ) -> None:
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")

        self.rank = rank
        self.world_size = world_size
        self._dp_group = dp_group
        self._dp_size = dp_size

    # ------------------------------------------------------------------
    # Accelerator-compatible API
    # ------------------------------------------------------------------

    def autocast(self) -> contextlib.AbstractContextManager[None]:
        """Return a bf16 autocast context manager.

        FSDP2 applies mixed precision to sharded parameters via
        ``MixedPrecisionPolicy``, but the non-chunked logprob path runs
        lm_head as part of the full model forward and needs autocast for
        consistent bf16 compute (e.g., embedding output, log_softmax inputs).
        """
        return torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore[return-value]

    def backward(self, loss: torch.Tensor) -> None:
        """Run the backward pass.

        Under FSDP2, gradient synchronization is handled by the sharded
        module hooks, so a plain ``.backward()`` is sufficient.
        """
        loss.backward()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``tensor`` across the DP process group.

        Uses the DP group (not the full world group) so that TP-replicated
        tensors are not double-counted.  Falls back to returning ``tensor``
        unchanged when not distributed.
        """
        gather_size = self._dp_size if self._dp_group is not None else self.world_size
        if gather_size <= 1 or not dist.is_initialized():
            return tensor

        gathered = [torch.zeros_like(tensor) for _ in range(gather_size)]
        dist.all_gather(gathered, tensor, group=self._dp_group)
        return torch.cat(gathered, dim=0)

    def unwrap_model(self, model: Any) -> Any:
        """Return the underlying model, unwrapping any distributed wrapper.

        FSDP2 does not require manual unwrapping for state-dict access, so
        the model is returned as-is.
        """
        return model
