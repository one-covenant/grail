"""Distributed training configuration for FSDP2-based parallelism.

Reads from environment variables prefixed with ``GRAIL_DIST_`` and provides
auto-detection heuristics for tensor/data parallelism degrees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DistributedConfig:
    """Parameters governing distributed training (FSDP2 + TP).

    Defaults are conservative: auto-detect parallelism, offload the reference
    model to CPU, and use asynchronous checkpointing.
    """

    tp_degree: int = 0  # 0 = auto-detect from world_size and model size
    ref_model_offload: bool = True  # CPU offload for ref model when beta > 0
    reshard_after_forward: bool = False  # ZeRO-2 style (faster, +8GB VRAM)
    gc_skip_last_n: int = 0  # Skip gradient checkpointing on last N layers
    async_checkpoint: bool = True  # Non-blocking checkpoint writes

    @classmethod
    def from_env(cls) -> DistributedConfig:
        """Construct a ``DistributedConfig`` from ``GRAIL_DIST_*`` env vars.

        Unset variables fall back to dataclass defaults.
        """

        def _bool(key: str, default: bool) -> bool:
            raw = os.getenv(key)
            if raw is None:
                return default
            return raw.lower() in ("1", "true", "yes")

        def _int(key: str, default: int) -> int:
            raw = os.getenv(key)
            if raw is None:
                return default
            return int(raw)

        return cls(
            tp_degree=_int("GRAIL_DIST_TP_DEGREE", 0),
            ref_model_offload=_bool("GRAIL_DIST_REF_MODEL_OFFLOAD", True),
            reshard_after_forward=_bool("GRAIL_DIST_RESHARD_AFTER_FORWARD", False),
            gc_skip_last_n=_int("GRAIL_DIST_GC_SKIP_LAST_N", 0),
            async_checkpoint=_bool("GRAIL_DIST_ASYNC_CHECKPOINT", True),
        )

    @staticmethod
    def auto_detect_parallelism(
        world_size: int,
        model_params_b: float,
    ) -> tuple[int, int]:
        """Return ``(tp_degree, dp_degree)`` for a given cluster and model size.

        Heuristics (tuned for Qwen-family models with 8 KV heads):

        * 4 GPUs: TP=2, DP=2
        * 8 GPUs, <=14B params: TP=2, DP=4
        * 8 GPUs, >14B params: TP=4, DP=2

        Raises:
            ValueError: If the resulting degrees are inconsistent with
                ``world_size`` or if TP does not evenly divide the 8 KV heads
                assumed for Qwen models.
        """
        if world_size == 4:
            tp, dp = 2, 2
        elif world_size == 8 and model_params_b <= 14.0:
            tp, dp = 2, 4
        elif world_size == 8 and model_params_b > 14.0:
            tp, dp = 4, 2
        else:
            # Fallback: no tensor parallelism, pure data parallelism.
            tp, dp = 1, world_size

        if tp * dp != world_size:
            raise ValueError(
                f"Parallelism mismatch: tp_degree={tp} * dp_degree={dp} = {tp * dp} "
                f"!= world_size={world_size}"
            )

        num_kv_heads = 8  # Qwen family default
        if num_kv_heads % tp != 0:
            raise ValueError(f"tp_degree={tp} does not evenly divide num_kv_heads={num_kv_heads}")

        return tp, dp
