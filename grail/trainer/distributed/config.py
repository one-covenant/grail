"""Distributed training configuration for multi-strategy parallelism.

Reads from environment variables prefixed with ``GRAIL_DIST_`` and provides
auto-detection heuristics for tensor/data parallelism degrees.

Supported strategies:
  - ``fsdp2``: Fully Sharded Data Parallelism v2 (default). Shards model
    parameters, gradients, and optimizer states across GPUs.
  - ``ddp``: DistributedDataParallel. Replicates the full model on each GPU,
    all-reduces gradients during backward. Simpler and faster when the model
    fits in a single GPU's memory.
  - ``diloco``: Distributed Low-Communication training. Each GPU runs
    independent inner optimization (AdamW) for H steps, then synchronizes
    via pseudo-gradient all-reduce and an outer Nesterov SGD step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

StrategyType = Literal["fsdp2", "ddp", "diloco"]

_VALID_STRATEGIES: frozenset[str] = frozenset({"fsdp2", "ddp", "diloco"})


@dataclass
class DistributedConfig:
    """Parameters governing distributed training.

    Defaults are conservative: FSDP2 strategy, auto-detect parallelism,
    offload the reference model to CPU, and use asynchronous checkpointing.
    """

    # Strategy selection
    strategy: StrategyType = "fsdp2"

    # FSDP2/TP settings (ignored for ddp strategy)
    tp_degree: int = 0  # 0 = auto-detect from world_size and model size
    ref_model_offload: bool = True  # CPU offload for ref model when beta > 0
    reshard_after_forward: bool = False  # ZeRO-2 style (faster, +8GB VRAM)
    gc_skip_last_n: int = 0  # Skip gradient checkpointing on last N layers
    async_checkpoint: bool = True  # Non-blocking checkpoint writes

    # DILOCO settings (ignored unless strategy == "diloco")
    diloco_inner_steps: int = 10  # H: number of inner optimizer steps between outer syncs
    diloco_outer_lr: float = 0.7  # Outer Nesterov SGD learning rate
    diloco_outer_momentum: float = 0.9  # Outer Nesterov momentum coefficient

    # PULSE-DiLoCo: BF16-gated sparse communication with residual buffer
    pulse_diloco: bool = False  # Enable PULSE-DiLoCo sparse outer step

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

        def _float(key: str, default: float) -> float:
            raw = os.getenv(key)
            if raw is None:
                return default
            return float(raw)

        strategy_raw = os.getenv("GRAIL_DIST_STRATEGY", "fsdp2").strip().lower()
        if strategy_raw not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid GRAIL_DIST_STRATEGY={strategy_raw!r}. "
                f"Must be one of: {sorted(_VALID_STRATEGIES)}"
            )

        return cls(
            strategy=strategy_raw,  # type: ignore[arg-type]
            tp_degree=_int("GRAIL_DIST_TP_DEGREE", 0),
            ref_model_offload=_bool("GRAIL_DIST_REF_MODEL_OFFLOAD", True),
            reshard_after_forward=_bool("GRAIL_DIST_RESHARD_AFTER_FORWARD", False),
            gc_skip_last_n=_int("GRAIL_DIST_GC_SKIP_LAST_N", 0),
            async_checkpoint=_bool("GRAIL_DIST_ASYNC_CHECKPOINT", True),
            diloco_inner_steps=_int("GRAIL_DIST_DILOCO_INNER_STEPS", 10),
            diloco_outer_lr=_float("GRAIL_DIST_DILOCO_OUTER_LR", 0.7),
            diloco_outer_momentum=_float("GRAIL_DIST_DILOCO_OUTER_MOMENTUM", 0.9),
            pulse_diloco=_bool("GRAIL_DIST_PULSE_DILOCO", False),
        )

    def validate(self, world_size: int) -> None:
        """Validate config against the runtime environment.

        Args:
            world_size: Total number of distributed ranks.

        Raises:
            ValueError: If the config is invalid for the given world_size.
        """
        if self.strategy == "ddp" and self.tp_degree > 1:
            raise ValueError("DDP does not support tensor parallelism (tp_degree must be 0 or 1)")

        if self.strategy == "diloco" and world_size < 2:
            raise ValueError("DILOCO requires at least 2 workers (GRAIL_DIST_NPROC >= 2)")

        if self.strategy == "diloco" and self.diloco_inner_steps < 1:
            raise ValueError(f"diloco_inner_steps must be >= 1, got {self.diloco_inner_steps}")

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
