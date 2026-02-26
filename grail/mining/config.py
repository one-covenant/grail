"""Pipeline configuration for 3-GPU pipelined mining.

GPU 0 = vLLM/SGLang generation, GPU 1 = HF proof computation, GPU 2 = Triton kernel eval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration for pipelined mining across multiple GPUs.

    All settings can be overridden via environment variables prefixed with
    ``GRAIL_PIPELINE_``.
    """

    enabled: bool = False  # GRAIL_PIPELINE_ENABLED
    backend: str = "vllm"  # GRAIL_PIPELINE_BACKEND (vllm|sglang)
    vllm_gpu: int = 0  # GRAIL_PIPELINE_VLLM_GPU
    proof_gpu: int = 1  # GRAIL_PIPELINE_PROOF_GPU
    # eval_gpu: already handled by KERNEL_EVAL_GPU_IDS

    # Generation server params
    gpu_memory_utilization: float = 0.90  # Higher than trainer (dedicated GPU)
    max_model_len: int = 12288
    max_num_seqs: int = 64
    max_concurrent_requests: int = 48
    server_timeout: float = 300.0

    # Symlink directory for vLLM weight reload (empty = use checkpoint parent dir)
    symlink_dir: str = ""  # GRAIL_PIPELINE_SYMLINK_DIR

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Construct from environment variables with sensible defaults."""

        def _bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "").strip().lower()
            if not val:
                return default
            return val in ("1", "true", "yes")

        def _int(key: str, default: int) -> int:
            val = os.getenv(key, "").strip()
            return int(val) if val else default

        def _float(key: str, default: float) -> float:
            val = os.getenv(key, "").strip()
            return float(val) if val else default

        def _str(key: str, default: str) -> str:
            val = os.getenv(key, "").strip()
            return val if val else default

        return cls(
            enabled=_bool("GRAIL_PIPELINE_ENABLED", cls.enabled),
            backend=_str("GRAIL_PIPELINE_BACKEND", cls.backend),
            vllm_gpu=_int("GRAIL_PIPELINE_VLLM_GPU", cls.vllm_gpu),
            proof_gpu=_int("GRAIL_PIPELINE_PROOF_GPU", cls.proof_gpu),
            gpu_memory_utilization=_float(
                "GRAIL_PIPELINE_GPU_MEM_UTIL", cls.gpu_memory_utilization
            ),
            max_model_len=_int("GRAIL_PIPELINE_MAX_MODEL_LEN", cls.max_model_len),
            max_num_seqs=_int("GRAIL_PIPELINE_MAX_NUM_SEQS", cls.max_num_seqs),
            max_concurrent_requests=_int(
                "GRAIL_PIPELINE_MAX_CONCURRENT", cls.max_concurrent_requests
            ),
            server_timeout=_float("GRAIL_PIPELINE_SERVER_TIMEOUT", cls.server_timeout),
            symlink_dir=_str("GRAIL_PIPELINE_SYMLINK_DIR", cls.symlink_dir),
        )
