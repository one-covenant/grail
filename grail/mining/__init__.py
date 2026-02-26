"""Pipelined mining engine for 3-GPU parallel rollout generation.

Public API:
  - PipelineConfig: Configuration dataclass (from env vars)
  - PipelinedMiningEngine: Orchestrator for pipelined generation
  - ProofWorker: HF model proof computation on dedicated GPU
  - WeightSyncStrategy: Abstract weight sync interface
  - SGLangWeightSync / VLLMWeightSync: Backend implementations
"""

from .config import PipelineConfig
from .engine import PipelinedMiningEngine
from .proof_worker import ProofWorker
from .weight_sync import SGLangWeightSync, VLLMWeightSync, WeightSyncStrategy

__all__ = [
    "PipelineConfig",
    "PipelinedMiningEngine",
    "ProofWorker",
    "SGLangWeightSync",
    "VLLMWeightSync",
    "WeightSyncStrategy",
]
