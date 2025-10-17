"""Algorithm registry for trainer.

Start minimal; evolve to a factory when we add more algorithms.
"""

from __future__ import annotations

from .base import TrainingAlgorithm
from .grpo import GRPOAlgorithm

__all__ = ["GRPOAlgorithm", "TrainingAlgorithm"]
