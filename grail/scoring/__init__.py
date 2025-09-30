"""GRAIL scoring package.

Converts validation results into miner scores and computes weights.
"""

from __future__ import annotations

from .scorer import MinerScorer
from .weights import WeightComputer

__all__ = ["MinerScorer", "WeightComputer"]
